"""Data processing components for ETL pipeline."""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import duckdb
import numpy as np
import pandas as pd
import polars as pl
from omegaconf import DictConfig

from ..validation.schemas import TrajectoryData
from ...features.trajectory_features import TrajectoryFeatureExtractor
from ...utils.data_quality import DataQualityMonitor

logger = logging.getLogger(__name__)


class DataProcessor:
    """Enhanced data processor with validation and quality monitoring."""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.quality_monitor = DataQualityMonitor(config)
        self.feature_extractor = TrajectoryFeatureExtractor(config)
        
    async def process(self, data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Process raw trajectory data into clean DataFrame."""
        if not data:
            logger.warning("No data to process")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        logger.info(f"Processing {len(df)} records")
        
        # Data cleaning and validation
        df = await self._clean_data(df)
        
        # Quality monitoring
        quality_report = await self.quality_monitor.assess_quality(df)
        logger.info(f"Data quality score: {quality_report.get('overall_score', 0):.2f}")
        
        # Feature extraction
        if self.config.get("extract_features", True):
            df = await self._extract_features(df)
        
        logger.info(f"Processed data shape: {df.shape}")
        return df
    
    async def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate trajectory data."""
        initial_count = len(df)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=["vehicle_id", "timestamp"])
        
        # Handle missing values
        df = df.dropna(subset=["vehicle_id", "x", "y", "timestamp"])
        
        # Sort by vehicle and time
        df = df.sort_values(["vehicle_id", "timestamp"])
        
        # Basic range validation
        df = df[
            (df["x"].abs() < 1e6) & 
            (df["y"].abs() < 1e6) &
            (df["velocity"] >= 0) & (df["velocity"] < 200) if "velocity" in df.columns else True
        ]
        
        cleaned_count = len(df)
        logger.info(f"Cleaned data: {initial_count} -> {cleaned_count} records")
        
        return df
    
    async def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract trajectory features."""
        logger.info("Extracting trajectory features")
        
        # Group by vehicle to extract trajectory-level features
        enhanced_records = []
        
        for vehicle_id, vehicle_df in df.groupby("vehicle_id"):
            vehicle_df = vehicle_df.sort_values("timestamp")
            
            # Extract features for this vehicle's trajectory
            features = await self.feature_extractor.extract_trajectory_features(vehicle_df)
            
            # Add features to each record
            for idx, record in vehicle_df.iterrows():
                record_dict = record.to_dict()
                record_dict.update(features)
                enhanced_records.append(record_dict)
        
        enhanced_df = pd.DataFrame(enhanced_records)
        logger.info(f"Added features, new shape: {enhanced_df.shape}")
        
        return enhanced_df


class ParquetProcessor(DataProcessor):
    """Processor for saving data in Parquet format with partitioning and compression."""
    
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.compression = config.get("compression", "snappy")
        self.partition_cols = config.get("partition_columns", ["dataset"])
    
    async def save_partitioned(
        self, 
        df: pd.DataFrame, 
        output_path: Path
    ) -> Dict[str, Any]:
        """Save DataFrame as partitioned Parquet files."""
        output_path.mkdir(parents=True, exist_ok=True)
        
        files_created = 0
        
        # Intelligent partitioning
        if any(col in df.columns for col in self.partition_cols):
            available_partition_cols = [col for col in self.partition_cols if col in df.columns]
            
            if len(available_partition_cols) == 1:
                partition_col = available_partition_cols[0]
                for partition_value in df[partition_col].unique():
                    partition_df = df[df[partition_col] == partition_value]
                    partition_path = output_path / f"{partition_col}={partition_value}"
                    partition_path.mkdir(exist_ok=True)
                    
                    parquet_file = partition_path / "data.parquet"
                    partition_df.to_parquet(
                        parquet_file, 
                        index=False, 
                        compression=self.compression,
                        engine="pyarrow"
                    )
                    
                    files_created += 1
                    logger.info(f"Saved {len(partition_df)} records to {parquet_file}")
            else:
                # Multi-level partitioning
                for partition_combo, partition_df in df.groupby(available_partition_cols):
                    if not isinstance(partition_combo, tuple):
                        partition_combo = (partition_combo,)
                    
                    partition_path = output_path
                    for col, value in zip(available_partition_cols, partition_combo):
                        partition_path = partition_path / f"{col}={value}"
                    
                    partition_path.mkdir(parents=True, exist_ok=True)
                    parquet_file = partition_path / "data.parquet"
                    
                    partition_df.to_parquet(
                        parquet_file, 
                        index=False, 
                        compression=self.compression,
                        engine="pyarrow"
                    )
                    
                    files_created += 1
                    logger.info(f"Saved {len(partition_df)} records to {parquet_file}")
        else:
            # No partitioning
            parquet_file = output_path / "data.parquet"
            df.to_parquet(
                parquet_file, 
                index=False, 
                compression=self.compression,
                engine="pyarrow"
            )
            files_created = 1
            logger.info(f"Saved {len(df)} records to {parquet_file}")
        
        return {
            "files_created": files_created,
            "records_saved": len(df),
            "output_path": str(output_path),
            "compression": self.compression
        }


class DuckDBProcessor(DataProcessor):
    """High-performance processor using DuckDB for analytics."""
    
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.db_path = config.get("duckdb_path", ":memory:")
        self.conn = None
    
    async def __aenter__(self):
        self.conn = duckdb.connect(self.db_path)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()
    
    async def create_trajectory_table(self, df: pd.DataFrame) -> None:
        """Create optimized trajectory table in DuckDB."""
        if not self.conn:
            self.conn = duckdb.connect(self.db_path)
        
        # Create table with appropriate types and indexes
        self.conn.execute("""
            CREATE OR REPLACE TABLE trajectories AS
            SELECT * FROM df
        """)
        
        # Create indexes for common queries
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_vehicle_time 
            ON trajectories(vehicle_id, timestamp)
        """)
        
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_spatial 
            ON trajectories(x, y)
        """)
        
        logger.info(f"Created trajectory table with {len(df)} records")
    
    async def aggregate_trajectories(self) -> pd.DataFrame:
        """Aggregate trajectory data for analysis."""
        if not self.conn:
            raise RuntimeError("DuckDB connection not established")
        
        query = """
        SELECT 
            vehicle_id,
            COUNT(*) as num_points,
            MIN(timestamp) as start_time,
            MAX(timestamp) as end_time,
            AVG(velocity) as avg_velocity,
            MAX(velocity) as max_velocity,
            STDDEV(velocity) as velocity_std,
            AVG(acceleration) as avg_acceleration,
            STDDEV(acceleration) as acceleration_std,
            MAX(x) - MIN(x) as x_range,
            MAX(y) - MIN(y) as y_range,
            SQRT(POWER(MAX(x) - MIN(x), 2) + POWER(MAX(y) - MIN(y), 2)) as total_distance
        FROM trajectories 
        GROUP BY vehicle_id
        ORDER BY num_points DESC
        """
        
        result_df = self.conn.execute(query).fetchdf()
        logger.info(f"Generated trajectory aggregations for {len(result_df)} vehicles")
        
        return result_df
    
    async def export_to_parquet(self, output_path: Path, query: Optional[str] = None) -> None:
        """Export query results to Parquet."""
        if not query:
            query = "SELECT * FROM trajectories"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.conn.execute(f"""
            COPY ({query}) TO '{output_path}' 
            (FORMAT PARQUET, COMPRESSION SNAPPY)
        """)
        
        logger.info(f"Exported data to {output_path}")


class PolarsProcessor(DataProcessor):
    """High-performance processor using Polars for large datasets."""
    
    async def process_lazy(self, data: List[Dict[str, Any]]) -> pl.LazyFrame:
        """Process data using Polars lazy evaluation."""
        # Convert to Polars DataFrame
        df = pl.DataFrame(data)
        
        # Create lazy frame for optimized processing
        lazy_df = (
            df.lazy()
            .drop_nulls(subset=["vehicle_id", "x", "y", "timestamp"])
            .filter(
                (pl.col("x").abs() < 1e6) & 
                (pl.col("y").abs() < 1e6)
            )
            .sort(["vehicle_id", "timestamp"])
            .with_columns([
                # Add derived columns
                pl.col("timestamp").dt.hour().alias("hour"),
                pl.col("timestamp").dt.dayofweek().alias("day_of_week"),
                # Calculate time differences
                (pl.col("timestamp") - pl.col("timestamp").shift(1))
                .over("vehicle_id").dt.seconds().alias("time_diff"),
            ])
        )
        
        logger.info("Created lazy processing pipeline")
        return lazy_df
    
    async def compute_trajectory_stats(self, lazy_df: pl.LazyFrame) -> pl.DataFrame:
        """Compute trajectory statistics using Polars."""
        stats = (
            lazy_df
            .group_by("vehicle_id")
            .agg([
                pl.count().alias("num_points"),
                pl.col("velocity").mean().alias("avg_velocity"),
                pl.col("velocity").max().alias("max_velocity"),
                pl.col("velocity").std().alias("velocity_std"),
                pl.col("acceleration").mean().alias("avg_acceleration"),
                pl.col("x").max().sub(pl.col("x").min()).alias("x_range"),
                pl.col("y").max().sub(pl.col("y").min()).alias("y_range"),
            ])
            .collect()
        )
        
        logger.info(f"Computed statistics for {len(stats)} vehicles")
        return stats