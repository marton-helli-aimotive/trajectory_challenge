"""
Async ETL pipeline for trajectory data processing.

Implements a scalable ETL pipeline with:
- Async data ingestion using aiohttp
- Columnar storage with Parquet partitioning
- Incremental loading with change data capture
- Data quality validation
- Feature extraction with temporal windowing
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Union

import aiofiles
import pandas as pd
from omegaconf import DictConfig

from ..sources.base import DataSource
from ..validation.schemas import TrajectoryData
from .extractors import DataExtractor, ParallelDataExtractor
from .processors import DataProcessor, ParquetProcessor, DuckDBProcessor
from ...features.trajectory_features import TrajectoryFeatureExtractor, SequenceFeatureExtractor
from ...utils.data_quality import DataQualityMonitor

logger = logging.getLogger(__name__)


class ETLProcessor(Protocol):
    """Protocol for ETL processing components."""
    
    async def process(self, data: Any) -> Any:
        """Process data asynchronously."""
        ...


class TrajectoryETLPipeline:
    """
    Enhanced async ETL pipeline for trajectory data processing.
    
    Features:
    - Concurrent data ingestion with multiple sources
    - Advanced data processing with feature extraction
    - Columnar storage with intelligent partitioning
    - Incremental loading with change detection
    - Comprehensive data validation and quality monitoring
    - Temporal windowing for sequence data
    """
    
    def __init__(
        self, 
        config: DictConfig,
        data_sources: Union[DataSource, List[DataSource]],
        max_concurrent: int = 50
    ):
        self.config = config
        self.data_sources = data_sources if isinstance(data_sources, list) else [data_sources]
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        # Initialize components
        self.extractor = ParallelDataExtractor(config)
        self.data_processor = DataProcessor(config)
        self.parquet_processor = ParquetProcessor(config)
        self.quality_monitor = DataQualityMonitor(config)
        self.feature_extractor = TrajectoryFeatureExtractor(config)
        self.sequence_extractor = SequenceFeatureExtractor(config)
        
        # Setup paths
        self.raw_data_path = Path(config.paths.data.raw)
        self.processed_data_path = Path(config.paths.data.processed)
        self.features_data_path = Path(config.paths.data.features)
        
        # Ensure directories exist
        for path in [self.raw_data_path, self.processed_data_path, self.features_data_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Pipeline state
        self.pipeline_state = {
            "last_run": None,
            "records_processed": 0,
            "quality_score": 0.0
        }
        
    async def run(
        self, 
        force_refresh: bool = False,
        batch_size: int = 1000,
        extract_features: bool = True,
        extract_sequences: bool = True,
        quality_check: bool = True
    ) -> Dict[str, Any]:
        """
        Run the complete enhanced ETL pipeline.
        
        Args:
            force_refresh: If True, reprocess all data
            batch_size: Number of records per batch
            extract_features: Whether to extract trajectory features
            extract_sequences: Whether to extract sequence features
            quality_check: Whether to perform quality assessment
            
        Returns:
            Comprehensive pipeline execution summary
        """
        logger.info("Starting enhanced trajectory ETL pipeline")
        start_time = asyncio.get_event_loop().time()
        
        try:
            summary = {
                "status": "success",
                "start_time": datetime.now().isoformat(),
                "stages": {}
            }
            
            # Stage 1: Extract data from multiple sources
            logger.info("Stage 1: Data extraction")
            stage_start = asyncio.get_event_loop().time()
            raw_data_dict = await self._extract_data_parallel()
            total_raw_records = sum(len(data) for data in raw_data_dict.values())
            summary["stages"]["extraction"] = {
                "duration": asyncio.get_event_loop().time() - stage_start,
                "sources_processed": len(raw_data_dict),
                "total_records": total_raw_records
            }
            
            # Stage 2: Data processing and validation
            logger.info("Stage 2: Data processing and validation")
            stage_start = asyncio.get_event_loop().time()
            processed_df = await self._process_data_comprehensive(raw_data_dict, batch_size)
            summary["stages"]["processing"] = {
                "duration": asyncio.get_event_loop().time() - stage_start,
                "records_processed": len(processed_df),
                "data_shape": list(processed_df.shape)
            }
            
            # Stage 3: Quality assessment
            if quality_check:
                logger.info("Stage 3: Quality assessment")
                stage_start = asyncio.get_event_loop().time()
                quality_report = await self.quality_monitor.assess_quality(processed_df)
                summary["stages"]["quality_assessment"] = {
                    "duration": asyncio.get_event_loop().time() - stage_start,
                    "overall_score": quality_report["overall_score"],
                    "recommendations": quality_report["recommendations"]
                }
            
            # Stage 4: Feature extraction
            features_summary = {}
            if extract_features:
                logger.info("Stage 4: Feature extraction")
                stage_start = asyncio.get_event_loop().time()
                features_summary = await self._extract_trajectory_features(processed_df)
                summary["stages"]["feature_extraction"] = {
                    "duration": asyncio.get_event_loop().time() - stage_start,
                    **features_summary
                }
            
            # Stage 5: Sequence feature extraction
            if extract_sequences:
                logger.info("Stage 5: Sequence feature extraction")
                stage_start = asyncio.get_event_loop().time()
                sequence_summary = await self._extract_sequence_features(processed_df)
                summary["stages"]["sequence_extraction"] = {
                    "duration": asyncio.get_event_loop().time() - stage_start,
                    **sequence_summary
                }
            
            # Stage 6: Data loading with partitioning
            logger.info("Stage 6: Data loading")
            stage_start = asyncio.get_event_loop().time()
            load_summary = await self._load_data_comprehensive(
                processed_df, 
                features_summary.get("features_df"), 
                summary["stages"].get("sequence_extraction", {}).get("sequences_df")
            )
            summary["stages"]["loading"] = {
                "duration": asyncio.get_event_loop().time() - stage_start,
                **load_summary
            }
            
            # Update pipeline state
            self.pipeline_state.update({
                "last_run": datetime.now().isoformat(),
                "records_processed": len(processed_df),
                "quality_score": summary["stages"].get("quality_assessment", {}).get("overall_score", 0.0)
            })
            
            # Final summary
            end_time = asyncio.get_event_loop().time()
            summary.update({
                "end_time": datetime.now().isoformat(),
                "total_duration_seconds": end_time - start_time,
                "pipeline_state": self.pipeline_state
            })
            
            logger.info(f"Enhanced ETL pipeline completed successfully in {summary['total_duration_seconds']:.2f}s")
            return summary
            
        except Exception as e:
            logger.error(f"Enhanced ETL pipeline failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "duration_seconds": asyncio.get_event_loop().time() - start_time
            }
    
    async def _extract_data_parallel(self) -> Dict[str, List[Dict[str, Any]]]:
        """Extract data from multiple sources in parallel."""
        return await self.extractor.extract_parallel(self.data_sources)
    
    async def _process_data_comprehensive(
        self, 
        raw_data_dict: Dict[str, List[Dict[str, Any]]], 
        batch_size: int
    ) -> pd.DataFrame:
        """Process raw data comprehensively with validation and cleaning."""
        all_raw_data = []
        for source_name, data in raw_data_dict.items():
            # Add source identifier to each record
            for record in data:
                record["source"] = source_name
            all_raw_data.extend(data)
        
        # Process data using enhanced processor
        return await self.data_processor.process(all_raw_data)
    
    async def _extract_trajectory_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract trajectory-level features."""
        features_list = []
        vehicles_processed = 0
        
        for vehicle_id, vehicle_df in df.groupby("vehicle_id"):
            try:
                features = await self.feature_extractor.extract_trajectory_features(vehicle_df)
                features["vehicle_id"] = vehicle_id
                features_list.append(features)
                vehicles_processed += 1
            except Exception as e:
                logger.warning(f"Failed to extract features for vehicle {vehicle_id}: {e}")
        
        features_df = pd.DataFrame(features_list) if features_list else pd.DataFrame()
        
        return {
            "vehicles_processed": vehicles_processed,
            "features_extracted": len(features_df.columns) if not features_df.empty else 0,
            "features_df": features_df
        }
    
    async def _extract_sequence_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract sequence-level features using sliding windows."""
        all_sequences = []
        vehicles_processed = 0
        
        for vehicle_id, vehicle_df in df.groupby("vehicle_id"):
            try:
                sequences = await self.sequence_extractor.extract_sequence_features(vehicle_df)
                all_sequences.extend(sequences)
                vehicles_processed += 1
            except Exception as e:
                logger.warning(f"Failed to extract sequences for vehicle {vehicle_id}: {e}")
        
        sequences_df = pd.DataFrame(all_sequences) if all_sequences else pd.DataFrame()
        
        return {
            "vehicles_processed": vehicles_processed,
            "sequences_extracted": len(all_sequences),
            "sequences_df": sequences_df
        }
    
    async def _load_data_comprehensive(
        self, 
        trajectories_df: pd.DataFrame,
        features_df: Optional[pd.DataFrame] = None,
        sequences_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """Load all processed data with intelligent partitioning."""
        load_results = {}
        
        # Load main trajectory data
        if not trajectories_df.empty:
            traj_result = await self.parquet_processor.save_partitioned(
                trajectories_df, 
                self.processed_data_path / "trajectories"
            )
            load_results["trajectories"] = traj_result
        
        # Load trajectory features
        if features_df is not None and not features_df.empty:
            features_result = await self.parquet_processor.save_partitioned(
                features_df,
                self.features_data_path / "trajectory_features"
            )
            load_results["features"] = features_result
        
        # Load sequence features
        if sequences_df is not None and not sequences_df.empty:
            sequences_result = await self.parquet_processor.save_partitioned(
                sequences_df,
                self.features_data_path / "sequence_features"
            )
            load_results["sequences"] = sequences_result
        
        return load_results
    
    async def _transform_data(
        self, 
        raw_data: List[Dict[str, Any]], 
        batch_size: int
    ) -> List[TrajectoryData]:
        """Transform raw data with validation and quality checks."""
        validated_data = []
        
        # Process in batches to manage memory
        for i in range(0, len(raw_data), batch_size):
            batch = raw_data[i:i + batch_size]
            
            # Validate batch concurrently
            batch_tasks = [
                self._validate_trajectory_record(record) 
                for record in batch
            ]
            
            # Limit concurrent validation
            async with self.semaphore:
                batch_results = await asyncio.gather(
                    *batch_tasks, 
                    return_exceptions=True
                )
            
            # Filter valid results
            for result in batch_results:
                if isinstance(result, TrajectoryData):
                    validated_data.append(result)
                elif isinstance(result, Exception):
                    logger.warning(f"Validation failed: {result}")
        
        logger.info(f"Validated {len(validated_data)}/{len(raw_data)} records")
        return validated_data
    
    async def _validate_trajectory_record(
        self, 
        record: Dict[str, Any]
    ) -> TrajectoryData:
        """Validate a single trajectory record."""
        return TrajectoryData(**record)
    
    async def _load_data(
        self, 
        processed_data: List[TrajectoryData]
    ) -> Dict[str, Any]:
        """Load processed data with partitioning strategy."""
        # Convert to DataFrame
        df_data = [record.model_dump() for record in processed_data]
        df = pd.DataFrame(df_data)
        
        # Load with partitioning
        return await self.processor.save_partitioned(
            df, 
            self.processed_data_path / "trajectories"
        )
    
    async def get_incremental_data(
        self, 
        last_update: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get incremental data updates since last_update timestamp.
        
        Args:
            last_update: ISO timestamp string
            
        Returns:
            List of new/updated trajectory records
        """
        return await self.data_source.get_incremental_data(last_update)
    
    async def validate_data_quality(
        self, 
        data: List[TrajectoryData]
    ) -> Dict[str, Any]:
        """
        Validate data quality metrics.
        
        Returns:
            Quality metrics and validation results
        """
        if not data:
            return {"status": "error", "message": "No data to validate"}
        
        # Convert to DataFrame for analysis
        df_data = [record.model_dump() for record in data]
        df = pd.DataFrame(df_data)
        
        quality_metrics = {
            "total_records": len(df),
            "completeness": {
                "vehicle_id": df["vehicle_id"].notna().sum() / len(df),
                "timestamp": df["timestamp"].notna().sum() / len(df),
                "position": df[["x", "y"]].notna().all(axis=1).sum() / len(df),
            },
            "temporal_consistency": self._check_temporal_consistency(df),
            "spatial_accuracy": self._check_spatial_accuracy(df),
            "trajectory_smoothness": self._check_trajectory_smoothness(df)
        }
        
        return quality_metrics
    
    def _check_temporal_consistency(self, df: pd.DataFrame) -> Dict[str, float]:
        """Check temporal consistency of trajectory data."""
        # Group by vehicle and check timestamp ordering
        temporal_issues = 0
        total_vehicles = df["vehicle_id"].nunique()
        
        for vehicle_id in df["vehicle_id"].unique():
            vehicle_data = df[df["vehicle_id"] == vehicle_id].copy()
            vehicle_data = vehicle_data.sort_values("timestamp")
            
            # Check for time gaps or overlaps
            time_diffs = vehicle_data["timestamp"].diff().dt.total_seconds()
            if (time_diffs < 0).any() or (time_diffs > 1.0).any():  # 1 second max gap
                temporal_issues += 1
        
        return {
            "consistency_score": 1.0 - (temporal_issues / total_vehicles),
            "vehicles_with_issues": temporal_issues
        }
    
    def _check_spatial_accuracy(self, df: pd.DataFrame) -> Dict[str, float]:
        """Check spatial accuracy of trajectory data."""
        # Basic spatial validation
        invalid_coords = (
            (df["x"].abs() > 1e6) | 
            (df["y"].abs() > 1e6) |
            df["x"].isna() | 
            df["y"].isna()
        ).sum()
        
        return {
            "accuracy_score": 1.0 - (invalid_coords / len(df)),
            "invalid_coordinates": invalid_coords
        }
    
    def _check_trajectory_smoothness(self, df: pd.DataFrame) -> Dict[str, float]:
        """Check trajectory smoothness (velocity/acceleration consistency)."""
        smoothness_scores = []
        
        for vehicle_id in df["vehicle_id"].unique():
            vehicle_data = df[df["vehicle_id"] == vehicle_id].copy()
            vehicle_data = vehicle_data.sort_values("timestamp")
            
            if len(vehicle_data) < 3:
                continue
                
            # Calculate velocity changes
            x_vel = vehicle_data["x"].diff() / vehicle_data["timestamp"].diff().dt.total_seconds()
            y_vel = vehicle_data["y"].diff() / vehicle_data["timestamp"].diff().dt.total_seconds()
            
            # Calculate acceleration changes
            x_acc = x_vel.diff() / vehicle_data["timestamp"].diff().dt.total_seconds()[1:]
            y_acc = y_vel.diff() / vehicle_data["timestamp"].diff().dt.total_seconds()[1:]
            
            # Score based on acceleration variance (lower = smoother)
            acc_variance = (x_acc.var() + y_acc.var()) / 2
            smoothness_score = 1.0 / (1.0 + acc_variance) if not pd.isna(acc_variance) else 0.0
            smoothness_scores.append(smoothness_score)
        
        avg_smoothness = sum(smoothness_scores) / len(smoothness_scores) if smoothness_scores else 0.0
        
        return {
            "smoothness_score": avg_smoothness,
            "vehicles_analyzed": len(smoothness_scores)
        }