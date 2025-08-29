"""
NGSIM dataset implementation.

Handles the NGSIM (Next Generation Simulation) trajectory dataset
with specific parsing and validation for US highway trajectory data.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiofiles
import aiohttp
import pandas as pd
from omegaconf import DictConfig

from .base import DataSource

logger = logging.getLogger(__name__)


class NGSIMDataSource(DataSource):
    """
    NGSIM trajectory dataset implementation.
    
    Handles data from the US Federal Highway Administration's
    Next Generation Simulation program, including I-80, US-101,
    and Lankershim Boulevard datasets.
    """
    
    # NGSIM dataset URLs (public domain)
    DATASET_URLS = {
        "i80": "https://www.fhwa.dot.gov/publications/research/operations/07030/ngsim_i80.csv",
        "us101": "https://www.fhwa.dot.gov/publications/research/operations/07030/ngsim_us101.csv",
        "lankershim": "https://www.fhwa.dot.gov/publications/research/operations/07030/ngsim_lankershim.csv"
    }
    
    # NGSIM column mapping to standardized schema
    COLUMN_MAPPING = {
        "Vehicle_ID": "vehicle_id",
        "Frame_ID": "frame_id", 
        "Total_Frames": "total_frames",
        "Global_Time": "timestamp",
        "Local_X": "x",
        "Local_Y": "y", 
        "Global_X": "global_x",
        "Global_Y": "global_y",
        "v_length": "vehicle_length",
        "v_Width": "vehicle_width",
        "v_Class": "vehicle_class",
        "v_Vel": "velocity",
        "v_Acc": "acceleration",
        "Lane_ID": "lane_id",
        "Preceding": "preceding_vehicle",
        "Following": "following_vehicle",
        "Space_Hdwy": "space_headway",
        "Time_Hdwy": "time_headway"
    }
    
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.dataset_name = config.get("dataset", "i80")
        self.data_path = Path(config.paths.data.raw) / "ngsim"
        self.data_path.mkdir(parents=True, exist_ok=True)
        
    async def load_data(self) -> List[Dict[str, Any]]:
        """
        Load NGSIM trajectory data.
        
        Downloads data if not available locally, then processes
        into standardized trajectory format.
        """
        logger.info(f"Loading NGSIM {self.dataset_name} dataset")
        
        # Ensure data is available
        await self._ensure_data_available()
        
        # Load and process data
        data_file = self.data_path / f"ngsim_{self.dataset_name}.csv"
        
        async with aiofiles.open(data_file, mode="r") as f:
            content = await f.read()
        
        # Parse CSV data
        df = pd.read_csv(data_file)
        
        # Standardize column names
        df = df.rename(columns=self.COLUMN_MAPPING)
        
        # Convert to trajectory records
        trajectories = await self._process_trajectories(df)
        
        logger.info(f"Loaded {len(trajectories)} trajectory records")
        return trajectories
    
    async def get_incremental_data(
        self, 
        last_update: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get incremental data updates.
        
        For NGSIM (static dataset), this returns empty list
        unless force refresh is needed.
        """
        # NGSIM is a static dataset, so no incremental updates
        if last_update is None:
            return await self.load_data()
        
        # Check if local data is newer than last_update
        data_file = self.data_path / f"ngsim_{self.dataset_name}.csv"
        if data_file.exists():
            file_mtime = data_file.stat().st_mtime
            # If file is newer, return data; otherwise empty
            # This is a simple implementation - could be more sophisticated
            return []
        
        return []
    
    def get_schema_info(self) -> Dict[str, Any]:
        """Get NGSIM schema information."""
        return {
            "source": "NGSIM",
            "description": "US highway trajectory data from FHWA NGSIM program",
            "columns": {
                "vehicle_id": {"type": "int", "description": "Unique vehicle identifier"},
                "frame_id": {"type": "int", "description": "Frame number in dataset"},
                "timestamp": {"type": "float", "description": "Global time in milliseconds"},
                "x": {"type": "float", "description": "Local X coordinate (feet)"},
                "y": {"type": "float", "description": "Local Y coordinate (feet)"},
                "global_x": {"type": "float", "description": "Global X coordinate"},
                "global_y": {"type": "float", "description": "Global Y coordinate"},
                "velocity": {"type": "float", "description": "Instantaneous velocity (ft/s)"},
                "acceleration": {"type": "float", "description": "Instantaneous acceleration (ft/s²)"},
                "vehicle_length": {"type": "float", "description": "Vehicle length (feet)"},
                "vehicle_width": {"type": "float", "description": "Vehicle width (feet)"},
                "vehicle_class": {"type": "int", "description": "Vehicle class (1=motorcycle, 2=auto, 3=truck)"},
                "lane_id": {"type": "int", "description": "Lane identifier"},
                "preceding_vehicle": {"type": "int", "description": "ID of preceding vehicle"},
                "following_vehicle": {"type": "int", "description": "ID of following vehicle"},
                "space_headway": {"type": "float", "description": "Space headway (feet)"},
                "time_headway": {"type": "float", "description": "Time headway (seconds)"}
            },
            "constraints": {
                "temporal_resolution": "0.1 seconds",
                "spatial_units": "feet",
                "coordinate_system": "local"
            }
        }
    
    async def validate_source(self) -> Dict[str, Any]:
        """Validate NGSIM data source."""
        validation_results = {
            "source_available": False,
            "data_integrity": False,
            "schema_compliance": False,
            "errors": []
        }
        
        try:
            # Check if dataset URL is accessible
            dataset_url = self.DATASET_URLS.get(self.dataset_name)
            if not dataset_url:
                validation_results["errors"].append(f"Unknown dataset: {self.dataset_name}")
                return validation_results
            
            # Check local data availability
            data_file = self.data_path / f"ngsim_{self.dataset_name}.csv"
            if data_file.exists():
                validation_results["source_available"] = True
                
                # Basic integrity check
                df = pd.read_csv(data_file, nrows=1000)  # Sample check
                expected_cols = set(self.COLUMN_MAPPING.keys())
                actual_cols = set(df.columns)
                
                if expected_cols.issubset(actual_cols):
                    validation_results["schema_compliance"] = True
                    validation_results["data_integrity"] = True
                else:
                    missing_cols = expected_cols - actual_cols
                    validation_results["errors"].append(f"Missing columns: {missing_cols}")
            else:
                # Try to access remote URL
                async with aiohttp.ClientSession() as session:
                    async with session.head(dataset_url) as response:
                        if response.status == 200:
                            validation_results["source_available"] = True
                        else:
                            validation_results["errors"].append(
                                f"Remote dataset not accessible: {response.status}"
                            )
        
        except Exception as e:
            validation_results["errors"].append(f"Validation error: {str(e)}")
        
        return validation_results
    
    async def _ensure_data_available(self) -> None:
        """Ensure NGSIM data is available locally."""
        data_file = self.data_path / f"ngsim_{self.dataset_name}.csv"
        
        if not data_file.exists():
            logger.info(f"Downloading NGSIM {self.dataset_name} dataset")
            await self._download_dataset()
    
    async def _download_dataset(self) -> None:
        """Download NGSIM dataset from official source."""
        dataset_url = self.DATASET_URLS[self.dataset_name]
        data_file = self.data_path / f"ngsim_{self.dataset_name}.csv"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(dataset_url) as response:
                if response.status == 200:
                    async with aiofiles.open(data_file, mode="wb") as f:
                        async for chunk in response.content.iter_chunked(8192):
                            await f.write(chunk)
                    logger.info(f"Downloaded NGSIM data to {data_file}")
                else:
                    raise RuntimeError(
                        f"Failed to download NGSIM data: HTTP {response.status}"
                    )
    
    async def _process_trajectories(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Process raw NGSIM data into standardized trajectory format.
        
        Args:
            df: Raw NGSIM DataFrame
            
        Returns:
            List of processed trajectory records
        """
        # Convert timestamp from milliseconds to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        
        # Filter out invalid records
        df = df.dropna(subset=["vehicle_id", "x", "y", "timestamp"])
        
        # Sort by vehicle and time
        df = df.sort_values(["vehicle_id", "timestamp"])
        
        # Convert to records
        trajectories = []
        
        # Process in batches to manage memory
        batch_size = 10000
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i + batch_size]
            batch_records = batch_df.to_dict("records")
            
            # Convert each record to proper types
            for record in batch_records:
                processed_record = await self._process_single_record(record)
                if processed_record:
                    trajectories.append(processed_record)
        
        return trajectories
    
    async def _process_single_record(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a single trajectory record.
        
        Args:
            record: Raw trajectory record
            
        Returns:
            Processed record or None if invalid
        """
        try:
            # Basic data cleaning and type conversion
            processed = {
                "vehicle_id": int(record["vehicle_id"]),
                "timestamp": record["timestamp"],
                "x": float(record["x"]),
                "y": float(record["y"]),
                "velocity": float(record.get("velocity", 0.0)),
                "acceleration": float(record.get("acceleration", 0.0)),
                "vehicle_length": float(record.get("vehicle_length", 16.0)),
                "vehicle_width": float(record.get("vehicle_width", 6.0)),
                "vehicle_class": int(record.get("vehicle_class", 2)),
                "lane_id": int(record.get("lane_id", 1)),
                
                # Optional fields
                "global_x": record.get("global_x"),
                "global_y": record.get("global_y"),
                "preceding_vehicle": record.get("preceding_vehicle"),
                "following_vehicle": record.get("following_vehicle"),
                "space_headway": record.get("space_headway"),
                "time_headway": record.get("time_headway"),
                
                # Metadata
                "dataset": "ngsim",
                "dataset_variant": self.dataset_name
            }
            
            # Basic validation
            if processed["x"] == 0 and processed["y"] == 0:
                return None  # Invalid position
            
            if abs(processed["velocity"]) > 200:  # > 200 ft/s is unrealistic
                processed["velocity"] = 0.0
            
            return processed
            
        except (ValueError, KeyError, TypeError) as e:
            logger.warning(f"Failed to process record: {e}")
            return None