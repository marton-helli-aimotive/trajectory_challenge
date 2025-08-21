"""NGSIM dataset integration with async ETL pipeline support.

This module provides utilities for loading and processing NGSIM (Next Generation
Simulation) trajectory datasets into our standardized data models with full
ETL pipeline integration.

NGSIM is a Federal Highway Administration program that collected detailed
vehicle trajectory data on real freeway segments.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from .etl import ETLPipeline

import pandas as pd

from .etl import DataExtractor, DataTransformer, ETLConfig
from .models import Dataset, Trajectory, TrajectoryPoint, Vehicle, VehicleType

logger = logging.getLogger(__name__)


class NGSIMDataExtractor(DataExtractor):
    """Async extractor for NGSIM trajectory data."""

    def __init__(self, config: ETLConfig, data_sources: list[str | Path]):
        super().__init__(config)
        self.data_sources = [Path(source) for source in data_sources]

    def extract(self) -> AsyncIterator[dict[str, Any]]:
        """Extract raw NGSIM data from files."""
        return self._extract_files()

    async def _extract_files(self) -> AsyncIterator[dict[str, Any]]:
        """Async generator for extracting data from multiple files."""
        for file_path in self.data_sources:
            logger.info(f"Extracting data from {file_path}")
            async for record in self._extract_file(file_path):
                yield record

    async def _extract_file(self, file_path: Path) -> AsyncIterator[dict[str, Any]]:
        """Extract data from a single NGSIM file."""
        # Run pandas read in executor to avoid blocking
        loop = asyncio.get_event_loop()
        df = await loop.run_in_executor(None, self._read_ngsim_file, file_path)

        # Process in chunks to avoid memory issues
        chunk_size = self.config.chunk_size
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i : i + chunk_size]
            for _, row in chunk.iterrows():
                yield row.to_dict()

    def _read_ngsim_file(self, file_path: Path) -> pd.DataFrame:
        """Read NGSIM file with proper column handling."""
        try:
            # NGSIM files are typically space-separated
            if file_path.suffix.lower() == ".csv":
                df = pd.read_csv(file_path)
            else:
                # Try space-separated format (typical NGSIM format)
                df = pd.read_csv(file_path, sep=r"\s+")

            logger.info(f"Loaded {len(df)} records from {file_path}")
            return df

        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}")
            raise


class NGSIMDataTransformer(DataTransformer):
    """Transform NGSIM raw data into validated trajectory objects."""

    def __init__(self, config: ETLConfig, ngsim_config: dict[str, Any] | None = None):
        super().__init__(config)
        self.ngsim_config = ngsim_config or {}
        self.column_mapping = self._get_column_mapping()
        self.vehicle_cache: dict[int, Vehicle] = {}
        self.trajectory_cache: dict[int, list[dict[str, Any]]] = {}

    def _get_column_mapping(self) -> dict[str, str]:
        """Get mapping from NGSIM column names to our standard field names."""
        return {
            "Vehicle_ID": "vehicle_id",
            "Frame_ID": "frame_id",
            "Total_Frames": "total_frames",
            "Global_Time": "timestamp",
            "Local_X": "x",
            "Local_Y": "y",
            "Global_X": "global_x",
            "Global_Y": "global_y",
            "v_Length": "length",
            "v_Width": "width",
            "v_Class": "vehicle_class",
            "v_Vel": "speed",
            "v_Acc": "acceleration",
            "Lane_ID": "lane_id",
            "O_Zone": "origin_zone",
            "D_Zone": "destination_zone",
            "Int_ID": "intersection_id",
            "Section_ID": "section_id",
            "Direction": "direction",
            "Movement": "movement",
            "Preceding": "preceding_vehicle",
            "Following": "following_vehicle",
            "Space_Hdwy": "space_headway",
            "Time_Hdwy": "time_headway",
        }

    async def _transform_item(self, item: dict[str, Any]) -> Trajectory:
        """Transform a single NGSIM record into trajectory points."""
        # This method will be called for each record
        # We need to accumulate records by vehicle and then create trajectories
        raise NotImplementedError("Use transform_batch for NGSIM data")

    async def _transform_batch(
        self, batch: list[dict[str, Any]]
    ) -> AsyncIterator[Trajectory]:
        """Transform a batch of NGSIM records into trajectories."""
        # Group records by vehicle
        for item in batch:
            vehicle_id = item.get("Vehicle_ID")
            if vehicle_id is None:
                continue

            if vehicle_id not in self.trajectory_cache:
                self.trajectory_cache[vehicle_id] = []

            self.trajectory_cache[vehicle_id].append(item)

        # Create trajectories for vehicles with enough data
        min_points = self.ngsim_config.get("min_trajectory_length", 10)

        for vehicle_id, points_data in list(self.trajectory_cache.items()):
            if len(points_data) >= min_points:
                try:
                    trajectory = await self._create_trajectory(vehicle_id, points_data)
                    yield trajectory
                    # Remove processed trajectory from cache
                    del self.trajectory_cache[vehicle_id]
                except Exception as e:
                    logger.warning(
                        f"Failed to create trajectory for vehicle {vehicle_id}: {e}"
                    )

    async def _create_trajectory(
        self, vehicle_id: int, points_data: list[dict[str, Any]]
    ) -> Trajectory:
        """Create a trajectory from NGSIM points data."""
        # Sort by frame ID to ensure temporal order
        points_data.sort(key=lambda x: x.get("Frame_ID", 0))

        # Create vehicle object
        if vehicle_id not in self.vehicle_cache:
            first_point = points_data[0]
            vehicle = Vehicle(
                vehicle_id=vehicle_id,
                vehicle_type=self._map_vehicle_type(first_point.get("v_Class", 1)),
                length=first_point.get("v_Length", 4.5),
                width=first_point.get("v_Width", 1.8),
                metadata={"ngsim_class": first_point.get("v_Class")},
            )
            self.vehicle_cache[vehicle_id] = vehicle
        else:
            vehicle = self.vehicle_cache[vehicle_id]

        # Create trajectory points
        trajectory_points = []
        for point_data in points_data:
            try:
                point = self._create_trajectory_point(point_data)
                trajectory_points.append(point)
            except Exception as e:
                logger.debug(f"Skipping invalid point for vehicle {vehicle_id}: {e}")

        if len(trajectory_points) < 2:
            raise ValueError(f"Not enough valid points for vehicle {vehicle_id}")

        # Create trajectory with placeholder quality scores (computed later)
        trajectory = Trajectory(
            trajectory_id=f"ngsim_{vehicle_id}_{trajectory_points[0].frame_id}",
            vehicle=vehicle,
            points=trajectory_points,
            dataset_name="ngsim",
            completeness_score=1.0,  # Will be computed properly
            temporal_consistency_score=1.0,  # Will be computed properly
            spatial_accuracy_score=1.0,  # Will be computed properly
            smoothness_score=1.0,  # Will be computed properly
            metadata={
                "origin_zone": points_data[0].get("O_Zone"),
                "destination_zone": points_data[0].get("D_Zone"),
                "section_id": points_data[0].get("Section_ID"),
                "intersection_id": points_data[0].get("Int_ID"),
            },
        )

        return trajectory

    def _create_trajectory_point(self, point_data: dict[str, Any]) -> TrajectoryPoint:
        """Convert NGSIM data point to TrajectoryPoint."""
        # Handle timestamp conversion (NGSIM uses milliseconds)
        timestamp = point_data.get("Global_Time", 0)
        if timestamp > 1e10:  # Milliseconds
            timestamp = timestamp / 1000.0

        # Calculate velocities if not provided
        speed = point_data.get("v_Vel", 0.0)
        acceleration = point_data.get("v_Acc", 0.0)

        # Convert speed from mph to m/s if needed
        if speed > 50:  # Likely in mph
            speed = speed * 0.44704

        return TrajectoryPoint(
            timestamp=timestamp,
            x=point_data.get("Local_X", 0.0),
            y=point_data.get("Local_Y", 0.0),
            speed=abs(speed),  # Ensure positive speed
            velocity_x=speed,  # Simplified - assuming motion in X direction
            velocity_y=0.0,
            acceleration_x=acceleration,
            acceleration_y=0.0,
            heading=None,  # NGSIM doesn't typically include heading
            lane_id=point_data.get("Lane_ID"),
            frame_id=point_data.get("Frame_ID"),
        )

    def _map_vehicle_type(self, ngsim_class: int) -> VehicleType:
        """Map NGSIM vehicle class to our VehicleType enum."""
        # NGSIM classes: 1=motorcycle, 2=car, 3=truck
        mapping = {
            1: VehicleType.MOTORCYCLE,
            2: VehicleType.CAR,
            3: VehicleType.TRUCK,
        }
        return mapping.get(ngsim_class, VehicleType.OTHER)


class NGSIMDataLoader:
    """Legacy NGSIM data loader for backwards compatibility."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize NGSIM data loader with configuration."""
        self.config = config or {}

    def load_dataset(
        self, file_path: Path | str, dataset_name: str | None = None
    ) -> Dataset:
        """Load NGSIM dataset from file.

        Args:
            file_path: Path to NGSIM data file
            dataset_name: Optional name for the dataset

        Returns:
            Dataset object with loaded trajectories
        """
        logger.info(f"Loading NGSIM dataset from: {file_path}")

        if dataset_name is None:
            dataset_name = f"ngsim_{Path(file_path).stem}"

        # For simplicity in legacy loader, process data directly
        import asyncio

        config = ETLConfig()
        extractor = NGSIMDataExtractor(config, [file_path])
        transformer = NGSIMDataTransformer(config, self.config)

        # Collect trajectories directly
        trajectories = []

        async def collect_trajectories() -> None:
            # Extract data
            extracted_data = []
            async for item in extractor.extract():
                extracted_data.append(item)

            # Transform in batches
            batch_size = 100
            for i in range(0, len(extracted_data), batch_size):
                batch = extracted_data[i : i + batch_size]
                async for trajectory in transformer._transform_batch(batch):
                    trajectories.append(trajectory)

        asyncio.run(collect_trajectories())

        return Dataset(
            name=dataset_name,
            version="1.0",
            description=f"NGSIM dataset loaded from {file_path}",
            coordinate_system="cartesian",
            citation="NGSIM - Next Generation Simulation Program, FHWA",
            trajectories=trajectories,
            metadata={
                "source_file": str(file_path),
                "total_trajectories": len(trajectories),
                "loader_type": "legacy_ngsim_loader",
            },
        )


def load_ngsim_dataset(
    file_path: Path | str,
    dataset_name: str | None = None,
    config: dict[str, Any] | None = None,
) -> Dataset:
    """Convenience function to load NGSIM dataset.

    Args:
        file_path: Path to NGSIM data file
        dataset_name: Optional name for the dataset
        config: Optional configuration for the loader

    Returns:
        Dataset object with loaded trajectories
    """
    # This is a legacy function - use NGSIMDataSource for new implementations
    raise NotImplementedError("Use NGSIMDataSource.create_etl_pipeline() instead")


class NGSIMDataSource:
    """Data source implementation for NGSIM datasets."""

    def __init__(self, data_path: Path | str, config: dict[str, Any] | None = None):
        """Initialize NGSIM data source.

        Args:
            data_path: Path to NGSIM data file
            config: Optional configuration dict
        """
        self.data_path = Path(data_path)
        self.config = config or {}

    def validate_source(self, data_path: Path) -> bool:
        """Validate if data path is compatible with NGSIM format."""
        # Check file extension and basic structure
        return data_path.suffix.lower() in [".csv", ".txt"]

    def get_metadata(self) -> dict[str, Any]:
        """Get metadata about NGSIM data source."""
        return {
            "source_type": "ngsim",
            "description": "NGSIM trajectory dataset",
            "supported_formats": [".csv", ".txt"],
            "coordinate_system": "cartesian",
        }

    async def create_etl_pipeline(self, output_path: Path) -> ETLPipeline:
        """Create ETL pipeline for NGSIM data processing."""
        from .etl import DataLoader, ETLConfig, ETLPipeline

        etl_config = ETLConfig(
            chunk_size=self.config.get("chunk_size", 1000),
            max_concurrent_downloads=self.config.get("max_concurrent", 5),
            partition_cols=["dataset_name", "date"],
            compression="snappy",
        )

        extractor = NGSIMDataExtractor(etl_config, [self.data_path])
        transformer = NGSIMDataTransformer(etl_config, self.config)
        loader = DataLoader(etl_config, output_path)

        return ETLPipeline(extractor, transformer, loader)


async def load_ngsim_dataset_async(
    file_path: Path | str,
    output_path: Path | str,
    dataset_name: str | None = None,
    config: dict[str, Any] | None = None,
) -> None:
    """Async function to load NGSIM dataset with ETL pipeline.

    Args:
        file_path: Path to NGSIM data file
        output_path: Path to store processed Parquet files
        dataset_name: Optional name for the dataset
        config: Optional configuration for the loader
    """
    if dataset_name is None:
        dataset_name = f"ngsim_{Path(file_path).stem}"

    # Create and run ETL pipeline
    data_source = NGSIMDataSource(file_path, config)
    pipeline = await data_source.create_etl_pipeline(Path(output_path))
    await pipeline.run()


def load_ngsim_dataset_sync(
    file_path: Path | str,
    output_path: Path | str,
    dataset_name: str | None = None,
    config: dict[str, Any] | None = None,
) -> None:
    """Convenience function to load NGSIM dataset.

    Args:
        file_path: Path to NGSIM data file
        output_path: Path to store processed Parquet files
        dataset_name: Optional name for the dataset
        config: Optional configuration for the loader
    """
    asyncio.run(load_ngsim_dataset_async(file_path, output_path, dataset_name, config))
