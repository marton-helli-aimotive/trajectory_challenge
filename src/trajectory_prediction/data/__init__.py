"""Data processing and ETL pipeline package.

This package provides comprehensive data processing capabilities including:
- Async ETL pipeline with extractors, transformers, and loaders
- Data models and validation using Pydantic
- NGSIM dataset integration with schema mapping
- Data quality pipeline with cleaning and validation
- Incremental data loading with checkpointing
- Data source factory pattern for extensibility
"""

from .etl import DataExtractor, DataLoader, DataTransformer, ETLConfig, ETLPipeline
from .incremental import (
    CheckpointManager,
    FileHashTracker,
    IncrementalConfig,
    IncrementalETLPipeline,
    create_source_id,
)
from .models import (
    Dataset,
    Trajectory,
    TrajectoryPoint,
    Vehicle,
    VehicleType,
)
from .ngsim import (
    NGSIMDataExtractor,
    NGSIMDataLoader,
    NGSIMDataSource,
    NGSIMDataTransformer,
    load_ngsim_dataset_async,
    load_ngsim_dataset_sync,
)
from .quality import (
    DataQualityPipeline,
    DataQualityRule,
    MinimumPointsRule,
    OutlierDetectionRule,
    PhysicsConsistencyRule,
    TemporalConsistencyRule,
    interpolate_missing_points,
)
from .sources import DataSource, DataSourceFactory
from .validation import TrajectoryValidator

__all__ = [
    # Core ETL
    "ETLConfig",
    "ETLPipeline",
    "DataExtractor",
    "DataTransformer",
    "DataLoader",
    # Data Models
    "Dataset",
    "Trajectory",
    "TrajectoryPoint",
    "Vehicle",
    "VehicleType",
    # NGSIM Integration
    "NGSIMDataExtractor",
    "NGSIMDataTransformer",
    "NGSIMDataLoader",
    "NGSIMDataSource",
    "load_ngsim_dataset_async",
    "load_ngsim_dataset_sync",
    # Data Quality
    "DataQualityPipeline",
    "DataQualityRule",
    "MinimumPointsRule",
    "TemporalConsistencyRule",
    "PhysicsConsistencyRule",
    "OutlierDetectionRule",
    "interpolate_missing_points",
    # Incremental Loading
    "IncrementalETLPipeline",
    "IncrementalConfig",
    "CheckpointManager",
    "FileHashTracker",
    "create_source_id",
    # Data Sources
    "DataSource",
    "DataSourceFactory",
    # Validation
    "TrajectoryValidator",
]
