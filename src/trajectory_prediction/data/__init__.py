"""Data processing and ETL pipeline package.

This package provides comprehensive data processing capabilities including:
- Async ETL pipeline with extractors, transformers, and loaders
- Data models and validation using Pydantic
- NGSIM dataset integration with schema mapping
- Data quality pipeline with cleaning and validation
- Incremental data loading with checkpointing
- Data source factory pattern for extensibility
- Data downloaders for fetching trajectory datasets
- Feature engineering and extraction pipeline
- Feature store for reusable features
- Data augmentation techniques
- Feature validation and quality assessment
"""

from .augmentation import (
    AugmentationPipeline,
    NoiseInjectionAugmenter,
    SpatialTransformationAugmenter,
    TemporalShiftAugmenter,
    TrajectoryAugmenter,
    TrajectoryInterpolationAugmenter,
    create_default_augmentation_pipeline,
)
from .downloaders import (
    DownloadError,
    NGSIMDownloader,
    download_all_ngsim_datasets,
    download_ngsim_dataset,
)
from .etl import DataExtractor, DataLoader, DataTransformer, ETLConfig, ETLPipeline
from .feature_store import (
    BatchFeatureProcessor,
    FeatureStorage,
    FeatureStore,
    FileFeatureStorage,
    InMemoryFeatureStorage,
    create_default_feature_store,
)
from .feature_validation import (
    FeatureCorrelationAnalyzer,
    FeatureQualityAssessor,
    FeatureQualityReport,
    generate_feature_quality_summary,
)
from .features import (
    ContextualFeatureExtractor,
    FeatureExtractor,
    FeatureInfo,
    KinematicFeatureExtractor,
    QualityFeatureExtractor,
    TemporalFeatureExtractor,
)
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
    NGSIMDataSource,
    NGSIMDataTransformer,
    download_and_load_ngsim_dataset,
    load_ngsim_dataset_async,
    load_ngsim_dataset_sync,
    sync_download_and_load_ngsim_dataset,
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
    "NGSIMDataSource",
    "load_ngsim_dataset_async",
    "load_ngsim_dataset_sync",
    "download_and_load_ngsim_dataset",
    "sync_download_and_load_ngsim_dataset",
    # Data Downloaders
    "NGSIMDownloader",
    "DownloadError",
    "download_ngsim_dataset",
    "download_all_ngsim_datasets",
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
    # Feature Engineering
    "FeatureExtractor",
    "FeatureInfo",
    "KinematicFeatureExtractor",
    "TemporalFeatureExtractor",
    "ContextualFeatureExtractor",
    "QualityFeatureExtractor",
    # Feature Store
    "FeatureStore",
    "FeatureStorage",
    "FileFeatureStorage",
    "InMemoryFeatureStorage",
    "BatchFeatureProcessor",
    "create_default_feature_store",
    # Data Augmentation
    "TrajectoryAugmenter",
    "NoiseInjectionAugmenter",
    "TemporalShiftAugmenter",
    "TrajectoryInterpolationAugmenter",
    "SpatialTransformationAugmenter",
    "AugmentationPipeline",
    "create_default_augmentation_pipeline",
    # Feature Validation
    "FeatureQualityAssessor",
    "FeatureQualityReport",
    "FeatureCorrelationAnalyzer",
    "generate_feature_quality_summary",
]
