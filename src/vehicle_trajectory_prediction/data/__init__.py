"""Data processing and ETL pipeline for vehicle trajectory prediction."""

# Import components that don't require external dependencies
from .datasets import NGSIMDataset, DatasetFactory, DataSource, DatasetConfig
from .quality import DataQualityPipeline, QualityConfig, QualityMetrics

# Try to import components that require external dependencies
try:
    from .etl import AsyncETLPipeline, ETLConfig
    ETL_AVAILABLE = True
except ImportError:
    ETL_AVAILABLE = False
    AsyncETLPipeline = None
    ETLConfig = None

try:
    from .storage import ParquetStorage, StorageConfig
    STORAGE_AVAILABLE = True
except ImportError:
    STORAGE_AVAILABLE = False
    ParquetStorage = None
    StorageConfig = None

try:
    from .integration import CompleteETLPipeline
    INTEGRATION_AVAILABLE = True
except ImportError:
    INTEGRATION_AVAILABLE = False
    CompleteETLPipeline = None

__all__ = [
    "NGSIMDataset",
    "DatasetFactory",
    "DataSource",
    "DatasetConfig",
    "DataQualityPipeline",
    "QualityConfig",
    "QualityMetrics",
]

# Add optional components if available
if ETL_AVAILABLE:
    __all__.extend(["AsyncETLPipeline", "ETLConfig"])

if STORAGE_AVAILABLE:
    __all__.extend(["ParquetStorage", "StorageConfig"])

if INTEGRATION_AVAILABLE:
    __all__.append("CompleteETLPipeline")