"""
Data processing and ETL pipeline components.

This package provides:
- Async ETL pipeline for trajectory data ingestion
- Data source factory pattern for multiple datasets
- Columnar storage with Apache Parquet
- Data validation and quality checks
"""

from .etl.pipeline import TrajectoryETLPipeline
from .sources.factory import DataSourceFactory
from .validation.schemas import TrajectoryData

__all__ = ["TrajectoryETLPipeline", "DataSourceFactory", "TrajectoryData"]