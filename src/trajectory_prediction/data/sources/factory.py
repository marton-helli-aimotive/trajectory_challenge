"""
Data source factory for creating trajectory data sources.

Implements factory pattern for easy extensibility to new datasets.
"""

from typing import Dict, Type

from omegaconf import DictConfig

from .base import DataSource
from .ngsim import NGSIMDataSource


class DataSourceFactory:
    """
    Factory for creating trajectory data sources.
    
    Supports extensible registration of new data source types
    while maintaining a clean interface for the ETL pipeline.
    """
    
    _sources: Dict[str, Type[DataSource]] = {
        "ngsim": NGSIMDataSource,
    }
    
    @classmethod
    def create(cls, source_type: str, config: DictConfig) -> DataSource:
        """
        Create a data source instance.
        
        Args:
            source_type: Type of data source (e.g., "ngsim", "highd")
            config: Configuration for the data source
            
        Returns:
            Configured data source instance
            
        Raises:
            ValueError: If source_type is not supported
        """
        if source_type not in cls._sources:
            available = ", ".join(cls._sources.keys())
            raise ValueError(
                f"Unknown data source type: {source_type}. "
                f"Available: {available}"
            )
        
        source_class = cls._sources[source_type]
        return source_class(config)
    
    @classmethod
    def register(cls, source_type: str, source_class: Type[DataSource]) -> None:
        """
        Register a new data source type.
        
        Args:
            source_type: Name for the data source type
            source_class: DataSource implementation class
        """
        cls._sources[source_type] = source_class
    
    @classmethod
    def get_available_sources(cls) -> Dict[str, Type[DataSource]]:
        """
        Get all available data source types.
        
        Returns:
            Dictionary mapping source names to classes
        """
        return cls._sources.copy()
    
    @classmethod
    def is_supported(cls, source_type: str) -> bool:
        """
        Check if a data source type is supported.
        
        Args:
            source_type: Data source type to check
            
        Returns:
            True if supported, False otherwise
        """
        return source_type in cls._sources