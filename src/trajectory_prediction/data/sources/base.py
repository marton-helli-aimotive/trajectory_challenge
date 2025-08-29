"""
Base classes for trajectory data sources.

Defines the interface for different trajectory datasets using
the factory pattern for easy extensibility.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig


class DataSource(ABC):
    """
    Abstract base class for trajectory data sources.
    
    Implements common interface for different trajectory datasets
    (NGSIM, HighD, etc.) using factory pattern.
    """
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.name = self.__class__.__name__
        
    @abstractmethod
    async def load_data(self) -> List[Dict[str, Any]]:
        """
        Load trajectory data from source.
        
        Returns:
            List of trajectory records as dictionaries
        """
        pass
    
    @abstractmethod
    async def get_incremental_data(
        self, 
        last_update: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get incremental data updates.
        
        Args:
            last_update: ISO timestamp string for last update
            
        Returns:
            List of new/updated trajectory records
        """
        pass
    
    @abstractmethod
    def get_schema_info(self) -> Dict[str, Any]:
        """
        Get schema information for this data source.
        
        Returns:
            Schema metadata including column types, constraints
        """
        pass
    
    @abstractmethod
    async def validate_source(self) -> Dict[str, Any]:
        """
        Validate data source availability and integrity.
        
        Returns:
            Validation results and metadata
        """
        pass
    
    def get_source_info(self) -> Dict[str, Any]:
        """
        Get general information about this data source.
        
        Returns:
            Source metadata and configuration
        """
        return {
            "name": self.name,
            "config": self.config,
            "schema": self.get_schema_info()
        }