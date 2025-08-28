"""Feature store implementation for vehicle trajectory prediction."""

import numpy as np
import pandas as pd
import pickle
import json
import hashlib
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import logging
import os

from ..core.config import BaseConfig
from ..core.types import TrajectoryData, TrajectoryPoint

logger = logging.getLogger(__name__)


@dataclass
class FeatureDefinition:
    """Definition of a feature for the feature store."""
    
    name: str
    description: str
    feature_type: str  # "velocity", "acceleration", "curvature", etc.
    data_type: str  # "float", "int", "array", "dict"
    shape: Optional[Tuple[int, ...]] = None
    default_value: Any = None
    validation_rules: Optional[Dict[str, Any]] = None
    dependencies: Optional[List[str]] = None
    created_at: Optional[datetime] = None
    version: str = "1.0.0"
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureDefinition':
        """Create from dictionary."""
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)
    
    def get_hash(self) -> str:
        """Get hash of feature definition."""
        data = self.to_dict()
        data['created_at'] = data['created_at'].isoformat() if data['created_at'] else None
        return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()


@dataclass
class FeatureVersion:
    """Version information for a feature."""
    
    feature_name: str
    version: str
    created_at: datetime
    description: str
    changes: List[str]
    is_active: bool = True
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureVersion':
        """Create from dictionary."""
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)


@dataclass
class FeatureCache:
    """Cache for computed features."""
    
    feature_name: str
    trajectory_id: str
    feature_data: Any
    computed_at: datetime
    expires_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['computed_at'] = self.computed_at.isoformat()
        if self.expires_at:
            data['expires_at'] = self.expires_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureCache':
        """Create from dictionary."""
        if 'computed_at' in data and isinstance(data['computed_at'], str):
            data['computed_at'] = datetime.fromisoformat(data['computed_at'])
        if 'expires_at' in data and isinstance(data['expires_at'], str):
            data['expires_at'] = datetime.fromisoformat(data['expires_at'])
        return cls(**data)


@dataclass
class FeatureStoreConfig(BaseConfig):
    """Configuration for feature store."""
    
    # Storage settings
    base_path: str = "data/feature_store"
    enable_caching: bool = True
    cache_expiration_hours: int = 24
    max_cache_size_mb: int = 1000
    
    # Feature computation settings
    enable_parallel_computation: bool = True
    max_workers: int = 4
    batch_size: int = 100
    
    # Versioning settings
    enable_versioning: bool = True
    max_versions_per_feature: int = 10
    
    # Validation settings
    enable_validation: bool = True
    strict_validation: bool = False
    
    # Performance settings
    enable_compression: bool = True
    compression_level: int = 6


class BaseFeatureStore(ABC):
    """Base class for feature store implementations."""
    
    def __init__(self, config: FeatureStoreConfig):
        self.config = config
        self.feature_definitions: Dict[str, FeatureDefinition] = {}
        self.feature_versions: Dict[str, List[FeatureVersion]] = {}
        self.cache: Dict[str, FeatureCache] = {}
        
    @abstractmethod
    def store_feature(self, feature_name: str, trajectory_id: str, feature_data: Any) -> bool:
        """Store a computed feature."""
        pass
    
    @abstractmethod
    def retrieve_feature(self, feature_name: str, trajectory_id: str) -> Optional[Any]:
        """Retrieve a stored feature."""
        pass
    
    @abstractmethod
    def list_features(self) -> List[str]:
        """List all available features."""
        pass
    
    @abstractmethod
    def delete_feature(self, feature_name: str, trajectory_id: str) -> bool:
        """Delete a stored feature."""
        pass


class LocalFeatureStore(BaseFeatureStore):
    """Local file-based feature store implementation."""
    
    def __init__(self, config: FeatureStoreConfig):
        super().__init__(config)
        self.base_path = Path(config.base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.base_path / "features").mkdir(exist_ok=True)
        (self.base_path / "definitions").mkdir(exist_ok=True)
        (self.base_path / "versions").mkdir(exist_ok=True)
        (self.base_path / "cache").mkdir(exist_ok=True)
        
        # Load existing definitions and versions
        self._load_definitions()
        self._load_versions()
        self._load_cache()
    
    def register_feature(self, feature_definition: FeatureDefinition) -> bool:
        """Register a new feature definition."""
        try:
            self.feature_definitions[feature_definition.name] = feature_definition
            
            # Save definition to file
            definition_path = self.base_path / "definitions" / f"{feature_definition.name}.json"
            with open(definition_path, 'w') as f:
                json.dump(feature_definition.to_dict(), f, indent=2)
            
            # Initialize version tracking
            if feature_definition.name not in self.feature_versions:
                self.feature_versions[feature_definition.name] = []
            
            logger.info(f"Registered feature: {feature_definition.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error registering feature {feature_definition.name}: {e}")
            return False
    
    def get_feature_definition(self, feature_name: str) -> Optional[FeatureDefinition]:
        """Get feature definition by name."""
        return self.feature_definitions.get(feature_name)
    
    def list_feature_definitions(self) -> List[FeatureDefinition]:
        """List all feature definitions."""
        return list(self.feature_definitions.values())
    
    def store_feature(self, feature_name: str, trajectory_id: str, feature_data: Any) -> bool:
        """Store a computed feature."""
        try:
            # Validate feature exists
            if feature_name not in self.feature_definitions:
                logger.error(f"Feature {feature_name} not registered")
                return False
            
            # Create feature directory
            feature_path = self.base_path / "features" / feature_name
            feature_path.mkdir(exist_ok=True)
            
            # Store feature data
            file_path = feature_path / f"{trajectory_id}.pkl"
            
            if self.config.enable_compression:
                import gzip
                with gzip.open(file_path, 'wb', compresslevel=self.config.compression_level) as f:
                    pickle.dump(feature_data, f)
            else:
                with open(file_path, 'wb') as f:
                    pickle.dump(feature_data, f)
            
            # Update cache
            if self.config.enable_caching:
                self._update_cache(feature_name, trajectory_id, feature_data)
            
            logger.debug(f"Stored feature {feature_name} for trajectory {trajectory_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing feature {feature_name} for trajectory {trajectory_id}: {e}")
            return False
    
    def retrieve_feature(self, feature_name: str, trajectory_id: str) -> Optional[Any]:
        """Retrieve a stored feature."""
        try:
            # Check cache first
            if self.config.enable_caching:
                cached_data = self._get_from_cache(feature_name, trajectory_id)
                if cached_data is not None:
                    return cached_data
            
            # Load from file
            feature_path = self.base_path / "features" / feature_name / f"{trajectory_id}.pkl"
            
            if not feature_path.exists():
                return None
            
            if self.config.enable_compression:
                import gzip
                with gzip.open(feature_path, 'rb') as f:
                    feature_data = pickle.load(f)
            else:
                with open(feature_path, 'rb') as f:
                    feature_data = pickle.load(f)
            
            # Update cache
            if self.config.enable_caching:
                self._update_cache(feature_name, trajectory_id, feature_data)
            
            return feature_data
            
        except Exception as e:
            logger.error(f"Error retrieving feature {feature_name} for trajectory {trajectory_id}: {e}")
            return None
    
    def list_features(self) -> List[str]:
        """List all available features."""
        return list(self.feature_definitions.keys())
    
    def list_trajectories_for_feature(self, feature_name: str) -> List[str]:
        """List all trajectories that have a specific feature computed."""
        try:
            feature_path = self.base_path / "features" / feature_name
            if not feature_path.exists():
                return []
            
            trajectory_ids = []
            for file_path in feature_path.glob("*.pkl"):
                trajectory_id = file_path.stem
                trajectory_ids.append(trajectory_id)
            
            return trajectory_ids
            
        except Exception as e:
            logger.error(f"Error listing trajectories for feature {feature_name}: {e}")
            return []
    
    def delete_feature(self, feature_name: str, trajectory_id: str) -> bool:
        """Delete a stored feature."""
        try:
            # Remove from file system
            feature_path = self.base_path / "features" / feature_name / f"{trajectory_id}.pkl"
            if feature_path.exists():
                feature_path.unlink()
            
            # Remove from cache
            if self.config.enable_caching:
                self._remove_from_cache(feature_name, trajectory_id)
            
            logger.debug(f"Deleted feature {feature_name} for trajectory {trajectory_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting feature {feature_name} for trajectory {trajectory_id}: {e}")
            return False
    
    def create_feature_version(self, feature_name: str, version: str, description: str, 
                             changes: List[str]) -> bool:
        """Create a new version for a feature."""
        try:
            if feature_name not in self.feature_definitions:
                logger.error(f"Feature {feature_name} not found")
                return False
            
            # Create version object
            feature_version = FeatureVersion(
                feature_name=feature_name,
                version=version,
                created_at=datetime.now(),
                description=description,
                changes=changes
            )
            
            # Add to versions
            if feature_name not in self.feature_versions:
                self.feature_versions[feature_name] = []
            
            self.feature_versions[feature_name].append(feature_version)
            
            # Save to file
            version_path = self.base_path / "versions" / f"{feature_name}_versions.json"
            versions_data = [v.to_dict() for v in self.feature_versions[feature_name]]
            with open(version_path, 'w') as f:
                json.dump(versions_data, f, indent=2)
            
            logger.info(f"Created version {version} for feature {feature_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating version for feature {feature_name}: {e}")
            return False
    
    def get_feature_versions(self, feature_name: str) -> List[FeatureVersion]:
        """Get all versions for a feature."""
        return self.feature_versions.get(feature_name, [])
    
    def get_latest_version(self, feature_name: str) -> Optional[FeatureVersion]:
        """Get the latest version for a feature."""
        versions = self.get_feature_versions(feature_name)
        if not versions:
            return None
        
        # Sort by creation date and return latest
        versions.sort(key=lambda v: v.created_at, reverse=True)
        return versions[0]
    
    def compute_feature_batch(self, feature_name: str, trajectories: List[TrajectoryData], 
                            compute_function: Callable[[TrajectoryData], Any]) -> Dict[str, Any]:
        """Compute features for a batch of trajectories."""
        results = {}
        
        if self.config.enable_parallel_computation:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            
            with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
                # Submit computation tasks
                future_to_trajectory = {
                    executor.submit(compute_function, trajectory): trajectory
                    for trajectory in trajectories
                }
                
                # Collect results
                for future in as_completed(future_to_trajectory):
                    trajectory = future_to_trajectory[future]
                    try:
                        feature_data = future.result()
                        results[trajectory.vehicle_id] = feature_data
                        
                        # Store feature
                        self.store_feature(feature_name, trajectory.vehicle_id, feature_data)
                        
                    except Exception as e:
                        logger.error(f"Error computing feature for trajectory {trajectory.vehicle_id}: {e}")
                        results[trajectory.vehicle_id] = None
        else:
            # Sequential computation
            for trajectory in trajectories:
                try:
                    feature_data = compute_function(trajectory)
                    results[trajectory.vehicle_id] = feature_data
                    
                    # Store feature
                    self.store_feature(feature_name, trajectory.vehicle_id, feature_data)
                    
                except Exception as e:
                    logger.error(f"Error computing feature for trajectory {trajectory.vehicle_id}: {e}")
                    results[trajectory.vehicle_id] = None
        
        return results
    
    def get_store_statistics(self) -> Dict[str, Any]:
        """Get statistics about the feature store."""
        stats = {
            "total_features": len(self.feature_definitions),
            "total_versions": sum(len(versions) for versions in self.feature_versions.values()),
            "cache_size": len(self.cache),
            "cache_hit_rate": 0.0,
            "storage_size_mb": 0.0
        }
        
        # Calculate storage size
        total_size = 0
        for feature_name in self.feature_definitions:
            feature_path = self.base_path / "features" / feature_name
            if feature_path.exists():
                for file_path in feature_path.glob("*.pkl"):
                    total_size += file_path.stat().st_size
        
        stats["storage_size_mb"] = total_size / (1024 * 1024)
        
        return stats
    
    def cleanup_cache(self) -> int:
        """Clean up expired cache entries."""
        if not self.config.enable_caching:
            return 0
        
        expired_keys = []
        for key, cache_entry in self.cache.items():
            if cache_entry.is_expired():
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
        
        logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
        return len(expired_keys)
    
    def _load_definitions(self):
        """Load feature definitions from disk."""
        definitions_path = self.base_path / "definitions"
        for file_path in definitions_path.glob("*.json"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    definition = FeatureDefinition.from_dict(data)
                    self.feature_definitions[definition.name] = definition
            except Exception as e:
                logger.error(f"Error loading definition from {file_path}: {e}")
    
    def _load_versions(self):
        """Load feature versions from disk."""
        versions_path = self.base_path / "versions"
        for file_path in versions_path.glob("*_versions.json"):
            try:
                feature_name = file_path.stem.replace("_versions", "")
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    versions = [FeatureVersion.from_dict(v) for v in data]
                    self.feature_versions[feature_name] = versions
            except Exception as e:
                logger.error(f"Error loading versions from {file_path}: {e}")
    
    def _load_cache(self):
        """Load cache from disk."""
        if not self.config.enable_caching:
            return
        
        cache_path = self.base_path / "cache" / "cache.json"
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                    for cache_data in data:
                        cache_entry = FeatureCache.from_dict(cache_data)
                        if not cache_entry.is_expired():
                            key = f"{cache_entry.feature_name}_{cache_entry.trajectory_id}"
                            self.cache[key] = cache_entry
            except Exception as e:
                logger.error(f"Error loading cache: {e}")
    
    def _save_cache(self):
        """Save cache to disk."""
        if not self.config.enable_caching:
            return
        
        try:
            cache_path = self.base_path / "cache" / "cache.json"
            cache_data = [entry.to_dict() for entry in self.cache.values()]
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
    
    def _update_cache(self, feature_name: str, trajectory_id: str, feature_data: Any):
        """Update cache with new feature data."""
        if not self.config.enable_caching:
            return
        
        # Calculate expiration time
        expires_at = datetime.now() + pd.Timedelta(hours=self.config.cache_expiration_hours)
        
        # Create cache entry
        cache_entry = FeatureCache(
            feature_name=feature_name,
            trajectory_id=trajectory_id,
            feature_data=feature_data,
            computed_at=datetime.now(),
            expires_at=expires_at
        )
        
        # Store in cache
        key = f"{feature_name}_{trajectory_id}"
        self.cache[key] = cache_entry
        
        # Check cache size limit
        self._enforce_cache_size_limit()
    
    def _get_from_cache(self, feature_name: str, trajectory_id: str) -> Optional[Any]:
        """Get feature data from cache."""
        if not self.config.enable_caching:
            return None
        
        key = f"{feature_name}_{trajectory_id}"
        cache_entry = self.cache.get(key)
        
        if cache_entry is None or cache_entry.is_expired():
            if cache_entry and cache_entry.is_expired():
                del self.cache[key]
            return None
        
        return cache_entry.feature_data
    
    def _remove_from_cache(self, feature_name: str, trajectory_id: str):
        """Remove feature from cache."""
        if not self.config.enable_caching:
            return
        
        key = f"{feature_name}_{trajectory_id}"
        if key in self.cache:
            del self.cache[key]
    
    def _enforce_cache_size_limit(self):
        """Enforce cache size limit by removing oldest entries."""
        if not self.config.enable_caching:
            return
        
        # Calculate current cache size
        cache_size_mb = sum(
            len(pickle.dumps(entry.feature_data)) / (1024 * 1024)
            for entry in self.cache.values()
        )
        
        if cache_size_mb > self.config.max_cache_size_mb:
            # Sort by computed_at and remove oldest entries
            sorted_entries = sorted(
                self.cache.items(),
                key=lambda x: x[1].computed_at
            )
            
            entries_to_remove = []
            current_size = cache_size_mb
            
            for key, entry in sorted_entries:
                entry_size = len(pickle.dumps(entry.feature_data)) / (1024 * 1024)
                if current_size - entry_size <= self.config.max_cache_size_mb:
                    break
                entries_to_remove.append(key)
                current_size -= entry_size
            
            # Remove entries
            for key in entries_to_remove:
                del self.cache[key]
            
            logger.info(f"Removed {len(entries_to_remove)} cache entries to enforce size limit")


class FeatureStore:
    """Main feature store interface."""
    
    def __init__(self, config: FeatureStoreConfig):
        self.config = config
        self.store = LocalFeatureStore(config)
    
    def register_feature(self, feature_definition: FeatureDefinition) -> bool:
        """Register a new feature definition."""
        return self.store.register_feature(feature_definition)
    
    def get_feature_definition(self, feature_name: str) -> Optional[FeatureDefinition]:
        """Get feature definition by name."""
        return self.store.get_feature_definition(feature_name)
    
    def list_feature_definitions(self) -> List[FeatureDefinition]:
        """List all feature definitions."""
        return self.store.list_feature_definitions()
    
    def store_feature(self, feature_name: str, trajectory_id: str, feature_data: Any) -> bool:
        """Store a computed feature."""
        return self.store.store_feature(feature_name, trajectory_id, feature_data)
    
    def retrieve_feature(self, feature_name: str, trajectory_id: str) -> Optional[Any]:
        """Retrieve a stored feature."""
        return self.store.retrieve_feature(feature_name, trajectory_id)
    
    def list_features(self) -> List[str]:
        """List all available features."""
        return self.store.list_features()
    
    def list_trajectories_for_feature(self, feature_name: str) -> List[str]:
        """List all trajectories that have a specific feature computed."""
        return self.store.list_trajectories_for_feature(feature_name)
    
    def delete_feature(self, feature_name: str, trajectory_id: str) -> bool:
        """Delete a stored feature."""
        return self.store.delete_feature(feature_name, trajectory_id)
    
    def create_feature_version(self, feature_name: str, version: str, description: str, 
                             changes: List[str]) -> bool:
        """Create a new version for a feature."""
        return self.store.create_feature_version(feature_name, version, description, changes)
    
    def get_feature_versions(self, feature_name: str) -> List[FeatureVersion]:
        """Get all versions for a feature."""
        return self.store.get_feature_versions(feature_name)
    
    def get_latest_version(self, feature_name: str) -> Optional[FeatureVersion]:
        """Get the latest version for a feature."""
        return self.store.get_latest_version(feature_name)
    
    def compute_feature_batch(self, feature_name: str, trajectories: List[TrajectoryData], 
                            compute_function: Callable[[TrajectoryData], Any]) -> Dict[str, Any]:
        """Compute features for a batch of trajectories."""
        return self.store.compute_feature_batch(feature_name, trajectories, compute_function)
    
    def get_store_statistics(self) -> Dict[str, Any]:
        """Get statistics about the feature store."""
        return self.store.get_store_statistics()
    
    def cleanup_cache(self) -> int:
        """Clean up expired cache entries."""
        return self.store.cleanup_cache()