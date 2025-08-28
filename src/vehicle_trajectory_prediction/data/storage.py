"""Parquet-based storage layer for trajectory data with partitioning and versioning."""

# Try to import pandas
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

# Try to import pyarrow
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False
    pa = None
    pq = None
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    DataFrame = pd.DataFrame if PANDAS_AVAILABLE else Any
else:
    DataFrame = Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import json
import shutil

# Use a simple base class since BaseConfig doesn't exist
class BaseConfig:
    """Base configuration class."""
    pass
from ..core.models import TrajectoryPoint, Trajectory, TrajectoryDataset
from ..core.exceptions import StorageError
from ..core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class StorageConfig(BaseConfig):
    """Configuration for Parquet storage."""
    
    # Storage settings
    base_path: str = "data/storage"
    partition_by: List[str] = None  # Will be set in __post_init__
    compression: str = "snappy"
    row_group_size: int = 100000
    
    # Versioning settings
    enable_versioning: bool = True
    max_versions: int = 10
    version_metadata: bool = True
    
    # Performance settings
    use_threads: bool = True
    memory_map: bool = True
    pre_buffer: bool = True
    
    # Query optimization
    enable_indexing: bool = True
    index_columns: List[str] = None  # Will be set in __post_init__
    
    def __post_init__(self):
        """Set default values for partitioning and indexing."""
        if self.partition_by is None:
            self.partition_by = ['vehicle_id', 'date']
        
        if self.index_columns is None:
            self.index_columns = ['vehicle_id', 'timestamp', 'frame_id']


class ParquetStorage:
    """Parquet-based storage with partitioning and versioning."""
    
    def __init__(self, config: StorageConfig):
        """Initialize Parquet storage.
        
        Args:
            config: Storage configuration
        """
        self.config = config
        self.base_path = Path(config.base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.data_path = self.base_path / "data"
        self.metadata_path = self.base_path / "metadata"
        self.index_path = self.base_path / "indexes"
        
        for path in [self.data_path, self.metadata_path, self.index_path]:
            path.mkdir(exist_ok=True)
        
        logger.info("Initialized Parquet storage", 
                   base_path=str(self.base_path),
                   partition_by=config.partition_by,
                   compression=config.compression)
    
    def _get_partition_path(self, partition_values: Dict[str, Any]) -> Path:
        """Generate partition path from partition values.
        
        Args:
            partition_values: Dictionary of partition column values
            
        Returns:
            Partition directory path
        """
        partition_parts = []
        for col in self.config.partition_by:
            if col in partition_values:
                value = partition_values[col]
                # Handle date partitioning
                if col == 'date' and isinstance(value, (datetime, pd.Timestamp)):
                    value = value.strftime('%Y-%m-%d')
                partition_parts.append(f"{col}={value}")
            else:
                partition_parts.append(f"{col}=unknown")
        
        return self.data_path / "/".join(partition_parts)
    
    def _extract_partition_values(self, df: DataFrame) -> Dict[str, Any]:
        """Extract partition values from DataFrame.
        
        Args:
            df: DataFrame to extract partition values from
            
        Returns:
            Dictionary of partition values
        """
        partition_values = {}
        
        for col in self.config.partition_by:
            if col in df.columns:
                if col == 'date':
                    # Extract date from timestamp
                    if 'timestamp' in df.columns:
                        partition_values[col] = pd.to_datetime(df['timestamp']).dt.date.iloc[0]
                    else:
                        partition_values[col] = datetime.now().date()
                else:
                    partition_values[col] = df[col].iloc[0]
            else:
                partition_values[col] = 'unknown'
        
        return partition_values
    
    def _generate_filename(self, partition_values: Dict[str, Any], suffix: str = "") -> str:
        """Generate filename for partition.
        
        Args:
            partition_values: Partition values
            suffix: Optional suffix for filename
            
        Returns:
            Generated filename
        """
        # Create hash of partition values for uniqueness
        partition_str = json.dumps(partition_values, sort_keys=True)
        partition_hash = hashlib.md5(partition_str.encode()).hexdigest()[:8]
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if suffix:
            return f"trajectory_{timestamp}_{partition_hash}_{suffix}.parquet"
        else:
            return f"trajectory_{timestamp}_{partition_hash}.parquet"
    
    def store_trajectories(self, 
                          trajectories: List[Trajectory], 
                          dataset_name: str,
                          version: Optional[str] = None) -> Dict[str, Any]:
        """Store trajectories in partitioned Parquet format.
        
        Args:
            trajectories: List of trajectories to store
            dataset_name: Name of the dataset
            version: Optional version identifier
            
        Returns:
            Storage metadata
        """
        if not trajectories:
            raise StorageError("No trajectories provided")
        
        # Convert trajectories to DataFrame
        all_points = []
        for trajectory in trajectories:
            for point in trajectory.points:
                point_dict = {
                    'vehicle_id': trajectory.vehicle_id,
                    'timestamp': point.timestamp,
                    'x': point.x,
                    'y': point.y,
                    'velocity': point.velocity,
                    'acceleration': point.acceleration,
                    'heading': point.heading,
                    'lane_id': point.lane_id,
                    'frame_id': point.attributes.get('frame_id', 0),
                    'vehicle_class': point.attributes.get('vehicle_class', ''),
                    'vehicle_length': point.attributes.get('vehicle_length', 0.0),
                    'vehicle_width': point.attributes.get('vehicle_width', 0.0),
                }
                all_points.append(point_dict)
        
        if PANDAS_AVAILABLE:
            df = pd.DataFrame(all_points)
        else:
            raise ImportError("pandas is required for trajectory storage")
        
        # Add date column for partitioning
        df['date'] = pd.to_datetime(df['timestamp'], unit='s').dt.date
        
        # Group by partition values
        partition_groups = df.groupby(self.config.partition_by)
        
        stored_files = []
        total_rows = 0
        
        for partition_values, group_df in partition_groups:
            # Create partition directory
            partition_path = self._get_partition_path(partition_values)
            partition_path.mkdir(parents=True, exist_ok=True)
            
            # Generate filename
            filename = self._generate_filename(partition_values, version or "")
            file_path = partition_path / filename
            
            # Convert to Arrow table
            table = pa.Table.from_pandas(group_df)
            
            # Write Parquet file
            pq.write_table(
                table,
                file_path,
                compression=self.config.compression,
                row_group_size=self.config.row_group_size,
                use_threads=self.config.use_threads
            )
            
            stored_files.append({
                'file_path': str(file_path),
                'partition_values': partition_values,
                'rows': len(group_df),
                'size_bytes': file_path.stat().st_size
            })
            
            total_rows += len(group_df)
            
            logger.debug("Stored partition", 
                        file_path=str(file_path),
                        rows=len(group_df),
                        partition_values=partition_values)
        
        # Create storage metadata
        metadata = {
            'dataset_name': dataset_name,
            'version': version or datetime.now().strftime('%Y%m%d_%H%M%S'),
            'timestamp': datetime.now().isoformat(),
            'total_trajectories': len(trajectories),
            'total_points': total_rows,
            'stored_files': stored_files,
            'partition_schema': self.config.partition_by,
            'compression': self.config.compression,
        }
        
        # Save metadata
        if self.config.version_metadata:
            metadata_file = self.metadata_path / f"{dataset_name}_{metadata['version']}.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
        
        logger.info("Trajectories stored successfully", 
                   dataset_name=dataset_name,
                   version=metadata['version'],
                   total_trajectories=len(trajectories),
                   total_points=total_rows,
                   num_files=len(stored_files))
        
        return metadata
    
    def load_trajectories(self, 
                         dataset_name: str,
                         version: Optional[str] = None,
                         filters: Optional[Dict[str, Any]] = None) -> DataFrame:
        """Load trajectories from storage.
        
        Args:
            dataset_name: Name of the dataset
            version: Optional version identifier
            filters: Optional filters for loading specific data
            
        Returns:
            DataFrame with trajectory data
        """
        # Find data files
        data_files = self._find_data_files(dataset_name, version, filters)
        
        if not data_files:
            raise StorageError(f"No data files found for dataset: {dataset_name}")
        
        # Load and combine data
        dataframes = []
        
        for file_path in data_files:
            try:
                # Read Parquet file
                table = pq.read_table(
                    file_path,
                    memory_map=self.config.memory_map,
                    pre_buffer=self.config.pre_buffer
                )
                
                df = table.to_pandas()
                dataframes.append(df)
                
                logger.debug("Loaded file", 
                           file_path=str(file_path),
                           rows=len(df))
                
            except Exception as e:
                logger.error("Failed to load file", 
                           file_path=str(file_path),
                           error=str(e))
                continue
        
        if not dataframes:
            raise StorageError("No data could be loaded")
        
        # Combine all dataframes
        combined_df = pd.concat(dataframes, ignore_index=True)
        
        # Apply additional filters if provided
        if filters:
            combined_df = self._apply_filters(combined_df, filters)
        
        logger.info("Trajectories loaded successfully", 
                   dataset_name=dataset_name,
                   version=version,
                   total_rows=len(combined_df),
                   num_files=len(data_files))
        
        return combined_df
    
    def _find_data_files(self, 
                         dataset_name: str,
                         version: Optional[str] = None,
                         filters: Optional[Dict[str, Any]] = None) -> List[Path]:
        """Find data files matching criteria.
        
        Args:
            dataset_name: Dataset name
            version: Version identifier
            filters: Data filters
            
        Returns:
            List of matching file paths
        """
        data_files = []
        
        # Search in data directory
        for file_path in self.data_path.rglob("*.parquet"):
            # Check if file matches criteria
            if self._file_matches_criteria(file_path, dataset_name, version, filters):
                data_files.append(file_path)
        
        return sorted(data_files)
    
    def _file_matches_criteria(self, 
                              file_path: Path,
                              dataset_name: str,
                              version: Optional[str] = None,
                              filters: Optional[Dict[str, Any]] = None) -> bool:
        """Check if file matches loading criteria.
        
        Args:
            file_path: File path to check
            dataset_name: Dataset name
            version: Version identifier
            filters: Data filters
            
        Returns:
            True if file matches criteria
        """
        # Check filename for version
        if version and version not in file_path.name:
            return False
        
        # Check partition values if filters provided
        if filters:
            partition_values = self._extract_partition_from_path(file_path)
            for key, value in filters.items():
                if key in partition_values and partition_values[key] != value:
                    return False
        
        return True
    
    def _extract_partition_from_path(self, file_path: Path) -> Dict[str, Any]:
        """Extract partition values from file path.
        
        Args:
            file_path: File path
            
        Returns:
            Dictionary of partition values
        """
        partition_values = {}
        
        # Extract from directory structure
        relative_path = file_path.relative_to(self.data_path)
        path_parts = relative_path.parts[:-1]  # Exclude filename
        
        for part in path_parts:
            if '=' in part:
                key, value = part.split('=', 1)
                partition_values[key] = value
        
        return partition_values
    
    def _apply_filters(self, df: DataFrame, filters: Dict[str, Any]) -> DataFrame:
        """Apply filters to DataFrame.
        
        Args:
            df: DataFrame to filter
            filters: Filter conditions
            
        Returns:
            Filtered DataFrame
        """
        for column, value in filters.items():
            if column in df.columns:
                if isinstance(value, (list, tuple)):
                    df = df[df[column].isin(value)]
                else:
                    df = df[df[column] == value]
        
        return df
    
    def list_versions(self, dataset_name: str) -> List[Dict[str, Any]]:
        """List available versions for a dataset.
        
        Args:
            dataset_name: Dataset name
            
        Returns:
            List of version information
        """
        versions = []
        
        # Check metadata files
        metadata_pattern = f"{dataset_name}_*.json"
        for metadata_file in self.metadata_path.glob(metadata_pattern):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                versions.append(metadata)
            except Exception as e:
                logger.error("Failed to read metadata file", 
                           file_path=str(metadata_file),
                           error=str(e))
        
        return sorted(versions, key=lambda x: x['timestamp'], reverse=True)
    
    def delete_version(self, dataset_name: str, version: str) -> bool:
        """Delete a specific version of a dataset.
        
        Args:
            dataset_name: Dataset name
            version: Version to delete
            
        Returns:
            True if deletion successful
        """
        try:
            # Find and delete data files
            data_files = self._find_data_files(dataset_name, version)
            
            for file_path in data_files:
                file_path.unlink()
                logger.debug("Deleted data file", file_path=str(file_path))
            
            # Delete metadata file
            metadata_file = self.metadata_path / f"{dataset_name}_{version}.json"
            if metadata_file.exists():
                metadata_file.unlink()
                logger.debug("Deleted metadata file", file_path=str(metadata_file))
            
            logger.info("Version deleted successfully", 
                       dataset_name=dataset_name,
                       version=version,
                       deleted_files=len(data_files))
            
            return True
            
        except Exception as e:
            logger.error("Failed to delete version", 
                        dataset_name=dataset_name,
                        version=version,
                        error=str(e))
            return False
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics.
        
        Args:
            Dictionary with storage statistics
        """
        total_files = 0
        total_size = 0
        
        # Count files and size
        for file_path in self.data_path.rglob("*.parquet"):
            total_files += 1
            total_size += file_path.stat().st_size
        
        # Count metadata files
        metadata_files = len(list(self.metadata_path.glob("*.json")))
        
        return {
            'base_path': str(self.base_path),
            'total_data_files': total_files,
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'metadata_files': metadata_files,
            'partition_schema': self.config.partition_by,
            'compression': self.config.compression,
        }
    
    def optimize_storage(self) -> Dict[str, Any]:
        """Optimize storage by compacting small files.
        
        Returns:
            Optimization statistics
        """
        logger.info("Starting storage optimization")
        
        # Find small files that can be compacted
        small_files = []
        for file_path in self.data_path.rglob("*.parquet"):
            if file_path.stat().st_size < 1024 * 1024:  # Less than 1MB
                small_files.append(file_path)
        
        if not small_files:
            logger.info("No small files to optimize")
            return {'optimized_files': 0, 'saved_space': 0}
        
        # Group small files by partition
        partition_groups = {}
        for file_path in small_files:
            partition_key = str(file_path.parent)
            if partition_key not in partition_groups:
                partition_groups[partition_key] = []
            partition_groups[partition_key].append(file_path)
        
        optimized_count = 0
        saved_space = 0
        
        for partition_key, files in partition_groups.items():
            if len(files) < 2:
                continue
            
            try:
                # Read and combine small files
                dataframes = []
                original_size = 0
                
                for file_path in files:
                    table = pq.read_table(file_path)
                    df = table.to_pandas()
                    dataframes.append(df)
                    original_size += file_path.stat().st_size
                
                # Combine and write optimized file
                combined_df = pd.concat(dataframes, ignore_index=True)
                table = pa.Table.from_pandas(combined_df)
                
                # Create optimized filename
                optimized_file = files[0].parent / f"optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
                
                pq.write_table(
                    table,
                    optimized_file,
                    compression=self.config.compression,
                    row_group_size=self.config.row_group_size
                )
                
                # Delete original files
                for file_path in files:
                    file_path.unlink()
                
                optimized_size = optimized_file.stat().st_size
                saved_space += original_size - optimized_size
                optimized_count += len(files)
                
                logger.debug("Optimized partition", 
                           partition_key=partition_key,
                           original_files=len(files),
                           optimized_file=str(optimized_file))
                
            except Exception as e:
                logger.error("Failed to optimize partition", 
                           partition_key=partition_key,
                           error=str(e))
        
        logger.info("Storage optimization completed", 
                   optimized_files=optimized_count,
                   saved_space_mb=saved_space / (1024 * 1024))
        
        return {
            'optimized_files': optimized_count,
            'saved_space_bytes': saved_space,
            'saved_space_mb': saved_space / (1024 * 1024)
        }