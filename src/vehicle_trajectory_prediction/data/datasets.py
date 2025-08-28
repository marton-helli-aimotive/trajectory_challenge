"""Dataset integration and data source factory for trajectory prediction."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    try:
        import pandas as pd
        DataFrame = pd.DataFrame
        Series = pd.Series
    except ImportError:
        DataFrame = Any
        Series = Any
else:
    DataFrame = Any
    Series = Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta

# Try to import numpy
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

# Try to import pandas
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

# Use a simple base class since BaseConfig doesn't exist
class BaseConfig:
    """Base configuration class."""
    pass
from ..core.models import TrajectoryPoint, Trajectory, TrajectoryDataset
from ..core.exceptions import DataSourceError
from ..core.logging import get_logger

logger = get_logger(__name__)


class DataSource(Protocol):
    """Protocol for data sources."""
    
    def load_data(self) -> DataFrame:
        """Load data from source."""
        ...
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get source metadata."""
        ...


@dataclass
class DatasetConfig(BaseConfig):
    """Configuration for dataset loading."""
    
    # NGSIM specific settings
    ngsim_data_dir: str = "data/ngsim"
    ngsim_columns: List[str] = None  # Will be set in __post_init__
    
    # Data processing
    min_trajectory_length: int = 10
    max_trajectory_length: int = 1000
    time_resolution: float = 0.1  # seconds
    
    # Quality filters
    min_velocity: float = 0.0
    max_velocity: float = 50.0  # m/s
    min_acceleration: float = -10.0  # m/s²
    max_acceleration: float = 10.0  # m/s²
    
    # Spatial filters
    min_x: float = -float('inf')
    max_x: float = float('inf')
    min_y: float = -float('inf')
    max_y: float = float('inf')
    
    def __post_init__(self):
        """Set default NGSIM columns if not provided."""
        if self.ngsim_columns is None:
            self.ngsim_columns = [
                'Vehicle_ID', 'Frame_ID', 'Total_Frames', 'Global_Time',
                'Local_X', 'Local_Y', 'Global_X', 'Global_Y',
                'v_length', 'v_Width', 'v_Class', 'v_Vel', 'v_Acc',
                'Lane_ID', 'Preceding', 'Following', 'Space_Headway',
                'Time_Headway'
            ]


class BaseDataset(ABC):
    """Base class for trajectory datasets."""
    
    def __init__(self, config: DatasetConfig):
        """Initialize dataset with configuration.
        
        Args:
            config: Dataset configuration
        """
        self.config = config
        self.data: Optional[DataFrame] = None
        self.metadata: Dict[str, Any] = {}
        
    @abstractmethod
    def load_data(self) -> DataFrame:
        """Load raw data from source."""
        pass
    
    @abstractmethod
    def preprocess_data(self, data: DataFrame) -> DataFrame:
        """Preprocess raw data."""
        pass
    
    def validate_data(self, data: DataFrame) -> bool:
        """Validate data quality.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas is required for data validation")
            
        required_columns = ['Vehicle_ID', 'Frame_ID', 'Local_X', 'Local_Y', 'v_Vel']
        
        # Check required columns
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            logger.error("Missing required columns", missing_columns=missing_columns)
            return False
        
        # Check data types
        if not pd.api.types.is_numeric_dtype(data['Local_X']):
            logger.error("Local_X must be numeric")
            return False
        
        if not pd.api.types.is_numeric_dtype(data['Local_Y']):
            logger.error("Local_Y must be numeric")
            return False
        
        # Check for NaN values in critical columns
        critical_columns = ['Vehicle_ID', 'Local_X', 'Local_Y']
        for col in critical_columns:
            if data[col].isna().any():
                logger.warning(f"NaN values found in {col}", 
                             nan_count=data[col].isna().sum())
        
        return True
    
    def filter_data(self, data: DataFrame) -> DataFrame:
        """Apply quality filters to data.
        
        Args:
            data: DataFrame to filter
            
        Returns:
            Filtered DataFrame
        """
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas is required for data filtering")
            
        initial_count = len(data)
        
        # Velocity filter
        if 'v_Vel' in data.columns:
            data = data[
                (data['v_Vel'] >= self.config.min_velocity) &
                (data['v_Vel'] <= self.config.max_velocity)
            ]
        
        # Acceleration filter
        if 'v_Acc' in data.columns:
            data = data[
                (data['v_Acc'] >= self.config.min_acceleration) &
                (data['v_Acc'] <= self.config.max_acceleration)
            ]
        
        # Spatial filters
        data = data[
            (data['Local_X'] >= self.config.min_x) &
            (data['Local_X'] <= self.config.max_x) &
            (data['Local_Y'] >= self.config.min_y) &
            (data['Local_Y'] <= self.config.max_y)
        ]
        
        filtered_count = len(data)
        logger.info("Data filtering completed", 
                   initial_count=initial_count,
                   filtered_count=filtered_count,
                   removed_count=initial_count - filtered_count)
        
        return data
    
    def to_trajectories(self, data: DataFrame) -> TrajectoryDataset:
        """Convert DataFrame to TrajectoryDataset.
        
        Args:
            data: Preprocessed DataFrame
            
        Returns:
            TrajectoryDataset object
        """
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas is required for trajectory conversion")
            
        trajectories = []
        
        # Group by vehicle ID
        for vehicle_id, vehicle_data in data.groupby('Vehicle_ID'):
            # Sort by frame ID
            vehicle_data = vehicle_data.sort_values('Frame_ID')
            
            # Skip if trajectory too short or too long
            if len(vehicle_data) < self.config.min_trajectory_length:
                continue
            if len(vehicle_data) > self.config.max_trajectory_length:
                continue
            
            # Convert to trajectory points
            points = []
            for _, row in vehicle_data.iterrows():
                point = TrajectoryPoint(
                    x=float(row['Local_X']),
                    y=float(row['Local_Y']),
                    timestamp=float(row.get('Global_Time', row['Frame_ID'] * self.config.time_resolution)),
                    velocity=float(row.get('v_Vel', 0.0)),
                    acceleration=float(row.get('v_Acc', 0.0)),
                    heading=float(row.get('v_Heading', 0.0)),
                    vehicle_id=str(vehicle_id),
                    lane_id=str(row.get('Lane_ID', '')),
                    attributes={
                        'frame_id': int(row['Frame_ID']),
                        'vehicle_class': str(row.get('v_Class', '')),
                        'vehicle_length': float(row.get('v_length', 0.0)),
                        'vehicle_width': float(row.get('v_Width', 0.0)),
                    }
                )
                points.append(point)
            
            # Create trajectory
            trajectory = Trajectory(
                points=points,
                vehicle_id=str(vehicle_id),
                metadata={
                    'source': self.__class__.__name__,
                    'num_points': len(points),
                    'duration': points[-1].timestamp - points[0].timestamp,
                    'total_distance': self._calculate_total_distance(points),
                }
            )
            trajectories.append(trajectory)
        
        # Create dataset
        dataset = TrajectoryDataset(
            trajectories=trajectories,
            name=f"{self.__class__.__name__}_Dataset",
            description=f"Dataset from {self.__class__.__name__}",
            source=self.__class__.__name__,
            version="1.0.0",
            metadata=self.metadata
        )
        
        logger.info("Trajectory conversion completed", 
                   num_trajectories=len(trajectories),
                   total_points=sum(len(t.points) for t in trajectories))
        
        return dataset
    
    def _calculate_total_distance(self, points: List[TrajectoryPoint]) -> float:
        """Calculate total distance of trajectory.
        
        Args:
            points: List of trajectory points
            
        Returns:
            Total distance in meters
        """
        total_distance = 0.0
        for i in range(1, len(points)):
            dx = points[i].x - points[i-1].x
            dy = points[i].y - points[i-1].y
            total_distance += np.sqrt(dx*dx + dy*dy)
        return total_distance
    
    def load(self) -> TrajectoryDataset:
        """Load and process dataset.
        
        Returns:
            Processed TrajectoryDataset
            
        Raises:
            DataSourceError: If loading fails
        """
        try:
            logger.info(f"Loading {self.__class__.__name__} dataset")
            
            # Load raw data
            raw_data = self.load_data()
            
            # Validate data
            if not self.validate_data(raw_data):
                raise DataSourceError("Data validation failed")
            
            # Preprocess data
            processed_data = self.preprocess_data(raw_data)
            
            # Filter data
            filtered_data = self.filter_data(processed_data)
            
            # Convert to trajectories
            dataset = self.to_trajectories(filtered_data)
            
            logger.info(f"Successfully loaded {self.__class__.__name__} dataset",
                       num_trajectories=len(dataset.trajectories))
            
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to load {self.__class__.__name__} dataset", 
                        error=str(e))
            raise DataSourceError(f"Dataset loading failed: {str(e)}") from e


class NGSIMDataset(BaseDataset):
    """NGSIM (Next Generation Simulation) dataset loader."""
    
    def __init__(self, config: DatasetConfig, data_path: Optional[Path] = None):
        """Initialize NGSIM dataset.
        
        Args:
            config: Dataset configuration
            data_path: Path to NGSIM data files
        """
        super().__init__(config)
        self.data_path = Path(data_path) if data_path else Path(config.ngsim_data_dir)
        
        # NGSIM specific metadata
        self.metadata = {
            'dataset_type': 'NGSIM',
            'data_path': str(self.data_path),
            'description': 'Next Generation Simulation dataset',
            'time_resolution': 0.1,  # seconds
            'spatial_resolution': 0.1,  # meters
        }
    
    def load_data(self) -> DataFrame:
        """Load NGSIM data from files.
        
        Returns:
            Combined DataFrame from all NGSIM files
            
        Raises:
            DataSourceError: If no data files found
        """
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas is required for NGSIM data loading")
            
        if not self.data_path.exists():
            raise DataSourceError(f"NGSIM data directory does not exist: {self.data_path}")
        
        # Find all CSV files
        csv_files = list(self.data_path.glob("*.csv"))
        if not csv_files:
            raise DataSourceError(f"No CSV files found in {self.data_path}")
        
        logger.info(f"Found {len(csv_files)} NGSIM data files")
        
        # Load and combine all files
        dataframes = []
        for file_path in csv_files:
            try:
                df = pd.read_csv(file_path)
                
                # Ensure required columns exist
                missing_columns = set(self.config.ngsim_columns) - set(df.columns)
                if missing_columns:
                    logger.warning(f"Missing columns in {file_path.name}", 
                                 missing_columns=missing_columns)
                    # Add missing columns with default values
                    for col in missing_columns:
                        df[col] = 0.0
                
                # Select only required columns
                df = df[self.config.ngsim_columns]
                
                # Add source file information
                df['source_file'] = file_path.name
                
                dataframes.append(df)
                logger.debug(f"Loaded {file_path.name}", rows=len(df))
                
            except Exception as e:
                logger.error(f"Failed to load {file_path}", error=str(e))
                continue
        
        if not dataframes:
            raise DataSourceError("No data files could be loaded")
        
        # Combine all dataframes
        combined_df = pd.concat(dataframes, ignore_index=True)
        
        logger.info("NGSIM data loading completed", 
                   total_files=len(csv_files),
                   total_rows=len(combined_df))
        
        return combined_df
    
    def preprocess_data(self, data: DataFrame) -> DataFrame:
        """Preprocess NGSIM data.
        
        Args:
            data: Raw NGSIM DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas is required for NGSIM data preprocessing")
            
        # Convert data types
        data['Vehicle_ID'] = data['Vehicle_ID'].astype(str)
        data['Frame_ID'] = data['Frame_ID'].astype(int)
        data['Local_X'] = data['Local_X'].astype(float)
        data['Local_Y'] = data['Local_Y'].astype(float)
        data['v_Vel'] = data['v_Vel'].astype(float)
        
        # Handle missing values
        data['v_Acc'] = data['v_Acc'].fillna(0.0)
        data['v_Heading'] = data['v_Heading'].fillna(0.0)
        data['Lane_ID'] = data['Lane_ID'].fillna('unknown')
        
        # Calculate heading if not available
        if 'v_Heading' not in data.columns or data['v_Heading'].isna().all():
            data['v_Heading'] = self._calculate_heading(data)
        
        # Remove duplicates
        data = data.drop_duplicates(subset=['Vehicle_ID', 'Frame_ID'])
        
        # Sort by vehicle ID and frame ID
        data = data.sort_values(['Vehicle_ID', 'Frame_ID'])
        
        logger.info("NGSIM data preprocessing completed", 
                   final_rows=len(data),
                   unique_vehicles=data['Vehicle_ID'].nunique())
        
        return data
    
    def _calculate_heading(self, data: DataFrame) -> Series:
        """Calculate vehicle heading from position changes.
        
        Args:
            data: DataFrame with position data
            
        Returns:
            Series with heading angles
        """
        if not PANDAS_AVAILABLE or not NUMPY_AVAILABLE:
            raise ImportError("pandas and numpy are required for heading calculation")
            
        # Group by vehicle and calculate heading
        headings = []
        
        for vehicle_id, vehicle_data in data.groupby('Vehicle_ID'):
            vehicle_data = vehicle_data.sort_values('Frame_ID')
            
            # Calculate heading from position changes
            dx = vehicle_data['Local_X'].diff()
            dy = vehicle_data['Local_Y'].diff()
            
            # Calculate angle
            angle = np.arctan2(dy, dx)
            
            # Convert to degrees
            heading_deg = np.degrees(angle)
            
            # Handle first row (no previous position)
            heading_deg.iloc[0] = heading_deg.iloc[1] if len(heading_deg) > 1 else 0.0
            
            headings.append(pd.Series(heading_deg, index=vehicle_data.index))
        
        return pd.concat(headings).reindex(data.index)


class DatasetFactory:
    """Factory for creating dataset instances."""
    
    _datasets = {
        'ngsim': NGSIMDataset,
    }
    
    @classmethod
    def register_dataset(cls, name: str, dataset_class: type):
        """Register a new dataset class.
        
        Args:
            name: Dataset name
            dataset_class: Dataset class to register
        """
        cls._datasets[name] = dataset_class
        logger.info(f"Registered dataset: {name}")
    
    @classmethod
    def create_dataset(cls, name: str, config: DatasetConfig, **kwargs) -> BaseDataset:
        """Create a dataset instance.
        
        Args:
            name: Dataset name
            config: Dataset configuration
            **kwargs: Additional arguments for dataset constructor
            
        Returns:
            Dataset instance
            
        Raises:
            DataSourceError: If dataset type not supported
        """
        if name not in cls._datasets:
            available = list(cls._datasets.keys())
            raise DataSourceError(f"Unknown dataset type: {name}. Available: {available}")
        
        dataset_class = cls._datasets[name]
        return dataset_class(config, **kwargs)
    
    @classmethod
    def list_datasets(cls) -> List[str]:
        """List available dataset types.
        
        Returns:
            List of available dataset names
        """
        return list(cls._datasets.keys())
    
    @classmethod
    def get_dataset_info(cls, name: str) -> Dict[str, Any]:
        """Get information about a dataset type.
        
        Args:
            name: Dataset name
            
        Returns:
            Dataset information dictionary
        """
        if name not in cls._datasets:
            raise DataSourceError(f"Unknown dataset type: {name}")
        
        dataset_class = cls._datasets[name]
        return {
            'name': name,
            'class': dataset_class.__name__,
            'description': dataset_class.__doc__ or '',
            'config_fields': list(DatasetConfig.__annotations__.keys()),
        }