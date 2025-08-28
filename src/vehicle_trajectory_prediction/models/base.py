"""Base classes for trajectory prediction models."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

# Try to import optional dependencies
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # Create a simple numpy-like interface for basic operations
    class SimpleArray:
        def __init__(self, data):
            self.data = data
            if isinstance(data, list):
                self.shape = (len(data),)
            else:
                self.shape = (1,)
        
        def __getitem__(self, key):
            return self.data[key]
        
        def __len__(self):
            return len(self.data)
        
        def __iter__(self):
            return iter(self.data)
    
    def array(data):
        return SimpleArray(data)
    
    def sqrt(x):
        import math
        return math.sqrt(x)
    
    def mean(data):
        return sum(data) / len(data)
    
    np = type('numpy', (), {
        'array': array,
        'sqrt': sqrt,
        'mean': mean,
        'inf': float('inf'),
        'nan': float('nan'),
        'zeros_like': lambda x: SimpleArray([0.0] * len(x)),
        'where': lambda condition, x, y: x if condition else y,
        'abs': abs,
        'linalg': type('linalg', (), {'norm': lambda x: sqrt(sum(xi**2 for xi in x))})(),
        'arctan2': lambda y, x: __import__('math').atan2(y, x),
        'cos': lambda x: __import__('math').cos(x),
        'sin': lambda x: __import__('math').sin(x),
        'ndarray': SimpleArray
    })()

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

from ..core.models import Trajectory, TrajectoryPoint
from ..core.config import ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Result of a trajectory prediction."""
    
    # Predicted trajectory points
    predicted_points: List[TrajectoryPoint]
    
    # Prediction timestamps
    timestamps: List[datetime]
    
    # Predicted positions (x, y)
    x_positions: Any
    y_positions: Any
    
    # Predicted velocities
    velocities: Any
    
    # Predicted accelerations
    accelerations: Any
    
    # Predicted headings
    headings: Any
    
    # Confidence scores (if available)
    confidence_scores: Optional[Any] = None
    
    # Uncertainty measures (if available)
    uncertainty: Optional[Dict[str, Any]] = None
    
    # Model metadata
    model_name: str = ""
    prediction_time: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_trajectory(self, vehicle_id: str) -> Trajectory:
        """Convert prediction result to Trajectory object."""
        if len(self.predicted_points) < 2:
            raise ValueError("Prediction must have at least 2 points")
        
        start_time = self.timestamps[0]
        end_time = self.timestamps[-1]
        duration = (end_time - start_time).total_seconds()
        
        # Calculate total distance
        total_distance = 0.0
        for i in range(1, len(self.predicted_points)):
            total_distance += self.predicted_points[i-1].distance_to(self.predicted_points[i])
        
        return Trajectory(
            vehicle_id=vehicle_id,
            points=self.predicted_points,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            total_distance=total_distance,
            metadata=self.metadata or {}
        )
    
    def to_dataframe(self):
        """Convert prediction result to pandas DataFrame."""
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas is required for DataFrame conversion")
        
        data = []
        for i, point in enumerate(self.predicted_points):
            row = {
                "timestamp": self.timestamps[i],
                "x": point.x,
                "y": point.y,
                "velocity": point.velocity,
                "acceleration": point.acceleration,
                "heading": point.heading,
            }
            
            if self.confidence_scores is not None:
                row["confidence"] = self.confidence_scores[i]
            
            if self.uncertainty is not None:
                for key, values in self.uncertainty.items():
                    row[f"uncertainty_{key}"] = values[i]
            
            data.append(row)
        
        return pd.DataFrame(data)


class BaseTrajectoryPredictor(ABC):
    """Base class for trajectory prediction models."""
    
    def __init__(self, config: ModelConfig, model_name: str = ""):
        self.config = config
        self.model_name = model_name or self.__class__.__name__
        self.is_trained = False
        self.training_data_size = 0
        self.last_training_time = None
        
        logger.info(f"Initialized {self.model_name} predictor")
    
    @abstractmethod
    def train(self, trajectories: List[Trajectory]) -> Dict[str, Any]:
        """Train the model on trajectory data."""
        pass
    
    @abstractmethod
    def predict(
        self, 
        trajectory: Trajectory, 
        prediction_horizon: Optional[int] = None,
        prediction_frequency: Optional[float] = None
    ) -> PredictionResult:
        """Predict future trajectory."""
        pass
    
    @abstractmethod
    def predict_batch(
        self, 
        trajectories: List[Trajectory],
        prediction_horizon: Optional[int] = None,
        prediction_frequency: Optional[float] = None
    ) -> List[PredictionResult]:
        """Predict future trajectories for multiple inputs."""
        pass
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model to disk."""
        raise NotImplementedError(f"Model saving not implemented for {self.model_name}")
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model from disk."""
        raise NotImplementedError(f"Model loading not implemented for {self.model_name}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        return {
            "model_name": self.model_name,
            "is_trained": self.is_trained,
            "training_data_size": self.training_data_size,
            "last_training_time": self.last_training_time,
            "config": self.config.__dict__ if hasattr(self.config, '__dict__') else str(self.config)
        }
    
    def validate_input(self, trajectory: Trajectory) -> None:
        """Validate input trajectory for prediction."""
        if not self.is_trained:
            raise RuntimeError(f"Model {self.model_name} must be trained before prediction")
        
        if trajectory.length < 2:
            raise ValueError("Trajectory must have at least 2 points")
        
        # Check for minimum trajectory length
        min_length = getattr(self.config, 'min_trajectory_length', 5)
        if trajectory.length < min_length:
            raise ValueError(f"Trajectory must have at least {min_length} points")
    
    def _prepare_prediction_timestamps(
        self, 
        trajectory: Trajectory,
        prediction_horizon: Optional[int] = None,
        prediction_frequency: Optional[float] = None
    ) -> List[datetime]:
        """Prepare timestamps for prediction."""
        horizon = prediction_horizon or self.config.prediction_horizon
        frequency = prediction_frequency or self.config.prediction_frequency
        
        # Start from the last timestamp in the trajectory
        start_time = trajectory.points[-1].timestamp
        
        # Generate prediction timestamps
        timestamps = []
        current_time = start_time
        
        for i in range(horizon):
            current_time += timedelta(seconds=frequency)
            timestamps.append(current_time)
        
        return timestamps
    
    def _interpolate_trajectory_point(
        self, 
        trajectory: Trajectory, 
        timestamp: datetime
    ) -> TrajectoryPoint:
        """Interpolate trajectory point at specific timestamp."""
        # Find the two points that bracket the timestamp
        points = trajectory.points
        
        # If timestamp is before first point, use first point
        if timestamp <= points[0].timestamp:
            return points[0]
        
        # If timestamp is after last point, use last point
        if timestamp >= points[-1].timestamp:
            return points[-1]
        
        # Find the two points that bracket the timestamp
        for i in range(len(points) - 1):
            if points[i].timestamp <= timestamp <= points[i + 1].timestamp:
                p1, p2 = points[i], points[i + 1]
                t1, t2 = p1.timestamp, p2.timestamp
                
                # Linear interpolation
                alpha = (timestamp - t1).total_seconds() / (t2 - t1).total_seconds()
                
                x = p1.x + alpha * (p2.x - p1.x)
                y = p1.y + alpha * (p2.y - p1.y)
                velocity = p1.velocity + alpha * (p2.velocity - p1.velocity)
                acceleration = p1.acceleration + alpha * (p2.acceleration - p1.acceleration)
                heading = p1.heading + alpha * (p2.heading - p1.heading)
                
                return TrajectoryPoint(
                    x=x, y=y, timestamp=timestamp, velocity=velocity,
                    acceleration=acceleration, heading=heading,
                    vehicle_id=p1.vehicle_id, lane_id=p1.lane_id
                )
        
        # Fallback to last point
        return points[-1]