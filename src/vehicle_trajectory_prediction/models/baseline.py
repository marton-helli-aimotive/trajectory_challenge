"""Baseline trajectory prediction models."""

from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
import math

# Import numpy from base module to ensure consistency
from .base import np, NUMPY_AVAILABLE

from .base import BaseTrajectoryPredictor, PredictionResult
from ..core.models import Trajectory, TrajectoryPoint
from ..core.config import ModelConfig

logger = logging.getLogger(__name__)


class ConstantVelocityPredictor(BaseTrajectoryPredictor):
    """Constant Velocity (CV) trajectory predictor.
    
    This baseline model assumes the vehicle maintains constant velocity
    in both x and y directions. It's a simple physics-based model that
    serves as a baseline for comparison with more sophisticated models.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__(config, "ConstantVelocity")
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        self.last_position = None
        self.last_timestamp = None
        
    def train(self, trajectories: List[Trajectory]) -> Dict[str, Any]:
        """Train the CV model (minimal training required)."""
        logger.info(f"Training {self.model_name} model with {len(trajectories)} trajectories")
        
        # For CV model, we don't need extensive training
        # We just validate that we have enough data
        if len(trajectories) < 1:
            raise ValueError("At least one trajectory is required for training")
        
        total_points = sum(traj.length for traj in trajectories)
        logger.info(f"Total trajectory points: {total_points}")
        
        self.is_trained = True
        self.training_data_size = len(trajectories)
        self.last_training_time = datetime.now()
        
        return {
            "model_name": self.model_name,
            "training_data_size": self.training_data_size,
            "total_points": total_points,
            "training_time": self.last_training_time
        }
    
    def predict(
        self, 
        trajectory: Trajectory, 
        prediction_horizon: Optional[int] = None,
        prediction_frequency: Optional[float] = None
    ) -> PredictionResult:
        """Predict future trajectory using constant velocity assumption."""
        self.validate_input(trajectory)
        
        # Get prediction parameters
        horizon = prediction_horizon or self.config.prediction_horizon
        frequency = prediction_frequency or self.config.prediction_frequency
        
        # Calculate current velocity from last two points
        if trajectory.length < 2:
            raise ValueError("Trajectory must have at least 2 points for velocity calculation")
        
        last_point = trajectory.points[-1]
        second_last_point = trajectory.points[-2]
        
        # Calculate time difference
        dt = (last_point.timestamp - second_last_point.timestamp).total_seconds()
        if dt <= 0:
            raise ValueError("Invalid time difference between trajectory points")
        
        # Calculate velocity components
        velocity_x = (last_point.x - second_last_point.x) / dt
        velocity_y = (last_point.y - second_last_point.y) / dt
        
        # Calculate velocity magnitude and heading
        velocity_magnitude = np.sqrt(velocity_x**2 + velocity_y**2)
        heading = np.arctan2(velocity_y, velocity_x)
        
        # Generate prediction timestamps
        timestamps = self._prepare_prediction_timestamps(trajectory, horizon, frequency)
        
        # Predict future positions
        predicted_points = []
        x_positions = []
        y_positions = []
        velocities = []
        accelerations = []
        headings_list = []
        
        current_x = last_point.x
        current_y = last_point.y
        
        for i, timestamp in enumerate(timestamps):
            # Calculate time step
            dt_pred = (timestamp - last_point.timestamp).total_seconds()
            
            # Predict position using constant velocity
            predicted_x = current_x + velocity_x * dt_pred
            predicted_y = current_y + velocity_y * dt_pred
            
            # Create predicted point
            predicted_point = TrajectoryPoint(
                x=predicted_x,
                y=predicted_y,
                timestamp=timestamp,
                velocity=velocity_magnitude,
                acceleration=0.0,  # Constant velocity means zero acceleration
                heading=heading,
                vehicle_id=last_point.vehicle_id,
                lane_id=last_point.lane_id
            )
            
            predicted_points.append(predicted_point)
            x_positions.append(predicted_x)
            y_positions.append(predicted_y)
            velocities.append(velocity_magnitude)
            accelerations.append(0.0)
            headings_list.append(heading)
            
            # Update current position for next iteration
            current_x = predicted_x
            current_y = predicted_y
        
        return PredictionResult(
            predicted_points=predicted_points,
            timestamps=timestamps,
            x_positions=np.array(x_positions),
            y_positions=np.array(y_positions),
            velocities=np.array(velocities),
            accelerations=np.array(accelerations),
            headings=np.array(headings_list),
            model_name=self.model_name,
            prediction_time=datetime.now(),
            metadata={
                "velocity_x": velocity_x,
                "velocity_y": velocity_y,
                "velocity_magnitude": velocity_magnitude,
                "heading": heading
            }
        )
    
    def predict_batch(
        self, 
        trajectories: List[Trajectory],
        prediction_horizon: Optional[int] = None,
        prediction_frequency: Optional[float] = None
    ) -> List[PredictionResult]:
        """Predict future trajectories for multiple inputs."""
        return [self.predict(traj, prediction_horizon, prediction_frequency) 
                for traj in trajectories]


class ConstantAccelerationPredictor(BaseTrajectoryPredictor):
    """Constant Acceleration (CA) trajectory predictor.
    
    This baseline model assumes the vehicle maintains constant acceleration
    in both x and y directions. It's a physics-based model that extends
    the CV model by considering acceleration.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__(config, "ConstantAcceleration")
        self.acceleration_x = 0.0
        self.acceleration_y = 0.0
        
    def train(self, trajectories: List[Trajectory]) -> Dict[str, Any]:
        """Train the CA model (minimal training required)."""
        logger.info(f"Training {self.model_name} model with {len(trajectories)} trajectories")
        
        # For CA model, we don't need extensive training
        # We just validate that we have enough data
        if len(trajectories) < 1:
            raise ValueError("At least one trajectory is required for training")
        
        total_points = sum(traj.length for traj in trajectories)
        logger.info(f"Total trajectory points: {total_points}")
        
        self.is_trained = True
        self.training_data_size = len(trajectories)
        self.last_training_time = datetime.now()
        
        return {
            "model_name": self.model_name,
            "training_data_size": self.training_data_size,
            "total_points": total_points,
            "training_time": self.last_training_time
        }
    
    def predict(
        self, 
        trajectory: Trajectory, 
        prediction_horizon: Optional[int] = None,
        prediction_frequency: Optional[float] = None
    ) -> PredictionResult:
        """Predict future trajectory using constant acceleration assumption."""
        self.validate_input(trajectory)
        
        # Get prediction parameters
        horizon = prediction_horizon or self.config.prediction_horizon
        frequency = prediction_frequency or self.config.prediction_frequency
        
        # Calculate current velocity and acceleration from last three points
        if trajectory.length < 3:
            raise ValueError("Trajectory must have at least 3 points for acceleration calculation")
        
        last_point = trajectory.points[-1]
        second_last_point = trajectory.points[-2]
        third_last_point = trajectory.points[-3]
        
        # Calculate time differences
        dt1 = (last_point.timestamp - second_last_point.timestamp).total_seconds()
        dt2 = (second_last_point.timestamp - third_last_point.timestamp).total_seconds()
        
        if dt1 <= 0 or dt2 <= 0:
            raise ValueError("Invalid time differences between trajectory points")
        
        # Calculate velocity components at current and previous time
        velocity_x_current = (last_point.x - second_last_point.x) / dt1
        velocity_y_current = (last_point.y - second_last_point.y) / dt1
        
        velocity_x_prev = (second_last_point.x - third_last_point.x) / dt2
        velocity_y_prev = (second_last_point.y - third_last_point.y) / dt2
        
        # Calculate acceleration components
        acceleration_x = (velocity_x_current - velocity_x_prev) / ((dt1 + dt2) / 2)
        acceleration_y = (velocity_y_current - velocity_y_prev) / ((dt1 + dt2) / 2)
        
        # Calculate current velocity magnitude and heading
        velocity_magnitude = np.sqrt(velocity_x_current**2 + velocity_y_current**2)
        heading = np.arctan2(velocity_y_current, velocity_x_current)
        
        # Generate prediction timestamps
        timestamps = self._prepare_prediction_timestamps(trajectory, horizon, frequency)
        
        # Predict future positions
        predicted_points = []
        x_positions = []
        y_positions = []
        velocities = []
        accelerations = []
        headings_list = []
        
        current_x = last_point.x
        current_y = last_point.y
        current_vx = velocity_x_current
        current_vy = velocity_y_current
        
        for i, timestamp in enumerate(timestamps):
            # Calculate time step
            dt_pred = (timestamp - last_point.timestamp).total_seconds()
            
            # Predict position using constant acceleration
            # x = x0 + v0*t + 0.5*a*t^2
            predicted_x = current_x + current_vx * dt_pred + 0.5 * acceleration_x * dt_pred**2
            predicted_y = current_y + current_vy * dt_pred + 0.5 * acceleration_y * dt_pred**2
            
            # Predict velocity using constant acceleration
            # v = v0 + a*t
            predicted_vx = current_vx + acceleration_x * dt_pred
            predicted_vy = current_vy + acceleration_y * dt_pred
            predicted_velocity = np.sqrt(predicted_vx**2 + predicted_vy**2)
            predicted_heading = np.arctan2(predicted_vy, predicted_vx)
            
            # Create predicted point
            predicted_point = TrajectoryPoint(
                x=predicted_x,
                y=predicted_y,
                timestamp=timestamp,
                velocity=predicted_velocity,
                acceleration=np.sqrt(acceleration_x**2 + acceleration_y**2),
                heading=predicted_heading,
                vehicle_id=last_point.vehicle_id,
                lane_id=last_point.lane_id
            )
            
            predicted_points.append(predicted_point)
            x_positions.append(predicted_x)
            y_positions.append(predicted_y)
            velocities.append(predicted_velocity)
            accelerations.append(np.sqrt(acceleration_x**2 + acceleration_y**2))
            headings_list.append(predicted_heading)
            
            # Update current state for next iteration
            current_x = predicted_x
            current_y = predicted_y
            current_vx = predicted_vx
            current_vy = predicted_vy
        
        return PredictionResult(
            predicted_points=predicted_points,
            timestamps=timestamps,
            x_positions=np.array(x_positions),
            y_positions=np.array(y_positions),
            velocities=np.array(velocities),
            accelerations=np.array(accelerations),
            headings=np.array(headings_list),
            model_name=self.model_name,
            prediction_time=datetime.now(),
            metadata={
                "initial_velocity_x": velocity_x_current,
                "initial_velocity_y": velocity_y_current,
                "acceleration_x": acceleration_x,
                "acceleration_y": acceleration_y,
                "initial_heading": heading
            }
        )
    
    def predict_batch(
        self, 
        trajectories: List[Trajectory],
        prediction_horizon: Optional[int] = None,
        prediction_frequency: Optional[float] = None
    ) -> List[PredictionResult]:
        """Predict future trajectories for multiple inputs."""
        return [self.predict(traj, prediction_horizon, prediction_frequency) 
                for traj in trajectories]