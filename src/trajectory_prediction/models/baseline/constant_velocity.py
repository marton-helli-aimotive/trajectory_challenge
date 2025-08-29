"""
Constant Velocity (CV) trajectory prediction model.

Simple baseline that assumes vehicles maintain constant velocity.
"""

import asyncio
from typing import Any, Dict, List, Optional

import numpy as np
from omegaconf import DictConfig

from ...data.validation.schemas import TrajectoryData
from ..base import TrajectoryPredictor, PredictionResult


class ConstantVelocityPredictor(TrajectoryPredictor):
    """
    Constant Velocity trajectory prediction model.
    
    Predicts future trajectory by assuming the vehicle maintains
    its current velocity vector (both magnitude and direction).
    
    This is a simple but effective baseline for short-term prediction.
    """
    
    def __init__(self, config: DictConfig, name: Optional[str] = None):
        super().__init__(config, name or "ConstantVelocity")
        
        # Model parameters
        self.min_history_length = config.get("min_history_length", 2)
        self.velocity_smoothing_window = config.get("velocity_smoothing_window", 3)
        
        # Statistics for uncertainty estimation
        self._velocity_noise_std = 0.1  # Will be learned during training
        self._position_noise_std = 0.5
    
    async def fit(
        self, 
        trajectories: List[TrajectoryData],
        validation_trajectories: Optional[List[TrajectoryData]] = None
    ) -> Dict[str, Any]:
        """
        Train the constant velocity model by learning noise parameters.
        
        Args:
            trajectories: Training trajectory data
            validation_trajectories: Optional validation data
            
        Returns:
            Training statistics
        """
        # Group trajectories by vehicle
        vehicle_trajectories = self._group_by_vehicle(trajectories)
        
        velocity_errors = []
        position_errors = []
        
        for vehicle_id, traj_points in vehicle_trajectories.items():
            if len(traj_points) < self.min_history_length + 1:
                continue
            
            # Sort by timestamp
            traj_points = sorted(traj_points, key=lambda x: x.timestamp)
            
            # Calculate velocity errors for noise estimation
            for i in range(self.min_history_length, len(traj_points) - 1):
                # Use history to predict next point
                history = traj_points[i-self.min_history_length:i]
                actual_next = traj_points[i]
                
                # Predict using current velocity
                pred_result = await self._predict_from_history(
                    history, 
                    (actual_next.timestamp - history[-1].timestamp).total_seconds()
                )
                
                if len(pred_result.x_coords) > 0:
                    # Calculate errors
                    pos_error = np.sqrt(
                        (pred_result.x_coords[0] - actual_next.x)**2 + 
                        (pred_result.y_coords[0] - actual_next.y)**2
                    )
                    position_errors.append(pos_error)
        
        # Estimate noise parameters
        if position_errors:
            self._position_noise_std = np.std(position_errors)
            self._velocity_noise_std = self._position_noise_std / 2.0  # Rough estimate
        
        self.is_trained = True
        
        return {
            "model": self.name,
            "training_samples": len(trajectories),
            "vehicles_analyzed": len(vehicle_trajectories),
            "position_noise_std": float(self._position_noise_std),
            "velocity_noise_std": float(self._velocity_noise_std),
            "mean_position_error": float(np.mean(position_errors)) if position_errors else 0.0
        }
    
    async def predict(
        self,
        history: List[TrajectoryData],
        prediction_horizon: float,
        **kwargs
    ) -> PredictionResult:
        """
        Predict future trajectory using constant velocity assumption.
        
        Args:
            history: Historical trajectory points
            prediction_horizon: How far to predict (seconds)
            **kwargs: Additional parameters (time_step_size, etc.)
            
        Returns:
            Predicted trajectory with uncertainty estimates
        """
        self._validate_history(history)
        
        if len(history) < self.min_history_length:
            raise ValueError(f"Need at least {self.min_history_length} history points")
        
        return await self._predict_from_history(history, prediction_horizon, **kwargs)
    
    async def _predict_from_history(
        self,
        history: List[TrajectoryData],
        prediction_horizon: float,
        time_step_size: float = 0.1,
        **kwargs
    ) -> PredictionResult:
        """Internal prediction method."""
        
        # Calculate current velocity from recent history
        velocity_x, velocity_y = self._estimate_velocity(history)
        
        # Generate future timestamps
        num_steps = int(prediction_horizon / time_step_size)
        if num_steps == 0:
            num_steps = 1
        
        time_steps = np.linspace(time_step_size, prediction_horizon, num_steps)
        
        # Current position (last known point)
        last_point = history[-1]
        current_x, current_y = last_point.x, last_point.y
        
        # Predict future positions
        future_x = current_x + velocity_x * time_steps
        future_y = current_y + velocity_y * time_steps
        
        # Generate timestamps
        base_timestamp = last_point.timestamp
        future_timestamps = np.array([
            (base_timestamp.timestamp() + t) * 1000 for t in time_steps
        ])
        
        # Estimate uncertainties (grows with time)
        uncertainty_growth_factor = np.sqrt(time_steps)  # Uncertainty grows with sqrt(time)
        x_std = self._position_noise_std * uncertainty_growth_factor
        y_std = self._position_noise_std * uncertainty_growth_factor
        
        # Estimate velocities (constant)
        velocities = np.full(num_steps, np.sqrt(velocity_x**2 + velocity_y**2))
        
        return PredictionResult(
            timestamps=future_timestamps,
            x_coords=future_x,
            y_coords=future_y,
            x_std=x_std,
            y_std=y_std,
            velocities=velocities,
            accelerations=np.zeros(num_steps),  # Zero acceleration
            prediction_horizon=prediction_horizon,
            model_name=self.name,
            metadata={
                "velocity_x": float(velocity_x),
                "velocity_y": float(velocity_y),
                "position_noise_std": float(self._position_noise_std),
                "time_step_size": time_step_size
            }
        )
    
    def _estimate_velocity(self, history: List[TrajectoryData]) -> tuple:
        """
        Estimate current velocity from trajectory history.
        
        Args:
            history: Trajectory points
            
        Returns:
            Tuple of (velocity_x, velocity_y)
        """
        if len(history) < 2:
            return 0.0, 0.0
        
        # Use smoothing window for more stable velocity estimate
        window_size = min(self.velocity_smoothing_window, len(history))
        recent_points = history[-window_size:]
        
        # Calculate velocities for each pair of points
        velocities_x = []
        velocities_y = []
        
        for i in range(1, len(recent_points)):
            dt = (recent_points[i].timestamp - recent_points[i-1].timestamp).total_seconds()
            
            if dt > 0:
                vx = (recent_points[i].x - recent_points[i-1].x) / dt
                vy = (recent_points[i].y - recent_points[i-1].y) / dt
                velocities_x.append(vx)
                velocities_y.append(vy)
        
        # Return average velocity
        if velocities_x:
            return np.mean(velocities_x), np.mean(velocities_y)
        else:
            return 0.0, 0.0
    
    def _group_by_vehicle(self, trajectories: List[TrajectoryData]) -> Dict[int, List[TrajectoryData]]:
        """Group trajectory points by vehicle ID."""
        vehicle_groups = {}
        
        for point in trajectories:
            vehicle_id = point.vehicle_id
            if vehicle_id not in vehicle_groups:
                vehicle_groups[vehicle_id] = []
            vehicle_groups[vehicle_id].append(point)
        
        return vehicle_groups
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and parameters."""
        return {
            "name": self.name,
            "type": "baseline",
            "description": "Constant velocity trajectory prediction",
            "parameters": {
                "min_history_length": self.min_history_length,
                "velocity_smoothing_window": self.velocity_smoothing_window,
                "position_noise_std": float(self._position_noise_std),
                "velocity_noise_std": float(self._velocity_noise_std)
            },
            "capabilities": {
                "uncertainty_estimation": True,
                "online_learning": False,
                "batch_prediction": True
            },
            "is_trained": self.is_trained
        }