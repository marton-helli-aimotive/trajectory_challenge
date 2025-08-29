"""
Constant Acceleration (CA) trajectory prediction model.

Baseline model that assumes vehicles maintain constant acceleration,
using kinematic equations for prediction.
"""

import asyncio
from typing import Any, Dict, List, Optional

import numpy as np
from omegaconf import DictConfig

from ...data.validation.schemas import TrajectoryData
from ..base import TrajectoryPredictor, PredictionResult


class ConstantAccelerationPredictor(TrajectoryPredictor):
    """
    Constant Acceleration trajectory prediction model.
    
    Uses kinematic equations: x(t) = x0 + v0*t + 0.5*a*t²
    
    Predicts future trajectory by assuming the vehicle maintains
    its current acceleration vector in both X and Y directions.
    """
    
    def __init__(self, config: DictConfig, name: Optional[str] = None):
        super().__init__(config, name or "ConstantAcceleration")
        
        # Model parameters
        self.min_history_length = config.get("min_history_length", 3)
        self.acceleration_smoothing_window = config.get("acceleration_smoothing_window", 3)
        
        # Noise parameters for uncertainty estimation
        self._acceleration_noise_std = 0.5  # m/s² or ft/s²
        self._velocity_noise_std = 0.2      # m/s or ft/s
        self._position_noise_std = 1.0      # m or ft
    
    async def fit(
        self, 
        trajectories: List[TrajectoryData],
        validation_trajectories: Optional[List[TrajectoryData]] = None
    ) -> Dict[str, Any]:
        """
        Train the constant acceleration model by learning noise parameters.
        
        Args:
            trajectories: Training trajectory data
            validation_trajectories: Optional validation data
            
        Returns:
            Training statistics and learned parameters
        """
        vehicle_trajectories = self._group_by_vehicle(trajectories)
        
        position_errors = []
        velocity_errors = []
        acceleration_errors = []
        
        for vehicle_id, traj_points in vehicle_trajectories.items():
            if len(traj_points) < self.min_history_length + 1:
                continue
            
            # Sort by timestamp
            traj_points = sorted(traj_points, key=lambda x: x.timestamp)
            
            # Analyze prediction accuracy for noise parameter estimation
            for i in range(self.min_history_length, len(traj_points) - 1):
                history = traj_points[i-self.min_history_length:i]
                actual_next = traj_points[i]
                
                try:
                    # Predict using current method
                    pred_horizon = (actual_next.timestamp - history[-1].timestamp).total_seconds()
                    pred_result = await self._predict_from_history(history, pred_horizon)
                    
                    if len(pred_result.x_coords) > 0:
                        # Calculate position error
                        pos_error = np.sqrt(
                            (pred_result.x_coords[0] - actual_next.x)**2 + 
                            (pred_result.y_coords[0] - actual_next.y)**2
                        )
                        position_errors.append(pos_error)
                        
                        # Calculate velocity error if available
                        if (pred_result.velocities is not None and 
                            actual_next.velocity is not None):
                            vel_error = abs(pred_result.velocities[0] - actual_next.velocity)
                            velocity_errors.append(vel_error)
                        
                        # Calculate acceleration error if available
                        if (pred_result.accelerations is not None and 
                            actual_next.acceleration is not None):
                            accel_error = abs(pred_result.accelerations[0] - actual_next.acceleration)
                            acceleration_errors.append(accel_error)
                            
                except Exception:
                    # Skip problematic predictions during training
                    continue
        
        # Update noise parameters based on observed errors
        if position_errors:
            self._position_noise_std = np.std(position_errors)
        
        if velocity_errors:
            self._velocity_noise_std = np.std(velocity_errors)
        
        if acceleration_errors:
            self._acceleration_noise_std = np.std(acceleration_errors)
        
        self.is_trained = True
        
        return {
            "model": self.name,
            "training_samples": len(trajectories),
            "vehicles_analyzed": len(vehicle_trajectories),
            "position_noise_std": float(self._position_noise_std),
            "velocity_noise_std": float(self._velocity_noise_std),
            "acceleration_noise_std": float(self._acceleration_noise_std),
            "mean_position_error": float(np.mean(position_errors)) if position_errors else 0.0,
            "mean_velocity_error": float(np.mean(velocity_errors)) if velocity_errors else 0.0,
            "mean_acceleration_error": float(np.mean(acceleration_errors)) if acceleration_errors else 0.0
        }
    
    async def predict(
        self,
        history: List[TrajectoryData],
        prediction_horizon: float,
        **kwargs
    ) -> PredictionResult:
        """
        Predict future trajectory using constant acceleration assumption.
        
        Uses kinematic equations:
        - x(t) = x₀ + v₀ₓ·t + ½·aₓ·t²
        - y(t) = y₀ + v₀ᵧ·t + ½·aᵧ·t²
        - v(t) = v₀ + a·t
        
        Args:
            history: Historical trajectory points (need >= 3 for acceleration)
            prediction_horizon: How far to predict (seconds)
            **kwargs: Additional parameters
            
        Returns:
            Predicted trajectory with uncertainty estimates
        """
        self._validate_history(history)
        
        if len(history) < self.min_history_length:
            raise ValueError(f"Need at least {self.min_history_length} history points for acceleration estimation")
        
        return await self._predict_from_history(history, prediction_horizon, **kwargs)
    
    async def _predict_from_history(
        self,
        history: List[TrajectoryData],
        prediction_horizon: float,
        time_step_size: float = 0.1,
        **kwargs
    ) -> PredictionResult:
        """Internal prediction method using kinematic equations."""
        
        # Estimate current velocity and acceleration
        velocity_x, velocity_y = self._estimate_velocity(history)
        acceleration_x, acceleration_y = self._estimate_acceleration(history)
        
        # Generate future time points
        num_steps = max(1, int(prediction_horizon / time_step_size))
        time_steps = np.linspace(time_step_size, prediction_horizon, num_steps)
        
        # Current position (last known point)
        last_point = history[-1]
        x0, y0 = last_point.x, last_point.y
        
        # Kinematic equations: x(t) = x0 + v0*t + 0.5*a*t²
        future_x = x0 + velocity_x * time_steps + 0.5 * acceleration_x * time_steps**2
        future_y = y0 + velocity_y * time_steps + 0.5 * acceleration_y * time_steps**2
        
        # Velocity evolution: v(t) = v0 + a*t
        future_vx = velocity_x + acceleration_x * time_steps
        future_vy = velocity_y + acceleration_y * time_steps
        future_velocities = np.sqrt(future_vx**2 + future_vy**2)
        
        # Constant acceleration
        future_accelerations = np.full(num_steps, np.sqrt(acceleration_x**2 + acceleration_y**2))
        
        # Generate timestamps
        base_timestamp = last_point.timestamp
        future_timestamps = np.array([
            (base_timestamp.timestamp() + t) * 1000 for t in time_steps
        ])
        
        # Uncertainty estimation (grows with time)
        # Position uncertainty grows as t² due to acceleration uncertainty
        time_factor = time_steps
        time_squared_factor = time_steps**2
        
        # Uncertainty sources: initial position, velocity, and acceleration
        position_uncertainty = self._position_noise_std
        velocity_uncertainty = self._velocity_noise_std * time_factor
        acceleration_uncertainty = self._acceleration_noise_std * 0.5 * time_squared_factor
        
        # Combined uncertainty (root sum of squares)
        x_std = np.sqrt(
            position_uncertainty**2 + 
            velocity_uncertainty**2 + 
            acceleration_uncertainty**2
        )
        y_std = x_std.copy()  # Assume similar uncertainty in both dimensions
        
        return PredictionResult(
            timestamps=future_timestamps,
            x_coords=future_x,
            y_coords=future_y,
            x_std=x_std,
            y_std=y_std,
            velocities=future_velocities,
            accelerations=future_accelerations,
            prediction_horizon=prediction_horizon,
            model_name=self.name,
            metadata={
                "initial_velocity_x": float(velocity_x),
                "initial_velocity_y": float(velocity_y),
                "acceleration_x": float(acceleration_x),
                "acceleration_y": float(acceleration_y),
                "position_noise_std": float(self._position_noise_std),
                "velocity_noise_std": float(self._velocity_noise_std),
                "acceleration_noise_std": float(self._acceleration_noise_std),
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
        
        # Use most recent velocity estimate
        recent_points = history[-2:]  # Use last two points
        
        dt = (recent_points[1].timestamp - recent_points[0].timestamp).total_seconds()
        
        if dt > 0:
            vx = (recent_points[1].x - recent_points[0].x) / dt
            vy = (recent_points[1].y - recent_points[0].y) / dt
            return vx, vy
        else:
            return 0.0, 0.0
    
    def _estimate_acceleration(self, history: List[TrajectoryData]) -> tuple:
        """
        Estimate current acceleration from trajectory history.
        
        Args:
            history: Trajectory points (need >= 3 for acceleration)
            
        Returns:
            Tuple of (acceleration_x, acceleration_y)
        """
        if len(history) < 3:
            return 0.0, 0.0
        
        # Use smoothing window for more stable estimate
        window_size = min(self.acceleration_smoothing_window, len(history))
        recent_points = history[-window_size:]
        
        # Calculate accelerations from velocity changes
        accelerations_x = []
        accelerations_y = []
        
        # Calculate velocities first
        velocities_x = []
        velocities_y = []
        time_points = []
        
        for i in range(1, len(recent_points)):
            dt = (recent_points[i].timestamp - recent_points[i-1].timestamp).total_seconds()
            if dt > 0:
                vx = (recent_points[i].x - recent_points[i-1].x) / dt
                vy = (recent_points[i].y - recent_points[i-1].y) / dt
                velocities_x.append(vx)
                velocities_y.append(vy)
                time_points.append(recent_points[i].timestamp)
        
        # Calculate accelerations from velocity changes
        if len(velocities_x) >= 2:
            for i in range(1, len(velocities_x)):
                dt = (time_points[i] - time_points[i-1]).total_seconds()
                if dt > 0:
                    ax = (velocities_x[i] - velocities_x[i-1]) / dt
                    ay = (velocities_y[i] - velocities_y[i-1]) / dt
                    accelerations_x.append(ax)
                    accelerations_y.append(ay)
        
        # Return average acceleration
        if accelerations_x:
            return np.mean(accelerations_x), np.mean(accelerations_y)
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
            "description": "Constant acceleration trajectory prediction using kinematic equations",
            "parameters": {
                "min_history_length": self.min_history_length,
                "acceleration_smoothing_window": self.acceleration_smoothing_window,
                "position_noise_std": float(self._position_noise_std),
                "velocity_noise_std": float(self._velocity_noise_std),
                "acceleration_noise_std": float(self._acceleration_noise_std)
            },
            "capabilities": {
                "uncertainty_estimation": True,
                "online_learning": False,
                "batch_prediction": True,
                "kinematic_consistency": True
            },
            "equations_used": [
                "x(t) = x₀ + v₀ₓ·t + ½·aₓ·t²",
                "y(t) = y₀ + v₀ᵧ·t + ½·aᵧ·t²", 
                "v(t) = v₀ + a·t"
            ],
            "is_trained": self.is_trained
        }