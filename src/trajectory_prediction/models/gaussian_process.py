"""
Gaussian Process trajectory prediction model with uncertainty quantification.

Uses Gaussian Processes for smooth trajectory prediction with principled
uncertainty estimation.
"""

import asyncio
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from omegaconf import DictConfig

from ..data.validation.schemas import TrajectoryData
from .base import TrajectoryPredictor, PredictionResult


class GaussianProcessPredictor(TrajectoryPredictor):
    """
    Gaussian Process trajectory prediction model with uncertainty quantification.
    
    Features:
    - Separate GP models for X and Y coordinates
    - RBF kernel with automatic hyperparameter optimization
    - Principled uncertainty quantification
    - Smooth predictions with temporal features
    """
    
    def __init__(self, config: DictConfig, name: Optional[str] = None):
        super().__init__(config, name or "GaussianProcess")
        
        # Model parameters
        self.length_scale = config.get("length_scale", 1.0)
        self.noise_level = config.get("noise_level", 0.1)
        self.min_history_length = config.get("min_history_length", 5)
        self.optimize_hyperparameters = config.get("optimize_hyperparameters", True)
        self.max_velocity = config.get("max_velocity", 100.0)
        
        # Model components
        self.gp_x = None
        self.gp_y = None
        self.feature_scaler = StandardScaler()
        
    async def fit(
        self, 
        trajectories: List[TrajectoryData],
        validation_trajectories: Optional[List[TrajectoryData]] = None
    ) -> Dict[str, Any]:
        """Train Gaussian Process models for trajectory prediction."""
        
        # Prepare training data
        X_train, y_train_x, y_train_y = await self._prepare_training_data(trajectories)
        
        if len(X_train) == 0:
            raise ValueError("No valid training data available")
        
        # Scale features
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        
        # Create kernel: Constant * RBF + White noise
        kernel = ConstantKernel(1.0) * RBF(length_scale=self.length_scale) + WhiteKernel(noise_level=self.noise_level)
        
        # Create GP models
        self.gp_x = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,  # Additional noise for numerical stability
            normalize_y=True,
            n_restarts_optimizer=10 if self.optimize_hyperparameters else 0
        )
        
        self.gp_y = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=10 if self.optimize_hyperparameters else 0
        )
        
        # Fit models
        self.gp_x.fit(X_train_scaled, y_train_x)
        self.gp_y.fit(X_train_scaled, y_train_y)
        
        # Calculate training metrics
        y_pred_x, y_std_x = self.gp_x.predict(X_train_scaled, return_std=True)
        y_pred_y, y_std_y = self.gp_y.predict(X_train_scaled, return_std=True)
        
        train_rmse_x = np.sqrt(mean_squared_error(y_train_x, y_pred_x))
        train_rmse_y = np.sqrt(mean_squared_error(y_train_y, y_pred_y))
        
        self.is_trained = True
        
        return {
            "model": self.name,
            "training_samples": len(X_train),
            "feature_dimension": X_train.shape[1],
            "train_rmse_x": float(train_rmse_x),
            "train_rmse_y": float(train_rmse_y),
            "mean_uncertainty_x": float(np.mean(y_std_x)),
            "mean_uncertainty_y": float(np.mean(y_std_y)),
            "optimized_kernel_x": str(self.gp_x.kernel_),
            "optimized_kernel_y": str(self.gp_y.kernel_)
        }
    
    async def predict(
        self,
        history: List[TrajectoryData],
        prediction_horizon: float,
        **kwargs
    ) -> PredictionResult:
        """Predict future trajectory using Gaussian Process."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        self._validate_history(history)
        
        if len(history) < self.min_history_length:
            raise ValueError(f"Need at least {self.min_history_length} history points")
        
        return await self._predict_from_history(history, prediction_horizon, **kwargs)
    
    async def _predict_from_history(
        self,
        history: List[TrajectoryData],
        prediction_horizon: float,
        time_step_size: float = 0.1,
        confidence_level: float = 0.95,
        **kwargs
    ) -> PredictionResult:
        """Internal prediction method using Gaussian Process."""
        
        # Generate future time points
        num_steps = max(1, int(prediction_horizon / time_step_size))
        time_steps = np.linspace(time_step_size, prediction_horizon, num_steps)
        
        # Create features for prediction
        X_pred = self._create_prediction_features(history, time_steps)
        X_pred_scaled = self.feature_scaler.transform(X_pred)
        
        # Make predictions with uncertainty
        pred_x_mean, pred_x_std = self.gp_x.predict(X_pred_scaled, return_std=True)
        pred_y_mean, pred_y_std = self.gp_y.predict(X_pred_scaled, return_std=True)
        
        # Apply velocity constraints
        pred_x_mean, pred_y_mean = self._apply_velocity_constraints(
            pred_x_mean, pred_y_mean, time_steps
        )
        
        # Calculate confidence intervals
        from scipy.stats import norm
        confidence_multiplier = norm.ppf((1 + confidence_level) / 2)
        pred_x_std_conf = pred_x_std * confidence_multiplier
        pred_y_std_conf = pred_y_std * confidence_multiplier
        
        # Calculate velocities and accelerations
        velocities, accelerations = self._calculate_kinematics(
            pred_x_mean, pred_y_mean, time_steps
        )
        
        # Generate timestamps
        last_point = history[-1]
        base_time = last_point.timestamp.timestamp()
        future_timestamps = np.array([(base_time + t) * 1000 for t in time_steps])
        
        return PredictionResult(
            timestamps=future_timestamps,
            x_coords=pred_x_mean,
            y_coords=pred_y_mean,
            x_std=pred_x_std_conf,
            y_std=pred_y_std_conf,
            velocities=velocities,
            accelerations=accelerations,
            prediction_horizon=prediction_horizon,
            confidence_level=confidence_level,
            model_name=self.name,
            metadata={
                "feature_dimension": X_pred.shape[1],
                "mean_uncertainty_x": float(np.mean(pred_x_std)),
                "mean_uncertainty_y": float(np.mean(pred_y_std)),
                "confidence_level": confidence_level
            }
        )
    
    async def _prepare_training_data(
        self, 
        trajectories: List[TrajectoryData]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare training data with feature engineering."""
        
        # Group by vehicle
        vehicle_trajectories = self._group_by_vehicle(trajectories)
        
        all_X = []
        all_y_x = []
        all_y_y = []
        
        for vehicle_id, traj_points in vehicle_trajectories.items():
            if len(traj_points) < self.min_history_length:
                continue
                
            # Sort by timestamp
            traj_points = sorted(traj_points, key=lambda x: x.timestamp)
            
            # Create training examples using sliding windows
            for i in range(self.min_history_length, len(traj_points)):
                history = traj_points[i-self.min_history_length:i]
                target_point = traj_points[i]
                
                # Extract features from history
                features = self._extract_features(history)
                
                if features is not None:
                    all_X.append(features)
                    all_y_x.append(target_point.x)
                    all_y_y.append(target_point.y)
        
        if not all_X:
            return np.array([]), np.array([]), np.array([])
        
        X = np.vstack(all_X)
        y_x = np.array(all_y_x)
        y_y = np.array(all_y_y)
        
        return X, y_x, y_y
    
    def _extract_features(self, history: List[TrajectoryData]) -> Optional[np.ndarray]:
        """Extract features from trajectory history for GP."""
        if len(history) < 2:
            return None
        
        features = []
        
        # Position features
        positions_x = np.array([p.x for p in history])
        positions_y = np.array([p.y for p in history])
        
        features.extend([
            positions_x[-1], positions_y[-1],  # Current position
            np.mean(positions_x), np.mean(positions_y),  # Average position
            positions_x[-1] - positions_x[0], positions_y[-1] - positions_y[0]  # Displacement
        ])
        
        # Velocity features
        velocities = []
        for i in range(1, len(history)):
            dt = (history[i].timestamp - history[i-1].timestamp).total_seconds()
            if dt > 0:
                vx = (history[i].x - history[i-1].x) / dt
                vy = (history[i].y - history[i-1].y) / dt
                velocities.append(np.sqrt(vx**2 + vy**2))
        
        if velocities:
            features.extend([
                velocities[-1],  # Current velocity
                np.mean(velocities),  # Average velocity
                np.std(velocities) if len(velocities) > 1 else 0.0  # Velocity variation
            ])
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # Time features
        times = np.array([p.timestamp.timestamp() for p in history])
        time_diffs = np.diff(times)
        features.extend([
            np.mean(time_diffs) if len(time_diffs) > 0 else 0.1,  # Average time step
            times[-1] - times[0]  # Total time span
        ])
        
        return np.array(features)
    
    def _create_prediction_features(
        self, 
        history: List[TrajectoryData], 
        future_times: np.ndarray
    ) -> np.ndarray:
        """Create features for prediction at future time points."""
        # Extract base features from history
        base_features = self._extract_features(history)
        
        if base_features is None:
            # Fallback features
            last_point = history[-1]
            base_features = np.array([
                last_point.x, last_point.y, 
                last_point.x, last_point.y,
                0.0, 0.0,  # displacement
                0.0, 0.0, 0.0,  # velocity features
                0.1, 0.0  # time features
            ])
        
        # Create prediction features for each future time
        X_pred = []
        for t_offset in future_times:
            features = base_features.copy()
            # Update time-related features
            features[-1] = t_offset  # Update time span to future offset
            X_pred.append(features)
        
        return np.array(X_pred)
    
    def _apply_velocity_constraints(
        self,
        pred_x: np.ndarray,
        pred_y: np.ndarray,
        time_steps: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply velocity constraints to predictions."""
        # Simple velocity limiting
        for i in range(1, len(pred_x)):
            dt = time_steps[i] - time_steps[i-1]
            if dt > 0:
                dx = pred_x[i] - pred_x[i-1]
                dy = pred_y[i] - pred_y[i-1]
                velocity = np.sqrt(dx**2 + dy**2) / dt
                
                if velocity > self.max_velocity:
                    scale = self.max_velocity / velocity
                    pred_x[i] = pred_x[i-1] + dx * scale
                    pred_y[i] = pred_y[i-1] + dy * scale
        
        return pred_x, pred_y
    
    def _calculate_kinematics(
        self, 
        pred_x: np.ndarray, 
        pred_y: np.ndarray, 
        time_steps: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate velocities and accelerations from positions."""
        velocities = np.zeros(len(pred_x))
        accelerations = np.zeros(len(pred_x))
        
        # Calculate velocities
        for i in range(1, len(pred_x)):
            dt = time_steps[i] - time_steps[i-1]
            if dt > 0:
                dx = pred_x[i] - pred_x[i-1]
                dy = pred_y[i] - pred_y[i-1]
                velocities[i] = np.sqrt(dx**2 + dy**2) / dt
        
        # Calculate accelerations
        for i in range(1, len(velocities)):
            dt = time_steps[i] - time_steps[i-1]
            if dt > 0:
                accelerations[i] = (velocities[i] - velocities[i-1]) / dt
        
        return velocities, accelerations
    
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
        kernel_info = {}
        if self.is_trained:
            kernel_info = {
                "trained_kernel_x": str(self.gp_x.kernel_),
                "trained_kernel_y": str(self.gp_y.kernel_),
            }
        
        return {
            "name": self.name,
            "type": "gaussian_process",
            "description": "Gaussian Process trajectory prediction with uncertainty quantification",
            "parameters": {
                "length_scale": self.length_scale,
                "noise_level": self.noise_level,
                "min_history_length": self.min_history_length,
                "optimize_hyperparameters": self.optimize_hyperparameters,
                "max_velocity": self.max_velocity
            },
            "capabilities": {
                "uncertainty_quantification": True,
                "principled_uncertainty": True,
                "smooth_predictions": True,
                "confidence_intervals": True,
                "hyperparameter_optimization": self.optimize_hyperparameters,
                "online_learning": False,
                "batch_prediction": True
            },
            **kernel_info,
            "is_trained": self.is_trained
        }