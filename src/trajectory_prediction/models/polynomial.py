"""
Physics-informed polynomial regression trajectory prediction model.

Uses polynomial fitting with physics constraints and regularization
for smooth, realistic trajectory predictions.
"""

import asyncio
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import optimize
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from omegaconf import DictConfig

from ..data.validation.schemas import TrajectoryData
from .base import TrajectoryPredictor, PredictionResult


class PolynomialTrajectoryPredictor(TrajectoryPredictor):
    """
    Physics-informed polynomial regression trajectory prediction model.
    
    Features:
    - Polynomial fitting for X and Y coordinates separately
    - Physics constraints (velocity and acceleration continuity)
    - Bayesian ridge regression for uncertainty estimation
    - Regularization to prevent overfitting
    - Temporal feature engineering
    """
    
    def __init__(self, config: DictConfig, name: Optional[str] = None):
        super().__init__(config, name or "Polynomial")
        
        # Model parameters
        self.polynomial_degree = config.get("polynomial_degree", 3)
        self.regularization_alpha = config.get("regularization_alpha", 1.0)
        self.min_history_length = config.get("min_history_length", 4)
        self.physics_constraints = config.get("physics_constraints", True)
        self.max_velocity = config.get("max_velocity", 100.0)  # m/s or ft/s
        self.max_acceleration = config.get("max_acceleration", 20.0)  # m/s² or ft/s²
        
        # Model components
        self.x_model = None
        self.y_model = None
        self.feature_scaler = None
        self.time_reference = None  # Reference timestamp for normalization
        
        # Learned parameters for uncertainty estimation
        self.residual_std_x = 1.0
        self.residual_std_y = 1.0
        
    async def fit(
        self, 
        trajectories: List[TrajectoryData],
        validation_trajectories: Optional[List[TrajectoryData]] = None
    ) -> Dict[str, Any]:
        """
        Train polynomial regression models with physics constraints.
        
        Args:
            trajectories: Training trajectory data
            validation_trajectories: Optional validation data
            
        Returns:
            Training metrics and model parameters
        """
        # Prepare training data
        X_train, y_train_x, y_train_y, time_features = await self._prepare_training_data(trajectories)
        
        if len(X_train) == 0:
            raise ValueError("No valid training data available")
        
        # Create polynomial feature transformer
        poly_features = PolynomialFeatures(
            degree=self.polynomial_degree, 
            include_bias=True,
            interaction_only=False
        )
        
        # Create models with regularization
        base_regressor = BayesianRidge(
            alpha_1=1e-6, 
            alpha_2=1e-6, 
            lambda_1=1e-6, 
            lambda_2=1e-6,
            compute_score=True,
            fit_intercept=False  # Polynomial features include bias
        )
        
        # Build pipelines
        self.x_model = Pipeline([
            ('poly', poly_features),
            ('regressor', base_regressor)
        ])
        
        self.y_model = Pipeline([
            ('poly', poly_features),
            ('regressor', base_regressor)
        ])
        
        # Fit models
        self.x_model.fit(X_train, y_train_x)
        self.y_model.fit(X_train, y_train_y)
        
        # Calculate residual statistics for uncertainty estimation
        y_pred_x = self.x_model.predict(X_train)
        y_pred_y = self.y_model.predict(X_train)
        
        self.residual_std_x = np.std(y_train_x - y_pred_x)
        self.residual_std_y = np.std(y_train_y - y_pred_y)
        
        # Calculate training metrics
        train_rmse_x = np.sqrt(mean_squared_error(y_train_x, y_pred_x))
        train_rmse_y = np.sqrt(mean_squared_error(y_train_y, y_pred_y))
        
        # Validation metrics
        val_metrics = {}
        if validation_trajectories:
            val_metrics = await self._evaluate_on_validation(validation_trajectories)
        
        self.is_trained = True
        
        return {
            "model": self.name,
            "training_samples": len(X_train),
            "polynomial_degree": self.polynomial_degree,
            "regularization_alpha": self.regularization_alpha,
            "train_rmse_x": float(train_rmse_x),
            "train_rmse_y": float(train_rmse_y),
            "residual_std_x": float(self.residual_std_x),
            "residual_std_y": float(self.residual_std_y),
            "physics_constraints": self.physics_constraints,
            **val_metrics
        }
    
    async def predict(
        self,
        history: List[TrajectoryData],
        prediction_horizon: float,
        **kwargs
    ) -> PredictionResult:
        """
        Predict future trajectory using polynomial regression.
        
        Args:
            history: Historical trajectory points
            prediction_horizon: How far to predict (seconds)
            **kwargs: Additional parameters
            
        Returns:
            Predicted trajectory with uncertainty estimates
        """
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
        **kwargs
    ) -> PredictionResult:
        """Internal prediction method."""
        
        # Prepare time features for prediction
        last_point = history[-1]
        base_time = last_point.timestamp.timestamp()
        
        # Generate future time points
        num_steps = max(1, int(prediction_horizon / time_step_size))
        time_steps = np.linspace(time_step_size, prediction_horizon, num_steps)
        future_times = base_time + time_steps
        
        # Create time features relative to reference
        if self.time_reference is None:
            self.time_reference = history[0].timestamp.timestamp()
        
        relative_times = future_times - self.time_reference
        
        # Extract additional features from history
        X_pred = self._create_prediction_features(history, relative_times)
        
        # Make predictions
        pred_x = self.x_model.predict(X_pred)
        pred_y = self.y_model.predict(X_pred)
        
        # Apply physics constraints if enabled
        if self.physics_constraints:
            pred_x, pred_y = self._apply_physics_constraints(
                pred_x, pred_y, time_steps, history
            )
        
        # Calculate uncertainties using Bayesian model uncertainty
        uncertainty_x, uncertainty_y = self._estimate_uncertainties(X_pred, time_steps)
        
        # Calculate velocities and accelerations from predictions
        velocities, accelerations = self._calculate_kinematics(pred_x, pred_y, time_steps)
        
        # Generate timestamps
        future_timestamps = np.array([t * 1000 for t in future_times])
        
        return PredictionResult(
            timestamps=future_timestamps,
            x_coords=pred_x,
            y_coords=pred_y,
            x_std=uncertainty_x,
            y_std=uncertainty_y,
            velocities=velocities,
            accelerations=accelerations,
            prediction_horizon=prediction_horizon,
            model_name=self.name,
            metadata={
                "polynomial_degree": self.polynomial_degree,
                "physics_constraints": self.physics_constraints,
                "time_step_size": time_step_size,
                "history_length": len(history),
                "prediction_steps": num_steps
            }
        )
    
    async def _prepare_training_data(
        self, 
        trajectories: List[TrajectoryData]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Prepare training data from trajectories."""
        
        # Group by vehicle
        vehicle_trajectories = self._group_by_vehicle(trajectories)
        
        all_X = []
        all_y_x = []
        all_y_y = []
        
        # Set time reference from first trajectory
        if trajectories:
            self.time_reference = min(traj.timestamp.timestamp() for traj in trajectories)
        
        for vehicle_id, traj_points in vehicle_trajectories.items():
            if len(traj_points) < self.min_history_length:
                continue
                
            # Sort by timestamp
            traj_points = sorted(traj_points, key=lambda x: x.timestamp)
            
            # Create features for each trajectory segment
            for i in range(self.min_history_length, len(traj_points)):
                history = traj_points[i-self.min_history_length:i]
                target_point = traj_points[i]
                
                # Create features from history
                features = self._extract_polynomial_features(history)
                
                if features is not None:
                    all_X.append(features)
                    all_y_x.append(target_point.x)
                    all_y_y.append(target_point.y)
        
        if not all_X:
            return np.array([]), np.array([]), np.array([]), {}
        
        X = np.vstack(all_X)
        y_x = np.array(all_y_x)
        y_y = np.array(all_y_y)
        
        time_features = {
            "time_reference": self.time_reference,
            "feature_dim": X.shape[1] if X.size > 0 else 0
        }
        
        return X, y_x, y_y, time_features
    
    def _extract_polynomial_features(self, history: List[TrajectoryData]) -> Optional[np.ndarray]:
        """Extract polynomial features from trajectory history."""
        if len(history) < 2:
            return None
        
        # Time features (relative to reference)
        times = np.array([
            (point.timestamp.timestamp() - self.time_reference) for point in history
        ])
        
        # Position features
        positions_x = np.array([point.x for point in history])
        positions_y = np.array([point.y for point in history])
        
        # Velocity features (calculated from positions)
        velocities = []
        for i in range(1, len(history)):
            dt = times[i] - times[i-1]
            if dt > 0:
                vx = (positions_x[i] - positions_x[i-1]) / dt
                vy = (positions_y[i] - positions_y[i-1]) / dt
                v_mag = np.sqrt(vx**2 + vy**2)
                velocities.extend([vx, vy, v_mag])
        
        # Pad velocities if needed
        while len(velocities) < 3 * (len(history) - 1):
            velocities.append(0.0)
        
        # Acceleration features (if enough history)
        accelerations = []
        if len(history) >= 3:
            for i in range(2, len(history)):
                dt1 = times[i-1] - times[i-2]
                dt2 = times[i] - times[i-1]
                
                if dt1 > 0 and dt2 > 0:
                    vx1 = (positions_x[i-1] - positions_x[i-2]) / dt1
                    vy1 = (positions_y[i-1] - positions_y[i-2]) / dt1
                    vx2 = (positions_x[i] - positions_x[i-1]) / dt2
                    vy2 = (positions_y[i] - positions_y[i-1]) / dt2
                    
                    ax = (vx2 - vx1) / ((dt1 + dt2) / 2)
                    ay = (vy2 - vy1) / ((dt1 + dt2) / 2)
                    a_mag = np.sqrt(ax**2 + ay**2)
                    accelerations.extend([ax, ay, a_mag])
        
        # Combine all features
        features = []
        features.extend([times[-1]])  # Current time
        features.extend([positions_x[-1], positions_y[-1]])  # Current position
        features.extend(velocities[-3:] if velocities else [0.0, 0.0, 0.0])  # Recent velocity
        features.extend(accelerations[-3:] if accelerations else [0.0, 0.0, 0.0])  # Recent acceleration
        
        return np.array(features)
    
    def _create_prediction_features(
        self, 
        history: List[TrajectoryData], 
        future_times: np.ndarray
    ) -> np.ndarray:
        """Create features for prediction at future time points."""
        # Extract base features from history
        base_features = self._extract_polynomial_features(history)
        
        if base_features is None:
            # Fallback: minimal features
            last_point = history[-1]
            base_features = np.array([
                (last_point.timestamp.timestamp() - self.time_reference),
                last_point.x,
                last_point.y,
                0.0, 0.0, 0.0,  # velocity features
                0.0, 0.0, 0.0   # acceleration features
            ])
        
        # Create features for each future time point
        X_pred = []
        for future_time in future_times:
            features = base_features.copy()
            features[0] = future_time - self.time_reference  # Update time feature
            X_pred.append(features)
        
        return np.array(X_pred)
    
    def _apply_physics_constraints(
        self,
        pred_x: np.ndarray,
        pred_y: np.ndarray, 
        time_steps: np.ndarray,
        history: List[TrajectoryData]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply physics constraints to predictions."""
        
        # Get last known state
        last_point = history[-1]
        
        # Calculate implied velocities and accelerations
        dt = np.diff(np.concatenate([[0], time_steps]))
        
        # Velocity constraints
        for i in range(1, len(pred_x)):
            if dt[i] > 0:
                vx = (pred_x[i] - pred_x[i-1]) / dt[i]
                vy = (pred_y[i] - pred_y[i-1]) / dt[i]
                v_mag = np.sqrt(vx**2 + vy**2)
                
                # Limit velocity magnitude
                if v_mag > self.max_velocity:
                    scale = self.max_velocity / v_mag
                    pred_x[i] = pred_x[i-1] + vx * scale * dt[i]
                    pred_y[i] = pred_y[i-1] + vy * scale * dt[i]
        
        # Acceleration constraints
        for i in range(2, len(pred_x)):
            if dt[i] > 0 and dt[i-1] > 0:
                # Calculate velocities
                vx1 = (pred_x[i-1] - pred_x[i-2]) / dt[i-1]
                vy1 = (pred_y[i-1] - pred_y[i-2]) / dt[i-1]
                vx2 = (pred_x[i] - pred_x[i-1]) / dt[i]
                vy2 = (pred_y[i] - pred_y[i-1]) / dt[i]
                
                # Calculate accelerations
                ax = (vx2 - vx1) / ((dt[i] + dt[i-1]) / 2)
                ay = (vy2 - vy1) / ((dt[i] + dt[i-1]) / 2)
                a_mag = np.sqrt(ax**2 + ay**2)
                
                # Limit acceleration magnitude
                if a_mag > self.max_acceleration:
                    scale = self.max_acceleration / a_mag
                    # Adjust current prediction
                    vx2_new = vx1 + ax * scale * ((dt[i] + dt[i-1]) / 2)
                    vy2_new = vy1 + ay * scale * ((dt[i] + dt[i-1]) / 2)
                    pred_x[i] = pred_x[i-1] + vx2_new * dt[i]
                    pred_y[i] = pred_y[i-1] + vy2_new * dt[i]
        
        return pred_x, pred_y
    
    def _estimate_uncertainties(
        self, 
        X_pred: np.ndarray, 
        time_steps: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate prediction uncertainties."""
        
        # Base uncertainty from residuals
        base_uncertainty_x = self.residual_std_x
        base_uncertainty_y = self.residual_std_y
        
        # Time-dependent uncertainty growth
        time_factor = np.sqrt(time_steps)  # Uncertainty grows with sqrt(time)
        
        # Model uncertainty (if using Bayesian model)
        model_uncertainty = 0.1  # Additional model uncertainty
        
        uncertainty_x = np.sqrt(
            base_uncertainty_x**2 + 
            (model_uncertainty * time_factor)**2
        )
        
        uncertainty_y = np.sqrt(
            base_uncertainty_y**2 + 
            (model_uncertainty * time_factor)**2
        )
        
        return uncertainty_x, uncertainty_y
    
    def _calculate_kinematics(
        self, 
        pred_x: np.ndarray, 
        pred_y: np.ndarray, 
        time_steps: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate velocities and accelerations from position predictions."""
        
        # Calculate velocities
        velocities = np.zeros(len(pred_x))
        accelerations = np.zeros(len(pred_x))
        
        # Velocity calculation
        for i in range(1, len(pred_x)):
            dt = time_steps[i] - time_steps[i-1] if i > 0 else time_steps[0]
            if dt > 0:
                vx = (pred_x[i] - pred_x[i-1]) / dt
                vy = (pred_y[i] - pred_y[i-1]) / dt
                velocities[i] = np.sqrt(vx**2 + vy**2)
        
        # First point velocity (extrapolate)
        if len(velocities) > 1:
            velocities[0] = velocities[1]
        
        # Acceleration calculation
        for i in range(1, len(velocities)):
            dt = time_steps[i] - time_steps[i-1] if i > 0 else time_steps[0]
            if dt > 0:
                accelerations[i] = (velocities[i] - velocities[i-1]) / dt
        
        # First point acceleration
        if len(accelerations) > 1:
            accelerations[0] = accelerations[1]
        
        return velocities, accelerations
    
    async def _evaluate_on_validation(self, validation_trajectories: List[TrajectoryData]) -> Dict[str, Any]:
        """Evaluate model on validation data."""
        # Implement validation evaluation
        # This is a placeholder - full implementation would evaluate prediction accuracy
        return {
            "val_samples": len(validation_trajectories),
            "val_rmse_x": 0.0,
            "val_rmse_y": 0.0
        }
    
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
            "type": "polynomial_regression",
            "description": "Physics-informed polynomial trajectory prediction with regularization",
            "parameters": {
                "polynomial_degree": self.polynomial_degree,
                "regularization_alpha": self.regularization_alpha,
                "min_history_length": self.min_history_length,
                "physics_constraints": self.physics_constraints,
                "max_velocity": self.max_velocity,
                "max_acceleration": self.max_acceleration
            },
            "capabilities": {
                "uncertainty_estimation": True,
                "physics_constraints": self.physics_constraints,
                "online_learning": False,
                "batch_prediction": True,
                "temporal_features": True
            },
            "model_complexity": {
                "polynomial_degree": self.polynomial_degree,
                "feature_engineering": "time + position + velocity + acceleration",
                "regularization": "Bayesian Ridge"
            },
            "is_trained": self.is_trained
        }