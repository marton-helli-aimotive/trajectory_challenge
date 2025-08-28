"""Polynomial regression model for trajectory prediction."""

from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
import math

# Import numpy from base module to ensure consistency
from .base import np, NUMPY_AVAILABLE

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

try:
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_squared_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    # Create mock sklearn classes
    class PolynomialFeatures:
        def __init__(self, degree=2, include_bias=False):
            self.degree = degree
            self.include_bias = include_bias
        
        def fit_transform(self, X):
            return X
        
        def transform(self, X):
            return X
        
        def get_feature_names_out(self):
            return ['feature_1', 'feature_2']
    
    class StandardScaler:
        def fit_transform(self, X):
            return X
        
        def transform(self, X):
            return X
    
    class LinearRegression:
        def __init__(self):
            pass
        
        def fit(self, X, y):
            pass
        
        def predict(self, X):
            return [[0.0, 0.0]]  # Mock prediction
    
    class Ridge:
        def __init__(self, alpha=1.0, random_state=None):
            self.alpha = alpha
            self.random_state = random_state
        
        def fit(self, X, y):
            pass
        
        def predict(self, X):
            return [[0.0, 0.0]]  # Mock prediction
    
    class Lasso:
        def __init__(self, alpha=1.0, random_state=None):
            self.alpha = alpha
            self.random_state = random_state
        
        def fit(self, X, y):
            pass
        
        def predict(self, X):
            return [[0.0, 0.0]]  # Mock prediction
    
    def cross_val_score(estimator, X, y, cv=5, scoring='neg_mean_squared_error'):
        return [-1.0] * cv  # Mock scores
    
    def mean_squared_error(y_true, y_pred):
        return 1.0  # Mock error

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    joblib = None

from .base import BaseTrajectoryPredictor, PredictionResult
from ..core.models import Trajectory, TrajectoryPoint
from ..core.config import ModelConfig

logger = logging.getLogger(__name__)


class PolynomialRegressionPredictor(BaseTrajectoryPredictor):
    """Polynomial regression model for trajectory prediction.
    
    This model uses polynomial features to capture non-linear relationships
    in trajectory data. It can handle multiple input features and provides
    a good balance between complexity and interpretability.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__(config, "PolynomialRegression")
        
        # Extract polynomial-specific configuration
        poly_config = getattr(config, 'polynomial_config', {})
        self.degree = poly_config.get('degree', 3)
        self.features = poly_config.get('features', ['position', 'velocity', 'acceleration'])
        self.regularization = poly_config.get('regularization', 'ridge')  # 'ridge', 'lasso', or 'linear'
        self.alpha = poly_config.get('alpha', 1.0)  # Regularization strength
        self.cross_validate = poly_config.get('cross_validate', True)
        self.cv_folds = poly_config.get('cv_folds', 5)
        
        # Model components
        self.poly_features = None
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = []
        
        # Training data storage
        self.X_train = None
        self.y_train = None
        
        logger.info(f"Initialized {self.model_name} with degree={self.degree}, features={self.features}")
    
    def _extract_features(self, trajectory: Trajectory) -> np.ndarray:
        """Extract features from trajectory for polynomial regression."""
        if trajectory.length < 3:
            raise ValueError("Trajectory must have at least 3 points for feature extraction")
        
        features_list = []
        
        # Use the last few points for feature extraction
        # This provides a window of context for prediction
        window_size = min(10, trajectory.length)
        recent_points = trajectory.points[-window_size:]
        
        for point in recent_points:
            point_features = []
            
            if 'position' in self.features:
                point_features.extend([point.x, point.y])
            
            if 'velocity' in self.features:
                point_features.append(point.velocity)
            
            if 'acceleration' in self.features:
                point_features.append(point.acceleration)
            
            if 'heading' in self.features:
                point_features.append(point.heading)
            
            if 'time' in self.features:
                # Normalize time relative to trajectory start
                time_from_start = (point.timestamp - trajectory.start_time).total_seconds()
                point_features.append(time_from_start)
            
            features_list.append(point_features)
        
        # Flatten features and pad if necessary
        flat_features = []
        for features in features_list:
            flat_features.extend(features)
        
        # Pad to consistent length if needed
        max_features = max(len(f) for f in features_list)
        while len(flat_features) < max_features * window_size:
            flat_features.append(0.0)
        
        return np.array(flat_features).reshape(1, -1)
    
    def _create_targets(self, trajectory: Trajectory, horizon: int, frequency: float) -> np.ndarray:
        """Create target values for training."""
        # Generate future timestamps
        timestamps = self._prepare_prediction_timestamps(trajectory, horizon, frequency)
        
        # Interpolate or extrapolate to get target positions
        target_x = []
        target_y = []
        
        for timestamp in timestamps:
            # Try to get actual trajectory point at this time
            actual_point = trajectory.get_point_at_time(timestamp)
            
            if actual_point is not None:
                # Use actual trajectory data if available
                target_x.append(actual_point.x)
                target_y.append(actual_point.y)
            else:
                # Extrapolate using constant velocity assumption
                last_point = trajectory.points[-1]
                dt = (timestamp - last_point.timestamp).total_seconds()
                
                # Simple extrapolation (can be improved)
                if len(trajectory.points) >= 2:
                    prev_point = trajectory.points[-2]
                    dt_prev = (last_point.timestamp - prev_point.timestamp).total_seconds()
                    if dt_prev > 0:
                        vx = (last_point.x - prev_point.x) / dt_prev
                        vy = (last_point.y - prev_point.y) / dt_prev
                        target_x.append(last_point.x + vx * dt)
                        target_y.append(last_point.y + vy * dt)
                    else:
                        target_x.append(last_point.x)
                        target_y.append(last_point.y)
                else:
                    target_x.append(last_point.x)
                    target_y.append(last_point.y)
        
        return np.array(target_x + target_y)
    
    def train(self, trajectories: List[Trajectory]) -> Dict[str, Any]:
        """Train the polynomial regression model."""
        logger.info(f"Training {self.model_name} model with {len(trajectories)} trajectories")
        
        if len(trajectories) < 2:
            raise ValueError("At least 2 trajectories are required for training")
        
        # Prepare training data
        X_list = []
        y_list = []
        
        horizon = self.config.prediction_horizon
        frequency = self.config.prediction_frequency
        
        for trajectory in trajectories:
            try:
                # Extract features
                features = self._extract_features(trajectory)
                X_list.append(features)
                
                # Create targets
                targets = self._create_targets(trajectory, horizon, frequency)
                y_list.append(targets)
                
            except Exception as e:
                logger.warning(f"Skipping trajectory {trajectory.vehicle_id}: {e}")
                continue
        
        if len(X_list) < 2:
            raise ValueError("Not enough valid trajectories for training")
        
        # Combine all training data
        self.X_train = np.vstack(X_list)
        self.y_train = np.vstack(y_list)
        
        logger.info(f"Training data shape: X={self.X_train.shape}, y={self.y_train.shape}")
        
        # Create polynomial features
        self.poly_features = PolynomialFeatures(degree=self.degree, include_bias=False)
        X_poly = self.poly_features.fit_transform(self.X_train)
        self.feature_names = self.poly_features.get_feature_names_out()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_poly)
        
        # Create and train model
        if self.regularization == 'ridge':
            self.model = Ridge(alpha=self.alpha, random_state=self.config.random_state)
        elif self.regularization == 'lasso':
            self.model = Lasso(alpha=self.alpha, random_state=self.config.random_state)
        else:
            self.model = LinearRegression()
        
        # Cross-validation if enabled
        cv_score = None
        if self.cross_validate and len(self.X_train) >= self.cv_folds:
            cv_scores = cross_val_score(
                self.model, X_scaled, self.y_train, 
                cv=self.cv_folds, scoring='neg_mean_squared_error'
            )
            cv_score = np.sqrt(-cv_scores.mean())
            logger.info(f"Cross-validation RMSE: {cv_score:.4f}")
        
        # Train the model
        self.model.fit(X_scaled, self.y_train)
        
        # Calculate training score
        y_pred = self.model.predict(X_scaled)
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_pred))
        
        self.is_trained = True
        self.training_data_size = len(trajectories)
        self.last_training_time = datetime.now()
        
        logger.info(f"Training completed. Train RMSE: {train_rmse:.4f}")
        
        return {
            "model_name": self.model_name,
            "training_data_size": self.training_data_size,
            "train_rmse": train_rmse,
            "cv_rmse": cv_score,
            "feature_count": len(self.feature_names),
            "polynomial_degree": self.degree,
            "regularization": self.regularization,
            "alpha": self.alpha,
            "training_time": self.last_training_time
        }
    
    def predict(
        self, 
        trajectory: Trajectory, 
        prediction_horizon: Optional[int] = None,
        prediction_frequency: Optional[float] = None
    ) -> PredictionResult:
        """Predict future trajectory using polynomial regression."""
        self.validate_input(trajectory)
        
        # Get prediction parameters
        horizon = prediction_horizon or self.config.prediction_horizon
        frequency = prediction_frequency or self.config.prediction_frequency
        
        # Extract features
        features = self._extract_features(trajectory)
        
        # Transform features
        X_poly = self.poly_features.transform(features)
        X_scaled = self.scaler.transform(X_poly)
        
        # Make prediction
        prediction = self.model.predict(X_scaled)[0]
        
        # Split prediction into x and y coordinates
        n_points = len(prediction) // 2
        predicted_x = prediction[:n_points]
        predicted_y = prediction[n_points:]
        
        # Generate timestamps
        timestamps = self._prepare_prediction_timestamps(trajectory, horizon, frequency)
        
        # Create predicted points
        predicted_points = []
        velocities = []
        accelerations = []
        headings_list = []
        
        last_point = trajectory.points[-1]
        
        for i, (timestamp, x, y) in enumerate(zip(timestamps, predicted_x, predicted_y)):
            # Calculate velocity and acceleration (simplified)
            if i == 0:
                # First prediction point
                dt = (timestamp - last_point.timestamp).total_seconds()
                if dt > 0:
                    velocity = np.sqrt((x - last_point.x)**2 + (y - last_point.y)**2) / dt
                    heading = np.arctan2(y - last_point.y, x - last_point.x)
                else:
                    velocity = last_point.velocity
                    heading = last_point.heading
                acceleration = 0.0
            else:
                # Subsequent points
                prev_x, prev_y = predicted_x[i-1], predicted_y[i-1]
                prev_timestamp = timestamps[i-1]
                dt = (timestamp - prev_timestamp).total_seconds()
                
                if dt > 0:
                    velocity = np.sqrt((x - prev_x)**2 + (y - prev_y)**2) / dt
                    heading = np.arctan2(y - prev_y, x - prev_x)
                else:
                    velocity = velocities[-1] if velocities else last_point.velocity
                    heading = headings_list[-1] if headings_list else last_point.heading
                
                # Simple acceleration calculation
                if len(velocities) > 0:
                    acceleration = (velocity - velocities[-1]) / dt
                else:
                    acceleration = 0.0
            
            # Create predicted point
            predicted_point = TrajectoryPoint(
                x=x, y=y, timestamp=timestamp, velocity=velocity,
                acceleration=acceleration, heading=heading,
                vehicle_id=last_point.vehicle_id, lane_id=last_point.lane_id
            )
            
            predicted_points.append(predicted_point)
            velocities.append(velocity)
            accelerations.append(acceleration)
            headings_list.append(heading)
        
        return PredictionResult(
            predicted_points=predicted_points,
            timestamps=timestamps,
            x_positions=np.array(predicted_x),
            y_positions=np.array(predicted_y),
            velocities=np.array(velocities),
            accelerations=np.array(accelerations),
            headings=np.array(headings_list),
            model_name=self.model_name,
            prediction_time=datetime.now(),
            metadata={
                "polynomial_degree": self.degree,
                "feature_count": len(self.feature_names),
                "regularization": self.regularization,
                "alpha": self.alpha
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
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model to disk."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before saving")
        
        model_data = {
            'poly_features': self.poly_features,
            'scaler': self.scaler,
            'model': self.model,
            'feature_names': self.feature_names,
            'config': {
                'degree': self.degree,
                'features': self.features,
                'regularization': self.regularization,
                'alpha': self.alpha
            }
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model from disk."""
        model_data = joblib.load(filepath)
        
        self.poly_features = model_data['poly_features']
        self.scaler = model_data['scaler']
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        
        # Restore configuration
        config = model_data['config']
        self.degree = config['degree']
        self.features = config['features']
        self.regularization = config['regularization']
        self.alpha = config['alpha']
        
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if not self.is_trained or self.model is None:
            return {}
        
        if hasattr(self.model, 'coef_'):
            coefficients = self.model.coef_
            if coefficients.ndim > 1:
                # For multi-output models, use mean of coefficients
                coefficients = np.mean(np.abs(coefficients), axis=0)
            else:
                coefficients = np.abs(coefficients)
            
            feature_importance = {}
            for i, feature_name in enumerate(self.feature_names):
                if i < len(coefficients):
                    feature_importance[feature_name] = float(coefficients[i])
            
            return feature_importance
        
        return {}