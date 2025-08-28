"""Tree-based ensemble models for trajectory prediction using Random Forest and XGBoost."""

from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SelectKBest, f_regression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    RandomForestRegressor = None
    GridSearchCV = None
    cross_val_score = None
    StandardScaler = None
    SelectKBest = None
    f_regression = None

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

from .base import BaseTrajectoryPredictor, PredictionResult
from ..core.models import Trajectory, TrajectoryPoint
from ..core.config import ModelConfig

logger = logging.getLogger(__name__)


class TreeEnsemblePredictor(BaseTrajectoryPredictor):
    """
    Tree-based ensemble model for trajectory prediction using Random Forest and XGBoost.
    
    This model uses ensemble methods to predict vehicle trajectories with feature importance
    analysis and robust performance across different trajectory patterns.
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """Initialize the Tree Ensemble predictor."""
        super().__init__(config)
        
        # Model-specific configuration
        self.model_type = getattr(self.config, 'model_type', 'random_forest')  # 'random_forest' or 'xgboost'
        self.n_estimators = getattr(self.config, 'n_estimators', 100)
        self.max_depth = getattr(self.config, 'max_depth', 10)
        self.min_samples_split = getattr(self.config, 'min_samples_split', 2)
        self.min_samples_leaf = getattr(self.config, 'min_samples_leaf', 1)
        self.random_state = getattr(self.config, 'random_state', 42)
        
        # XGBoost specific parameters
        self.learning_rate = getattr(self.config, 'learning_rate', 0.1)
        self.subsample = getattr(self.config, 'subsample', 0.8)
        self.colsample_bytree = getattr(self.config, 'colsample_bytree', 0.8)
        self.reg_alpha = getattr(self.config, 'reg_alpha', 0.0)
        self.reg_lambda = getattr(self.config, 'reg_lambda', 1.0)
        
        # Feature engineering parameters
        self.use_feature_selection = getattr(self.config, 'use_feature_selection', True)
        self.n_features = getattr(self.config, 'n_features', 50)
        self.use_scaling = getattr(self.config, 'use_scaling', True)
        
        # Model components
        self.models = {}  # Dictionary to store models for each output
        self.feature_names = None
        self.scaler = None
        self.feature_selector = None
        self.feature_importance = {}
        
        # Validate model type
        if self.model_type not in ['random_forest', 'xgboost']:
            raise ValueError("model_type must be 'random_forest' or 'xgboost'")
        
        # Validate availability
        if self.model_type == 'random_forest' and not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for Random Forest")
        
        if self.model_type == 'xgboost' and not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is required for XGBoost model")
    
    def _extract_trajectory_features(self, trajectory: Trajectory) -> Tuple[np.ndarray, List[str]]:
        """
        Extract comprehensive features from trajectory for tree-based models.
        
        Args:
            trajectory: Input trajectory
            
        Returns:
            Feature array and feature names
        """
        points = trajectory.points
        
        if len(points) < 3:
            raise ValueError("Trajectory must have at least 3 points for feature extraction")
        
        features = []
        feature_names = []
        
        # Basic position features
        x_positions = [p.x for p in points]
        y_positions = [p.y for p in points]
        
        # Time features
        start_time = points[0].timestamp
        times = [(p.timestamp - start_time).total_seconds() for p in points]
        
        # Velocity features (if available or calculated)
        velocities = []
        if hasattr(points[0], 'velocity') and points[0].velocity is not None:
            velocities = [p.velocity for p in points]
        else:
            # Calculate velocities from positions
            for i in range(1, len(points)):
                dt = (points[i].timestamp - points[i-1].timestamp).total_seconds()
                if dt > 0:
                    dx = points[i].x - points[i-1].x
                    dy = points[i].y - points[i-1].y
                    velocity = np.sqrt(dx**2 + dy**2) / dt
                    velocities.append(velocity)
                else:
                    velocities.append(0.0)
            velocities.insert(0, velocities[0] if velocities else 0.0)
        
        # Acceleration features (if available or calculated)
        accelerations = []
        if hasattr(points[0], 'acceleration') and points[0].acceleration is not None:
            accelerations = [p.acceleration for p in points]
        else:
            # Calculate accelerations from velocities
            for i in range(1, len(velocities)):
                dt = (points[i].timestamp - points[i-1].timestamp).total_seconds()
                if dt > 0:
                    acceleration = (velocities[i] - velocities[i-1]) / dt
                    accelerations.append(acceleration)
                else:
                    accelerations.append(0.0)
            accelerations.insert(0, accelerations[0] if accelerations else 0.0)
        
        # Heading features (if available or calculated)
        headings = []
        if hasattr(points[0], 'heading') and points[0].heading is not None:
            headings = [p.heading for p in points]
        else:
            # Calculate headings from positions
            for i in range(1, len(points)):
                dx = points[i].x - points[i-1].x
                dy = points[i].y - points[i-1].y
                heading = np.arctan2(dy, dx)
                headings.append(heading)
            headings.insert(0, headings[0] if headings else 0.0)
        
        # Statistical features
        features.extend([
            np.mean(x_positions), np.std(x_positions), np.min(x_positions), np.max(x_positions),
            np.mean(y_positions), np.std(y_positions), np.min(y_positions), np.max(y_positions),
            np.mean(velocities), np.std(velocities), np.min(velocities), np.max(velocities),
            np.mean(accelerations), np.std(accelerations), np.min(accelerations), np.max(accelerations),
            np.mean(headings), np.std(headings), np.min(headings), np.max(headings),
            np.mean(times), np.std(times), np.max(times)
        ])
        feature_names.extend([
            'mean_x', 'std_x', 'min_x', 'max_x',
            'mean_y', 'std_y', 'min_y', 'max_y',
            'mean_velocity', 'std_velocity', 'min_velocity', 'max_velocity',
            'mean_acceleration', 'std_acceleration', 'min_acceleration', 'max_acceleration',
            'mean_heading', 'std_heading', 'min_heading', 'max_heading',
            'mean_time', 'std_time', 'max_time'
        ])
        
        # Trajectory shape features
        total_distance = sum(np.sqrt((x_positions[i] - x_positions[i-1])**2 + 
                                   (y_positions[i] - y_positions[i-1])**2) 
                           for i in range(1, len(x_positions)))
        straight_line_distance = np.sqrt((x_positions[-1] - x_positions[0])**2 + 
                                       (y_positions[-1] - y_positions[0])**2)
        tortuosity = total_distance / straight_line_distance if straight_line_distance > 0 else 1.0
        
        features.extend([total_distance, straight_line_distance, tortuosity])
        feature_names.extend(['total_distance', 'straight_line_distance', 'tortuosity'])
        
        # Recent trajectory features (last few points)
        recent_points = min(5, len(points))
        for i in range(recent_points):
            idx = -(i + 1)
            features.extend([x_positions[idx], y_positions[idx], velocities[idx], 
                           accelerations[idx], headings[idx]])
            feature_names.extend([f'recent_x_{i}', f'recent_y_{i}', f'recent_velocity_{i}',
                                f'recent_acceleration_{i}', f'recent_heading_{i}'])
        
        # Velocity and acceleration trends
        if len(velocities) >= 3:
            velocity_trend = np.polyfit(range(len(velocities)), velocities, 1)[0]
            acceleration_trend = np.polyfit(range(len(accelerations)), accelerations, 1)[0]
        else:
            velocity_trend = 0.0
            acceleration_trend = 0.0
        
        features.extend([velocity_trend, acceleration_trend])
        feature_names.extend(['velocity_trend', 'acceleration_trend'])
        
        # Curvature features
        curvatures = []
        for i in range(1, len(points) - 1):
            p1, p2, p3 = points[i-1], points[i], points[i+1]
            # Calculate curvature using three points
            a = np.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)
            b = np.sqrt((p3.x - p2.x)**2 + (p3.y - p2.y)**2)
            c = np.sqrt((p3.x - p1.x)**2 + (p3.y - p1.y)**2)
            
            if a * b * c > 0:
                s = (a + b + c) / 2
                area = np.sqrt(s * (s - a) * (s - b) * (s - c))
                curvature = 4 * area / (a * b * c) if a * b * c > 0 else 0
            else:
                curvature = 0
            curvatures.append(curvature)
        
        if curvatures:
            features.extend([np.mean(curvatures), np.std(curvatures), np.max(curvatures)])
            feature_names.extend(['mean_curvature', 'std_curvature', 'max_curvature'])
        else:
            features.extend([0.0, 0.0, 0.0])
            feature_names.extend(['mean_curvature', 'std_curvature', 'max_curvature'])
        
        return np.array(features), feature_names
    
    def _create_random_forest_model(self) -> RandomForestRegressor:
        """Create Random Forest model with configured parameters."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is not available")
        
        return RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=-1
        )
    
    def _create_xgboost_model(self) -> xgb.XGBRegressor:
        """Create XGBoost model with configured parameters."""
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not available")
        
        return xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            n_jobs=-1
        )
    
    def train(self, trajectories: List[Trajectory]) -> None:
        """
        Train the Tree Ensemble model.
        
        Args:
            trajectories: List of training trajectories
        """
        if not trajectories:
            raise ValueError("No trajectories provided for training")
        
        logger.info(f"Training {self.model_type} model with {len(trajectories)} trajectories")
        
        # Extract features and targets
        X_list = []
        y_x_list = []
        y_y_list = []
        
        for trajectory in trajectories:
            try:
                # Extract features from input trajectory
                features, feature_names = self._extract_trajectory_features(trajectory)
                X_list.append(features)
                
                # Extract target (future positions)
                future_points = trajectory.points[-self.prediction_horizon:]
                future_x = [p.x for p in future_points]
                future_y = [p.y for p in future_points]
                y_x_list.extend(future_x)
                y_y_list.extend(future_y)
                
            except Exception as e:
                logger.warning(f"Skipping trajectory due to error: {e}")
                continue
        
        if not X_list:
            raise ValueError("No valid trajectories for training")
        
        # Convert to numpy arrays
        X = np.array(X_list)
        y_x = np.array(y_x_list)
        y_y = np.array(y_y_list)
        self.feature_names = feature_names
        
        logger.info(f"Training data shape: X={X.shape}, y_x={y_x.shape}, y_y={y_y.shape}")
        
        # Feature scaling
        if self.use_scaling:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        
        # Feature selection
        if self.use_feature_selection and X.shape[1] > self.n_features:
            self.feature_selector = SelectKBest(score_func=f_regression, k=self.n_features)
            X = self.feature_selector.fit_transform(X, y_x)  # Use y_x for feature selection
        
        # Train separate models for x and y coordinates
        for coord, y in [('x', y_x), ('y', y_y)]:
            if self.model_type == 'random_forest':
                model = self._create_random_forest_model()
            else:  # xgboost
                model = self._create_xgboost_model()
            
            logger.info(f"Training {coord}-coordinate model...")
            model.fit(X, y)
            self.models[coord] = model
            
            # Store feature importance
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[coord] = model.feature_importances_
        
        self.is_trained = True
        logger.info(f"{self.model_type} model training completed")
    
    def predict(self, trajectory: Trajectory) -> PredictionResult:
        """
        Predict future trajectory using Tree Ensemble regression.
        
        Args:
            trajectory: Input trajectory for prediction
            
        Returns:
            PredictionResult with predicted trajectory
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        # Extract features from input trajectory
        features, _ = self._extract_trajectory_features(trajectory)
        X_test = features.reshape(1, -1)
        
        # Apply scaling if used during training
        if self.use_scaling and self.scaler is not None:
            X_test = self.scaler.transform(X_test)
        
        # Apply feature selection if used during training
        if self.use_feature_selection and self.feature_selector is not None:
            X_test = self.feature_selector.transform(X_test)
        
        # Make predictions
        x_predictions = self.models['x'].predict(X_test).flatten()
        y_predictions = self.models['y'].predict(X_test).flatten()
        
        # Generate timestamps
        last_timestamp = trajectory.points[-1].timestamp
        timestamps = [
            last_timestamp + timedelta(seconds=i * self.time_step)
            for i in range(1, self.prediction_horizon + 1)
        ]
        
        # Create predicted trajectory points
        predicted_points = []
        for i in range(self.prediction_horizon):
            point = TrajectoryPoint(
                x=x_predictions[i],
                y=y_predictions[i],
                timestamp=timestamps[i],
                velocity=None,  # Could be calculated if needed
                acceleration=None,
                heading=None
            )
            predicted_points.append(point)
        
        return PredictionResult(
            predicted_points=predicted_points,
            timestamps=timestamps,
            x_positions=x_predictions,
            y_positions=y_predictions,
            velocities=None,
            accelerations=None,
            headings=None,
            confidence_scores=None,
            uncertainty=None
        )
    
    def get_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """Get feature importance for each coordinate."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before getting feature importance")
        
        importance_dict = {}
        for coord, importance in self.feature_importance.items():
            # Get feature names (considering feature selection)
            if self.use_feature_selection and self.feature_selector is not None:
                selected_features = self.feature_selector.get_support()
                feature_names = [self.feature_names[i] for i in range(len(self.feature_names)) if selected_features[i]]
            else:
                feature_names = self.feature_names
            
            # Create feature importance dictionary
            importance_dict[coord] = {
                feature_names[i]: float(importance[i])
                for i in range(min(len(feature_names), len(importance)))
            }
        
        return importance_dict
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained model."""
        info = super().get_model_info()
        info.update({
            'model_type': f'Tree Ensemble ({self.model_type})',
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'learning_rate': self.learning_rate if self.model_type == 'xgboost' else None,
            'use_feature_selection': self.use_feature_selection,
            'n_features': self.n_features if self.use_feature_selection else len(self.feature_names),
            'use_scaling': self.use_scaling,
            'feature_names': self.feature_names,
            'feature_importance_available': bool(self.feature_importance)
        })
        return info
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model to disk."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before saving")
        
        import joblib
        
        model_data = {
            'config': self.config,
            'models': self.models,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'model_type': self.model_type,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model from disk."""
        import joblib
        
        model_data = joblib.load(filepath)
        
        # Restore model components
        self.config = model_data['config']
        self.models = model_data['models']
        self.feature_names = model_data['feature_names']
        self.feature_importance = model_data['feature_importance']
        self.scaler = model_data['scaler']
        self.feature_selector = model_data['feature_selector']
        self.model_type = model_data['model_type']
        self.is_trained = model_data['is_trained']
        
        logger.info(f"Model loaded from {filepath}")