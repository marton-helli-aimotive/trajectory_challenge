"""K-Nearest Neighbors model with Dynamic Time Warping for trajectory prediction."""

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
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    # Create mock sklearn classes
    class NearestNeighbors:
        def __init__(self, n_neighbors=5, metric='euclidean'):
            self.n_neighbors = n_neighbors
            self.metric = metric
        
        def fit(self, X):
            pass
        
        def kneighbors(self, X):
            return [[1.0] * self.n_neighbors], [[0] * self.n_neighbors]
    
    class StandardScaler:
        def fit_transform(self, X):
            return X
        
        def transform(self, X):
            return X

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


def dtw_distance(traj1: np.ndarray, traj2: np.ndarray) -> float:
    """Calculate Dynamic Time Warping distance between two trajectories.
    
    Args:
        traj1: First trajectory as numpy array of shape (n_points, n_features)
        traj2: Second trajectory as numpy array of shape (m_points, n_features)
    
    Returns:
        DTW distance between the trajectories
    """
    n, m = len(traj1), len(traj2)
    
    # Initialize DTW matrix
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0
    
    # Fill DTW matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.linalg.norm(traj1[i-1] - traj2[j-1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],    # insertion
                dtw_matrix[i, j-1],    # deletion
                dtw_matrix[i-1, j-1]   # match
            )
    
    return dtw_matrix[n, m]


class KNearestNeighborsPredictor(BaseTrajectoryPredictor):
    """K-Nearest Neighbors predictor with Dynamic Time Warping.
    
    This model finds the most similar historical trajectories using DTW
    and predicts future trajectory by aggregating their continuations.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__(config, "KNearestNeighbors")
        
        # Extract KNN-specific configuration
        knn_config = getattr(config, 'knn_config', {})
        self.n_neighbors = knn_config.get('n_neighbors', 5)
        self.weights = knn_config.get('weights', 'uniform')  # 'uniform' or 'distance'
        self.use_dtw = knn_config.get('use_dtw', True)
        self.feature_window = knn_config.get('feature_window', 10)
        self.distance_metric = knn_config.get('distance_metric', 'euclidean')
        self.aggregation_method = knn_config.get('aggregation_method', 'weighted_mean')
        
        # Model components
        self.training_trajectories = []
        self.training_features = []
        self.scaler = StandardScaler()
        self.knn_model = None
        
        # Feature extraction parameters
        self.feature_names = ['x', 'y', 'velocity', 'acceleration', 'heading']
        
        logger.info(f"Initialized {self.model_name} with k={self.n_neighbors}, weights={self.weights}")
    
    def _extract_trajectory_features(self, trajectory: Trajectory) -> np.ndarray:
        """Extract features from trajectory for similarity comparison."""
        if trajectory.length < 2:
            raise ValueError("Trajectory must have at least 2 points")
        
        # Use the last few points for feature extraction
        window_size = min(self.feature_window, trajectory.length)
        recent_points = trajectory.points[-window_size:]
        
        features = []
        for point in recent_points:
            point_features = [
                point.x,
                point.y,
                point.velocity,
                point.acceleration,
                point.heading
            ]
            features.append(point_features)
        
        # Pad to consistent length if needed
        while len(features) < self.feature_window:
            # Repeat the last point's features
            features.append(features[-1])
        
        return np.array(features)
    
    def _calculate_trajectory_similarity(
        self, 
        query_features: np.ndarray, 
        training_features: np.ndarray
    ) -> float:
        """Calculate similarity between query and training trajectory."""
        if self.use_dtw:
            return dtw_distance(query_features, training_features)
        else:
            # Use Euclidean distance on flattened features
            return np.linalg.norm(query_features.flatten() - training_features.flatten())
    
    def _find_nearest_neighbors(
        self, 
        query_features: np.ndarray
    ) -> Tuple[List[int], List[float]]:
        """Find k nearest neighbors for the query trajectory."""
        if self.use_dtw:
            # Use DTW distance for similarity
            distances = []
            for i, training_features in enumerate(self.training_features):
                distance = self._calculate_trajectory_similarity(query_features, training_features)
                distances.append((distance, i))
            
            # Sort by distance and get top k
            distances.sort()
            neighbor_indices = [idx for _, idx in distances[:self.n_neighbors]]
            neighbor_distances = [dist for dist, _ in distances[:self.n_neighbors]]
            
        else:
            # Use sklearn's NearestNeighbors
            if self.knn_model is None:
                raise RuntimeError("KNN model not trained")
            
            # Reshape features for sklearn
            query_flat = query_features.flatten().reshape(1, -1)
            query_scaled = self.scaler.transform(query_flat)
            
            distances, indices = self.knn_model.kneighbors(query_scaled)
            neighbor_indices = indices[0].tolist()
            neighbor_distances = distances[0].tolist()
        
        return neighbor_indices, neighbor_distances
    
    def _get_trajectory_continuation(
        self, 
        trajectory: Trajectory, 
        horizon: int, 
        frequency: float
    ) -> np.ndarray:
        """Get the continuation of a trajectory for the prediction horizon."""
        # Generate future timestamps
        timestamps = self._prepare_prediction_timestamps(trajectory, horizon, frequency)
        
        # Get actual trajectory points at these timestamps
        continuations_x = []
        continuations_y = []
        
        for timestamp in timestamps:
            actual_point = trajectory.get_point_at_time(timestamp)
            if actual_point is not None:
                continuations_x.append(actual_point.x)
                continuations_y.append(actual_point.y)
            else:
                # If no actual point, use extrapolation
                last_point = trajectory.points[-1]
                if len(trajectory.points) >= 2:
                    prev_point = trajectory.points[-2]
                    dt_prev = (last_point.timestamp - prev_point.timestamp).total_seconds()
                    dt = (timestamp - last_point.timestamp).total_seconds()
                    
                    if dt_prev > 0:
                        vx = (last_point.x - prev_point.x) / dt_prev
                        vy = (last_point.y - prev_point.y) / dt_prev
                        continuations_x.append(last_point.x + vx * dt)
                        continuations_y.append(last_point.y + vy * dt)
                    else:
                        continuations_x.append(last_point.x)
                        continuations_y.append(last_point.y)
                else:
                    continuations_x.append(last_point.x)
                    continuations_y.append(last_point.y)
        
        return np.array(continuations_x), np.array(continuations_y)
    
    def _aggregate_predictions(
        self, 
        neighbor_continuations: List[Tuple[np.ndarray, np.ndarray]], 
        neighbor_distances: List[float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Aggregate predictions from nearest neighbors."""
        if not neighbor_continuations:
            raise ValueError("No neighbor continuations provided")
        
        if self.aggregation_method == 'weighted_mean' and self.weights == 'distance':
            # Weighted average based on distance
            weights = np.array(neighbor_distances)
            # Convert distances to weights (closer = higher weight)
            weights = 1.0 / (weights + 1e-8)  # Add small epsilon to avoid division by zero
            weights = weights / np.sum(weights)
            
            # Weighted average of x and y coordinates
            weighted_x = np.zeros_like(neighbor_continuations[0][0])
            weighted_y = np.zeros_like(neighbor_continuations[0][1])
            
            for i, ((x_cont, y_cont), weight) in enumerate(zip(neighbor_continuations, weights)):
                weighted_x += weight * x_cont
                weighted_y += weight * y_cont
            
            return weighted_x, weighted_y
        
        else:
            # Simple mean
            x_continuations = [x for x, _ in neighbor_continuations]
            y_continuations = [y for _, y in neighbor_continuations]
            
            return np.mean(x_continuations, axis=0), np.mean(y_continuations, axis=0)
    
    def train(self, trajectories: List[Trajectory]) -> Dict[str, Any]:
        """Train the KNN model."""
        logger.info(f"Training {self.model_name} model with {len(trajectories)} trajectories")
        
        if len(trajectories) < self.n_neighbors:
            raise ValueError(f"Need at least {self.n_neighbors} trajectories for training")
        
        # Extract features from all training trajectories
        self.training_trajectories = []
        self.training_features = []
        
        for trajectory in trajectories:
            try:
                features = self._extract_trajectory_features(trajectory)
                self.training_trajectories.append(trajectory)
                self.training_features.append(features)
            except Exception as e:
                logger.warning(f"Skipping trajectory {trajectory.vehicle_id}: {e}")
                continue
        
        if len(self.training_trajectories) < self.n_neighbors:
            raise ValueError(f"Not enough valid trajectories for training (need at least {self.n_neighbors})")
        
        # Prepare features for sklearn if not using DTW
        if not self.use_dtw:
            # Flatten features for sklearn
            flat_features = []
            for features in self.training_features:
                flat_features.append(features.flatten())
            
            X_train = np.vstack(flat_features)
            X_scaled = self.scaler.fit_transform(X_train)
            
            # Train sklearn's NearestNeighbors
            self.knn_model = NearestNeighbors(
                n_neighbors=self.n_neighbors,
                metric=self.distance_metric
            )
            self.knn_model.fit(X_scaled)
        
        self.is_trained = True
        self.training_data_size = len(self.training_trajectories)
        self.last_training_time = datetime.now()
        
        logger.info(f"Training completed with {self.training_data_size} trajectories")
        
        return {
            "model_name": self.model_name,
            "training_data_size": self.training_data_size,
            "n_neighbors": self.n_neighbors,
            "use_dtw": self.use_dtw,
            "weights": self.weights,
            "aggregation_method": self.aggregation_method,
            "training_time": self.last_training_time
        }
    
    def predict(
        self, 
        trajectory: Trajectory, 
        prediction_horizon: Optional[int] = None,
        prediction_frequency: Optional[float] = None
    ) -> PredictionResult:
        """Predict future trajectory using KNN with DTW."""
        self.validate_input(trajectory)
        
        # Get prediction parameters
        horizon = prediction_horizon or self.config.prediction_horizon
        frequency = prediction_frequency or self.config.prediction_frequency
        
        # Extract features from query trajectory
        query_features = self._extract_trajectory_features(trajectory)
        
        # Find nearest neighbors
        neighbor_indices, neighbor_distances = self._find_nearest_neighbors(query_features)
        
        # Get continuations from nearest neighbors
        neighbor_continuations = []
        for idx in neighbor_indices:
            neighbor_traj = self.training_trajectories[idx]
            x_cont, y_cont = self._get_trajectory_continuation(neighbor_traj, horizon, frequency)
            neighbor_continuations.append((x_cont, y_cont))
        
        # Aggregate predictions
        predicted_x, predicted_y = self._aggregate_predictions(neighbor_continuations, neighbor_distances)
        
        # Generate timestamps
        timestamps = self._prepare_prediction_timestamps(trajectory, horizon, frequency)
        
        # Create predicted points
        predicted_points = []
        velocities = []
        accelerations = []
        headings_list = []
        
        last_point = trajectory.points[-1]
        
        for i, (timestamp, x, y) in enumerate(zip(timestamps, predicted_x, predicted_y)):
            # Calculate velocity and acceleration
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
            x_positions=predicted_x,
            y_positions=predicted_y,
            velocities=np.array(velocities),
            accelerations=np.array(accelerations),
            headings=np.array(headings_list),
            model_name=self.model_name,
            prediction_time=datetime.now(),
            metadata={
                "n_neighbors": self.n_neighbors,
                "neighbor_distances": neighbor_distances,
                "use_dtw": self.use_dtw,
                "weights": self.weights,
                "aggregation_method": self.aggregation_method
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
            'training_trajectories': self.training_trajectories,
            'training_features': self.training_features,
            'scaler': self.scaler,
            'knn_model': self.knn_model,
            'config': {
                'n_neighbors': self.n_neighbors,
                'weights': self.weights,
                'use_dtw': self.use_dtw,
                'feature_window': self.feature_window,
                'distance_metric': self.distance_metric,
                'aggregation_method': self.aggregation_method
            }
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model from disk."""
        model_data = joblib.load(filepath)
        
        self.training_trajectories = model_data['training_trajectories']
        self.training_features = model_data['training_features']
        self.scaler = model_data['scaler']
        self.knn_model = model_data['knn_model']
        
        # Restore configuration
        config = model_data['config']
        self.n_neighbors = config['n_neighbors']
        self.weights = config['weights']
        self.use_dtw = config['use_dtw']
        self.feature_window = config['feature_window']
        self.distance_metric = config['distance_metric']
        self.aggregation_method = config['aggregation_method']
        
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")
    
    def get_neighbor_info(self, trajectory: Trajectory) -> Dict[str, Any]:
        """Get information about the nearest neighbors for a trajectory."""
        if not self.is_trained:
            return {}
        
        query_features = self._extract_trajectory_features(trajectory)
        neighbor_indices, neighbor_distances = self._find_nearest_neighbors(query_features)
        
        neighbor_info = []
        for i, (idx, distance) in enumerate(zip(neighbor_indices, neighbor_distances)):
            neighbor_traj = self.training_trajectories[idx]
            neighbor_info.append({
                'rank': i + 1,
                'vehicle_id': neighbor_traj.vehicle_id,
                'distance': distance,
                'trajectory_length': neighbor_traj.length,
                'duration': neighbor_traj.duration,
                'total_distance': neighbor_traj.total_distance
            })
        
        return {
            'n_neighbors': self.n_neighbors,
            'neighbors': neighbor_info,
            'use_dtw': self.use_dtw,
            'weights': self.weights
        }