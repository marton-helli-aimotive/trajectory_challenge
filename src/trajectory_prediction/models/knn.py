"""
K-Nearest Neighbors trajectory prediction model with similarity matching.

Uses trajectory shape similarity to find similar historical patterns
and predicts based on how those patterns continued.
"""

import asyncio
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import dtw
from scipy.interpolate import interp1d
from omegaconf import DictConfig

from ..data.validation.schemas import TrajectoryData
from .base import TrajectoryPredictor, PredictionResult


class KNNTrajectoryPredictor(TrajectoryPredictor):
    """
    K-Nearest Neighbors trajectory prediction model with similarity matching.
    
    Features:
    - Trajectory shape similarity using DTW (Dynamic Time Warping)
    - Multiple distance metrics (Euclidean, cosine, DTW)
    - Weighted prediction based on similarity scores
    - Uncertainty estimation from prediction variance
    - Context-aware matching (velocity, acceleration patterns)
    """
    
    def __init__(self, config: DictConfig, name: Optional[str] = None):
        super().__init__(config, name or "KNN")
        
        # Model parameters
        self.n_neighbors = config.get("n_neighbors", 5)
        self.min_history_length = config.get("min_history_length", 5)
        self.trajectory_length = config.get("trajectory_length", 10)  # Length for comparison
        self.similarity_metric = config.get("similarity_metric", "dtw")  # dtw, euclidean, cosine
        self.weight_function = config.get("weight_function", "distance")  # distance, uniform
        self.context_features = config.get("context_features", True)
        
        # Distance thresholds
        self.max_distance_threshold = config.get("max_distance_threshold", 1000.0)
        self.velocity_weight = config.get("velocity_weight", 0.3)
        self.acceleration_weight = config.get("acceleration_weight", 0.2)
        
        # Training data storage
        self.training_trajectories = []
        self.training_features = None
        self.feature_scaler = StandardScaler()
        self.knn_model = None
        
        # Prediction database
        self.trajectory_database = {}  # vehicle_id -> list of trajectory segments
        
    async def fit(
        self, 
        trajectories: List[TrajectoryData],
        validation_trajectories: Optional[List[TrajectoryData]] = None
    ) -> Dict[str, Any]:
        """
        Train KNN model by building trajectory database and feature representations.
        
        Args:
            trajectories: Training trajectory data
            validation_trajectories: Optional validation data
            
        Returns:
            Training metrics and database statistics
        """
        # Build trajectory database
        await self._build_trajectory_database(trajectories)
        
        # Extract features for all training trajectory segments
        all_features = []
        all_continuations = []
        
        for vehicle_id, segments in self.trajectory_database.items():
            for segment_data in segments:
                history = segment_data["history"]
                continuation = segment_data["continuation"]
                
                # Extract features for this trajectory segment
                features = self._extract_trajectory_features(history)
                if features is not None:
                    all_features.append(features)
                    all_continuations.append(continuation)
        
        if not all_features:
            raise ValueError("No valid trajectory features extracted from training data")
        
        # Convert to numpy arrays
        self.training_features = np.array(all_features)
        self.training_continuations = all_continuations
        
        # Fit feature scaler
        self.feature_scaler.fit(self.training_features)
        scaled_features = self.feature_scaler.transform(self.training_features)
        
        # Build KNN index
        self.knn_model = NearestNeighbors(
            n_neighbors=min(self.n_neighbors, len(scaled_features)),
            metric='euclidean' if self.similarity_metric != 'dtw' else 'euclidean',
            algorithm='auto'
        )
        self.knn_model.fit(scaled_features)
        
        self.is_trained = True
        
        # Calculate training statistics
        avg_trajectory_length = np.mean([
            len(segments) for segments in self.trajectory_database.values()
        ])
        
        return {
            "model": self.name,
            "training_trajectories": len(trajectories),
            "trajectory_database_size": len(self.trajectory_database),
            "trajectory_segments": len(all_features),
            "feature_dimension": self.training_features.shape[1],
            "avg_trajectory_length": float(avg_trajectory_length),
            "n_neighbors": self.n_neighbors,
            "similarity_metric": self.similarity_metric
        }
    
    async def predict(
        self,
        history: List[TrajectoryData],
        prediction_horizon: float,
        **kwargs
    ) -> PredictionResult:
        """
        Predict future trajectory using K-nearest neighbors approach.
        
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
        """Internal prediction method using KNN similarity matching."""
        
        # Extract features from query trajectory
        query_features = self._extract_trajectory_features(history)
        if query_features is None:
            raise ValueError("Cannot extract features from history")
        
        # Scale features
        query_features_scaled = self.feature_scaler.transform([query_features])
        
        # Find k nearest neighbors
        if self.similarity_metric == "dtw":
            # Use DTW for trajectory shape matching
            neighbors_indices, similarities = await self._find_dtw_neighbors(history)
        else:
            # Use standard KNN
            distances, neighbors_indices = self.knn_model.kneighbors(query_features_scaled)
            distances = distances[0]
            neighbors_indices = neighbors_indices[0]
            similarities = 1.0 / (1.0 + distances)  # Convert distance to similarity
        
        # Get neighbor continuations
        neighbor_predictions = []
        neighbor_weights = []
        
        for i, neighbor_idx in enumerate(neighbors_indices):
            if neighbor_idx < len(self.training_continuations):
                continuation = self.training_continuations[neighbor_idx]
                weight = similarities[i] if hasattr(similarities, '__iter__') else similarities
                
                # Convert continuation to prediction
                pred = self._continuation_to_prediction(
                    continuation, history, prediction_horizon, time_step_size
                )
                
                if pred is not None:
                    neighbor_predictions.append(pred)
                    neighbor_weights.append(weight)
        
        if not neighbor_predictions:
            # Fallback: use last known position
            last_point = history[-1]
            return self._create_fallback_prediction(last_point, prediction_horizon, time_step_size)
        
        # Combine neighbor predictions
        combined_prediction = self._combine_neighbor_predictions(
            neighbor_predictions, neighbor_weights
        )
        
        return combined_prediction
    
    async def _build_trajectory_database(self, trajectories: List[TrajectoryData]):
        """Build database of trajectory segments for similarity matching."""
        # Group by vehicle
        vehicle_trajectories = self._group_by_vehicle(trajectories)
        
        for vehicle_id, traj_points in vehicle_trajectories.items():
            if len(traj_points) < self.min_history_length * 2:  # Need history + continuation
                continue
            
            # Sort by timestamp
            traj_points = sorted(traj_points, key=lambda x: x.timestamp)
            
            # Create overlapping segments
            segments = []
            for i in range(len(traj_points) - self.min_history_length * 2):
                history_end = i + self.min_history_length
                continuation_end = min(
                    history_end + self.trajectory_length,
                    len(traj_points)
                )
                
                if continuation_end > history_end:
                    history = traj_points[i:history_end]
                    continuation = traj_points[history_end:continuation_end]
                    
                    segments.append({
                        "history": history,
                        "continuation": continuation,
                        "start_idx": i,
                        "vehicle_id": vehicle_id
                    })
            
            if segments:
                self.trajectory_database[vehicle_id] = segments
    
    def _extract_trajectory_features(self, trajectory: List[TrajectoryData]) -> Optional[np.ndarray]:
        """Extract features from trajectory for similarity matching."""
        if len(trajectory) < 2:
            return None
        
        features = []
        
        # Basic statistics
        positions_x = np.array([p.x for p in trajectory])
        positions_y = np.array([p.y for p in trajectory])
        
        # Position features
        features.extend([
            np.mean(positions_x), np.std(positions_x),
            np.mean(positions_y), np.std(positions_y),
            positions_x[-1] - positions_x[0],  # Total displacement X
            positions_y[-1] - positions_y[0],  # Total displacement Y
        ])
        
        # Calculate velocities
        velocities = []
        headings = []
        
        for i in range(1, len(trajectory)):
            dt = (trajectory[i].timestamp - trajectory[i-1].timestamp).total_seconds()
            if dt > 0:
                dx = trajectory[i].x - trajectory[i-1].x
                dy = trajectory[i].y - trajectory[i-1].y
                
                velocity = np.sqrt(dx**2 + dy**2) / dt
                heading = np.arctan2(dy, dx)
                
                velocities.append(velocity)
                headings.append(heading)
        
        # Velocity features
        if velocities:
            features.extend([
                np.mean(velocities), np.std(velocities),
                np.max(velocities), np.min(velocities)
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        # Heading features  
        if headings:
            # Handle circular nature of headings
            mean_heading = np.arctan2(np.mean(np.sin(headings)), np.mean(np.cos(headings)))
            heading_changes = np.diff(headings)
            # Wrap heading changes to [-π, π]
            heading_changes = np.arctan2(np.sin(heading_changes), np.cos(heading_changes))
            
            features.extend([
                mean_heading,
                np.std(heading_changes),
                np.sum(np.abs(heading_changes))  # Total heading change
            ])
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # Acceleration features (if enough points)
        if len(velocities) >= 2:
            accelerations = []
            for i in range(1, len(velocities)):
                dt = (trajectory[i+1].timestamp - trajectory[i].timestamp).total_seconds()
                if dt > 0:
                    acc = (velocities[i] - velocities[i-1]) / dt
                    accelerations.append(acc)
            
            if accelerations:
                features.extend([
                    np.mean(accelerations), np.std(accelerations),
                    np.max(accelerations), np.min(accelerations)
                ])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        # Trajectory shape features (normalized positions)
        if len(trajectory) >= 3:
            # Normalize positions relative to start and end
            start_x, start_y = positions_x[0], positions_y[0] 
            end_x, end_y = positions_x[-1], positions_y[-1]
            
            # Rotation to align with displacement vector
            displacement_length = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
            
            if displacement_length > 1e-6:
                cos_theta = (end_x - start_x) / displacement_length
                sin_theta = (end_y - start_y) / displacement_length
                
                # Rotate and normalize positions
                normalized_positions = []
                for x, y in zip(positions_x, positions_y):
                    # Translate to origin
                    x_t = x - start_x
                    y_t = y - start_y
                    
                    # Rotate to align with X-axis
                    x_r = x_t * cos_theta + y_t * sin_theta
                    y_r = -x_t * sin_theta + y_t * cos_theta
                    
                    # Normalize by displacement length
                    x_n = x_r / displacement_length
                    y_n = y_r / displacement_length
                    
                    normalized_positions.extend([x_n, y_n])
                
                # Take fixed number of shape features
                max_shape_features = 10
                if len(normalized_positions) > max_shape_features:
                    # Sample evenly
                    indices = np.linspace(0, len(normalized_positions)-1, max_shape_features, dtype=int)
                    normalized_positions = [normalized_positions[i] for i in indices]
                
                features.extend(normalized_positions[:max_shape_features])
                
                # Pad if necessary
                while len(features) < len(features) + (max_shape_features - len(normalized_positions)):
                    features.append(0.0)
        
        return np.array(features)
    
    async def _find_dtw_neighbors(self, query_history: List[TrajectoryData]) -> Tuple[np.ndarray, np.ndarray]:
        """Find neighbors using Dynamic Time Warping distance."""
        # Extract position sequences for DTW
        query_positions = np.array([[p.x, p.y] for p in query_history])
        
        dtw_distances = []
        
        for i, continuation_data in enumerate(self.training_continuations):
            training_history = continuation_data if isinstance(continuation_data, list) else []
            
            if len(training_history) > 0:
                train_positions = np.array([[p.x, p.y] for p in training_history[:len(query_history)]])
                
                # Calculate DTW distance
                try:
                    distance = dtw(query_positions, train_positions, dist=np.linalg.norm)[0]
                    dtw_distances.append((i, distance))
                except:
                    # Fallback to Euclidean if DTW fails
                    distance = np.linalg.norm(query_positions[-1] - train_positions[-1])
                    dtw_distances.append((i, distance))
        
        # Sort by distance and take k nearest
        dtw_distances.sort(key=lambda x: x[1])
        k = min(self.n_neighbors, len(dtw_distances))
        
        indices = np.array([dtw_distances[i][0] for i in range(k)])
        distances = np.array([dtw_distances[i][1] for i in range(k)])
        similarities = 1.0 / (1.0 + distances)
        
        return indices, similarities
    
    def _continuation_to_prediction(
        self,
        continuation: List[TrajectoryData],
        query_history: List[TrajectoryData],
        prediction_horizon: float,
        time_step_size: float
    ) -> Optional[Dict[str, np.ndarray]]:
        """Convert trajectory continuation to prediction format."""
        if not continuation:
            return None
        
        # Get time information
        last_query_point = query_history[-1]
        base_timestamp = last_query_point.timestamp.timestamp()
        
        # Extract positions from continuation
        continuation_times = np.array([
            (p.timestamp.timestamp() - continuation[0].timestamp.timestamp()) 
            for p in continuation
        ])
        continuation_x = np.array([p.x for p in continuation])
        continuation_y = np.array([p.y for p in continuation])
        
        # Interpolate to match requested prediction horizon
        num_steps = max(1, int(prediction_horizon / time_step_size))
        target_times = np.linspace(0, prediction_horizon, num_steps)
        
        if len(continuation_times) > 1 and continuation_times[-1] > 0:
            # Interpolate positions
            try:
                interp_x = interp1d(continuation_times, continuation_x, 
                                  kind='linear', fill_value='extrapolate')
                interp_y = interp1d(continuation_times, continuation_y, 
                                  kind='linear', fill_value='extrapolate')
                
                pred_x = interp_x(target_times)
                pred_y = interp_y(target_times)
                
                # Adjust to start from query position
                offset_x = last_query_point.x - pred_x[0]
                offset_y = last_query_point.y - pred_y[0]
                
                pred_x += offset_x
                pred_y += offset_y
                
                return {
                    "x": pred_x,
                    "y": pred_y,
                    "times": target_times
                }
                
            except Exception:
                # Fallback: linear extrapolation
                if len(continuation) >= 2:
                    dx = continuation[-1].x - continuation[0].x
                    dy = continuation[-1].y - continuation[0].y
                    dt = continuation_times[-1] - continuation_times[0]
                    
                    if dt > 0:
                        vx = dx / dt
                        vy = dy / dt
                        
                        pred_x = last_query_point.x + vx * target_times
                        pred_y = last_query_point.y + vy * target_times
                        
                        return {
                            "x": pred_x,
                            "y": pred_y,
                            "times": target_times
                        }
        
        return None
    
    def _combine_neighbor_predictions(
        self,
        predictions: List[Dict[str, np.ndarray]],
        weights: List[float]
    ) -> PredictionResult:
        """Combine predictions from multiple neighbors."""
        if not predictions:
            raise ValueError("No predictions to combine")
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(predictions)] * len(predictions)
        
        # Weighted average of predictions
        weighted_x = np.zeros_like(predictions[0]["x"])
        weighted_y = np.zeros_like(predictions[0]["y"])
        
        for pred, weight in zip(predictions, weights):
            weighted_x += weight * pred["x"]
            weighted_y += weight * pred["y"]
        
        # Calculate uncertainty from prediction variance
        x_variance = np.var([pred["x"] for pred in predictions], axis=0)
        y_variance = np.var([pred["y"] for pred in predictions], axis=0)
        
        x_std = np.sqrt(x_variance)
        y_std = np.sqrt(y_variance)
        
        # Calculate velocities and accelerations
        times = predictions[0]["times"]
        velocities = np.zeros_like(weighted_x)
        accelerations = np.zeros_like(weighted_x)
        
        for i in range(1, len(weighted_x)):
            dt = times[i] - times[i-1]
            if dt > 0:
                dx = weighted_x[i] - weighted_x[i-1]
                dy = weighted_y[i] - weighted_y[i-1]
                velocities[i] = np.sqrt(dx**2 + dy**2) / dt
        
        for i in range(1, len(velocities)):
            dt = times[i] - times[i-1]
            if dt > 0:
                accelerations[i] = (velocities[i] - velocities[i-1]) / dt
        
        # Generate timestamps
        base_time = predictions[0]["times"][0] if "base_time" in predictions[0] else 0
        timestamps = np.array([(base_time + t) * 1000 for t in times])
        
        return PredictionResult(
            timestamps=timestamps,
            x_coords=weighted_x,
            y_coords=weighted_y,
            x_std=x_std,
            y_std=y_std,
            velocities=velocities,
            accelerations=accelerations,
            prediction_horizon=times[-1],
            model_name=self.name,
            metadata={
                "n_neighbors_used": len(predictions),
                "similarity_metric": self.similarity_metric,
                "weights": weights,
                "prediction_variance": {
                    "x_var": float(np.mean(x_variance)),
                    "y_var": float(np.mean(y_variance))
                }
            }
        )
    
    def _create_fallback_prediction(
        self,
        last_point: TrajectoryData,
        prediction_horizon: float,
        time_step_size: float
    ) -> PredictionResult:
        """Create fallback prediction when no neighbors found."""
        num_steps = max(1, int(prediction_horizon / time_step_size))
        times = np.linspace(time_step_size, prediction_horizon, num_steps)
        
        # Simple constant position prediction
        pred_x = np.full(num_steps, last_point.x)
        pred_y = np.full(num_steps, last_point.y)
        
        # High uncertainty for fallback
        x_std = np.full(num_steps, 10.0)
        y_std = np.full(num_steps, 10.0)
        
        timestamps = np.array([(last_point.timestamp.timestamp() + t) * 1000 for t in times])
        
        return PredictionResult(
            timestamps=timestamps,
            x_coords=pred_x,
            y_coords=pred_y,
            x_std=x_std,
            y_std=y_std,
            velocities=np.zeros(num_steps),
            accelerations=np.zeros(num_steps),
            prediction_horizon=prediction_horizon,
            model_name=self.name,
            metadata={"fallback": True, "reason": "no_neighbors_found"}
        )
    
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
            "type": "k_nearest_neighbors",
            "description": "Trajectory prediction using similarity-based neighbor matching",
            "parameters": {
                "n_neighbors": self.n_neighbors,
                "min_history_length": self.min_history_length,
                "trajectory_length": self.trajectory_length,
                "similarity_metric": self.similarity_metric,
                "weight_function": self.weight_function,
                "context_features": self.context_features
            },
            "capabilities": {
                "uncertainty_estimation": True,
                "similarity_matching": True,
                "trajectory_shape_analysis": True,
                "online_learning": False,
                "batch_prediction": True
            },
            "database_stats": {
                "vehicles_in_database": len(self.trajectory_database),
                "total_segments": sum(len(segments) for segments in self.trajectory_database.values()),
                "feature_dimension": self.training_features.shape[1] if self.training_features is not None else 0
            },
            "is_trained": self.is_trained
        }