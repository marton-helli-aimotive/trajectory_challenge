"""K-Nearest Neighbors model with Dynamic Time Warping for trajectory prediction."""

import time
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler

from ..data.models import Trajectory
from .base import ModelConfig, PredictionResult, TrajectoryPredictor

# mypy: disable-error-code="no-any-return"


def dtw_distance(traj1: np.ndarray, traj2: np.ndarray) -> float:
    """Compute Dynamic Time Warping distance between two trajectories.

    Args:
        traj1: First trajectory as (n_points, 2) array
        traj2: Second trajectory as (m_points, 2) array

    Returns:
        DTW distance between trajectories
    """
    n, m = len(traj1), len(traj2)

    # Initialize DTW matrix
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0

    # Fill DTW matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = euclidean(traj1[i - 1], traj2[j - 1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],  # insertion
                dtw_matrix[i, j - 1],  # deletion
                dtw_matrix[i - 1, j - 1],  # match
            )

    return float(dtw_matrix[n, m])


def normalize_trajectory_length(
    trajectory: np.ndarray, target_length: int
) -> np.ndarray:
    """Normalize trajectory to target length using interpolation.

    Args:
        trajectory: Input trajectory as (n_points, n_features) array
        target_length: Desired number of points

    Returns:
        Normalized trajectory with target_length points
    """
    if len(trajectory) == target_length:
        return trajectory

    # Create normalized time indices
    original_indices = np.linspace(0, 1, len(trajectory))
    target_indices = np.linspace(0, 1, target_length)

    # Interpolate each feature dimension separately
    interpolated_features = []
    for feature_idx in range(trajectory.shape[1]):
        feature_interp = np.interp(
            target_indices, original_indices, trajectory[:, feature_idx]
        )
        interpolated_features.append(feature_interp)

    return np.column_stack(interpolated_features)


class KNearestNeighborsPredictor(TrajectoryPredictor):
    """K-Nearest Neighbors model using Dynamic Time Warping for trajectory similarity."""

    def __init__(self, config: ModelConfig | None = None):
        if config is None:
            config = ModelConfig(
                name="KNearestNeighbors",
                model_type="classical",
                hyperparameters={
                    "n_neighbors": 5,
                    "normalize_length": True,
                    "target_length": 10,
                    "distance_metric": "dtw",
                    "use_velocity": True,
                },
            )
        super().__init__(config)

        # Hyperparameters
        self.n_neighbors = config.hyperparameters.get("n_neighbors", 5)
        self.normalize_length = config.hyperparameters.get("normalize_length", True)
        self.target_length = config.hyperparameters.get("target_length", 10)
        self.distance_metric = config.hyperparameters.get("distance_metric", "dtw")
        self.use_velocity = config.hyperparameters.get("use_velocity", True)

        # Training data storage
        self.training_trajectories: list[Trajectory] = []
        self.training_features: list[np.ndarray] = []
        self.scaler: StandardScaler | None = None

    def _extract_trajectory_features(self, trajectory: Trajectory) -> np.ndarray:
        """Extract feature representation of trajectory for similarity matching."""
        points = trajectory.points
        if len(points) < 2:
            return np.array([])

        # Extract position trajectory
        positions = np.array([[p.x, p.y] for p in points])

        if self.use_velocity:
            # Include velocity information
            velocities = np.array([[p.velocity_x, p.velocity_y] for p in points])
            features = np.hstack([positions, velocities])
        else:
            features = positions

        # Normalize trajectory length if requested
        if self.normalize_length:
            features = normalize_trajectory_length(features, self.target_length)

        return features

    def _compute_similarity(self, traj1: np.ndarray, traj2: np.ndarray) -> float:
        """Compute similarity between two trajectory feature arrays."""
        if len(traj1) == 0 or len(traj2) == 0:
            return np.inf

        if self.distance_metric == "dtw":
            # DTW returns numpy scalar which mypy sees as Any
            distance: Any = dtw_distance(traj1, traj2)
            return float(distance)
        elif self.distance_metric == "euclidean":
            # For euclidean, trajectories must have same length
            if len(traj1) != len(traj2):
                min_len = min(len(traj1), len(traj2))
                traj1 = traj1[:min_len]
                traj2 = traj2[:min_len]

            return float(euclidean(traj1.flatten(), traj2.flatten()))
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

    def fit(
        self, trajectories: list[Trajectory], features: pd.DataFrame | None = None
    ) -> "KNearestNeighborsPredictor":
        """Train the KNN model by storing training trajectories."""
        _ = features  # Not used for this model
        start_time = time.time()

        self.training_trajectories = []
        self.training_features = []

        # Extract features from all training trajectories
        valid_features = []
        for trajectory in trajectories:
            if len(trajectory.points) < 3:  # Need at least 3 points for prediction
                continue

            traj_features = self._extract_trajectory_features(trajectory)
            if len(traj_features) > 0:
                self.training_trajectories.append(trajectory)
                self.training_features.append(traj_features)
                valid_features.append(traj_features.flatten())

        if not self.training_features:
            raise ValueError("No valid training trajectories found")

        # Fit scaler on flattened features for normalization
        if valid_features:
            # Pad features to same length for scaler
            max_len = max(len(f) for f in valid_features)
            padded_features = []
            for f in valid_features:
                if len(f) < max_len:
                    padded = np.pad(f, (0, max_len - len(f)), mode="constant")
                else:
                    padded = f
                padded_features.append(padded)

            self.scaler = StandardScaler()
            self.scaler.fit(padded_features)

        self.is_fitted = True
        self._training_time = time.time() - start_time

        return self

    def predict(
        self,
        trajectory: Trajectory,
        prediction_horizon: float,
        features: pd.DataFrame | None = None,
    ) -> PredictionResult:
        """Predict future trajectory using KNN with DTW similarity."""
        _ = features  # Not used for this model
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        start_time = time.time()

        points = trajectory.points
        if len(points) < 2:
            return PredictionResult(
                trajectory_id=trajectory.trajectory_id,
                predicted_points=[],
                prediction_horizon=prediction_horizon,
                metadata={"error": "Insufficient points for prediction"},
            )

        # Extract features from query trajectory
        query_features = self._extract_trajectory_features(trajectory)
        if len(query_features) == 0:
            return PredictionResult(
                trajectory_id=trajectory.trajectory_id,
                predicted_points=[],
                prediction_horizon=prediction_horizon,
                metadata={"error": "Could not extract features"},
            )

        # Find k nearest neighbors
        distances = []
        for i, train_features in enumerate(self.training_features):
            distance = self._compute_similarity(query_features, train_features)
            distances.append((distance, i))

        # Sort by distance and get k nearest
        distances.sort(key=lambda x: x[0])
        nearest_indices = [idx for _, idx in distances[: self.n_neighbors]]
        nearest_distances = [dist for dist, _ in distances[: self.n_neighbors]]

        # Get corresponding trajectories
        nearest_trajectories = [self.training_trajectories[i] for i in nearest_indices]

        # Generate predictions by averaging future points from neighbors
        prediction_frequency = 10  # 10 Hz predictions
        time_step = 1.0 / prediction_frequency
        num_steps = int(prediction_horizon * prediction_frequency)

        predicted_points = []

        # For each time step, predict by averaging neighbors
        for step in range(num_steps):
            step_predictions = []
            weights = []

            for traj, distance in zip(nearest_trajectories, nearest_distances):
                # Try to find a continuation pattern from this trajectory
                traj_points = traj.points

                # Simple approach: use the average velocity from the neighbor
                if len(traj_points) >= 2:
                    # Calculate average velocity
                    velocities = []
                    for i in range(1, len(traj_points)):
                        dt = traj_points[i].timestamp - traj_points[i - 1].timestamp
                        if dt > 0:
                            vx = (traj_points[i].x - traj_points[i - 1].x) / dt
                            vy = (traj_points[i].y - traj_points[i - 1].y) / dt
                            velocities.append([vx, vy])

                    if velocities:
                        avg_velocity = np.mean(velocities, axis=0)

                        # Project from last point of query trajectory
                        last_point = trajectory.points[-1]
                        future_time = (step + 1) * time_step
                        future_pos = (
                            np.array([last_point.x, last_point.y])
                            + avg_velocity * future_time
                        )

                        step_predictions.append(future_pos)
                        # Weight by inverse distance (add small epsilon to avoid division by zero)
                        weight = 1.0 / (distance + 1e-6)
                        weights.append(weight)

            if step_predictions:
                # Weighted average of predictions
                step_predictions = np.array(step_predictions)
                weights = np.array(weights)
                weights = weights / np.sum(weights)  # Normalize weights

                weighted_pred = np.average(step_predictions, axis=0, weights=weights)
                predicted_points.append(
                    (float(weighted_pred[0]), float(weighted_pred[1]))
                )
            else:
                # Fallback: constant velocity from last point
                if len(points) >= 2:
                    last_point = points[-1]
                    prev_point = points[-2]
                    dt = last_point.timestamp - prev_point.timestamp
                    if dt > 0:
                        vx = (last_point.x - prev_point.x) / dt
                        vy = (last_point.y - prev_point.y) / dt
                        future_time = (step + 1) * time_step
                        future_pos = (
                            np.array([last_point.x, last_point.y])
                            + np.array([vx, vy]) * future_time
                        )
                        predicted_points.append(
                            (float(future_pos[0]), float(future_pos[1]))
                        )

        inference_time = time.time() - start_time

        return PredictionResult(
            trajectory_id=trajectory.trajectory_id,
            predicted_points=predicted_points,
            prediction_horizon=prediction_horizon,
            metadata={
                "inference_time": inference_time,
                "n_neighbors": self.n_neighbors,
                "distance_metric": self.distance_metric,
                "nearest_distances": nearest_distances,
                "prediction_frequency": prediction_frequency,
            },
        )

    def get_feature_importance(self) -> dict[str, float] | None:
        """KNN doesn't have traditional feature importance."""
        if not self.is_fitted:
            return None

        # Return some basic statistics about the model
        return {
            "n_training_trajectories": float(len(self.training_trajectories)),
            "n_neighbors": float(self.n_neighbors),
            "avg_trajectory_length": float(
                np.mean([len(t.points) for t in self.training_trajectories])
            ),
            "distance_metric_weight": 1.0,  # All features weighted equally in DTW
        }
