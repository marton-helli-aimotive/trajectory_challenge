"""Polynomial regression model for trajectory prediction."""

import time

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from ..data.features import KinematicFeatureExtractor
from ..data.models import Trajectory
from .base import FeatureBasedPredictor, ModelConfig, PredictionResult


class PolynomialRegressionPredictor(FeatureBasedPredictor):
    """Polynomial regression model with engineered features."""

    def __init__(self, config: ModelConfig | None = None):
        if config is None:
            config = ModelConfig(
                name="PolynomialRegression",
                model_type="classical",
                hyperparameters={
                    "degree": 3,
                    "alpha": 0.01,  # Ridge regularization
                    "include_bias": True,
                    "interaction_only": False,
                },
            )
        super().__init__(config)

        # Initialize components
        self.degree = config.hyperparameters.get("degree", 3)
        self.alpha = config.hyperparameters.get("alpha", 0.01)
        self.include_bias = config.hyperparameters.get("include_bias", True)
        self.interaction_only = config.hyperparameters.get("interaction_only", False)

        # Create sklearn pipeline
        self.model_x = Pipeline(
            [
                (
                    "poly",
                    PolynomialFeatures(
                        degree=self.degree,
                        include_bias=self.include_bias,
                        interaction_only=self.interaction_only,
                    ),
                ),
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=self.alpha, random_state=config.random_state)),
            ]
        )

        self.model_y = Pipeline(
            [
                (
                    "poly",
                    PolynomialFeatures(
                        degree=self.degree,
                        include_bias=self.include_bias,
                        interaction_only=self.interaction_only,
                    ),
                ),
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=self.alpha, random_state=config.random_state)),
            ]
        )

        self.feature_extractor = KinematicFeatureExtractor()

    def _extract_features(self, trajectory: Trajectory) -> np.ndarray:
        """Extract polynomial-appropriate features from trajectory."""
        points = trajectory.points
        if len(points) < 2:
            return np.array([])

        # Use kinematic feature extractor for rich features
        features = self.feature_extractor.extract(trajectory)

        # Select relevant features for polynomial regression
        feature_list = []

        # Position features
        if "total_displacement" in features:
            feature_list.append(features["total_displacement"])
        if "path_length" in features:
            feature_list.append(features["path_length"])
        if "straightness" in features:
            feature_list.append(features["straightness"])

        # Velocity features
        if "avg_speed" in features:
            feature_list.append(features["avg_speed"])
        if "max_speed" in features:
            feature_list.append(features["max_speed"])
        if "speed_variation" in features:
            feature_list.append(features["speed_variation"])

        # Acceleration features
        if "avg_acceleration_magnitude" in features:
            feature_list.append(features["avg_acceleration_magnitude"])
        if "max_acceleration_magnitude" in features:
            feature_list.append(features["max_acceleration_magnitude"])

        # Curvature features
        if "avg_curvature" in features:
            feature_list.append(features["avg_curvature"])
        if "max_curvature" in features:
            feature_list.append(features["max_curvature"])

        # Temporal features
        if "duration" in features:
            feature_list.append(features["duration"])

        # Current state features (most recent point)
        last_point = points[-1]
        feature_list.extend(
            [
                last_point.x,
                last_point.y,
                last_point.velocity_x,
                last_point.velocity_y,
                last_point.acceleration_x,
                last_point.acceleration_y,
            ]
        )

        # If we have enough points, add velocity and acceleration deltas
        if len(points) >= 2:
            prev_point = points[-2]
            dt = last_point.timestamp - prev_point.timestamp
            if dt > 0:
                # Velocity delta
                vel_delta_x = last_point.velocity_x - prev_point.velocity_x
                vel_delta_y = last_point.velocity_y - prev_point.velocity_y
                feature_list.extend([vel_delta_x, vel_delta_y])

                # Acceleration delta
                acc_delta_x = last_point.acceleration_x - prev_point.acceleration_x
                acc_delta_y = last_point.acceleration_y - prev_point.acceleration_y
                feature_list.extend([acc_delta_x, acc_delta_y])

        return np.array(feature_list) if feature_list else np.array([])

    def _prepare_training_data(
        self, trajectories: list[Trajectory]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Prepare features and targets for training."""
        X_list = []
        y_list = []

        for trajectory in trajectories:
            points = trajectory.points
            if len(points) < 3:  # Need at least 3 points
                continue

            # For each trajectory, create multiple training examples
            # by predicting each point from the previous points
            for i in range(2, len(points)):
                # Create a sub-trajectory up to point i-1
                sub_trajectory = Trajectory(
                    trajectory_id=f"{trajectory.trajectory_id}_sub_{i}",
                    vehicle=trajectory.vehicle,
                    points=points[:i],
                    dataset_name=trajectory.dataset_name,
                    completeness_score=None,
                    temporal_consistency_score=None,
                    spatial_accuracy_score=None,
                    smoothness_score=None,
                )

                # Extract features from sub-trajectory
                features = self._extract_features(sub_trajectory)
                if len(features) == 0:
                    continue

                # Target is the next point
                target_point = points[i]
                target = np.array([target_point.x, target_point.y])

                X_list.append(features)
                y_list.append(target)

        if not X_list:
            return np.array([]), np.array([])

        X = np.vstack(X_list)
        y = np.vstack(y_list)

        return X, y

    def fit(
        self, trajectories: list[Trajectory], features: pd.DataFrame | None = None
    ) -> "PolynomialRegressionPredictor":
        """Train the polynomial regression model."""
        _ = features  # Not used for this model
        start_time = time.time()

        # Prepare training data
        X, y = self._prepare_training_data(trajectories)

        if X.shape[0] == 0:
            raise ValueError("No valid training data could be extracted")

        # Store feature names for later reference
        self._feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        # Train separate models for x and y coordinates
        self.model_x.fit(X, y[:, 0])
        self.model_y.fit(X, y[:, 1])

        self.is_fitted = True
        self._training_time = time.time() - start_time

        return self

    def predict(
        self,
        trajectory: Trajectory,
        prediction_horizon: float,
        features: pd.DataFrame | None = None,
    ) -> PredictionResult:
        """Predict future trajectory using polynomial regression."""
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

        # Extract features from current trajectory
        current_features = self._extract_features(trajectory)
        if len(current_features) == 0:
            return PredictionResult(
                trajectory_id=trajectory.trajectory_id,
                predicted_points=[],
                prediction_horizon=prediction_horizon,
                metadata={"error": "Could not extract features"},
            )

        # Reshape for prediction
        X = current_features.reshape(1, -1)

        # Generate predictions
        prediction_frequency = 10  # 10 Hz predictions
        num_steps = int(prediction_horizon * prediction_frequency)

        predicted_points = []

        # Iteratively predict next points
        for _ in range(num_steps):
            # Predict next position
            pred_x = self.model_x.predict(X)[0]
            pred_y = self.model_y.predict(X)[0]

            predicted_points.append((float(pred_x), float(pred_y)))

            # For next iteration, we would ideally update the trajectory
            # with the predicted point, but for simplicity, we'll use
            # a simple approach here
            break  # For now, just predict one step

        # If we only have one prediction, extrapolate using the same pattern
        if len(predicted_points) == 1 and num_steps > 1:
            # Simple linear extrapolation from last real point to first prediction
            last_real = np.array([points[-1].x, points[-1].y])
            first_pred = np.array(predicted_points[0])
            step_vector = first_pred - last_real

            # Generate more points by stepping
            for i in range(1, num_steps):
                next_point = first_pred + step_vector * i
                predicted_points.append((float(next_point[0]), float(next_point[1])))

        inference_time = time.time() - start_time

        return PredictionResult(
            trajectory_id=trajectory.trajectory_id,
            predicted_points=predicted_points,
            prediction_horizon=prediction_horizon,
            metadata={
                "inference_time": inference_time,
                "model_degree": self.degree,
                "regularization_alpha": self.alpha,
                "prediction_frequency": prediction_frequency,
            },
        )

    def get_feature_importance(self) -> dict[str, float] | None:
        """Get feature importance from the polynomial model."""
        if not self.is_fitted or self._feature_names is None:
            return None

        # Get coefficients from the Ridge models
        coef_x = self.model_x.named_steps["ridge"].coef_
        coef_y = self.model_y.named_steps["ridge"].coef_

        # Combine coefficients by taking the magnitude
        combined_coef = np.sqrt(coef_x**2 + coef_y**2)

        # Get feature names from polynomial transformer
        poly_feature_names = self.model_x.named_steps["poly"].get_feature_names_out(
            self._feature_names
        )

        # Create importance dictionary
        importance_dict = {}
        for i, name in enumerate(poly_feature_names):
            if i < len(combined_coef):
                importance_dict[name] = float(combined_coef[i])

        return importance_dict
