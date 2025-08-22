"""Mixture Density Network for probabilistic trajectory prediction using scikit-learn."""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from ..data.models import Trajectory
from .base import FeatureBasedPredictor, ModelConfig, PredictionResult


class SimplifiedMDNPredictor(FeatureBasedPredictor):
    """Simplified probabilistic predictor using multiple neural networks."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._models_x: list[MLPRegressor] = []
        self._models_y: list[MLPRegressor] = []
        self._scaler: StandardScaler = StandardScaler()

        # Extract hyperparameters
        params = config.hyperparameters
        self.n_components = params.get("n_components", 3)
        self.hidden_size = params.get("hidden_size", (64, 32))
        self.max_iter = params.get("max_iter", 500)
        self.learning_rate_init = params.get("learning_rate_init", 0.001)

        # Create multiple MLPs to simulate mixture components
        for _ in range(self.n_components):
            mlp_x = MLPRegressor(
                hidden_layer_sizes=self.hidden_size,
                max_iter=self.max_iter,
                learning_rate_init=self.learning_rate_init,
                random_state=config.random_state,
                early_stopping=True,
                validation_fraction=0.1,
            )
            mlp_y = MLPRegressor(
                hidden_layer_sizes=self.hidden_size,
                max_iter=self.max_iter,
                learning_rate_init=self.learning_rate_init,
                random_state=config.random_state,
                early_stopping=True,
                validation_fraction=0.1,
            )
            self._models_x.append(mlp_x)
            self._models_y.append(mlp_y)

    def _extract_features(self, trajectory: Trajectory) -> np.ndarray:
        """Extract features suitable for neural network."""
        points = trajectory.points
        if len(points) < 3:
            return np.array([])

        # Use recent trajectory points
        n_points = min(5, len(points))
        recent_points = points[-n_points:]

        features = []

        # Current state
        last_point = recent_points[-1]
        features.extend(
            [
                last_point.x,
                last_point.y,
                last_point.velocity_x,
                last_point.velocity_y,
                last_point.acceleration_x,
                last_point.acceleration_y,
            ]
        )

        # Derived kinematic features
        speed = np.sqrt(last_point.velocity_x**2 + last_point.velocity_y**2)
        heading = np.arctan2(last_point.velocity_y, last_point.velocity_x)
        features.extend([speed, heading])

        # Historical features (pad if necessary)
        max_history = 3
        for i in range(max_history):
            if i < len(recent_points) - 1:
                point = recent_points[-(i + 2)]
                features.extend(
                    [
                        point.x,
                        point.y,
                        point.velocity_x,
                        point.velocity_y,
                    ]
                )
            else:
                # Pad with last known values
                features.extend(
                    [
                        last_point.x,
                        last_point.y,
                        last_point.velocity_x,
                        last_point.velocity_y,
                    ]
                )

        # Temporal features
        if len(recent_points) >= 2:
            dt = recent_points[-1].timestamp - recent_points[-2].timestamp
            features.append(dt)
        else:
            features.append(0.1)  # Default time step

        return np.array(features)

    def fit(
        self, trajectories: list[Trajectory], features: pd.DataFrame | None = None
    ) -> "SimplifiedMDNPredictor":
        """Train multiple neural networks as mixture components."""
        del features  # Unused parameter

        X = self._extract_features_batch(trajectories)

        if X.size == 0:
            raise ValueError("No valid features extracted from trajectories")

        # Prepare targets
        y_list = []
        for trajectory in trajectories:
            points = trajectory.points
            if len(points) >= 2:
                last_point = points[-1]
                if len(points) >= 2:
                    second_last = points[-2]
                    dt = last_point.timestamp - second_last.timestamp
                    next_x = last_point.x + last_point.velocity_x * dt
                    next_y = last_point.y + last_point.velocity_y * dt
                else:
                    next_x = last_point.x
                    next_y = last_point.y

                y_list.append([next_x, next_y])

        if not y_list:
            raise ValueError("No valid targets extracted from trajectories")

        y = np.array(y_list)

        # Scale features
        X_scaled = self._scaler.fit_transform(X)

        # Train multiple models on different subsets/with different initialization
        for i, (mlp_x, mlp_y) in enumerate(zip(self._models_x, self._models_y)):
            # Add some noise to create diversity among components
            noise_scale = 0.1 * (i + 1) / self.n_components
            X_noisy = X_scaled + np.random.normal(0, noise_scale, X_scaled.shape)
            y_noisy = y + np.random.normal(0, noise_scale, y.shape)

            mlp_x.fit(X_noisy, y_noisy[:, 0])
            mlp_y.fit(X_noisy, y_noisy[:, 1])

        self.is_fitted = True
        self._feature_names = [f"feature_{i}" for i in range(X_scaled.shape[1])]

        return self

    def predict(
        self,
        trajectory: Trajectory,
        prediction_horizon: float,
        features: pd.DataFrame | None = None,
    ) -> PredictionResult:
        """Predict future trajectory with probabilistic outputs."""
        del features  # Unused parameter

        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Extract features
        X = self._extract_features(trajectory).reshape(1, -1)
        if X.size == 0:
            return PredictionResult(
                trajectory_id=trajectory.trajectory_id,
                predicted_points=[],
                prediction_horizon=prediction_horizon,
                confidence_scores=[],
            )

        X_scaled = self._scaler.transform(X)

        # Get predictions from all components
        predictions_x = []
        predictions_y = []

        for mlp_x, mlp_y in zip(self._models_x, self._models_y):
            pred_x = mlp_x.predict(X_scaled)[0]
            pred_y = mlp_y.predict(X_scaled)[0]
            predictions_x.append(pred_x)
            predictions_y.append(pred_y)

        # Use weighted average (simplified mixture)
        weights = np.ones(self.n_components) / self.n_components
        predicted_x = np.average(predictions_x, weights=weights)
        predicted_y = np.average(predictions_y, weights=weights)

        # Calculate uncertainty from component variance
        uncertainty_x = np.std(predictions_x)
        uncertainty_y = np.std(predictions_y)

        # Generate trajectory points
        points = trajectory.points
        if not points:
            return PredictionResult(
                trajectory_id=trajectory.trajectory_id,
                predicted_points=[],
                prediction_horizon=prediction_horizon,
                confidence_scores=[],
            )

        last_point = points[-1]
        dt = 0.1
        n_steps = int(prediction_horizon / dt)

        predicted_points = []
        confidence_scores = []

        # Start from predicted next position
        current_x, current_y = float(predicted_x), float(predicted_y)
        current_vx, current_vy = last_point.velocity_x, last_point.velocity_y

        # Calculate base confidence from prediction variance
        base_uncertainty = np.sqrt(uncertainty_x**2 + uncertainty_y**2)
        base_confidence = max(0.0, 1.0 - base_uncertainty / 10.0)

        for _ in range(n_steps):
            predicted_points.append((current_x, current_y))
            confidence_scores.append(base_confidence)

            # Propagate using constant velocity
            current_x += current_vx * dt
            current_y += current_vy * dt

        return PredictionResult(
            trajectory_id=trajectory.trajectory_id,
            predicted_points=predicted_points,
            prediction_horizon=prediction_horizon,
            confidence_scores=confidence_scores,
            metadata={
                "component_predictions_x": predictions_x,
                "component_predictions_y": predictions_y,
                "mixture_uncertainty_x": float(uncertainty_x),
                "mixture_uncertainty_y": float(uncertainty_y),
                "prediction_method": "simplified_mixture_density",
            },
        )

    def get_feature_importance(self) -> dict[str, float] | None:
        """Neural networks don't have traditional feature importance."""
        return None

    def sample_trajectories(
        self,
        trajectory: Trajectory,
        prediction_horizon: float,
        n_samples: int = 100,
    ) -> list[list[tuple[float, float]]]:
        """Generate multiple trajectory samples from the learned distribution."""
        if not self.is_fitted:
            return []

        X = self._extract_features(trajectory).reshape(1, -1)
        if X.size == 0:
            return []

        X_scaled = self._scaler.transform(X)

        # Get predictions from all components
        predictions_x = []
        predictions_y = []

        for mlp_x, mlp_y in zip(self._models_x, self._models_y):
            pred_x = mlp_x.predict(X_scaled)[0]
            pred_y = mlp_y.predict(X_scaled)[0]
            predictions_x.append(pred_x)
            predictions_y.append(pred_y)

        samples = []
        dt = 0.1
        n_steps = int(prediction_horizon / dt)
        last_point = trajectory.points[-1]

        for _ in range(n_samples):
            # Sample from one of the components randomly
            component_idx = np.random.randint(0, self.n_components)

            # Add some noise to create variation
            noise_x = np.random.normal(0, abs(predictions_x[component_idx]) * 0.1)
            noise_y = np.random.normal(0, abs(predictions_y[component_idx]) * 0.1)

            current_x = predictions_x[component_idx] + noise_x
            current_y = predictions_y[component_idx] + noise_y
            current_vx, current_vy = last_point.velocity_x, last_point.velocity_y

            trajectory_sample = []
            for _ in range(n_steps):
                trajectory_sample.append((current_x, current_y))
                current_x += current_vx * dt
                current_y += current_vy * dt

            samples.append(trajectory_sample)

        return samples

    def predict_distribution(
        self,
        trajectory: Trajectory,
    ) -> dict[str, float]:
        """Get distribution statistics for the next prediction."""
        if not self.is_fitted:
            return {}

        X = self._extract_features(trajectory).reshape(1, -1)
        if X.size == 0:
            return {}

        X_scaled = self._scaler.transform(X)

        # Get predictions from all components
        predictions_x = []
        predictions_y = []

        for mlp_x, mlp_y in zip(self._models_x, self._models_y):
            pred_x = mlp_x.predict(X_scaled)[0]
            pred_y = mlp_y.predict(X_scaled)[0]
            predictions_x.append(pred_x)
            predictions_y.append(pred_y)

        return {
            "mean_x": float(np.mean(predictions_x)),
            "mean_y": float(np.mean(predictions_y)),
            "std_x": float(np.std(predictions_x)),
            "std_y": float(np.std(predictions_y)),
            "min_x": float(np.min(predictions_x)),
            "max_x": float(np.max(predictions_x)),
            "min_y": float(np.min(predictions_y)),
            "max_y": float(np.max(predictions_y)),
        }
