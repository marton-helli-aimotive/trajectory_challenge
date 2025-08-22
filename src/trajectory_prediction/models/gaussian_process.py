"""Gaussian Process Regression for trajectory prediction with uncertainty quantification."""

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

from ..data.models import Trajectory
from .base import FeatureBasedPredictor, ModelConfig, PredictionResult


class GaussianProcessPredictor(FeatureBasedPredictor):
    """Gaussian Process Regression for trajectory prediction with uncertainty."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._gp_x: GaussianProcessRegressor
        self._gp_y: GaussianProcessRegressor

        # Override parent's _scaler to ensure it's properly typed
        self._scaler: StandardScaler = StandardScaler()

        # Extract hyperparameters
        params = config.hyperparameters
        kernel_type = params.get("kernel", "rbf")
        length_scale = params.get("length_scale", 1.0)
        noise_level = params.get("noise_level", 1e-5)

        # Create kernel based on configuration
        if kernel_type == "rbf":
            kernel = RBF(length_scale=length_scale) + WhiteKernel(
                noise_level=noise_level
            )
        elif kernel_type == "matern":
            nu = params.get("nu", 1.5)
            kernel = Matern(length_scale=length_scale, nu=nu) + WhiteKernel(
                noise_level=noise_level
            )
        else:
            kernel = RBF(length_scale=length_scale) + WhiteKernel(
                noise_level=noise_level
            )

        # Initialize separate GPs for x and y coordinates
        self._gp_x = GaussianProcessRegressor(
            kernel=kernel,
            random_state=config.random_state,
            normalize_y=True,
            alpha=1e-10,
        )
        self._gp_y = GaussianProcessRegressor(
            kernel=kernel,
            random_state=config.random_state,
            normalize_y=True,
            alpha=1e-10,
        )

    def _extract_features(self, trajectory: Trajectory) -> np.ndarray:
        """Extract features for GP including temporal and kinematic information."""
        points = trajectory.points
        if len(points) < 3:
            return np.array([])

        # Use multiple recent points for better prediction
        n_points = min(5, len(points))
        recent_points = points[-n_points:]

        features = []

        # Position features (recent trajectory)
        for point in recent_points:
            features.extend([point.x, point.y])

        # Velocity features
        for point in recent_points:
            features.extend([point.velocity_x, point.velocity_y])

        # Acceleration features
        for point in recent_points:
            features.extend([point.acceleration_x, point.acceleration_y])

        # Derived features
        last_point = recent_points[-1]

        # Speed and heading
        speed = np.sqrt(last_point.velocity_x**2 + last_point.velocity_y**2)
        heading = np.arctan2(last_point.velocity_y, last_point.velocity_x)
        features.extend([speed, heading])

        # Curvature (change in heading)
        if len(recent_points) >= 2:
            prev_point = recent_points[-2]
            prev_heading = np.arctan2(prev_point.velocity_y, prev_point.velocity_x)
            curvature = heading - prev_heading
            features.append(curvature)
        else:
            features.append(0.0)

        # Time since start
        features.append(last_point.timestamp - recent_points[0].timestamp)

        return np.array(features)

    def fit(
        self, trajectories: list[Trajectory], features: pd.DataFrame | None = None
    ) -> "GaussianProcessPredictor":
        """Train the Gaussian Process models."""
        del features  # Unused parameter

        X = self._extract_features_batch(trajectories)

        if X.size == 0:
            raise ValueError("No valid features extracted from trajectories")

        # Prepare targets - predict next position
        y_x, y_y = [], []
        for trajectory in trajectories:
            points = trajectory.points
            if len(points) >= 2:
                # Predict position delta for next time step
                last_point = points[-1]
                if len(points) >= 2:
                    second_last = points[-2]
                    dt = last_point.timestamp - second_last.timestamp
                    # Extrapolate next position
                    next_x = last_point.x + last_point.velocity_x * dt
                    next_y = last_point.y + last_point.velocity_y * dt
                else:
                    next_x = last_point.x
                    next_y = last_point.y

                y_x.append(next_x)
                y_y.append(next_y)

        if not y_x:
            raise ValueError("No valid targets extracted from trajectories")

        y_x = np.array(y_x)
        y_y = np.array(y_y)

        # Scale features
        X_scaled = self._scaler.fit_transform(X)

        # Train separate GPs for x and y coordinates
        self._gp_x.fit(X_scaled, y_x)
        self._gp_y.fit(X_scaled, y_y)

        self.is_fitted = True
        self._feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        return self

    def predict(
        self,
        trajectory: Trajectory,
        prediction_horizon: float,
        features: pd.DataFrame | None = None,
    ) -> PredictionResult:
        """Predict future trajectory with uncertainty."""
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

        # Predict with uncertainty
        gp_result_x = self._gp_x.predict(X_scaled, return_std=True)
        gp_result_y = self._gp_y.predict(X_scaled, return_std=True)

        # Handle tuple unpacking properly
        if isinstance(gp_result_x, tuple):
            pred_x, std_x = gp_result_x[0], gp_result_x[1]
        else:
            pred_x, std_x = gp_result_x, np.zeros_like(gp_result_x)

        if isinstance(gp_result_y, tuple):
            pred_y, std_y = gp_result_y[0], gp_result_y[1]
        else:
            pred_y, std_y = gp_result_y, np.zeros_like(gp_result_y)

        # Generate trajectory points for the prediction horizon
        points = trajectory.points
        if not points:
            return PredictionResult(
                trajectory_id=trajectory.trajectory_id,
                predicted_points=[],
                prediction_horizon=prediction_horizon,
                confidence_scores=[],
            )

        last_point = points[-1]
        dt = 0.1  # 100ms time steps
        n_steps = int(prediction_horizon / dt)

        predicted_points = []
        confidence_scores = []

        # Start from predicted next position
        current_x, current_y = float(pred_x[0]), float(pred_y[0])
        current_vx, current_vy = last_point.velocity_x, last_point.velocity_y

        for _ in range(n_steps):
            predicted_points.append((current_x, current_y))

            # Confidence based on prediction uncertainty
            uncertainty = np.sqrt(std_x[0] ** 2 + std_y[0] ** 2)
            confidence = max(0.0, 1.0 - uncertainty / 10.0)  # Normalize uncertainty
            confidence_scores.append(confidence)

            # Propagate using constant velocity assumption
            current_x += current_vx * dt
            current_y += current_vy * dt

        return PredictionResult(
            trajectory_id=trajectory.trajectory_id,
            predicted_points=predicted_points,
            prediction_horizon=prediction_horizon,
            confidence_scores=confidence_scores,
            metadata={
                "epistemic_uncertainty_x": float(std_x[0]),
                "epistemic_uncertainty_y": float(std_y[0]),
                "prediction_method": "gaussian_process",
            },
        )

    def get_feature_importance(self) -> dict[str, float] | None:
        """GP doesn't have traditional feature importance, return kernel parameters."""
        if not self.is_fitted:
            return None

        # Return kernel hyperparameters as proxy for importance
        kernel_params = self._gp_x.kernel_.get_params()
        return {
            f"kernel_{k}": float(v)
            for k, v in kernel_params.items()
            if isinstance(v, int | float)
        }

    def predict_with_uncertainty(
        self,
        trajectory: Trajectory,
        prediction_horizon: float,
        n_samples: int = 100,
    ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        """Generate uncertainty-aware predictions by sampling from posterior.

        Returns:
            Tuple of (mean_predictions, uncertainty_bounds)
        """
        if not self.is_fitted:
            return [], []

        X = self._extract_features(trajectory).reshape(1, -1)
        if X.size == 0:
            return [], []

        X_scaled = self._scaler.transform(X)

        # Sample from posterior
        samples_x = self._gp_x.sample_y(
            X_scaled, n_samples=n_samples, random_state=self.config.random_state
        )
        samples_y = self._gp_y.sample_y(
            X_scaled, n_samples=n_samples, random_state=self.config.random_state
        )

        # Calculate statistics
        mean_x, mean_y = np.mean(samples_x, axis=1)[0], np.mean(samples_y, axis=1)[0]
        std_x, std_y = np.std(samples_x, axis=1)[0], np.std(samples_y, axis=1)[0]

        # Generate trajectory
        points = trajectory.points
        if not points:
            return [], []

        last_point = points[-1]
        dt = 0.1
        n_steps = int(prediction_horizon / dt)

        mean_predictions = []
        uncertainty_bounds = []

        current_x, current_y = float(mean_x), float(mean_y)
        current_vx, current_vy = last_point.velocity_x, last_point.velocity_y

        for _ in range(n_steps):
            mean_predictions.append((current_x, current_y))

            # 95% confidence interval
            bound_x = 1.96 * std_x
            bound_y = 1.96 * std_y
            uncertainty_bounds.append((bound_x, bound_y))

            current_x += current_vx * dt
            current_y += current_vy * dt

        return mean_predictions, uncertainty_bounds
