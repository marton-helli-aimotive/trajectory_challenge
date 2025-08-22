"""Tree-based ensemble models for trajectory prediction with uncertainty quantification."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from ..data.models import Trajectory
from .base import FeatureBasedPredictor, ModelConfig, PredictionResult


class TreeEnsemblePredictor(FeatureBasedPredictor):
    """Tree-based ensemble for trajectory prediction with uncertainty estimation."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._model_x: RandomForestRegressor | XGBRegressor
        self._model_y: RandomForestRegressor | XGBRegressor
        self._scaler: StandardScaler = StandardScaler()

        # Extract hyperparameters
        params = config.hyperparameters
        model_type = params.get("model_type", "random_forest")
        n_estimators = params.get("n_estimators", 100)
        max_depth = params.get("max_depth", 10)

        # Initialize models based on type
        if model_type == "random_forest":
            self._model_x = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=config.random_state,
                n_jobs=-1,
            )
            self._model_y = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=config.random_state,
                n_jobs=-1,
            )
        elif model_type == "xgboost":
            self._model_x = XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=config.random_state,
                n_jobs=-1,
            )
            self._model_y = XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=config.random_state,
                n_jobs=-1,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _extract_features(self, trajectory: Trajectory) -> np.ndarray:
        """Extract rich features for tree-based models."""
        points = trajectory.points
        if len(points) < 3:
            return np.array([])

        # Use recent trajectory points
        n_points = min(10, len(points))
        recent_points = points[-n_points:]

        features = []

        # Current state features
        last_point = recent_points[-1]

        # Position features
        features.extend([last_point.x, last_point.y])

        # Velocity features
        features.extend([last_point.velocity_x, last_point.velocity_y])

        # Acceleration features
        features.extend([last_point.acceleration_x, last_point.acceleration_y])

        # Speed and heading
        speed = np.sqrt(last_point.velocity_x**2 + last_point.velocity_y**2)
        heading = np.arctan2(last_point.velocity_y, last_point.velocity_x)
        features.extend([speed, heading])

        # Trajectory history features
        if len(recent_points) >= 2:
            # Velocity change
            prev_point = recent_points[-2]
            velocity_change_x = last_point.velocity_x - prev_point.velocity_x
            velocity_change_y = last_point.velocity_y - prev_point.velocity_y
            features.extend([velocity_change_x, velocity_change_y])

            # Speed change
            prev_speed = np.sqrt(prev_point.velocity_x**2 + prev_point.velocity_y**2)
            speed_change = speed - prev_speed
            features.append(speed_change)

            # Heading change (curvature)
            prev_heading = np.arctan2(prev_point.velocity_y, prev_point.velocity_x)
            heading_change = heading - prev_heading
            # Normalize to [-pi, pi]
            while heading_change > np.pi:
                heading_change -= 2 * np.pi
            while heading_change < -np.pi:
                heading_change += 2 * np.pi
            features.append(heading_change)

            # Time delta
            dt = last_point.timestamp - prev_point.timestamp
            features.append(dt)
        else:
            # Fill with zeros if not enough history
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0])

        # Statistical features over trajectory window
        if len(recent_points) >= 3:
            # Position statistics
            x_positions = [p.x for p in recent_points]
            y_positions = [p.y for p in recent_points]

            features.extend(
                [
                    np.mean(x_positions),
                    np.std(x_positions),
                    np.mean(y_positions),
                    np.std(y_positions),
                ]
            )

            # Velocity statistics
            vx_values = [p.velocity_x for p in recent_points]
            vy_values = [p.velocity_y for p in recent_points]

            features.extend(
                [
                    np.mean(vx_values),
                    np.std(vx_values),
                    np.mean(vy_values),
                    np.std(vy_values),
                ]
            )

            # Speed statistics
            speeds = [np.sqrt(p.velocity_x**2 + p.velocity_y**2) for p in recent_points]
            features.extend(
                [
                    np.mean(speeds),
                    np.std(speeds),
                    np.min(speeds),
                    np.max(speeds),
                ]
            )
        else:
            # Fill with current values if not enough history
            features.extend(
                [
                    last_point.x,
                    0.0,
                    last_point.y,
                    0.0,  # Position stats
                    last_point.velocity_x,
                    0.0,
                    last_point.velocity_y,
                    0.0,  # Velocity stats
                    speed,
                    0.0,
                    speed,
                    speed,  # Speed stats
                ]
            )

        # Trajectory length and duration
        total_distance = 0.0
        for i in range(1, len(recent_points)):
            dx = recent_points[i].x - recent_points[i - 1].x
            dy = recent_points[i].y - recent_points[i - 1].y
            total_distance += np.sqrt(dx**2 + dy**2)

        duration = recent_points[-1].timestamp - recent_points[0].timestamp
        features.extend([total_distance, duration])

        return np.array(features)

    def fit(
        self, trajectories: list[Trajectory], features: pd.DataFrame | None = None
    ) -> "TreeEnsemblePredictor":
        """Train the tree ensemble models."""
        del features  # Unused parameter

        X = self._extract_features_batch(trajectories)

        if X.size == 0:
            raise ValueError("No valid features extracted from trajectories")

        # Prepare targets - predict next position
        y_x, y_y = [], []
        for trajectory in trajectories:
            points = trajectory.points
            if len(points) >= 2:
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

        y_x_array = np.array(y_x)
        y_y_array = np.array(y_y)

        # Scale features
        X_scaled = self._scaler.fit_transform(X)

        # Train separate models for x and y coordinates
        self._model_x.fit(X_scaled, y_x_array)
        self._model_y.fit(X_scaled, y_y_array)

        self.is_fitted = True
        self._feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        return self

    def predict(
        self,
        trajectory: Trajectory,
        prediction_horizon: float,
        features: pd.DataFrame | None = None,
    ) -> PredictionResult:
        """Predict future trajectory with uncertainty from ensemble variance."""
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

        # Get predictions and uncertainty estimates
        pred_x = self._model_x.predict(X_scaled)[0]
        pred_y = self._model_y.predict(X_scaled)[0]

        # Calculate uncertainty from ensemble variance
        uncertainty_x = self._calculate_prediction_uncertainty(X_scaled, "x")
        uncertainty_y = self._calculate_prediction_uncertainty(X_scaled, "y")

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
        current_x, current_y = float(pred_x), float(pred_y)
        current_vx, current_vy = last_point.velocity_x, last_point.velocity_y

        for _ in range(n_steps):
            predicted_points.append((current_x, current_y))

            # Confidence based on prediction uncertainty
            uncertainty = np.sqrt(uncertainty_x**2 + uncertainty_y**2)
            confidence = max(0.0, 1.0 - uncertainty / 10.0)
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
                "aleatoric_uncertainty_x": float(uncertainty_x),
                "aleatoric_uncertainty_y": float(uncertainty_y),
                "prediction_method": "tree_ensemble",
            },
        )

    def _calculate_prediction_uncertainty(
        self, X: np.ndarray, coordinate: str
    ) -> float:
        """Calculate prediction uncertainty from ensemble variance."""
        model = self._model_x if coordinate == "x" else self._model_y

        if hasattr(model, "estimators_"):
            # For RandomForest - get predictions from individual trees
            tree_predictions = []
            for estimator in model.estimators_:
                pred = estimator.predict(X)
                tree_predictions.append(pred[0])

            # Return standard deviation as uncertainty measure
            return float(np.std(tree_predictions))
        else:
            # For XGBoost - approximate uncertainty (this is a simplification)
            # In practice, you might use quantile regression or other methods
            pred = model.predict(X)
            return float(abs(pred[0]) * 0.1)  # 10% of prediction as uncertainty

    def get_feature_importance(self) -> dict[str, float] | None:
        """Get feature importance from tree-based models."""
        if not self.is_fitted:
            return None

        # Average importance from both models
        importance_x = self._model_x.feature_importances_
        importance_y = self._model_y.feature_importances_

        avg_importance = (importance_x + importance_y) / 2

        if self._feature_names:
            return dict(zip(self._feature_names, avg_importance))
        else:
            return {
                f"feature_{i}": importance
                for i, importance in enumerate(avg_importance)
            }

    def predict_with_quantiles(
        self,
        trajectory: Trajectory,
        prediction_horizon: float,
        quantiles: list[float] | None = None,
    ) -> dict[str, list[tuple[float, float]]]:
        """Predict with quantile estimates for uncertainty bounds.

        Returns:
            Dictionary with quantile predictions
        """
        if quantiles is None:
            quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]

        if not self.is_fitted:
            return {}

        # For now, return simplified quantile estimates
        # In practice, you would train quantile regression models
        base_pred = self.predict(trajectory, prediction_horizon)

        result = {}
        for q in quantiles:
            # Simple approximation - scale predictions by quantile
            scale_factor = 0.5 + q  # Simple scaling
            scaled_points = [
                (x * scale_factor, y * scale_factor)
                for x, y in base_pred.predicted_points
            ]
            result[f"quantile_{q}"] = scaled_points

        return result
