"""Baseline trajectory prediction models."""

import time

import numpy as np
import pandas as pd

from ..data.models import Trajectory
from .base import ModelConfig, PredictionResult, TrajectoryPredictor


class ConstantVelocityPredictor(TrajectoryPredictor):
    """Baseline model using constant velocity extrapolation."""

    def __init__(self, config: ModelConfig | None = None):
        if config is None:
            config = ModelConfig(name="ConstantVelocity", model_type="baseline")
        super().__init__(config)

    def fit(
        self, trajectories: list[Trajectory], features: pd.DataFrame | None = None
    ) -> "ConstantVelocityPredictor":
        """Fit the model (no training needed for constant velocity)."""
        # Note: trajectories and features are required by interface but not used for baseline models
        _ = trajectories, features
        start_time = time.time()

        # No actual training needed - just mark as fitted
        self.is_fitted = True
        self._training_time = time.time() - start_time

        return self

    def predict(
        self,
        trajectory: Trajectory,
        prediction_horizon: float,
        features: pd.DataFrame | None = None,
    ) -> PredictionResult:
        """Predict future trajectory using constant velocity extrapolation."""
        # Note: features not used for baseline models
        _ = features
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        start_time = time.time()

        points = trajectory.points
        if len(points) < 2:
            # Cannot predict with less than 2 points
            return PredictionResult(
                trajectory_id=trajectory.trajectory_id,
                predicted_points=[],
                prediction_horizon=prediction_horizon,
                metadata={"error": "Insufficient points for prediction"},
            )

        # Use the last point's velocity
        last_point = points[-1]
        current_pos = np.array([last_point.x, last_point.y])
        velocity = np.array([last_point.velocity_x, last_point.velocity_y])

        # If velocity is zero or very small, use velocity from last two points
        if np.linalg.norm(velocity) < 1e-3 and len(points) >= 2:
            prev_point = points[-2]
            dt = last_point.timestamp - prev_point.timestamp
            if dt > 0:
                velocity = np.array(
                    [
                        (last_point.x - prev_point.x) / dt,
                        (last_point.y - prev_point.y) / dt,
                    ]
                )

        # Generate prediction points
        prediction_frequency = 10  # 10 Hz predictions
        time_step = 1.0 / prediction_frequency
        num_steps = int(prediction_horizon * prediction_frequency)

        predicted_points = []
        for i in range(1, num_steps + 1):
            t = i * time_step
            future_pos = current_pos + velocity * t
            predicted_points.append((float(future_pos[0]), float(future_pos[1])))

        inference_time = time.time() - start_time

        return PredictionResult(
            trajectory_id=trajectory.trajectory_id,
            predicted_points=predicted_points,
            prediction_horizon=prediction_horizon,
            metadata={
                "inference_time": inference_time,
                "velocity": velocity.tolist(),
                "prediction_frequency": prediction_frequency,
            },
        )

    def get_feature_importance(self) -> dict[str, float] | None:
        """Constant velocity model doesn't use features."""
        return None


class ConstantAccelerationPredictor(TrajectoryPredictor):
    """Baseline model using constant acceleration extrapolation."""

    def __init__(self, config: ModelConfig | None = None):
        if config is None:
            config = ModelConfig(name="ConstantAcceleration", model_type="baseline")
        super().__init__(config)

    def fit(
        self, trajectories: list[Trajectory], features: pd.DataFrame | None = None
    ) -> "ConstantAccelerationPredictor":
        """Fit the model (no training needed for constant acceleration)."""
        # Note: trajectories and features are required by interface but not used for baseline models
        _ = trajectories, features
        start_time = time.time()

        # No actual training needed - just mark as fitted
        self.is_fitted = True
        self._training_time = time.time() - start_time

        return self

    def predict(
        self,
        trajectory: Trajectory,
        prediction_horizon: float,
        features: pd.DataFrame | None = None,
    ) -> PredictionResult:
        """Predict future trajectory using constant acceleration extrapolation."""
        # Note: features not used for baseline models
        _ = features
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        start_time = time.time()

        points = trajectory.points
        if len(points) < 3:
            # Fall back to constant velocity if less than 3 points
            cv_predictor = ConstantVelocityPredictor()
            cv_predictor.is_fitted = True
            result = cv_predictor.predict(trajectory, prediction_horizon)
            result.metadata.update({"fallback": "constant_velocity"})
            return result

        # Use the last point's position, velocity, and acceleration
        last_point = points[-1]
        current_pos = np.array([last_point.x, last_point.y])
        velocity = np.array([last_point.velocity_x, last_point.velocity_y])
        acceleration = np.array([last_point.acceleration_x, last_point.acceleration_y])

        # If acceleration is not available or zero, estimate from recent points
        if np.linalg.norm(acceleration) < 1e-6 and len(points) >= 3:
            # Estimate acceleration from velocity changes
            prev_point = points[-2]
            prev_prev_point = points[-3]

            dt1 = last_point.timestamp - prev_point.timestamp
            dt2 = prev_point.timestamp - prev_prev_point.timestamp

            if dt1 > 0 and dt2 > 0:
                # Current velocity
                v_current = np.array([last_point.velocity_x, last_point.velocity_y])
                # Previous velocity
                v_prev = np.array([prev_point.velocity_x, prev_point.velocity_y])

                # If velocities are not available, estimate from positions
                if np.linalg.norm(v_current) < 1e-6:
                    v_current = np.array(
                        [
                            (last_point.x - prev_point.x) / dt1,
                            (last_point.y - prev_point.y) / dt1,
                        ]
                    )

                if np.linalg.norm(v_prev) < 1e-6:
                    v_prev = np.array(
                        [
                            (prev_point.x - prev_prev_point.x) / dt2,
                            (prev_point.y - prev_prev_point.y) / dt2,
                        ]
                    )

                # Estimate acceleration
                acceleration = (v_current - v_prev) / dt1

        # Generate prediction points using kinematic equations
        # x(t) = x0 + v0*t + 0.5*a*t^2
        prediction_frequency = 10  # 10 Hz predictions
        time_step = 1.0 / prediction_frequency
        num_steps = int(prediction_horizon * prediction_frequency)

        predicted_points = []
        for i in range(1, num_steps + 1):
            t = i * time_step
            future_pos = current_pos + velocity * t + 0.5 * acceleration * t**2
            predicted_points.append((float(future_pos[0]), float(future_pos[1])))

        inference_time = time.time() - start_time

        return PredictionResult(
            trajectory_id=trajectory.trajectory_id,
            predicted_points=predicted_points,
            prediction_horizon=prediction_horizon,
            metadata={
                "inference_time": inference_time,
                "velocity": velocity.tolist(),
                "acceleration": acceleration.tolist(),
                "prediction_frequency": prediction_frequency,
            },
        )

    def get_feature_importance(self) -> dict[str, float] | None:
        """Constant acceleration model doesn't use features."""
        return None
