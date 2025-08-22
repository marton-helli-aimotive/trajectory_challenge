"""Base model interface for trajectory prediction."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel

from ..data.models import Trajectory


class ModelConfig(BaseModel):  # type: ignore[misc]
    """Configuration for trajectory prediction models."""

    name: str
    model_type: str
    hyperparameters: dict[str, Any] = {}
    random_state: int | None = 42


class PredictionResult(BaseModel):  # type: ignore[misc]
    """Result of trajectory prediction."""

    trajectory_id: str
    predicted_points: list[tuple[float, float]]  # (x, y) coordinates
    prediction_horizon: float  # seconds
    confidence_scores: list[float] | None = None
    metadata: dict[str, Any] = {}


class ModelPerformance(BaseModel):  # type: ignore[misc]
    """Model performance metrics."""

    rmse: float
    ade: float  # Average Displacement Error
    fde: float  # Final Displacement Error
    training_time: float
    inference_time: float
    metadata: dict[str, Any] = {}


class TrajectoryPredictor(ABC):
    """Abstract base class for trajectory prediction models."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.is_fitted = False
        self._feature_names: list[str] | None = None

    @abstractmethod
    def fit(
        self, trajectories: list[Trajectory], features: pd.DataFrame | None = None
    ) -> "TrajectoryPredictor":
        """Train the model on trajectory data.

        Args:
            trajectories: List of training trajectories
            features: Optional pre-computed features

        Returns:
            Self for method chaining
        """
        pass

    @abstractmethod
    def predict(
        self,
        trajectory: Trajectory,
        prediction_horizon: float,
        features: pd.DataFrame | None = None,
    ) -> PredictionResult:
        """Predict future trajectory points.

        Args:
            trajectory: Input trajectory to predict from
            prediction_horizon: How far into the future to predict (seconds)
            features: Optional pre-computed features

        Returns:
            Prediction result with future trajectory points
        """
        pass

    def predict_batch(
        self,
        trajectories: list[Trajectory],
        prediction_horizon: float,
        features: pd.DataFrame | None = None,
    ) -> list[PredictionResult]:
        """Predict multiple trajectories efficiently.

        Args:
            trajectories: List of trajectories to predict
            prediction_horizon: How far into the future to predict (seconds)
            features: Optional pre-computed features

        Returns:
            List of prediction results
        """
        return [
            self.predict(traj, prediction_horizon, features) for traj in trajectories
        ]

    @abstractmethod
    def get_feature_importance(self) -> dict[str, float] | None:
        """Get feature importance scores if supported.

        Returns:
            Dictionary mapping feature names to importance scores, or None
        """
        pass

    def save_model(self, filepath: str) -> None:
        """Save model to disk."""
        import pickle
        from pathlib import Path

        with Path(filepath).open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_model(cls, filepath: str) -> "TrajectoryPredictor":
        """Load model from disk."""
        import pickle
        from pathlib import Path

        with Path(filepath).open("rb") as f:
            return pickle.load(f)  # type: ignore[no-any-return]

    @property
    def model_name(self) -> str:
        """Get model name."""
        return self.config.name

    @property
    def feature_names(self) -> list[str] | None:
        """Get feature names used by the model."""
        return self._feature_names


class FeatureBasedPredictor(TrajectoryPredictor):
    """Base class for models that use engineered features."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._scaler: Any | None = None

    def _extract_features(self, trajectory: Trajectory) -> np.ndarray:
        """Extract features from a single trajectory.

        This should be overridden by subclasses to define their feature extraction.
        """
        # Default: use basic kinematic features
        points = trajectory.points
        if len(points) < 2:
            return np.array([])

        # Extract basic features: position, velocity, acceleration
        last_point = points[-1]
        features = [
            last_point.x,
            last_point.y,
            last_point.velocity_x,
            last_point.velocity_y,
            last_point.acceleration_x,
            last_point.acceleration_y,
        ]

        return np.array(features)

    def _extract_features_batch(self, trajectories: list[Trajectory]) -> np.ndarray:
        """Extract features from multiple trajectories."""
        feature_list = []
        for trajectory in trajectories:
            features = self._extract_features(trajectory)
            if len(features) > 0:
                feature_list.append(features)

        if not feature_list:
            return np.array([])

        return np.vstack(feature_list)

    def _prepare_training_data(
        self, trajectories: list[Trajectory]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Prepare features and targets for training.

        Returns:
            Tuple of (features, targets) arrays
        """
        X = self._extract_features_batch(trajectories)

        # For now, use next position as target (this will be refined by subclasses)
        y_list = []
        for trajectory in trajectories:
            points = trajectory.points
            if len(points) >= 2:
                # Target is the last point's position
                last_point = points[-1]
                y_list.append([last_point.x, last_point.y])

        y = np.vstack(y_list) if y_list else np.array([])

        return X, y
