"""
Base classes for trajectory prediction models.

Defines the common interface and prediction result format
for all trajectory prediction models.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from ..data.validation.schemas import TrajectoryData


@dataclass
class PredictionResult:
    """
    Result of trajectory prediction.
    
    Contains predicted trajectory points with uncertainty estimates
    and additional metadata about the prediction.
    """
    # Predicted trajectory points
    timestamps: np.ndarray  # Future timestamps
    x_coords: np.ndarray    # Predicted X coordinates
    y_coords: np.ndarray    # Predicted Y coordinates
    
    # Uncertainty estimates (optional)
    x_std: Optional[np.ndarray] = None  # Standard deviation for X
    y_std: Optional[np.ndarray] = None  # Standard deviation for Y
    
    # Predicted velocities/accelerations (optional)
    velocities: Optional[np.ndarray] = None
    accelerations: Optional[np.ndarray] = None
    
    # Prediction metadata
    prediction_horizon: float = 0.0  # Prediction horizon in seconds
    confidence_level: float = 0.95   # Confidence level for uncertainty
    model_name: str = "unknown"      # Name of the prediction model
    
    # Additional model-specific data
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def trajectory_points(self) -> List[Tuple[float, float, float]]:
        """Get trajectory as list of (timestamp, x, y) tuples."""
        return list(zip(self.timestamps, self.x_coords, self.y_coords))
    
    @property 
    def has_uncertainty(self) -> bool:
        """Check if prediction includes uncertainty estimates."""
        return self.x_std is not None and self.y_std is not None
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert prediction result to pandas DataFrame."""
        data = {
            "timestamp": self.timestamps,
            "x": self.x_coords, 
            "y": self.y_coords,
            "model": self.model_name
        }
        
        if self.has_uncertainty:
            data["x_std"] = self.x_std
            data["y_std"] = self.y_std
            
        if self.velocities is not None:
            data["velocity"] = self.velocities
            
        if self.accelerations is not None:
            data["acceleration"] = self.accelerations
            
        return pd.DataFrame(data)


class TrajectoryPredictor(ABC):
    """
    Abstract base class for trajectory prediction models.
    
    Defines the interface that all trajectory prediction models must implement,
    supporting both training and prediction with uncertainty quantification.
    """
    
    def __init__(self, config: DictConfig, name: Optional[str] = None):
        self.config = config
        self.name = name or self.__class__.__name__
        self.is_trained = False
        self._model_params = {}
        
    @abstractmethod
    async def fit(
        self, 
        trajectories: List[TrajectoryData],
        validation_trajectories: Optional[List[TrajectoryData]] = None
    ) -> Dict[str, Any]:
        """
        Train the trajectory prediction model.
        
        Args:
            trajectories: Training trajectory data
            validation_trajectories: Optional validation data
            
        Returns:
            Training metrics and metadata
        """
        pass
    
    @abstractmethod
    async def predict(
        self,
        history: List[TrajectoryData],
        prediction_horizon: float,
        **kwargs
    ) -> PredictionResult:
        """
        Predict future trajectory points.
        
        Args:
            history: Historical trajectory points for a single vehicle
            prediction_horizon: How far to predict into the future (seconds)
            **kwargs: Additional prediction parameters
            
        Returns:
            PredictionResult with predicted trajectory and uncertainty
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Model metadata including parameters, capabilities, etc.
        """
        pass
    
    def predict_batch(
        self,
        batch_histories: List[List[TrajectoryData]],
        prediction_horizon: float,
        **kwargs
    ) -> List[PredictionResult]:
        """
        Predict trajectories for multiple vehicles.
        
        Args:
            batch_histories: List of trajectory histories
            prediction_horizon: Prediction horizon in seconds
            **kwargs: Additional parameters
            
        Returns:
            List of prediction results
        """
        results = []
        for history in batch_histories:
            try:
                result = self.predict(history, prediction_horizon, **kwargs)
                results.append(result)
            except Exception as e:
                # Create empty result for failed predictions
                empty_result = PredictionResult(
                    timestamps=np.array([]),
                    x_coords=np.array([]),
                    y_coords=np.array([]),
                    model_name=self.name,
                    metadata={"error": str(e)}
                )
                results.append(empty_result)
        
        return results
    
    def save_model(self, path: str) -> None:
        """
        Save trained model to disk.
        
        Args:
            path: File path to save model
        """
        import pickle
        
        model_data = {
            "name": self.name,
            "config": self.config,
            "is_trained": self.is_trained,
            "params": self._model_params,
            "model_state": self._get_model_state()
        }
        
        with open(path, "wb") as f:
            pickle.dump(model_data, f)
    
    def load_model(self, path: str) -> None:
        """
        Load trained model from disk.
        
        Args:
            path: File path to load model from
        """
        import pickle
        
        with open(path, "rb") as f:
            model_data = pickle.load(f)
        
        self.name = model_data["name"]
        self.config = model_data["config"] 
        self.is_trained = model_data["is_trained"]
        self._model_params = model_data["params"]
        self._set_model_state(model_data["model_state"])
    
    def _get_model_state(self) -> Dict[str, Any]:
        """Get model-specific state for serialization."""
        return {}
    
    def _set_model_state(self, state: Dict[str, Any]) -> None:
        """Set model-specific state from serialization."""
        pass
    
    def _validate_history(self, history: List[TrajectoryData]) -> None:
        """
        Validate trajectory history data.
        
        Args:
            history: Trajectory history to validate
            
        Raises:
            ValueError: If history is invalid
        """
        if not history:
            raise ValueError("Trajectory history cannot be empty")
        
        # Check temporal ordering
        for i in range(1, len(history)):
            if history[i].timestamp <= history[i-1].timestamp:
                raise ValueError("Trajectory history must be temporally ordered")
        
        # Check vehicle consistency
        vehicle_id = history[0].vehicle_id
        for point in history:
            if point.vehicle_id != vehicle_id:
                raise ValueError("All trajectory points must belong to same vehicle")
    
    def _extract_features(self, history: List[TrajectoryData]) -> np.ndarray:
        """
        Extract features from trajectory history.
        
        Args:
            history: Trajectory history
            
        Returns:
            Feature array for prediction
        """
        if len(history) == 0:
            return np.array([])
        
        # Basic features: position, velocity, acceleration
        features = []
        
        for i, point in enumerate(history):
            point_features = [point.x, point.y]
            
            # Add velocity if available
            if point.velocity is not None:
                point_features.append(point.velocity)
            else:
                # Estimate velocity from position differences
                if i > 0:
                    dt = (point.timestamp - history[i-1].timestamp).total_seconds()
                    if dt > 0:
                        dx = point.x - history[i-1].x
                        dy = point.y - history[i-1].y
                        velocity = np.sqrt(dx**2 + dy**2) / dt
                        point_features.append(velocity)
                    else:
                        point_features.append(0.0)
                else:
                    point_features.append(0.0)
            
            # Add acceleration if available
            if point.acceleration is not None:
                point_features.append(point.acceleration)
            else:
                # Could estimate from velocity changes
                point_features.append(0.0)
            
            features.append(point_features)
        
        return np.array(features)


class EnsemblePredictor(TrajectoryPredictor):
    """
    Base class for ensemble prediction models.
    
    Combines predictions from multiple models with uncertainty estimation.
    """
    
    def __init__(self, config: DictConfig, base_models: List[TrajectoryPredictor]):
        super().__init__(config, "EnsemblePredictor")
        self.base_models = base_models
        self.weights = None  # Model weights (learned during training)
    
    async def fit(
        self, 
        trajectories: List[TrajectoryData],
        validation_trajectories: Optional[List[TrajectoryData]] = None
    ) -> Dict[str, Any]:
        """Train all base models and learn ensemble weights."""
        training_results = {}
        
        # Train each base model
        for model in self.base_models:
            model_result = await model.fit(trajectories, validation_trajectories)
            training_results[model.name] = model_result
        
        # Learn ensemble weights (could be more sophisticated)
        self.weights = np.ones(len(self.base_models)) / len(self.base_models)
        
        self.is_trained = True
        return training_results
    
    async def predict(
        self,
        history: List[TrajectoryData], 
        prediction_horizon: float,
        **kwargs
    ) -> PredictionResult:
        """Generate ensemble prediction from base models."""
        if not self.is_trained:
            raise ValueError("Ensemble model must be trained before prediction")
        
        # Get predictions from all base models
        base_predictions = []
        for model in self.base_models:
            try:
                pred = await model.predict(history, prediction_horizon, **kwargs)
                base_predictions.append(pred)
            except Exception:
                # Skip failed predictions
                continue
        
        if not base_predictions:
            raise ValueError("All base models failed to generate predictions")
        
        # Combine predictions (weighted average)
        return self._combine_predictions(base_predictions)
    
    def _combine_predictions(self, predictions: List[PredictionResult]) -> PredictionResult:
        """Combine multiple predictions into ensemble result."""
        if not predictions:
            raise ValueError("No predictions to combine")
        
        # Use first prediction as template
        template = predictions[0]
        
        # Weighted average of coordinates
        weighted_x = np.zeros_like(template.x_coords)
        weighted_y = np.zeros_like(template.y_coords)
        
        total_weight = 0.0
        for i, pred in enumerate(predictions):
            if i < len(self.weights):
                weight = self.weights[i]
            else:
                weight = 1.0 / len(predictions)
            
            weighted_x += weight * pred.x_coords
            weighted_y += weight * pred.y_coords
            total_weight += weight
        
        # Normalize
        if total_weight > 0:
            weighted_x /= total_weight
            weighted_y /= total_weight
        
        # Estimate uncertainty from prediction variance
        x_variance = np.var([pred.x_coords for pred in predictions], axis=0)
        y_variance = np.var([pred.y_coords for pred in predictions], axis=0)
        
        return PredictionResult(
            timestamps=template.timestamps,
            x_coords=weighted_x,
            y_coords=weighted_y,
            x_std=np.sqrt(x_variance),
            y_std=np.sqrt(y_variance),
            prediction_horizon=template.prediction_horizon,
            model_name=self.name,
            metadata={
                "base_models": [model.name for model in self.base_models],
                "weights": self.weights.tolist() if self.weights is not None else None,
                "num_predictions": len(predictions)
            }
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get ensemble model information."""
        base_model_info = [model.get_model_info() for model in self.base_models]
        
        return {
            "name": self.name,
            "type": "ensemble",
            "base_models": base_model_info,
            "weights": self.weights.tolist() if self.weights is not None else None,
            "is_trained": self.is_trained
        }