"""Machine learning models for trajectory prediction."""

from .base import (
    FeatureBasedPredictor,
    ModelConfig,
    ModelPerformance,
    PredictionResult,
    TrajectoryPredictor,
)
from .baseline import ConstantAccelerationPredictor, ConstantVelocityPredictor
from .factory import ModelFactory, ModelRegistry, create_model, get_available_models
from .knn import KNearestNeighborsPredictor
from .polynomial import PolynomialRegressionPredictor

__all__ = [
    # Base classes and types
    "TrajectoryPredictor",
    "FeatureBasedPredictor",
    "ModelConfig",
    "PredictionResult",
    "ModelPerformance",
    # Model implementations
    "ConstantVelocityPredictor",
    "ConstantAccelerationPredictor",
    "PolynomialRegressionPredictor",
    "KNearestNeighborsPredictor",
    # Factory and utilities
    "ModelFactory",
    "ModelRegistry",
    "create_model",
    "get_available_models",
]
