"""Machine learning models for trajectory prediction."""

from .base import (
    FeatureBasedPredictor,
    ModelConfig,
    ModelPerformance,
    PredictionResult,
    TrajectoryPredictor,
)
from .baseline import ConstantAccelerationPredictor, ConstantVelocityPredictor
from .ensemble import EnsemblePredictor
from .factory import ModelFactory, ModelRegistry, create_model, get_available_models
from .gaussian_process import GaussianProcessPredictor
from .knn import KNearestNeighborsPredictor
from .mixture_density import SimplifiedMDNPredictor
from .polynomial import PolynomialRegressionPredictor
from .tree_ensemble import TreeEnsemblePredictor

__all__ = [
    # Base classes and types
    "TrajectoryPredictor",
    "FeatureBasedPredictor",
    "ModelConfig",
    "PredictionResult",
    "ModelPerformance",
    # Baseline models
    "ConstantVelocityPredictor",
    "ConstantAccelerationPredictor",
    # Classical ML models
    "PolynomialRegressionPredictor",
    "KNearestNeighborsPredictor",
    # Advanced ML models with uncertainty quantification
    "GaussianProcessPredictor",
    "TreeEnsemblePredictor",
    "SimplifiedMDNPredictor",
    # Ensemble methods
    "EnsemblePredictor",
    # Factory and utilities
    "ModelFactory",
    "ModelRegistry",
    "create_model",
    "get_available_models",
]
