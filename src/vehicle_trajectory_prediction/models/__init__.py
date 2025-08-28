"""Trajectory prediction models package."""

from .base import BaseTrajectoryPredictor, PredictionResult
from .baseline import ConstantVelocityPredictor, ConstantAccelerationPredictor
from .polynomial import PolynomialRegressionPredictor
from .knn import KNearestNeighborsPredictor
from .gaussian_process import GaussianProcessPredictor
from .tree_ensemble import TreeEnsemblePredictor
from .mixture_density import MixtureDensityPredictor
from .ensemble import (
    EnsemblePredictor, 
    EnsembleStrategy, 
    WeightedAverageStrategy, 
    VotingStrategy, 
    DynamicEnsembleStrategy
)

__all__ = [
    "BaseTrajectoryPredictor",
    "PredictionResult", 
    "ConstantVelocityPredictor",
    "ConstantAccelerationPredictor",
    "PolynomialRegressionPredictor",
    "KNearestNeighborsPredictor",
    "GaussianProcessPredictor",
    "TreeEnsemblePredictor",
    "MixtureDensityPredictor",
    "EnsemblePredictor",
    "EnsembleStrategy",
    "WeightedAverageStrategy",
    "VotingStrategy",
    "DynamicEnsembleStrategy"
]