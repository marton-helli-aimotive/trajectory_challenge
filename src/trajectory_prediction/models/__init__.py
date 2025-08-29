"""
Trajectory prediction models.

Implements 5 distinct prediction approaches:
1. Baseline models (Constant Velocity, Constant Acceleration)
2. Polynomial regression with physics-informed features  
3. K-Nearest Neighbors with trajectory similarity
4. Gaussian Process regression with spatial kernels
5. Tree-based ensemble (XGBoost) with trajectory features
"""

from .base import TrajectoryPredictor, PredictionResult
from .factory import ModelFactory
from .baseline import ConstantVelocityPredictor, ConstantAccelerationPredictor
from .polynomial import PolynomialTrajectoryPredictor
from .knn import KNNTrajectoryPredictor
from .gaussian_process import GaussianProcessPredictor
from .ensemble import EnsembleTrajectoryPredictor

__all__ = [
    "TrajectoryPredictor", 
    "PredictionResult",
    "ModelFactory",
    "ConstantVelocityPredictor",
    "ConstantAccelerationPredictor", 
    "PolynomialTrajectoryPredictor",
    "KNNTrajectoryPredictor",
    "GaussianProcessPredictor",
    "EnsembleTrajectoryPredictor"
]