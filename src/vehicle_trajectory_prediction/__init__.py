"""Vehicle Trajectory Prediction System.

A comprehensive system for predicting vehicle trajectories using advanced ML techniques.
"""

__version__ = "0.1.0"
__author__ = "ML Engineer"
__email__ = "engineer@example.com"

from .core.config import get_config
from .core.logging import setup_logging
from .core.exceptions import (
    TrajectoryPredictionError,
    DataValidationError,
    ModelError,
    ConfigurationError,
)

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "get_config",
    "setup_logging",
    "TrajectoryPredictionError",
    "DataValidationError",
    "ModelError",
    "ConfigurationError",
]