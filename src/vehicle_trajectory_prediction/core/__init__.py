"""Core functionality for the vehicle trajectory prediction system."""

from .config import get_config, Config
from .logging import setup_logging, get_logger
from .exceptions import (
    TrajectoryPredictionError,
    DataValidationError,
    ModelError,
    ConfigurationError,
)

__all__ = [
    "get_config",
    "Config",
    "setup_logging",
    "get_logger",
    "TrajectoryPredictionError",
    "DataValidationError",
    "ModelError",
    "ConfigurationError",
]