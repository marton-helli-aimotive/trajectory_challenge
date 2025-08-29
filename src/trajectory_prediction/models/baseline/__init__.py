"""Baseline trajectory prediction models."""

from .constant_velocity import ConstantVelocityPredictor
from .constant_acceleration import ConstantAccelerationPredictor

__all__ = ["ConstantVelocityPredictor", "ConstantAccelerationPredictor"]