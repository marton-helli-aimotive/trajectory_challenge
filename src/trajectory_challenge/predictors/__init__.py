"""Predictor registry and exports."""
from __future__ import annotations

from typing import Dict, Type

from .base import Predictor
from .baselines import ConstantVelocityPredictor, LinearKinematicsPredictor


_REGISTRY: Dict[str, Type[Predictor]] = {
    ConstantVelocityPredictor.name: ConstantVelocityPredictor,
    LinearKinematicsPredictor.name: LinearKinematicsPredictor,
}


def list_predictors() -> list[str]:
    return sorted(_REGISTRY.keys())


def get_predictor(name: str) -> Type[Predictor]:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown predictor: {name}")
    return _REGISTRY[name]


__all__ = [
    "Predictor",
    "list_predictors",
    "get_predictor",
    "ConstantVelocityPredictor",
    "LinearKinematicsPredictor",
]
