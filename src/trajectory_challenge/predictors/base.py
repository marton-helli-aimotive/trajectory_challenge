from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypedDict, Any
import numpy as np


@dataclass
class PredictionResult:
    # future time stamps (seconds from epoch or relative)
    t: np.ndarray  # shape (T,)
    # axes follow data convention: lateral (x), longitudinal (y)
    x: np.ndarray  # shape (T,)
    y: np.ndarray  # shape (T,)
    meta: dict[str, Any] | None = None


class History(TypedDict):
    t: np.ndarray  # (N,)
    x: np.ndarray  # (N,)
    y: np.ndarray  # (N,)


class Predictor(Protocol):
    """Minimal predictor interface for deterministic baselines."""

    # Unique name used in registry
    name: str

    def get_params(self) -> dict[str, Any]:
        ...

    def set_params(self, params: dict[str, Any]) -> None:
        ...

    def fit(self, hist: History) -> None:
        ...

    def predict(self, hist: History, horizon_s: float, dt_s: float) -> PredictionResult:
        ...
