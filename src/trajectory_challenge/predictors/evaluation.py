from __future__ import annotations

import numpy as np


def rmse(e: np.ndarray) -> float:
    e = np.asarray(e, dtype=float)
    return float(np.sqrt(np.mean(np.square(e))))


def mae(e: np.ndarray) -> float:
    e = np.asarray(e, dtype=float)
    return float(np.mean(np.abs(e)))


def ade(dx: np.ndarray, dy: np.ndarray) -> float:
    dx = np.asarray(dx, dtype=float)
    dy = np.asarray(dy, dtype=float)
    d = np.hypot(dx, dy)
    return float(np.mean(d))


def fde(dx: np.ndarray, dy: np.ndarray) -> float:
    dx = np.asarray(dx, dtype=float)
    dy = np.asarray(dy, dtype=float)
    d = np.hypot(dx, dy)
    return float(d[-1]) if d.size > 0 else float("nan")
