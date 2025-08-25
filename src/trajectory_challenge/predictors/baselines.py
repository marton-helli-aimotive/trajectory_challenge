from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .base import Predictor, History, PredictionResult


@dataclass
class _BaseParams:
    velocity_window: int = 5  # samples for velocity estimation


class ConstantVelocityPredictor(Predictor):
    name = "Constant Velocity"

    def __init__(self, velocity_window: int = 5):
        self.params = _BaseParams(velocity_window=velocity_window)

    def get_params(self) -> dict[str, int]:
        return {"velocity_window": int(self.params.velocity_window)}

    def set_params(self, params: dict[str, int]) -> None:
        if "velocity_window" in params:
            self.params.velocity_window = int(params["velocity_window"])

    def fit(self, hist: History) -> None:  # no-op for baseline
        return None

    def predict(self, hist: History, horizon_s: float, dt_s: float) -> PredictionResult:
        t = np.asarray(hist["t"], dtype=float)  # (N,)
        x = np.asarray(hist["x"], dtype=float)  # lateral
        y = np.asarray(hist["y"], dtype=float)  # longitudinal
        assert t.ndim == x.ndim == y.ndim == 1 and len(t) >= 2

        # Estimate velocity with a robust window
        k = max(1, min(max(2, int(self.params.velocity_window)), len(t) - 1))
        dt = (t[-k:] - t[-k - 1 : -1]).astype(float)
        dt[dt == 0] = np.finfo(float).eps
        vx = np.median((x[-k:] - x[-k - 1 : -1]) / dt)
        vy = np.median((y[-k:] - y[-k - 1 : -1]) / dt)

        # Future times
        T = max(1, int(np.round(horizon_s / max(dt_s, 1e-6))))
        t0 = float(t[-1])
        tf = t0 + np.arange(1, T + 1) * float(dt_s)

        xf = x[-1] + vx * (tf - t0)
        yf = y[-1] + vy * (tf - t0)
        return PredictionResult(t=tf, x=xf, y=yf, meta={"vx": float(vx), "vy": float(vy)})


class LinearKinematicsPredictor(Predictor):
    name = "Linear Kinematics (deg1)"

    def __init__(self, window: int = 12):
        self.window = int(window)

    def get_params(self) -> dict[str, int]:
        return {"window": int(self.window)}

    def set_params(self, params: dict[str, int]) -> None:
        if "window" in params:
            self.window = int(params["window"])

    def fit(self, hist: History) -> None:  # no-op
        return None

    def predict(self, hist: History, horizon_s: float, dt_s: float) -> PredictionResult:
        t = np.asarray(hist["t"], dtype=float)  # (N,)
        x = np.asarray(hist["x"], dtype=float)  # lateral
        y = np.asarray(hist["y"], dtype=float)  # longitudinal
        assert t.ndim == x.ndim == y.ndim == 1 and len(t) >= 2

        n = len(t)
        w = max(2, min(int(self.window), n))
        tw = t[-w:]
        xw = x[-w:]
        yw = y[-w:]
        # Fit degree-1 per axis
        cx: np.ndarray = np.asarray(np.polyfit(tw - tw[0], xw, deg=1), dtype=float)  # type: ignore[no-untyped-call]
        cy: np.ndarray = np.asarray(np.polyfit(tw - tw[0], yw, deg=1), dtype=float)  # type: ignore[no-untyped-call]

        T = max(1, int(np.round(horizon_s / max(dt_s, 1e-6))))
        t0 = float(t[-1])
        tf = t0 + np.arange(1, T + 1) * float(dt_s)
        dt_rel = tf - tw[0]

        xf: np.ndarray = np.asarray(np.polyval(cx, dt_rel), dtype=float)  # type: ignore[no-untyped-call]
        yf: np.ndarray = np.asarray(np.polyval(cy, dt_rel), dtype=float)  # type: ignore[no-untyped-call]
        return PredictionResult(t=tf, x=xf, y=yf, meta={"coef_x": [float(v) for v in cx.tolist()], "coef_y": [float(v) for v in cy.tolist()]})
