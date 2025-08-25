from __future__ import annotations

# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownLambdaType=false, reportUnknownParameterType=false

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

from .base import History


@dataclass
class WindowSpec:
    track_id: str
    t0: float
    history_s: float
    horizon_s: float
    dt_s: float


def coerce_time_seconds(series: pd.Series) -> pd.Series:
    t = pd.to_numeric(series, errors="coerce")
    # Check if values look like milliseconds (Unix epoch ms are > 1e12, regular seconds are < 1e10)
    if t.max(skipna=True) > 1e10:
        t = t / 1000.0
    return t


def extract_track(df: pd.DataFrame, track_id: str, time_col: str, x_col: str, y_col: str) -> pd.DataFrame:
    g = df[df["track_id"] == track_id].copy()
    g[time_col] = coerce_time_seconds(g[time_col])
    g = g.dropna(subset=[time_col, x_col, y_col]).sort_values(time_col)
    return g


def slice_history_future(g: pd.DataFrame, spec: WindowSpec, time_col: str, x_col: str, y_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    t0 = float(spec.t0)
    hist = g[(g[time_col] >= t0 - float(spec.history_s)) & (g[time_col] <= t0)].copy()
    fut = g[(g[time_col] > t0) & (g[time_col] <= t0 + float(spec.horizon_s))].copy()
    return hist, fut


def resample_uniform(df: pd.DataFrame, time_col: str, x_col: str, y_col: str, dt_s: float) -> History:
    t = df[time_col].to_numpy(dtype=float)
    x = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
    if t.size == 0:
        return {"t": t, "x": x, "y": y}
    t0, t1 = float(t.min()), float(t.max())
    n = max(1, int(np.floor((t1 - t0) / max(dt_s, 1e-6))) + 1)
    tu = np.linspace(t0, t0 + (n - 1) * dt_s, n)
    xu = np.interp(tu, t, x)
    yu = np.interp(tu, t, y)
    return {"t": tu, "x": xu, "y": yu}
