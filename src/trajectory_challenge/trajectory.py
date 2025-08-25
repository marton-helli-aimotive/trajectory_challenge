"""Trajectory processing and transformation functions.

Public API
- filter_by_vehicle(df, vehicle_ids) -> DataFrame
- filter_by_frame_range(df, start=None, end=None) -> DataFrame
- filter_by_lane(df, lanes) -> DataFrame
- detect_trajectory_segments(veh_data, time_threshold=20.0) -> list[DataFrame]
- generate_track_ids(df, time_threshold=20.0) -> DataFrame
- prepare_animation_data(df, column_mapping) -> DataFrame
- interpolate_position(vehicle_data, target_time, column_mapping) -> tuple[float, float] | None
- get_vehicle_trail(vehicle_data, current_time, trail_length, column_mapping) -> dict[str, list[float]]
"""
from __future__ import annotations

from typing import Any, Iterable, cast

import numpy as np
import pandas as pd


def filter_by_vehicle(df: pd.DataFrame, vehicle_ids: int | Iterable[int]) -> pd.DataFrame:
    if isinstance(vehicle_ids, int):
        vehicle_ids = [vehicle_ids]
    return df[df["vehicle_id"].isin(vehicle_ids)].copy()


def filter_by_frame_range(df: pd.DataFrame, start: int | None = None, end: int | None = None) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    if start is not None:
        mask &= df["frame_id"] >= start
    if end is not None:
        mask &= df["frame_id"] <= end
    return df[mask].copy()


def filter_by_lane(df: pd.DataFrame, lanes: int | Iterable[int]) -> pd.DataFrame:
    if isinstance(lanes, int):
        lanes = [lanes]
    return df[df["lane_id"].isin(lanes)].copy()


def detect_trajectory_segments(
    veh_data: pd.DataFrame,
    time_threshold: float = 20.0,
    *,
    # advanced controls (defaults per approved design)
    bridge_gap: float | None = 5.0,
    small_gap_floor: float = 0.25,
    s_jump: float = 300.0,
    v_max: float = 180.0,
    s_stop: float = 7.5,
    min_points: int = 4,
    min_duration: float = 1.5,
    merge_gap: float = 8.0,
    merge_dist: float = 80.0,
    merge_lane: int = 1,
) -> list[pd.DataFrame]:
    """Detect trajectory segments for a single vehicle using robust rules.

    - Prefer global_time (ms) else infer seconds from frame_id.
    - Hysteresis around small gaps, bridging up to bridge_gap when spatially plausible.
    - Hard split beyond time_threshold unless the vehicle is stationary.
    - Single-pass post-merge of micro segments.
    """
    if len(veh_data) < 2:
        return [veh_data]

    has_time = "global_time" in veh_data.columns or "frame_id" in veh_data.columns
    if not has_time:
        return [veh_data]

    df = veh_data.copy()
    # Build t_sec (seconds)
    if "global_time" in df.columns:
        t_sec = pd.to_numeric(df["global_time"], errors="coerce") / 1000.0  # type: ignore[call-overload]
    else:
        fr = pd.to_numeric(df["frame_id"], errors="coerce")  # type: ignore[call-overload]
        dfr = fr.sort_values().diff()
        pos = dfr[dfr > 0]  # type: ignore[operator]
        base = float(pos.median()) if not pos.empty else 1.0  # type: ignore[union-attr]
        period = 0.1 if base == 0 or np.isnan(base) else base * 0.1
        t_sec = fr * period
    df = df.assign(t_sec=t_sec)
    df = df.sort_values("t_sec").reset_index()

    # Deltas and features
    dt = df["t_sec"].diff()
    has_xy = ("local_x" in df.columns) and ("local_y" in df.columns)
    if has_xy:
        x = pd.to_numeric(df["local_x"], errors="coerce")  # type: ignore[call-overload]
        y = pd.to_numeric(df["local_y"], errors="coerce")  # type: ignore[call-overload]
        dx = x.diff()
        dy = y.diff()
        d = np.sqrt((dx.fillna(0) ** 2) + (dy.fillna(0) ** 2))  # type: ignore[call-overload]
        with np.errstate(divide="ignore", invalid="ignore"):
            inst_v = d / dt
    else:
        d = pd.Series(0.0, index=df.index)
        inst_v = pd.Series(np.nan, index=df.index)

    has_lane = "lane_id" in df.columns
    if has_lane:
        lane_vals = pd.to_numeric(df["lane_id"], errors="coerce")  # type: ignore[call-overload]
        lane_change = lane_vals.diff().abs().fillna(0)  # type: ignore[call-overload]
    else:
        lane_change = pd.Series(0.0, index=df.index)

    # Dynamic small_gap from median positive dt
    dt_pos = dt[dt > 0]  # type: ignore[operator]
    dt_base = float(dt_pos.median()) if not dt_pos.empty else 0.1  # type: ignore[union-attr]
    small_gap = max(2.0 * dt_base, small_gap_floor)
    b_gap = bridge_gap if bridge_gap is not None else 3.0

    # Splitting logic
    split_flags = [True]
    for i in range(1, len(df)):
        dti = float(dt.iat[i]) if pd.notna(dt.iat[i]) else 0.0  # type: ignore[arg-type]
        di = float(d.iat[i]) if pd.notna(d.iat[i]) else 0.0  # type: ignore[arg-type]
        vi = float(inst_v.iat[i]) if pd.notna(inst_v.iat[i]) else np.nan  # type: ignore[arg-type]
        lci = float(lane_change.iat[i]) if pd.notna(lane_change.iat[i]) else 0.0  # type: ignore[arg-type]

        do_split = False
        if dti <= small_gap:
            do_split = False
        elif dti <= b_gap:
            # Prefer to keep continuity across moderate gaps unless there's a clear implausibility.
            if has_xy and ((di > 2 * s_jump) or (not np.isnan(vi) and vi > 1.25 * v_max)):
                do_split = True
        elif dti <= time_threshold:
            if has_xy and ((di > s_jump) or (not np.isnan(vi) and vi > v_max) or (lci > 2)):
                do_split = True
            else:
                if has_xy and di <= s_stop:
                    do_split = False
                else:
                    do_split = False
        else:
            if has_xy and di <= s_stop:
                do_split = False
            else:
                do_split = True
        split_flags.append(do_split)

    # Build initial segments as index ranges
    seg_ranges: list[tuple[int, int]] = []
    start = 0
    for i in range(1, len(df)):
        if split_flags[i]:
            seg_ranges.append((start, i))
            start = i
    seg_ranges.append((start, len(df)))

    def seg_duration(a: int, b: int) -> float:
        return float(df["t_sec"].iat[b - 1] - df["t_sec"].iat[a]) if b - a > 1 else 0.0

    def seg_end_start_gap(prev: tuple[int, int], nxt: tuple[int, int]) -> float:
        return float(df["t_sec"].iat[nxt[0]] - df["t_sec"].iat[prev[1] - 1])

    def seg_end_start_dist(prev: tuple[int, int], nxt: tuple[int, int]) -> float:
        if not has_xy:
            return 0.0
        x1, y1 = float(x.iat[prev[1] - 1]), float(y.iat[prev[1] - 1])
        x2, y2 = float(x.iat[nxt[0]]), float(y.iat[nxt[0]])
        return float(np.hypot(x2 - x1, y2 - y1))

    def seg_lane_jump(prev: tuple[int, int], nxt: tuple[int, int]) -> float:
        if not has_lane:
            return 0.0
        a = pd.to_numeric(df["lane_id"].iat[prev[1] - 1], errors="coerce")
        b = pd.to_numeric(df["lane_id"].iat[nxt[0]], errors="coerce")
        if pd.isna(a) or pd.isna(b):
            return 0.0
        return float(abs(b - a))

    # Single-pass merge for micro segments
    merged: list[tuple[int, int]] = []
    i = 0
    while i < len(seg_ranges):
        cur = seg_ranges[i]
        pts = cur[1] - cur[0]
        dur = seg_duration(*cur)
        if (pts < min_points) or (dur < min_duration):
            merged_flag = False
            # try merge with previous
            if merged:
                prev = merged[-1]
                gap = seg_end_start_gap(prev, cur)
                dist = seg_end_start_dist(prev, cur)
                lj = seg_lane_jump(prev, cur)
                if gap <= merge_gap and (not has_xy or dist <= merge_dist) and (lj <= merge_lane):
                    merged[-1] = (prev[0], cur[1])
                    merged_flag = True
            # else try merge with next
            if (not merged_flag) and (i + 1 < len(seg_ranges)):
                nxt = seg_ranges[i + 1]
                gap = seg_end_start_gap(cur, nxt)
                dist = seg_end_start_dist(cur, nxt)
                lj = seg_lane_jump(cur, nxt)
                if gap <= merge_gap and (not has_xy or dist <= merge_dist) and (lj <= merge_lane):
                    # merge cur with next by advancing i and pushing combined
                    merged.append((cur[0], nxt[1]))
                    i += 2  # consume next as well
                    continue
            if not merged_flag:
                merged.append(cur)
                i += 1
                continue
            # if merged with previous, just advance
            i += 1
        else:
            merged.append(cur)
            i += 1

    # Build DataFrames for segments with >=2 points
    segments: list[pd.DataFrame] = []
    for a, b in merged:
        if b - a >= 2:
            idx_slice = df.loc[a:b - 1, "index"].values
            segments.append(veh_data.loc[idx_slice].copy())
    if not segments:
        return [veh_data]
    return segments


def generate_track_ids(
    df: pd.DataFrame,
    time_threshold: float = 20.0,
    **advanced_params: Any,
) -> pd.DataFrame:
    """Add a 'track_id' for each continuous segment (time-only segmentation).

    Input: DataFrame with at least vehicle_id and global_time/frame_id.
    Output: Copy with 'track_id' string column like 'V123_T1'.
    """
    out = df.copy()
    out["track_id"] = ""
    # Group by vehicle_id, skipping NaN IDs to avoid unassignable rows
    for vehicle_id, veh in out.groupby("vehicle_id", dropna=True, sort=False):
        if len(veh) < 2:
            out.loc[veh.index, "track_id"] = f"V{vehicle_id}_T1"
            continue
        segments = detect_trajectory_segments(veh, time_threshold, **advanced_params)
        for idx, seg in enumerate(segments):
            if len(seg) > 0:
                out.loc[seg.index, "track_id"] = f"V{vehicle_id}_T{idx+1}"
    # Fill any remaining empty track IDs (e.g., if segmentation produced no segments)
    empty_mask = (out["track_id"].astype(str).str.len() == 0) & out["vehicle_id"].notna()
    if empty_mask.any():
        for vid, idxs in out[empty_mask].groupby("vehicle_id").groups.items():
            out.loc[idxs, "track_id"] = f"V{vid}_T1"
    return out


def prepare_animation_data(df: pd.DataFrame, column_mapping: dict) -> pd.DataFrame:
    """Prepare cleaned numeric subset for animation components."""
    x_col = column_mapping["x"]
    y_col = column_mapping["y"]
    time_col = column_mapping["time"]
    vehicle_col = column_mapping["vehicle"]

    df_clean = df.copy()
    df_clean[x_col] = pd.to_numeric(df_clean[x_col], errors="coerce")
    df_clean[y_col] = pd.to_numeric(df_clean[y_col], errors="coerce")
    df_clean[time_col] = pd.to_numeric(df_clean[time_col], errors="coerce")
    df_clean[vehicle_col] = pd.to_numeric(df_clean[vehicle_col], errors="coerce")
    df_clean = df_clean.dropna(subset=[x_col, y_col, time_col, vehicle_col])
    if time_col == "global_time" and df_clean[time_col].max() > 10000:
        df_clean[time_col] = df_clean[time_col] / 1000.0
    return df_clean.sort_values([time_col, vehicle_col])


def interpolate_position(vehicle_data: pd.DataFrame, target_time: float, column_mapping: dict):
    """Interpolate a vehicle's (x,y) at target_time or return None if out of range."""
    if len(vehicle_data) == 0:
        return None
    x_col = column_mapping["x"]
    y_col = column_mapping["y"]
    time_col = column_mapping["time"]

    times = vehicle_data[time_col].values
    x_positions = vehicle_data[x_col].values
    y_positions = vehicle_data[y_col].values

    if target_time < times.min() or target_time > times.max():
        return None

    exact = vehicle_data[vehicle_data[time_col] == target_time]
    if not exact.empty:
        return (exact[x_col].iloc[0], exact[y_col].iloc[0])

    x_interp = np.interp(target_time, times, x_positions)
    y_interp = np.interp(target_time, times, y_positions)
    return (x_interp, y_interp)


def get_vehicle_trail(vehicle_data: pd.DataFrame, current_time: float, trail_length: int, column_mapping: dict):
    """Return recent trail coordinates up to current_time."""
    x_col = column_mapping["x"]
    y_col = column_mapping["y"]
    time_col = column_mapping["time"]
    recent = vehicle_data[vehicle_data[time_col] <= current_time].tail(trail_length)
    return {"x": recent[x_col].tolist(), "y": recent[y_col].tolist()}


__all__ = [
    "filter_by_vehicle",
    "filter_by_frame_range",
    "filter_by_lane",
    "detect_trajectory_segments",
    "generate_track_ids",
    "build_trajectories",
    "prepare_animation_data",
    "interpolate_position",
    "get_vehicle_trail",
]


def build_trajectories(
    df: pd.DataFrame,
    time_threshold: float = 20.0,
    **advanced_params: Any,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (df_with_track_ids, trajectory_table) ready for GUI consumption.

    - Annotates input with 'track_id' using generate_track_ids.
    - Builds a one-row-per-trajectory summary including frame/time spans and counts.

    Columns in the summary follow the existing GUI convention:
      ['track_id', 'vehicle_id', 'frame_id_min', 'frame_id_max', 'frame_id_count',
       'global_time_min', 'global_time_max'] when those source columns exist.

    If required columns are missing, returns empty summary.
    """
    if df.empty:
        return df.copy(), pd.DataFrame()

    tracks = generate_track_ids(df, time_threshold=time_threshold, **advanced_params)

    # If no track_id could be produced, return early with empty summary.
    if ("track_id" not in tracks.columns) or tracks["track_id"].astype(str).str.strip().eq("").all():
        return tracks, pd.DataFrame()

    # Build aggregation similar to the GUI's previous logic
    cols: dict[str, list[str]] = {}
    if "frame_id" in tracks.columns:
        cols["frame_id"] = ["min", "max"]
    if "global_time" in tracks.columns:
        cols["global_time"] = ["min", "max"]

    if not cols:
        # Fallback: just counts per track and vehicle
        summary = (
            tracks.groupby(["track_id", "vehicle_id"], as_index=False)
            .size()
            .rename(columns={"size": "count"})
        )
        return tracks, summary

    summary = tracks.groupby(["track_id", "vehicle_id"], as_index=False).agg(cast(Any, cols))  # type: ignore[reportGeneralTypeIssues]
    summary = cast(pd.DataFrame, summary)

    # Flatten MultiIndex columns
    flat_cols: list[str] = []
    for col in list(summary.columns.values):  # type: ignore[reportUnknownMemberType]
        if isinstance(col, tuple):
            flat_cols.append("_".join([str(p) for p in col if p]))  # type: ignore[reportGeneralTypeIssues]
        else:
            flat_cols.append(str(col))  # type: ignore[reportGeneralTypeIssues]
    summary.columns = flat_cols

    # For convenience, add a distinct frame_id_count if frame_id present
    if "frame_id" in tracks.columns:
        frame_numeric = cast(Any, pd.to_numeric(tracks["frame_id"], errors="coerce"))  # type: ignore[reportUnknownMemberType]
        frame_counts = (
            frame_numeric.groupby(tracks["track_id"])  # type: ignore[arg-type]
            .nunique()
            .rename("frame_id_count")
        )
        summary = summary.merge(frame_counts.reset_index(), on="track_id", how="left")

    # Reorder to keep identifiers first
    id_cols = [c for c in ["track_id", "vehicle_id"] if c in summary.columns]
    other_cols = [c for c in summary.columns if c not in id_cols]
    summary = summary[id_cols + other_cols]
    return tracks, summary
