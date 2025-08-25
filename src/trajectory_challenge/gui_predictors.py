from __future__ import annotations

# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit as st

from .trajectory import build_trajectories
from .predictors import list_predictors, get_predictor
from .predictors.features import WindowSpec, extract_track, slice_history_future, resample_uniform, coerce_time_seconds
from .predictors.evaluation import rmse, ade, fde


def _column_mapping() -> dict[str, str]:
    return {
        "x": "local_x",
        "y": "local_y",
        "time": "global_time",
        "vehicle": "vehicle_id",
        "frame": "frame_id",
        "lane": "lane_id",
    }


@st.cache_data(show_spinner=False, hash_funcs={pd.DataFrame: lambda _df: None})
def _cached_tracks(df: pd.DataFrame, dataset_id: str, time_threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    _ = dataset_id
    return build_trajectories(df, time_threshold=time_threshold)


def run(df: pd.DataFrame) -> None:
    st.subheader("Predictor Frameworks")
    mapping = _column_mapping()

    # Ensure trajectories exist and pick a track
    dsid = st.session_state.get("dataset_id", "dataset")
    tsec = st.number_input(
        "Segmentation time gap (s)",
        min_value=1.0,
        max_value=180.0,
        value=60.0,
        step=1.0,
        key="pred_seg_time_gap",
    )
    df_tracks, summary = _cached_tracks(df, dataset_id=str(dsid), time_threshold=float(tsec))
    if summary.empty or "track_id" not in df_tracks.columns:
        st.info("No trajectories available.")
        return

    # Filters similar to Trajectory List
    cfa, cfb = st.columns(2)
    with cfa:
        min_frames = st.number_input(
            "Min frames", min_value=0, max_value=1000, value=20, step=5, key="pred_min_frames"
        )
    with cfb:
        top_n = st.selectbox("Top N", options=[100, 250, 500, 1000], index=2, key="pred_top_n")

    filtered = summary.copy()
    if "frame_id_count" in filtered.columns:
        filtered["frame_id_count"] = pd.to_numeric(filtered["frame_id_count"], errors="coerce")
        filtered = filtered[filtered["frame_id_count"] >= float(min_frames)]
        filtered = filtered.sort_values("frame_id_count", ascending=False)
    if filtered.empty:
        st.caption("No tracks matched the filter; showing all tracks instead.")
        filtered = summary.copy()
    track_options = [t for t in filtered["track_id"].dropna().astype(str).tolist() if t.strip() != ""][: int(top_n)]
    if not track_options:
        # Fallback to any available track ids from df_tracks
        fallback = [t for t in df_tracks.get("track_id", pd.Series(dtype=str)).dropna().astype(str).unique().tolist() if t.strip() != ""]
        track_options = fallback[: int(top_n)]
    if not track_options:
        st.info("No trajectories available with the current filters. Try lowering Min frames or increasing the segmentation gap.")
        return
    track_id = st.selectbox("Track", options=track_options, key="pred_track_select")

    # Window spec controls
    time_col, x_col, y_col = mapping["time"], mapping["x"], mapping["y"]  # keep data axes; swap only in plots
    g = extract_track(df_tracks, track_id, time_col, x_col, y_col)
    if g.empty:
        st.warning("Selected track has no data after cleaning.")
        return

    # Debug: show raw time values before and after coercion
    raw_time = g[time_col]
    t = coerce_time_seconds(raw_time)
    
    # Additional debug: check if this matches what trajectory list would show
    try:
        orig_raw = df[df["vehicle_id"] == g["vehicle_id"].iloc[0]][time_col] if len(g) > 0 else pd.Series(dtype=float)
        orig_raw_numeric = pd.to_numeric(orig_raw, errors="coerce")
        raw_time_numeric = pd.to_numeric(raw_time, errors="coerce")
        
        st.caption(f"Original data time range: {orig_raw_numeric.min():.0f} to {orig_raw_numeric.max():.0f}")
        st.caption(f"Track data time range: {raw_time_numeric.min():.0f} to {raw_time_numeric.max():.0f}")
        st.caption(f"Converted time range: {t.min():.1f}s to {t.max():.1f}s")
    except Exception as e:
        st.caption(f"Debug error: {e}")
        st.caption(f"Time column type: {type(raw_time.iloc[0]) if len(raw_time) > 0 else 'empty'}")
        st.caption(f"Sample time values: {raw_time.head(3).tolist()}")
    
    t_np = t.to_numpy(dtype=float)
    t_min = float(np.nanmin(t_np)) if t_np.size else 0.0
    t_max = float(np.nanmax(t_np)) if t_np.size else 0.0
    track_duration = t_max - t_min
    
    # Show track info for debugging
    st.caption(f"Track {track_id}: {len(g)} points, {track_duration:.1f}s duration ({t_min:.1f}s to {t_max:.1f}s)")
    
    # Use adaptive defaults based on track length
    max_hist = max(0.5, min(3.0, track_duration * 0.3))
    max_horiz = max(0.5, min(3.0, track_duration * 0.3))
    
    c1, c2, c3, c4 = st.columns(4)
    with c2:
        hist_s = st.number_input(
            "History (s)", 
            min_value=0.5, 
            max_value=float(max_hist), 
            value=min(2.0, float(max_hist)), 
            step=0.1, 
            key="pred_history"
        )
    with c3:
        horizon_s = st.number_input(
            "Horizon (s)", 
            min_value=0.5, 
            max_value=float(max_horiz), 
            value=min(2.0, float(max_horiz)), 
            step=0.1, 
            key="pred_horizon"
        )
    with c4:
        dt_s = st.number_input("dt (s)", min_value=0.05, max_value=1.0, value=0.2, step=0.05, key="pred_dt")

    # Compute a feasible range for t0 given history and horizon
    t0_lo = t_min + float(hist_s)
    t0_hi = t_max - float(horizon_s)
    
    if track_duration < 2.0:
        st.warning(f"Track too short ({track_duration:.1f}s). Try a longer track from the filter.")
        return
    
    if t0_hi < t0_lo:
        needed = hist_s + horizon_s
        st.warning(f"History+Horizon ({needed:.1f}s) exceeds track duration ({track_duration:.1f}s). Reduce values or pick a longer track.")
        return
        
    t0_default = float(np.clip(np.nanmedian(t_np), t0_lo, t0_hi))
    with c1:
        t0 = st.slider(
            "t0 (s)",
            min_value=float(np.round(t0_lo, 1)),
            max_value=float(np.round(t0_hi, 1)),
            value=float(np.round(t0_default, 1)),
            step=0.1,
            key="pred_t0",
        )

    spec = WindowSpec(track_id=str(track_id), t0=float(t0), history_s=float(hist_s), horizon_s=float(horizon_s), dt_s=float(dt_s))

    hist_df, fut_df = slice_history_future(g, spec, time_col, x_col, y_col)
    hist = resample_uniform(hist_df, time_col, x_col, y_col, dt_s=spec.dt_s)
    fut = resample_uniform(fut_df, time_col, x_col, y_col, dt_s=spec.dt_s)
    if len(hist["t"]) < 2 or len(fut["t"]) == 0:
        st.info("Insufficient history or future in the chosen window. Try moving t0 slider, increasing history/horizon, or choosing another track.")
        return

    # Predictor selection and params
    pred_name = st.selectbox("Predictor", options=list_predictors(), index=0, key="pred_predictor_name")
    PredClass = get_predictor(pred_name)
    predictor = PredClass()
    with st.expander("Predictor params", expanded=False):
        params: dict[str, object] = {}
        if hasattr(predictor, "get_params"):
            try:
                params = dict(predictor.get_params())  # type: ignore[assignment]
            except Exception:
                params = {}
        for k, v in list(params.items()):
            if isinstance(v, int):
                params[k] = st.number_input(k, value=int(v), key=f"pred_param_{k}_int")  # type: ignore[call-overload]
            elif isinstance(v, float):
                params[k] = st.number_input(k, value=float(v), key=f"pred_param_{k}_float")  # type: ignore[call-overload]
            else:
                st.write(f"{k}: {v}")
        if params:
            predictor.set_params(params)  # type: ignore[arg-type]

    predictor.fit(hist)
    pred = predictor.predict(hist, horizon_s=spec.horizon_s, dt_s=spec.dt_s)

    # Align prediction and future by time for metrics
    # Simple nearest-neighbor alignment on timestamps
    tf = fut["t"]
    idx = np.searchsorted(pred.t, tf)
    idx = np.clip(idx, 0, len(pred.t) - 1)
    dx = pred.x[idx] - fut["x"]
    dy = pred.y[idx] - fut["y"]

    # Metrics
    colm1, colm2, colm3, colm4 = st.columns(4)
    with colm1:
        st.metric("RMSE-x (ft)", f"{rmse(dx):.2f}")
    with colm2:
        st.metric("RMSE-y (ft)", f"{rmse(dy):.2f}")
    with colm3:
        st.metric("ADE (ft)", f"{ade(dx, dy):.2f}")
    with colm4:
        st.metric("FDE (ft)", f"{fde(dx, dy):.2f}")

    # Plot: swap axes for visual convention (x <- longitudinal, y <- lateral)
    fig = go.Figure()
    fig.update_layout(template="plotly_white", height=520)
    # History in gray
    fig.add_trace(go.Scatter(x=hist["y"], y=hist["x"], mode="lines", name="history", line=dict(color="#888", width=2)))
    # Future GT in green
    fig.add_trace(go.Scatter(x=fut["y"], y=fut["x"], mode="lines+markers", name="future (gt)", line=dict(color="#2ca02c", width=2), marker=dict(size=4)))
    # Prediction in orange
    fig.add_trace(go.Scatter(x=pred.y, y=pred.x, mode="lines+markers", name="prediction", line=dict(color="#ff7f0e", width=2, dash="dash"), marker=dict(size=4)))
    fig.update_xaxes(title_text="Longitudinal (ft)")
    fig.update_yaxes(title_text="Lateral (ft)", scaleanchor="x", scaleratio=1)
    st.plotly_chart(fig, use_container_width=True)
