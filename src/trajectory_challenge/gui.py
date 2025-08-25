"""Streamlined Streamlit GUI (three tabs): Data Explorer, Trajectory List, Road Visualization."""
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownLambdaType=false, reportUntypedFunctionDecorator=false
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Support both module and script execution
try:
    from . import ngsim
    from .trajectory import build_trajectories
    from .utils import canonicalize_columns
except Exception:  # pragma: no cover
    # Running as a script: ensure project root (parent of 'src') is on sys.path
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    import trajectory_challenge.ngsim as ngsim
    from trajectory_challenge.trajectory import build_trajectories
    from trajectory_challenge.utils import canonicalize_columns


st.set_page_config(page_title="Trajectory Explorer", layout="wide")


@st.cache_data(show_spinner=False)
def load_data(location: str, limit: int) -> pd.DataFrame:
    where = f"location='{location}'"
    df = ngsim.load_ngsim_portal(limit=limit, where=where, json_endpoint=True)
    return canonicalize_columns(df)


@st.cache_data(show_spinner=False)
def load_data_cached(cache_dir: str, force_refresh: bool) -> pd.DataFrame:
    """Load the full cached dataset from disk or portal cache."""
    df = ngsim.load_ngsim_cached(cache_dir=cache_dir, force_refresh=force_refresh)
    return canonicalize_columns(df)


@st.cache_data(show_spinner=False, hash_funcs={pd.DataFrame: lambda _df: None})
def build_trajectories_cached(
    df: pd.DataFrame,
    *,
    dataset_id: str,
    time_threshold: float,
    **advanced_params: object,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cached wrapper for trajectory.build_trajectories.

    Avoid hashing the entire DataFrame; key the cache by a lightweight dataset_id and parameters.
    """
    _ = dataset_id  # used only for cache keying
    return build_trajectories(df, time_threshold=time_threshold, **advanced_params)


def _column_mapping() -> dict[str, str]:
    return {
        "x": "local_x",
        "y": "local_y",
        "time": "global_time",
        "vehicle": "vehicle_id",
        "frame": "frame_id",
        "lane": "lane_id",
    }


def _lane_bands(df: pd.DataFrame, *, swap_axes: bool = False) -> list[dict[str, object]]:
    mapping = _column_mapping()
    x_col, y_col, lane_col = mapping["x"], mapping["y"], mapping["lane"]
    if lane_col not in df.columns or x_col not in df.columns or y_col not in df.columns:
        return []
    # Coerce to numeric to avoid string comparisons/quantiles
    y_series = pd.to_numeric(df[y_col], errors="coerce")
    if y_series.dropna().empty:
        return []
    # Global longitudinal span (used when not swapping) and will be used as x-span when swapped
    y0, y1 = float(y_series.min(skipna=True)), float(y_series.max(skipna=True))
    shapes: list[dict[str, object]] = []
    for lane in sorted(pd.to_numeric(df[lane_col], errors="coerce").dropna().unique()):
        lane_mask = pd.to_numeric(df[lane_col], errors="coerce") == lane
        lane_df = df[lane_mask]
        if lane_df.empty:
            continue
        x_series = pd.to_numeric(lane_df[x_col], errors="coerce").dropna()
        if x_series.empty:
            continue
        q10, q90 = np.nanquantile(x_series.values, 0.1), np.nanquantile(x_series.values, 0.9)
        if not swap_axes:
            # Vertical lane bands: lateral on X (x0/x1 from lateral quantiles), longitudinal on Y (full span)
            shapes.append(
                {
                    "type": "rect",
                    "xref": "x",
                    "yref": "y",
                    "x0": float(q10),
                    "x1": float(q90),
                    "y0": y0,
                    "y1": y1,
                    "fillcolor": "rgba(200,200,200,0.2)",
                    "line": {"width": 0},
                    "layer": "below",
                }
            )
        else:
            # Horizontal lane bands when axes are swapped: longitudinal on X (full span from y0/y1), lateral on Y (q10/q90)
            shapes.append(
                {
                    "type": "rect",
                    "xref": "x",
                    "yref": "y",
                    "x0": y0,
                    "x1": y1,
                    "y0": float(q10),
                    "y1": float(q90),
                    "fillcolor": "rgba(200,200,200,0.2)",
                    "line": {"width": 0},
                    "layer": "below",
                }
            )
    return shapes


def _dataset_id(df: pd.DataFrame) -> str:
    """Build a lightweight identifier for caching; uses location (if present), row and vehicle counts."""
    try:
        nrows = int(len(df))
        veh_col = _column_mapping()["vehicle"]
        nveh = int(pd.to_numeric(df.get(veh_col, pd.Series(dtype=float)), errors="coerce").nunique())
        loc = ""
        if "location" in df.columns:
            loc_series = df["location"].dropna()
            if not loc_series.empty:
                loc = str(loc_series.mode().iloc[0])
        return f"{loc}|rows={nrows}|veh={nveh}"
    except Exception:
        return f"rows={int(len(df))}"


def _paginate(df: pd.DataFrame, page_size: int, page: int) -> pd.DataFrame:
    start = page * page_size
    end = start + page_size
    return df.iloc[start:end]


def data_explorer(df: pd.DataFrame) -> None:
    st.subheader("Data Explorer")
    mapping = _column_mapping()

    cols = st.multiselect("Columns to display", list(df.columns), default=list(df.columns))

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        vehicle_filter = st.text_input("Vehicle IDs (comma-separated)", "")
    with c2:
        if mapping["lane"] in df.columns:
            lanes_series = pd.to_numeric(df[mapping["lane"]], errors="coerce")
            lanes = sorted(int(x) for x in lanes_series.dropna().unique())
        else:
            lanes = []
        lane_filter = st.multiselect("Lane IDs", options=lanes)
    with c3:
        if mapping["frame"] in df.columns:
            frame_series = pd.to_numeric(df[mapping["frame"]], errors="coerce")
            fr_min, fr_max = int(frame_series.min(skipna=True)), int(frame_series.max(skipna=True))
            if fr_min >= fr_max:
                # Degenerate range; show as text and fix to single value
                st.write(f"Frame: {fr_min}")
                frame_range = (fr_min, fr_max)
            else:
                frame_range = st.slider("Frame range", min_value=fr_min, max_value=fr_max, value=(fr_min, fr_max))
        else:
            frame_range = None
    with c4:
        sort_col = st.selectbox("Sort by", options=list(df.columns), index=min(1, len(df.columns) - 1), key="data_explorer_sort_by")
        sort_asc = st.toggle("Ascending", value=True)

    filtered = df
    if vehicle_filter.strip():
        try:
            ids = [int(s) for s in vehicle_filter.split(',') if s.strip()]
            veh_series = pd.to_numeric(filtered[mapping["vehicle"]], errors="coerce")
            filtered = filtered[veh_series.isin(ids)]
        except Exception:
            st.warning("Invalid vehicle ID list; expected comma-separated integers.")
    if lane_filter:
        lane_series = pd.to_numeric(filtered[mapping["lane"]], errors="coerce")
        filtered = filtered[lane_series.isin(lane_filter)]
    if frame_range and mapping["frame"] in filtered.columns:
        a, b = frame_range
        fr_series = pd.to_numeric(filtered[mapping["frame"]], errors="coerce")
        filtered = filtered[(fr_series >= a) & (fr_series <= b)]

    if sort_col in filtered.columns:
        filtered = filtered.sort_values(sort_col, ascending=sort_asc)

    total_rows = int(len(filtered))
    if total_rows == 0:
        st.info("No rows match the current filters.")
        st.dataframe(filtered[cols], use_container_width=True, height=320)
        return

    page_size = st.selectbox("Rows per page", options=[50, 100, 200, 500], index=1, key="data_explorer_page_size")
    max_page = max(0, (total_rows - 1) // page_size)
    if max_page == 0:
        page = 0
        st.caption("Page 1 of 1")
    else:
        page = st.slider("Page", min_value=0, max_value=int(max_page), value=0)
    page_df = _paginate(filtered[cols], page_size, page)

    st.dataframe(page_df, use_container_width=True, height=480)


def trajectory_list(df: pd.DataFrame) -> None:
    st.subheader("Trajectory List")
    mapping = _column_mapping()
    tsec = st.number_input(
        "Time gap threshold (s)",
        min_value=1.0,
        max_value=180.0,
        value=60.0,
        step=1.0,
        help="Increase this if trajectories are split too often (default 60s)."
    )
    colA, _ = st.columns([1, 1])
    with colA:
        recompute = st.button("Recompute trajectories (clear cache)")
    if recompute:
        build_trajectories_cached.clear()
        st.rerun()
    use_adv = False
    with st.expander("Advanced segmentation", expanded=False):
        use_adv = st.checkbox("Apply advanced overrides", value=False, help="When disabled, the default detector parameters are used.")
        c1, c2, c3 = st.columns(3)
        with c1:
            bridge_gap = st.number_input("Bridge gap (s)", min_value=0.0, max_value=10.0, value=5.0, step=0.5)
            small_gap_floor = st.number_input("Small gap floor (s)", min_value=0.0, max_value=2.0, value=0.25, step=0.05)
            s_stop = st.number_input("Stationary threshold (ft)", min_value=0.0, max_value=20.0, value=7.5, step=0.5)
        with c2:
            s_jump = st.number_input("Spatial jump (ft)", min_value=50.0, max_value=1000.0, value=300.0, step=10.0)
            v_max = st.number_input("Max plausible speed (ft/s)", min_value=30.0, max_value=300.0, value=180.0, step=5.0)
            min_points = st.number_input("Min points/segment", min_value=1, max_value=50, value=4, step=1)
        with c3:
            min_duration = st.number_input("Min duration (s)", min_value=0.0, max_value=30.0, value=1.5, step=0.5)
            merge_gap = st.number_input("Merge gap (s)", min_value=0.0, max_value=20.0, value=8.0, step=0.5)
            merge_dist = st.number_input("Merge distance (ft)", min_value=0.0, max_value=200.0, value=80.0, step=5.0)
            merge_lane = st.number_input("Max lane jump on merge", min_value=0, max_value=3, value=1, step=1)

    adv = dict(
        bridge_gap=float(bridge_gap),
        small_gap_floor=float(small_gap_floor),
        s_jump=float(s_jump),
        v_max=float(v_max),
        s_stop=float(s_stop),
        min_points=int(min_points),
        min_duration=float(min_duration),
        merge_gap=float(merge_gap),
        merge_dist=float(merge_dist),
        merge_lane=int(merge_lane),
    )

    dsid = st.session_state.get("dataset_id") or _dataset_id(df)
    if use_adv:
        df_tracks, summary = build_trajectories_cached(df=df, dataset_id=str(dsid), time_threshold=float(tsec), **adv)
    else:
        df_tracks, summary = build_trajectories_cached(df=df, dataset_id=str(dsid), time_threshold=float(tsec))
    if summary.empty or "track_id" not in df_tracks.columns:
        st.error("Failed to build trajectories. Required columns: vehicle_id plus global_time or frame_id.")
        return
    # Reorder common columns for readability if present
    preferred = [
        "track_id",
        mapping["vehicle"],
        f"{mapping['frame']}_min",
        f"{mapping['frame']}_max",
        "frame_id_count",
        f"{mapping['time']}_min",
        f"{mapping['time']}_max",
    ]
    cols = [c for c in preferred if c in summary.columns] + [c for c in summary.columns if c not in preferred]
    # Summary filters to reduce frontend load
    csa, csb, csc = st.columns(3)
    with csa:
        min_frames = st.number_input("Min frames", min_value=0, max_value=1000, value=20, step=5)
    with csb:
        top_n = st.selectbox("Show top N", options=[100, 250, 500, 1000, 5000], index=2)
    with csc:
        sort_desc = st.toggle("Sort by length desc", value=True)

    df_sum = summary.copy()
    if "frame_id_count" in df_sum.columns:
        df_sum["frame_id_count"] = pd.to_numeric(df_sum["frame_id_count"], errors="coerce")
        df_sum = df_sum[df_sum["frame_id_count"] >= float(min_frames)]
        if sort_desc:
            df_sum = df_sum.sort_values("frame_id_count", ascending=False)
    df_sum_head = df_sum[cols].head(int(top_n))
    st.dataframe(df_sum_head, use_container_width=True, height=300)

    # Build a smaller selection list from the filtered summary
    if "track_id" in df_tracks.columns and "track_id" in df_sum_head.columns:
        options = [t for t in df_sum_head["track_id"].tolist() if pd.notna(t) and str(t).strip() != ""]
    elif "track_id" in df_tracks.columns:
        options = [t for t in df_tracks["track_id"].dropna().unique().tolist() if str(t).strip() != ""][:500]
    else:
        options = []
    selected = st.multiselect("Select trajectories to visualize", options=options)
    if not selected:
        return

    # Visualization controls
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        overlay = st.toggle("Overlay tracks", value=False, help="Show all selected tracks on one plot")
    with c2:
        equal_aspect = st.toggle("Equal aspect", value=True, help="Use equal X/Y scale for true geometry")
    with c3:
        break_jumps = st.toggle("Break on jumps", value=True, help="Insert gaps for large spatial jumps")
    with c4:
        jump_ft = st.number_input("Jump threshold (ft)", min_value=10.0, max_value=1000.0, value=150.0, step=10.0)
    decim = st.slider("Decimate (every Nth point)", min_value=1, max_value=10, value=1, help="Reduce clutter by plotting fewer points")

    # Prepare cleaned data: numeric x/y, time ordering per track, optional decimation and jump breaks
    working = df_tracks[df_tracks["track_id"].isin(selected)].copy()
    working[mapping["x"]] = pd.to_numeric(working[mapping["x"]], errors="coerce")
    working[mapping["y"]] = pd.to_numeric(working[mapping["y"]], errors="coerce")
    time_col = mapping["time"] if mapping["time"] in working.columns else mapping["frame"]
    t = pd.to_numeric(working[time_col], errors="coerce") if time_col in working.columns else pd.Series(index=working.index, dtype=float)
    if time_col == "global_time" and t.max(skipna=True) > 10000:
        t = t / 1000.0
    working["__t__"] = t
    working = working.dropna(subset=[mapping["x"], mapping["y"], "__t__"]).copy()

    def _series_with_breaks(g: pd.DataFrame) -> tuple[list[float | None], list[float | None]]:
        g = g.sort_values("__t__")
        if decim > 1:
            g = g.iloc[:: int(decim)]
        xs = pd.to_numeric(g[mapping["x"]], errors="coerce").tolist()
        ys = pd.to_numeric(g[mapping["y"]], errors="coerce").tolist()
        if not break_jumps or len(xs) < 2:
            return xs, ys
        # Insert None where spatial jump exceeds threshold
        out_x: list[float | None] = [xs[0]]
        out_y: list[float | None] = [ys[0]]
        for i in range(1, len(xs)):
            dx = float(xs[i] - xs[i - 1]) if xs[i] is not None and xs[i - 1] is not None else 0.0
            dy = float(ys[i] - ys[i - 1]) if ys[i] is not None and ys[i - 1] is not None else 0.0
            if np.hypot(dx, dy) > float(jump_ft):
                out_x.append(None)
                out_y.append(None)
            out_x.append(xs[i])
            out_y.append(ys[i])
        return out_x, out_y

    groups = list(working.groupby("track_id", sort=False))
    shapes_once = _lane_bands(working, swap_axes=True)
    if overlay:
        fig = go.Figure()
        fig.update_layout(template="plotly_white", height=520)
        for shp in shapes_once:
            fig.add_shape(shp)
        for tid, grp in groups:
            xs, ys = _series_with_breaks(grp)
            # Swap axes: x <- longitudinal (ys), y <- lateral (xs)
            fig.add_trace(
                go.Scatter(
                    x=ys,
                    y=xs,
                    mode="lines",
                    name=str(tid),
                    line=dict(width=2),
                )
            )
        fig.update_xaxes(title_text="Longitudinal (ft)")
        fig.update_yaxes(title_text="Lateral (ft)")
        if equal_aspect:
            fig.update_yaxes(scaleanchor="x", scaleratio=1)
        st.plotly_chart(fig, use_container_width=True)
    else:
        rows = len(groups)
        fig = make_subplots(rows=rows, cols=1, shared_xaxes=False, shared_yaxes=False, vertical_spacing=0.04, subplot_titles=[str(tid) for tid, _ in groups])
        for r, (tid, grp) in enumerate(groups, start=1):
            xs, ys = _series_with_breaks(grp)
            # Swap axes per subplot
            fig.add_trace(go.Scatter(x=ys, y=xs, mode="lines", name=str(tid), line=dict(width=2)), row=r, col=1)
            for shp in shapes_once:
                fig.add_shape(shp, row=r, col=1)
            fig.update_xaxes(title_text="Longitudinal (ft)", row=r, col=1)
            fig.update_yaxes(title_text="Lateral (ft)", row=r, col=1)
            if equal_aspect:
                fig.update_yaxes(scaleanchor=f"x{r if r>1 else ''}", scaleratio=1, row=r, col=1)
        fig.update_layout(template="plotly_white", height=360 * rows)
        st.plotly_chart(fig, use_container_width=True)


def road_visualization(df: pd.DataFrame) -> None:
    st.subheader("Road Visualization (Animated)")
    mapping = _column_mapping()
    if mapping["time"] not in df.columns:
        st.info("No time column available for animation.")
        return

    df_anim = df.copy()
    t = pd.to_numeric(df_anim[mapping["time"]], errors="coerce")
    if t.max() > 10000:
        t = t / 1000.0
    df_anim["t_sec"] = t.round(1)
    df_anim = df_anim.dropna(subset=["t_sec", mapping["x"], mapping["y"], mapping["vehicle"]])

    # Performance controls: constrain time window, vehicle count, and frame density
    with st.expander("Animation settings", expanded=False):
        max_seconds = st.number_input("Time window (seconds)", min_value=1.0, max_value=120.0, value=10.0, step=1.0)
        max_vehicles = st.number_input("Max vehicles", min_value=5, max_value=200, value=30, step=5)
        frame_step = st.number_input("Frame decimation (every Nth)", min_value=1, max_value=10, value=2, step=1)

    # Apply windowing and decimation
    t0 = float(t.min(skipna=True))
    t1 = t0 + float(max_seconds)
    df_anim = df_anim[(t >= t0) & (t <= t1)].copy()
    veh_col = mapping["vehicle"]
    top_veh = df_anim[veh_col].value_counts(dropna=True).head(int(max_vehicles)).index.tolist()
    df_anim = df_anim[df_anim[veh_col].isin(top_veh)].copy()
    if int(frame_step) > 1:
        df_anim = df_anim.iloc[:: int(frame_step)].copy()

    # Swap axes for animation too: x <- longitudinal (y), y <- lateral (x)
    fig = px.scatter(
        df_anim,
        x=mapping["y"],
        y=mapping["x"],
        animation_frame="t_sec",
        color=mapping["vehicle"],
        size_max=6,
        opacity=0.8,
        labels={mapping["y"]: "Longitudinal (ft)", mapping["x"]: "Lateral (ft)"},
        template="plotly_white",
    )
    fig.update_layout(height=550)
    fig.update_layout(shapes=_lane_bands(df_anim, swap_axes=True))
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.title("NGSIM Trajectory Explorer")

    with st.sidebar:
        st.header("Data Source")
        tabs = st.tabs(["Portal slice", "Cached dataset"])  # sidebar tabs for source selection
        with tabs[0]:
            _ = st.selectbox(
                "Location",
                options=["us-101", "i-80", "lankershim"],
                index=0,
                key="sidebar_location_select",
            )
            _ = st.number_input("Rows to load", min_value=1000, max_value=200000, value=20000, step=1000, key="sidebar_limit")
            if st.button("Load data", type="primary", key="btn_load_portal"):
                st.session_state["_trigger_load"] = True
                st.session_state["_load_source"] = "portal"
        with tabs[1]:
            _ = st.text_input("Cache directory", value=".ngsim_cache", key="sidebar_cache_dir")
            _ = st.toggle("Force refresh", value=False, help="Re-download and rebuild the cached dataset", key="sidebar_force_refresh")
            if st.button("Load cached", type="primary", key="btn_load_cached"):
                st.session_state["_trigger_load"] = True
                st.session_state["_load_source"] = "cached"

    if st.session_state.get("_trigger_load"):
        if st.session_state.get("_load_source") == "cached":
            df = load_data_cached(
                cache_dir=st.session_state.get("sidebar_cache_dir", ".ngsim_cache"),
                force_refresh=bool(st.session_state.get("sidebar_force_refresh", False)),
            )
        else:
            df = load_data(
                st.session_state.get("sidebar_location_select", "us-101"),
                int(st.session_state.get("sidebar_limit", 20000)),
            )
    else:
        df = load_data("us-101", 5000)

    # Store a stable dataset id for caching downstream computations
    st.session_state["dataset_id"] = _dataset_id(df)

    tabs = st.tabs(["Data Explorer", "Trajectory List", "Road Visualization", "Predictors"])
    with tabs[0]:
        data_explorer(df)
    with tabs[1]:
        trajectory_list(df)
    with tabs[2]:
        road_visualization(df)
    with tabs[3]:
        try:
            from . import gui_predictors
        except Exception:
            import trajectory_challenge.gui_predictors as gui_predictors  # type: ignore[no-redef]
        gui_predictors.run(df)


# Standard guard: Streamlit runs this file as __main__; avoid complex re-entry logic
if __name__ == "__main__":  # pragma: no cover
    main()
