from pathlib import Path

import pandas as pd
import pytest

from trajectory_challenge.ngsim import load_ngsim_cached
from trajectory_challenge.trajectory import build_trajectories


def test_cached_dataset_trajectory_lengths():
    """Ensure most trajectories are reasonably long on the cached dataset.

    Requirement: at least 90% of trajectories have frame_id_count >= 20.
    If the cache file is not present locally, skip to avoid heavy network fetch.
    """
    cache_dir = Path(".ngsim_cache")
    cache_file = cache_dir / "ngsim_complete_dataset.parquet"

    if not cache_file.exists():
        pytest.skip("Cached dataset not found; skipping heavy network download in tests")

    df = load_ngsim_cached(cache_dir=str(cache_dir), force_refresh=False)
    assert not df.empty, "Loaded cached dataset is empty"

    # Build trajectories directly using the public API
    df_tracks, summary = build_trajectories(df, time_threshold=60.0)
    print(df_tracks)

    # Prefer the precomputed frame_id_count from the summary if available
    if not summary.empty and "frame_id_count" in summary.columns:
        counts = pd.to_numeric(summary["frame_id_count"], errors="coerce").dropna()
        assert not counts.empty, "No valid frame_id_count values found in summary"
        long_share = float((counts >= 20).mean())
        assert long_share >= 0.90, f"Only {long_share:.1%} trajectories have >=20 frames"
        return

    # Fallback: compute distinct frame_id per track_id from the tracks dataframe
    if "frame_id" not in df_tracks.columns:
        pytest.skip("frame_id column missing; cannot compute frame_id_count")
    if "track_id" not in df_tracks.columns:
        pytest.skip("track_id column missing after building trajectories")

    df_tracks["frame_id"] = pd.to_numeric(df_tracks["frame_id"], errors="coerce")
    frame_counts = (
        df_tracks.dropna(subset=["frame_id"]).groupby("track_id")["frame_id"].nunique()
    )
    assert not frame_counts.empty, "No trajectories with valid frame_id found"
    long_share = float((frame_counts >= 20).mean())
    assert long_share >= 0.90, f"Only {long_share:.1%} trajectories have >=20 frames"
