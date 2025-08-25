"""Entry point demonstrating NGSIM data access."""
from __future__ import annotations

from trajectory_challenge import ngsim


def main() -> None:
    # Fetch a small slice (limit) from JSON endpoint without specifying columns to reduce risk of mismatch.
    try:
        df = ngsim.load_ngsim_portal(limit=1000, json_endpoint=True)
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"Failed to load NGSIM portal sample: {exc}")
        return

    print(f"Fetched {len(df)} rows. Columns: {list(df.columns)}")
    # Show simple aggregate if columns available.
    if {"vehicle_id", "frame_id"}.issubset(df.columns):
        unique_vehicles = df["vehicle_id"].nunique()
        min_frame = df["frame_id"].min()
        max_frame = df["frame_id"].max()
        print(f"Vehicles: {unique_vehicles}; frame range: {min_frame}..{max_frame}")
    else:
        print("Could not compute frame statistics (missing canonical columns).")


if __name__ == "__main__":  # pragma: no cover
    main()
