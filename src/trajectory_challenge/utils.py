"""Shared utilities and types for trajectory_challenge.

Public API
- DatasetName: Literal of supported dataset names.
- canonicalize_columns(df) -> DataFrame: rename columns to the canonical NGSIM schema.
- add_numbers(a, b) -> number: trivial helper used in tests.
- divide(a, b) -> number: trivial helper used in tests; raises ValueError on division by zero.
"""
from __future__ import annotations

from typing import Literal
import pandas as pd

# Supported dataset names
DatasetName = Literal["us_101", "i_80", "lankershim"]

# Column name canonicalization map based on dataset schema in .github/instructions/dataset_record.instructions.md
_CANONICAL_COLUMNS: dict[str, str] = {
    # Canonical names
    "vehicle_id": "vehicle_id",
    "frame_id": "frame_id",
    "total_frames": "total_frames",
    "global_time": "global_time",
    "local_x": "local_x",
    "local_y": "local_y",
    "global_x": "global_x",
    "global_y": "global_y",
    "v_length": "v_length",
    "v_width": "v_width",
    "v_class": "v_class",
    "v_vel": "v_vel",
    "v_acc": "v_acc",
    "lane_id": "lane_id",
    "o_zone": "o_zone",
    "d_zone": "d_zone",
    "int_id": "int_id",
    "section_id": "section_id",
    "direction": "direction",
    "movement": "movement",
    "preceding": "preceding",
    "following": "following",
    "space_headway": "space_headway",
    "time_headway": "time_headway",
    "location": "location",
    # Legacy/variant names
    "Vehicle_ID": "vehicle_id",
    "Frame_ID": "frame_id",
    "Total_Frames": "total_frames",
    "Global_Time": "global_time",
    "Local_X": "local_x",
    "Local_Y": "local_y",
    "Global_X": "global_x",
    "Global_Y": "global_y",
    "v_Length": "v_length",
    "v_Width": "v_width",
    "v_Class": "v_class",
    "v_Vel": "v_vel",
    "v_Acc": "v_acc",
    "Lane_ID": "lane_id",
    "O_Zone": "o_zone",
    "D_Zone": "d_zone",
    "Int_ID": "int_id",
    "Section_ID": "section_id",
    "Direction": "direction",
    "Movement": "movement",
    "Preceding": "preceding",
    "Following": "following",
    "Space_Headway": "space_headway",
    "Time_Headway": "time_headway",
    "Location": "location",
    # API/JSON variants
    "frame": "frame_id",
    "lane": "lane_id",
}


def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with columns renamed to the canonical NGSIM schema.

    Input: pandas.DataFrame with any mix of canonical or legacy/variant column names.
    Output: pandas.DataFrame with standardized column names. Unrecognized columns are unchanged.
    """
    rename_map = {src: dst for src, dst in _CANONICAL_COLUMNS.items() if src in df.columns}
    return df.rename(columns=rename_map)


def add_numbers(a: float | int, b: float | int) -> float | int:
    """Add two numbers.

    Input: a, b numbers. Output: sum with standard Python numeric promotion.
    """
    return a + b


def divide(a: float | int, b: float | int) -> float:
    """Divide a by b; raises ValueError on division by zero."""
    if b == 0:
        raise ValueError("division by zero")
    return float(a) / float(b)


__all__ = [
    "DatasetName",
    "canonicalize_columns",
    "add_numbers",
    "divide",
]
