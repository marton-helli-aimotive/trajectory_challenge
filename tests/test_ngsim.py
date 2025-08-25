import pandas as pd
from trajectory_challenge.ngsim import (
    _canonicalize_columns as canon,  # type: ignore
    filter_by_vehicle,
    filter_by_frame_range,
    filter_by_lane,
)


def test_ngsim_filters():
    df = pd.DataFrame(
        {
            "Vehicle_ID": [1, 1, 2],
            "Frame_ID": [10, 11, 10],
            "Lane_ID": [2, 2, 3],
        }
    )
    df = canon(df)
    assert set(df.columns) >= {"vehicle_id", "frame_id", "lane_id"}
    assert filter_by_vehicle(df, 1)["vehicle_id"].nunique() == 1
    assert filter_by_frame_range(df, start=11)["frame_id"].min() == 11
    assert filter_by_lane(df, [3])["lane_id"].nunique() == 1
