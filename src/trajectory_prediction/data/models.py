"""Pydantic data models for trajectory prediction with validation."""

from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator


class VehicleType(str, Enum):
    """Vehicle type enumeration."""

    CAR = "car"
    TRUCK = "truck"
    MOTORCYCLE = "motorcycle"
    BUS = "bus"
    OTHER = "other"


class TrajectoryPoint(BaseModel):  # type: ignore[misc]
    """Individual trajectory point with spatial-temporal information."""

    model_config = ConfigDict(validate_assignment=True)

    timestamp: float = Field(..., description="Unix timestamp in seconds")
    x: float = Field(..., description="X coordinate in meters")
    y: float = Field(..., description="Y coordinate in meters")
    speed: float = Field(0.0, ge=0.0, le=200.0, description="Speed in m/s")
    velocity_x: float = Field(0.0, description="X velocity component in m/s")
    velocity_y: float = Field(0.0, description="Y velocity component in m/s")
    acceleration_x: float = Field(0.0, description="X acceleration component in m/s²")
    acceleration_y: float = Field(0.0, description="Y acceleration component in m/s²")
    heading: float | None = Field(
        None, ge=0.0, lt=360.0, description="Heading in degrees"
    )
    lane_id: int | None = Field(None, description="Lane identifier")
    frame_id: int | None = Field(None, description="Frame identifier")

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, v: float) -> float:
        """Validate timestamp is reasonable."""
        if v < 946684800:  # Year 2000
            raise ValueError("Timestamp too old (before year 2000)")
        if v > 4102444800:  # Year 2100
            raise ValueError("Timestamp too far in future (after year 2100)")
        return v

    @field_validator("x", "y")
    @classmethod
    def validate_coordinates(cls, v: float) -> float:
        """Validate coordinates are reasonable."""
        if abs(v) > 1000000:  # 1000 km limit
            raise ValueError("Coordinate value too large")
        return v

    @field_validator("velocity_x", "velocity_y")
    @classmethod
    def validate_velocity(cls, v: float) -> float:
        """Validate velocity components."""
        if abs(v) > 200:  # 200 m/s = 720 km/h limit
            raise ValueError("Velocity too high")
        return v

    @field_validator("acceleration_x", "acceleration_y")
    @classmethod
    def validate_acceleration(cls, v: float) -> float:
        """Validate acceleration components."""
        if abs(v) > 50:  # 50 m/s² limit (very aggressive)
            raise ValueError("Acceleration too high")
        return v


class Vehicle(BaseModel):  # type: ignore[misc]
    """Vehicle characteristics and metadata."""

    model_config = ConfigDict(validate_assignment=True)

    vehicle_id: int = Field(..., description="Unique vehicle identifier")
    vehicle_type: VehicleType = Field(VehicleType.CAR, description="Type of vehicle")
    length: float = Field(4.5, gt=0.0, le=30.0, description="Vehicle length in meters")
    width: float = Field(1.8, gt=0.0, le=4.0, description="Vehicle width in meters")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    @field_validator("length", "width")
    @classmethod
    def validate_dimensions(cls, v: float) -> float:
        """Validate vehicle dimensions are reasonable."""
        if v <= 0:
            raise ValueError("Vehicle dimensions must be positive")
        return v


class Trajectory(BaseModel):  # type: ignore[misc]
    """Complete trajectory with quality metrics."""

    model_config = ConfigDict(validate_assignment=True)

    trajectory_id: str = Field(..., description="Unique trajectory identifier")
    vehicle: Vehicle = Field(..., description="Vehicle information")
    points: list[TrajectoryPoint] = Field(
        ..., min_length=2, description="Trajectory points"
    )
    dataset_name: str = Field(..., description="Source dataset name")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    # Quality metrics (computed on demand)
    completeness_score: float | None = Field(None, ge=0.0, le=1.0)
    temporal_consistency_score: float | None = Field(None, ge=0.0, le=1.0)
    spatial_accuracy_score: float | None = Field(None, ge=0.0, le=1.0)
    smoothness_score: float | None = Field(None, ge=0.0, le=1.0)

    @field_validator("points")
    @classmethod
    def validate_points_temporal_order(
        cls, v: list[TrajectoryPoint]
    ) -> list[TrajectoryPoint]:
        """Validate points are in temporal order."""
        if len(v) < 2:
            raise ValueError("Trajectory must have at least 2 points")

        timestamps = [p.timestamp for p in v]
        if timestamps != sorted(timestamps):
            raise ValueError("Trajectory points must be in temporal order")

        # Check for reasonable time gaps (max 5 seconds between consecutive points)
        for i in range(1, len(timestamps)):
            gap = timestamps[i] - timestamps[i - 1]
            if gap > 5.0:
                raise ValueError(f"Time gap too large: {gap}s between points")
            if gap <= 0:
                raise ValueError("Non-positive time gap between points")

        return v

    @property
    def duration(self) -> float:
        """Calculate trajectory duration in seconds."""
        if len(self.points) < 2:
            return 0.0
        return self.points[-1].timestamp - self.points[0].timestamp

    @property
    def length(self) -> float:
        """Calculate total trajectory length in meters."""
        if len(self.points) < 2:
            return 0.0

        total_length = 0.0
        for i in range(1, len(self.points)):
            dx = self.points[i].x - self.points[i - 1].x
            dy = self.points[i].y - self.points[i - 1].y
            total_length += np.sqrt(dx**2 + dy**2)

        return total_length

    def compute_quality_metrics(self) -> None:
        """Compute trajectory quality metrics."""
        if len(self.points) < 2:
            self.completeness_score = 0.0
            self.temporal_consistency_score = 0.0
            self.spatial_accuracy_score = 0.0
            self.smoothness_score = 0.0
            return

        # Completeness: based on number of points and temporal coverage
        expected_points = max(10, int(self.duration * 10))  # Expect ~10 Hz
        actual_points = len(self.points)
        self.completeness_score = min(1.0, actual_points / expected_points)

        # Temporal consistency: check for uniform time intervals
        time_intervals = []
        for i in range(1, len(self.points)):
            interval = self.points[i].timestamp - self.points[i - 1].timestamp
            time_intervals.append(interval)

        if time_intervals:
            mean_interval = float(np.mean(time_intervals))
            std_interval = float(np.std(time_intervals))
            cv = std_interval / mean_interval if mean_interval > 0 else 1.0
            self.temporal_consistency_score = float(max(0.0, 1.0 - cv))
        else:
            self.temporal_consistency_score = 0.0

        # Spatial accuracy: check for physically reasonable movements
        unreasonable_moves = 0
        for i in range(1, len(self.points)):
            dt = self.points[i].timestamp - self.points[i - 1].timestamp
            dx = self.points[i].x - self.points[i - 1].x
            dy = self.points[i].y - self.points[i - 1].y
            distance = np.sqrt(dx**2 + dy**2)

            if dt > 0:
                implied_speed = distance / dt
                if implied_speed > 100:  # 100 m/s = 360 km/h
                    unreasonable_moves += 1

        total_moves = len(self.points) - 1
        self.spatial_accuracy_score = (
            1.0 - (unreasonable_moves / total_moves) if total_moves > 0 else 1.0
        )

        # Smoothness: analyze acceleration changes (jerk)
        if len(self.points) >= 3:
            jerk_values = []
            for i in range(2, len(self.points)):
                dt1 = self.points[i - 1].timestamp - self.points[i - 2].timestamp
                dt2 = self.points[i].timestamp - self.points[i - 1].timestamp

                if dt1 > 0 and dt2 > 0:
                    # Calculate acceleration at consecutive time steps
                    ax1 = self.points[i - 1].acceleration_x
                    ax2 = self.points[i].acceleration_x
                    ay1 = self.points[i - 1].acceleration_y
                    ay2 = self.points[i].acceleration_y

                    # Jerk magnitude
                    jerk_x = (ax2 - ax1) / dt2
                    jerk_y = (ay2 - ay1) / dt2
                    jerk_magnitude = np.sqrt(jerk_x**2 + jerk_y**2)
                    jerk_values.append(jerk_magnitude)

            if jerk_values:
                mean_jerk = float(np.mean(jerk_values))
                # Normalize jerk (10 m/s³ is considered high)
                normalized_jerk = min(1.0, mean_jerk / 10.0)
                self.smoothness_score = float(max(0.0, 1.0 - normalized_jerk))
            else:
                self.smoothness_score = 1.0
        else:
            self.smoothness_score = 1.0


class Dataset(BaseModel):  # type: ignore[misc]
    """Collection of trajectories with metadata and quality metrics."""

    model_config = ConfigDict(validate_assignment=True)

    name: str = Field(..., description="Dataset name")
    version: str = Field("1.0", description="Dataset version")
    description: str = Field("", description="Dataset description")
    trajectories: list[Trajectory] = Field(
        default_factory=list, description="List of trajectories"
    )
    coordinate_system: str = Field("cartesian", description="Coordinate system used")
    units: dict[str, str] = Field(
        default_factory=lambda: {
            "position": "meters",
            "velocity": "m/s",
            "time": "seconds",
        },
        description="Units for measurements",
    )
    citation: str = Field("", description="Citation information")
    created_at: datetime = Field(
        default_factory=datetime.now, description="Creation timestamp"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    @property
    def total_trajectories(self) -> int:
        """Get total number of trajectories."""
        return len(self.trajectories)

    @property
    def total_points(self) -> int:
        """Get total number of trajectory points."""
        return sum(len(traj.points) for traj in self.trajectories)

    def get_quality_summary(self) -> dict[str, float]:
        """Get aggregated quality metrics across all trajectories."""
        if not self.trajectories:
            return {
                "avg_completeness": 0.0,
                "avg_temporal_consistency": 0.0,
                "avg_spatial_accuracy": 0.0,
                "avg_smoothness": 0.0,
            }

        # Ensure quality metrics are computed
        for trajectory in self.trajectories:
            if trajectory.completeness_score is None:
                trajectory.compute_quality_metrics()

        all_timestamps: list[float] = []
        completeness_scores = []
        temporal_scores = []
        spatial_scores = []
        smoothness_scores = []

        for trajectory in self.trajectories:
            all_timestamps.extend([p.timestamp for p in trajectory.points])
            if trajectory.completeness_score is not None:
                completeness_scores.append(trajectory.completeness_score)
            if trajectory.temporal_consistency_score is not None:
                temporal_scores.append(trajectory.temporal_consistency_score)
            if trajectory.spatial_accuracy_score is not None:
                spatial_scores.append(trajectory.spatial_accuracy_score)
            if trajectory.smoothness_score is not None:
                smoothness_scores.append(trajectory.smoothness_score)

        return {
            "avg_completeness": float(np.mean(completeness_scores))
            if completeness_scores
            else 0.0,
            "avg_temporal_consistency": float(np.mean(temporal_scores))
            if temporal_scores
            else 0.0,
            "avg_spatial_accuracy": float(np.mean(spatial_scores))
            if spatial_scores
            else 0.0,
            "avg_smoothness": float(np.mean(smoothness_scores))
            if smoothness_scores
            else 0.0,
        }

    def add_trajectory(self, trajectory: Trajectory) -> None:
        """Add a trajectory to the dataset."""
        self.trajectories.append(trajectory)

    def filter_by_duration(
        self, min_duration: float, max_duration: float | None = None
    ) -> list[Trajectory]:
        """Filter trajectories by duration."""
        filtered = []
        for trajectory in self.trajectories:
            duration = trajectory.duration
            if duration >= min_duration and (
                max_duration is None or duration <= max_duration
            ):
                filtered.append(trajectory)
        return filtered

    def filter_by_vehicle_type(self, vehicle_type: VehicleType) -> list[Trajectory]:
        """Filter trajectories by vehicle type."""
        return [
            traj
            for traj in self.trajectories
            if traj.vehicle.vehicle_type == vehicle_type
        ]
