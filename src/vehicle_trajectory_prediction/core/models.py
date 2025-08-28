"""Core data models for trajectory prediction system."""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

# Try to import optional dependencies
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from pydantic import BaseModel, Field, validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

try:
    from shapely.geometry import Point
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False


class TrajectoryPoint:
    """A single point in a vehicle trajectory."""
    
    def __init__(
        self,
        x: float,
        y: float,
        timestamp: datetime,
        velocity: float,
        acceleration: float,
        heading: float,
        vehicle_id: str,
        lane_id: Optional[str] = None,
        attributes: Dict[str, Any] = None
    ):
        self.x = x
        self.y = y
        self.timestamp = timestamp
        self.velocity = velocity
        self.acceleration = acceleration
        self.heading = heading
        self.vehicle_id = vehicle_id
        self.lane_id = lane_id
        self.attributes = attributes or {}
        
        # Validate inputs
        if velocity < 0:
            raise ValueError("Velocity must be non-negative")
        
        # Normalize heading to [0, 2π)
        if NUMPY_AVAILABLE:
            self.heading = heading % (2 * np.pi)
        else:
            import math
            self.heading = heading % (2 * math.pi)
    
    def to_point(self):
        """Convert to Shapely Point."""
        if SHAPELY_AVAILABLE:
            return Point(self.x, self.y)
        else:
            return (self.x, self.y)
    
    def distance_to(self, other: "TrajectoryPoint") -> float:
        """Calculate Euclidean distance to another point."""
        if NUMPY_AVAILABLE:
            return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)
        else:
            import math
            return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)
    
    def time_delta(self, other: "TrajectoryPoint") -> float:
        """Calculate time difference in seconds."""
        return (other.timestamp - self.timestamp).total_seconds()
    
    def velocity_delta(self, other: "TrajectoryPoint") -> float:
        """Calculate velocity difference."""
        return other.velocity - self.velocity
    
    def acceleration_delta(self, other: "TrajectoryPoint") -> float:
        """Calculate acceleration difference."""
        return other.acceleration - self.acceleration


class Trajectory:
    """A complete vehicle trajectory."""
    
    def __init__(
        self,
        vehicle_id: str,
        points: List[TrajectoryPoint],
        start_time: datetime,
        end_time: datetime,
        duration: float,
        total_distance: float,
        quality_score: float = 1.0,
        completeness: float = 1.0,
        smoothness: float = 1.0,
        metadata: Dict[str, Any] = None
    ):
        self.vehicle_id = vehicle_id
        self.points = points
        self.start_time = start_time
        self.end_time = end_time
        self.duration = duration
        self.total_distance = total_distance
        self.quality_score = quality_score
        self.completeness = completeness
        self.smoothness = smoothness
        self.metadata = metadata or {}
        
        # Validate inputs
        if len(points) < 2:
            raise ValueError("Trajectory must have at least 2 points")
        
        # Check that all points have the same vehicle_id
        vehicle_ids = {point.vehicle_id for point in points}
        if len(vehicle_ids) > 1:
            raise ValueError("All points must have the same vehicle_id")
        
        # Check that points are ordered by timestamp
        timestamps = [point.timestamp for point in points]
        if timestamps != sorted(timestamps):
            raise ValueError("Points must be ordered by timestamp")
        
        # Check that start time is before end time
        if start_time >= end_time:
            raise ValueError("Start time must be before end time")
    
    @property
    def length(self) -> int:
        """Number of points in trajectory."""
        return len(self.points)
    
    def get_point_at_time(self, timestamp: datetime) -> Optional[TrajectoryPoint]:
        """Get trajectory point at specific time (interpolated if needed)."""
        if not self.points:
            return None
        
        # Find exact match
        for point in self.points:
            if point.timestamp == timestamp:
                return point
        
        # Find closest point
        closest_point = min(self.points, key=lambda p: abs((p.timestamp - timestamp).total_seconds()))
        return closest_point
    
    def get_segment(self, start_time: datetime, end_time: datetime) -> "Trajectory":
        """Get trajectory segment between two times."""
        segment_points = [
            point for point in self.points
            if start_time <= point.timestamp <= end_time
        ]
        
        if len(segment_points) < 2:
            raise ValueError("Segment must have at least 2 points")
        
        return Trajectory(
            vehicle_id=self.vehicle_id,
            points=segment_points,
            start_time=start_time,
            end_time=end_time,
            duration=(end_time - start_time).total_seconds(),
            total_distance=self._calculate_distance(segment_points),
            metadata=self.metadata.copy()
        )
    
    def to_dataframe(self):
        """Convert trajectory to pandas DataFrame."""
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas is required for DataFrame conversion")
        
        data = []
        for point in self.points:
            data.append({
                "vehicle_id": point.vehicle_id,
                "timestamp": point.timestamp,
                "x": point.x,
                "y": point.y,
                "velocity": point.velocity,
                "acceleration": point.acceleration,
                "heading": point.heading,
                "lane_id": point.lane_id,
                **point.attributes
            })
        
        return pd.DataFrame(data)
    
    def _calculate_distance(self, points: List[TrajectoryPoint]) -> float:
        """Calculate total distance of trajectory points."""
        if len(points) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(1, len(points)):
            total_distance += points[i-1].distance_to(points[i])
        
        return total_distance


class TrajectoryDataset:
    """A collection of trajectories."""
    
    def __init__(
        self,
        name: str,
        trajectories: List[Trajectory],
        description: Optional[str] = None,
        source: Optional[str] = None,
        version: str = "1.0.0",
        num_vehicles: int = None,
        num_trajectories: int = None,
        total_points: int = None,
        start_time: datetime = None,
        end_time: datetime = None,
        bounds: Tuple[float, float, float, float] = None
    ):
        self.name = name
        self.trajectories = trajectories
        self.description = description
        self.source = source
        self.version = version
        self.num_vehicles = num_vehicles or len(set(traj.vehicle_id for traj in trajectories))
        self.num_trajectories = num_trajectories or len(trajectories)
        self.total_points = total_points or sum(traj.length for traj in trajectories)
        self.start_time = start_time or min(traj.start_time for traj in trajectories)
        self.end_time = end_time or max(traj.end_time for traj in trajectories)
        self.bounds = bounds or self._calculate_bounds()
        
        # Validate inputs
        if not trajectories:
            raise ValueError("Dataset must contain at least one trajectory")
    
    @property
    def vehicle_ids(self) -> List[str]:
        """Get list of unique vehicle IDs."""
        return list(set(traj.vehicle_id for traj in self.trajectories))
    
    def get_trajectory_by_vehicle(self, vehicle_id: str) -> List[Trajectory]:
        """Get all trajectories for a specific vehicle."""
        return [traj for traj in self.trajectories if traj.vehicle_id == vehicle_id]
    
    def get_trajectories_in_time_range(self, start_time: datetime, end_time: datetime) -> List[Trajectory]:
        """Get trajectories that overlap with the given time range."""
        return [
            traj for traj in self.trajectories
            if traj.start_time <= end_time and traj.end_time >= start_time
        ]
    
    def to_dataframe(self):
        """Convert dataset to pandas DataFrame."""
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas is required for DataFrame conversion")
        
        dfs = []
        for trajectory in self.trajectories:
            df = trajectory.to_dataframe()
            dfs.append(df)
        
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    
    def split_by_time(self, split_time: datetime) -> Tuple["TrajectoryDataset", "TrajectoryDataset"]:
        """Split dataset into two parts based on time."""
        before_trajectories = []
        after_trajectories = []
        
        for trajectory in self.trajectories:
            if trajectory.end_time <= split_time:
                before_trajectories.append(trajectory)
            elif trajectory.start_time >= split_time:
                after_trajectories.append(trajectory)
            else:
                # Split trajectory
                before_segment = trajectory.get_segment(trajectory.start_time, split_time)
                after_segment = trajectory.get_segment(split_time, trajectory.end_time)
                before_trajectories.append(before_segment)
                after_trajectories.append(after_segment)
        
        before_dataset = TrajectoryDataset(
            name=f"{self.name}_before",
            trajectories=before_trajectories,
            description=f"Trajectories before {split_time}",
            source=self.source,
            version=self.version
        )
        
        after_dataset = TrajectoryDataset(
            name=f"{self.name}_after",
            trajectories=after_trajectories,
            description=f"Trajectories after {split_time}",
            source=self.source,
            version=self.version
        )
        
        return before_dataset, after_dataset
    
    def _calculate_bounds(self) -> Tuple[float, float, float, float]:
        """Calculate spatial bounds of the dataset."""
        if not self.trajectories:
            return (0.0, 0.0, 0.0, 0.0)
        
        all_x = []
        all_y = []
        for trajectory in self.trajectories:
            for point in trajectory.points:
                all_x.append(point.x)
                all_y.append(point.y)
        
        return (min(all_x), min(all_y), max(all_x), max(all_y))


class PredictionRequest:
    """Request for trajectory prediction."""
    
    def __init__(
        self,
        vehicle_id: str,
        current_state: TrajectoryPoint,
        prediction_horizon: int = 30,
        prediction_frequency: float = 0.1,
        context_trajectories: List[Trajectory] = None,
        road_geometry: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = None,
        confidence_level: float = 0.95
    ):
        self.vehicle_id = vehicle_id
        self.current_state = current_state
        self.prediction_horizon = prediction_horizon
        self.prediction_frequency = prediction_frequency
        self.context_trajectories = context_trajectories or []
        self.road_geometry = road_geometry
        self.model_name = model_name
        self.confidence_level = confidence_level


class PredictionResult:
    """Result of trajectory prediction."""
    
    def __init__(
        self,
        vehicle_id: str,
        prediction_horizon: int,
        prediction_frequency: float,
        predicted_trajectory: Trajectory,
        model_name: str,
        model_confidence: float,
        inference_time: float,
        confidence_intervals: Optional[List[Tuple[float, float]]] = None,
        uncertainty_scores: Optional[List[float]] = None,
        memory_usage: Optional[float] = None,
        metadata: Dict[str, Any] = None
    ):
        self.vehicle_id = vehicle_id
        self.prediction_horizon = prediction_horizon
        self.prediction_frequency = prediction_frequency
        self.predicted_trajectory = predicted_trajectory
        self.model_name = model_name
        self.model_confidence = model_confidence
        self.inference_time = inference_time
        self.confidence_intervals = confidence_intervals
        self.uncertainty_scores = uncertainty_scores
        self.memory_usage = memory_usage
        self.metadata = metadata or {}
        
        # Validate inputs
        if predicted_trajectory.length < 2:
            raise ValueError("Predicted trajectory must have at least 2 points")
        
        if confidence_intervals is not None and len(confidence_intervals) != predicted_trajectory.length:
            raise ValueError("Number of confidence intervals must match trajectory length")
        
        if uncertainty_scores is not None and len(uncertainty_scores) != predicted_trajectory.length:
            raise ValueError("Number of uncertainty scores must match trajectory length")


class TrajectoryData:
    """Data structure for trajectory data with numpy arrays for efficient processing."""
    
    def __init__(
        self,
        vehicle_id: str,
        timestamps: np.ndarray,
        x_positions: np.ndarray,
        y_positions: np.ndarray,
        velocities: np.ndarray,
        headings: np.ndarray,
        accelerations: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.vehicle_id = vehicle_id
        self.timestamps = timestamps
        self.x_positions = x_positions
        self.y_positions = y_positions
        self.velocities = velocities
        self.headings = headings
        self.accelerations = accelerations
        self.metadata = metadata or {}
        
        # Validate inputs
        if not all(len(arr) == len(timestamps) for arr in [x_positions, y_positions, velocities, headings, accelerations]):
            raise ValueError("All arrays must have the same length as timestamps")
        
        if len(timestamps) < 2:
            raise ValueError("Trajectory must have at least 2 points")
    
    @property
    def length(self) -> int:
        """Number of points in trajectory."""
        return len(self.timestamps)
    
    def to_trajectory(self) -> Trajectory:
        """Convert to Trajectory object."""
        points = []
        for i in range(len(self.timestamps)):
            point = TrajectoryPoint(
                vehicle_id=self.vehicle_id,
                x=self.x_positions[i],
                y=self.y_positions[i],
                timestamp=datetime.fromtimestamp(self.timestamps[i]) if isinstance(self.timestamps[i], (int, float)) else self.timestamps[i],
                velocity=self.velocities[i],
                acceleration=self.accelerations[i],
                heading=self.headings[i]
            )
            points.append(point)
        
        start_time = points[0].timestamp
        end_time = points[-1].timestamp
        duration = (end_time - start_time).total_seconds()
        total_distance = self._calculate_total_distance()
        
        return Trajectory(
            vehicle_id=self.vehicle_id,
            points=points,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            total_distance=total_distance,
            metadata=self.metadata
        )
    
    def _calculate_total_distance(self) -> float:
        """Calculate total distance of the trajectory."""
        if len(self.x_positions) < 2:
            return 0.0
        
        distances = np.sqrt(
            np.diff(self.x_positions) ** 2 + np.diff(self.y_positions) ** 2
        )
        return float(np.sum(distances))
    
    def get_segment(self, start_idx: int, end_idx: int) -> "TrajectoryData":
        """Get trajectory segment between two indices."""
        if start_idx < 0 or end_idx > len(self.timestamps) or start_idx >= end_idx:
            raise ValueError("Invalid indices for trajectory segment")
        
        return TrajectoryData(
            vehicle_id=self.vehicle_id,
            timestamps=self.timestamps[start_idx:end_idx],
            x_positions=self.x_positions[start_idx:end_idx],
            y_positions=self.y_positions[start_idx:end_idx],
            velocities=self.velocities[start_idx:end_idx],
            headings=self.headings[start_idx:end_idx],
            accelerations=self.accelerations[start_idx:end_idx],
            metadata=self.metadata.copy()
        )