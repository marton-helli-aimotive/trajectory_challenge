"""
Pydantic models for trajectory data validation.

Provides comprehensive validation for trajectory data including:
- Spatial-temporal constraints
- Physics-based validation
- Data quality checks
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, validator


class VehicleClass(str, Enum):
    """Vehicle classification."""
    MOTORCYCLE = "motorcycle"
    CAR = "car" 
    TRUCK = "truck"
    BUS = "bus"
    OTHER = "other"


class TrajectoryPoint(BaseModel):
    """
    Single trajectory point with spatial and temporal information.
    
    Represents a vehicle's state at a specific point in time.
    """
    timestamp: datetime = Field(..., description="Timestamp of the observation")
    x: float = Field(..., description="X coordinate (meters or feet)", ge=-1e6, le=1e6)
    y: float = Field(..., description="Y coordinate (meters or feet)", ge=-1e6, le=1e6)
    
    # Optional kinematic data
    velocity: Optional[float] = Field(None, description="Speed (m/s or ft/s)", ge=0.0, le=200.0)
    acceleration: Optional[float] = Field(None, description="Acceleration (m/s² or ft/s²)", ge=-20.0, le=20.0)
    heading: Optional[float] = Field(None, description="Heading angle (radians)", ge=-3.14159, le=3.14159)
    
    # Optional lane information
    lane_id: Optional[int] = Field(None, description="Lane identifier")
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }


class VehicleInfo(BaseModel):
    """
    Vehicle characteristics and metadata.
    """
    vehicle_id: int = Field(..., description="Unique vehicle identifier", ge=0)
    vehicle_class: VehicleClass = Field(default=VehicleClass.CAR, description="Vehicle type")
    length: float = Field(default=4.5, description="Vehicle length (meters)", ge=1.0, le=30.0)
    width: float = Field(default=1.8, description="Vehicle width (meters)", ge=0.5, le=5.0)
    
    @validator("vehicle_id")
    def validate_vehicle_id(cls, v):
        if v <= 0:
            raise ValueError("Vehicle ID must be positive")
        return v


class TrajectoryData(BaseModel):
    """
    Complete trajectory data with validation and constraints.
    
    Combines spatial-temporal trajectory points with vehicle information
    and provides comprehensive validation for trajectory quality.
    """
    # Core identification
    vehicle_id: int = Field(..., description="Unique vehicle identifier", ge=0)
    timestamp: datetime = Field(..., description="Timestamp of the observation") 
    
    # Position data (required)
    x: float = Field(..., description="X coordinate", ge=-1e6, le=1e6)
    y: float = Field(..., description="Y coordinate", ge=-1e6, le=1e6)
    
    # Kinematic data (optional but recommended)
    velocity: Optional[float] = Field(None, description="Instantaneous velocity", ge=0.0, le=200.0)
    acceleration: Optional[float] = Field(None, description="Instantaneous acceleration", ge=-20.0, le=20.0)
    heading: Optional[float] = Field(None, description="Vehicle heading (radians)", ge=-3.14159, le=3.14159)
    
    # Vehicle characteristics
    vehicle_length: float = Field(default=4.5, description="Vehicle length", ge=1.0, le=30.0)
    vehicle_width: float = Field(default=1.8, description="Vehicle width", ge=0.5, le=5.0)
    vehicle_class: VehicleClass = Field(default=VehicleClass.CAR, description="Vehicle classification")
    
    # Lane and traffic context
    lane_id: Optional[int] = Field(None, description="Current lane identifier")
    preceding_vehicle: Optional[int] = Field(None, description="ID of preceding vehicle")
    following_vehicle: Optional[int] = Field(None, description="ID of following vehicle")
    space_headway: Optional[float] = Field(None, description="Space headway to preceding vehicle", ge=0.0)
    time_headway: Optional[float] = Field(None, description="Time headway to preceding vehicle", ge=0.0)
    
    # Dataset metadata
    dataset: str = Field(default="unknown", description="Source dataset name")
    dataset_variant: Optional[str] = Field(None, description="Dataset variant (e.g., 'i80', 'us101')")
    
    # Optional global coordinates
    global_x: Optional[float] = Field(None, description="Global X coordinate")
    global_y: Optional[float] = Field(None, description="Global Y coordinate")
    
    @validator("timestamp")
    def validate_timestamp(cls, v):
        """Ensure timestamp is reasonable (not too far in future/past)."""
        if v.year < 1990 or v.year > 2030:
            raise ValueError("Timestamp must be between 1990 and 2030")
        return v
    
    @validator("velocity")
    def validate_velocity(cls, v):
        """Validate velocity is physically reasonable."""
        if v is not None and v < 0:
            raise ValueError("Velocity cannot be negative")
        if v is not None and v > 200:  # ~720 km/h, unrealistic for ground vehicles
            raise ValueError("Velocity too high for ground vehicle")
        return v
    
    @validator("acceleration") 
    def validate_acceleration(cls, v):
        """Validate acceleration is physically reasonable."""
        if v is not None and abs(v) > 20:  # ~2g, very high for typical driving
            raise ValueError("Acceleration magnitude too high")
        return v
    
    @validator("space_headway")
    def validate_space_headway(cls, v):
        """Validate space headway is reasonable."""
        if v is not None and v < 0:
            raise ValueError("Space headway cannot be negative")
        if v is not None and v > 1000:  # 1km, unreasonably large
            raise ValueError("Space headway unreasonably large")
        return v
    
    @validator("time_headway")
    def validate_time_headway(cls, v):
        """Validate time headway is reasonable."""
        if v is not None and v < 0:
            raise ValueError("Time headway cannot be negative")
        if v is not None and v > 30:  # 30 seconds, very large gap
            raise ValueError("Time headway unreasonably large")
        return v
    
    @validator("x", "y", "global_x", "global_y")
    def validate_coordinates(cls, v):
        """Validate coordinates are not NaN or infinite."""
        if v is not None:
            import math
            if math.isnan(v) or math.isinf(v):
                raise ValueError("Coordinates cannot be NaN or infinite")
        return v
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }
        validate_assignment = True
        
    def to_point(self) -> TrajectoryPoint:
        """Convert to TrajectoryPoint for simplified access."""
        return TrajectoryPoint(
            timestamp=self.timestamp,
            x=self.x,
            y=self.y,
            velocity=self.velocity,
            acceleration=self.acceleration,
            heading=self.heading,
            lane_id=self.lane_id
        )
    
    def to_vehicle_info(self) -> VehicleInfo:
        """Extract vehicle information."""
        return VehicleInfo(
            vehicle_id=self.vehicle_id,
            vehicle_class=self.vehicle_class,
            length=self.vehicle_length,
            width=self.vehicle_width
        )


class TrajectorySequence(BaseModel):
    """
    Complete trajectory sequence for a single vehicle.
    
    Represents the full trajectory of a vehicle with validation
    for temporal consistency and trajectory quality.
    """
    vehicle_id: int = Field(..., description="Vehicle identifier")
    vehicle_info: VehicleInfo = Field(..., description="Vehicle characteristics")
    points: List[TrajectoryPoint] = Field(..., description="Trajectory points", min_items=1)
    
    # Trajectory metadata
    total_duration: float = Field(..., description="Total trajectory duration (seconds)", ge=0.0)
    total_distance: float = Field(..., description="Total distance traveled", ge=0.0)
    avg_velocity: float = Field(..., description="Average velocity", ge=0.0)
    max_velocity: float = Field(..., description="Maximum velocity", ge=0.0)
    
    @validator("points")
    def validate_temporal_order(cls, v):
        """Ensure trajectory points are temporally ordered."""
        if len(v) < 2:
            return v
            
        for i in range(1, len(v)):
            if v[i].timestamp <= v[i-1].timestamp:
                raise ValueError("Trajectory points must be temporally ordered")
        return v
    
    @validator("points")
    def validate_trajectory_continuity(cls, v):
        """Check for reasonable temporal gaps in trajectory."""
        if len(v) < 2:
            return v
            
        max_gap_seconds = 5.0  # Maximum allowed gap
        for i in range(1, len(v)):
            gap = (v[i].timestamp - v[i-1].timestamp).total_seconds()
            if gap > max_gap_seconds:
                raise ValueError(f"Trajectory gap too large: {gap:.2f} seconds")
        return v
    
    @validator("points") 
    def validate_spatial_continuity(cls, v):
        """Check for reasonable spatial jumps in trajectory."""
        if len(v) < 2:
            return v
            
        max_jump_distance = 100.0  # Maximum allowed spatial jump (meters)
        for i in range(1, len(v)):
            dx = v[i].x - v[i-1].x
            dy = v[i].y - v[i-1].y
            distance = (dx**2 + dy**2)**0.5
            
            time_gap = (v[i].timestamp - v[i-1].timestamp).total_seconds()
            if time_gap > 0:
                implied_velocity = distance / time_gap
                if implied_velocity > 100:  # > 100 m/s = 360 km/h
                    raise ValueError(f"Implied velocity too high: {implied_velocity:.2f} m/s")
        return v