"""
Unit tests for data schemas and validation.

This module tests:
- TrajectoryData validation and structure
- Position and Velocity data classes
- Schema validation and error handling
"""

import pytest
import numpy as np
from datetime import datetime
from typing import List

from src.trajectory_prediction.data.schemas import TrajectoryData, Position, Velocity
from tests.conftest import validate_trajectory_data


class TestPosition:
    """Test Position data class."""
    
    def test_position_creation(self):
        """Test basic position creation."""
        pos = Position(x=1.0, y=2.0)
        assert pos.x == 1.0
        assert pos.y == 2.0
    
    def test_position_distance(self):
        """Test distance calculation between positions."""
        pos1 = Position(x=0.0, y=0.0)
        pos2 = Position(x=3.0, y=4.0)
        
        distance = pos1.distance_to(pos2)
        assert abs(distance - 5.0) < 1e-6
    
    def test_position_equality(self):
        """Test position equality comparison."""
        pos1 = Position(x=1.0, y=2.0)
        pos2 = Position(x=1.0, y=2.0)
        pos3 = Position(x=1.1, y=2.0)
        
        assert pos1 == pos2
        assert pos1 != pos3
    
    def test_position_invalid_types(self):
        """Test position creation with invalid types."""
        with pytest.raises((TypeError, ValueError)):
            Position(x="invalid", y=2.0)
        
        with pytest.raises((TypeError, ValueError)):
            Position(x=1.0, y=None)
    
    @pytest.mark.parametrize("x,y", [
        (0.0, 0.0),
        (-1.0, 1.0),
        (1000.0, -1000.0),
        (np.inf, 0.0),
        (0.0, -np.inf),
    ])
    def test_position_edge_cases(self, x, y):
        """Test position creation with edge case values."""
        if np.isinf(x) or np.isinf(y):
            with pytest.raises(ValueError):
                Position(x=x, y=y)
        else:
            pos = Position(x=x, y=y)
            assert pos.x == x
            assert pos.y == y


class TestVelocity:
    """Test Velocity data class."""
    
    def test_velocity_creation(self):
        """Test basic velocity creation."""
        vel = Velocity(vx=1.0, vy=2.0)
        assert vel.vx == 1.0
        assert vel.vy == 2.0
    
    def test_velocity_magnitude(self):
        """Test velocity magnitude calculation."""
        vel = Velocity(vx=3.0, vy=4.0)
        assert abs(vel.magnitude - 5.0) < 1e-6
    
    def test_velocity_direction(self):
        """Test velocity direction calculation."""
        vel = Velocity(vx=1.0, vy=1.0)
        direction = vel.direction
        expected_direction = np.pi / 4  # 45 degrees
        assert abs(direction - expected_direction) < 1e-6
    
    def test_velocity_zero_magnitude(self):
        """Test zero velocity magnitude."""
        vel = Velocity(vx=0.0, vy=0.0)
        assert vel.magnitude == 0.0
    
    def test_velocity_operations(self):
        """Test velocity arithmetic operations."""
        vel1 = Velocity(vx=1.0, vy=2.0)
        vel2 = Velocity(vx=3.0, vy=4.0)
        
        # Addition
        vel_sum = vel1 + vel2
        assert vel_sum.vx == 4.0
        assert vel_sum.vy == 6.0
        
        # Subtraction
        vel_diff = vel2 - vel1
        assert vel_diff.vx == 2.0
        assert vel_diff.vy == 2.0
        
        # Scalar multiplication
        vel_scaled = vel1 * 2.0
        assert vel_scaled.vx == 2.0
        assert vel_scaled.vy == 4.0


class TestTrajectoryData:
    """Test TrajectoryData schema."""
    
    def test_trajectory_creation(self, sample_trajectory):
        """Test basic trajectory creation."""
        assert sample_trajectory.trajectory_id == "test_001"
        assert sample_trajectory.vehicle_id == "vehicle_001"
        assert len(sample_trajectory.positions) == 20
        assert len(sample_trajectory.velocities) == 20
        assert len(sample_trajectory.timestamps) == 20
        assert validate_trajectory_data(sample_trajectory)
    
    def test_trajectory_validation_missing_positions(self):
        """Test trajectory validation with missing positions."""
        trajectory = TrajectoryData(
            trajectory_id="test",
            vehicle_id="vehicle",
            positions=[],
            velocities=[],
            timestamps=[]
        )
        assert not validate_trajectory_data(trajectory)
    
    def test_trajectory_validation_mismatched_lengths(self, sample_positions, sample_velocities, sample_timestamps):
        """Test trajectory validation with mismatched array lengths."""
        # Truncate velocities
        short_velocities = sample_velocities[:10]
        
        trajectory = TrajectoryData(
            trajectory_id="test",
            vehicle_id="vehicle",
            positions=sample_positions,  # Length 20
            velocities=short_velocities,  # Length 10
            timestamps=sample_timestamps
        )
        assert not validate_trajectory_data(trajectory)
    
    def test_trajectory_validation_unordered_timestamps(self, sample_positions, sample_velocities):
        """Test trajectory validation with unordered timestamps."""
        # Create unordered timestamps
        unordered_timestamps = [1.0, 3.0, 2.0, 4.0, 5.0] + list(range(6, 16))
        
        trajectory = TrajectoryData(
            trajectory_id="test",
            vehicle_id="vehicle",
            positions=sample_positions,
            velocities=sample_velocities,
            timestamps=unordered_timestamps
        )
        assert not validate_trajectory_data(trajectory)
    
    def test_trajectory_duration(self, sample_trajectory):
        """Test trajectory duration calculation."""
        duration = sample_trajectory.duration
        expected_duration = sample_trajectory.timestamps[-1] - sample_trajectory.timestamps[0]
        assert abs(duration - expected_duration) < 1e-6
    
    def test_trajectory_length(self, sample_trajectory):
        """Test trajectory length calculation."""
        length = sample_trajectory.total_length
        assert length > 0
        
        # Verify length calculation
        expected_length = 0.0
        for i in range(1, len(sample_trajectory.positions)):
            expected_length += sample_trajectory.positions[i-1].distance_to(sample_trajectory.positions[i])
        
        assert abs(length - expected_length) < 1e-6
    
    def test_trajectory_speed_statistics(self, sample_trajectory):
        """Test trajectory speed statistics."""
        if sample_trajectory.velocities:
            speeds = [vel.magnitude for vel in sample_trajectory.velocities]
            
            assert sample_trajectory.average_speed == np.mean(speeds)
            assert sample_trajectory.max_speed == np.max(speeds)
            assert sample_trajectory.min_speed == np.min(speeds)
    
    def test_trajectory_bounding_box(self, sample_trajectory):
        """Test trajectory bounding box calculation."""
        bbox = sample_trajectory.bounding_box
        
        x_coords = [pos.x for pos in sample_trajectory.positions]
        y_coords = [pos.y for pos in sample_trajectory.positions]
        
        assert bbox['min_x'] == min(x_coords)
        assert bbox['max_x'] == max(x_coords)
        assert bbox['min_y'] == min(y_coords)
        assert bbox['max_y'] == max(y_coords)
    
    def test_trajectory_sampling(self, sample_trajectory):
        """Test trajectory sampling functionality."""
        # Sample every 2nd point
        sampled = sample_trajectory.sample(step=2)
        
        assert len(sampled.positions) == len(sample_trajectory.positions) // 2
        assert len(sampled.velocities) == len(sample_trajectory.velocities) // 2
        assert len(sampled.timestamps) == len(sample_trajectory.timestamps) // 2
        
        # Check that sampled points match original
        for i, pos in enumerate(sampled.positions):
            original_pos = sample_trajectory.positions[i * 2]
            assert pos.x == original_pos.x
            assert pos.y == original_pos.y
    
    def test_trajectory_truncation(self, sample_trajectory):
        """Test trajectory truncation functionality."""
        # Truncate to first 10 points
        truncated = sample_trajectory.truncate(max_length=10)
        
        assert len(truncated.positions) == 10
        assert len(truncated.velocities) == 10
        assert len(truncated.timestamps) == 10
        
        # Check that truncated points match original
        for i in range(10):
            assert truncated.positions[i] == sample_trajectory.positions[i]
    
    def test_trajectory_metadata(self, sample_trajectory):
        """Test trajectory metadata handling."""
        assert "source" in sample_trajectory.metadata
        assert sample_trajectory.metadata["source"] == "test"
        
        # Test metadata update
        sample_trajectory.add_metadata("test_field", "test_value")
        assert sample_trajectory.metadata["test_field"] == "test_value"
    
    def test_trajectory_serialization(self, sample_trajectory):
        """Test trajectory serialization to/from dict."""
        # Serialize to dict
        trajectory_dict = sample_trajectory.to_dict()
        
        assert isinstance(trajectory_dict, dict)
        assert trajectory_dict["trajectory_id"] == sample_trajectory.trajectory_id
        assert len(trajectory_dict["positions"]) == len(sample_trajectory.positions)
        
        # Deserialize from dict
        reconstructed = TrajectoryData.from_dict(trajectory_dict)
        
        assert reconstructed.trajectory_id == sample_trajectory.trajectory_id
        assert len(reconstructed.positions) == len(sample_trajectory.positions)
        assert len(reconstructed.velocities) == len(sample_trajectory.velocities)
    
    @pytest.mark.parametrize("trajectory_length", [1, 5, 50, 100, 1000])
    def test_trajectory_various_lengths(self, trajectory_length):
        """Test trajectory creation with various lengths."""
        positions = [Position(x=float(i), y=float(i*0.5)) for i in range(trajectory_length)]
        velocities = [Velocity(vx=1.0, vy=0.5) for _ in range(trajectory_length)]
        timestamps = [float(i * 0.1) for i in range(trajectory_length)]
        
        trajectory = TrajectoryData(
            trajectory_id=f"test_{trajectory_length}",
            vehicle_id="vehicle",
            positions=positions,
            velocities=velocities,
            timestamps=timestamps
        )
        
        assert len(trajectory.positions) == trajectory_length
        assert validate_trajectory_data(trajectory)


class TestTrajectoryValidation:
    """Test trajectory validation functions."""
    
    def test_valid_trajectory(self, sample_trajectory):
        """Test validation of valid trajectory."""
        assert validate_trajectory_data(sample_trajectory)
    
    def test_empty_trajectory(self):
        """Test validation of empty trajectory."""
        trajectory = TrajectoryData(
            trajectory_id="empty",
            vehicle_id="vehicle",
            positions=[],
            velocities=[],
            timestamps=[]
        )
        assert not validate_trajectory_data(trajectory)
    
    def test_mismatched_arrays(self, sample_positions, sample_velocities, sample_timestamps):
        """Test validation with mismatched array lengths."""
        # Test different combinations of mismatched lengths
        test_cases = [
            (sample_positions[:10], sample_velocities, sample_timestamps),  # Short positions
            (sample_positions, sample_velocities[:10], sample_timestamps),  # Short velocities
            (sample_positions, sample_velocities, sample_timestamps[:10]),  # Short timestamps
        ]
        
        for positions, velocities, timestamps in test_cases:
            trajectory = TrajectoryData(
                trajectory_id="test",
                vehicle_id="vehicle",
                positions=positions,
                velocities=velocities,
                timestamps=timestamps
            )
            assert not validate_trajectory_data(trajectory)
    
    def test_invalid_timestamps(self, sample_positions, sample_velocities):
        """Test validation with invalid timestamp sequences."""
        # Non-monotonic timestamps
        invalid_timestamps = [1.0, 2.0, 1.5, 3.0] + list(range(4, 16))
        
        trajectory = TrajectoryData(
            trajectory_id="test",
            vehicle_id="vehicle",
            positions=sample_positions,
            velocities=sample_velocities,
            timestamps=invalid_timestamps
        )
        assert not validate_trajectory_data(trajectory)
    
    def test_duplicate_timestamps(self, sample_positions, sample_velocities):
        """Test validation with duplicate timestamps."""
        # Duplicate timestamps
        duplicate_timestamps = [1.0, 2.0, 2.0, 3.0] + list(range(4, 16))
        
        trajectory = TrajectoryData(
            trajectory_id="test",
            vehicle_id="vehicle",
            positions=sample_positions,
            velocities=sample_velocities,
            timestamps=duplicate_timestamps
        )
        assert not validate_trajectory_data(trajectory)


# Property-based tests
try:
    from hypothesis import given, strategies as st
    from hypothesis import assume, settings
    
    class TestTrajectoryProperties:
        """Property-based tests for trajectory data."""
        
        @given(st.integers(min_value=1, max_value=100))
        def test_trajectory_length_property(self, length):
            """Test that trajectory length is preserved."""
            positions = [Position(x=float(i), y=float(i)) for i in range(length)]
            velocities = [Velocity(vx=1.0, vy=1.0) for _ in range(length)]
            timestamps = [float(i) for i in range(length)]
            
            trajectory = TrajectoryData(
                trajectory_id="test",
                vehicle_id="vehicle",
                positions=positions,
                velocities=velocities,
                timestamps=timestamps
            )
            
            assert len(trajectory.positions) == length
            assert len(trajectory.velocities) == length
            assert len(trajectory.timestamps) == length
        
        @given(
            st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
            st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False)
        )
        def test_position_distance_property(self, x, y):
            """Test position distance properties."""
            pos1 = Position(x=0.0, y=0.0)
            pos2 = Position(x=x, y=y)
            
            distance = pos1.distance_to(pos2)
            
            # Distance should be non-negative
            assert distance >= 0
            
            # Distance to self should be zero
            assert pos1.distance_to(pos1) == 0
            
            # Distance should be symmetric
            assert abs(distance - pos2.distance_to(pos1)) < 1e-10
        
        @given(
            st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
            st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)
        )
        def test_velocity_magnitude_property(self, vx, vy):
            """Test velocity magnitude properties."""
            vel = Velocity(vx=vx, vy=vy)
            magnitude = vel.magnitude
            
            # Magnitude should be non-negative
            assert magnitude >= 0
            
            # Magnitude should equal sqrt(vx^2 + vy^2)
            expected_magnitude = np.sqrt(vx**2 + vy**2)
            assert abs(magnitude - expected_magnitude) < 1e-10
            
            # Zero velocity should have zero magnitude
            if vx == 0 and vy == 0:
                assert magnitude == 0

except ImportError:
    # Hypothesis not available, skip property-based tests
    pass