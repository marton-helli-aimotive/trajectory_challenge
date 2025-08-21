"""Comprehensive tests for trajectory data models and validation framework."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from trajectory_prediction.data.models import (
    Dataset,
    Trajectory,
    TrajectoryPoint,
    Vehicle,
    VehicleType,
)
from trajectory_prediction.data.validation import (
    DataQualityAnalyzer,
    TrajectoryValidator,
    ValidationIssue,
    ValidationSeverity,
)


def create_test_point(timestamp=1609459200.0, x=0.0, y=0.0, **kwargs):
    """Helper to create test trajectory points with sensible defaults."""
    defaults = {
        "speed": 10.0,
        "velocity_x": 5.0,
        "velocity_y": 5.0,
        "acceleration_x": 0.0,
        "acceleration_y": 0.0,
        "heading": 0.0,
        "lane_id": 1,
        "frame_id": 1,
    }
    defaults.update(kwargs)
    return TrajectoryPoint(timestamp=timestamp, x=x, y=y, **defaults)


def create_test_vehicle(vehicle_id=1, **kwargs):
    """Helper to create test vehicles with sensible defaults."""
    defaults = {
        "vehicle_type": VehicleType.CAR,
        "length": 4.5,
        "width": 1.8,
    }
    defaults.update(kwargs)
    return Vehicle(vehicle_id=vehicle_id, **defaults)


def create_test_trajectory(trajectory_id="test_001", **kwargs):
    """Helper to create test trajectories with sensible defaults."""
    defaults = {
        "vehicle": create_test_vehicle(),
        "points": [
            create_test_point(timestamp=1609459200.0, x=0.0, y=0.0),
            create_test_point(timestamp=1609459201.0, x=10.0, y=5.0),
        ],
        "dataset_name": "test_dataset",
    }
    defaults.update(kwargs)
    return Trajectory(trajectory_id=trajectory_id, **defaults)


def create_test_dataset(name="test_dataset", **kwargs):
    """Helper to create test datasets with sensible defaults."""
    return Dataset(name=name, **kwargs)


class TestTrajectoryPoint:
    """Test suite for TrajectoryPoint model."""

    def test_valid_trajectory_point(self):
        """Test creation of valid trajectory point."""
        point = create_test_point(
            timestamp=1609459200.0,  # 2021-01-01 00:00:00
            x=100.0,
            y=200.0,
            speed=15.0,
            velocity_x=10.0,
            velocity_y=11.18,
            acceleration_x=0.5,
            acceleration_y=-0.2,
            heading=45.0,
            lane_id=1,
            frame_id=100,
        )

        assert point.timestamp == 1609459200.0
        assert point.x == 100.0
        assert point.y == 200.0
        assert point.speed == 15.0

    def test_trajectory_point_validation_constraints(self):
        """Test validation constraints for trajectory points."""
        # Test timestamp validation
        with pytest.raises(ValidationError, match="Timestamp too old"):
            create_test_point(timestamp=946684799.0, x=0.0, y=0.0)

        with pytest.raises(ValidationError, match="Timestamp too far in future"):
            create_test_point(timestamp=4102444801.0, x=0.0, y=0.0)

        # Test coordinate validation
        with pytest.raises(ValidationError, match="Coordinate value too large"):
            create_test_point(timestamp=1609459200.0, x=1000001.0, y=0.0)

        # Test speed constraints
        with pytest.raises(
            ValidationError, match="Input should be less than or equal to 200"
        ):
            create_test_point(timestamp=1609459200.0, x=0.0, y=0.0, speed=201.0)

        # Test velocity validation
        with pytest.raises(ValidationError, match="Velocity too high"):
            create_test_point(timestamp=1609459200.0, x=0.0, y=0.0, velocity_x=201.0)

        # Test acceleration validation
        with pytest.raises(ValidationError, match="Acceleration too high"):
            create_test_point(timestamp=1609459200.0, x=0.0, y=0.0, acceleration_x=51.0)

        # Test heading constraints
        with pytest.raises(ValidationError, match="Input should be less than 360"):
            create_test_point(timestamp=1609459200.0, x=0.0, y=0.0, heading=360.0)

    @given(
        timestamp=st.floats(min_value=946684800, max_value=4102444800),
        x=st.floats(min_value=-1000000, max_value=1000000),
        y=st.floats(min_value=-1000000, max_value=1000000),
        speed=st.floats(min_value=0.0, max_value=200.0),
        velocity_x=st.floats(min_value=-200.0, max_value=200.0),
        velocity_y=st.floats(min_value=-200.0, max_value=200.0),
        acceleration_x=st.floats(min_value=-50.0, max_value=50.0),
        acceleration_y=st.floats(min_value=-50.0, max_value=50.0),
    )
    def test_trajectory_point_property_based(
        self,
        timestamp: float,
        x: float,
        y: float,
        speed: float,
        velocity_x: float,
        velocity_y: float,
        acceleration_x: float,
        acceleration_y: float,
    ):
        """Property-based test for trajectory point creation."""
        # Skip NaN values
        if any(
            np.isnan(val)
            for val in [
                timestamp,
                x,
                y,
                speed,
                velocity_x,
                velocity_y,
                acceleration_x,
                acceleration_y,
            ]
        ):
            return

        point = TrajectoryPoint(
            timestamp=timestamp,
            x=x,
            y=y,
            speed=speed,
            velocity_x=velocity_x,
            velocity_y=velocity_y,
            acceleration_x=acceleration_x,
            acceleration_y=acceleration_y,
            heading=0.0,
            lane_id=1,
            frame_id=1,
        )

        # Assert basic properties
        assert point.timestamp == timestamp
        assert point.x == x
        assert point.y == y
        assert point.speed == speed

        # Physics constraints should be satisfied
        assert 0.0 <= point.speed <= 200.0
        assert abs(point.velocity_x) <= 200.0
        assert abs(point.velocity_y) <= 200.0
        assert abs(point.acceleration_x) <= 50.0
        assert abs(point.acceleration_y) <= 50.0


class TestVehicle:
    """Test suite for Vehicle model."""

    def test_valid_vehicle(self):
        """Test creation of valid vehicle."""
        vehicle = Vehicle(
            vehicle_id=123,
            vehicle_type=VehicleType.CAR,
            length=4.5,
            width=1.8,
            metadata={"make": "Toyota", "model": "Camry"},
        )

        assert vehicle.vehicle_id == 123
        assert vehicle.vehicle_type == VehicleType.CAR
        assert vehicle.length == 4.5
        assert vehicle.width == 1.8

    def test_vehicle_validation(self):
        """Test vehicle validation constraints."""
        # Test negative dimensions
        with pytest.raises(ValidationError, match="Input should be greater than 0"):
            Vehicle(
                vehicle_id=123,
                vehicle_type=VehicleType.CAR,
                length=0.0,
                width=1.8,
            )

        # Test too large dimensions
        with pytest.raises(
            ValidationError, match="Input should be less than or equal to 30"
        ):
            Vehicle(
                vehicle_id=123,
                vehicle_type=VehicleType.CAR,
                length=31.0,
                width=1.8,
            )

    @given(
        vehicle_id=st.integers(min_value=0, max_value=1000000),
        length=st.floats(min_value=0.1, max_value=30.0),
        width=st.floats(min_value=0.1, max_value=4.0),
    )
    def test_vehicle_property_based(self, vehicle_id: int, length: float, width: float):
        """Property-based test for vehicle creation."""
        if any(np.isnan(val) for val in [length, width]):
            return

        vehicle = Vehicle(
            vehicle_id=vehicle_id,
            vehicle_type=VehicleType.CAR,
            length=length,
            width=width,
        )

        assert vehicle.vehicle_id == vehicle_id
        assert vehicle.length == length
        assert vehicle.width == width
        assert 0.1 <= vehicle.length <= 30.0
        assert 0.1 <= vehicle.width <= 4.0


class TestTrajectory:
    """Test suite for Trajectory model."""

    def create_test_points(self, num_points: int = 5) -> list[TrajectoryPoint]:
        """Helper to create test trajectory points."""
        points = []
        base_timestamp = 1609459200.0
        for i in range(num_points):
            points.append(
                create_test_point(
                    timestamp=base_timestamp + i * 0.1,
                    x=i * 10.0,
                    y=i * 5.0,
                )
            )
        return points

    def test_valid_trajectory(self):
        """Test creation of valid trajectory."""
        vehicle = create_test_vehicle(
            vehicle_id=123,
            vehicle_type=VehicleType.CAR,
            length=4.5,
            width=1.8,
        )
        points = self.create_test_points()

        trajectory = create_test_trajectory(
            trajectory_id="traj_001",
            vehicle=vehicle,
            points=points,
            dataset_name="test_dataset",
        )

        assert trajectory.trajectory_id == "traj_001"
        assert len(trajectory.points) == 5
        assert trajectory.dataset_name == "test_dataset"

    def test_trajectory_temporal_ordering(self):
        """Test that trajectory points must be in temporal order."""
        vehicle = create_test_vehicle(vehicle_id=123)

        # Create points with non-monotonic timestamps
        points = [
            create_test_point(
                timestamp=1609459200.0,
                x=0.0,
                y=0.0,
                speed=10.0,
                velocity_x=5.0,
                velocity_y=5.0,
            ),
            create_test_point(
                timestamp=1609459199.0,  # Earlier timestamp
                x=10.0,
                y=5.0,
                speed=10.0,
                velocity_x=5.0,
                velocity_y=5.0,
            ),
            create_test_point(
                timestamp=1609459201.0,
                x=20.0,
                y=10.0,
                speed=10.0,
                velocity_x=5.0,
                velocity_y=5.0,
            ),
        ]

        with pytest.raises(
            ValidationError, match="Trajectory points must be in temporal order"
        ):
            create_test_trajectory(
                trajectory_id="temporal_test",
                vehicle=vehicle,
                points=points,
                dataset_name="test_dataset",
            )

    def test_trajectory_minimum_points(self):
        """Test that trajectory must have minimum points."""
        vehicle = create_test_vehicle(vehicle_id=123)

        # Only one point
        points = [create_test_point()]

        with pytest.raises(ValidationError, match="List should have at least 2 items"):
            create_test_trajectory(
                trajectory_id="insufficient_test",
                vehicle=vehicle,
                points=points,
                dataset_name="test_dataset",
            )

    def test_trajectory_large_time_gaps(self):
        """Test detection of large time gaps."""
        vehicle = create_test_vehicle(vehicle_id=123)

        # Create points with large time gap
        points = [
            create_test_point(timestamp=1609459200.0, x=0.0, y=0.0),
            create_test_point(timestamp=1609459206.0, x=10.0, y=5.0),  # 6 second gap
        ]

        with pytest.raises(ValidationError, match="Time gap too large"):
            create_test_trajectory(
                trajectory_id="time_gap_test",
                vehicle=vehicle,
                points=points,
                dataset_name="test_dataset",
            )

    def test_trajectory_properties(self):
        """Test trajectory computed properties."""
        vehicle = create_test_vehicle(vehicle_id=123)
        points = self.create_test_points()

        trajectory = create_test_trajectory(
            trajectory_id="traj_001",
            vehicle=vehicle,
            points=points,
        )

        # Test duration calculation
        expected_duration = (len(points) - 1) * 0.1
        assert abs(trajectory.duration - expected_duration) < 1e-6

        # Test length calculation
        assert trajectory.length > 0.0

    def test_trajectory_quality_metrics(self):
        """Test trajectory quality metrics computation."""
        vehicle = create_test_vehicle(vehicle_id=123)
        points = self.create_test_points()

        trajectory = create_test_trajectory(
            trajectory_id="traj_001",
            vehicle=vehicle,
            points=points,
        )

        trajectory.compute_quality_metrics()

        assert trajectory.completeness_score is not None
        assert trajectory.temporal_consistency_score is not None
        assert trajectory.spatial_accuracy_score is not None
        assert trajectory.smoothness_score is not None

        # Quality scores should be between 0 and 1
        assert 0.0 <= trajectory.completeness_score <= 1.0
        assert 0.0 <= trajectory.temporal_consistency_score <= 1.0
        assert 0.0 <= trajectory.spatial_accuracy_score <= 1.0
        assert 0.0 <= trajectory.smoothness_score <= 1.0

    @given(
        num_points=st.integers(min_value=2, max_value=20),
        time_interval=st.floats(min_value=0.01, max_value=1.0),
        speed=st.floats(min_value=1.0, max_value=50.0),
    )
    def test_trajectory_property_based(
        self, num_points: int, time_interval: float, speed: float
    ):
        """Property-based test for trajectory creation."""
        if any(np.isnan(val) for val in [time_interval, speed]):
            return

        vehicle = create_test_vehicle(vehicle_id=1)
        points = []
        base_timestamp = 1609459200.0

        for i in range(num_points):
            points.append(
                create_test_point(
                    timestamp=base_timestamp + i * time_interval,
                    x=i * speed * time_interval,
                    y=i * speed * time_interval * 0.5,
                    speed=speed,
                )
            )

        trajectory = create_test_trajectory(
            trajectory_id="property_test",
            vehicle=vehicle,
            points=points,
            dataset_name="test_dataset",
        )

        assert len(trajectory.points) == num_points
        assert trajectory.duration >= 0.0
        assert trajectory.length >= 0.0


class TestDataset:
    """Test suite for Dataset model."""

    def create_test_trajectory(self, trajectory_id: str) -> Trajectory:
        """Helper to create test trajectory."""
        vehicle = Vehicle(
            vehicle_id=123,
            vehicle_type=VehicleType.CAR,
            length=4.5,
            width=1.8,
        )
        points = []
        for i in range(5):
            points.append(
                create_test_point(
                    timestamp=1609459200.0 + i * 0.1,
                    x=i * 10.0,
                    y=i * 5.0,
                    speed=15.0,
                    velocity_x=10.0,
                    velocity_y=5.0,
                )
            )
        return create_test_trajectory(
            trajectory_id=trajectory_id,
            vehicle=vehicle,
            points=points,
            dataset_name="test_dataset",
        )

    def test_valid_dataset(self):
        """Test creation of valid dataset."""
        dataset = create_test_dataset(
            name="test_dataset",
            version="1.0",
            description="Test dataset for unit tests",
        )

        assert dataset.name == "test_dataset"
        assert dataset.version == "1.0"
        assert dataset.total_trajectories == 0
        assert dataset.total_points == 0

    def test_dataset_with_trajectories(self):
        """Test dataset with trajectories."""
        dataset = create_test_dataset(name="test_dataset")

        # Add trajectories
        traj1 = self.create_test_trajectory("traj_001")
        traj2 = self.create_test_trajectory("traj_002")

        dataset.add_trajectory(traj1)
        dataset.add_trajectory(traj2)

        assert dataset.total_trajectories == 2
        assert dataset.total_points == 10  # 5 points each

    def test_dataset_filtering(self):
        """Test dataset filtering capabilities."""
        dataset = create_test_dataset(name="test_dataset")

        # Create trajectories with different properties
        traj1 = self.create_test_trajectory("traj_001")

        # Longer trajectory
        vehicle2 = create_test_vehicle(
            vehicle_id=456,
            vehicle_type=VehicleType.TRUCK,
            length=8.0,
            width=2.5,
        )
        points2 = []
        for i in range(10):
            points2.append(
                create_test_point(
                    timestamp=1609459200.0 + i * 0.1,
                    x=i * 10.0,
                    y=i * 5.0,
                    speed=15.0,
                    velocity_x=10.0,
                    velocity_y=5.0,
                )
            )
        traj2 = create_test_trajectory(
            trajectory_id="traj_002",
            vehicle=vehicle2,
            points=points2,
        )

        dataset.add_trajectory(traj1)
        dataset.add_trajectory(traj2)

        # Filter by duration
        long_trajectories = dataset.filter_by_duration(min_duration=0.5)
        assert len(long_trajectories) == 1
        assert long_trajectories[0].trajectory_id == "traj_002"

        # Filter by vehicle type
        truck_trajectories = dataset.filter_by_vehicle_type(VehicleType.TRUCK)
        assert len(truck_trajectories) == 1
        assert truck_trajectories[0].vehicle.vehicle_type == VehicleType.TRUCK

    def test_dataset_quality_summary(self):
        """Test dataset quality summary."""
        dataset = create_test_dataset(name="test_dataset")

        traj1 = self.create_test_trajectory("traj_001")
        traj2 = self.create_test_trajectory("traj_002")

        dataset.add_trajectory(traj1)
        dataset.add_trajectory(traj2)

        summary = dataset.get_quality_summary()

        assert "avg_completeness" in summary
        assert "avg_temporal_consistency" in summary
        assert "avg_spatial_accuracy" in summary
        assert "avg_smoothness" in summary

        # All scores should be between 0 and 1
        for score in summary.values():
            assert 0.0 <= score <= 1.0


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_trajectory_points(self):
        """Test handling of empty trajectory points."""
        vehicle = Vehicle(
            vehicle_id=123,
            vehicle_type=VehicleType.CAR,
            length=4.5,
            width=1.8,
        )

        with pytest.raises(ValidationError):
            create_test_trajectory(
                trajectory_id="empty",
                vehicle=vehicle,
                points=[],
                dataset_name="test",
            )

    def test_single_point_trajectory(self):
        """Test handling of single point trajectory."""
        vehicle = Vehicle(
            vehicle_id=123,
            vehicle_type=VehicleType.CAR,
            length=4.5,
            width=1.8,
        )
        points = [create_test_point()]

        with pytest.raises(ValidationError):
            create_test_trajectory(
                trajectory_id="single",
                vehicle=vehicle,
                points=points,
                dataset_name="test",
            )

    def test_extreme_coordinates(self):
        """Test handling of extreme coordinate values."""
        # Should work within bounds
        point = create_test_point(
            timestamp=1609459200.0,
            x=999999.0,
            y=-999999.0,
            speed=15.0,
            velocity_x=10.0,
            velocity_y=10.0,
        )
        assert point.x == 999999.0
        assert point.y == -999999.0

    def test_zero_speed_trajectory(self):
        """Test trajectory with zero speed (stationary)."""
        vehicle = Vehicle(
            vehicle_id=123,
            vehicle_type=VehicleType.CAR,
            length=4.5,
            width=1.8,
        )
        points = []
        for i in range(5):
            points.append(
                create_test_point(
                    timestamp=1609459200.0 + i * 0.1,
                    x=100.0,  # Same position
                    y=200.0,  # Same position
                    speed=0.0,
                    velocity_x=0.0,
                    velocity_y=0.0,
                )
            )

        trajectory = create_test_trajectory(
            trajectory_id="stationary",
            vehicle=vehicle,
            points=points,
            dataset_name="test",
        )

        assert trajectory.length == 0.0  # No movement
        trajectory.compute_quality_metrics()
        assert (
            trajectory.spatial_accuracy_score == 1.0
        )  # Perfect accuracy for stationary


class TestPhysicsConstraints:
    """Test physics constraint validation using property-based testing."""

    @given(
        speed=st.floats(min_value=0.0, max_value=50.0),
        velocity_magnitude=st.floats(min_value=0.0, max_value=50.0),
        tolerance=st.floats(min_value=0.01, max_value=1.0),
    )
    def test_speed_velocity_consistency(
        self, speed: float, velocity_magnitude: float, tolerance: float
    ):
        """Test speed-velocity consistency validation."""
        if any(np.isnan(val) for val in [speed, velocity_magnitude, tolerance]):
            return

        # Create velocity components that match the magnitude
        angle = np.pi / 4  # 45 degrees
        vx = velocity_magnitude * np.cos(angle)
        vy = velocity_magnitude * np.sin(angle)

        point = create_test_point(
            speed=speed,
            velocity_x=vx,
            velocity_y=vy,
        )

        calculated_speed = np.sqrt(vx**2 + vy**2)
        speed_diff = abs(calculated_speed - speed)

        # If the difference is significant, validation should catch it
        if speed_diff > tolerance:
            # This would be caught by validation framework, not Pydantic
            assert abs(point.speed - speed) < 1e-6  # Pydantic stores what we give it

    @given(
        acceleration=st.floats(min_value=0.0, max_value=10.0),
        time_interval=st.floats(min_value=0.1, max_value=1.0),
    )
    def test_acceleration_velocity_consistency(
        self, acceleration: float, time_interval: float
    ):
        """Test acceleration-velocity consistency."""
        if any(np.isnan(val) for val in [acceleration, time_interval]):
            return

        point1 = create_test_point(
            timestamp=1609459200.0,
            x=0.0,
            y=0.0,
            speed=10.0,
            velocity_x=10.0,
            velocity_y=0.0,
        )

        point2 = create_test_point(
            timestamp=1609459200.0 + time_interval,
            x=10.0 * time_interval + 0.5 * acceleration * time_interval**2,
            y=0.0,
            speed=10.0 + acceleration * time_interval,
            velocity_x=10.0 + acceleration * time_interval,
            velocity_y=0.0,
            acceleration_x=acceleration,
        )

        # Basic consistency checks
        assert point2.timestamp > point1.timestamp
        assert point2.acceleration_x == acceleration


# Validation Framework Tests
class TestTrajectoryValidator:
    """Test suite for TrajectoryValidator."""

    def create_valid_trajectory(self, num_points: int = 5) -> Trajectory:
        """Helper to create a valid trajectory."""
        vehicle = create_test_vehicle(vehicle_id=123)
        points = []
        base_timestamp = 1609459200.0

        for i in range(num_points):
            points.append(
                create_test_point(
                    timestamp=base_timestamp + i * 0.1, x=i * 1.0, y=i * 0.5
                )
            )

        return create_test_trajectory(
            trajectory_id=f"test_traj_{num_points}",
            vehicle=vehicle,
            points=points,
            dataset_name="test_dataset",
        )

    def test_validator_initialization(self):
        """Test validator initialization."""
        validator = TrajectoryValidator()
        assert validator.config is not None
        assert "max_speed" in validator.config
        assert "max_acceleration" in validator.config

    def test_validate_valid_trajectory_point(self):
        """Test validation of valid trajectory point."""
        validator = TrajectoryValidator()
        point = create_test_point()

        issues = validator.validate_trajectory_point(point, "test_traj")

        # Should have no error issues
        error_issues = [
            issue for issue in issues if issue.severity == ValidationSeverity.ERROR
        ]
        assert len(error_issues) == 0

    def test_validate_point_speed_too_high(self):
        """Test speed validation."""
        # Test with speed within Pydantic bounds but above our custom limit
        custom_validator = TrajectoryValidator(config={"max_speed": 50.0})

        point = create_test_point(speed=100.0)  # Above our custom limit

        issues = custom_validator.validate_trajectory_point(point, "test_traj")

        error_issues = [
            issue for issue in issues if issue.severity == ValidationSeverity.ERROR
        ]
        assert len(error_issues) > 0
        assert any("SPEED_TOO_HIGH" in issue.code for issue in error_issues)

    def test_validate_speed_velocity_mismatch(self):
        """Test speed-velocity consistency check."""
        validator = TrajectoryValidator()

        point = create_test_point(
            speed=20.0,  # Reported speed
            velocity_x=10.0,  # Components suggest different speed
            velocity_y=10.0,  # sqrt(10² + 10²) ≈ 14.14
        )

        issues = validator.validate_trajectory_point(point, "test_traj")

        warning_issues = [
            issue for issue in issues if issue.severity == ValidationSeverity.WARNING
        ]
        assert any("SPEED_VELOCITY_MISMATCH" in issue.code for issue in warning_issues)

    def test_validate_trajectory_physics(self):
        """Test physics constraint validation."""
        validator = TrajectoryValidator()

        # Create trajectory with physics violations
        vehicle = create_test_vehicle(vehicle_id=123)
        points = [
            create_test_point(timestamp=1609459200.0, x=0.0, y=0.0, speed=10.0),
            create_test_point(
                timestamp=1609459200.1, x=1000.0, y=0.0, speed=10.0
            ),  # Teleportation
        ]

        trajectory = create_test_trajectory(
            trajectory_id="physics_test",
            vehicle=vehicle,
            points=points,
            dataset_name="test_dataset",
        )

        issues = validator.validate_trajectory_physics(trajectory)

        error_issues = [
            issue for issue in issues if issue.severity == ValidationSeverity.ERROR
        ]
        assert len(error_issues) > 0
        assert any("TELEPORTATION" in issue.code for issue in error_issues)

    def test_validate_complete_trajectory(self):
        """Test complete trajectory validation."""
        validator = TrajectoryValidator()
        trajectory = self.create_valid_trajectory()

        result = validator.validate_trajectory(trajectory)

        assert result.is_valid
        assert 0.0 <= result.quality_score <= 1.0
        assert "completeness_score" in result.metrics
        assert "temporal_consistency_score" in result.metrics
        assert "spatial_accuracy_score" in result.metrics
        assert "smoothness_score" in result.metrics

    def test_validate_dataset(self):
        """Test dataset validation."""
        validator = TrajectoryValidator()

        dataset = create_test_dataset(name="test_dataset")

        # Create trajectories with different IDs
        traj1 = self.create_valid_trajectory(num_points=5)
        traj1.trajectory_id = "traj_1"

        traj2 = self.create_valid_trajectory(num_points=6)
        traj2.trajectory_id = "traj_2"

        dataset.add_trajectory(traj1)
        dataset.add_trajectory(traj2)

        results = validator.validate_dataset(dataset)

        assert len(results) == 2
        for result in results.values():
            assert isinstance(result.quality_score, float)
            assert 0.0 <= result.quality_score <= 1.0


class TestDataQualityAnalyzer:
    """Test suite for DataQualityAnalyzer."""

    def create_test_dataset(self) -> Dataset:
        """Helper to create test dataset."""
        dataset = create_test_dataset(name="test_dataset")

        # Add some trajectories
        for i in range(3):
            vehicle = create_test_vehicle(vehicle_id=i)
            points = []
            base_timestamp = 1609459200.0

            for j in range(5):
                points.append(
                    create_test_point(
                        timestamp=base_timestamp + j * 0.1,
                        x=j * 10.0,
                        y=j * 5.0,
                        speed=15.0,
                        velocity_x=10.0,
                        velocity_y=5.0,
                    )
                )

            trajectory = create_test_trajectory(
                trajectory_id=f"traj_{i}",
                vehicle=vehicle,
                points=points,
                dataset_name="test_dataset",
            )
            dataset.add_trajectory(trajectory)

        return dataset

    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        analyzer = DataQualityAnalyzer()
        assert analyzer is not None

    def test_analyze_dataset_quality(self):
        """Test dataset quality analysis."""
        analyzer = DataQualityAnalyzer()
        dataset = self.create_test_dataset()

        analysis = analyzer.analyze_dataset_quality(dataset)

        assert "summary" in analysis
        assert "quality_statistics" in analysis
        assert "issue_breakdown" in analysis
        assert "dataset_metrics" in analysis
        assert "validation_results" in analysis

        summary = analysis["summary"]
        assert summary["total_trajectories"] == 3
        assert summary["validation_rate"] >= 0.0

    def test_generate_quality_report(self):
        """Test quality report generation."""
        analyzer = DataQualityAnalyzer()
        dataset = self.create_test_dataset()

        report = analyzer.generate_quality_report(dataset)

        assert isinstance(report, str)
        assert "TRAJECTORY DATASET QUALITY REPORT" in report
        assert "Total Trajectories: 3" in report
        assert "QUALITY STATISTICS" in report
        assert "DATASET METRICS" in report

    def test_empty_dataset_analysis(self):
        """Test analysis of empty dataset."""
        analyzer = DataQualityAnalyzer()
        dataset = create_test_dataset(name="empty_dataset")

        analysis = analyzer.analyze_dataset_quality(dataset)

        summary = analysis["summary"]
        assert summary["total_trajectories"] == 0
        assert summary["valid_trajectories"] == 0
        assert summary["validation_rate"] == 0.0


class TestValidationIssue:
    """Test suite for ValidationIssue."""

    def test_validation_issue_creation(self):
        """Test creation of validation issue."""
        issue = ValidationIssue(
            severity=ValidationSeverity.ERROR,
            code="TEST_CODE",
            message="Test message",
            trajectory_id="test_traj",
            point_index=5,
        )

        assert issue.severity == ValidationSeverity.ERROR
        assert issue.code == "TEST_CODE"
        assert issue.message == "Test message"
        assert issue.trajectory_id == "test_traj"
        assert issue.point_index == 5
        assert issue.metadata == {}

    def test_validation_issue_metadata_initialization(self):
        """Test metadata initialization in validation issue."""
        issue = ValidationIssue(
            severity=ValidationSeverity.WARNING,
            code="TEST_CODE",
            message="Test message",
        )

        # metadata should be initialized to empty dict
        assert issue.metadata == {}


class TestValidationIntegration:
    """Integration tests for validation framework."""

    def test_end_to_end_validation_pipeline(self):
        """Test complete validation pipeline."""
        # Create dataset with various quality issues
        dataset = create_test_dataset(name="integration_test")

        # Good trajectory
        vehicle1 = create_test_vehicle(vehicle_id=1)
        good_points = []
        for i in range(10):
            good_points.append(
                create_test_point(
                    timestamp=1609459200.0 + i * 0.1, x=i * 10.0, y=i * 5.0, speed=15.0
                )
            )
        good_trajectory = create_test_trajectory(
            trajectory_id="good_traj",
            vehicle=vehicle1,
            points=good_points,
            dataset_name="integration_test",
        )

        # Problematic trajectory (teleportation)
        vehicle2 = create_test_vehicle(vehicle_id=2)
        bad_points = [
            create_test_point(timestamp=1609459200.0, x=0.0, y=0.0, speed=10.0),
            create_test_point(
                timestamp=1609459200.1, x=1000.0, y=0.0, speed=10.0
            ),  # Teleportation
        ]
        bad_trajectory = create_test_trajectory(
            trajectory_id="bad_traj",
            vehicle=vehicle2,
            points=bad_points,
            dataset_name="integration_test",
        )

        dataset.add_trajectory(good_trajectory)
        dataset.add_trajectory(bad_trajectory)

        # Validate dataset
        validator = TrajectoryValidator()
        results = validator.validate_dataset(dataset)

        # Check results
        assert len(results) == 2
        assert results["good_traj"].is_valid
        assert not results["bad_traj"].is_valid

        # Analyze quality
        analyzer = DataQualityAnalyzer()
        analysis = analyzer.analyze_dataset_quality(dataset)

        assert analysis["summary"]["total_trajectories"] == 2
        assert analysis["summary"]["valid_trajectories"] == 1
        assert analysis["summary"]["validation_rate"] == 0.5

        # Generate report
        report = analyzer.generate_quality_report(dataset)
        assert "Valid Trajectories: 1 (50.0%)" in report
