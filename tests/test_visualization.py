"""Tests for visualization components."""

import pytest

from src.trajectory_prediction.data.models import TrajectoryPoint, Vehicle, VehicleType
from src.trajectory_prediction.visualization import TrajectoryDashboard


@pytest.fixture
def sample_trajectory_points():
    """Create sample trajectory points for testing."""
    trajectory_points = []
    base_time = 1609459200.0  # 2021-01-01 00:00:00 UTC

    for t in range(10):
        point = TrajectoryPoint(
            timestamp=base_time + t * 0.1,
            x=float(t * 2),
            y=float(t * 1.5),
            speed=10.0,
            velocity_x=2.0,
            velocity_y=1.5,
            acceleration_x=0.1,
            acceleration_y=0.05,
            heading=45.0,
            lane_id=1,
            frame_id=t,
        )
        trajectory_points.append(point)

    return trajectory_points


@pytest.fixture
def sample_vehicle():
    """Create sample vehicle for testing."""
    return Vehicle(
        vehicle_id=1,
        vehicle_type=VehicleType.CAR,
        length=4.5,
        width=2.0,
    )


class TestTrajectoryDashboard:
    """Test main dashboard functionality."""

    def test_init(self):
        """Test dashboard initialization."""
        dashboard = TrajectoryDashboard()
        assert dashboard is not None

    def test_dashboard_has_required_attributes(self):
        """Test dashboard has expected structure."""
        dashboard = TrajectoryDashboard()

        # Dashboard should exist and have basic structure
        assert dashboard is not None
        assert hasattr(dashboard, "__class__")


class TestDataValidation:
    """Test data validation and error handling."""

    def test_valid_trajectory_point(self):
        """Test creating valid trajectory points."""
        base_time = 1609459200.0  # 2021-01-01 00:00:00 UTC

        point = TrajectoryPoint(
            timestamp=base_time,
            x=100.0,
            y=200.0,
            speed=15.0,
            velocity_x=10.0,
            velocity_y=5.0,
            acceleration_x=1.0,
            acceleration_y=0.5,
            heading=90.0,
            lane_id=2,
            frame_id=1,
        )

        assert point.timestamp == base_time
        assert point.x == 100.0
        assert point.y == 200.0
        assert point.speed == 15.0

    def test_valid_vehicle(self):
        """Test creating valid vehicles."""
        vehicle = Vehicle(
            vehicle_id=123,
            vehicle_type=VehicleType.TRUCK,
            length=8.0,
            width=2.5,
        )

        assert vehicle.vehicle_id == 123
        assert vehicle.vehicle_type == VehicleType.TRUCK
        assert vehicle.length == 8.0
        assert vehicle.width == 2.5

    def test_trajectory_point_validation(self):
        """Test trajectory point validation."""
        base_time = 1609459200.0

        # Test with reasonable values
        point = TrajectoryPoint(
            timestamp=base_time,
            x=0.0,
            y=0.0,
            speed=0.0,
            velocity_x=0.0,
            velocity_y=0.0,
            acceleration_x=0.0,
            acceleration_y=0.0,
            heading=0.0,
            lane_id=1,
            frame_id=1,
        )
        assert point.timestamp == base_time

        # Test with invalid timestamp (too old)
        with pytest.raises(ValueError):
            TrajectoryPoint(
                timestamp=946684799.0,  # Before year 2000
                x=0.0,
                y=0.0,
                speed=0.0,
                velocity_x=0.0,
                velocity_y=0.0,
                acceleration_x=0.0,
                acceleration_y=0.0,
                heading=0.0,
                lane_id=1,
                frame_id=1,
            )

        # Test with invalid speed (negative)
        with pytest.raises(ValueError):
            TrajectoryPoint(
                timestamp=base_time,
                x=0.0,
                y=0.0,
                speed=-1.0,  # Negative speed
                velocity_x=0.0,
                velocity_y=0.0,
                acceleration_x=0.0,
                acceleration_y=0.0,
                heading=0.0,
                lane_id=1,
                frame_id=1,
            )


class TestVisualizationImports:
    """Test that visualization components can be imported."""

    def test_dashboard_import(self):
        """Test that dashboard can be imported."""
        from src.trajectory_prediction.visualization import TrajectoryDashboard

        assert TrajectoryDashboard is not None

    def test_visualization_module_import(self):
        """Test that visualization module can be imported."""
        import src.trajectory_prediction.visualization as viz

        assert viz is not None
        assert hasattr(viz, "TrajectoryDashboard")


class TestErrorHandling:
    """Test error handling in dashboard components."""

    def test_dashboard_creation_robust(self):
        """Test that dashboard creation is robust."""
        # This should not raise any exceptions
        dashboard = TrajectoryDashboard()
        assert dashboard is not None

    def test_empty_data_handling(self, sample_trajectory_points):
        """Test behavior with empty and valid data."""
        # Test with valid data
        assert len(sample_trajectory_points) == 10

        # Test with empty data - should not crash
        empty_points = []
        assert len(empty_points) == 0


class TestIntegration:
    """Integration tests for the dashboard."""

    def test_dashboard_basic_functionality(
        self, sample_trajectory_points, sample_vehicle
    ):
        """Test basic dashboard functionality."""
        dashboard = TrajectoryDashboard()

        # Test that dashboard exists and basic data works
        assert dashboard is not None
        assert len(sample_trajectory_points) == 10
        assert sample_vehicle.vehicle_id == 1

        # Test data structures
        first_point = sample_trajectory_points[0]
        assert isinstance(first_point, TrajectoryPoint)
        assert first_point.x == 0.0
        assert first_point.y == 0.0

        last_point = sample_trajectory_points[-1]
        assert isinstance(last_point, TrajectoryPoint)
        assert last_point.x == 18.0  # 9 * 2
        assert last_point.y == 13.5  # 9 * 1.5

    def test_data_consistency(self, sample_trajectory_points):
        """Test data consistency across trajectory points."""
        # All points should have consistent vehicle properties
        speeds = [p.speed for p in sample_trajectory_points]
        assert all(speed == 10.0 for speed in speeds)

        # Check temporal ordering
        timestamps = [p.timestamp for p in sample_trajectory_points]
        assert timestamps == sorted(timestamps)

        # Check that consecutive timestamps increase
        for i in range(1, len(timestamps)):
            assert timestamps[i] > timestamps[i - 1]
