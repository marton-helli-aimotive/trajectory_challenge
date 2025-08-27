"""Unit tests for core data models."""

import pytest
from datetime import datetime, timedelta
from typing import List

from vehicle_trajectory_prediction.core.models import (
    TrajectoryPoint,
    Trajectory,
    TrajectoryDataset,
    PredictionRequest,
    PredictionResult,
)


class TestTrajectoryPoint:
    """Test TrajectoryPoint model."""
    
    def test_valid_trajectory_point(self) -> None:
        """Test creating a valid trajectory point."""
        point = TrajectoryPoint(
            x=100.0,
            y=200.0,
            timestamp=datetime.now(),
            velocity=25.0,
            acceleration=0.5,
            heading=1.57,
            vehicle_id="vehicle_001",
            lane_id="lane_1"
        )
        
        assert point.x == 100.0
        assert point.y == 200.0
        assert point.velocity == 25.0
        assert point.acceleration == 0.5
        assert point.vehicle_id == "vehicle_001"
        assert point.lane_id == "lane_1"
    
    def test_heading_normalization(self) -> None:
        """Test heading normalization to [0, 2π)."""
        point = TrajectoryPoint(
            x=0.0,
            y=0.0,
            timestamp=datetime.now(),
            velocity=0.0,
            acceleration=0.0,
            heading=3 * 3.14159,  # 3π
            vehicle_id="vehicle_001"
        )
        
        # Heading should be normalized to [0, 2π)
        assert 0 <= point.heading < 2 * 3.14159
    
    def test_negative_velocity_raises_error(self) -> None:
        """Test that negative velocity raises validation error."""
        with pytest.raises(ValueError, match="Velocity must be non-negative"):
            TrajectoryPoint(
                x=0.0,
                y=0.0,
                timestamp=datetime.now(),
                velocity=-5.0,
                acceleration=0.0,
                heading=0.0,
                vehicle_id="vehicle_001"
            )
    
    def test_distance_calculation(self) -> None:
        """Test distance calculation between points."""
        point1 = TrajectoryPoint(
            x=0.0,
            y=0.0,
            timestamp=datetime.now(),
            velocity=0.0,
            acceleration=0.0,
            heading=0.0,
            vehicle_id="vehicle_001"
        )
        
        point2 = TrajectoryPoint(
            x=3.0,
            y=4.0,
            timestamp=datetime.now(),
            velocity=0.0,
            acceleration=0.0,
            heading=0.0,
            vehicle_id="vehicle_001"
        )
        
        distance = point1.distance_to(point2)
        assert distance == 5.0  # 3-4-5 triangle
    
    def test_time_delta_calculation(self) -> None:
        """Test time difference calculation."""
        time1 = datetime.now()
        time2 = time1 + timedelta(seconds=10)
        
        point1 = TrajectoryPoint(
            x=0.0,
            y=0.0,
            timestamp=time1,
            velocity=0.0,
            acceleration=0.0,
            heading=0.0,
            vehicle_id="vehicle_001"
        )
        
        point2 = TrajectoryPoint(
            x=0.0,
            y=0.0,
            timestamp=time2,
            velocity=0.0,
            acceleration=0.0,
            heading=0.0,
            vehicle_id="vehicle_001"
        )
        
        delta = point1.time_delta(point2)
        assert delta == 10.0


class TestTrajectory:
    """Test Trajectory model."""
    
    def create_sample_trajectory(self) -> Trajectory:
        """Create a sample trajectory for testing."""
        base_time = datetime.now()
        points = []
        
        for i in range(10):
            point = TrajectoryPoint(
                x=float(i * 10),
                y=float(i * 5),
                timestamp=base_time + timedelta(seconds=i),
                velocity=20.0 + i,
                acceleration=0.5,
                heading=0.1 * i,
                vehicle_id="vehicle_001",
                lane_id="lane_1"
            )
            points.append(point)
        
        return Trajectory(
            vehicle_id="vehicle_001",
            points=points,
            start_time=base_time,
            end_time=base_time + timedelta(seconds=9),
            duration=9.0,
            total_distance=100.0
        )
    
    def test_valid_trajectory(self) -> None:
        """Test creating a valid trajectory."""
        trajectory = self.create_sample_trajectory()
        
        assert trajectory.vehicle_id == "vehicle_001"
        assert len(trajectory.points) == 10
        assert trajectory.length == 10
        assert trajectory.duration == 9.0
        assert trajectory.total_distance == 100.0
    
    def test_trajectory_with_insufficient_points_raises_error(self) -> None:
        """Test that trajectory with less than 2 points raises error."""
        base_time = datetime.now()
        point = TrajectoryPoint(
            x=0.0,
            y=0.0,
            timestamp=base_time,
            velocity=0.0,
            acceleration=0.0,
            heading=0.0,
            vehicle_id="vehicle_001"
        )
        
        with pytest.raises(ValueError, match="Trajectory must have at least 2 points"):
            Trajectory(
                vehicle_id="vehicle_001",
                points=[point],
                start_time=base_time,
                end_time=base_time + timedelta(seconds=1),
                duration=1.0,
                total_distance=0.0
            )
    
    def test_trajectory_with_mixed_vehicle_ids_raises_error(self) -> None:
        """Test that trajectory with mixed vehicle IDs raises error."""
        base_time = datetime.now()
        points = [
            TrajectoryPoint(
                x=0.0,
                y=0.0,
                timestamp=base_time,
                velocity=0.0,
                acceleration=0.0,
                heading=0.0,
                vehicle_id="vehicle_001"
            ),
            TrajectoryPoint(
                x=10.0,
                y=0.0,
                timestamp=base_time + timedelta(seconds=1),
                velocity=0.0,
                acceleration=0.0,
                heading=0.0,
                vehicle_id="vehicle_002"  # Different vehicle ID
            )
        ]
        
        with pytest.raises(ValueError, match="All points must have the same vehicle_id"):
            Trajectory(
                vehicle_id="vehicle_001",
                points=points,
                start_time=base_time,
                end_time=base_time + timedelta(seconds=1),
                duration=1.0,
                total_distance=10.0
            )
    
    def test_trajectory_with_unordered_timestamps_raises_error(self) -> None:
        """Test that trajectory with unordered timestamps raises error."""
        base_time = datetime.now()
        points = [
            TrajectoryPoint(
                x=0.0,
                y=0.0,
                timestamp=base_time + timedelta(seconds=1),  # Later time first
                velocity=0.0,
                acceleration=0.0,
                heading=0.0,
                vehicle_id="vehicle_001"
            ),
            TrajectoryPoint(
                x=10.0,
                y=0.0,
                timestamp=base_time,  # Earlier time second
                velocity=0.0,
                acceleration=0.0,
                heading=0.0,
                vehicle_id="vehicle_001"
            )
        ]
        
        with pytest.raises(ValueError, match="Points must be ordered by timestamp"):
            Trajectory(
                vehicle_id="vehicle_001",
                points=points,
                start_time=base_time,
                end_time=base_time + timedelta(seconds=1),
                duration=1.0,
                total_distance=10.0
            )
    
    def test_trajectory_to_dataframe(self) -> None:
        """Test converting trajectory to DataFrame."""
        trajectory = self.create_sample_trajectory()
        df = trajectory.to_dataframe()
        
        assert len(df) == 10
        assert "vehicle_id" in df.columns
        assert "timestamp" in df.columns
        assert "x" in df.columns
        assert "y" in df.columns
        assert "velocity" in df.columns
        assert "acceleration" in df.columns
        assert "heading" in df.columns
        assert "lane_id" in df.columns
    
    def test_get_segment(self) -> None:
        """Test getting trajectory segment."""
        trajectory = self.create_sample_trajectory()
        base_time = trajectory.start_time
        
        segment = trajectory.get_segment(
            base_time + timedelta(seconds=2),
            base_time + timedelta(seconds=5)
        )
        
        assert len(segment.points) == 4  # Points at seconds 2, 3, 4, 5
        assert segment.start_time == base_time + timedelta(seconds=2)
        assert segment.end_time == base_time + timedelta(seconds=5)
        assert segment.duration == 3.0


class TestTrajectoryDataset:
    """Test TrajectoryDataset model."""
    
    def create_sample_dataset(self) -> TrajectoryDataset:
        """Create a sample dataset for testing."""
        base_time = datetime.now()
        trajectories = []
        
        for i in range(3):
            points = []
            for j in range(5):
                point = TrajectoryPoint(
                    x=float(i * 10 + j),
                    y=float(i * 5 + j),
                    timestamp=base_time + timedelta(seconds=i * 5 + j),
                    velocity=20.0 + j,
                    acceleration=0.5,
                    heading=0.1 * j,
                    vehicle_id=f"vehicle_{i:03d}",
                    lane_id="lane_1"
                )
                points.append(point)
            
            trajectory = Trajectory(
                vehicle_id=f"vehicle_{i:03d}",
                points=points,
                start_time=base_time + timedelta(seconds=i * 5),
                end_time=base_time + timedelta(seconds=i * 5 + 4),
                duration=4.0,
                total_distance=50.0
            )
            trajectories.append(trajectory)
        
        return TrajectoryDataset(
            name="test_dataset",
            trajectories=trajectories,
            description="Test dataset",
            source="synthetic",
            version="1.0.0",
            num_vehicles=3,
            num_trajectories=3,
            total_points=15,
            start_time=base_time,
            end_time=base_time + timedelta(seconds=14),
            bounds=(0.0, 0.0, 34.0, 19.0)
        )
    
    def test_valid_dataset(self) -> None:
        """Test creating a valid dataset."""
        dataset = self.create_sample_dataset()
        
        assert dataset.name == "test_dataset"
        assert len(dataset.trajectories) == 3
        assert dataset.num_vehicles == 3
        assert dataset.num_trajectories == 3
        assert dataset.total_points == 15
    
    def test_dataset_without_trajectories_raises_error(self) -> None:
        """Test that dataset without trajectories raises error."""
        with pytest.raises(ValueError, match="Dataset must contain at least one trajectory"):
            TrajectoryDataset(
                name="empty_dataset",
                trajectories=[],
                description="Empty dataset",
                source="synthetic",
                version="1.0.0",
                num_vehicles=0,
                num_trajectories=0,
                total_points=0,
                start_time=datetime.now(),
                end_time=datetime.now(),
                bounds=(0.0, 0.0, 0.0, 0.0)
            )
    
    def test_vehicle_ids_property(self) -> None:
        """Test vehicle_ids property."""
        dataset = self.create_sample_dataset()
        vehicle_ids = dataset.vehicle_ids
        
        assert len(vehicle_ids) == 3
        assert "vehicle_000" in vehicle_ids
        assert "vehicle_001" in vehicle_ids
        assert "vehicle_002" in vehicle_ids
    
    def test_get_trajectory_by_vehicle(self) -> None:
        """Test getting trajectories by vehicle ID."""
        dataset = self.create_sample_dataset()
        trajectories = dataset.get_trajectory_by_vehicle("vehicle_001")
        
        assert len(trajectories) == 1
        assert trajectories[0].vehicle_id == "vehicle_001"
    
    def test_dataset_to_dataframe(self) -> None:
        """Test converting dataset to DataFrame."""
        dataset = self.create_sample_dataset()
        df = dataset.to_dataframe()
        
        assert len(df) == 15  # Total number of points
        assert "vehicle_id" in df.columns
        assert "timestamp" in df.columns
        assert "x" in df.columns
        assert "y" in df.columns


class TestPredictionRequest:
    """Test PredictionRequest model."""
    
    def test_valid_prediction_request(self) -> None:
        """Test creating a valid prediction request."""
        current_state = TrajectoryPoint(
            x=100.0,
            y=200.0,
            timestamp=datetime.now(),
            velocity=25.0,
            acceleration=0.5,
            heading=1.57,
            vehicle_id="vehicle_001",
            lane_id="lane_1"
        )
        
        request = PredictionRequest(
            vehicle_id="vehicle_001",
            current_state=current_state,
            prediction_horizon=30,
            prediction_frequency=0.1,
            model_name="cv",
            confidence_level=0.95
        )
        
        assert request.vehicle_id == "vehicle_001"
        assert request.prediction_horizon == 30
        assert request.prediction_frequency == 0.1
        assert request.model_name == "cv"
        assert request.confidence_level == 0.95


class TestPredictionResult:
    """Test PredictionResult model."""
    
    def test_valid_prediction_result(self) -> None:
        """Test creating a valid prediction result."""
        # Create a simple predicted trajectory
        base_time = datetime.now()
        points = [
            TrajectoryPoint(
                x=100.0 + i * 10,
                y=200.0 + i * 5,
                timestamp=base_time + timedelta(seconds=i * 0.1),
                velocity=25.0,
                acceleration=0.0,
                heading=1.57,
                vehicle_id="vehicle_001"
            )
            for i in range(5)
        ]
        
        predicted_trajectory = Trajectory(
            vehicle_id="vehicle_001",
            points=points,
            start_time=base_time,
            end_time=base_time + timedelta(seconds=0.4),
            duration=0.4,
            total_distance=50.0
        )
        
        result = PredictionResult(
            vehicle_id="vehicle_001",
            prediction_horizon=5,
            prediction_frequency=0.1,
            predicted_trajectory=predicted_trajectory,
            model_name="cv",
            model_confidence=0.85,
            inference_time=0.1
        )
        
        assert result.vehicle_id == "vehicle_001"
        assert result.prediction_horizon == 5
        assert result.prediction_frequency == 0.1
        assert result.model_name == "cv"
        assert result.model_confidence == 0.85
        assert result.inference_time == 0.1
        assert len(result.predicted_trajectory.points) == 5