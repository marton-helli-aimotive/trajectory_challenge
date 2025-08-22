"""Tests for Milestone 4: Feature Engineering Framework."""

import sys
import tempfile
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / ".." / "src"))

import numpy as np
import pandas as pd
import pytest

from trajectory_prediction.data.feature_store import (
    create_default_feature_store,
)
from trajectory_prediction.data.feature_validation import (
    FeatureCorrelationAnalyzer,
    FeatureQualityAssessor,
)
from trajectory_prediction.data.features import KinematicFeatureExtractor
from trajectory_prediction.data.models import (
    Trajectory,
    TrajectoryPoint,
    Vehicle,
    VehicleType,
)


@pytest.fixture
def sample_trajectory():
    """Create a sample trajectory for testing."""
    # Use realistic timestamps (current time + offsets)
    base_time = 1609459200.0  # Jan 1, 2021 timestamp
    timestamps = base_time + np.linspace(0, 5, 25)  # 5 seconds, 25 points
    x = 10 * np.linspace(0, 5, 25)  # Constant velocity in x
    y = 0.5 * np.linspace(0, 5, 25)  # Slow movement in y

    # Calculate velocities
    v_x = np.full_like(timestamps, 10.0)  # constant 10 m/s
    v_y = np.full_like(timestamps, 0.5)  # constant 0.5 m/s

    # Create trajectory points
    points = []
    for i, (t, px, py, vx, vy) in enumerate(zip(timestamps, x, y, v_x, v_y)):
        points.append(
            TrajectoryPoint(
                timestamp=float(t),
                x=float(px),
                y=float(py),
                speed=float(np.sqrt(vx**2 + vy**2)),
                velocity_x=float(vx),
                velocity_y=float(vy),
                acceleration_x=0.0,
                acceleration_y=0.0,
                heading=float(np.arctan2(vy, vx)),
                lane_id=1,
                frame_id=i,
            )
        )

    vehicle = Vehicle(vehicle_id=1, vehicle_type=VehicleType.CAR, length=4.0, width=1.8)

    return Trajectory(
        trajectory_id="test_trajectory_001",
        vehicle=vehicle,
        points=points,
        dataset_name="test",
        metadata={"test": True},
        completeness_score=1.0,
        temporal_consistency_score=1.0,
        spatial_accuracy_score=1.0,
        smoothness_score=1.0,
    )


def test_feature_extractor_basic(sample_trajectory):
    """Test basic feature extraction functionality."""
    extractor = KinematicFeatureExtractor()

    # Test kinematic features
    features = extractor.extract(sample_trajectory)
    assert isinstance(features, dict)
    assert len(features) > 0

    # Test that we get expected kinematic features
    expected_features = ["path_length", "total_displacement", "straightness"]
    for feature in expected_features:
        assert feature in features
        assert isinstance(features[feature], int | float)


def test_feature_store_basic():
    """Test basic feature store functionality."""
    with tempfile.TemporaryDirectory() as temp_dir:
        feature_store = create_default_feature_store(temp_dir)

        # Test storing and retrieving features
        test_features = {"speed_mean": 10.0, "acceleration_max": 2.0}
        feature_store.storage.store_features("test_trajectory", test_features)

        retrieved = feature_store.storage.retrieve_features("test_trajectory")
        assert retrieved == test_features


def test_trajectory_augmentation_basic(sample_trajectory):
    """Test basic trajectory augmentation functionality."""
    # For this basic test, let's just test that augmentation structure works
    # Create a simple test by checking if we can create an augmented trajectory
    from trajectory_prediction.data.augmentation import NoiseInjectionAugmenter

    augmenter = NoiseInjectionAugmenter(
        position_noise_std=0.1, velocity_noise_std=0.05, num_augmentations=1
    )

    # Test noise injection
    augmented_trajectories = augmenter.augment(sample_trajectory)
    assert len(augmented_trajectories) == 1

    augmented = augmented_trajectories[0]
    assert len(augmented.points) == len(sample_trajectory.points)
    assert augmented.trajectory_id != sample_trajectory.trajectory_id


def test_feature_quality_assessment():
    """Test feature quality assessment functionality."""
    assessor = FeatureQualityAssessor()

    # Create test feature data
    feature_data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], name="test_feature")
    target_data = pd.Series([2, 4, 6, 8, 10, 12, 14, 16, 18, 20], name="target")

    # Assess quality
    report = assessor.assess_feature_quality(feature_data, target_data)

    assert report.feature_name == "test_feature"
    assert 0 <= report.completeness <= 1
    assert 0 <= report.uniqueness <= 1
    assert report.distribution_type is not None
    assert 0 <= report.outlier_ratio <= 1
    assert 0 <= report.stability <= 1


def test_feature_correlation_analysis():
    """Test feature correlation analysis functionality."""
    analyzer = FeatureCorrelationAnalyzer()

    # Create test feature dataframe
    feature_df = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [2, 4, 6, 8, 10],  # Highly correlated with feature1
            "feature3": [5, 4, 3, 2, 1],  # Anti-correlated with feature1
            "feature4": [1, 3, 2, 5, 4],  # Random
        }
    )

    analysis = analyzer.analyze_correlations(feature_df)

    assert "correlation_matrix" in analysis
    assert "highly_correlated_pairs" in analysis
    assert "redundant_features" in analysis
    assert "recommendations" in analysis

    # Should detect high correlation between feature1 and feature2
    assert len(analysis["highly_correlated_pairs"]) > 0


if __name__ == "__main__":
    # Simple test runner for when pytest is not available
    import traceback

    # Create sample data
    sample_traj = None
    try:
        # Manually create the sample trajectory
        # Use realistic timestamps (current time + offsets)
        base_time = 1609459200.0  # Jan 1, 2021 timestamp
        timestamps = base_time + np.linspace(0, 5, 25)  # 5 seconds, 25 points
        x = 10 * np.linspace(0, 5, 25)  # Constant velocity in x
        y = 0.5 * np.linspace(0, 5, 25)  # Slow movement in y

        # Calculate velocities
        v_x = np.full_like(timestamps, 10.0)  # constant 10 m/s
        v_y = np.full_like(timestamps, 0.5)  # constant 0.5 m/s

        # Create trajectory points
        points = []
        for i, (t, px, py, vx, vy) in enumerate(zip(timestamps, x, y, v_x, v_y)):
            points.append(
                TrajectoryPoint(
                    timestamp=float(t),
                    x=float(px),
                    y=float(py),
                    speed=float(np.sqrt(vx**2 + vy**2)),
                    velocity_x=float(vx),
                    velocity_y=float(vy),
                    acceleration_x=0.0,
                    acceleration_y=0.0,
                    heading=float(np.arctan2(vy, vx)),
                    lane_id=1,
                    frame_id=i,
                )
            )

        vehicle = Vehicle(
            vehicle_id=1, vehicle_type=VehicleType.CAR, length=4.0, width=1.8
        )

        sample_traj = Trajectory(
            trajectory_id="test_trajectory_001",
            vehicle=vehicle,
            points=points,
            dataset_name="test",
            metadata={"test": True},
            completeness_score=1.0,
            temporal_consistency_score=1.0,
            spatial_accuracy_score=1.0,
            smoothness_score=1.0,
        )

        print("✓ Sample trajectory created successfully")
    except Exception as e:
        print(f"✗ Failed to create sample trajectory: {e}")
        traceback.print_exc()
        exit(1)

    # Test feature extraction
    try:
        test_feature_extractor_basic(sample_traj)
        print("✓ Feature extraction test passed")
    except Exception as e:
        print(f"✗ Feature extraction test failed: {e}")
        traceback.print_exc()

    # Test feature store
    try:
        test_feature_store_basic()
        print("✓ Feature store test passed")
    except Exception as e:
        print(f"✗ Feature store test failed: {e}")
        traceback.print_exc()

    # Test augmentation
    try:
        test_trajectory_augmentation_basic(sample_traj)
        print("✓ Trajectory augmentation test passed")
    except Exception as e:
        print(f"✗ Trajectory augmentation test failed: {e}")
        print("  (This is expected if NoiseInjectionAugmenter is not implemented)")
        traceback.print_exc()

    # Test quality assessment
    try:
        test_feature_quality_assessment()
        print("✓ Feature quality assessment test passed")
    except Exception as e:
        print(f"✗ Feature quality assessment test failed: {e}")
        traceback.print_exc()

    # Test correlation analysis
    try:
        test_feature_correlation_analysis()
        print("✓ Feature correlation analysis test passed")
    except Exception as e:
        print(f"✗ Feature correlation analysis test failed: {e}")
        traceback.print_exc()

    print("\nAll tests completed!")
