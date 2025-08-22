"""Simple tests for Milestone 4: Feature Engineering Framework."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trajectory_prediction.data.augmentation import TrajectoryAugmenter
from trajectory_prediction.data.feature_store import FeatureStore
from trajectory_prediction.data.features import FeatureExtractor
from trajectory_prediction.data.models import Trajectory, TrajectoryPoint, Vehicle


def create_test_trajectory():
    """Create a simple test trajectory."""
    vehicle = Vehicle(vehicle_id=1, vehicle_class="car", length=4.0, width=1.8)

    points = [
        TrajectoryPoint(
            timestamp=0.0,
            x=0.0,
            y=0.0,
            speed=0.0,
            velocity_x=0.0,
            velocity_y=0.0,
            acceleration_x=0.0,
            acceleration_y=0.0,
            heading=0.0,
            lane_id=1,
            frame_id=0,
        ),
        TrajectoryPoint(
            timestamp=1.0,
            x=1.0,
            y=0.0,
            speed=1.0,
            velocity_x=1.0,
            velocity_y=0.0,
            acceleration_x=0.0,
            acceleration_y=0.0,
            heading=0.0,
            lane_id=1,
            frame_id=1,
        ),
        TrajectoryPoint(
            timestamp=2.0,
            x=2.0,
            y=0.0,
            speed=1.0,
            velocity_x=1.0,
            velocity_y=0.0,
            acceleration_x=0.0,
            acceleration_y=0.0,
            heading=0.0,
            lane_id=1,
            frame_id=2,
        ),
    ]

    return Trajectory(
        trajectory_id="test_trajectory",
        vehicle=vehicle,
        points=points,
        dataset_name="test",
        metadata={},
    )


def test_feature_extraction():
    """Test basic feature extraction."""
    trajectory = create_test_trajectory()
    extractor = FeatureExtractor()

    features = extractor.extract_features(trajectory)

    # Check that we get basic features
    expected_features = [
        "duration",
        "distance",
        "avg_speed",
        "max_speed",
        "avg_acceleration",
        "max_acceleration",
    ]

    for feature in expected_features:
        assert feature in features, f"Missing feature: {feature}"
        print(f"✓ Feature {feature}: {features[feature]}")

    print("✓ Feature extraction test passed")


def test_feature_store():
    """Test feature store functionality."""
    trajectory = create_test_trajectory()
    extractor = FeatureExtractor()
    store = FeatureStore()

    # Extract and store features
    features = extractor.extract_features(trajectory)
    store.store_features(trajectory.trajectory_id, features)

    # Retrieve features
    retrieved = store.get_features(trajectory.trajectory_id)

    assert retrieved is not None, "Features not retrieved"
    assert len(retrieved) > 0, "No features retrieved"

    print("✓ Feature store test passed")


def test_data_augmentation():
    """Test data augmentation functionality."""
    trajectory = create_test_trajectory()
    augmenter = TrajectoryAugmenter()

    # Test noise injection
    noisy_trajectory = augmenter.add_noise(trajectory, noise_std=0.1)

    assert len(noisy_trajectory.points) == len(trajectory.points)
    assert noisy_trajectory.trajectory_id != trajectory.trajectory_id

    print("✓ Data augmentation test passed")


def run_all_tests():
    """Run all tests."""
    print("Running Milestone 4 tests...")

    try:
        test_feature_extraction()
        test_feature_store()
        test_data_augmentation()
        print("\n✅ All Milestone 4 tests passed!")
        return True
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
