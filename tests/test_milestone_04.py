"""Consolidated tests for Milestone 4: Feature Engineering Framework."""

import sys
import tempfile
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / ".." / "src"))

import numpy as np
import pandas as pd
import pytest

from trajectory_prediction.data.augmentation import (
    AugmentationPipeline,
    NoiseInjectionAugmenter,
    SpatialTransformationAugmenter,
    TemporalShiftAugmenter,
)
from trajectory_prediction.data.feature_store import (
    FeatureStore,
    create_default_feature_store,
)
from trajectory_prediction.data.feature_validation import (
    FeatureCorrelationAnalyzer,
    FeatureQualityAssessor,
)
from trajectory_prediction.data.features import (
    ContextualFeatureExtractor,
    KinematicFeatureExtractor,
    QualityFeatureExtractor,
    TemporalFeatureExtractor,
)
from trajectory_prediction.data.models import (
    Trajectory,
    TrajectoryPoint,
    Vehicle,
    VehicleType,
)


@pytest.fixture
def sample_trajectory():
    """Create a sample trajectory for testing with realistic timestamps."""
    # Use realistic timestamps (Jan 1, 2021 + offsets)
    base_time = 1609459200.0  # Jan 1, 2021 timestamp
    timestamps = base_time + np.linspace(0, 5, 25)  # 5 seconds, 25 points
    x = 10 * np.linspace(0, 5, 25)  # Constant velocity in x
    y = 0.5 * np.linspace(0, 5, 25)  # Slow movement in y

    # Calculate velocities
    vx = np.full_like(timestamps, 10.0)
    vy = np.full_like(timestamps, 0.5)
    speeds = np.sqrt(vx**2 + vy**2)

    # Create trajectory points
    points = []
    for i, (t, px, py, v_x, v_y, speed) in enumerate(
        zip(timestamps, x, y, vx, vy, speeds)
    ):
        points.append(
            TrajectoryPoint(
                timestamp=float(t),
                x=float(px),
                y=float(py),
                speed=float(speed),
                velocity_x=float(v_x),
                velocity_y=float(v_y),
                acceleration_x=0.0,
                acceleration_y=0.0,
                heading=np.arctan2(v_y, v_x),
                lane_id=1,
                frame_id=i,
            )
        )

    vehicle = Vehicle(
        vehicle_id=1,
        vehicle_type=VehicleType.CAR,
        length=4.5,
        width=1.8,
    )

    return Trajectory(
        trajectory_id="test_trajectory",
        vehicle=vehicle,
        points=points,
        dataset_name="test_dataset",
        metadata={"test": True},
        completeness_score=1.0,
        temporal_consistency_score=1.0,
        spatial_accuracy_score=1.0,
        smoothness_score=1.0,
    )


@pytest.fixture
def simple_trajectory():
    """Create a simple trajectory for basic testing."""
    base_time = 1609459200.0  # Jan 1, 2021 timestamp
    vehicle = Vehicle(vehicle_id=1, vehicle_type=VehicleType.CAR, length=4.0, width=1.8)

    points = [
        TrajectoryPoint(
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
            frame_id=0,
        ),
        TrajectoryPoint(
            timestamp=base_time + 1.0,
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
            timestamp=base_time + 2.0,
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
        trajectory_id="simple_test_trajectory",
        vehicle=vehicle,
        points=points,
        dataset_name="test_dataset",
        metadata={"test": True},
        completeness_score=1.0,
        temporal_consistency_score=1.0,
        spatial_accuracy_score=1.0,
        smoothness_score=1.0,
    )


class TestKinematicFeatureExtractor:
    """Test kinematic feature extraction."""

    def test_extract_basic_features(self, sample_trajectory):
        """Test basic kinematic feature extraction."""
        extractor = KinematicFeatureExtractor()
        features = extractor.extract(sample_trajectory)

        # Check expected features are present
        expected_features = [
            "total_displacement",
            "path_length",
            "straightness",
            "speed_mean",
            "speed_std",
            "acceleration_mean",
        ]

        for feature in expected_features:
            assert feature in features
            assert isinstance(features[feature], int | float)

        # Check some basic physics
        assert features["total_displacement"] > 0
        assert features["path_length"] > 0
        assert 0 <= features["straightness"] <= 1
        assert features["speed_mean"] > 0

    def test_extract_with_minimal_trajectory(self, simple_trajectory):
        """Test feature extraction with minimal valid trajectory."""
        extractor = KinematicFeatureExtractor()
        features = extractor.extract(simple_trajectory)

        # Should still produce valid features
        assert isinstance(features, dict)
        assert len(features) > 0

        # Basic sanity checks
        assert features["total_displacement"] >= 0
        assert features["path_length"] >= 0

    def test_get_feature_info(self):
        """Test getting feature information."""
        extractor = KinematicFeatureExtractor()
        info = extractor.get_feature_info()

        assert isinstance(info, list)
        assert len(info) > 0

        # Check that each feature has required metadata
        for feature_info in info:
            assert hasattr(feature_info, "name")
            assert hasattr(feature_info, "description")
            assert hasattr(feature_info, "feature_type")


class TestTemporalFeatureExtractor:
    """Test temporal feature extraction."""

    def test_extract_temporal_features(self, sample_trajectory):
        """Test temporal feature extraction."""
        extractor = TemporalFeatureExtractor()
        features = extractor.extract(sample_trajectory)

        expected_features = [
            "total_duration",
            "sampling_rate_mean",
            "num_samples",
        ]

        for feature in expected_features:
            assert feature in features
            assert isinstance(features[feature], int | float)

        # Check temporal properties
        assert features["total_duration"] > 0
        assert features["sampling_rate_mean"] > 0
        assert features["num_samples"] > 0


class TestContextualFeatureExtractor:
    """Test contextual feature extraction."""

    def test_extract_contextual_features(self, sample_trajectory):
        """Test contextual feature extraction."""
        extractor = ContextualFeatureExtractor()
        features = extractor.extract(sample_trajectory)

        expected_features = [
            "lane_changes",
            "following_detected",
            "lane_consistency",
        ]

        for feature in expected_features:
            assert feature in features
            assert isinstance(features[feature], int | float)


class TestQualityFeatureExtractor:
    """Test quality feature extraction."""

    def test_extract_quality_features(self, sample_trajectory):
        """Test quality feature extraction."""
        extractor = QualityFeatureExtractor()
        features = extractor.extract(sample_trajectory)

        expected_features = [
            "completeness_score",
            "position_smoothness",
            "missing_speed_ratio",
        ]

        for feature in expected_features:
            assert feature in features
            assert isinstance(features[feature], int | float)

        # Quality metrics should be between 0 and 1
        assert 0 <= features["completeness_score"] <= 1
        assert 0 <= features["position_smoothness"] <= 1
        assert 0 <= features["missing_speed_ratio"] <= 1


class TestFeatureStore:
    """Test feature store functionality."""

    def test_feature_store_creation(self):
        """Test creating a feature store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = create_default_feature_store(Path(tmpdir))
            assert isinstance(store, FeatureStore)

    def test_feature_extraction_and_caching(self, sample_trajectory):
        """Test feature extraction and caching."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = create_default_feature_store(Path(tmpdir))

            # Extract features
            features = store.extract_features(sample_trajectory)
            assert isinstance(features, dict)
            assert len(features) > 0

            # Test caching by extracting again
            cached_features = store.extract_features(sample_trajectory)
            assert features == cached_features

    def test_feature_filtering(self, sample_trajectory):
        """Test feature filtering."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = create_default_feature_store(Path(tmpdir))

            # Extract only specific features
            features = store.extract_features(
                sample_trajectory, feature_names=["speed_mean", "path_length"]
            )

            # Should contain only requested features (if available)
            assert isinstance(features, dict)
            assert len(features) <= 2  # Should only have requested features


class TestDataAugmentation:
    """Test data augmentation functionality."""

    def test_noise_injection_augmenter(self, sample_trajectory):
        """Test noise injection augmentation."""
        augmenter = NoiseInjectionAugmenter(
            position_noise_std=0.1, velocity_noise_std=0.05, num_augmentations=1
        )
        augmented_list = augmenter.augment(sample_trajectory)

        assert isinstance(augmented_list, list)
        assert len(augmented_list) == 1

        augmented = augmented_list[0]
        assert isinstance(augmented, Trajectory)
        assert len(augmented.points) == len(sample_trajectory.points)

        # Check that some noise was added
        original_x = [p.x for p in sample_trajectory.points]
        augmented_x = [p.x for p in augmented.points]
        assert original_x != augmented_x

    def test_temporal_shift_augmenter(self, sample_trajectory):
        """Test temporal shift augmentation."""
        augmenter = TemporalShiftAugmenter(
            time_shift_range=(-1.0, 1.0), num_augmentations=1
        )
        augmented_list = augmenter.augment(sample_trajectory)

        assert isinstance(augmented_list, list)
        assert len(augmented_list) == 1

        augmented = augmented_list[0]
        assert isinstance(augmented, Trajectory)
        assert len(augmented.points) == len(sample_trajectory.points)

    def test_spatial_transformation_augmenter(self, sample_trajectory):
        """Test spatial transformation augmentation."""
        augmenter = SpatialTransformationAugmenter(
            translation_range=(-1.0, 1.0),
            rotation_range=(-0.1, 0.1),
            num_augmentations=1,
        )
        augmented_list = augmenter.augment(sample_trajectory)

        assert isinstance(augmented_list, list)
        assert len(augmented_list) == 1

        augmented = augmented_list[0]
        assert isinstance(augmented, Trajectory)
        assert len(augmented.points) == len(sample_trajectory.points)

    def test_augmentation_pipeline(self, sample_trajectory):
        """Test augmentation pipeline with multiple augmenters."""
        pipeline = AugmentationPipeline(
            [
                NoiseInjectionAugmenter(position_noise_std=0.05, num_augmentations=1),
                TemporalShiftAugmenter(
                    time_shift_range=(-0.5, 0.5), num_augmentations=1
                ),
                SpatialTransformationAugmenter(
                    translation_range=(-0.5, 0.5),
                    rotation_range=(-0.05, 0.05),
                    num_augmentations=1,
                ),
            ]
        )

        # Test augmenting a dataset
        trajectories = [sample_trajectory]
        augmented_dataset = pipeline.augment_dataset(trajectories)
        assert isinstance(augmented_dataset, list)
        assert len(augmented_dataset) >= len(
            trajectories
        )  # Should include originals + augmented

    def test_basic_augmentation(self, simple_trajectory):
        """Test basic augmentation functionality."""
        # Test individual augmenters
        noise_augmenter = NoiseInjectionAugmenter(num_augmentations=1)
        noisy_list = noise_augmenter.augment(simple_trajectory)
        assert len(noisy_list) == 1
        assert len(noisy_list[0].points) == len(simple_trajectory.points)


class TestFeatureValidation:
    """Test feature validation functionality."""

    def test_feature_quality_assessor(self, sample_trajectory):
        """Test feature quality assessment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = create_default_feature_store(Path(tmpdir))
            features = store.extract_features(sample_trajectory)

            # Convert to DataFrame for assessment
            feature_df = pd.DataFrame([features])

            assessor = FeatureQualityAssessor()
            quality_reports = assessor.assess_feature_set(feature_df)

            assert isinstance(quality_reports, dict)
            for _feature_name, report in quality_reports.items():
                assert hasattr(report, "completeness")
                assert hasattr(report, "outlier_ratio")

    def test_feature_correlation_analyzer(self):
        """Test feature correlation analysis."""
        # Create sample feature data
        feature_data = pd.DataFrame(
            {
                "feature_1": np.random.normal(0, 1, 100),
                "feature_2": np.random.normal(0, 1, 100),
                "feature_3": np.random.normal(0, 1, 100),
            }
        )

        analyzer = FeatureCorrelationAnalyzer()
        correlation_report = analyzer.analyze_correlations(feature_data)

        assert isinstance(correlation_report, dict)
        assert "correlation_matrix" in correlation_report
        assert "highly_correlated_pairs" in correlation_report


class TestIntegration:
    """Test end-to-end integration scenarios."""

    def test_end_to_end_feature_pipeline(self, sample_trajectory):
        """Test complete feature engineering pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create feature store
            store = create_default_feature_store(Path(tmpdir))

            # Extract features
            features = store.extract_features(sample_trajectory)
            assert len(features) > 0

            # Validate features
            feature_df = pd.DataFrame([features])
            assessor = FeatureQualityAssessor()
            quality_reports = assessor.assess_feature_set(feature_df)
            assert isinstance(quality_reports, dict)

            # Test augmentation
            augmenter = NoiseInjectionAugmenter(num_augmentations=1)
            augmented_trajectories = augmenter.augment(sample_trajectory)

            # Extract features from augmented trajectory
            augmented_features = store.extract_features(augmented_trajectories[0])
            assert len(augmented_features) > 0

    def test_batch_processing_workflow(self, sample_trajectory):
        """Test batch processing multiple trajectories."""
        trajectories = [sample_trajectory for _ in range(3)]

        with tempfile.TemporaryDirectory() as tmpdir:
            store = create_default_feature_store(Path(tmpdir))

            # Process batch
            all_features = []
            for trajectory in trajectories:
                features = store.extract_features(trajectory)
                all_features.append(features)

            assert len(all_features) == 3
            assert all(isinstance(f, dict) for f in all_features)


# Simple function-based tests for basic functionality
def test_feature_extraction_basic(simple_trajectory):
    """Test basic feature extraction functionality."""
    extractor = KinematicFeatureExtractor()
    features = extractor.extract(simple_trajectory)

    assert isinstance(features, dict)
    assert len(features) > 0

    # Check basic feature presence
    basic_features = ["speed_mean", "acceleration_mean", "path_length"]
    for feature in basic_features:
        if feature in features:
            assert isinstance(features[feature], int | float)


def test_feature_store_basic():
    """Test basic feature store creation and operation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = create_default_feature_store(Path(tmpdir))
        assert isinstance(store, FeatureStore)

        # Test default extractors are loaded
        assert len(store.extractors) > 0


def test_trajectory_augmentation_basic(simple_trajectory):
    """Test basic trajectory augmentation."""
    # Test individual augmenters
    noise_augmenter = NoiseInjectionAugmenter(num_augmentations=1)
    noisy_trajectories = noise_augmenter.augment(simple_trajectory)
    assert len(noisy_trajectories) == 1
    assert len(noisy_trajectories[0].points) == len(simple_trajectory.points)

    # Verify data types are preserved
    for point in noisy_trajectories[0].points:
        assert isinstance(point.x, float)
        assert isinstance(point.y, float)
        assert isinstance(point.timestamp, float)


def test_feature_quality_assessment_basic():
    """Test basic feature quality assessment."""
    # Create sample features DataFrame
    feature_data = pd.DataFrame(
        {
            "speed_mean": [15.0, 12.0, 18.0],
            "acceleration_mean": [0.5, 0.3, 0.7],
            "path_length": [100.0, 120.0, 90.0],
            "outlier_feature": [999999.0, 15.0, 20.0],  # Has obvious outlier
        }
    )

    assessor = FeatureQualityAssessor()
    quality_reports = assessor.assess_feature_set(feature_data)

    assert isinstance(quality_reports, dict)
    assert len(quality_reports) > 0


def test_feature_correlation_analysis_basic():
    """Test basic feature correlation analysis."""
    # Create correlated feature data
    n_samples = 50
    x = np.random.normal(0, 1, n_samples)
    feature_data = pd.DataFrame(
        {
            "feature_a": x,
            "feature_b": x + np.random.normal(0, 0.1, n_samples),  # Highly correlated
            "feature_c": np.random.normal(0, 1, n_samples),  # Independent
        }
    )

    analyzer = FeatureCorrelationAnalyzer()
    correlation_report = analyzer.analyze_correlations(feature_data)

    assert isinstance(correlation_report, dict)
    assert "correlation_matrix" in correlation_report
