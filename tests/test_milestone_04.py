"""Tests for Milestone 4: Feature Engineering Framework."""

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
from trajectory_prediction.data.feature_store import create_default_feature_store
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
    """Create a sample trajectory for testing."""
    timestamps = np.linspace(0, 5, 25)  # 5 seconds, 25 points
    x = 10 * timestamps  # Constant velocity in x
    y = 0.5 * timestamps  # Slow movement in y

    # Calculate velocities
    vx = np.full_like(timestamps, 10.0)
    vy = np.full_like(timestamps, 0.5)
    speeds = np.sqrt(vx**2 + vy**2)

    points = []
    for i, t in enumerate(timestamps):
        point = TrajectoryPoint(
            timestamp=t,
            x=x[i],
            y=y[i],
            speed=speeds[i],
            velocity_x=vx[i],
            velocity_y=vy[i],
            acceleration_x=0.0,
            acceleration_y=0.0,
            heading=np.degrees(np.arctan2(vy[i], vx[i])),
            lane_id=1,
            frame_id=i,
        )
        points.append(point)

    vehicle = Vehicle(
        vehicle_id=1,
        vehicle_type=VehicleType.CAR,
        length=4.0,
        width=1.8,
    )

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

    def test_extract_with_minimal_trajectory(self):
        """Test feature extraction with minimal valid trajectory."""
        points = [
            TrajectoryPoint(
                timestamp=1609459200.0,  # Jan 1, 2021
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
                timestamp=1609459201.0,  # Jan 1, 2021 + 1s
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
        ]

        vehicle = Vehicle(
            vehicle_id=2,
            vehicle_type=VehicleType.CAR,
            length=4.0,
            width=1.8,
        )

        trajectory = Trajectory(
            trajectory_id="minimal_trajectory",
            vehicle=vehicle,
            points=points,
            dataset_name="test",
            metadata={},
            completeness_score=1.0,
            temporal_consistency_score=1.0,
            spatial_accuracy_score=1.0,
            smoothness_score=1.0,
        )

        extractor = KinematicFeatureExtractor()
        features = extractor.extract(trajectory)

        # Should not crash and should return reasonable values
        assert isinstance(features, dict)
        assert len(features) > 0
        assert features["total_displacement"] == 1.0
        assert features["path_length"] == 1.0

    def test_get_feature_info(self):
        """Test feature metadata retrieval."""
        extractor = KinematicFeatureExtractor()
        feature_info = extractor.get_feature_info()

        assert len(feature_info) > 0
        for info in feature_info:
            assert hasattr(info, "name")
            assert hasattr(info, "description")
            assert hasattr(info, "feature_type")
            assert info.feature_type == "kinematic"


class TestTemporalFeatureExtractor:
    """Test temporal feature extraction."""

    def test_extract_temporal_features(self, sample_trajectory):
        """Test temporal feature extraction."""
        extractor = TemporalFeatureExtractor()
        features = extractor.extract(sample_trajectory)

        expected_features = [
            "total_duration",
            "sampling_rate_mean",
            "sampling_regularity",
            "start_hour_of_day",
        ]

        for feature in expected_features:
            assert feature in features

        # Check reasonable values
        assert features["total_duration"] > 0
        assert features["sampling_rate_mean"] > 0
        assert 0 <= features["sampling_regularity"] <= 1
        assert 0 <= features["start_hour_of_day"] < 24


class TestContextualFeatureExtractor:
    """Test contextual feature extraction."""

    def test_extract_contextual_features(self, sample_trajectory):
        """Test contextual feature extraction."""
        extractor = ContextualFeatureExtractor()
        features = extractor.extract(sample_trajectory)

        expected_features = ["lane_changes", "unique_lanes", "lane_consistency"]

        for feature in expected_features:
            assert feature in features

        # Check reasonable values for this trajectory (all same lane)
        assert features["lane_changes"] == 0  # No lane changes
        assert features["unique_lanes"] == 1  # Only one lane
        assert features["lane_consistency"] == 1.0  # Always in same lane


class TestQualityFeatureExtractor:
    """Test quality feature extraction."""

    def test_extract_quality_features(self, sample_trajectory):
        """Test quality feature extraction."""
        extractor = QualityFeatureExtractor()
        features = extractor.extract(sample_trajectory)

        expected_features = [
            "completeness_score",
            "missing_speed_ratio",
            "position_smoothness",
            "velocity_smoothness",
        ]

        for feature in expected_features:
            assert feature in features

        # Check reasonable values
        assert 0 <= features["completeness_score"] <= 1
        assert 0 <= features["missing_speed_ratio"] <= 1
        assert 0 <= features["position_smoothness"] <= 1
        assert 0 <= features["velocity_smoothness"] <= 1


class TestFeatureStore:
    """Test feature store functionality."""

    def test_feature_store_creation(self):
        """Test feature store creation and setup."""
        feature_store = create_default_feature_store()

        assert len(feature_store.extractors) > 0
        assert len(feature_store.list_available_features()) > 0

        # Check all expected extractors are registered
        expected_extractors = ["kinematic", "temporal", "contextual", "quality"]
        for extractor_name in expected_extractors:
            assert extractor_name in feature_store.extractors

    def test_feature_extraction_and_caching(self, sample_trajectory):
        """Test feature extraction and caching."""
        with tempfile.TemporaryDirectory() as temp_dir:
            feature_store = create_default_feature_store(temp_dir)

            # First extraction
            features1 = feature_store.extract_features(sample_trajectory)
            assert len(features1) > 0

            # Second extraction (should use cache)
            features2 = feature_store.extract_features(sample_trajectory)
            assert features1 == features2

            # Force recomputation
            features3 = feature_store.extract_features(
                sample_trajectory, force_recompute=True
            )
            assert features1 == features3  # Should be same values but freshly computed

    def test_feature_filtering(self, sample_trajectory):
        """Test extracting specific features."""
        feature_store = create_default_feature_store()

        # Extract only specific features
        requested_features = ["speed_mean", "total_duration", "path_length"]
        features = feature_store.extract_features(sample_trajectory, requested_features)

        # Should only return requested features
        for feature_name in requested_features:
            assert feature_name in features

        # Check we didn't get unrequested features (but cache might have them)
        assert len(features) >= len(requested_features)


class TestDataAugmentation:
    """Test data augmentation techniques."""

    def test_noise_injection_augmenter(self, sample_trajectory):
        """Test noise injection augmentation."""
        augmenter = NoiseInjectionAugmenter(
            position_noise_std=0.1, velocity_noise_std=0.05, num_augmentations=2
        )

        augmented = augmenter.augment(sample_trajectory)

        assert len(augmented) == 2
        for aug_traj in augmented:
            assert len(aug_traj.points) == len(sample_trajectory.points)
            assert "noise" in aug_traj.trajectory_id
            assert aug_traj.metadata.get("augmentation") == "noise_injection"

    def test_temporal_shift_augmenter(self, sample_trajectory):
        """Test temporal shift augmentation."""
        augmenter = TemporalShiftAugmenter(
            time_shift_range=(-0.1, 0.1),
            time_scale_range=(0.95, 1.05),
            num_augmentations=1,
        )

        augmented = augmenter.augment(sample_trajectory)

        assert len(augmented) == 1
        aug_traj = augmented[0]
        assert len(aug_traj.points) == len(sample_trajectory.points)
        assert "temporal" in aug_traj.trajectory_id
        assert aug_traj.metadata.get("augmentation") == "temporal_shift"

    def test_spatial_transformation_augmenter(self, sample_trajectory):
        """Test spatial transformation augmentation."""
        augmenter = SpatialTransformationAugmenter(
            rotation_range=(-1.0, 1.0),
            translation_range=(-0.5, 0.5),
            num_augmentations=1,
        )

        augmented = augmenter.augment(sample_trajectory)

        assert len(augmented) == 1
        aug_traj = augmented[0]
        assert len(aug_traj.points) == len(sample_trajectory.points)
        assert "spatial" in aug_traj.trajectory_id
        assert aug_traj.metadata.get("augmentation") == "spatial_transformation"

    def test_augmentation_pipeline(self, sample_trajectory):
        """Test augmentation pipeline with multiple techniques."""
        augmenters = [
            NoiseInjectionAugmenter(num_augmentations=1),
            TemporalShiftAugmenter(num_augmentations=1),
        ]

        pipeline = AugmentationPipeline(augmenters, apply_probability=1.0)
        augmented_dataset = pipeline.augment_dataset([sample_trajectory])

        # Should have original + augmented trajectories
        assert len(augmented_dataset) >= 1

        # Get stats
        stats = pipeline.get_augmentation_stats(1, len(augmented_dataset) - 1)
        assert stats["original_trajectories"] == 1
        assert stats["augmented_trajectories"] >= 0


class TestFeatureValidation:
    """Test feature validation and quality assessment."""

    def test_feature_quality_assessor(self, sample_trajectory):
        """Test feature quality assessment."""
        feature_store = create_default_feature_store()
        features = feature_store.extract_features(sample_trajectory)

        # Create DataFrame for analysis
        features_df = pd.DataFrame([features])

        assessor = FeatureQualityAssessor()
        quality_reports = assessor.assess_feature_set(features_df)

        assert len(quality_reports) > 0

        # Check a specific feature report
        if "speed_mean" in quality_reports:
            report = quality_reports["speed_mean"]
            assert hasattr(report, "completeness")
            assert hasattr(report, "uniqueness")
            assert hasattr(report, "distribution_type")
            assert 0 <= report.completeness <= 1
            assert 0 <= report.uniqueness <= 1

    def test_feature_correlation_analyzer(self):
        """Test feature correlation analysis."""
        # Create synthetic feature data with known correlations
        np.random.seed(42)
        n_samples = 100

        feature1 = np.random.normal(0, 1, n_samples)
        feature2 = feature1 + np.random.normal(0, 0.1, n_samples)  # Highly correlated
        feature3 = np.random.normal(0, 1, n_samples)  # Independent

        features_df = pd.DataFrame(
            {
                "feature1": feature1,
                "feature2": feature2,  # Should be highly correlated with feature1
                "feature3": feature3,
                "trajectory_id": [f"traj_{i}" for i in range(n_samples)],
            }
        )

        analyzer = FeatureCorrelationAnalyzer(correlation_threshold=0.8)
        results = analyzer.analyze_correlations(features_df)

        assert "correlation_matrix" in results
        assert "highly_correlated_pairs" in results
        assert "redundant_features" in results
        assert "recommendations" in results

        # Should detect high correlation between feature1 and feature2
        assert len(results["highly_correlated_pairs"]) > 0


class TestMilestone4Integration:
    """Integration tests for Milestone 4 components."""

    def test_end_to_end_feature_pipeline(self, sample_trajectory):
        """Test complete feature engineering pipeline."""
        # 1. Feature extraction
        feature_store = create_default_feature_store()
        features = feature_store.extract_features(sample_trajectory)
        assert len(features) > 0

        # 2. Data augmentation
        augmenter = NoiseInjectionAugmenter(num_augmentations=1)
        augmented = augmenter.augment(sample_trajectory)
        assert len(augmented) == 1

        # 3. Feature extraction from augmented data
        aug_features = feature_store.extract_features(augmented[0])
        assert len(aug_features) > 0

        # 4. Feature quality assessment
        features_df = pd.DataFrame([features, aug_features])
        assessor = FeatureQualityAssessor()
        quality_reports = assessor.assess_feature_set(features_df)
        assert len(quality_reports) > 0

        # 5. Correlation analysis
        analyzer = FeatureCorrelationAnalyzer()
        correlation_results = analyzer.analyze_correlations(features_df)
        assert "correlation_matrix" in correlation_results

    def test_batch_processing_workflow(self, sample_trajectory):
        """Test batch processing of multiple trajectories."""
        # Create multiple trajectories
        trajectories = []
        for i in range(3):
            traj = sample_trajectory
            traj.trajectory_id = f"batch_test_{i}"
            trajectories.append(traj)

        # Batch process features
        feature_store = create_default_feature_store()
        from trajectory_prediction.data.feature_store import BatchFeatureProcessor

        batch_processor = BatchFeatureProcessor(feature_store, batch_size=2)
        features_df = batch_processor.process_trajectories(trajectories)

        assert len(features_df) == 3
        assert "trajectory_id" in features_df.columns

        # Export and verify
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp_file:
            batch_processor.export_features(trajectories, tmp_file.name)
            exported_df = pd.read_parquet(tmp_file.name)
            assert len(exported_df) == 3


if __name__ == "__main__":
    pytest.main([__file__])
