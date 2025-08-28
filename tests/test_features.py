"""Tests for feature engineering components."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from datetime import datetime
import tempfile
import shutil
from pathlib import Path

from vehicle_trajectory_prediction.features.extraction import (
    TrajectoryFeatureExtractor,
    VelocityFeatureExtractor,
    AccelerationFeatureExtractor,
    CurvatureFeatureExtractor,
    LaneChangeFeatureExtractor,
    SpatialTemporalFeatureExtractor,
    ContextualFeatureExtractor,
    FeatureExtractionConfig
)
from vehicle_trajectory_prediction.features.quality import (
    TrajectoryQualityMetrics,
    CompletenessMetrics,
    SmoothnessMetrics,
    ConsistencyMetrics,
    PhysicsConstraintValidator,
    QualityMetricsConfig
)
from vehicle_trajectory_prediction.features.augmentation import (
    TrajectoryAugmentor,
    NoiseInjectionAugmentor,
    InterpolationAugmentor,
    SyntheticScenarioGenerator,
    AdversarialExampleGenerator,
    AugmentationConfig
)
from vehicle_trajectory_prediction.features.store import (
    FeatureStore,
    FeatureDefinition,
    FeatureVersion,
    FeatureCache,
    FeatureStoreConfig
)
from vehicle_trajectory_prediction.features.validation import (
    ValidationReport,
    FeatureValidator,
    PhysicsInformedValidator,
    FeatureQualityScorer,
    ValidationConfig
)
from vehicle_trajectory_prediction.core.types import TrajectoryData


class TestFeatureExtraction:
    """Test feature extraction components."""
    
    @pytest.fixture
    def sample_trajectory(self):
        """Create a sample trajectory for testing."""
        timestamps = np.linspace(0, 10, 100)
        x_positions = 10 * timestamps + np.sin(timestamps)  # Forward motion with oscillation
        y_positions = 5 * np.sin(2 * timestamps)  # Lateral oscillation
        
        return TrajectoryData(
            vehicle_id="test_vehicle_1",
            timestamps=timestamps,
            x_positions=x_positions,
            y_positions=y_positions
        )
    
    @pytest.fixture
    def extraction_config(self):
        """Create feature extraction configuration."""
        return FeatureExtractionConfig()
    
    def test_velocity_feature_extractor(self, sample_trajectory, extraction_config):
        """Test velocity feature extraction."""
        extractor = VelocityFeatureExtractor(extraction_config)
        features = extractor.extract_features(sample_trajectory)
        
        assert 'velocity_magnitude' in features
        assert 'velocity_x' in features
        assert 'velocity_y' in features
        assert 'velocity_smoothness' in features
        assert 'velocity_variability' in features
        assert 'max_velocity' in features
        assert 'min_velocity' in features
        assert 'mean_velocity' in features
        assert 'velocity_acceleration_correlation' in features
        
        # Check that velocity magnitudes are positive
        assert np.all(features['velocity_magnitude'] >= 0)
        
        # Check that statistics are reasonable
        assert features['max_velocity'] >= features['min_velocity']
        assert features['mean_velocity'] >= 0
    
    def test_acceleration_feature_extractor(self, sample_trajectory, extraction_config):
        """Test acceleration feature extraction."""
        extractor = AccelerationFeatureExtractor(extraction_config)
        features = extractor.extract_features(sample_trajectory)
        
        assert 'acceleration_magnitude' in features
        assert 'acceleration_x' in features
        assert 'acceleration_y' in features
        assert 'jerk_magnitude' in features
        assert 'jerk_x' in features
        assert 'jerk_y' in features
        assert 'acceleration_smoothness' in features
        assert 'jerk_smoothness' in features
        assert 'max_acceleration' in features
        assert 'min_acceleration' in features
        assert 'mean_acceleration' in features
        assert 'max_jerk' in features
        assert 'min_jerk' in features
        assert 'mean_jerk' in features
        assert 'acceleration_pattern' in features
        assert 'jerk_pattern' in features
        
        # Check that acceleration magnitudes are positive
        if len(features['acceleration_magnitude']) > 0:
            assert np.all(features['acceleration_magnitude'] >= 0)
    
    def test_curvature_feature_extractor(self, sample_trajectory, extraction_config):
        """Test curvature feature extraction."""
        extractor = CurvatureFeatureExtractor(extraction_config)
        features = extractor.extract_features(sample_trajectory)
        
        assert 'curvature' in features
        assert 'curvature_smoothness' in features
        assert 'curvature_variability' in features
        assert 'max_curvature' in features
        assert 'min_curvature' in features
        assert 'mean_curvature' in features
        assert 'curvature_rate' in features
        assert 'turning_radius' in features
        assert 'turning_angle' in features
        assert 'straight_line_ratio' in features
        assert 'curve_complexity' in features
        
        # Check that curvature values are reasonable
        assert features['max_curvature'] >= 0
        assert features['straight_line_ratio'] >= 0 and features['straight_line_ratio'] <= 1
    
    def test_lane_change_feature_extractor(self, sample_trajectory, extraction_config):
        """Test lane change feature extraction."""
        extractor = LaneChangeFeatureExtractor(extraction_config)
        features = extractor.extract_features(sample_trajectory)
        
        assert 'lane_change_detected' in features
        assert 'lane_change_count' in features
        assert 'lane_change_duration' in features
        assert 'lane_change_distance' in features
        assert 'lane_change_velocity' in features
        assert 'lane_change_acceleration' in features
        assert 'lane_change_angle' in features
        assert 'lane_change_smoothness' in features
        assert 'lane_position' in features
        assert 'lane_deviation' in features
        assert 'lane_center_distance' in features
        assert 'lane_boundary_distance' in features
        
        # Check that lane change count is non-negative
        assert features['lane_change_count'] >= 0
    
    def test_spatial_temporal_feature_extractor(self, sample_trajectory, extraction_config):
        """Test spatial-temporal feature extraction."""
        extractor = SpatialTemporalFeatureExtractor(extraction_config)
        features = extractor.extract_features(sample_trajectory)
        
        assert 'spatial_density' in features
        assert 'temporal_density' in features
        assert 'spatial_entropy' in features
        assert 'temporal_entropy' in features
        assert 'spatial_clustering' in features
        assert 'temporal_clustering' in features
        assert 'spatial_regularity' in features
        assert 'temporal_regularity' in features
        assert 'spatial_coverage' in features
        assert 'temporal_coverage' in features
        assert 'spatial_dispersion' in features
        assert 'temporal_dispersion' in features
        
        # Check that densities are positive
        assert features['spatial_density'] >= 0
        assert features['temporal_density'] >= 0
    
    def test_contextual_feature_extractor(self, sample_trajectory, extraction_config):
        """Test contextual feature extraction."""
        extractor = ContextualFeatureExtractor(extraction_config)
        features = extractor.extract_features(sample_trajectory)
        
        assert 'time_of_day_features' in features
        assert 'day_of_week_features' in features
        assert 'seasonal_features' in features
        assert 'road_geometry_features' in features
        assert 'urban_features' in features
        assert 'rural_features' in features
        assert 'highway_features' in features
        assert 'intersection_features' in features
        assert 'traffic_density_features' in features
        assert 'weather_features' in features
    
    def test_trajectory_feature_extractor(self, sample_trajectory, extraction_config):
        """Test main trajectory feature extractor."""
        extractor = TrajectoryFeatureExtractor(extraction_config)
        features = extractor.extract_features(sample_trajectory)
        
        # Check that all expected feature categories are present
        expected_categories = ['velocity', 'acceleration', 'curvature', 'lane_change', 'spatial_temporal', 'contextual']
        for category in expected_categories:
            assert category in features
            assert isinstance(features[category], dict)
        
        # Check total feature count
        total_features = extractor.get_total_feature_count()
        assert total_features > 0
        
        # Check feature names
        feature_names = extractor.get_all_feature_names()
        assert len(feature_names) == len(expected_categories)


class TestQualityMetrics:
    """Test quality metrics components."""
    
    @pytest.fixture
    def sample_trajectory(self):
        """Create a sample trajectory for testing."""
        timestamps = np.linspace(0, 10, 100)
        x_positions = 10 * timestamps + np.sin(timestamps)
        y_positions = 5 * np.sin(2 * timestamps)
        
        return TrajectoryData(
            vehicle_id="test_vehicle_1",
            timestamps=timestamps,
            x_positions=x_positions,
            y_positions=y_positions
        )
    
    @pytest.fixture
    def quality_config(self):
        """Create quality metrics configuration."""
        return QualityMetricsConfig()
    
    def test_completeness_metrics(self, sample_trajectory, quality_config):
        """Test completeness metrics calculation."""
        metrics = CompletenessMetrics(quality_config)
        results = metrics.calculate_metric(sample_trajectory)
        
        assert 'total_points' in results
        assert 'is_sufficient_length' in results
        assert 'length_score' in results
        assert 'missing_values' in results
        assert 'missing_ratio' in results
        assert 'is_acceptable_missing' in results
        assert 'completeness_score' in results
        assert 'time_coverage' in results
        assert 'is_acceptable_time_coverage' in results
        assert 'has_valid_timestamps' in results
        assert 'has_valid_positions' in results
        assert 'overall_completeness_score' in results
        
        # Check that scores are between 0 and 1
        assert 0 <= results['length_score'] <= 1
        assert 0 <= results['completeness_score'] <= 1
        assert 0 <= results['time_coverage'] <= 1
        assert 0 <= results['overall_completeness_score'] <= 1
    
    def test_smoothness_metrics(self, sample_trajectory, quality_config):
        """Test smoothness metrics calculation."""
        metrics = SmoothnessMetrics(quality_config)
        results = metrics.calculate_metric(sample_trajectory)
        
        assert 'position_smoothness_score' in results
        assert 'velocity_smoothness_score' in results
        assert 'acceleration_smoothness_score' in results
        assert 'jerk_smoothness_score' in results
        assert 'overall_smoothness_score' in results
        assert 'position_jumps' in results
        assert 'velocity_jumps' in results
        assert 'acceleration_jumps' in results
        assert 'jerk_violations' in results
        
        # Check that scores are between 0 and 1
        assert 0 <= results['position_smoothness_score'] <= 1
        assert 0 <= results['velocity_smoothness_score'] <= 1
        assert 0 <= results['acceleration_smoothness_score'] <= 1
        assert 0 <= results['jerk_smoothness_score'] <= 1
        assert 0 <= results['overall_smoothness_score'] <= 1
    
    def test_consistency_metrics(self, sample_trajectory, quality_config):
        """Test consistency metrics calculation."""
        metrics = ConsistencyMetrics(quality_config)
        results = metrics.calculate_metric(sample_trajectory)
        
        assert 'velocity_consistency_score' in results
        assert 'acceleration_consistency_score' in results
        assert 'temporal_consistency_score' in results
        assert 'spatial_consistency_score' in results
        assert 'overall_consistency_score' in results
        assert 'velocity_violations' in results
        assert 'acceleration_violations' in results
        assert 'temporal_violations' in results
        assert 'spatial_violations' in results
        
        # Check that scores are between 0 and 1
        assert 0 <= results['velocity_consistency_score'] <= 1
        assert 0 <= results['acceleration_consistency_score'] <= 1
        assert 0 <= results['temporal_consistency_score'] <= 1
        assert 0 <= results['spatial_consistency_score'] <= 1
        assert 0 <= results['overall_consistency_score'] <= 1
    
    def test_physics_constraint_validator(self, sample_trajectory, quality_config):
        """Test physics constraint validation."""
        validator = PhysicsConstraintValidator(quality_config)
        results = validator.calculate_metric(sample_trajectory)
        
        assert 'centripetal_physics_score' in results
        assert 'turning_radius_physics_score' in results
        assert 'angular_velocity_physics_score' in results
        assert 'energy_physics_score' in results
        assert 'overall_physics_score' in results
        assert 'centripetal_violations' in results
        assert 'turning_radius_violations' in results
        assert 'angular_velocity_violations' in results
        assert 'energy_violations' in results
        
        # Check that scores are between 0 and 1
        assert 0 <= results['centripetal_physics_score'] <= 1
        assert 0 <= results['turning_radius_physics_score'] <= 1
        assert 0 <= results['angular_velocity_physics_score'] <= 1
        assert 0 <= results['energy_physics_score'] <= 1
        assert 0 <= results['overall_physics_score'] <= 1
    
    def test_trajectory_quality_metrics(self, sample_trajectory, quality_config):
        """Test main trajectory quality metrics."""
        metrics = TrajectoryQualityMetrics(quality_config)
        results = metrics.calculate_quality_metrics(sample_trajectory)
        
        # Check that all metric categories are present
        expected_categories = ['completeness', 'smoothness', 'consistency', 'physics', 'overall']
        for category in expected_categories:
            assert category in results
        
        # Check overall quality score
        overall_score = results['overall']['quality_score']
        assert 0 <= overall_score <= 1
        
        # Test quality summary
        summary = metrics.get_quality_summary(results)
        assert 'overall_quality_score' in summary
        assert 'completeness_score' in summary
        assert 'smoothness_score' in summary
        assert 'consistency_score' in summary
        assert 'physics_score' in summary
        assert 'total_violations' in summary
        assert 'quality_level' in summary
        
        # Check quality level
        assert summary['quality_level'] in ['excellent', 'good', 'fair', 'poor', 'very_poor']


class TestDataAugmentation:
    """Test data augmentation components."""
    
    @pytest.fixture
    def sample_trajectory(self):
        """Create a sample trajectory for testing."""
        timestamps = np.linspace(0, 10, 50)
        x_positions = 10 * timestamps + np.sin(timestamps)
        y_positions = 5 * np.sin(2 * timestamps)
        
        return TrajectoryData(
            vehicle_id="test_vehicle_1",
            timestamps=timestamps,
            x_positions=x_positions,
            y_positions=y_positions
        )
    
    @pytest.fixture
    def augmentation_config(self):
        """Create augmentation configuration."""
        return AugmentationConfig()
    
    def test_noise_injection_augmentor(self, sample_trajectory, augmentation_config):
        """Test noise injection augmentation."""
        augmentor = NoiseInjectionAugmentor(augmentation_config)
        augmented = augmentor.augment_trajectory(sample_trajectory)
        
        # Check that augmented trajectory has same structure
        assert augmented.vehicle_id == sample_trajectory.vehicle_id
        assert len(augmented.timestamps) == len(sample_trajectory.timestamps)
        assert len(augmented.x_positions) == len(sample_trajectory.x_positions)
        assert len(augmented.y_positions) == len(sample_trajectory.y_positions)
        
        # Check that some noise was added (positions should be different)
        assert not np.allclose(augmented.x_positions, sample_trajectory.x_positions)
        assert not np.allclose(augmented.y_positions, sample_trajectory.y_positions)
    
    def test_interpolation_augmentor(self, sample_trajectory, augmentation_config):
        """Test interpolation augmentation."""
        augmentor = InterpolationAugmentor(augmentation_config)
        augmented = augmentor.augment_trajectory(sample_trajectory)
        
        # Check that augmented trajectory has more points
        assert len(augmented.timestamps) >= len(sample_trajectory.timestamps)
        assert len(augmented.x_positions) >= len(sample_trajectory.x_positions)
        assert len(augmented.y_positions) >= len(sample_trajectory.y_positions)
        
        # Check that timestamps are monotonically increasing
        assert np.all(np.diff(augmented.timestamps) > 0)
    
    def test_synthetic_scenario_generator(self, sample_trajectory, augmentation_config):
        """Test synthetic scenario generation."""
        generator = SyntheticScenarioGenerator(augmentation_config)
        synthetic = generator.augment_trajectory(sample_trajectory)
        
        # Check that synthetic trajectory has reasonable structure
        assert synthetic.vehicle_id.startswith("synthetic_")
        assert len(synthetic.timestamps) > 0
        assert len(synthetic.x_positions) > 0
        assert len(synthetic.y_positions) > 0
        assert len(synthetic.timestamps) == len(synthetic.x_positions) == len(synthetic.y_positions)
        
        # Check that timestamps are monotonically increasing
        assert np.all(np.diff(synthetic.timestamps) > 0)
    
    def test_adversarial_example_generator(self, sample_trajectory, augmentation_config):
        """Test adversarial example generation."""
        generator = AdversarialExampleGenerator(augmentation_config)
        adversarial = generator.augment_trajectory(sample_trajectory)
        
        # Check that adversarial trajectory has same structure
        assert adversarial.vehicle_id.startswith("adversarial_")
        assert len(adversarial.timestamps) == len(sample_trajectory.timestamps)
        assert len(adversarial.x_positions) == len(sample_trajectory.x_positions)
        assert len(adversarial.y_positions) == len(sample_trajectory.y_positions)
        
        # Check that some perturbation was added
        assert not np.allclose(adversarial.x_positions, sample_trajectory.x_positions)
        assert not np.allclose(adversarial.y_positions, sample_trajectory.y_positions)
    
    def test_trajectory_augmentor(self, sample_trajectory, augmentation_config):
        """Test main trajectory augmentor."""
        augmentor = TrajectoryAugmentor(augmentation_config)
        
        # Test single trajectory augmentation
        augmented = augmentor.augment_trajectory(sample_trajectory)
        assert augmented.vehicle_id != sample_trajectory.vehicle_id or len(augmented.timestamps) != len(sample_trajectory.timestamps)
        
        # Test batch augmentation
        trajectories = [sample_trajectory]
        augmented_batch = augmentor.augment_trajectories(trajectories)
        assert len(augmented_batch) == 1
        
        # Test available methods
        methods = augmentor.get_available_methods()
        assert len(methods) > 0
        assert all(method in ['noise', 'interpolation', 'synthetic', 'adversarial'] for method in methods)
        
        # Test augmentation statistics
        stats = augmentor.get_augmentation_statistics(trajectories, augmented_batch)
        assert 'total_trajectories' in stats
        assert 'successful_augmentations' in stats
        assert 'failed_augmentations' in stats


class TestFeatureStore:
    """Test feature store components."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def store_config(self, temp_dir):
        """Create feature store configuration."""
        return FeatureStoreConfig(base_path=temp_dir)
    
    @pytest.fixture
    def sample_feature_definition(self):
        """Create sample feature definition."""
        return FeatureDefinition(
            name="test_velocity_features",
            description="Test velocity features for trajectory",
            feature_type="velocity",
            data_type="dict",
            shape=None,
            default_value={},
            validation_rules={"max_velocity": 50.0},
            dependencies=["trajectory_data"],
            version="1.0.0"
        )
    
    def test_feature_definition(self, sample_feature_definition):
        """Test feature definition functionality."""
        # Test to_dict and from_dict
        definition_dict = sample_feature_definition.to_dict()
        reconstructed = FeatureDefinition.from_dict(definition_dict)
        
        assert reconstructed.name == sample_feature_definition.name
        assert reconstructed.description == sample_feature_definition.description
        assert reconstructed.feature_type == sample_feature_definition.feature_type
        assert reconstructed.data_type == sample_feature_definition.data_type
        assert reconstructed.version == sample_feature_definition.version
        
        # Test hash generation
        hash_value = sample_feature_definition.get_hash()
        assert isinstance(hash_value, str)
        assert len(hash_value) > 0
    
    def test_feature_version(self):
        """Test feature version functionality."""
        version = FeatureVersion(
            feature_name="test_feature",
            version="1.1.0",
            created_at=datetime.now(),
            description="Updated feature version",
            changes=["Added new velocity metrics", "Fixed curvature calculation"],
            is_active=True
        )
        
        # Test to_dict and from_dict
        version_dict = version.to_dict()
        reconstructed = FeatureVersion.from_dict(version_dict)
        
        assert reconstructed.feature_name == version.feature_name
        assert reconstructed.version == version.version
        assert reconstructed.description == version.description
        assert reconstructed.changes == version.changes
        assert reconstructed.is_active == version.is_active
    
    def test_feature_cache(self):
        """Test feature cache functionality."""
        cache_entry = FeatureCache(
            feature_name="test_feature",
            trajectory_id="test_trajectory",
            feature_data={"velocity": [1.0, 2.0, 3.0]},
            computed_at=datetime.now(),
            expires_at=datetime.now()
        )
        
        # Test to_dict and from_dict
        cache_dict = cache_entry.to_dict()
        reconstructed = FeatureCache.from_dict(cache_dict)
        
        assert reconstructed.feature_name == cache_entry.feature_name
        assert reconstructed.trajectory_id == cache_entry.trajectory_id
        assert reconstructed.feature_data == cache_entry.feature_data
        
        # Test expiration
        assert cache_entry.is_expired()
    
    def test_feature_store_operations(self, store_config, sample_feature_definition):
        """Test feature store operations."""
        store = FeatureStore(store_config)
        
        # Test feature registration
        success = store.register_feature(sample_feature_definition)
        assert success
        
        # Test feature definition retrieval
        retrieved_definition = store.get_feature_definition("test_velocity_features")
        assert retrieved_definition is not None
        assert retrieved_definition.name == sample_feature_definition.name
        
        # Test feature storage and retrieval
        feature_data = {"velocity_magnitude": [1.0, 2.0, 3.0], "velocity_x": [0.5, 1.0, 1.5]}
        trajectory_id = "test_trajectory_1"
        
        # Store feature
        store_success = store.store_feature("test_velocity_features", trajectory_id, feature_data)
        assert store_success
        
        # Retrieve feature
        retrieved_data = store.retrieve_feature("test_velocity_features", trajectory_id)
        assert retrieved_data is not None
        assert retrieved_data["velocity_magnitude"] == feature_data["velocity_magnitude"]
        assert retrieved_data["velocity_x"] == feature_data["velocity_x"]
        
        # Test feature listing
        features = store.list_features()
        assert "test_velocity_features" in features
        
        # Test trajectory listing
        trajectories = store.list_trajectories_for_feature("test_velocity_features")
        assert trajectory_id in trajectories
        
        # Test feature deletion
        delete_success = store.delete_feature("test_velocity_features", trajectory_id)
        assert delete_success
        
        # Verify deletion
        retrieved_after_delete = store.retrieve_feature("test_velocity_features", trajectory_id)
        assert retrieved_after_delete is None
    
    def test_feature_versioning(self, store_config, sample_feature_definition):
        """Test feature versioning functionality."""
        store = FeatureStore(store_config)
        
        # Register feature
        store.register_feature(sample_feature_definition)
        
        # Create version
        success = store.create_feature_version(
            "test_velocity_features",
            "1.1.0",
            "Updated velocity features",
            ["Added acceleration correlation", "Improved smoothing"]
        )
        assert success
        
        # Get versions
        versions = store.get_feature_versions("test_velocity_features")
        assert len(versions) > 0
        
        # Get latest version
        latest_version = store.get_latest_version("test_velocity_features")
        assert latest_version is not None
        assert latest_version.version == "1.1.0"
    
    def test_feature_store_statistics(self, store_config, sample_feature_definition):
        """Test feature store statistics."""
        store = FeatureStore(store_config)
        
        # Register feature
        store.register_feature(sample_feature_definition)
        
        # Get statistics
        stats = store.get_store_statistics()
        assert 'total_features' in stats
        assert 'total_versions' in stats
        assert 'cache_size' in stats
        assert 'storage_size_mb' in stats
        
        assert stats['total_features'] > 0
        assert stats['cache_size'] >= 0
        assert stats['storage_size_mb'] >= 0


class TestFeatureValidation:
    """Test feature validation components."""
    
    @pytest.fixture
    def sample_trajectory(self):
        """Create a sample trajectory for testing."""
        timestamps = np.linspace(0, 10, 50)
        x_positions = 10 * timestamps + np.sin(timestamps)
        y_positions = 5 * np.sin(2 * timestamps)
        
        return TrajectoryData(
            vehicle_id="test_vehicle_1",
            timestamps=timestamps,
            x_positions=x_positions,
            y_positions=y_positions
        )
    
    @pytest.fixture
    def sample_features(self):
        """Create sample features for testing."""
        return {
            'velocity': {
                'velocity_magnitude': [10.0, 12.0, 15.0, 11.0, 13.0],
                'velocity_x': [8.0, 10.0, 12.0, 9.0, 11.0],
                'velocity_y': [6.0, 7.0, 9.0, 5.0, 8.0]
            },
            'acceleration': {
                'acceleration_magnitude': [2.0, 3.0, 1.0, 2.5, 1.5],
                'acceleration_x': [1.5, 2.0, 0.8, 2.0, 1.2],
                'acceleration_y': [1.0, 1.5, 0.6, 1.5, 0.9]
            },
            'curvature': {
                'curvature': [0.01, 0.02, 0.015, 0.025, 0.018],
                'turning_radius': [100.0, 50.0, 66.7, 40.0, 55.6]
            },
            'spatial_temporal': {
                'spatial_density': 0.5,
                'temporal_density': 5.0,
                'spatial_entropy': 2.5,
                'temporal_entropy': 1.8
            }
        }
    
    @pytest.fixture
    def validation_config(self):
        """Create validation configuration."""
        return ValidationConfig()
    
    def test_feature_validator(self, sample_features, sample_trajectory, validation_config):
        """Test basic feature validation."""
        validator = FeatureValidator(validation_config)
        results = validator.validate_features(sample_features, sample_trajectory)
        
        assert 'validation_name' in results
        assert 'trajectory_id' in results
        assert 'validation_timestamp' in results
        assert 'overall_score' in results
        assert 'issues' in results
        assert 'warnings' in results
        assert 'passed' in results
        assert 'completeness_score' in results
        assert 'consistency_score' in results
        assert 'duplicate_score' in results
        assert 'type_score' in results
        
        # Check that scores are between 0 and 1
        assert 0 <= results['overall_score'] <= 1
        assert 0 <= results['completeness_score'] <= 1
        assert 0 <= results['consistency_score'] <= 1
        assert 0 <= results['duplicate_score'] <= 1
        assert 0 <= results['type_score'] <= 1
    
    def test_physics_informed_validator(self, sample_features, sample_trajectory, validation_config):
        """Test physics-informed validation."""
        validator = PhysicsInformedValidator(validation_config)
        results = validator.validate_features(sample_features, sample_trajectory)
        
        assert 'validation_name' in results
        assert 'trajectory_id' in results
        assert 'validation_timestamp' in results
        assert 'overall_score' in results
        assert 'physics_violations' in results
        assert 'issues' in results
        assert 'warnings' in results
        assert 'passed' in results
        assert 'velocity_physics_score' in results
        assert 'acceleration_physics_score' in results
        assert 'curvature_physics_score' in results
        assert 'spatial_temporal_physics_score' in results
        
        # Check that scores are between 0 and 1
        assert 0 <= results['overall_score'] <= 1
        assert 0 <= results['velocity_physics_score'] <= 1
        assert 0 <= results['acceleration_physics_score'] <= 1
        assert 0 <= results['curvature_physics_score'] <= 1
        assert 0 <= results['spatial_temporal_physics_score'] <= 1
    
    def test_feature_quality_scorer(self, sample_features, sample_trajectory, validation_config):
        """Test feature quality scoring."""
        scorer = FeatureQualityScorer(validation_config)
        results = scorer.validate_features(sample_features, sample_trajectory)
        
        assert 'validation_name' in results
        assert 'trajectory_id' in results
        assert 'validation_timestamp' in results
        assert 'overall_quality_score' in results
        assert 'quality_breakdown' in results
        assert 'quality_level' in results
        assert 'recommendations' in results
        
        # Check quality breakdown
        quality_breakdown = results['quality_breakdown']
        assert 'completeness' in quality_breakdown
        assert 'consistency' in quality_breakdown
        assert 'physics' in quality_breakdown
        assert 'statistics' in quality_breakdown
        
        # Check that scores are between 0 and 1
        assert 0 <= results['overall_quality_score'] <= 1
        assert 0 <= quality_breakdown['completeness'] <= 1
        assert 0 <= quality_breakdown['consistency'] <= 1
        assert 0 <= quality_breakdown['physics'] <= 1
        assert 0 <= quality_breakdown['statistics'] <= 1
        
        # Check quality level
        assert results['quality_level'] in ['excellent', 'good', 'fair', 'poor', 'very_poor']
        
        # Check recommendations
        assert isinstance(results['recommendations'], list)
    
    def test_validation_report(self, sample_features, sample_trajectory, validation_config):
        """Test validation report generation."""
        report_generator = ValidationReport(validation_config)
        report = report_generator.generate_validation_report(sample_features, sample_trajectory)
        
        assert 'report_metadata' in report
        assert 'validation_results' in report
        assert 'summary' in report
        assert 'recommendations' in report
        
        # Check report metadata
        metadata = report['report_metadata']
        assert 'trajectory_id' in metadata
        assert 'generation_timestamp' in metadata
        assert 'feature_count' in metadata
        assert 'trajectory_length' in metadata
        
        # Check validation results
        validation_results = report['validation_results']
        assert 'basic' in validation_results
        assert 'physics' in validation_results
        assert 'quality' in validation_results
        
        # Check summary
        summary = report['summary']
        assert 'overall_passed' in summary
        assert 'overall_score' in summary
        assert 'validator_scores' in summary
        assert 'critical_issues' in summary
        assert 'warnings' in summary
        
        # Check recommendations
        assert isinstance(report['recommendations'], list)
    
    def test_batch_validation(self, sample_features, sample_trajectory, validation_config):
        """Test batch validation functionality."""
        report_generator = ValidationReport(validation_config)
        
        # Create batch data
        features_batch = {
            "test_vehicle_1": sample_features,
            "test_vehicle_2": sample_features
        }
        trajectories = [sample_trajectory, sample_trajectory]
        
        # Generate batch reports
        batch_reports = report_generator.batch_validate_features(features_batch, trajectories)
        
        assert len(batch_reports) == 2
        assert "test_vehicle_1" in batch_reports
        assert "test_vehicle_2" in batch_reports
        
        # Check that each report has the expected structure
        for trajectory_id, report in batch_reports.items():
            assert 'report_metadata' in report
            assert 'validation_results' in report
            assert 'summary' in report
            assert 'recommendations' in report


if __name__ == "__main__":
    pytest.main([__file__])