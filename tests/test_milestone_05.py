"""Tests for Milestone 5: Baseline & Classical Prediction Models."""

import numpy as np
import pandas as pd
import pytest

from trajectory_prediction.data.models import (
    Trajectory,
    TrajectoryPoint,
    Vehicle,
    VehicleType,
)
from trajectory_prediction.evaluation import (
    CrossValidationRunner,
    ModelEvaluator,
    TimeSeriesCrossValidator,
)
from trajectory_prediction.models.base import ModelConfig, TrajectoryPredictor
from trajectory_prediction.models.baseline import (
    ConstantAccelerationPredictor,
    ConstantVelocityPredictor,
)
from trajectory_prediction.models.factory import ModelFactory, create_model
from trajectory_prediction.models.knn import KNearestNeighborsPredictor
from trajectory_prediction.models.polynomial import PolynomialRegressionPredictor


@pytest.fixture
def sample_trajectory():
    """Create a sample trajectory for testing."""
    points = []
    base_timestamp = 1704067200.0  # 2024-01-01 00:00:00 UTC
    for i in range(20):  # 2 seconds at 10 Hz
        t = base_timestamp + i * 0.1
        x = 10 * (i * 0.1)
        y = 5 + 0.5 * (i * 0.1)
        speed = 10.0

        point = TrajectoryPoint(
            timestamp=t,
            x=x,
            y=y,
            speed=speed,
            velocity_x=10.0,
            velocity_y=0.5,
            acceleration_x=0.0,
            acceleration_y=0.0,
            heading=2.86,  # atan2(0.5, 10.0) in degrees
            lane_id=1,
            frame_id=i,
        )
        points.append(point)

    vehicle = Vehicle(vehicle_id=1, vehicle_type=VehicleType.CAR, length=4.5, width=2.0)

    return Trajectory(
        trajectory_id="test_001",
        vehicle=vehicle,
        points=points,
        dataset_name="test",
        completeness_score=1.0,
        temporal_consistency_score=1.0,
        spatial_accuracy_score=1.0,
        smoothness_score=1.0,
    )


@pytest.fixture
def sample_trajectories():
    """Create multiple sample trajectories for testing."""
    trajectories = []
    np.random.seed(42)
    base_timestamp = 1704067200.0  # 2024-01-01 00:00:00 UTC

    for traj_id in range(5):
        points = []
        start_x = traj_id * 100
        start_y = traj_id * 10

        for i in range(30):  # 3 seconds at 10 Hz
            t = base_timestamp + traj_id * 5.0 + i * 0.1  # Offset each trajectory
            x = start_x + 15 * (i * 0.1) + np.random.normal(0, 0.1)
            y = start_y + 2 * (i * 0.1) + np.random.normal(0, 0.1)

            point = TrajectoryPoint(
                timestamp=t,
                x=x,
                y=y,
                speed=15.0,
                velocity_x=15.0,
                velocity_y=2.0,
                acceleration_x=0.0,
                acceleration_y=0.0,
                heading=7.6,  # atan2(2, 15) in degrees
                lane_id=1,
                frame_id=i,
            )
            points.append(point)

        vehicle = Vehicle(
            vehicle_id=traj_id, vehicle_type=VehicleType.CAR, length=4.5, width=2.0
        )

        trajectory = Trajectory(
            trajectory_id=f"test_{traj_id:03d}",
            vehicle=vehicle,
            points=points,
            dataset_name="test",
            completeness_score=1.0,
            temporal_consistency_score=1.0,
            spatial_accuracy_score=1.0,
            smoothness_score=1.0,
        )
        trajectories.append(trajectory)

    return trajectories


class TestModelFactory:
    """Test model factory functionality."""

    def test_factory_creation(self):
        """Test that factory can be created."""
        factory = ModelFactory()
        assert factory is not None
        assert hasattr(factory, "registry")

    def test_available_models(self):
        """Test that factory lists available models."""
        factory = ModelFactory()
        available = factory.registry.get_available_models()

        expected_models = {
            "constant_velocity",
            "constant_acceleration",
            "polynomial",
            "knn",
        }
        assert expected_models.issubset(available)

    def test_create_models(self):
        """Test creating different model types."""
        configs = [
            ("constant_velocity", {}),
            ("constant_acceleration", {}),
            ("polynomial", {"degree": 2}),
            ("knn", {"n_neighbors": 3}),
        ]

        for model_type, params in configs:
            config = ModelConfig(
                name=f"test_{model_type}", model_type=model_type, hyperparameters=params
            )

            model = create_model(model_type, config)
            assert isinstance(model, TrajectoryPredictor)
            assert model.config.model_type == model_type


class TestBaselineModels:
    """Test baseline prediction models."""

    def test_constant_velocity_model(self, sample_trajectory):
        """Test constant velocity model."""
        config = ModelConfig(
            name="cv_test", model_type="constant_velocity", hyperparameters={}
        )

        model = ConstantVelocityPredictor(config)
        assert not model.is_fitted

        # Train model
        model.fit([sample_trajectory])
        assert model.is_fitted

        # Test prediction
        history_points = sample_trajectory.points[:10]
        history_trajectory = Trajectory(
            trajectory_id="history_test",
            vehicle=sample_trajectory.vehicle,
            points=history_points,
            dataset_name="test",
            completeness_score=1.0,
            temporal_consistency_score=1.0,
            spatial_accuracy_score=1.0,
            smoothness_score=1.0,
        )
        prediction = model.predict(history_trajectory, prediction_horizon=1.0)

        assert prediction is not None
        assert prediction.trajectory_id == "history_test"
        assert len(prediction.predicted_points) > 0

    def test_constant_acceleration_model(self, sample_trajectory):
        """Test constant acceleration model."""
        config = ModelConfig(
            name="ca_test", model_type="constant_acceleration", hyperparameters={}
        )

        model = ConstantAccelerationPredictor(config)
        assert not model.is_fitted

        # Train model
        model.fit([sample_trajectory])
        assert model.is_fitted

        # Test prediction
        history_points = sample_trajectory.points[:10]
        history_trajectory = Trajectory(
            trajectory_id="history_test",
            vehicle=sample_trajectory.vehicle,
            points=history_points,
            dataset_name="test",
            completeness_score=1.0,
            temporal_consistency_score=1.0,
            spatial_accuracy_score=1.0,
            smoothness_score=1.0,
        )
        prediction = model.predict(history_trajectory, prediction_horizon=1.0)

        assert prediction is not None
        assert len(prediction.predicted_points) > 0


class TestClassicalModels:
    """Test classical ML prediction models."""

    def test_polynomial_model(self, sample_trajectories):
        """Test polynomial regression model."""
        config = ModelConfig(
            name="poly_test",
            model_type="polynomial",
            hyperparameters={"degree": 2, "alpha": 0.1},
        )

        model = PolynomialRegressionPredictor(config)
        assert not model.is_fitted

        # Train model
        model.fit(sample_trajectories)
        assert model.is_fitted

        # Test prediction
        history_points = sample_trajectories[0].points[:10]
        history_trajectory = Trajectory(
            trajectory_id="history_test",
            vehicle=sample_trajectories[0].vehicle,
            points=history_points,
            dataset_name="test",
            completeness_score=1.0,
            temporal_consistency_score=1.0,
            spatial_accuracy_score=1.0,
            smoothness_score=1.0,
        )
        prediction = model.predict(history_trajectory, prediction_horizon=1.0)

        assert prediction is not None
        assert len(prediction.predicted_points) > 0

    def test_knn_model(self, sample_trajectories):
        """Test KNN model."""
        config = ModelConfig(
            name="knn_test",
            model_type="knn",
            hyperparameters={"n_neighbors": 3, "history_length": 1.0},
        )

        model = KNearestNeighborsPredictor(config)
        assert not model.is_fitted

        # Train model
        model.fit(sample_trajectories)
        assert model.is_fitted

        # Test prediction
        history_points = sample_trajectories[0].points[:10]
        history_trajectory = Trajectory(
            trajectory_id="history_test",
            vehicle=sample_trajectories[0].vehicle,
            points=history_points,
            dataset_name="test",
            completeness_score=1.0,
            temporal_consistency_score=1.0,
            spatial_accuracy_score=1.0,
            smoothness_score=1.0,
        )
        prediction = model.predict(history_trajectory, prediction_horizon=1.0)

        assert prediction is not None
        assert len(prediction.predicted_points) > 0


class TestEvaluationFramework:
    """Test evaluation framework."""

    def test_model_evaluator(self, sample_trajectories):
        """Test model evaluator."""
        evaluator = ModelEvaluator()

        # Create a simple model for testing
        config = ModelConfig(
            name="test_model", model_type="constant_velocity", hyperparameters={}
        )
        model = ConstantVelocityPredictor(config)
        model.fit(sample_trajectories)

        # Prepare test data
        test_cases = evaluator.prepare_test_data(
            sample_trajectories, prediction_horizon=1.0
        )
        assert len(test_cases) > 0

        # Evaluate model
        performance = evaluator.evaluate_model(
            model, test_cases, prediction_horizon=1.0
        )

        assert performance is not None
        assert performance.rmse >= 0
        assert performance.ade >= 0
        assert performance.fde >= 0
        assert performance.training_time >= 0
        assert performance.inference_time >= 0

    def test_cross_validation(self, sample_trajectories):
        """Test cross-validation framework."""
        cv = TimeSeriesCrossValidator(n_splits=2, test_size=0.3)

        # Test splitting
        splits = list(cv.split(sample_trajectories))
        assert len(splits) == 2

        for train_trajs, test_trajs in splits:
            assert len(train_trajs) > 0
            assert len(test_trajs) > 0

    def test_cross_validation_runner(self, sample_trajectories):
        """Test cross-validation runner."""
        # Create models
        models = {
            "cv": ConstantVelocityPredictor(
                ModelConfig(
                    name="cv_test", model_type="constant_velocity", hyperparameters={}
                )
            ),
            "ca": ConstantAccelerationPredictor(
                ModelConfig(
                    name="ca_test",
                    model_type="constant_acceleration",
                    hyperparameters={},
                )
            ),
        }

        # Run cross-validation
        cv_runner = CrossValidationRunner()
        cv_runner.cv.n_splits = 2  # Smaller for testing

        comparison_df = cv_runner.compare_models_cv(
            models, sample_trajectories, refit_per_fold=True
        )

        assert isinstance(comparison_df, pd.DataFrame)
        assert len(comparison_df) == 2  # Two models
        assert "model_name" in comparison_df.columns
        assert "mean_rmse" in comparison_df.columns
        assert "mean_ade" in comparison_df.columns
        assert "mean_fde" in comparison_df.columns


class TestModelPersistence:
    """Test model persistence functionality."""

    def test_model_config_serialization(self):
        """Test that model configurations can be serialized."""
        config = ModelConfig(
            name="test_model",
            model_type="polynomial",
            hyperparameters={"degree": 3, "alpha": 0.1},
            random_state=42,
        )

        # Test JSON serialization
        config_dict = config.model_dump()
        assert config_dict["name"] == "test_model"
        assert config_dict["model_type"] == "polynomial"
        assert config_dict["hyperparameters"]["degree"] == 3

        # Test deserialization
        restored_config = ModelConfig(**config_dict)
        assert restored_config.name == config.name
        assert restored_config.model_type == config.model_type
        assert restored_config.hyperparameters == config.hyperparameters


class TestModelValidation:
    """Test model validation and error handling."""

    def test_invalid_model_type(self):
        """Test creating model with invalid type."""
        config = ModelConfig(
            name="invalid_test", model_type="invalid_model_type", hyperparameters={}
        )

        with pytest.raises(ValueError):
            create_model("invalid_model_type", config)

    def test_empty_training_data(self):
        """Test training with empty data."""
        config = ModelConfig(
            name="empty_test", model_type="constant_velocity", hyperparameters={}
        )

        model = ConstantVelocityPredictor(config)

        with pytest.raises(ValueError):
            model.fit([])

    def test_invalid_prediction_horizon(self, sample_trajectory):
        """Test prediction with invalid horizon."""
        config = ModelConfig(
            name="invalid_horizon_test",
            model_type="constant_velocity",
            hyperparameters={},
        )

        model = ConstantVelocityPredictor(config)
        model.fit([sample_trajectory])

        history_points = sample_trajectory.points[:5]
        history_trajectory = Trajectory(
            trajectory_id="history_test",
            vehicle=sample_trajectory.vehicle,
            points=history_points,
            dataset_name="test",
            completeness_score=1.0,
            temporal_consistency_score=1.0,
            spatial_accuracy_score=1.0,
            smoothness_score=1.0,
        )

        # Test negative horizon
        with pytest.raises(ValueError):
            model.predict(history_trajectory, prediction_horizon=-1.0)

        # Test zero horizon
        with pytest.raises(ValueError):
            model.predict(history_trajectory, prediction_horizon=0.0)


def test_milestone_05_success_criteria():
    """Test that all Milestone 5 success criteria are met."""

    # Success Criterion 1: 4 baseline/classical models implemented
    factory = ModelFactory()
    available_models = factory.registry.get_available_models()
    required_models = {
        "constant_velocity",
        "constant_acceleration",
        "polynomial",
        "knn",
    }
    assert required_models.issubset(available_models), (
        "Not all required models available"
    )

    # Success Criterion 2: Model factory pattern for easy model selection
    for model_type in required_models:
        config = ModelConfig(name="test", model_type=model_type, hyperparameters={})
        model = create_model(model_type, config)
        assert isinstance(model, TrajectoryPredictor), f"Failed to create {model_type}"

    # Success Criterion 3: Hyperparameter optimization support (configuration)
    poly_config = ModelConfig(
        name="poly_test",
        model_type="polynomial",
        hyperparameters={"degree": 3, "alpha": 0.1},
    )
    poly_model = create_model("polynomial", poly_config)
    assert poly_model.config.hyperparameters["degree"] == 3

    # Success Criterion 4: Cross-validation pipeline for temporal data
    cv = TimeSeriesCrossValidator(n_splits=3)
    assert cv.n_splits == 3
    assert hasattr(cv, "split")

    # Success Criterion 5: Initial evaluation metrics implementation
    evaluator = ModelEvaluator()
    assert hasattr(evaluator, "evaluate_model")
    assert hasattr(evaluator, "prepare_test_data")

    print("✅ All Milestone 5 success criteria verified!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
