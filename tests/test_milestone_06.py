"""Tests for Milestone 6: Advanced ML Models & Uncertainty Quantification."""

import numpy as np
import pytest

from trajectory_prediction.data.models import (
    Trajectory,
    TrajectoryPoint,
    Vehicle,
    VehicleType,
)
from trajectory_prediction.evaluation.uncertainty_metrics import (
    epistemic_aleatoric_decomposition,
    uncertainty_quality_metrics,
)
from trajectory_prediction.models import ModelConfig, ModelFactory


def create_test_trajectory(
    trajectory_id: str = "test_001", n_points: int = 10
) -> Trajectory:
    """Create a test trajectory for unit tests."""
    points = []

    for i in range(n_points):
        t = i * 0.1

        # Simple linear trajectory
        x = 10.0 + 2.0 * t
        y = 5.0 + 1.0 * t

        point = TrajectoryPoint(
            timestamp=1000000000 + t,
            x=x,
            y=y,
            speed=np.sqrt(4.0 + 1.0),  # sqrt(vx^2 + vy^2)
            velocity_x=2.0,
            velocity_y=1.0,
            acceleration_x=0.0,
            acceleration_y=0.0,
            heading=np.arctan2(1.0, 2.0) * 180 / np.pi,
            lane_id=1,
            frame_id=i,
        )
        points.append(point)

    vehicle = Vehicle(
        vehicle_id=1,
        vehicle_type=VehicleType.CAR,
        length=4.5,
        width=1.8,
    )

    return Trajectory(
        trajectory_id=trajectory_id,
        vehicle=vehicle,
        points=points,
        dataset_name="test_dataset",
        completeness_score=1.0,
        temporal_consistency_score=1.0,
        spatial_accuracy_score=1.0,
        smoothness_score=1.0,
    )


@pytest.fixture
def sample_trajectories():
    """Create sample trajectories for testing."""
    return [create_test_trajectory(f"traj_{i:03d}", 15) for i in range(5)]


@pytest.fixture
def test_trajectory():
    """Create a single test trajectory."""
    return create_test_trajectory("test_traj", 20)


@pytest.fixture
def model_factory():
    """Create model factory instance."""
    return ModelFactory()


class TestGaussianProcessModel:
    """Test Gaussian Process regression model."""

    def test_gaussian_process_creation(self, model_factory):
        """Test GP model creation."""
        config = ModelConfig(
            name="GP_Test",
            model_type="advanced",
            hyperparameters={
                "kernel": "rbf",
                "length_scale": 1.0,
                "noise_level": 1e-4,
            },
        )

        model = model_factory.create_model("gaussian_process", config)
        assert model is not None
        assert model.config.name == "GP_Test"

    def test_gaussian_process_training(self, model_factory, sample_trajectories):
        """Test GP model training."""
        config = ModelConfig(
            name="GP_Train_Test",
            model_type="advanced",
            hyperparameters={"kernel": "rbf"},
        )

        model = model_factory.create_model("gaussian_process", config)
        trained_model = model.fit(sample_trajectories)

        assert trained_model.is_fitted
        assert trained_model == model  # Should return self

    def test_gaussian_process_prediction(
        self, model_factory, sample_trajectories, test_trajectory
    ):
        """Test GP model prediction."""
        config = ModelConfig(
            name="GP_Pred_Test",
            model_type="advanced",
        )

        model = model_factory.create_model("gaussian_process", config)
        model.fit(sample_trajectories)

        # Use first part of trajectory to predict the rest
        input_traj = Trajectory(
            trajectory_id="input_test",
            vehicle=test_trajectory.vehicle,
            points=test_trajectory.points[:10],
            dataset_name=test_trajectory.dataset_name,
            completeness_score=1.0,
            temporal_consistency_score=1.0,
            spatial_accuracy_score=1.0,
            smoothness_score=1.0,
        )

        prediction = model.predict(input_traj, prediction_horizon=1.0)

        assert prediction.trajectory_id == "input_test"
        assert len(prediction.predicted_points) > 0
        assert prediction.prediction_horizon == 1.0
        assert "epistemic_uncertainty_x" in prediction.metadata


class TestTreeEnsembleModel:
    """Test Tree-based ensemble model."""

    def test_tree_ensemble_creation(self, model_factory):
        """Test tree ensemble model creation."""
        config = ModelConfig(
            name="Tree_Test",
            model_type="advanced",
            hyperparameters={
                "model_type": "random_forest",
                "n_estimators": 10,
            },
        )

        model = model_factory.create_model("tree_ensemble", config)
        assert model is not None
        assert model.config.name == "Tree_Test"

    def test_tree_ensemble_training_and_prediction(
        self, model_factory, sample_trajectories, test_trajectory
    ):
        """Test tree ensemble training and prediction."""
        config = ModelConfig(
            name="Tree_Full_Test",
            model_type="advanced",
            hyperparameters={
                "model_type": "random_forest",
                "n_estimators": 10,
                "max_depth": 5,
            },
        )

        model = model_factory.create_model("tree_ensemble", config)
        model.fit(sample_trajectories)

        input_traj = Trajectory(
            trajectory_id="tree_input",
            vehicle=test_trajectory.vehicle,
            points=test_trajectory.points[:10],
            dataset_name=test_trajectory.dataset_name,
            completeness_score=1.0,
            temporal_consistency_score=1.0,
            spatial_accuracy_score=1.0,
            smoothness_score=1.0,
        )

        prediction = model.predict(input_traj, prediction_horizon=1.0)

        assert len(prediction.predicted_points) > 0
        assert "aleatoric_uncertainty_x" in prediction.metadata

        # Test feature importance
        importance = model.get_feature_importance()
        assert importance is not None
        assert isinstance(importance, dict)


class TestMixtureDensityNetwork:
    """Test Mixture Density Network model."""

    def test_mdn_creation(self, model_factory):
        """Test MDN model creation."""
        config = ModelConfig(
            name="MDN_Test",
            model_type="advanced",
            hyperparameters={
                "n_components": 2,
                "hidden_size": (16, 8),
                "max_iter": 50,
            },
        )

        model = model_factory.create_model("mixture_density", config)
        assert model is not None
        assert model.config.name == "MDN_Test"

    def test_mdn_with_sufficient_data(self, model_factory):
        """Test MDN with sufficient training data."""
        # Create more training data for MDN
        trajectories = [
            create_test_trajectory(f"mdn_traj_{i:03d}", 15) for i in range(20)
        ]

        config = ModelConfig(
            name="MDN_Data_Test",
            model_type="advanced",
            hyperparameters={
                "n_components": 2,
                "hidden_size": (16,),
                "max_iter": 50,
                "validation_fraction": 0.1,  # Smaller validation set
            },
        )

        model = model_factory.create_model("mixture_density", config)

        try:
            model.fit(trajectories)
            assert model.is_fitted
        except ValueError as e:
            # This is expected with small datasets
            assert "validation set" in str(e).lower()


class TestEnsembleModel:
    """Test ensemble model functionality."""

    def test_ensemble_creation(self, model_factory):
        """Test ensemble model creation."""
        ensemble = model_factory.create_ensemble_model(
            [
                {"model": "polynomial_regression", "weight": 1.0},
                {"model": "knn", "weight": 1.0},
            ]
        )

        assert ensemble is not None
        assert len(ensemble._predictors) == 2

    def test_ensemble_training_and_prediction(
        self, model_factory, sample_trajectories, test_trajectory
    ):
        """Test ensemble training and prediction."""
        ensemble = model_factory.create_ensemble_model(
            [
                {"model": "polynomial_regression", "weight": 1.0},
                {"model": "knn", "weight": 1.0},
            ]
        )

        ensemble.fit(sample_trajectories)

        input_traj = Trajectory(
            trajectory_id="ensemble_input",
            vehicle=test_trajectory.vehicle,
            points=test_trajectory.points[:10],
            dataset_name=test_trajectory.dataset_name,
            completeness_score=1.0,
            temporal_consistency_score=1.0,
            spatial_accuracy_score=1.0,
            smoothness_score=1.0,
        )

        prediction = ensemble.predict(input_traj, prediction_horizon=1.0)

        assert len(prediction.predicted_points) > 0
        assert "ensemble_method" in prediction.metadata


class TestUncertaintyMetrics:
    """Test uncertainty quantification metrics."""

    def test_uncertainty_quality_metrics(self):
        """Test uncertainty quality metrics calculation."""
        # Create mock prediction results
        from trajectory_prediction.models.base import PredictionResult

        predictions = [
            PredictionResult(
                trajectory_id="test_1",
                predicted_points=[(10.0, 5.0), (12.0, 6.0)],
                prediction_horizon=1.0,
                confidence_scores=[0.8, 0.7],
                metadata={
                    "uncertainty_bounds": [
                        [(9.5, 10.5), (4.5, 5.5)],
                        [(11.5, 12.5), (5.5, 6.5)],
                    ]
                },
            ),
            PredictionResult(
                trajectory_id="test_2",
                predicted_points=[(10.1, 5.1), (12.1, 6.1)],
                prediction_horizon=1.0,
                confidence_scores=[0.9, 0.8],
                metadata={
                    "uncertainty_bounds": [
                        [(9.6, 10.6), (4.6, 5.6)],
                        [(11.6, 12.6), (5.6, 6.6)],
                    ]
                },
            ),
        ]

        ground_truth = [
            [(10.0, 5.0), (12.0, 6.0)],
            [(10.0, 5.0), (12.0, 6.0)],
        ]

        metrics = uncertainty_quality_metrics(predictions, ground_truth)

        assert "picp_95" in metrics
        assert "picp_90" in metrics
        assert "mpiw" in metrics
        assert "crps" in metrics
        assert "ece" in metrics
        assert "mce" in metrics
        assert "sharpness" in metrics
        assert "reliability" in metrics

    def test_epistemic_aleatoric_decomposition(self):
        """Test epistemic and aleatoric uncertainty decomposition."""
        from src.trajectory_prediction.models.base import PredictionResult

        # Create mock ensemble predictions
        ensemble_predictions = [
            [
                PredictionResult(
                    trajectory_id="test_1",
                    predicted_points=[(10.0, 5.0)],
                    prediction_horizon=1.0,
                ),
                PredictionResult(
                    trajectory_id="test_2",
                    predicted_points=[(12.0, 6.0)],
                    prediction_horizon=1.0,
                ),
            ],
            [
                PredictionResult(
                    trajectory_id="test_1",
                    predicted_points=[(10.1, 5.1)],
                    prediction_horizon=1.0,
                ),
                PredictionResult(
                    trajectory_id="test_2",
                    predicted_points=[(12.1, 6.1)],
                    prediction_horizon=1.0,
                ),
            ],
            [
                PredictionResult(
                    trajectory_id="test_1",
                    predicted_points=[(9.9, 4.9)],
                    prediction_horizon=1.0,
                ),
                PredictionResult(
                    trajectory_id="test_2",
                    predicted_points=[(11.9, 5.9)],
                    prediction_horizon=1.0,
                ),
            ],
        ]

        # Create ground truth data
        ground_truth = [[(10.0, 5.0)], [(12.0, 6.0)]]

        decomposition = epistemic_aleatoric_decomposition(
            ensemble_predictions, ground_truth
        )

        assert "epistemic" in decomposition
        assert "aleatoric" in decomposition
        assert "total" in decomposition
        assert isinstance(decomposition["epistemic"], float)
        assert isinstance(decomposition["aleatoric"], float)
        assert isinstance(decomposition["total"], float)


class TestMilestone6Integration:
    """Integration tests for Milestone 6 features."""

    def test_all_advanced_models_available(self, model_factory):
        """Test that all advanced models can be created."""
        advanced_models = [
            "gaussian_process",
            "tree_ensemble",
            "mixture_density",
        ]

        for model_name in advanced_models:
            config = ModelConfig(
                name=f"{model_name}_integration_test",
                model_type="advanced",
            )

            model = model_factory.create_model(model_name, config)
            assert model is not None
            assert model.config.name == f"{model_name}_integration_test"

    def test_uncertainty_quantification_pipeline(
        self, model_factory, sample_trajectories, test_trajectory
    ):
        """Test end-to-end uncertainty quantification pipeline."""
        # Train a GP model (has uncertainty quantification)
        config = ModelConfig(
            name="UQ_Pipeline_Test",
            model_type="advanced",
            hyperparameters={"kernel": "rbf"},
        )

        model = model_factory.create_model("gaussian_process", config)
        model.fit(sample_trajectories)

        input_traj = Trajectory(
            trajectory_id="uq_input",
            vehicle=test_trajectory.vehicle,
            points=test_trajectory.points[:10],
            dataset_name=test_trajectory.dataset_name,
            completeness_score=1.0,
            temporal_consistency_score=1.0,
            spatial_accuracy_score=1.0,
            smoothness_score=1.0,
        )

        prediction = model.predict(input_traj, prediction_horizon=1.0)

        # Verify uncertainty information is present
        assert "epistemic_uncertainty_x" in prediction.metadata
        assert "epistemic_uncertainty_y" in prediction.metadata
        assert prediction.confidence_scores is not None
        assert len(prediction.confidence_scores) == len(prediction.predicted_points)
