"""Test suite for Milestone 5 advanced ML models."""

import pytest
import numpy as np
from datetime import datetime, timedelta
from typing import List

from ..core.models import Trajectory, TrajectoryPoint
from ..core.config import ModelConfig
from .gaussian_process import GaussianProcessPredictor
from .tree_ensemble import TreeEnsemblePredictor
from .mixture_density import MixtureDensityPredictor
from .ensemble import (
    EnsemblePredictor, 
    WeightedAverageStrategy, 
    VotingStrategy, 
    DynamicEnsembleStrategy
)
from .baseline import ConstantVelocityPredictor
from .polynomial import PolynomialRegressionPredictor


def create_sample_trajectory(n_points: int = 10) -> Trajectory:
    """Create a sample trajectory for testing."""
    points = []
    start_time = datetime.now()
    
    for i in range(n_points):
        point = TrajectoryPoint(
            x=float(i * 10),  # Linear motion in x
            y=float(i * 5),   # Linear motion in y
            timestamp=start_time + timedelta(seconds=i),
            velocity=float(np.sqrt(125)),  # sqrt(10^2 + 5^2)
            acceleration=0.0,
            heading=np.arctan2(5, 10)
        )
        points.append(point)
    
    return Trajectory(points=points)


def create_sample_trajectories(n_trajectories: int = 5) -> List[Trajectory]:
    """Create multiple sample trajectories for training."""
    trajectories = []
    
    for i in range(n_trajectories):
        # Create trajectories with different patterns
        points = []
        start_time = datetime.now() + timedelta(hours=i)
        
        for j in range(10):
            # Different motion patterns
            if i % 3 == 0:
                # Linear motion
                x = j * 10 + i * 5
                y = j * 5 + i * 3
            elif i % 3 == 1:
                # Curved motion
                x = j * 10 + i * 5
                y = j * 5 + i * 3 + np.sin(j * 0.5) * 10
            else:
                # Accelerating motion
                x = j * 10 + i * 5 + 0.1 * j * j
                y = j * 5 + i * 3 + 0.05 * j * j
            
            point = TrajectoryPoint(
                x=float(x),
                y=float(y),
                timestamp=start_time + timedelta(seconds=j),
                velocity=float(np.sqrt(125)),
                acceleration=0.0,
                heading=np.arctan2(5, 10)
            )
            points.append(point)
        
        trajectories.append(Trajectory(points=points))
    
    return trajectories


class TestGaussianProcessPredictor:
    """Test Gaussian Process Regression model."""
    
    def test_initialization(self):
        """Test model initialization."""
        config = ModelConfig(
            prediction_horizon=5,
            time_step=1.0,
            kernel_type='rbf',
            noise_variance=1e-6,
            gp_backend='gpytorch'  # Use GPyTorch for testing
        )
        
        model = GaussianProcessPredictor(config)
        assert model.kernel_type == 'rbf'
        assert model.noise_variance == 1e-6
        assert model.backend == 'gpytorch'
        assert not model.is_trained
    
    def test_feature_extraction(self):
        """Test feature extraction from trajectory."""
        config = ModelConfig(prediction_horizon=3, time_step=1.0)
        model = GaussianProcessPredictor(config)
        
        trajectory = create_sample_trajectory(5)
        features, feature_names = model._extract_features(trajectory)
        
        assert isinstance(features, np.ndarray)
        assert isinstance(feature_names, list)
        assert len(features) > 0
        assert len(feature_names) > 0
    
    def test_training_and_prediction(self):
        """Test model training and prediction."""
        config = ModelConfig(
            prediction_horizon=3,
            time_step=1.0,
            kernel_type='rbf',
            gp_backend='gpytorch'
        )
        
        model = GaussianProcessPredictor(config)
        trajectories = create_sample_trajectories(3)
        
        # Train model
        model.train(trajectories)
        assert model.is_trained
        
        # Make prediction
        test_trajectory = create_sample_trajectory(5)
        prediction = model.predict(test_trajectory)
        
        assert prediction is not None
        assert len(prediction.predicted_points) == 3
        assert prediction.uncertainty is not None
        assert 'x_std' in prediction.uncertainty
        assert 'y_std' in prediction.uncertainty


class TestTreeEnsemblePredictor:
    """Test Tree-based Ensemble model."""
    
    def test_initialization(self):
        """Test model initialization."""
        config = ModelConfig(
            prediction_horizon=5,
            time_step=1.0,
            model_type='random_forest',
            n_estimators=10,
            max_depth=5
        )
        
        model = TreeEnsemblePredictor(config)
        assert model.model_type == 'random_forest'
        assert model.n_estimators == 10
        assert model.max_depth == 5
        assert not model.is_trained
    
    def test_feature_extraction(self):
        """Test trajectory feature extraction."""
        config = ModelConfig(prediction_horizon=3, time_step=1.0)
        model = TreeEnsemblePredictor(config)
        
        trajectory = create_sample_trajectory(5)
        features, feature_names = model._extract_trajectory_features(trajectory)
        
        assert isinstance(features, np.ndarray)
        assert isinstance(feature_names, list)
        assert len(features) > 0
        assert len(feature_names) > 0
    
    def test_random_forest_training_and_prediction(self):
        """Test Random Forest training and prediction."""
        config = ModelConfig(
            prediction_horizon=3,
            time_step=1.0,
            model_type='random_forest',
            n_estimators=10,
            max_depth=5
        )
        
        model = TreeEnsemblePredictor(config)
        trajectories = create_sample_trajectories(5)
        
        # Train model
        model.train(trajectories)
        assert model.is_trained
        
        # Make prediction
        test_trajectory = create_sample_trajectory(5)
        prediction = model.predict(test_trajectory)
        
        assert prediction is not None
        assert len(prediction.predicted_points) == 3
        
        # Test feature importance
        importance = model.get_feature_importance()
        assert 'x' in importance
        assert 'y' in importance
    
    def test_xgboost_training_and_prediction(self):
        """Test XGBoost training and prediction."""
        config = ModelConfig(
            prediction_horizon=3,
            time_step=1.0,
            model_type='xgboost',
            n_estimators=10,
            max_depth=5,
            learning_rate=0.1
        )
        
        model = TreeEnsemblePredictor(config)
        trajectories = create_sample_trajectories(5)
        
        # Train model
        model.train(trajectories)
        assert model.is_trained
        
        # Make prediction
        test_trajectory = create_sample_trajectory(5)
        prediction = model.predict(test_trajectory)
        
        assert prediction is not None
        assert len(prediction.predicted_points) == 3


class TestMixtureDensityPredictor:
    """Test Mixture Density Network model."""
    
    def test_initialization(self):
        """Test model initialization."""
        config = ModelConfig(
            prediction_horizon=5,
            time_step=1.0,
            n_components=3,
            hidden_dim=64,
            learning_rate=0.001,
            batch_size=16,
            n_epochs=5  # Small number for testing
        )
        
        model = MixtureDensityPredictor(config)
        assert model.n_components == 3
        assert model.hidden_dim == 64
        assert model.learning_rate == 0.001
        assert not model.is_trained
    
    def test_feature_extraction(self):
        """Test feature extraction from trajectory."""
        config = ModelConfig(prediction_horizon=3, time_step=1.0)
        model = MixtureDensityPredictor(config)
        
        trajectory = create_sample_trajectory(5)
        features, feature_names = model._extract_features(trajectory)
        
        assert isinstance(features, np.ndarray)
        assert isinstance(feature_names, list)
        assert len(features) > 0
        assert len(feature_names) > 0
    
    def test_training_and_prediction(self):
        """Test model training and prediction."""
        config = ModelConfig(
            prediction_horizon=3,
            time_step=1.0,
            n_components=3,
            hidden_dim=32,
            learning_rate=0.001,
            batch_size=8,
            n_epochs=3  # Small number for testing
        )
        
        model = MixtureDensityPredictor(config)
        trajectories = create_sample_trajectories(4)
        
        # Train model
        model.train(trajectories)
        assert model.is_trained
        
        # Make prediction
        test_trajectory = create_sample_trajectory(5)
        prediction = model.predict(test_trajectory)
        
        assert prediction is not None
        assert len(prediction.predicted_points) == 3
        assert prediction.confidence_scores is not None
        assert prediction.uncertainty is not None
        assert 'mixing_coefficients' in prediction.uncertainty


class TestEnsembleStrategies:
    """Test ensemble combination strategies."""
    
    def test_weighted_average_strategy(self):
        """Test weighted average combination strategy."""
        from .base import PredictionResult
        
        # Create sample predictions
        predictions = []
        for i in range(3):
            pred = PredictionResult(
                predicted_points=[],
                timestamps=[datetime.now() + timedelta(seconds=j) for j in range(1, 4)],
                x_positions=[float(j + i) for j in range(3)],
                y_positions=[float(j * 2 + i) for j in range(3)],
                velocities=None,
                accelerations=None,
                headings=None,
                confidence_scores=None,
                uncertainty=None
            )
            predictions.append(pred)
        
        strategy = WeightedAverageStrategy()
        weights = [0.5, 0.3, 0.2]
        
        combined = strategy.combine_predictions(predictions, weights)
        
        assert combined is not None
        assert len(combined.x_positions) == 3
        assert len(combined.y_positions) == 3
        assert combined.confidence_scores == weights
    
    def test_voting_strategy(self):
        """Test voting combination strategy."""
        from .base import PredictionResult
        
        # Create sample predictions
        predictions = []
        for i in range(3):
            pred = PredictionResult(
                predicted_points=[],
                timestamps=[datetime.now() + timedelta(seconds=j) for j in range(1, 4)],
                x_positions=[float(j + i) for j in range(3)],
                y_positions=[float(j * 2 + i) for j in range(3)],
                velocities=None,
                accelerations=None,
                headings=None,
                confidence_scores=None,
                uncertainty=None
            )
            predictions.append(pred)
        
        strategy = VotingStrategy(voting_method='median')
        combined = strategy.combine_predictions(predictions)
        
        assert combined is not None
        assert len(combined.x_positions) == 3
        assert len(combined.y_positions) == 3


class TestEnsemblePredictor:
    """Test Ensemble predictor."""
    
    def test_initialization(self):
        """Test ensemble initialization."""
        config = ModelConfig(
            prediction_horizon=5,
            time_step=1.0,
            strategy_type='weighted_average',
            enable_online_learning=True
        )
        
        ensemble = EnsemblePredictor(config)
        assert ensemble.strategy_type == 'weighted_average'
        assert ensemble.enable_online_learning is True
        assert len(ensemble.base_models) == 0
    
    def test_add_remove_models(self):
        """Test adding and removing models from ensemble."""
        config = ModelConfig(prediction_horizon=3, time_step=1.0)
        ensemble = EnsemblePredictor(config)
        
        # Add models
        model1 = ConstantVelocityPredictor(config)
        model2 = PolynomialRegressionPredictor(config)
        
        ensemble.add_model(model1, weight=0.6)
        ensemble.add_model(model2, weight=0.4)
        
        assert len(ensemble.base_models) == 2
        assert len(ensemble.weights) == 2
        
        # Remove model
        ensemble.remove_model(0)
        assert len(ensemble.base_models) == 1
        assert len(ensemble.weights) == 1
    
    def test_ensemble_training_and_prediction(self):
        """Test ensemble training and prediction."""
        config = ModelConfig(
            prediction_horizon=3,
            time_step=1.0,
            strategy_type='weighted_average'
        )
        
        ensemble = EnsemblePredictor(config)
        
        # Add models
        model1 = ConstantVelocityPredictor(config)
        model2 = PolynomialRegressionPredictor(config)
        
        ensemble.add_model(model1, weight=0.6)
        ensemble.add_model(model2, weight=0.4)
        
        # Train ensemble
        trajectories = create_sample_trajectories(4)
        ensemble.train(trajectories)
        assert ensemble.is_trained
        
        # Make prediction
        test_trajectory = create_sample_trajectory(5)
        prediction = ensemble.predict(test_trajectory)
        
        assert prediction is not None
        assert len(prediction.predicted_points) == 3
        
        # Test performance info
        performance = ensemble.get_model_performance()
        assert performance['n_models'] == 2
        assert len(performance['weights']) == 2


def test_model_integration():
    """Test integration of all Milestone 5 models."""
    # Create configuration
    config = ModelConfig(
        prediction_horizon=3,
        time_step=1.0
    )
    
    # Create all model types
    models = [
        GaussianProcessPredictor(config),
        TreeEnsemblePredictor(config),
        MixtureDensityPredictor(config)
    ]
    
    # Create training data
    trajectories = create_sample_trajectories(5)
    
    # Test each model
    for i, model in enumerate(models):
        print(f"Testing model {i+1}: {type(model).__name__}")
        
        try:
            # Train model
            model.train(trajectories)
            assert model.is_trained
            
            # Make prediction
            test_trajectory = create_sample_trajectory(5)
            prediction = model.predict(test_trajectory)
            
            assert prediction is not None
            assert len(prediction.predicted_points) == 3
            
            # Get model info
            info = model.get_model_info()
            assert info is not None
            
            print(f"✓ {type(model).__name__} passed all tests")
            
        except Exception as e:
            print(f"✗ {type(model).__name__} failed: {e}")
            raise


if __name__ == "__main__":
    # Run integration test
    test_model_integration()
    print("All Milestone 5 models integration test passed!")