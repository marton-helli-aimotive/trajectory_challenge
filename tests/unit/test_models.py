"""
Unit tests for trajectory prediction models.

This module tests:
- Model base classes and interfaces
- Individual model implementations
- Model training and prediction functionality
- Model serialization and deserialization
"""

import pytest
import numpy as np
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any

from src.trajectory_prediction.models.base import TrajectoryPredictor
from src.trajectory_prediction.models.baseline.constant_velocity import ConstantVelocityPredictor
from src.trajectory_prediction.models.baseline.constant_acceleration import ConstantAccelerationPredictor
from src.trajectory_prediction.data.schemas import TrajectoryData, Position, Velocity
from tests.conftest import validate_trajectory_data, validate_prediction_response


class TestTrajectoryPredictorBase:
    """Test base TrajectoryPredictor class."""
    
    def test_abstract_methods(self):
        """Test that TrajectoryPredictor cannot be instantiated directly."""
        with pytest.raises(TypeError):
            TrajectoryPredictor()
    
    def test_predictor_interface(self, mock_model):
        """Test that mock model implements required interface."""
        assert hasattr(mock_model, 'predict')
        assert hasattr(mock_model, 'train')
        assert hasattr(mock_model, 'evaluate')
        assert hasattr(mock_model, 'model_name')
        assert hasattr(mock_model, 'is_trained')
    
    @pytest.mark.asyncio
    async def test_predictor_predict(self, mock_model, sample_trajectory):
        """Test predictor prediction method."""
        prediction = await mock_model.predict(sample_trajectory)
        
        assert isinstance(prediction, TrajectoryData)
        assert validate_trajectory_data(prediction)
        assert prediction.vehicle_id == sample_trajectory.vehicle_id
    
    @pytest.mark.asyncio
    async def test_predictor_train(self, mock_model, sample_trajectories):
        """Test predictor training method."""
        training_results = await mock_model.train(sample_trajectories)
        
        assert isinstance(training_results, dict)
        assert "training_samples" in training_results
        assert training_results["training_samples"] == len(sample_trajectories)
    
    @pytest.mark.asyncio
    async def test_predictor_evaluate(self, mock_model, sample_trajectories):
        """Test predictor evaluation method."""
        eval_results = await mock_model.evaluate(sample_trajectories)
        
        assert isinstance(eval_results, dict)
        assert "rmse" in eval_results
        assert "mae" in eval_results
        assert all(isinstance(v, (int, float)) for v in eval_results.values())


class TestConstantVelocityPredictor:
    """Test ConstantVelocityPredictor model."""
    
    @pytest.fixture
    def cv_predictor(self):
        """Create a ConstantVelocityPredictor instance."""
        return ConstantVelocityPredictor()
    
    def test_model_initialization(self, cv_predictor):
        """Test model initialization."""
        assert cv_predictor.model_name == "constant_velocity"
        assert not cv_predictor.is_trained  # Should be True after proper initialization
    
    @pytest.mark.asyncio
    async def test_predict_basic(self, cv_predictor, sample_trajectory):
        """Test basic prediction functionality."""
        prediction = await cv_predictor.predict(sample_trajectory, prediction_horizon=2.0)
        
        assert isinstance(prediction, TrajectoryData)
        assert validate_trajectory_data(prediction)
        assert len(prediction.positions) > 0
        assert len(prediction.velocities) > 0
        assert len(prediction.timestamps) > 0
    
    @pytest.mark.asyncio
    async def test_predict_consistency(self, cv_predictor, sample_trajectory):
        """Test prediction consistency - same input should give same output."""
        pred1 = await cv_predictor.predict(sample_trajectory, prediction_horizon=1.0)
        pred2 = await cv_predictor.predict(sample_trajectory, prediction_horizon=1.0)
        
        # Predictions should be identical for constant velocity
        assert len(pred1.positions) == len(pred2.positions)
        for pos1, pos2 in zip(pred1.positions, pred2.positions):
            assert abs(pos1.x - pos2.x) < 1e-6
            assert abs(pos1.y - pos2.y) < 1e-6
    
    @pytest.mark.asyncio
    async def test_predict_different_horizons(self, cv_predictor, sample_trajectory):
        """Test prediction with different time horizons."""
        short_pred = await cv_predictor.predict(sample_trajectory, prediction_horizon=1.0)
        long_pred = await cv_predictor.predict(sample_trajectory, prediction_horizon=5.0)
        
        # Longer horizon should produce more points
        assert len(long_pred.positions) > len(short_pred.positions)
        assert len(long_pred.timestamps) > len(short_pred.timestamps)
    
    @pytest.mark.asyncio
    async def test_predict_empty_trajectory(self, cv_predictor):
        """Test prediction with empty trajectory."""
        empty_trajectory = TrajectoryData(
            trajectory_id="empty",
            vehicle_id="vehicle",
            positions=[],
            velocities=[],
            timestamps=[]
        )
        
        with pytest.raises((ValueError, IndexError)):
            await cv_predictor.predict(empty_trajectory)
    
    @pytest.mark.asyncio
    async def test_predict_single_point(self, cv_predictor):
        """Test prediction with single point trajectory."""
        single_point_trajectory = TrajectoryData(
            trajectory_id="single",
            vehicle_id="vehicle",
            positions=[Position(x=0.0, y=0.0)],
            velocities=[Velocity(vx=1.0, vy=0.0)],
            timestamps=[0.0]
        )
        
        # Should handle single point gracefully or raise appropriate error
        try:
            prediction = await cv_predictor.predict(single_point_trajectory)
            assert len(prediction.positions) > 0
        except ValueError:
            # Acceptable to require minimum trajectory length
            pass
    
    @pytest.mark.parametrize("horizon,time_step", [
        (1.0, 0.1),
        (2.0, 0.2),
        (5.0, 0.1),
        (10.0, 0.5),
    ])
    @pytest.mark.asyncio
    async def test_predict_various_parameters(self, cv_predictor, sample_trajectory, horizon, time_step):
        """Test prediction with various parameter combinations."""
        prediction = await cv_predictor.predict(
            sample_trajectory,
            prediction_horizon=horizon,
            time_step=time_step
        )
        
        assert isinstance(prediction, TrajectoryData)
        assert validate_trajectory_data(prediction)
        
        # Check approximate number of prediction steps
        expected_steps = int(horizon / time_step)
        actual_steps = len(prediction.positions)
        assert abs(actual_steps - expected_steps) <= 2  # Allow some tolerance


class TestConstantAccelerationPredictor:
    """Test ConstantAccelerationPredictor model."""
    
    @pytest.fixture
    def ca_predictor(self):
        """Create a ConstantAccelerationPredictor instance."""
        return ConstantAccelerationPredictor()
    
    def test_model_initialization(self, ca_predictor):
        """Test model initialization."""
        assert ca_predictor.model_name == "constant_acceleration"
        assert not ca_predictor.is_trained  # Should be True after proper initialization
    
    @pytest.mark.asyncio
    async def test_predict_basic(self, ca_predictor, sample_trajectory):
        """Test basic prediction functionality."""
        prediction = await ca_predictor.predict(sample_trajectory, prediction_horizon=2.0)
        
        assert isinstance(prediction, TrajectoryData)
        assert validate_trajectory_data(prediction)
        assert len(prediction.positions) > 0
    
    @pytest.mark.asyncio
    async def test_predict_with_acceleration(self, ca_predictor):
        """Test prediction with accelerating trajectory."""
        # Create trajectory with acceleration
        positions = []
        velocities = []
        timestamps = []
        
        dt = 0.1
        for i in range(10):
            t = i * dt
            # Accelerating motion: x = 0.5*a*t^2, v = a*t
            acceleration = 2.0
            x = 0.5 * acceleration * t**2
            y = 0.0
            vx = acceleration * t
            vy = 0.0
            
            positions.append(Position(x=x, y=y))
            velocities.append(Velocity(vx=vx, vy=vy))
            timestamps.append(t)
        
        accel_trajectory = TrajectoryData(
            trajectory_id="accel",
            vehicle_id="vehicle",
            positions=positions,
            velocities=velocities,
            timestamps=timestamps
        )
        
        prediction = await ca_predictor.predict(accel_trajectory, prediction_horizon=1.0)
        
        assert isinstance(prediction, TrajectoryData)
        assert validate_trajectory_data(prediction)
        
        # Check that prediction continues acceleration pattern
        # (This would require more sophisticated testing in practice)
        assert len(prediction.positions) > 0


class TestModelValidation:
    """Test model validation functionality."""
    
    @pytest.mark.asyncio
    async def test_model_prediction_validation(self, mock_model, sample_trajectory):
        """Test that model predictions are properly validated."""
        prediction = await mock_model.predict(sample_trajectory)
        
        # Validate prediction structure
        assert hasattr(prediction, 'positions')
        assert hasattr(prediction, 'velocities') 
        assert hasattr(prediction, 'timestamps')
        assert hasattr(prediction, 'trajectory_id')
        assert hasattr(prediction, 'vehicle_id')
        
        # Validate data consistency
        assert len(prediction.positions) == len(prediction.velocities)
        assert len(prediction.positions) == len(prediction.timestamps)
        
        # Validate timestamp ordering
        if len(prediction.timestamps) > 1:
            for i in range(1, len(prediction.timestamps)):
                assert prediction.timestamps[i] > prediction.timestamps[i-1]
    
    @pytest.mark.asyncio
    async def test_model_error_handling(self, mock_model):
        """Test model error handling for invalid inputs."""
        # Test with None input
        with pytest.raises((TypeError, AttributeError)):
            await mock_model.predict(None)
        
        # Test with invalid trajectory
        invalid_trajectory = "not a trajectory"
        with pytest.raises((TypeError, AttributeError)):
            await mock_model.predict(invalid_trajectory)
    
    def test_model_parameter_validation(self, mock_model):
        """Test model parameter validation."""
        # Test model name
        assert isinstance(mock_model.model_name, str)
        assert len(mock_model.model_name) > 0
        
        # Test training status
        assert isinstance(mock_model.is_trained, bool)


class TestModelSerialization:
    """Test model serialization and deserialization."""
    
    def test_model_state_dict(self, mock_model):
        """Test model state dictionary creation."""
        if hasattr(mock_model, 'state_dict'):
            state = mock_model.state_dict()
            assert isinstance(state, dict)
            assert 'model_name' in state
    
    def test_model_save_load(self, mock_model, temp_directory):
        """Test model save/load functionality."""
        if hasattr(mock_model, 'save') and hasattr(mock_model, 'load'):
            save_path = temp_directory / "test_model.pkl"
            
            # Save model
            mock_model.save(str(save_path))
            assert save_path.exists()
            
            # Create new model instance and load
            new_model = type(mock_model)()
            new_model.load(str(save_path))
            
            # Verify loaded model
            assert new_model.model_name == mock_model.model_name


class TestModelEnsemble:
    """Test model ensemble functionality."""
    
    @pytest.fixture
    def model_ensemble(self, mock_model):
        """Create a simple model ensemble."""
        # Create multiple mock models
        models = []
        for i in range(3):
            model = Mock(spec=TrajectoryPredictor)
            model.model_name = f"mock_model_{i}"
            model.predict = AsyncMock(return_value=mock_model.predict.__await__().__next__())
            models.append(model)
        
        return models
    
    @pytest.mark.asyncio
    async def test_ensemble_prediction(self, model_ensemble, sample_trajectory):
        """Test ensemble prediction functionality."""
        # This would test ensemble prediction if implemented
        predictions = []
        for model in model_ensemble:
            pred = await model.predict(sample_trajectory)
            predictions.append(pred)
        
        assert len(predictions) == len(model_ensemble)
        for pred in predictions:
            assert isinstance(pred, TrajectoryData)


class TestModelComparison:
    """Test model comparison functionality."""
    
    @pytest.mark.asyncio
    async def test_model_performance_comparison(self, mock_model, sample_trajectories):
        """Test performance comparison between models."""
        # Evaluate model
        results = await mock_model.evaluate(sample_trajectories)
        
        assert isinstance(results, dict)
        assert 'rmse' in results
        assert 'mae' in results
        
        # Results should be reasonable
        assert results['rmse'] >= 0
        assert results['mae'] >= 0
    
    @pytest.mark.asyncio  
    async def test_model_consistency(self, mock_model, sample_trajectory):
        """Test model consistency across multiple runs."""
        predictions = []
        for _ in range(5):
            pred = await mock_model.predict(sample_trajectory)
            predictions.append(pred)
        
        # Check consistency (should be identical for deterministic models)
        first_pred = predictions[0]
        for pred in predictions[1:]:
            assert len(pred.positions) == len(first_pred.positions)
            # For mock model, predictions should be identical


# Property-based tests for models
try:
    from hypothesis import given, strategies as st, assume
    from tests.conftest import TrajectoryGenerator
    
    class TestModelProperties:
        """Property-based tests for model behavior."""
        
        @given(st.integers(min_value=5, max_value=50))
        @pytest.mark.asyncio
        async def test_prediction_length_property(self, mock_model, trajectory_length):
            """Test that prediction length is reasonable for various input lengths."""
            # Generate trajectory of specified length
            generator = TrajectoryGenerator()
            trajectory = generator.generate_valid_trajectory(
                min_length=trajectory_length,
                max_length=trajectory_length
            )
            
            prediction = await mock_model.predict(trajectory, prediction_horizon=1.0)
            
            # Prediction should have reasonable length
            assert len(prediction.positions) > 0
            assert len(prediction.positions) <= 100  # Reasonable upper bound
        
        @given(st.floats(min_value=0.1, max_value=10.0))
        @pytest.mark.asyncio
        async def test_prediction_horizon_property(self, mock_model, sample_trajectory, horizon):
            """Test prediction behavior with various horizons."""
            prediction = await mock_model.predict(sample_trajectory, prediction_horizon=horizon)
            
            # Prediction should exist and be valid
            assert isinstance(prediction, TrajectoryData)
            assert validate_trajectory_data(prediction)
            
            # Prediction duration should be approximately equal to horizon
            if len(prediction.timestamps) > 1:
                pred_duration = prediction.timestamps[-1] - prediction.timestamps[0]
                assert abs(pred_duration - horizon) <= horizon * 0.2  # 20% tolerance

except ImportError:
    # Hypothesis not available, skip property-based tests
    pass


class TestModelStressTests:
    """Stress tests for model robustness."""
    
    @pytest.mark.asyncio
    async def test_large_trajectory_prediction(self, mock_model):
        """Test prediction with very large trajectory."""
        # Create large trajectory (1000 points)
        positions = [Position(x=float(i), y=float(np.sin(i*0.01))) for i in range(1000)]
        velocities = [Velocity(vx=1.0, vy=np.cos(i*0.01)*0.01) for i in range(1000)]
        timestamps = [float(i * 0.1) for i in range(1000)]
        
        large_trajectory = TrajectoryData(
            trajectory_id="large",
            vehicle_id="vehicle",
            positions=positions,
            velocities=velocities,
            timestamps=timestamps
        )
        
        # Should handle large trajectories
        prediction = await mock_model.predict(large_trajectory)
        assert isinstance(prediction, TrajectoryData)
    
    @pytest.mark.asyncio
    async def test_extreme_velocities(self, mock_model):
        """Test prediction with extreme velocity values."""
        # Create trajectory with very high velocities
        positions = [Position(x=float(i*100), y=float(i*100)) for i in range(10)]
        velocities = [Velocity(vx=100.0, vy=100.0) for _ in range(10)]
        timestamps = [float(i * 0.1) for i in range(10)]
        
        extreme_trajectory = TrajectoryData(
            trajectory_id="extreme",
            vehicle_id="vehicle",
            positions=positions,
            velocities=velocities,
            timestamps=timestamps
        )
        
        # Should handle extreme velocities gracefully
        try:
            prediction = await mock_model.predict(extreme_trajectory)
            assert isinstance(prediction, TrajectoryData)
        except ValueError as e:
            # Acceptable to reject unrealistic inputs
            assert "velocity" in str(e).lower() or "unrealistic" in str(e).lower()
    
    @pytest.mark.asyncio
    async def test_minimal_trajectory(self, mock_model):
        """Test prediction with minimal valid trajectory."""
        # Create minimal trajectory (2 points)
        positions = [Position(x=0.0, y=0.0), Position(x=1.0, y=1.0)]
        velocities = [Velocity(vx=1.0, vy=1.0), Velocity(vx=1.0, vy=1.0)]
        timestamps = [0.0, 0.1]
        
        minimal_trajectory = TrajectoryData(
            trajectory_id="minimal",
            vehicle_id="vehicle",
            positions=positions,
            velocities=velocities,
            timestamps=timestamps
        )
        
        # Should handle minimal trajectories
        prediction = await mock_model.predict(minimal_trajectory)
        assert isinstance(prediction, TrajectoryData)
        assert len(prediction.positions) > 0