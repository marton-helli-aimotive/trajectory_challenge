"""
Unit tests for evaluation metrics.

This module tests:
- Trajectory metrics (RMSE, MAE, ADE, FDE)
- Safety metrics (TTC, minimum distance)
- Probabilistic metrics (NLL, calibration)
- Statistical evaluation methods
"""

import pytest
import numpy as np
from typing import List, Dict, Any

from src.trajectory_prediction.evaluation.metrics import TrajectoryMetrics, SafetyMetrics, ProbabilisticMetrics
from src.trajectory_prediction.data.schemas import TrajectoryData, Position, Velocity
from tests.conftest import validate_trajectory_data


class TestTrajectoryMetrics:
    """Test TrajectoryMetrics class."""
    
    @pytest.fixture
    def metrics_calculator(self):
        """Create TrajectoryMetrics instance."""
        return TrajectoryMetrics()
    
    def test_rmse_calculation(self, metrics_calculator):
        """Test RMSE calculation."""
        # Create simple test trajectories
        predicted = [Position(x=1.0, y=1.0), Position(x=2.0, y=2.0), Position(x=3.0, y=3.0)]
        actual = [Position(x=1.1, y=0.9), Position(x=2.1, y=1.9), Position(x=3.1, y=2.9)]
        
        rmse = metrics_calculator.rmse(predicted, actual)
        
        # Expected RMSE calculation
        errors = [np.sqrt((p.x - a.x)**2 + (p.y - a.y)**2) for p, a in zip(predicted, actual)]
        expected_rmse = np.sqrt(np.mean([e**2 for e in errors]))
        
        assert abs(rmse - expected_rmse) < 1e-10
    
    def test_mae_calculation(self, metrics_calculator):
        """Test MAE calculation."""
        predicted = [Position(x=1.0, y=1.0), Position(x=2.0, y=2.0), Position(x=3.0, y=3.0)]
        actual = [Position(x=1.2, y=0.8), Position(x=2.2, y=1.8), Position(x=3.2, y=2.8)]
        
        mae = metrics_calculator.mae(predicted, actual)
        
        # Expected MAE calculation
        errors = [np.sqrt((p.x - a.x)**2 + (p.y - a.y)**2) for p, a in zip(predicted, actual)]
        expected_mae = np.mean(errors)
        
        assert abs(mae - expected_mae) < 1e-10
    
    def test_ade_calculation(self, metrics_calculator):
        """Test ADE (Average Displacement Error) calculation."""
        predicted = [Position(x=0.0, y=0.0), Position(x=1.0, y=1.0), Position(x=2.0, y=2.0)]
        actual = [Position(x=0.1, y=0.1), Position(x=1.1, y=1.1), Position(x=2.1, y=2.1)]
        
        ade = metrics_calculator.ade(predicted, actual)
        
        # ADE should be average of all displacement errors
        displacements = [np.sqrt((p.x - a.x)**2 + (p.y - a.y)**2) for p, a in zip(predicted, actual)]
        expected_ade = np.mean(displacements)
        
        assert abs(ade - expected_ade) < 1e-10
    
    def test_fde_calculation(self, metrics_calculator):
        """Test FDE (Final Displacement Error) calculation."""
        predicted = [Position(x=0.0, y=0.0), Position(x=1.0, y=1.0), Position(x=2.0, y=2.0)]
        actual = [Position(x=0.1, y=0.1), Position(x=1.1, y=1.1), Position(x=2.2, y=2.2)]
        
        fde = metrics_calculator.fde(predicted, actual)
        
        # FDE should be displacement error of final positions
        final_pred = predicted[-1]
        final_actual = actual[-1]
        expected_fde = np.sqrt((final_pred.x - final_actual.x)**2 + (final_pred.y - final_actual.y)**2)
        
        assert abs(fde - expected_fde) < 1e-10
    
    def test_perfect_prediction(self, metrics_calculator):
        """Test metrics with perfect prediction."""
        positions = [Position(x=1.0, y=1.0), Position(x=2.0, y=2.0), Position(x=3.0, y=3.0)]
        predicted = positions
        actual = positions
        
        rmse = metrics_calculator.rmse(predicted, actual)
        mae = metrics_calculator.mae(predicted, actual)
        ade = metrics_calculator.ade(predicted, actual)
        fde = metrics_calculator.fde(predicted, actual)
        
        # All metrics should be zero for perfect prediction
        assert rmse == 0.0
        assert mae == 0.0
        assert ade == 0.0
        assert fde == 0.0
    
    def test_mismatched_lengths(self, metrics_calculator):
        """Test metrics with mismatched trajectory lengths."""
        predicted = [Position(x=1.0, y=1.0), Position(x=2.0, y=2.0)]
        actual = [Position(x=1.0, y=1.0), Position(x=2.0, y=2.0), Position(x=3.0, y=3.0)]
        
        with pytest.raises(ValueError):
            metrics_calculator.rmse(predicted, actual)
    
    def test_empty_trajectories(self, metrics_calculator):
        """Test metrics with empty trajectories."""
        with pytest.raises(ValueError):
            metrics_calculator.rmse([], [])
    
    @pytest.mark.parametrize("metric_name", ["rmse", "mae", "ade", "fde"])
    def test_metric_non_negative(self, metrics_calculator, metric_name):
        """Test that all metrics are non-negative."""
        predicted = [Position(x=np.random.random(), y=np.random.random()) for _ in range(10)]
        actual = [Position(x=np.random.random(), y=np.random.random()) for _ in range(10)]
        
        metric_func = getattr(metrics_calculator, metric_name)
        result = metric_func(predicted, actual)
        
        assert result >= 0.0
    
    def test_trajectory_metrics_integration(self, metrics_calculator, sample_trajectory):
        """Test metrics calculation with full trajectory objects."""
        # Create slightly modified trajectory as prediction
        predicted_positions = []
        for pos in sample_trajectory.positions:
            # Add small noise
            noise_x = np.random.normal(0, 0.1)
            noise_y = np.random.normal(0, 0.1)
            predicted_positions.append(Position(x=pos.x + noise_x, y=pos.y + noise_y))
        
        predicted_trajectory = TrajectoryData(
            trajectory_id="predicted",
            vehicle_id=sample_trajectory.vehicle_id,
            positions=predicted_positions,
            velocities=sample_trajectory.velocities,
            timestamps=sample_trajectory.timestamps
        )
        
        # Calculate all metrics
        metrics = metrics_calculator.calculate_all_metrics(
            predicted_trajectory.positions,
            sample_trajectory.positions
        )
        
        assert isinstance(metrics, dict)
        assert 'rmse' in metrics
        assert 'mae' in metrics  
        assert 'ade' in metrics
        assert 'fde' in metrics
        assert all(v >= 0 for v in metrics.values())


class TestSafetyMetrics:
    """Test SafetyMetrics class."""
    
    @pytest.fixture
    def safety_metrics(self):
        """Create SafetyMetrics instance."""
        return SafetyMetrics()
    
    def test_time_to_collision_calculation(self, safety_metrics):
        """Test TTC calculation."""
        # Create two vehicles on collision course
        traj1_positions = [Position(x=0.0, y=0.0), Position(x=1.0, y=0.0), Position(x=2.0, y=0.0)]
        traj1_velocities = [Velocity(vx=1.0, vy=0.0) for _ in range(3)]
        
        traj2_positions = [Position(x=5.0, y=0.0), Position(x=4.0, y=0.0), Position(x=3.0, y=0.0)]
        traj2_velocities = [Velocity(vx=-1.0, vy=0.0) for _ in range(3)]
        
        traj1 = TrajectoryData("traj1", "v1", traj1_positions, traj1_velocities, [0.0, 1.0, 2.0])
        traj2 = TrajectoryData("traj2", "v2", traj2_positions, traj2_velocities, [0.0, 1.0, 2.0])
        
        ttc = safety_metrics.time_to_collision(traj1, traj2)
        
        # Vehicles start 5m apart, approaching at 2m/s combined speed
        # Should collide in 2.5 seconds
        assert abs(ttc - 2.5) < 0.1
    
    def test_minimum_distance_calculation(self, safety_metrics):
        """Test minimum distance calculation."""
        # Create two parallel trajectories
        traj1_positions = [Position(x=0.0, y=0.0), Position(x=1.0, y=0.0), Position(x=2.0, y=0.0)]
        traj2_positions = [Position(x=0.0, y=2.0), Position(x=1.0, y=2.0), Position(x=2.0, y=2.0)]
        
        traj1 = TrajectoryData("traj1", "v1", traj1_positions, [], [0.0, 1.0, 2.0])
        traj2 = TrajectoryData("traj2", "v2", traj2_positions, [], [0.0, 1.0, 2.0])
        
        min_distance = safety_metrics.minimum_distance(traj1, traj2)
        
        # Minimum distance should be 2.0 (constant separation)
        assert abs(min_distance - 2.0) < 1e-10
    
    def test_lateral_distance_calculation(self, safety_metrics):
        """Test lateral distance calculation."""
        # Create trajectories with lateral separation
        reference_positions = [Position(x=0.0, y=0.0), Position(x=1.0, y=0.0), Position(x=2.0, y=0.0)]
        test_positions = [Position(x=0.0, y=1.0), Position(x=1.0, y=1.0), Position(x=2.0, y=1.0)]
        
        lateral_distances = safety_metrics.lateral_distance(reference_positions, test_positions)
        
        # All lateral distances should be 1.0
        assert all(abs(d - 1.0) < 1e-10 for d in lateral_distances)
    
    def test_no_collision_case(self, safety_metrics):
        """Test TTC when vehicles are not on collision course."""
        # Create diverging trajectories
        traj1_positions = [Position(x=0.0, y=0.0), Position(x=1.0, y=0.0), Position(x=2.0, y=0.0)]
        traj1_velocities = [Velocity(vx=1.0, vy=0.0) for _ in range(3)]
        
        traj2_positions = [Position(x=0.0, y=1.0), Position(x=1.0, y=2.0), Position(x=2.0, y=3.0)]
        traj2_velocities = [Velocity(vx=1.0, vy=1.0) for _ in range(3)]
        
        traj1 = TrajectoryData("traj1", "v1", traj1_positions, traj1_velocities, [0.0, 1.0, 2.0])
        traj2 = TrajectoryData("traj2", "v2", traj2_positions, traj2_velocities, [0.0, 1.0, 2.0])
        
        ttc = safety_metrics.time_to_collision(traj1, traj2)
        
        # Should return infinity or very large value for no collision
        assert ttc > 1000.0 or np.isinf(ttc)


class TestProbabilisticMetrics:
    """Test ProbabilisticMetrics class."""
    
    @pytest.fixture
    def prob_metrics(self):
        """Create ProbabilisticMetrics instance."""
        return ProbabilisticMetrics()
    
    def test_negative_log_likelihood(self, prob_metrics):
        """Test negative log-likelihood calculation."""
        # Create sample predictions with uncertainties
        predictions = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        actuals = np.array([[1.1, 0.9], [2.1, 1.9], [3.1, 2.9]])
        uncertainties = np.array([0.1, 0.1, 0.1])  # Standard deviations
        
        nll = prob_metrics.negative_log_likelihood(predictions, actuals, uncertainties)
        
        assert isinstance(nll, float)
        assert nll > 0  # NLL should be positive
    
    def test_calibration_error(self, prob_metrics):
        """Test calibration error calculation."""
        # Create sample data
        predictions = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        actuals = np.array([[1.1, 0.9], [2.1, 1.9], [3.1, 2.9]])
        confidences = np.array([0.8, 0.9, 0.7])
        
        calibration = prob_metrics.calibration_error(predictions, actuals, confidences)
        
        assert isinstance(calibration, float)
        assert 0 <= calibration <= 1  # Calibration error should be between 0 and 1
    
    def test_expected_calibration_error(self, prob_metrics):
        """Test Expected Calibration Error (ECE) calculation."""
        # Create perfectly calibrated predictions
        predictions = np.array([[i, i] for i in range(100)])
        actuals = predictions + np.random.normal(0, 0.1, (100, 2))
        uncertainties = np.full(100, 0.1)
        
        ece = prob_metrics.expected_calibration_error(predictions, actuals, uncertainties)
        
        assert isinstance(ece, float)
        assert ece >= 0
    
    def test_prediction_interval_coverage(self, prob_metrics):
        """Test prediction interval coverage calculation."""
        # Create sample data where 80% falls within intervals
        n_samples = 100
        predictions = np.zeros((n_samples, 2))
        actuals = np.random.normal(0, 1, (n_samples, 2))
        lower_bounds = np.full((n_samples, 2), -1.96)  # 95% interval
        upper_bounds = np.full((n_samples, 2), 1.96)
        
        coverage = prob_metrics.prediction_interval_coverage(
            predictions, actuals, lower_bounds, upper_bounds
        )
        
        assert isinstance(coverage, float)
        assert 0 <= coverage <= 1
        # Should be close to 0.95 for 95% prediction intervals
        assert 0.8 <= coverage <= 1.0


class TestMetricIntegration:
    """Test integration of different metric types."""
    
    def test_comprehensive_evaluation(self, sample_trajectory, mock_model):
        """Test comprehensive evaluation using all metric types."""
        # This would test integration of all metrics in a real evaluation
        trajectory_metrics = TrajectoryMetrics()
        safety_metrics = SafetyMetrics()
        prob_metrics = ProbabilisticMetrics()
        
        # Create sample prediction
        predicted_positions = []
        for pos in sample_trajectory.positions:
            noise_x = np.random.normal(0, 0.1)
            noise_y = np.random.normal(0, 0.1)
            predicted_positions.append(Position(x=pos.x + noise_x, y=pos.y + noise_y))
        
        # Calculate trajectory metrics
        traj_results = trajectory_metrics.calculate_all_metrics(
            predicted_positions, sample_trajectory.positions
        )
        
        assert isinstance(traj_results, dict)
        assert all(key in traj_results for key in ['rmse', 'mae', 'ade', 'fde'])
    
    def test_metric_consistency(self):
        """Test consistency between different metric calculations."""
        # Create identical trajectories
        positions = [Position(x=float(i), y=float(i)) for i in range(10)]
        
        metrics_calc = TrajectoryMetrics()
        
        # All metrics should be zero for identical trajectories
        rmse = metrics_calc.rmse(positions, positions)
        mae = metrics_calc.mae(positions, positions)
        ade = metrics_calc.ade(positions, positions)
        fde = metrics_calc.fde(positions, positions)
        
        assert rmse == 0.0
        assert mae == 0.0
        assert ade == 0.0
        assert fde == 0.0
    
    def test_metric_bounds(self):
        """Test that metrics have reasonable bounds."""
        # Create test trajectories with known properties
        predicted = [Position(x=0.0, y=0.0), Position(x=1.0, y=0.0)]
        actual = [Position(x=1.0, y=0.0), Position(x=0.0, y=0.0)]
        
        metrics_calc = TrajectoryMetrics()
        
        rmse = metrics_calc.rmse(predicted, actual)
        mae = metrics_calc.mae(predicted, actual)
        
        # RMSE should be >= MAE (equality case)
        assert rmse >= mae
        
        # Both should be positive
        assert rmse > 0
        assert mae > 0


# Property-based tests for metrics
try:
    from hypothesis import given, strategies as st, assume
    
    class TestMetricProperties:
        """Property-based tests for metric calculations."""
        
        @given(
            st.lists(
                st.tuples(
                    st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
                    st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False)
                ),
                min_size=1,
                max_size=100
            )
        )
        def test_rmse_properties(self, coordinates):
            """Test RMSE properties with various coordinate sets."""
            predicted = [Position(x=x, y=y) for x, y in coordinates]
            actual = [Position(x=x + np.random.normal(0, 0.1), y=y + np.random.normal(0, 0.1)) 
                     for x, y in coordinates]
            
            metrics_calc = TrajectoryMetrics()
            rmse = metrics_calc.rmse(predicted, actual)
            
            # RMSE should always be non-negative
            assert rmse >= 0
            
            # RMSE should be zero only when trajectories are identical
            if all(p.x == a.x and p.y == a.y for p, a in zip(predicted, actual)):
                assert rmse == 0
        
        @given(st.integers(min_value=1, max_value=50))
        def test_metric_scaling_property(self, n_points):
            """Test that metrics scale appropriately with trajectory length."""
            # Create trajectory with constant error
            error_magnitude = 1.0
            predicted = [Position(x=float(i), y=0.0) for i in range(n_points)]
            actual = [Position(x=float(i) + error_magnitude, y=0.0) for i in range(n_points)]
            
            metrics_calc = TrajectoryMetrics()
            rmse = metrics_calc.rmse(predicted, actual)
            mae = metrics_calc.mae(predicted, actual)
            
            # For constant error, RMSE and MAE should equal error magnitude
            assert abs(rmse - error_magnitude) < 1e-10
            assert abs(mae - error_magnitude) < 1e-10

except ImportError:
    # Hypothesis not available, skip property-based tests
    pass


class TestMetricEdgeCases:
    """Test edge cases for metric calculations."""
    
    def test_single_point_metrics(self):
        """Test metrics with single-point trajectories."""
        predicted = [Position(x=1.0, y=1.0)]
        actual = [Position(x=2.0, y=2.0)]
        
        metrics_calc = TrajectoryMetrics()
        
        rmse = metrics_calc.rmse(predicted, actual)
        mae = metrics_calc.mae(predicted, actual)
        ade = metrics_calc.ade(predicted, actual)
        fde = metrics_calc.fde(predicted, actual)
        
        # All metrics should equal the single displacement error
        expected_error = np.sqrt((2.0 - 1.0)**2 + (2.0 - 1.0)**2)
        assert abs(rmse - expected_error) < 1e-10
        assert abs(mae - expected_error) < 1e-10
        assert abs(ade - expected_error) < 1e-10
        assert abs(fde - expected_error) < 1e-10
    
    def test_very_small_errors(self):
        """Test metrics with very small errors."""
        predicted = [Position(x=1.0, y=1.0)]
        actual = [Position(x=1.0 + 1e-15, y=1.0 + 1e-15)]
        
        metrics_calc = TrajectoryMetrics()
        
        rmse = metrics_calc.rmse(predicted, actual)
        assert rmse >= 0
        assert rmse < 1e-10  # Should be very small
    
    def test_very_large_errors(self):
        """Test metrics with very large errors."""
        predicted = [Position(x=0.0, y=0.0)]
        actual = [Position(x=1e6, y=1e6)]
        
        metrics_calc = TrajectoryMetrics()
        
        rmse = metrics_calc.rmse(predicted, actual)
        mae = metrics_calc.mae(predicted, actual)
        
        # Should handle large errors without overflow
        assert np.isfinite(rmse)
        assert np.isfinite(mae)
        assert rmse > 1e6
        assert mae > 1e6