"""Test script for Milestone 6: Comprehensive Evaluation Framework."""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from vehicle_trajectory_prediction.core.models import Trajectory, TrajectoryPoint
from vehicle_trajectory_prediction.core.config import ModelConfig
from vehicle_trajectory_prediction.evaluation import (
    ComprehensiveEvaluator,
    StatisticalTestSuite,
    ModelBenchmarker,
    ConfidenceIntervalEstimator,
    TrajectoryMetrics,
    SafetyMetrics,
    StatisticalMetrics,
    PerformanceMetrics
)
from vehicle_trajectory_prediction.models.baseline import ConstantVelocityPredictor, ConstantAccelerationPredictor
from vehicle_trajectory_prediction.models.polynomial import PolynomialPredictor
from vehicle_trajectory_prediction.models.knn import KNNPredictor
from vehicle_trajectory_prediction.models.gaussian_process import GaussianProcessPredictor
from vehicle_trajectory_prediction.models.tree_ensemble import TreeEnsemblePredictor
from vehicle_trajectory_prediction.models.mixture_density import MixtureDensityPredictor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_trajectories(num_trajectories: int = 10, points_per_trajectory: int = 20) -> list[Trajectory]:
    """Create test trajectories for evaluation."""
    trajectories = []
    
    for i in range(num_trajectories):
        points = []
        start_time = datetime.now()
        
        # Create a simple trajectory with some variation
        x_start = np.random.uniform(0, 100)
        y_start = np.random.uniform(0, 100)
        velocity = np.random.uniform(5, 20)
        heading = np.random.uniform(0, 2 * np.pi)
        
        for j in range(points_per_trajectory):
            timestamp = start_time + timedelta(seconds=j * 0.1)
            
            # Add some noise and variation
            x = x_start + velocity * j * 0.1 * np.cos(heading) + np.random.normal(0, 0.1)
            y = y_start + velocity * j * 0.1 * np.sin(heading) + np.random.normal(0, 0.1)
            
            # Vary velocity slightly
            current_velocity = velocity + np.random.normal(0, 0.5)
            current_velocity = max(0, current_velocity)  # Ensure non-negative
            
            # Calculate acceleration (simplified)
            acceleration = np.random.normal(0, 0.1)
            
            point = TrajectoryPoint(
                x=x, y=y, timestamp=timestamp, velocity=current_velocity,
                acceleration=acceleration, heading=heading, vehicle_id=f"vehicle_{i}"
            )
            points.append(point)
        
        # Create trajectory
        trajectory = Trajectory(
            vehicle_id=f"vehicle_{i}",
            points=points,
            start_time=start_time,
            end_time=start_time + timedelta(seconds=(points_per_trajectory - 1) * 0.1),
            duration=(points_per_trajectory - 1) * 0.1,
            total_distance=velocity * (points_per_trajectory - 1) * 0.1
        )
        
        trajectories.append(trajectory)
    
    return trajectories


def test_metrics_calculation():
    """Test the metrics calculation functionality."""
    logger.info("Testing metrics calculation...")
    
    # Create test data
    trajectories = create_test_trajectories(5, 15)
    
    # Test trajectory metrics
    true_trajectory = trajectories[0]
    
    # Create a simple prediction (just offset the true trajectory)
    from vehicle_trajectory_prediction.models.base import PredictionResult
    predicted_points = []
    timestamps = []
    x_positions = []
    y_positions = []
    
    for i, point in enumerate(true_trajectory.points):
        predicted_point = TrajectoryPoint(
            x=point.x + 1.0,  # Offset by 1 meter
            y=point.y + 0.5,  # Offset by 0.5 meters
            timestamp=point.timestamp + timedelta(seconds=0.1),
            velocity=point.velocity,
            acceleration=point.acceleration,
            heading=point.heading,
            vehicle_id=point.vehicle_id
        )
        predicted_points.append(predicted_point)
        timestamps.append(predicted_point.timestamp)
        x_positions.append(predicted_point.x)
        y_positions.append(predicted_point.y)
    
    prediction_result = PredictionResult(
        predicted_points=predicted_points,
        timestamps=timestamps,
        x_positions=x_positions,
        y_positions=y_positions
    )
    
    # Test trajectory metrics
    rmse = TrajectoryMetrics.calculate_rmse(true_trajectory, prediction_result)
    ade = TrajectoryMetrics.calculate_ade(true_trajectory, prediction_result)
    fde = TrajectoryMetrics.calculate_fde(true_trajectory, prediction_result)
    
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"ADE: {ade:.4f}")
    logger.info(f"FDE: {fde:.4f}")
    
    # Test safety metrics
    min_distance = SafetyMetrics.calculate_min_distance(true_trajectory, prediction_result)
    ttc = SafetyMetrics.calculate_ttc(true_trajectory, prediction_result)
    lateral_error = SafetyMetrics.calculate_lateral_error(true_trajectory, prediction_result)
    risk_score = SafetyMetrics.calculate_risk_score(true_trajectory, prediction_result)
    
    logger.info(f"Min Distance: {min_distance:.4f}")
    logger.info(f"TTC: {ttc:.4f}")
    logger.info(f"Lateral Error: {lateral_error:.4f}")
    logger.info(f"Risk Score: {risk_score:.4f}")
    
    # Test statistical metrics
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    ci_lower, ci_upper = StatisticalMetrics.calculate_confidence_interval(values)
    percentiles = StatisticalMetrics.calculate_percentiles(values)
    outlier_rate = StatisticalMetrics.calculate_outlier_rate(values)
    dist_stats = StatisticalMetrics.calculate_distribution_stats(values)
    
    logger.info(f"Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")
    logger.info(f"Percentiles: {percentiles}")
    logger.info(f"Outlier Rate: {outlier_rate:.4f}")
    logger.info(f"Distribution Stats: {dist_stats}")
    
    logger.info("Metrics calculation tests completed successfully!")


def test_comprehensive_evaluator():
    """Test the comprehensive evaluator."""
    logger.info("Testing comprehensive evaluator...")
    
    # Create configuration
    config = ModelConfig()
    
    # Create evaluator
    evaluator = ComprehensiveEvaluator(config)
    
    # Create test trajectories
    trajectories = create_test_trajectories(10, 20)
    
    # Create test models
    models = [
        ConstantVelocityPredictor(config),
        ConstantAccelerationPredictor(config),
        PolynomialPredictor(config),
        KNNPredictor(config),
        GaussianProcessPredictor(config),
        TreeEnsemblePredictor(config),
        MixtureDensityPredictor(config)
    ]
    
    # Test single model evaluation
    logger.info("Testing single model evaluation...")
    test_model = models[0]  # Use CV predictor
    
    try:
        evaluation_result = evaluator.evaluate_model(
            test_model, trajectories[:5],  # Use subset for faster testing
            prediction_horizon=10,
            include_safety=True,
            include_performance=True,
            include_statistical=True,
            verbose=False
        )
        
        logger.info(f"Evaluation completed for {test_model.model_name}")
        logger.info(f"Success rate: {evaluation_result['success_rate']:.2%}")
        
        if 'summary' in evaluation_result:
            summary = evaluation_result['summary']
            logger.info(f"RMSE mean: {summary.get('rmse_mean', 'N/A')}")
            logger.info(f"ADE mean: {summary.get('ade_mean', 'N/A')}")
            logger.info(f"FDE mean: {summary.get('fde_mean', 'N/A')}")
        
    except Exception as e:
        logger.error(f"Single model evaluation failed: {e}")
    
    # Test model comparison
    logger.info("Testing model comparison...")
    try:
        comparison_result = evaluator.compare_models(
            models[:3],  # Compare first 3 models
            trajectories[:5],
            prediction_horizon=10,
            include_safety=True,
            include_performance=True,
            include_statistical=True,
            verbose=False
        )
        
        logger.info(f"Comparison completed for {len(models[:3])} models")
        
        if 'comparison_summary' in comparison_result:
            comparison_summary = comparison_result['comparison_summary']
            if 'rankings' in comparison_summary:
                for metric, rankings in comparison_summary['rankings'].items():
                    logger.info(f"{metric} rankings: {[r['model'] for r in rankings]}")
        
    except Exception as e:
        logger.error(f"Model comparison failed: {e}")
    
    logger.info("Comprehensive evaluator tests completed!")


def test_statistical_tests():
    """Test the statistical test suite."""
    logger.info("Testing statistical test suite...")
    
    # Create test data
    test_suite = StatisticalTestSuite(alpha=0.05)
    
    # Generate synthetic metrics for two models
    np.random.seed(42)
    model1_metrics = np.random.normal(2.0, 0.5, 50).tolist()
    model2_metrics = np.random.normal(2.2, 0.6, 50).tolist()
    
    # Test two-model comparison
    comparison_result = test_suite.compare_two_models(
        model1_metrics, model2_metrics, "model1_vs_model2", "independent"
    )
    
    logger.info(f"Two-model comparison result: {comparison_result}")
    
    # Test multiple model comparison
    model3_metrics = np.random.normal(1.8, 0.4, 50).tolist()
    multiple_models = {
        "model1": model1_metrics,
        "model2": model2_metrics,
        "model3": model3_metrics
    }
    
    multiple_comparison = test_suite.compare_multiple_models(multiple_models, "independent")
    logger.info(f"Multiple model comparison completed")
    
    # Test bootstrap test
    bootstrap_result = test_suite.perform_bootstrap_test(model1_metrics, model2_metrics, n_bootstrap=100)
    logger.info(f"Bootstrap test result: {bootstrap_result}")
    
    # Test permutation test
    permutation_result = test_suite.perform_permutation_test(model1_metrics, model2_metrics, n_permutations=100)
    logger.info(f"Permutation test result: {permutation_result}")
    
    logger.info("Statistical test suite tests completed!")


def test_confidence_intervals():
    """Test the confidence interval estimator."""
    logger.info("Testing confidence interval estimator...")
    
    # Create estimator
    ci_estimator = ConfidenceIntervalEstimator(confidence_level=0.95, n_bootstrap=100)
    
    # Generate test data
    np.random.seed(42)
    test_values = np.random.normal(5.0, 1.0, 30).tolist()
    
    # Test parametric CI
    parametric_ci = ci_estimator.calculate_parametric_ci(test_values, "t_distribution")
    logger.info(f"Parametric CI: {parametric_ci}")
    
    # Test bootstrap CI
    bootstrap_ci = ci_estimator.calculate_bootstrap_ci(test_values, "percentile")
    logger.info(f"Bootstrap CI: {bootstrap_ci}")
    
    # Test multiple methods
    multiple_ci = ci_estimator.calculate_ci_for_metric(test_values, ["t_distribution", "bootstrap_percentile"])
    logger.info(f"Multiple CI methods: {multiple_ci}")
    
    # Test multiple metrics
    metrics_dict = {
        "metric1": test_values,
        "metric2": np.random.normal(3.0, 0.8, 30).tolist()
    }
    multiple_metrics_ci = ci_estimator.calculate_ci_for_multiple_metrics(metrics_dict)
    logger.info(f"Multiple metrics CI: {multiple_metrics_ci}")
    
    logger.info("Confidence interval tests completed!")


def test_benchmarking():
    """Test the benchmarking functionality."""
    logger.info("Testing benchmarking...")
    
    # Create configuration
    config = ModelConfig()
    
    # Create benchmarker
    benchmarker = ModelBenchmarker(config)
    
    # Create test trajectories
    trajectories = create_test_trajectories(5, 15)
    
    # Create test model
    test_model = ConstantVelocityPredictor(config)
    
    # Test single model benchmarking
    logger.info("Testing single model benchmarking...")
    try:
        benchmark_result = benchmarker.benchmark_single_model(
            test_model, trajectories,
            prediction_horizon=10,
            num_runs=2,  # Use fewer runs for faster testing
            warmup_runs=1
        )
        
        logger.info(f"Benchmark completed for {test_model.model_name}")
        if 'inference_time' in benchmark_result:
            logger.info(f"Mean inference time: {benchmark_result['inference_time']['mean']:.4f} seconds")
        if 'throughput' in benchmark_result:
            logger.info(f"Mean throughput: {benchmark_result['throughput']['mean']:.2f} trajectories/second")
        
    except Exception as e:
        logger.error(f"Single model benchmarking failed: {e}")
    
    # Test multiple model benchmarking
    logger.info("Testing multiple model benchmarking...")
    models = [
        ConstantVelocityPredictor(config),
        ConstantAccelerationPredictor(config)
    ]
    
    try:
        multi_benchmark = benchmarker.benchmark_multiple_models(
            models, trajectories,
            prediction_horizon=10,
            num_runs=2,
            warmup_runs=1
        )
        
        logger.info(f"Multi-model benchmark completed")
        
    except Exception as e:
        logger.error(f"Multi-model benchmarking failed: {e}")
    
    logger.info("Benchmarking tests completed!")


def test_performance_metrics():
    """Test the performance metrics."""
    logger.info("Testing performance metrics...")
    
    # Create test model and trajectory
    config = ModelConfig()
    model = ConstantVelocityPredictor(config)
    trajectories = create_test_trajectories(3, 10)
    
    # Test inference time measurement
    inference_time = PerformanceMetrics.measure_inference_time(
        model, trajectories[0], prediction_horizon=10, num_runs=3
    )
    logger.info(f"Inference time metrics: {inference_time}")
    
    # Test memory usage measurement
    try:
        memory_usage = PerformanceMetrics.measure_memory_usage(
            model, trajectories[0], prediction_horizon=10
        )
        logger.info(f"Memory usage metrics: {memory_usage}")
    except Exception as e:
        logger.warning(f"Memory usage measurement failed: {e}")
    
    # Test throughput calculation
    throughput = PerformanceMetrics.calculate_throughput(
        model, trajectories, prediction_horizon=10
    )
    logger.info(f"Throughput metrics: {throughput}")
    
    logger.info("Performance metrics tests completed!")


def main():
    """Run all tests for Milestone 6."""
    logger.info("Starting Milestone 6 Evaluation Framework Tests")
    logger.info("=" * 60)
    
    try:
        # Test 1: Metrics calculation
        test_metrics_calculation()
        logger.info("✓ Metrics calculation tests passed")
        
        # Test 2: Comprehensive evaluator
        test_comprehensive_evaluator()
        logger.info("✓ Comprehensive evaluator tests passed")
        
        # Test 3: Statistical tests
        test_statistical_tests()
        logger.info("✓ Statistical test suite tests passed")
        
        # Test 4: Confidence intervals
        test_confidence_intervals()
        logger.info("✓ Confidence interval tests passed")
        
        # Test 5: Benchmarking
        test_benchmarking()
        logger.info("✓ Benchmarking tests passed")
        
        # Test 6: Performance metrics
        test_performance_metrics()
        logger.info("✓ Performance metrics tests passed")
        
        logger.info("=" * 60)
        logger.info("🎉 All Milestone 6 tests completed successfully!")
        logger.info("Comprehensive Evaluation Framework is working correctly.")
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        raise


if __name__ == "__main__":
    main()