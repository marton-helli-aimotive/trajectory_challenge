"""Test script for trajectory prediction models."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from vehicle_trajectory_prediction.core.models import Trajectory, TrajectoryPoint
from vehicle_trajectory_prediction.core.config import ModelConfig
from vehicle_trajectory_prediction.models import (
    ConstantVelocityPredictor,
    ConstantAccelerationPredictor,
    PolynomialRegressionPredictor,
    KNearestNeighborsPredictor
)
from vehicle_trajectory_prediction.models.evaluation import TrajectoryEvaluator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_trajectory(vehicle_id: str, start_time: datetime, duration: float = 30.0) -> Trajectory:
    """Create a sample trajectory for testing."""
    points = []
    current_time = start_time
    dt = 0.1  # 100ms intervals
    
    # Initial state
    x, y = 0.0, 0.0
    velocity = 10.0  # 10 m/s
    heading = 0.0  # East direction
    acceleration = 0.0
    
    # Generate trajectory points
    while current_time <= start_time + timedelta(seconds=duration):
        # Create trajectory point
        point = TrajectoryPoint(
            x=x, y=y, timestamp=current_time, velocity=velocity,
            acceleration=acceleration, heading=heading,
            vehicle_id=vehicle_id, lane_id="lane_1"
        )
        points.append(point)
        
        # Update state for next point
        current_time += timedelta(seconds=dt)
        
        # Simple motion model: constant velocity with slight curve
        heading += 0.01  # Slight turn
        x += velocity * np.cos(heading) * dt
        y += velocity * np.sin(heading) * dt
        
        # Add some noise
        x += np.random.normal(0, 0.1)
        y += np.random.normal(0, 0.1)
    
    # Calculate trajectory properties
    end_time = points[-1].timestamp
    total_duration = (end_time - start_time).total_seconds()
    
    # Calculate total distance
    total_distance = 0.0
    for i in range(1, len(points)):
        total_distance += points[i-1].distance_to(points[i])
    
    return Trajectory(
        vehicle_id=vehicle_id,
        points=points,
        start_time=start_time,
        end_time=end_time,
        duration=total_duration,
        total_distance=total_distance
    )


def create_test_dataset(n_trajectories: int = 20) -> list[Trajectory]:
    """Create a test dataset with multiple trajectories."""
    trajectories = []
    base_time = datetime.now()
    
    for i in range(n_trajectories):
        # Create trajectories with different start times and slight variations
        start_time = base_time + timedelta(seconds=i * 60)  # 1 minute apart
        vehicle_id = f"vehicle_{i:03d}"
        
        # Add some variation to make trajectories different
        duration = 25.0 + np.random.normal(0, 5.0)  # 25 ± 5 seconds
        trajectory = create_sample_trajectory(vehicle_id, start_time, duration)
        trajectories.append(trajectory)
    
    return trajectories


def test_baseline_models():
    """Test the baseline models (CV and CA)."""
    logger.info("Testing baseline models...")
    
    # Create configuration
    config = ModelConfig(
        prediction_horizon=10,
        prediction_frequency=0.1,
        random_state=42
    )
    
    # Create test trajectory
    test_trajectory = create_sample_trajectory("test_vehicle", datetime.now(), 20.0)
    
    # Test Constant Velocity model
    cv_model = ConstantVelocityPredictor(config)
    cv_model.train([test_trajectory])  # Minimal training for baseline models
    
    cv_prediction = cv_model.predict(test_trajectory)
    logger.info(f"CV Model prediction: {len(cv_prediction.predicted_points)} points")
    logger.info(f"CV Model first predicted position: ({cv_prediction.x_positions[0]:.2f}, {cv_prediction.y_positions[0]:.2f})")
    
    # Test Constant Acceleration model
    ca_model = ConstantAccelerationPredictor(config)
    ca_model.train([test_trajectory])
    
    ca_prediction = ca_model.predict(test_trajectory)
    logger.info(f"CA Model prediction: {len(ca_prediction.predicted_points)} points")
    logger.info(f"CA Model first predicted position: ({ca_prediction.x_positions[0]:.2f}, {ca_prediction.y_positions[0]:.2f})")
    
    return cv_model, ca_model


def test_polynomial_model():
    """Test the polynomial regression model."""
    logger.info("Testing polynomial regression model...")
    
    # Create configuration
    config = ModelConfig(
        prediction_horizon=10,
        prediction_frequency=0.1,
        random_state=42,
        polynomial_config={
            'degree': 2,
            'features': ['position', 'velocity'],
            'regularization': 'ridge',
            'alpha': 0.1
        }
    )
    
    # Create training dataset
    train_trajectories = create_test_dataset(15)
    
    # Create and train model
    poly_model = PolynomialRegressionPredictor(config)
    training_result = poly_model.train(train_trajectories)
    logger.info(f"Polynomial model training result: {training_result}")
    
    # Test prediction
    test_trajectory = create_sample_trajectory("test_vehicle", datetime.now(), 20.0)
    poly_prediction = poly_model.predict(test_trajectory)
    logger.info(f"Polynomial model prediction: {len(poly_prediction.predicted_points)} points")
    
    return poly_model


def test_knn_model():
    """Test the KNN model with DTW."""
    logger.info("Testing KNN model with DTW...")
    
    # Create configuration
    config = ModelConfig(
        prediction_horizon=10,
        prediction_frequency=0.1,
        random_state=42,
        knn_config={
            'n_neighbors': 3,
            'weights': 'distance',
            'use_dtw': True,
            'feature_window': 8
        }
    )
    
    # Create training dataset
    train_trajectories = create_test_dataset(20)
    
    # Create and train model
    knn_model = KNearestNeighborsPredictor(config)
    training_result = knn_model.train(train_trajectories)
    logger.info(f"KNN model training result: {training_result}")
    
    # Test prediction
    test_trajectory = create_sample_trajectory("test_vehicle", datetime.now(), 20.0)
    knn_prediction = knn_model.predict(test_trajectory)
    logger.info(f"KNN model prediction: {len(knn_prediction.predicted_points)} points")
    
    # Get neighbor information
    neighbor_info = knn_model.get_neighbor_info(test_trajectory)
    logger.info(f"KNN neighbor info: {neighbor_info}")
    
    return knn_model


def test_model_evaluation():
    """Test the model evaluation framework."""
    logger.info("Testing model evaluation...")
    
    # Create configuration
    config = ModelConfig(
        prediction_horizon=10,
        prediction_frequency=0.1,
        random_state=42
    )
    
    # Create test dataset
    test_trajectories = create_test_dataset(10)
    
    # Create models
    cv_model = ConstantVelocityPredictor(config)
    ca_model = ConstantAccelerationPredictor(config)
    
    # Train models (minimal training for baseline models)
    cv_model.train(test_trajectories[:5])
    ca_model.train(test_trajectories[:5])
    
    # Create evaluator
    evaluator = TrajectoryEvaluator(config)
    
    # Evaluate individual models
    cv_results = evaluator.evaluate_model(cv_model, test_trajectories[5:])
    ca_results = evaluator.evaluate_model(ca_model, test_trajectories[5:])
    
    logger.info(f"CV Model evaluation summary: {cv_results['summary']}")
    logger.info(f"CA Model evaluation summary: {ca_results['summary']}")
    
    # Compare models
    comparison_results = evaluator.compare_models(
        [cv_model, ca_model], test_trajectories[5:]
    )
    
    logger.info("Model comparison completed")
    
    return evaluator, comparison_results


def main():
    """Main test function."""
    logger.info("Starting trajectory prediction model tests...")
    
    try:
        # Test baseline models
        cv_model, ca_model = test_baseline_models()
        
        # Test polynomial model
        poly_model = test_polynomial_model()
        
        # Test KNN model
        knn_model = test_knn_model()
        
        # Test evaluation framework
        evaluator, comparison_results = test_model_evaluation()
        
        logger.info("All tests completed successfully!")
        
        # Print summary
        print("\n" + "="*50)
        print("TRAJECTORY PREDICTION MODELS - TEST SUMMARY")
        print("="*50)
        print(f"✓ Constant Velocity (CV) Model: {cv_model.model_name}")
        print(f"✓ Constant Acceleration (CA) Model: {ca_model.model_name}")
        print(f"✓ Polynomial Regression Model: {poly_model.model_name}")
        print(f"✓ K-Nearest Neighbors (KNN) Model: {knn_model.model_name}")
        print(f"✓ Model Evaluation Framework: {evaluator.__class__.__name__}")
        print("\nAll models implemented and tested successfully!")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        raise


if __name__ == "__main__":
    main()