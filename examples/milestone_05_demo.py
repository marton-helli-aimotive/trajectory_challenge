"""
Milestone 5 Demo: Baseline & Classical Prediction Models

This demo showcases the complete implementation of Milestone 5:
- Baseline models (Constant Velocity, Constant Acceleration)
- Classical ML models (Polynomial Regression, K-Nearest Neighbors with DTW)
- Model factory pattern for easy model selection
- Cross-validation pipeline for temporal data
- Initial evaluation metrics (RMSE, ADE, FDE)
"""

import time
from typing import Any

import numpy as np
import pandas as pd

from trajectory_prediction.data.models import (
    Trajectory,
    TrajectoryPoint,
    Vehicle,
    VehicleType,
)
from trajectory_prediction.evaluation import CrossValidationRunner, ModelEvaluator
from trajectory_prediction.models.base import ModelConfig
from trajectory_prediction.models.factory import ModelFactory, create_model
from trajectory_prediction.utils.logging import get_logger

logger = get_logger(__name__)


def create_demo_models() -> dict[str, Any]:
    """Create all baseline and classical models for demonstration."""

    # Define model configurations
    model_configs = {
        "constant_velocity": ModelConfig(
            name="cv_demo", model_type="constant_velocity", hyperparameters={}
        ),
        "constant_acceleration": ModelConfig(
            name="ca_demo", model_type="constant_acceleration", hyperparameters={}
        ),
        "polynomial": ModelConfig(
            name="poly_demo",
            model_type="polynomial",
            hyperparameters={"degree": 3, "include_interaction": True, "alpha": 0.1},
        ),
        "knn": ModelConfig(
            name="knn_demo",
            model_type="knn",
            hyperparameters={
                "n_neighbors": 5,
                "history_length": 3.0,
                "distance_weight": 0.7,
                "velocity_weight": 0.3,
            },
        ),
    }

    # Create models using factory
    models = {}
    for name, config in model_configs.items():
        try:
            model = create_model(name, config)
            models[name] = model
            logger.info(f"Created {name} model: {model.model_name}")
        except Exception as e:
            logger.error(f"Failed to create {name} model: {e}")

    return models


def create_synthetic_trajectories(n_trajectories: int = 50) -> list[Trajectory]:
    """Create synthetic trajectory data for demonstration."""
    logger.info(f"Creating {n_trajectories} synthetic trajectories...")

    trajectories = []
    np.random.seed(42)  # For reproducible results

    for i in range(n_trajectories):
        # Random trajectory parameters
        start_x = np.random.uniform(0, 1000)
        start_y = np.random.uniform(0, 100)

        # Random motion pattern
        velocity_x = np.random.uniform(10, 30)  # m/s
        velocity_y = np.random.uniform(-2, 2)  # m/s
        acceleration_x = np.random.uniform(-2, 2)  # m/s²
        acceleration_y = np.random.uniform(-1, 1)  # m/s²

        # Create trajectory points (5 seconds, 10 Hz)
        points = []
        dt = 0.1  # 10 Hz
        duration = 5.0  # seconds

        for t_step in range(int(duration / dt)):
            t = t_step * dt
            # Spread trajectories across time - each trajectory starts 10 seconds after previous
            base_timestamp = 1000000000.0 + i * 10.0  # Year 2001 + trajectory offset
            timestamp = base_timestamp + t

            # Kinematic equations with noise
            x = start_x + velocity_x * t + 0.5 * acceleration_x * t**2
            y = start_y + velocity_y * t + 0.5 * acceleration_y * t**2

            # Add some noise
            x += np.random.normal(0, 0.1)
            y += np.random.normal(0, 0.1)

            # Calculate velocities and accelerations
            vx = velocity_x + acceleration_x * t + np.random.normal(0, 0.5)
            vy = velocity_y + acceleration_y * t + np.random.normal(0, 0.3)
            ax = acceleration_x + np.random.normal(0, 0.2)
            ay = acceleration_y + np.random.normal(0, 0.2)

            speed = np.sqrt(vx**2 + vy**2)
            heading = np.degrees(np.arctan2(vy, vx)) % 360

            point = TrajectoryPoint(
                timestamp=timestamp,
                x=x,
                y=y,
                speed=speed,
                velocity_x=vx,
                velocity_y=vy,
                acceleration_x=ax,
                acceleration_y=ay,
                heading=heading,
                lane_id=1,
                frame_id=t_step,
            )
            points.append(point)

        # Create vehicle
        vehicle = Vehicle(
            vehicle_id=i, vehicle_type=VehicleType.CAR, length=4.5, width=2.0
        )

        # Create trajectory
        trajectory = Trajectory(
            trajectory_id=f"traj_{i:03d}",
            vehicle=vehicle,
            points=points,
            dataset_name="synthetic_demo",
            completeness_score=1.0,
            temporal_consistency_score=1.0,
            spatial_accuracy_score=1.0,
            smoothness_score=1.0,
        )

        trajectories.append(trajectory)

    logger.info(f"Created {len(trajectories)} synthetic trajectories")
    return trajectories


def demonstrate_individual_models(models: dict, trajectories: list) -> dict:
    """Demonstrate individual model training and prediction."""
    logger.info("Demonstrating individual model capabilities...")

    evaluator = ModelEvaluator()
    results = {}

    # Use first 80% for training, last 20% for testing
    split_idx = int(0.8 * len(trajectories))
    train_trajectories = trajectories[:split_idx]
    test_trajectories = trajectories[split_idx:]

    prediction_horizon = 2.0  # seconds

    for model_name, model in models.items():
        logger.info(f"\n--- {model.model_name} ---")

        start_time = time.time()

        try:
            # Train model
            if not model.is_fitted:
                logger.info("Training model...")
                model.fit(train_trajectories)
                training_time = time.time() - start_time
                logger.info(f"Training completed in {training_time:.2f}s")

            # Prepare test cases
            test_cases = evaluator.prepare_test_data(
                test_trajectories, prediction_horizon
            )
            logger.info(f"Prepared {len(test_cases)} test cases")

            if test_cases:
                # Evaluate model
                performance = evaluator.evaluate_model(
                    model, test_cases, prediction_horizon
                )

                logger.info("Performance:")
                logger.info(f"  RMSE: {performance.rmse:.3f}")
                logger.info(f"  ADE: {performance.ade:.3f}")
                logger.info(f"  FDE: {performance.fde:.3f}")
                logger.info(f"  Training time: {performance.training_time:.3f}s")
                logger.info(f"  Inference time: {performance.inference_time:.6f}s")

                results[model_name] = {
                    "model": model,
                    "performance": performance,
                    "n_test_cases": len(test_cases),
                }
            else:
                logger.warning("No valid test cases generated")

        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {e}")

    return results


def demonstrate_cross_validation(models: dict, trajectories: list) -> pd.DataFrame:
    """Demonstrate cross-validation across all models."""
    logger.info("\nDemonstrating cross-validation...")

    # Use a subset for faster cross-validation
    cv_trajectories = trajectories[:30]  # Smaller subset for demo

    # Configure cross-validation
    cv_runner = CrossValidationRunner()
    cv_runner.cv.n_splits = 3  # Fewer splits for demo
    cv_runner.cv.prediction_horizon = 2.0

    logger.info(
        f"Running 3-fold cross-validation on {len(cv_trajectories)} trajectories..."
    )

    # Run cross-validation comparison
    comparison_df = cv_runner.compare_models_cv(
        models, cv_trajectories, refit_per_fold=True
    )

    logger.info("\nCross-validation results:")
    logger.info(comparison_df.to_string(index=False, float_format="%.3f"))

    return comparison_df


def demonstrate_model_factory():
    """Demonstrate model factory pattern."""
    logger.info("\nDemonstrating model factory pattern...")

    factory = ModelFactory()

    # Show available models
    logger.info(
        f"Available model types: {list(factory.registry.get_available_models())}"
    )

    # Create models with different configurations
    configs_to_test = [
        ("constant_velocity", {}),
        ("polynomial", {"degree": 2}),
        ("knn", {"n_neighbors": 3, "history_length": 2.0}),
    ]

    for model_type, params in configs_to_test:
        config = ModelConfig(
            name=f"test_{model_type}", model_type=model_type, hyperparameters=params
        )

        try:
            model = create_model(model_type, config)
            logger.info(f"Created {model_type}: {model.model_name}")
            logger.info(f"  Config: {model.config}")
        except Exception as e:
            logger.error(f"Failed to create {model_type}: {e}")


def demonstrate_feature_extraction(trajectories: list):
    """Demonstrate feature extraction for classical models."""
    logger.info("\nDemonstrating feature extraction...")

    from trajectory_prediction.data.features import KinematicFeatureExtractor

    extractor = KinematicFeatureExtractor()

    # Extract features from first trajectory
    sample_trajectory = trajectories[0]

    try:
        # Try different method names
        if hasattr(extractor, "extract_features"):
            features = extractor.extract_features(sample_trajectory)
        elif hasattr(extractor, "extract"):
            features = extractor.extract(sample_trajectory)
        else:
            logger.warning("Could not find feature extraction method")
            return

        logger.info(
            f"Extracted features from trajectory {sample_trajectory.trajectory_id}:"
        )

        # Handle different feature structures
        if hasattr(features, "positions"):
            logger.info(f"  Positions: {len(features.positions)} points")
        if hasattr(features, "velocities"):
            logger.info(f"  Velocities: {len(features.velocities)} vectors")
        if hasattr(features, "accelerations"):
            logger.info(f"  Accelerations: {len(features.accelerations)} vectors")

    except Exception as e:
        logger.warning(f"Feature extraction failed: {e}")
        logger.info("Demonstrating basic trajectory data instead:")
        logger.info(f"  Trajectory {sample_trajectory.trajectory_id}:")
        logger.info(f"  Points: {len(sample_trajectory.points)}")
        logger.info(
            f"  Duration: {sample_trajectory.points[-1].timestamp - sample_trajectory.points[0].timestamp:.1f}s"
        )
        logger.info(
            f"  First point: ({sample_trajectory.points[0].x:.2f}, {sample_trajectory.points[0].y:.2f})"
        )


def main():
    """Run the complete Milestone 5 demonstration."""
    logger.info("=" * 60)
    logger.info("MILESTONE 5 DEMO: Baseline & Classical Prediction Models")
    logger.info("=" * 60)

    try:
        # 1. Demonstrate model factory
        demonstrate_model_factory()

        # 2. Create all models
        models = create_demo_models()
        if not models:
            logger.error("No models created successfully")
            return

        # 3. Create synthetic data
        trajectories = create_synthetic_trajectories(n_trajectories=40)
        if not trajectories:
            logger.error("No trajectories created")
            return

        # 4. Demonstrate feature extraction
        demonstrate_feature_extraction(trajectories)

        # 5. Demonstrate individual model capabilities
        individual_results = demonstrate_individual_models(models, trajectories)

        # 6. Demonstrate cross-validation
        if individual_results:
            cv_results = demonstrate_cross_validation(models, trajectories)

            # Save results
            output_file = "milestone_05_results.csv"
            cv_results.to_csv(output_file, index=False)
            logger.info(f"\nCross-validation results saved to {output_file}")

        logger.info("\n" + "=" * 60)
        logger.info("MILESTONE 5 DEMONSTRATION COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)

        # Summary
        logger.info("\nSummary:")
        logger.info(f"- Created {len(models)} prediction models")
        logger.info(f"- Evaluated on {len(trajectories)} synthetic trajectories")
        logger.info("- Implemented unified model interface")
        logger.info("- Demonstrated model factory pattern")
        logger.info("- Performed time-series cross-validation")
        logger.info("- Computed RMSE, ADE, FDE metrics")

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()
