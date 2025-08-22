#!/usr/bin/env python3
"""
Milestone 7 Demo: Comprehensive Evaluation Framework

This script demonstrates the comprehensive evaluation framework with:
- Safety-critical metrics (TTC, minimum distance, lateral error)
- Statistical significance testing
- Uncertainty quantification
- Multi-model comparison and ranking
- Automated report generation
"""

import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ruff: noqa: E402
from src.trajectory_prediction.data.models import (
    Trajectory,
    TrajectoryPoint,
    Vehicle,
    VehicleType,
)
from src.trajectory_prediction.evaluation import ComprehensiveEvaluator
from src.trajectory_prediction.models.factory import ModelFactory


def create_synthetic_trajectories(n_trajectories: int = 50) -> list[Trajectory]:
    """Create synthetic trajectory data for demonstration."""
    print(f"Creating {n_trajectories} synthetic trajectories...")

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

            # Kinematic motion with noise
            x = start_x + velocity_x * t + 0.5 * acceleration_x * t**2
            y = start_y + velocity_y * t + 0.5 * acceleration_y * t**2

            # Add some noise for realism
            x += np.random.normal(0, 0.5)
            y += np.random.normal(0, 0.2)

            # Calculate instantaneous velocities
            vx = velocity_x + acceleration_x * t + np.random.normal(0, 0.5)
            vy = velocity_y + acceleration_y * t + np.random.normal(0, 0.2)

            point = TrajectoryPoint(
                timestamp=timestamp,
                x=x,
                y=y,
                speed=np.sqrt(vx**2 + vy**2),
                velocity_x=vx,
                velocity_y=vy,
                acceleration_x=acceleration_x + np.random.normal(0, 0.1),
                acceleration_y=acceleration_y + np.random.normal(0, 0.1),
                heading=np.degrees(np.arctan2(vy, vx)) % 360,
                lane_id=1,
                frame_id=t_step,
            )
            points.append(point)

        # Create vehicle and trajectory
        vehicle = Vehicle(
            vehicle_id=i,  # Use integer
            vehicle_type=VehicleType.CAR,  # Use CAR instead of PASSENGER_CAR
            length=4.5,
            width=2.0,
        )

        trajectory = Trajectory(
            trajectory_id=f"traj_{i:03d}",
            vehicle=vehicle,
            points=points,
            dataset_name="synthetic",
            completeness_score=1.0,
            temporal_consistency_score=1.0,
            spatial_accuracy_score=1.0,
            smoothness_score=1.0,
        )

        trajectories.append(trajectory)

    return trajectories


def run_milestone_07_demo():
    """Run comprehensive evaluation framework demonstration."""
    print("=" * 80)
    print("MILESTONE 7: COMPREHENSIVE EVALUATION FRAMEWORK DEMO")
    print("=" * 80)
    print()

    # Step 1: Create synthetic trajectory data
    print("Step 1: Creating synthetic trajectory data...")
    trajectories = create_synthetic_trajectories(50)
    print(f"Created {len(trajectories)} trajectories for evaluation")

    if len(trajectories) < 10:
        print("Not enough trajectories for meaningful evaluation")
        return

    # Step 2: Create and train multiple models
    print("\nStep 2: Creating and training models...")
    factory = ModelFactory()

    models_to_evaluate = {
        "Constant Velocity": factory.create_model("constant_velocity"),
        "Polynomial Regression": factory.create_model("polynomial"),
        "K-Nearest Neighbors": factory.create_model("knn"),
    }

    # Train models
    prediction_horizon = 3.0  # 3 seconds
    train_trajectories = trajectories[:40]  # Use first 40 for training
    test_trajectories = trajectories[40:]  # Use last 10 for testing

    print(f"Training {len(models_to_evaluate)} models...")
    trained_models = {}

    for model_name, model in models_to_evaluate.items():
        print(f"  Training {model_name}...")
        start_time = time.time()
        try:
            model.fit(train_trajectories)
            trained_models[model_name] = model
            training_time = time.time() - start_time
            print(f"    ✓ Trained in {training_time:.2f}s")
        except Exception as e:
            print(f"    ✗ Training failed: {e}")

    if not trained_models:
        print("No models trained successfully")
        return

    # Step 3: Create comprehensive evaluator
    print("\nStep 3: Setting up comprehensive evaluation framework...")
    evaluator = ComprehensiveEvaluator()

    # Prepare test cases
    print("  Preparing test cases...")
    test_cases = evaluator.base_evaluator.prepare_test_data(
        test_trajectories, prediction_horizon
    )
    print(f"  Created {len(test_cases)} test cases")

    if len(test_cases) < 5:
        print("Not enough test cases for evaluation")
        return

    # Step 4: Run comprehensive evaluation
    print("\nStep 4: Running comprehensive model evaluation...")
    print("  This includes:")
    print("    - Basic trajectory metrics (RMSE, ADE, FDE)")
    print("    - Safety-critical metrics (TTC, minimum distance, lateral error)")
    print("    - Multi-horizon evaluation")
    print("    - Scenario-based analysis")
    print("    - Statistical significance testing")
    print()

    comparison_results = evaluator.compare_models_comprehensive(
        trained_models, test_cases, prediction_horizon
    )

    # Step 5: Display results
    print("\nStep 5: Evaluation Results")
    print("=" * 50)

    # Generate and display report
    report = evaluator.generate_evaluation_report(comparison_results)
    print(report)

    # Save detailed results
    output_file = PROJECT_ROOT / "milestone_07_evaluation_report.txt"
    evaluator.generate_evaluation_report(comparison_results, str(output_file))

    # Step 6: Demonstrate individual model evaluation
    print("\nStep 6: Detailed Individual Model Analysis")
    print("=" * 50)

    # Pick the best model for detailed analysis
    comparison_summary = comparison_results.get("comparison_summary", {})
    best_models = comparison_summary.get("best_models", {})
    best_overall = best_models.get("overall")

    if best_overall and best_overall in trained_models:
        print(f"Detailed analysis of best overall model: {best_overall}")

        detailed_results = evaluator.evaluate_model_comprehensive(
            trained_models[best_overall],
            test_cases,
            prediction_horizon,
            include_safety=True,
            include_uncertainty=True,
        )

        # Display safety metrics
        safety_metrics = detailed_results.get("aggregated_safety", {})
        if safety_metrics:
            print("\nSafety-Critical Metrics:")
            print(
                f"  Mean Minimum Distance: {safety_metrics.get('mean_minimum_distance', 'N/A'):.3f}m"
            )
            print(
                f"  Critical Scenario Rate: {safety_metrics.get('critical_scenario_rate', 0) * 100:.1f}%"
            )
            print(
                f"  Mean Collision Probability: {safety_metrics.get('mean_collision_probability', 'N/A'):.4f}"
            )

        # Display multi-horizon results
        multi_horizon = detailed_results.get("multi_horizon", {})
        if multi_horizon:
            print("\nMulti-Horizon Performance:")
            for horizon, metrics in multi_horizon.items():
                ade_val = metrics.get("ade", float("inf"))
                if np.isfinite(ade_val):
                    print(f"  {horizon}: ADE = {ade_val:.4f}m")

        # Display scenario analysis
        scenario_analysis = detailed_results.get("scenario_analysis", {})
        if scenario_analysis:
            print("\nScenario-Based Performance:")
            for scenario, metrics in scenario_analysis.items():
                ade_val = metrics.get("ade", float("inf"))
                success_rate = metrics.get("success_rate", 0) * 100
                if np.isfinite(ade_val):
                    print(
                        f"  {scenario.replace('_', ' ').title()}: ADE = {ade_val:.4f}m (Success: {success_rate:.1f}%)"
                    )

    # Step 7: Statistical testing demonstration
    print("\nStep 7: Statistical Significance Testing")
    print("=" * 50)

    if len(trained_models) >= 2:
        model_names = list(trained_models.keys())
        model_a, model_b = model_names[0], model_names[1]

        print(f"Comparing {model_a} vs {model_b}")

        # Extract ADE results for statistical testing
        results_a = []
        results_b = []

        for input_traj, ground_truth in test_cases[:10]:  # Use subset for demo
            try:
                pred_a = trained_models[model_a].predict(input_traj, prediction_horizon)
                pred_b = trained_models[model_b].predict(input_traj, prediction_horizon)

                from src.trajectory_prediction.evaluation.metrics import ade

                ade_a = ade(pred_a.predicted_points, ground_truth)
                ade_b = ade(pred_b.predicted_points, ground_truth)

                if np.isfinite(ade_a) and np.isfinite(ade_b):
                    results_a.append(ade_a)
                    results_b.append(ade_b)
            except Exception:
                continue

        if len(results_a) >= 3:  # Need minimum for statistical test
            from src.trajectory_prediction.evaluation.statistical_testing import (
                paired_t_test,
            )

            try:
                test_result = paired_t_test(results_a, results_b)
                print("  Paired t-test results:")
                print(f"    p-value: {test_result.get('p_value', 'N/A'):.4f}")
                print(
                    f"    Significant difference: {test_result.get('significant', False)}"
                )
                print(
                    f"    Effect size (Cohen's d): {test_result.get('effect_size', 'N/A'):.3f}"
                )
            except Exception as e:
                print(f"  Statistical test failed: {e}")
        else:
            print("  Not enough data points for statistical testing")

    print("\nStep 8: Evaluation Framework Summary")
    print("=" * 50)
    print("✓ Complete suite of trajectory prediction evaluation metrics")
    print("✓ Safety-critical evaluation (collision prediction, TTC, lateral error)")
    print("✓ Multi-horizon and scenario-based performance analysis")
    print("✓ Statistical significance testing framework")
    print("✓ Automated model comparison and ranking system")
    print("✓ Comprehensive evaluation reports")

    print(f"\nDetailed evaluation report saved to: {output_file}")
    print("\nMilestone 7 comprehensive evaluation framework is complete!")


if __name__ == "__main__":
    run_milestone_07_demo()
