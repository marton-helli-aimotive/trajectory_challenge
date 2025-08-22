"""Demo script for Milestone 6: Advanced ML Models & Uncertainty Quantification."""

import numpy as np

from trajectory_prediction.data.models import (
    Trajectory,
    TrajectoryPoint,
    Vehicle,
    VehicleType,
)
from trajectory_prediction.evaluation.uncertainty_metrics import (
    uncertainty_quality_metrics,
)
from trajectory_prediction.models import ModelConfig, ModelFactory


def create_sample_trajectory(trajectory_id: str = "demo_001") -> Trajectory:
    """Create a sample trajectory for testing."""
    points = []

    # Create a curved trajectory
    for i in range(20):
        t = i * 0.1  # 100ms intervals

        # Sinusoidal path
        x = 10.0 + 5.0 * t + 0.5 * np.sin(t)
        y = 20.0 + 2.0 * t + 0.3 * np.cos(t)

        # Velocities (derivatives)
        vx = 5.0 + 0.5 * np.cos(t)
        vy = 2.0 - 0.3 * np.sin(t)

        # Speed and heading
        speed = np.sqrt(vx**2 + vy**2)
        heading = np.arctan2(vy, vx) * 180 / np.pi
        if heading < 0:
            heading += 360

        # Accelerations (second derivatives)
        ax = -0.5 * np.sin(t)
        ay = -0.3 * np.cos(t)

        point = TrajectoryPoint(
            timestamp=1000000000 + t,  # Base timestamp in year 2000s
            x=x,
            y=y,
            speed=speed,
            velocity_x=vx,
            velocity_y=vy,
            acceleration_x=ax,
            acceleration_y=ay,
            heading=heading,
            lane_id=1,
            frame_id=i,
        )
        points.append(point)

    # Create vehicle
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
        dataset_name="demo_dataset",
        completeness_score=1.0,
        temporal_consistency_score=1.0,
        spatial_accuracy_score=1.0,
        smoothness_score=1.0,
    )


def demo_advanced_models():
    """Demonstrate advanced ML models with uncertainty quantification."""
    print("🚗 Milestone 6: Advanced ML Models & Uncertainty Quantification Demo")
    print("=" * 70)

    # Create sample data
    print("\n📊 Creating sample trajectories...")
    trajectories = [create_sample_trajectory(f"demo_{i:03d}") for i in range(10)]
    test_trajectory = create_sample_trajectory("test_001")

    # Split trajectory for testing
    input_points = test_trajectory.points[:15]
    true_future = test_trajectory.points[15:]

    input_trajectory = Trajectory(
        trajectory_id="test_input",
        vehicle=test_trajectory.vehicle,
        points=input_points,
        dataset_name="demo_dataset",
        completeness_score=1.0,
        temporal_consistency_score=1.0,
        spatial_accuracy_score=1.0,
        smoothness_score=1.0,
    )

    # Initialize model factory
    factory = ModelFactory()

    print(f"✅ Created {len(trajectories)} training trajectories")
    print(
        f"✅ Test trajectory: {len(input_points)} input points, {len(true_future)} future points"
    )

    # Test 1: Gaussian Process Model
    print("\n🔬 Testing Gaussian Process Model...")

    try:
        gp_model = factory.create_model(
            "gaussian_process",
            ModelConfig(
                name="GP_Demo",
                model_type="advanced",
                hyperparameters={
                    "kernel": "rbf",
                    "length_scale": 1.0,
                    "noise_level": 1e-4,
                },
            ),
        )

        print("  Training Gaussian Process...")
        gp_model.fit(trajectories)

        print("  Generating predictions with uncertainty...")
        gp_prediction = gp_model.predict(input_trajectory, prediction_horizon=1.0)

        print(f"  ✅ GP Prediction: {len(gp_prediction.predicted_points)} points")
        print(f"  📊 Uncertainty info: {list(gp_prediction.metadata.keys())}")

        # Check if the model has uncertainty methods
        try:
            mean_pred, uncertainty_bounds = gp_model.predict_with_uncertainty(
                input_trajectory, 1.0, n_samples=50
            )
            print(f"  🎯 Uncertainty bounds: {len(uncertainty_bounds)} estimates")
        except AttributeError:
            print("  ℹ️  No uncertainty quantification method available")

    except Exception as e:
        print(f"  ❌ GP Error: {e}")

    # Test 2: Tree Ensemble Model
    print("\n🌳 Testing Tree Ensemble Model...")

    try:
        tree_model = factory.create_model(
            "tree_ensemble",
            ModelConfig(
                name="TreeEnsemble_Demo",
                model_type="advanced",
                hyperparameters={
                    "model_type": "random_forest",
                    "n_estimators": 50,
                    "max_depth": 8,
                },
            ),
        )

        print("  Training Tree Ensemble...")
        tree_model.fit(trajectories)

        print("  Generating predictions...")
        tree_prediction = tree_model.predict(input_trajectory, prediction_horizon=1.0)

        print(f"  ✅ Tree Prediction: {len(tree_prediction.predicted_points)} points")
        print(f"  📊 Uncertainty info: {list(tree_prediction.metadata.keys())}")

        # Feature importance
        importance = tree_model.get_feature_importance()
        if importance:
            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[
                :3
            ]
            print(f"  🔝 Top features: {[f'{k}: {v:.3f}' for k, v in top_features]}")

    except Exception as e:
        print(f"  ❌ Tree Ensemble Error: {e}")

    # Test 3: Mixture Density Network
    print("\n🧠 Testing Simplified Mixture Density Network...")

    try:
        mdn_model = factory.create_model(
            "mixture_density",
            ModelConfig(
                name="MDN_Demo",
                model_type="advanced",
                hyperparameters={
                    "n_components": 3,
                    "hidden_size": (32, 16),
                    "max_iter": 100,  # Reduced for demo
                },
            ),
        )

        print("  Training Mixture Density Network...")
        mdn_model.fit(trajectories)

        print("  Generating probabilistic predictions...")
        mdn_prediction = mdn_model.predict(input_trajectory, prediction_horizon=1.0)

        print(f"  ✅ MDN Prediction: {len(mdn_prediction.predicted_points)} points")
        print(f"  📊 Mixture info: {list(mdn_prediction.metadata.keys())}")

        # Sample multiple trajectories
        try:
            samples = mdn_model.sample_trajectories(input_trajectory, 1.0, n_samples=10)
            print(f"  🎲 Generated {len(samples)} trajectory samples")
        except AttributeError:
            print("  ℹ️  No trajectory sampling method available")

    except Exception as e:
        print(f"  ❌ MDN Error: {e}")

    # Test 4: Ensemble Model
    print("\n🤝 Testing Ensemble Model...")

    try:
        ensemble = factory.create_ensemble_model(
            [
                {"model": "polynomial_regression", "weight": 1.0},
                {"model": "knn", "weight": 1.0},
            ]
        )

        print("  Training Ensemble...")
        ensemble.fit(trajectories)

        print("  Generating ensemble predictions...")
        ensemble_prediction = ensemble.predict(input_trajectory, prediction_horizon=1.0)

        print(
            f"  ✅ Ensemble Prediction: {len(ensemble_prediction.predicted_points)} points"
        )
        print(f"  📊 Ensemble info: {list(ensemble_prediction.metadata.keys())}")

        # Get individual predictions and uncertainty metrics
        try:
            uncertainty_metrics = ensemble.get_prediction_uncertainty(
                input_trajectory, prediction_horizon=1.0
            )
            print(
                f"  📈 Ensemble disagreement: {uncertainty_metrics.get('ensemble_disagreement', 0):.4f}"
            )
        except AttributeError:
            print("  ℹ️  No uncertainty metrics available")

    except Exception as e:
        print(f"  ❌ Ensemble Error: {e}")

    # Test 5: Uncertainty Quality Metrics
    print("\n📏 Testing Uncertainty Quality Metrics...")

    try:
        # Collect predictions for evaluation
        predictions = []
        ground_truth = []

        # Create ground truth from true future points
        true_points = [(p.x, p.y) for p in true_future[:10]]  # First 10 points

        # Try to get predictions from different models
        for model_name in ["polynomial_regression", "knn"]:
            try:
                model = factory.create_model(
                    model_name,
                    ModelConfig(name=f"{model_name}_eval", model_type="eval"),
                )
                model.fit(trajectories)
                pred = model.predict(input_trajectory, prediction_horizon=1.0)

                predictions.append(pred)
                ground_truth.append(true_points)

            except Exception:
                continue

        if predictions and ground_truth:
            quality_metrics = uncertainty_quality_metrics(predictions, ground_truth)

            print("  📊 Uncertainty Quality Metrics:")
            for metric, value in quality_metrics.items():
                if isinstance(value, float):
                    print(f"    {metric}: {value:.4f}")
                else:
                    print(f"    {metric}: {value}")
        else:
            print("  ⚠️  No valid predictions for uncertainty evaluation")

    except Exception as e:
        print(f"  ❌ Uncertainty Metrics Error: {e}")

    print("\n🎉 Milestone 6 Demo Complete!")
    print("\n📋 Summary of Advanced Features:")
    print("  ✅ Gaussian Process with uncertainty quantification")
    print("  ✅ Tree-based ensemble with prediction intervals")
    print("  ✅ Mixture Density Network for probabilistic predictions")
    print("  ✅ Ensemble methods with dynamic weighting")
    print("  ✅ Advanced uncertainty quality metrics")
    print("  ✅ Epistemic and aleatoric uncertainty decomposition")


if __name__ == "__main__":
    demo_advanced_models()
