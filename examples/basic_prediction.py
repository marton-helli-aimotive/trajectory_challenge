#!/usr/bin/env python3
"""
Basic Trajectory Prediction Example

This example demonstrates the fundamental usage of the Trajectory Prediction System:
1. Creating trajectory data
2. Using different prediction models
3. Visualizing results
4. Basic evaluation

Usage:
    python examples/basic_prediction.py [--model MODEL_NAME] [--horizon SECONDS]

Examples:
    python examples/basic_prediction.py --model constant_velocity --horizon 3.0
    python examples/basic_prediction.py --model constant_acceleration --horizon 5.0
"""

import asyncio
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import trajectory prediction modules
from trajectory_prediction.data.schemas import TrajectoryData, Position, Velocity
from trajectory_prediction.models.factory import ModelFactory
from trajectory_prediction.evaluation.metrics import TrajectoryMetrics
from trajectory_prediction.data.generators import SyntheticDataGenerator


def create_sample_trajectory(scenario='straight', duration=3.0, dt=0.1):
    """
    Create a sample trajectory for demonstration.
    
    Args:
        scenario: Type of trajectory ('straight', 'curved', 'accelerating')
        duration: Trajectory duration in seconds
        dt: Time step in seconds
    
    Returns:
        TrajectoryData: Sample trajectory
    """
    timestamps = np.arange(0, duration + dt, dt)
    positions = []
    velocities = []
    
    if scenario == 'straight':
        # Straight line with constant velocity
        start_pos = (0, 0)
        velocity = (12, 1.5)  # 12 m/s forward, 1.5 m/s sideways
        
        for t in timestamps:
            x = start_pos[0] + velocity[0] * t
            y = start_pos[1] + velocity[1] * t
            positions.append(Position(x=x, y=y))
            velocities.append(Velocity(vx=velocity[0], vy=velocity[1]))
    
    elif scenario == 'curved':
        # Circular arc
        radius = 25
        angular_velocity = 0.1  # radians per second
        
        for t in timestamps:
            angle = angular_velocity * t
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            
            # Tangential velocity
            vx = -radius * angular_velocity * np.sin(angle)
            vy = radius * angular_velocity * np.cos(angle)
            
            positions.append(Position(x=x, y=y))
            velocities.append(Velocity(vx=vx, vy=vy))
    
    elif scenario == 'accelerating':
        # Accelerating motion
        start_pos = (0, 0)
        initial_velocity = (8, 0)
        acceleration = (2, 0.5)  # 2 m/s² forward, 0.5 m/s² sideways
        
        for t in timestamps:
            # x = x₀ + v₀t + ½at²
            x = start_pos[0] + initial_velocity[0] * t + 0.5 * acceleration[0] * t**2
            y = start_pos[1] + initial_velocity[1] * t + 0.5 * acceleration[1] * t**2
            
            # v = v₀ + at
            vx = initial_velocity[0] + acceleration[0] * t
            vy = initial_velocity[1] + acceleration[1] * t
            
            positions.append(Position(x=x, y=y))
            velocities.append(Velocity(vx=vx, vy=vy))
    
    return TrajectoryData(
        trajectory_id=f"sample_{scenario}",
        vehicle_id="demo_vehicle",
        positions=positions,
        velocities=velocities,
        timestamps=timestamps.tolist()
    )


def plot_trajectory_with_prediction(input_trajectory, prediction, ground_truth=None, title="Trajectory Prediction"):
    """
    Plot trajectory with prediction overlay.
    
    Args:
        input_trajectory: Input trajectory data
        prediction: Predicted trajectory
        ground_truth: Optional ground truth for comparison
        title: Plot title
    """
    plt.figure(figsize=(12, 8))
    
    # Plot input trajectory
    input_x = [pos.x for pos in input_trajectory.positions]
    input_y = [pos.y for pos in input_trajectory.positions]
    plt.plot(input_x, input_y, 'bo-', linewidth=3, markersize=6, label='Input Trajectory', alpha=0.8)
    
    # Plot prediction
    pred_x = [pos.x for pos in prediction.positions]
    pred_y = [pos.y for pos in prediction.positions]
    plt.plot(pred_x, pred_y, 'ro--', linewidth=2, markersize=4, label='Prediction', alpha=0.8)
    
    # Plot ground truth if available
    if ground_truth:
        # Only plot the future part (after input trajectory)
        gt_future_x = [pos.x for pos in ground_truth.positions[len(input_trajectory.positions):]]
        gt_future_y = [pos.y for pos in ground_truth.positions[len(input_trajectory.positions):]]
        if gt_future_x:
            plt.plot(gt_future_x, gt_future_y, 'g--', linewidth=3, alpha=0.7, label='Ground Truth')
    
    # Mark start and end points
    plt.plot(input_x[0], input_y[0], 'go', markersize=10, label='Start')
    plt.plot(pred_x[-1], pred_y[-1], 'rs', markersize=8, label='Predicted End')
    
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


async def demonstrate_prediction_models(trajectory, prediction_horizon=3.0):
    """
    Demonstrate different prediction models on a trajectory.
    
    Args:
        trajectory: Input trajectory
        prediction_horizon: How far to predict (seconds)
    
    Returns:
        dict: Predictions from different models
    """
    # Available baseline models (no training required)
    models = {
        'constant_velocity': ModelFactory.create_model('constant_velocity'),
        'constant_acceleration': ModelFactory.create_model('constant_acceleration')
    }
    
    predictions = {}
    
    print(f"Making predictions with horizon: {prediction_horizon:.1f} seconds")
    print("=" * 50)
    
    for model_name, model in models.items():
        print(f"\nTesting {model_name} model...")
        
        try:
            # Make prediction
            prediction = await model.predict(
                trajectory,
                prediction_horizon=prediction_horizon,
                time_step=0.1
            )
            
            predictions[model_name] = prediction
            
            print(f"  ✓ Predicted {len(prediction.positions)} future points")
            print(f"  ✓ Final predicted position: ({prediction.positions[-1].x:.2f}, {prediction.positions[-1].y:.2f})")
            
            # Show uncertainty if available
            if hasattr(prediction, 'metadata') and 'uncertainty' in prediction.metadata:
                uncertainty = prediction.metadata['uncertainty']
                if isinstance(uncertainty, dict) and 'total_uncertainty' in uncertainty:
                    avg_uncertainty = np.mean(uncertainty['total_uncertainty'])
                    print(f"  ✓ Average uncertainty: {avg_uncertainty:.3f} m")
        
        except Exception as e:
            print(f"  ✗ Error with {model_name}: {str(e)}")
    
    return predictions


def evaluate_predictions(ground_truth, predictions, input_length):
    """
    Evaluate prediction quality using standard metrics.
    
    Args:
        ground_truth: Complete trajectory with future points
        predictions: Dictionary of model predictions
        input_length: Length of input trajectory
    
    Returns:
        dict: Evaluation results
    """
    if len(ground_truth.positions) <= input_length:
        print("⚠️  No ground truth data available for evaluation")
        return {}
    
    # Extract ground truth future trajectory
    gt_future = TrajectoryData(
        trajectory_id=ground_truth.trajectory_id,
        vehicle_id=ground_truth.vehicle_id,
        positions=ground_truth.positions[input_length:],
        velocities=ground_truth.velocities[input_length:],
        timestamps=ground_truth.timestamps[input_length:]
    )
    
    metrics_calculator = TrajectoryMetrics()
    evaluation_results = {}
    
    print("\nPrediction Evaluation Results:")
    print("=" * 50)
    
    for model_name, prediction in predictions.items():
        try:
            # Calculate standard trajectory prediction metrics
            rmse = metrics_calculator.rmse(prediction, gt_future)
            mae = metrics_calculator.mae(prediction, gt_future)
            ade = metrics_calculator.ade(prediction, gt_future)
            fde = metrics_calculator.fde(prediction, gt_future)
            
            evaluation_results[model_name] = {
                'RMSE': rmse,
                'MAE': mae,
                'ADE': ade,
                'FDE': fde
            }
            
            print(f"\n{model_name.upper()}:")
            print(f"  RMSE: {rmse:.3f} m")
            print(f"  MAE:  {mae:.3f} m") 
            print(f"  ADE:  {ade:.3f} m")
            print(f"  FDE:  {fde:.3f} m")
            
        except Exception as e:
            print(f"  ✗ Error evaluating {model_name}: {str(e)}")
    
    return evaluation_results


async def run_basic_example(model_name=None, scenario='straight', prediction_horizon=3.0):
    """
    Run the basic prediction example.
    
    Args:
        model_name: Specific model to use (None for all)
        scenario: Type of trajectory scenario
        prediction_horizon: Prediction time horizon
    """
    print("🚗 Trajectory Prediction System - Basic Example")
    print("=" * 55)
    
    # Create sample trajectory
    print(f"\n📍 Creating {scenario} trajectory scenario...")
    full_trajectory = create_sample_trajectory(scenario=scenario, duration=5.0)
    
    # Use first part as input, rest as ground truth
    input_duration = 2.0  # Use first 2 seconds as input
    input_length = int(input_duration / 0.1) + 1
    
    input_trajectory = TrajectoryData(
        trajectory_id=full_trajectory.trajectory_id,
        vehicle_id=full_trajectory.vehicle_id,
        positions=full_trajectory.positions[:input_length],
        velocities=full_trajectory.velocities[:input_length],
        timestamps=full_trajectory.timestamps[:input_length]
    )
    
    print(f"  ✓ Created trajectory with {len(full_trajectory.positions)} total points")
    print(f"  ✓ Using first {len(input_trajectory.positions)} points as input")
    print(f"  ✓ Predicting next {prediction_horizon:.1f} seconds")
    
    # Make predictions
    if model_name:
        # Use specific model
        try:
            model = ModelFactory.create_model(model_name)
            prediction = await model.predict(
                input_trajectory,
                prediction_horizon=prediction_horizon,
                time_step=0.1
            )
            predictions = {model_name: prediction}
            print(f"\n✓ Using {model_name} model")
        except Exception as e:
            print(f"\n✗ Error creating model {model_name}: {str(e)}")
            return
    else:
        # Use all available models
        predictions = await demonstrate_prediction_models(input_trajectory, prediction_horizon)
    
    if not predictions:
        print("✗ No predictions were generated")
        return
    
    # Evaluate predictions
    evaluation_results = evaluate_predictions(full_trajectory, predictions, len(input_trajectory.positions))
    
    # Visualize results
    print(f"\n📊 Visualizing results...")
    
    for model_name, prediction in predictions.items():
        plot_trajectory_with_prediction(
            input_trajectory,
            prediction, 
            ground_truth=full_trajectory,
            title=f"{model_name.replace('_', ' ').title()} - {scenario.title()} Trajectory Prediction"
        )
    
    # Summary
    print(f"\n📋 Summary:")
    print(f"  • Scenario: {scenario}")
    print(f"  • Input duration: {input_trajectory.timestamps[-1]:.1f} seconds")
    print(f"  • Prediction horizon: {prediction_horizon:.1f} seconds")
    print(f"  • Models tested: {', '.join(predictions.keys())}")
    
    if evaluation_results:
        print(f"  • Best model (lowest RMSE): {min(evaluation_results.keys(), key=lambda k: evaluation_results[k]['RMSE'])}")
    
    print("\n✅ Basic prediction example completed!")


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Basic Trajectory Prediction Example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--model',
        type=str,
        choices=['constant_velocity', 'constant_acceleration'],
        help='Specific model to use (default: use all available models)'
    )
    
    parser.add_argument(
        '--scenario',
        type=str,
        choices=['straight', 'curved', 'accelerating'],
        default='straight',
        help='Type of trajectory scenario'
    )
    
    parser.add_argument(
        '--horizon',
        type=float,
        default=3.0,
        help='Prediction horizon in seconds'
    )
    
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Skip plotting (useful for automated testing)'
    )
    
    args = parser.parse_args()
    
    # Disable plotting if requested
    if args.no_plot:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
    
    # Run the example
    try:
        asyncio.run(run_basic_example(
            model_name=args.model,
            scenario=args.scenario,
            prediction_horizon=args.horizon
        ))
    except KeyboardInterrupt:
        print("\n⚠️  Example interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running example: {str(e)}")
        raise


if __name__ == '__main__':
    main()