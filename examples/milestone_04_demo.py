"""Demonstration of Milestone 4: Feature Engineering Framework

This script demonstrates the comprehensive feature engineering capabilities:
1. Feature extraction from trajectories
2. Feature store usage for caching and retrieval
3. Data augmentation techniques
4. Feature quality assessment and validation
"""

import asyncio
from pathlib import Path

import numpy as np
import pandas as pd

from trajectory_prediction.data import (
    FeatureCorrelationAnalyzer,
    FeatureQualityAssessor,
    Trajectory,
    TrajectoryPoint,
    Vehicle,
    VehicleType,
    create_default_augmentation_pipeline,
    create_default_feature_store,
)


def create_sample_trajectory() -> Trajectory:
    """Create a sample trajectory for demonstration."""
    # Generate realistic trajectory points with realistic timestamps
    import time

    current_time = time.time()  # Current timestamp
    timestamps = np.linspace(
        current_time, current_time + 10, 50
    )  # 10 seconds, 50 points

    # Simple curved path with realistic physics (in meters, not based on timestamp)
    t_relative = np.linspace(0, 10, 50)  # Use relative time for physics
    x = 20 * t_relative + 0.5 * np.sin(0.5 * t_relative)  # Slight curve
    y = 2 * t_relative + 0.1 * np.cos(0.3 * t_relative)  # Minor lateral movement

    # Calculate velocities and accelerations - simplified
    vx = np.full_like(timestamps, 20.0)  # Approximately constant velocity in x
    vy = np.full_like(timestamps, 2.0)  # Approximately constant velocity in y
    ax = np.full_like(timestamps, 0.0)  # Zero acceleration (simplified)
    ay = np.full_like(timestamps, 0.0)  # Zero acceleration (simplified)

    speeds = np.sqrt(vx**2 + vy**2)

    # Create trajectory points
    points = []
    for i in range(len(timestamps)):
        point = TrajectoryPoint(
            timestamp=timestamps[i],
            x=x[i],
            y=y[i],
            speed=speeds[i],
            velocity_x=vx[i],
            velocity_y=vy[i],
            acceleration_x=ax[i],
            acceleration_y=ay[i],
            heading=np.degrees(np.arctan2(vy[i], vx[i])) % 360,
            lane_id=1,
            frame_id=i,
        )
        points.append(point)

    vehicle = Vehicle(vehicle_id=1, vehicle_type=VehicleType.CAR, length=4.5, width=1.8)

    return Trajectory(
        trajectory_id="demo_trajectory_001",
        vehicle=vehicle,
        points=points,
        dataset_name="demo",
        metadata={"demo": True, "synthetic": True},
    )


async def demonstrate_feature_extraction():
    """Demonstrate feature extraction capabilities."""
    print("=== Milestone 4: Feature Engineering Framework Demo ===\n")

    # Create sample data
    print("1. Creating sample trajectory...")
    trajectory = create_sample_trajectory()
    print(f"   - Trajectory ID: {trajectory.trajectory_id}")
    print(f"   - Duration: {trajectory.duration:.2f} seconds")
    print(f"   - Points: {len(trajectory.points)}")
    print(f"   - Distance: {trajectory.length:.2f} meters\n")

    # Initialize feature store
    print("2. Setting up feature store...")
    cache_dir = Path("feature_cache_demo")
    feature_store = create_default_feature_store(cache_dir)
    print(f"   - Cache directory: {cache_dir}")
    print(f"   - Registered extractors: {len(feature_store.extractors)}")
    print(f"   - Available features: {len(feature_store.list_available_features())}\n")

    # Extract features
    print("3. Extracting features...")
    features = feature_store.extract_features(trajectory)
    print(f"   - Extracted {len(features)} features")

    # Display some key features
    print("   - Key features:")
    key_features = [
        "total_displacement",
        "path_length",
        "straightness",
        "speed_mean",
        "speed_std",
        "acceleration_mean",
        "curvature_mean",
        "total_duration",
        "sampling_rate_mean",
    ]

    for feature_name in key_features:
        if feature_name in features:
            value = features[feature_name]
            print(f"     * {feature_name}: {value:.4f}")
    print()

    # Test feature caching
    print("4. Testing feature caching...")
    print("   - Extracting features again (should use cache)...")
    cached_features = feature_store.extract_features(trajectory)
    print(f"   - Retrieved {len(cached_features)} features from cache")
    print(f"   - Cache working: {features == cached_features}\n")

    # Feature store statistics
    print("5. Feature store statistics:")
    stats = feature_store.get_cache_stats()
    for key, value in stats.items():
        print(f"   - {key}: {value}")
    print()

    return trajectory, features


async def demonstrate_data_augmentation(trajectory: Trajectory):
    """Demonstrate data augmentation techniques."""
    print("6. Demonstrating data augmentation...")

    # Create augmentation pipeline
    aug_pipeline = create_default_augmentation_pipeline()
    print(f"   - Augmenters: {len(aug_pipeline.augmenters)}")

    # Apply augmentation
    augmented_trajectories = aug_pipeline.augment_dataset([trajectory])
    print("   - Original trajectories: 1")
    print(f"   - Total after augmentation: {len(augmented_trajectories)}")

    # Show augmentation types
    augmentation_types = {}
    for traj in augmented_trajectories:
        aug_type = traj.metadata.get("augmentation", "original")
        augmentation_types[aug_type] = augmentation_types.get(aug_type, 0) + 1

    print("   - Augmentation breakdown:")
    for aug_type, count in augmentation_types.items():
        print(f"     * {aug_type}: {count}")
    print()

    return augmented_trajectories


async def demonstrate_feature_validation(features_dict: dict):
    """Demonstrate feature quality assessment."""
    print("7. Feature quality assessment...")

    # Convert features to DataFrame for analysis
    features_df = pd.DataFrame([features_dict])

    # Initialize quality assessor
    quality_assessor = FeatureQualityAssessor()

    # Assess feature quality
    quality_reports = quality_assessor.assess_feature_set(features_df)
    print(f"   - Assessed {len(quality_reports)} features")

    # Show some quality metrics
    print("   - Sample quality metrics:")
    sample_features = ["speed_mean", "acceleration_mean", "curvature_mean"]

    for feature_name in sample_features:
        if feature_name in quality_reports:
            report = quality_reports[feature_name]
            print(f"     * {feature_name}:")
            print(f"       - Completeness: {report.completeness:.3f}")
            print(f"       - Distribution: {report.distribution_type}")
            print(f"       - Outlier ratio: {report.outlier_ratio:.3f}")
            print(f"       - Stability: {report.stability:.3f}")

    # Correlation analysis
    print("\n8. Feature correlation analysis...")
    correlation_analyzer = FeatureCorrelationAnalyzer()
    correlation_results = correlation_analyzer.analyze_correlations(features_df)

    print(
        f"   - Analyzed correlations for {len(features_df.select_dtypes(include=[np.number]).columns)} numeric features"
    )

    if correlation_results["highly_correlated_pairs"]:
        print(
            f"   - Found {len(correlation_results['highly_correlated_pairs'])} highly correlated pairs"
        )
    else:
        print("   - No highly correlated features detected")

    if correlation_results["recommendations"]:
        print("   - Recommendations:")
        for rec in correlation_results["recommendations"][:3]:
            print(f"     * {rec}")
    print()


async def demonstrate_batch_processing():
    """Demonstrate batch feature processing."""
    print("9. Batch feature processing...")

    # Create multiple sample trajectories
    trajectories = []
    for i in range(5):
        trajectory = create_sample_trajectory()
        trajectory.trajectory_id = f"demo_trajectory_{i:03d}"
        # Add some variation
        for point in trajectory.points:
            point.x += np.random.normal(0, 0.5)
            point.y += np.random.normal(0, 0.2)
        trajectories.append(trajectory)

    print(f"   - Created {len(trajectories)} sample trajectories")

    # Initialize batch processor
    feature_store = create_default_feature_store()
    from trajectory_prediction.data.feature_store import BatchFeatureProcessor

    batch_processor = BatchFeatureProcessor(feature_store, batch_size=3)

    # Process trajectories
    features_df = batch_processor.process_trajectories(trajectories)
    print(
        f"   - Processed features: {features_df.shape[0]} rows, {features_df.shape[1]} columns"
    )

    # Show feature summary
    numeric_features = features_df.select_dtypes(include=[np.number])
    print("   - Feature statistics:")
    print(
        f"     * Mean speed: {numeric_features['speed_mean'].mean():.2f} ± {numeric_features['speed_mean'].std():.2f}"
    )
    print(
        f"     * Mean path length: {numeric_features['path_length'].mean():.2f} ± {numeric_features['path_length'].std():.2f}"
    )
    print(
        f"     * Mean duration: {numeric_features['total_duration'].mean():.2f} ± {numeric_features['total_duration'].std():.2f}"
    )

    # Export features
    output_path = Path("demo_features.parquet")
    batch_processor.export_features(trajectories, output_path)
    print(f"   - Exported features to: {output_path}")
    print()


async def load_real_data_demo():
    """Demonstrate with real NGSIM data if available."""
    print("10. Real data demonstration (skipped)...")
    print("    - Real data demo disabled for this milestone demo")
    print()


async def main():
    """Main demonstration function."""
    try:
        # Feature extraction and caching
        trajectory, features = await demonstrate_feature_extraction()

        # Data augmentation
        await demonstrate_data_augmentation(trajectory)

        # Feature validation
        await demonstrate_feature_validation(features)

        # Batch processing
        await demonstrate_batch_processing()

        # Real data demo
        await load_real_data_demo()

        print("=== Milestone 4 Demo Complete ===")
        print("\nKey capabilities demonstrated:")
        print(
            "✓ Comprehensive feature extraction (kinematic, temporal, contextual, quality)"
        )
        print("✓ Feature store with caching and retrieval")
        print(
            "✓ Data augmentation techniques (noise, temporal, spatial, interpolation)"
        )
        print("✓ Feature quality assessment and validation")
        print("✓ Batch processing and export capabilities")
        print("✓ Performance-optimized computation with caching")

    except Exception as e:
        print(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
