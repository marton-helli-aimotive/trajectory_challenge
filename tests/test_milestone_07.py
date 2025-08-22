#!/usr/bin/env python3
"""
Test script for Milestone 7: Comprehensive Evaluation Framework
"""

import numpy as np

from src.trajectory_prediction.evaluation.safety_metrics import minimum_distance
from src.trajectory_prediction.evaluation.statistical_testing import paired_t_test


def test_safety_metrics():
    """Test safety-critical metrics."""
    print("Testing safety metrics...")

    # Test minimum distance
    traj1 = np.array([[0, 0], [1, 1], [2, 2]])
    traj2 = np.array([[0.5, 0], [1.5, 1], [2.5, 2]])
    min_dist = minimum_distance(traj1, traj2)
    assert min_dist > 0, "Minimum distance should be positive"
    print(f"  ✓ Minimum distance: {min_dist:.3f}m")


def test_statistical_testing():
    """Test statistical significance testing."""
    print("Testing statistical testing...")

    # Test paired t-test
    data1 = [0.1, 0.2, 0.3, 0.4, 0.5]
    data2 = [0.6, 0.7, 0.8, 0.9, 1.0]
    result = paired_t_test(data1, data2)
    assert "p_value" in result, "Should have p_value in result"
    print(f"  ✓ Paired t-test p-value: {result['p_value']:.4f}")


def test_comprehensive_evaluator():
    """Test comprehensive evaluation framework."""
    from src.trajectory_prediction.evaluation.metrics import ade, fde
    from src.trajectory_prediction.evaluation.safety_metrics import (
        lateral_error,
        minimum_distance,
    )

    print("Testing comprehensive evaluator...")

    # Test the evaluation utility functions directly

    # Create dummy trajectory data
    predicted_points = [(i * 1.0, i * 0.5) for i in range(10)]
    ground_truth_points = [(i * 1.1, i * 0.6) for i in range(10)]

    # Test basic metrics
    ade_score = ade(predicted_points, ground_truth_points)
    fde_score = fde(predicted_points, ground_truth_points)

    # Test safety metrics
    pred_array = np.array(predicted_points)
    gt_array = np.array(ground_truth_points)
    min_dist = minimum_distance(pred_array, gt_array)
    lat_errors = lateral_error(predicted_points, ground_truth_points)

    # Check results
    assert ade_score > 0, "ADE should be positive"
    assert fde_score > 0, "FDE should be positive"
    assert min_dist >= 0, "Minimum distance should be non-negative"
    assert len(lat_errors) == len(predicted_points), (
        "Should have lateral error for each point"
    )

    print("  ✓ Comprehensive evaluation components tested")
    print(f"  ✓ ADE: {ade_score:.3f}")
    print(f"  ✓ FDE: {fde_score:.3f}")
    print(f"  ✓ Min distance: {min_dist:.3f}")
    print(f"  ✓ Lateral errors computed: {len(lat_errors)} points")


def main():
    """Run all tests."""
    print("=" * 60)
    print("MILESTONE 7: COMPREHENSIVE EVALUATION FRAMEWORK TESTS")
    print("=" * 60)

    try:
        test_safety_metrics()
        print()
        test_statistical_testing()
        print()
        test_comprehensive_evaluator()
        print()
        print(
            "✓ All tests passed! Milestone 7 evaluation framework is working correctly."
        )

    except Exception as e:
        print(f"✗ Test failed: {e}")
        raise


if __name__ == "__main__":
    main()
