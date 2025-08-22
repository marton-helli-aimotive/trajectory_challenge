#!/usr/bin/env python3
"""
Test script for Milestone 7: Comprehensive Evaluation Framework
"""

import numpy as np

from src.trajectory_prediction.evaluation.comprehensive_evaluator import (
    ComprehensiveEvaluator,
)
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
    print("Testing comprehensive evaluator...")

    # Create dummy data
    predictions = [np.random.randn(10, 2) for _ in range(5)]
    ground_truth = [np.random.randn(10, 2) for _ in range(5)]

    evaluator = ComprehensiveEvaluator()
    results = evaluator.evaluate_model_comprehensive(
        predictions, ground_truth, "TestModel"
    )

    # Check results structure
    assert "basic_metrics" in results, "Should have basic metrics"
    assert "safety_metrics" in results, "Should have safety metrics"
    print("  ✓ Comprehensive evaluation completed")
    print(f"  ✓ ADE: {results['basic_metrics']['ade']:.3f}")
    print(f"  ✓ Safety score: {results['safety_metrics']['min_distance']:.3f}")


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
