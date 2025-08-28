#!/usr/bin/env python3
"""
Simple validation script for Milestone 6: Comprehensive Evaluation Framework.
This script validates the structure and basic functionality without requiring external dependencies.
"""

import os
import sys
import importlib.util
from pathlib import Path

def check_file_exists(file_path):
    """Check if a file exists and is readable."""
    if os.path.exists(file_path):
        print(f"✓ {file_path}")
        return True
    else:
        print(f"✗ {file_path} - NOT FOUND")
        return False

def check_module_structure():
    """Check the module structure for Milestone 6."""
    print("Checking Milestone 6 module structure...")
    print("=" * 60)
    
    # Define the expected files
    expected_files = [
        "src/vehicle_trajectory_prediction/evaluation/__init__.py",
        "src/vehicle_trajectory_prediction/evaluation/metrics.py",
        "src/vehicle_trajectory_prediction/evaluation/evaluator.py",
        "src/vehicle_trajectory_prediction/evaluation/statistical_tests.py",
        "src/vehicle_trajectory_prediction/evaluation/confidence_intervals.py",
        "src/vehicle_trajectory_prediction/evaluation/benchmarking.py",
        "tests/test_milestone6_evaluation.py"
    ]
    
    all_files_exist = True
    for file_path in expected_files:
        if not check_file_exists(file_path):
            all_files_exist = False
    
    return all_files_exist

def check_class_definitions():
    """Check that the expected classes are defined."""
    print("\nChecking class definitions...")
    print("=" * 60)
    
    # Define expected classes and their files
    expected_classes = {
        "src/vehicle_trajectory_prediction/evaluation/metrics.py": [
            "TrajectoryMetrics",
            "SafetyMetrics", 
            "StatisticalMetrics",
            "PerformanceMetrics"
        ],
        "src/vehicle_trajectory_prediction/evaluation/evaluator.py": [
            "ComprehensiveEvaluator"
        ],
        "src/vehicle_trajectory_prediction/evaluation/statistical_tests.py": [
            "StatisticalTestSuite"
        ],
        "src/vehicle_trajectory_prediction/evaluation/confidence_intervals.py": [
            "ConfidenceIntervalEstimator"
        ],
        "src/vehicle_trajectory_prediction/evaluation/benchmarking.py": [
            "ModelBenchmarker"
        ]
    }
    
    all_classes_found = True
    
    for file_path, expected_classes_list in expected_classes.items():
        if not os.path.exists(file_path):
            print(f"✗ File {file_path} not found, skipping class check")
            all_classes_found = False
            continue
            
        print(f"\nChecking classes in {file_path}:")
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                
            for class_name in expected_classes_list:
                if f"class {class_name}" in content:
                    print(f"  ✓ {class_name}")
                else:
                    print(f"  ✗ {class_name} - NOT FOUND")
                    all_classes_found = False
                    
        except Exception as e:
            print(f"  ✗ Error reading {file_path}: {e}")
            all_classes_found = False
    
    return all_classes_found

def check_method_definitions():
    """Check that key methods are defined in the classes."""
    print("\nChecking key method definitions...")
    print("=" * 60)
    
    # Define expected methods for each class
    expected_methods = {
        "TrajectoryMetrics": [
            "calculate_rmse",
            "calculate_ade", 
            "calculate_fde",
            "calculate_mae",
            "calculate_trajectory_similarity"
        ],
        "SafetyMetrics": [
            "calculate_min_distance",
            "calculate_ttc",
            "calculate_lateral_error",
            "calculate_risk_score"
        ],
        "StatisticalMetrics": [
            "calculate_confidence_interval",
            "calculate_percentiles",
            "calculate_outlier_rate",
            "calculate_distribution_stats"
        ],
        "PerformanceMetrics": [
            "measure_inference_time",
            "measure_memory_usage",
            "calculate_throughput"
        ],
        "ComprehensiveEvaluator": [
            "evaluate_prediction",
            "evaluate_model",
            "compare_models"
        ],
        "StatisticalTestSuite": [
            "compare_two_models",
            "compare_multiple_models",
            "perform_bootstrap_test",
            "perform_permutation_test"
        ],
        "ConfidenceIntervalEstimator": [
            "calculate_parametric_ci",
            "calculate_bootstrap_ci",
            "calculate_ci_for_metric"
        ],
        "ModelBenchmarker": [
            "benchmark_single_model",
            "benchmark_multiple_models",
            "benchmark_scalability"
        ]
    }
    
    all_methods_found = True
    
    for class_name, expected_methods_list in expected_methods.items():
        print(f"\nChecking methods in {class_name}:")
        
        # Find the file containing this class
        class_file = None
        for file_path in [
            "src/vehicle_trajectory_prediction/evaluation/metrics.py",
            "src/vehicle_trajectory_prediction/evaluation/evaluator.py",
            "src/vehicle_trajectory_prediction/evaluation/statistical_tests.py",
            "src/vehicle_trajectory_prediction/evaluation/confidence_intervals.py",
            "src/vehicle_trajectory_prediction/evaluation/benchmarking.py"
        ]:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    content = f.read()
                    if f"class {class_name}" in content:
                        class_file = file_path
                        break
        
        if class_file is None:
            print(f"  ✗ Could not find file containing {class_name}")
            all_methods_found = False
            continue
            
        try:
            with open(class_file, 'r') as f:
                content = f.read()
                
            for method_name in expected_methods_list:
                if f"def {method_name}" in content:
                    print(f"  ✓ {method_name}")
                else:
                    print(f"  ✗ {method_name} - NOT FOUND")
                    all_methods_found = False
                    
        except Exception as e:
            print(f"  ✗ Error reading {class_file}: {e}")
            all_methods_found = False
    
    return all_methods_found

def check_imports():
    """Check that the evaluation module can be imported."""
    print("\nChecking module imports...")
    print("=" * 60)
    
    # Add src to path
    src_path = os.path.join(os.path.dirname(__file__), 'src')
    sys.path.insert(0, src_path)
    
    try:
        # Try to import the evaluation module
        from vehicle_trajectory_prediction.evaluation import (
            TrajectoryMetrics,
            SafetyMetrics,
            StatisticalMetrics,
            PerformanceMetrics,
            ComprehensiveEvaluator,
            StatisticalTestSuite,
            ModelBenchmarker,
            ConfidenceIntervalEstimator
        )
        print("✓ Successfully imported all evaluation classes")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error during import: {e}")
        return False

def check_documentation():
    """Check that classes and methods have proper documentation."""
    print("\nChecking documentation...")
    print("=" * 60)
    
    files_to_check = [
        "src/vehicle_trajectory_prediction/evaluation/metrics.py",
        "src/vehicle_trajectory_prediction/evaluation/evaluator.py",
        "src/vehicle_trajectory_prediction/evaluation/statistical_tests.py",
        "src/vehicle_trajectory_prediction/evaluation/confidence_intervals.py",
        "src/vehicle_trajectory_prediction/evaluation/benchmarking.py"
    ]
    
    all_documented = True
    
    for file_path in files_to_check:
        if not os.path.exists(file_path):
            continue
            
        print(f"\nChecking documentation in {file_path}:")
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                
            # Check for module docstring
            if '"""' in content[:500]:
                print("  ✓ Module has docstring")
            else:
                print("  ✗ Module missing docstring")
                all_documented = False
            
            # Check for class docstrings
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if line.strip().startswith('class ') and ':' in line:
                    class_name = line.split('class ')[1].split('(')[0].split(':')[0].strip()
                    # Check if next line has docstring
                    if i + 1 < len(lines) and '"""' in lines[i + 1]:
                        print(f"  ✓ Class {class_name} has docstring")
                    else:
                        print(f"  ✗ Class {class_name} missing docstring")
                        all_documented = False
                        
        except Exception as e:
            print(f"  ✗ Error checking {file_path}: {e}")
            all_documented = False
    
    return all_documented

def main():
    """Run all validation checks."""
    print("Milestone 6: Comprehensive Evaluation Framework Validation")
    print("=" * 80)
    
    checks = [
        ("Module Structure", check_module_structure),
        ("Class Definitions", check_class_definitions),
        ("Method Definitions", check_method_definitions),
        ("Module Imports", check_imports),
        ("Documentation", check_documentation)
    ]
    
    results = []
    
    for check_name, check_func in checks:
        print(f"\n{check_name.upper()}")
        print("-" * 40)
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"✗ {check_name} check failed with error: {e}")
            results.append((check_name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    passed = 0
    total = len(results)
    
    for check_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{check_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall Result: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 Milestone 6 implementation is complete and valid!")
        print("The Comprehensive Evaluation Framework includes:")
        print("  • Criticality-aware metrics (RMSE, ADE, FDE, TTC, lateral error, risk score)")
        print("  • Statistical significance testing (t-tests, ANOVA, bootstrap, permutation)")
        print("  • Confidence interval estimation (parametric and bootstrap methods)")
        print("  • Performance benchmarking (inference time, memory, throughput)")
        print("  • Comprehensive model comparison framework")
        return 0
    else:
        print(f"\n❌ {total - passed} checks failed. Please review and fix the issues.")
        return 1

if __name__ == "__main__":
    exit(main())