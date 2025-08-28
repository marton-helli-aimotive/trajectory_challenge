#!/usr/bin/env python3
"""
Simplified validation script for Milestone 8: Production MLOps & Deployment

This script validates that all required MLOps components are properly implemented.
"""

import os
import sys
import importlib
import inspect
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def check_file_exists(filepath: str, description: str) -> bool:
    """Check if a file exists."""
    exists = Path(filepath).exists()
    status = "✅ PASSED" if exists else "❌ FAILED"
    print(f"{status} {description}: {filepath}")
    return exists


def check_module_import(module_name: str, description: str) -> bool:
    """Check if a module can be imported."""
    try:
        importlib.import_module(module_name)
        print(f"✅ PASSED {description}: {module_name}")
        return True
    except ImportError as e:
        print(f"❌ FAILED {description}: {module_name} - {e}")
        return False


def check_class_exists(module_name: str, class_name: str, description: str) -> bool:
    """Check if a class exists in a module."""
    try:
        module = importlib.import_module(module_name)
        if hasattr(module, class_name):
            print(f"✅ PASSED {description}: {module_name}.{class_name}")
            return True
        else:
            print(f"❌ FAILED {description}: {module_name}.{class_name} - Class not found")
            return False
    except ImportError as e:
        print(f"❌ FAILED {description}: {module_name}.{class_name} - {e}")
        return False


def check_method_exists(module_name: str, class_name: str, method_name: str, description: str) -> bool:
    """Check if a method exists in a class."""
    try:
        module = importlib.import_module(module_name)
        if hasattr(module, class_name):
            cls = getattr(module, class_name)
            if hasattr(cls, method_name):
                print(f"✅ PASSED {description}: {module_name}.{class_name}.{method_name}")
                return True
            else:
                print(f"❌ FAILED {description}: {module_name}.{class_name}.{method_name} - Method not found")
                return False
        else:
            print(f"❌ FAILED {description}: {module_name}.{class_name} - Class not found")
            return False
    except ImportError as e:
        print(f"❌ FAILED {description}: {module_name}.{class_name}.{method_name} - {e}")
        return False


def validate_milestone8_simple() -> Dict[str, Any]:
    """Validate Milestone 8 implementation (simplified)."""
    print("=" * 80)
    print("VALIDATING MILESTONE 8: Production MLOps & Deployment (Simplified)")
    print("=" * 80)
    
    results = {
        "file_structure": [],
        "dependencies": [],
        "mlops_components": []
    }
    
    # 1. File Structure Validation
    print("\n1. FILE STRUCTURE VALIDATION")
    print("-" * 40)
    
    required_files = [
        ("src/vehicle_trajectory_prediction/mlops/__init__.py", "MLOps module init file"),
        ("src/vehicle_trajectory_prediction/mlops/tracking.py", "MLflow tracking component"),
        ("src/vehicle_trajectory_prediction/mlops/serving.py", "FastAPI serving component"),
        ("src/vehicle_trajectory_prediction/mlops/monitoring.py", "Monitoring and drift detection"),
        ("src/vehicle_trajectory_prediction/mlops/registry.py", "Model registry component"),
        ("src/vehicle_trajectory_prediction/cli/mlops.py", "MLOps CLI commands"),
    ]
    
    for filepath, description in required_files:
        if check_file_exists(filepath, description):
            results["file_structure"].append(True)
        else:
            results["file_structure"].append(False)
    
    # 2. Dependencies Validation
    print("\n2. DEPENDENCIES VALIDATION")
    print("-" * 40)
    
    required_dependencies = [
        ("mlflow", "MLflow for experiment tracking"),
        ("fastapi", "FastAPI for model serving"),
        ("uvicorn", "Uvicorn ASGI server"),
        ("evidently", "Evidently for drift detection"),
        ("pydantic", "Pydantic for data validation"),
        ("click", "Click for CLI"),
    ]
    
    for dep, description in required_dependencies:
        if check_module_import(dep, description):
            results["dependencies"].append(True)
        else:
            results["dependencies"].append(False)
    
    # 3. MLOps Components Validation (Direct file checks)
    print("\n3. MLOPS COMPONENTS VALIDATION")
    print("-" * 40)
    
    # Check MLflowTracker class
    try:
        # Import just the tracking module
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        from vehicle_trajectory_prediction.mlops.tracking import MLflowTracker
        print("✅ PASSED MLflowTracker class import")
        results["mlops_components"].append(True)
    except Exception as e:
        print(f"❌ FAILED MLflowTracker class import: {e}")
        results["mlops_components"].append(False)
    
    # Check PredictionService class
    try:
        from vehicle_trajectory_prediction.mlops.serving import PredictionService
        print("✅ PASSED PredictionService class import")
        results["mlops_components"].append(True)
    except Exception as e:
        print(f"❌ FAILED PredictionService class import: {e}")
        results["mlops_components"].append(False)
    
    # Check ModelMonitor class
    try:
        from vehicle_trajectory_prediction.mlops.monitoring import ModelMonitor
        print("✅ PASSED ModelMonitor class import")
        results["mlops_components"].append(True)
    except Exception as e:
        print(f"❌ FAILED ModelMonitor class import: {e}")
        results["mlops_components"].append(False)
    
    # Check DataDriftDetector class
    try:
        from vehicle_trajectory_prediction.mlops.monitoring import DataDriftDetector
        print("✅ PASSED DataDriftDetector class import")
        results["mlops_components"].append(True)
    except Exception as e:
        print(f"❌ FAILED DataDriftDetector class import: {e}")
        results["mlops_components"].append(False)
    
    # Check ModelRegistry class
    try:
        from vehicle_trajectory_prediction.mlops.registry import ModelRegistry
        print("✅ PASSED ModelRegistry class import")
        results["mlops_components"].append(True)
    except Exception as e:
        print(f"❌ FAILED ModelRegistry class import: {e}")
        results["mlops_components"].append(False)
    
    # Check CLI mlops module
    try:
        from vehicle_trajectory_prediction.cli.mlops import mlops
        print("✅ PASSED MLOps CLI module import")
        results["mlops_components"].append(True)
    except Exception as e:
        print(f"❌ FAILED MLOps CLI module import: {e}")
        results["mlops_components"].append(False)
    
    # Check FastAPI app creation
    try:
        from vehicle_trajectory_prediction.mlops.serving import create_app
        app = create_app()
        if hasattr(app, 'routes'):
            print("✅ PASSED FastAPI app creation")
            results["mlops_components"].append(True)
        else:
            print("❌ FAILED FastAPI app creation: No routes found")
            results["mlops_components"].append(False)
    except Exception as e:
        print(f"❌ FAILED FastAPI app creation: {e}")
        results["mlops_components"].append(False)
    
    # Summary
    print("\n" + "=" * 80)
    print("MILESTONE 8 VALIDATION SUMMARY (Simplified)")
    print("=" * 80)
    
    total_checks = 0
    passed_checks = 0
    
    for category, checks in results.items():
        category_passed = sum(checks)
        category_total = len(checks)
        total_checks += category_total
        passed_checks += category_passed
        
        status = "✅ PASSED" if category_passed == category_total else "❌ FAILED"
        print(f"{status} {category.replace('_', ' ').title()}: {category_passed}/{category_total} checks passed")
    
    overall_status = "✅ PASSED" if passed_checks == total_checks else "❌ FAILED"
    print(f"\n{overall_status} Overall: {passed_checks}/{total_checks} checks passed")
    
    if passed_checks == total_checks:
        print("\n🎉 MILESTONE 8 IMPLEMENTATION IS COMPLETE!")
        print("All required MLOps components for Production MLOps & Deployment have been implemented.")
    else:
        print(f"\n⚠️  MILESTONE 8 IMPLEMENTATION IS INCOMPLETE!")
        print(f"Missing {total_checks - passed_checks} required components.")
    
    return {
        "overall_passed": passed_checks == total_checks,
        "total_checks": total_checks,
        "passed_checks": passed_checks,
        "results": results
    }


if __name__ == "__main__":
    validate_milestone8_simple()