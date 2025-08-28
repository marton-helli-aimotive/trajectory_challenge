#!/usr/bin/env python3
"""
Validation script for Milestone 8: Production MLOps & Deployment

This script validates that all required components for Milestone 8 are properly implemented.
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


def check_cli_command_exists(command_name: str, description: str) -> bool:
    """Check if a CLI command exists."""
    try:
        from vehicle_trajectory_prediction.cli.mlops import mlops
        
        # Check if command exists in mlops group
        if hasattr(mlops, 'commands') and command_name in mlops.commands:
            print(f"✅ PASSED {description}: {command_name}")
            return True
        else:
            print(f"❌ FAILED {description}: {command_name} - Command not found")
            return False
    except ImportError as e:
        print(f"❌ FAILED {description}: {command_name} - {e}")
        return False


def validate_milestone8() -> Dict[str, Any]:
    """Validate Milestone 8 implementation."""
    print("=" * 80)
    print("VALIDATING MILESTONE 8: Production MLOps & Deployment")
    print("=" * 80)
    
    results = {
        "file_structure": [],
        "dependencies": [],
        "module_imports": [],
        "class_structure": [],
        "method_implementation": [],
        "cli_integration": [],
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
    ]
    
    for dep, description in required_dependencies:
        if check_module_import(dep, description):
            results["dependencies"].append(True)
        else:
            results["dependencies"].append(False)
    
    # 3. Module Imports Validation
    print("\n3. MODULE IMPORTS VALIDATION")
    print("-" * 40)
    
    required_modules = [
        ("vehicle_trajectory_prediction.mlops", "MLOps module"),
        ("vehicle_trajectory_prediction.mlops.tracking", "MLflow tracking module"),
        ("vehicle_trajectory_prediction.mlops.serving", "Model serving module"),
        ("vehicle_trajectory_prediction.mlops.monitoring", "Monitoring module"),
        ("vehicle_trajectory_prediction.mlops.registry", "Model registry module"),
        ("vehicle_trajectory_prediction.cli.mlops", "MLOps CLI module"),
    ]
    
    for module, description in required_modules:
        if check_module_import(module, description):
            results["module_imports"].append(True)
        else:
            results["module_imports"].append(False)
    
    # 4. Class Structure Validation
    print("\n4. CLASS STRUCTURE VALIDATION")
    print("-" * 40)
    
    required_classes = [
        ("vehicle_trajectory_prediction.mlops.tracking", "MLflowTracker", "MLflow tracking class"),
        ("vehicle_trajectory_prediction.mlops.serving", "PredictionService", "Prediction service class"),
        ("vehicle_trajectory_prediction.mlops.monitoring", "ModelMonitor", "Model monitoring class"),
        ("vehicle_trajectory_prediction.mlops.monitoring", "DataDriftDetector", "Data drift detector class"),
        ("vehicle_trajectory_prediction.mlops.registry", "ModelRegistry", "Model registry class"),
        ("vehicle_trajectory_prediction.mlops.registry", "ModelVersion", "Model version dataclass"),
        ("vehicle_trajectory_prediction.mlops.registry", "DeploymentInfo", "Deployment info dataclass"),
    ]
    
    for module, class_name, description in required_classes:
        if check_class_exists(module, class_name, description):
            results["class_structure"].append(True)
        else:
            results["class_structure"].append(False)
    
    # 5. Method Implementation Validation
    print("\n5. METHOD IMPLEMENTATION VALIDATION")
    print("-" * 40)
    
    required_methods = [
        # MLflowTracker methods
        ("vehicle_trajectory_prediction.mlops.tracking", "MLflowTracker", "start_run", "Start MLflow run"),
        ("vehicle_trajectory_prediction.mlops.tracking", "MLflowTracker", "log_parameters", "Log parameters"),
        ("vehicle_trajectory_prediction.mlops.tracking", "MLflowTracker", "log_metrics", "Log metrics"),
        ("vehicle_trajectory_prediction.mlops.tracking", "MLflowTracker", "log_model", "Log model"),
        ("vehicle_trajectory_prediction.mlops.tracking", "MLflowTracker", "get_best_model", "Get best model"),
        
        # PredictionService methods
        ("vehicle_trajectory_prediction.mlops.serving", "PredictionService", "predict_trajectory", "Predict trajectory"),
        ("vehicle_trajectory_prediction.mlops.serving", "PredictionService", "predict_batch", "Batch prediction"),
        ("vehicle_trajectory_prediction.mlops.serving", "PredictionService", "get_health", "Health check"),
        
        # ModelMonitor methods
        ("vehicle_trajectory_prediction.mlops.monitoring", "ModelMonitor", "log_prediction", "Log prediction"),
        ("vehicle_trajectory_prediction.mlops.monitoring", "ModelMonitor", "get_model_performance", "Get performance"),
        ("vehicle_trajectory_prediction.mlops.monitoring", "ModelMonitor", "check_alerts", "Check alerts"),
        
        # DataDriftDetector methods
        ("vehicle_trajectory_prediction.mlops.monitoring", "DataDriftDetector", "detect_drift", "Detect drift"),
        ("vehicle_trajectory_prediction.mlops.monitoring", "DataDriftDetector", "get_drift_summary", "Get drift summary"),
        
        # ModelRegistry methods
        ("vehicle_trajectory_prediction.mlops.registry", "ModelRegistry", "register_model", "Register model"),
        ("vehicle_trajectory_prediction.mlops.registry", "ModelRegistry", "deploy_model", "Deploy model"),
        ("vehicle_trajectory_prediction.mlops.registry", "ModelRegistry", "list_models", "List models"),
        ("vehicle_trajectory_prediction.mlops.registry", "ModelRegistry", "get_latest_version", "Get latest version"),
    ]
    
    for module, class_name, method_name, description in required_methods:
        if check_method_exists(module, class_name, method_name, description):
            results["method_implementation"].append(True)
        else:
            results["method_implementation"].append(False)
    
    # 6. CLI Integration Validation
    print("\n6. CLI INTEGRATION VALIDATION")
    print("-" * 40)
    
    required_cli_commands = [
        ("tracking", "MLflow tracking commands"),
        ("registry", "Model registry commands"),
        ("monitoring", "Model monitoring commands"),
        ("drift", "Data drift detection commands"),
        ("serve", "Model serving command"),
        ("health_check", "Health check command"),
    ]
    
    for command, description in required_cli_commands:
        if check_cli_command_exists(command, description):
            results["cli_integration"].append(True)
        else:
            results["cli_integration"].append(False)
    
    # 7. MLOps Components Validation
    print("\n7. MLOPS COMPONENTS VALIDATION")
    print("-" * 40)
    
    # Check FastAPI app creation
    try:
        from vehicle_trajectory_prediction.mlops.serving import create_app
        app = create_app()
        if hasattr(app, 'routes'):
            print("✅ PASSED FastAPI app creation: create_app()")
            results["mlops_components"].append(True)
        else:
            print("❌ FAILED FastAPI app creation: No routes found")
            results["mlops_components"].append(False)
    except Exception as e:
        print(f"❌ FAILED FastAPI app creation: {e}")
        results["mlops_components"].append(False)
    
    # Check MLflow tracker initialization
    try:
        from vehicle_trajectory_prediction.mlops.tracking import MLflowTracker
        tracker = MLflowTracker()
        if hasattr(tracker, 'client'):
            print("✅ PASSED MLflow tracker initialization")
            results["mlops_components"].append(True)
        else:
            print("❌ FAILED MLflow tracker initialization: No client found")
            results["mlops_components"].append(False)
    except Exception as e:
        print(f"❌ FAILED MLflow tracker initialization: {e}")
        results["mlops_components"].append(False)
    
    # Check Model registry initialization
    try:
        from vehicle_trajectory_prediction.mlops.registry import ModelRegistry
        registry = ModelRegistry()
        if hasattr(registry, 'registry_path'):
            print("✅ PASSED Model registry initialization")
            results["mlops_components"].append(True)
        else:
            print("❌ FAILED Model registry initialization: No registry path found")
            results["mlops_components"].append(False)
    except Exception as e:
        print(f"❌ FAILED Model registry initialization: {e}")
        results["mlops_components"].append(False)
    
    # Check Model monitor initialization
    try:
        from vehicle_trajectory_prediction.mlops.monitoring import ModelMonitor
        monitor = ModelMonitor()
        if hasattr(monitor, 'monitoring_data'):
            print("✅ PASSED Model monitor initialization")
            results["mlops_components"].append(True)
        else:
            print("❌ FAILED Model monitor initialization: No monitoring data found")
            results["mlops_components"].append(False)
    except Exception as e:
        print(f"❌ FAILED Model monitor initialization: {e}")
        results["mlops_components"].append(False)
    
    # Check Data drift detector initialization
    try:
        from vehicle_trajectory_prediction.mlops.monitoring import DataDriftDetector
        detector = DataDriftDetector()
        if hasattr(detector, 'reference_data'):
            print("✅ PASSED Data drift detector initialization")
            results["mlops_components"].append(True)
        else:
            print("❌ FAILED Data drift detector initialization: No reference data found")
            results["mlops_components"].append(False)
    except Exception as e:
        print(f"❌ FAILED Data drift detector initialization: {e}")
        results["mlops_components"].append(False)
    
    # Summary
    print("\n" + "=" * 80)
    print("MILESTONE 8 VALIDATION SUMMARY")
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
        print("All required components for Production MLOps & Deployment have been implemented.")
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
    validate_milestone8()