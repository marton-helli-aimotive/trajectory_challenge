#!/usr/bin/env python3
"""
Minimal test script for MLOps components.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_mlops_components_minimal():
    """Test MLOps components with minimal imports."""
    print("Testing MLOps components (minimal)...")
    
    # Test file existence
    files_to_check = [
        "src/vehicle_trajectory_prediction/mlops/__init__.py",
        "src/vehicle_trajectory_prediction/mlops/tracking.py",
        "src/vehicle_trajectory_prediction/mlops/serving.py",
        "src/vehicle_trajectory_prediction/mlops/monitoring.py",
        "src/vehicle_trajectory_prediction/mlops/registry.py",
        "src/vehicle_trajectory_prediction/cli/mlops.py"
    ]
    
    for file_path in files_to_check:
        if Path(file_path).exists():
            print(f"✅ {file_path}: EXISTS")
        else:
            print(f"❌ {file_path}: MISSING")
    
    # Test basic imports without instantiating
    try:
        import mlflow
        print("✅ mlflow: IMPORTED")
    except Exception as e:
        print(f"❌ mlflow: {e}")
    
    try:
        import fastapi
        print("✅ fastapi: IMPORTED")
    except Exception as e:
        print(f"❌ fastapi: {e}")
    
    try:
        import evidently
        print("✅ evidently: IMPORTED")
    except Exception as e:
        print(f"❌ evidently: {e}")
    
    try:
        import pydantic
        print("✅ pydantic: IMPORTED")
    except Exception as e:
        print(f"❌ pydantic: {e}")
    
    try:
        import click
        print("✅ click: IMPORTED")
    except Exception as e:
        print(f"❌ click: {e}")
    
    # Test direct file imports
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("tracking", "src/vehicle_trajectory_prediction/mlops/tracking.py")
        tracking_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tracking_module)
        print("✅ tracking.py: LOADED")
    except Exception as e:
        print(f"❌ tracking.py: {e}")
    
    try:
        spec = importlib.util.spec_from_file_location("serving", "src/vehicle_trajectory_prediction/mlops/serving.py")
        serving_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(serving_module)
        print("✅ serving.py: LOADED")
    except Exception as e:
        print(f"❌ serving.py: {e}")
    
    try:
        spec = importlib.util.spec_from_file_location("monitoring", "src/vehicle_trajectory_prediction/mlops/monitoring.py")
        monitoring_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(monitoring_module)
        print("✅ monitoring.py: LOADED")
    except Exception as e:
        print(f"❌ monitoring.py: {e}")
    
    try:
        spec = importlib.util.spec_from_file_location("registry", "src/vehicle_trajectory_prediction/mlops/registry.py")
        registry_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(registry_module)
        print("✅ registry.py: LOADED")
    except Exception as e:
        print(f"❌ registry.py: {e}")

if __name__ == "__main__":
    test_mlops_components_minimal()