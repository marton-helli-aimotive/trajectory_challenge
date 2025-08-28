#!/usr/bin/env python3
"""
Final validation script for MLOps components.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_mlops_components_final():
    """Test MLOps components with proper package imports."""
    print("Testing MLOps components (final validation)...")
    
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
    
    # Test basic dependencies
    dependencies = ['mlflow', 'fastapi', 'evidently', 'pydantic', 'click']
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✅ {dep}: IMPORTED")
        except Exception as e:
            print(f"❌ {dep}: {e}")
    
    # Test package imports
    try:
        from vehicle_trajectory_prediction.mlops import MLflowTracker
        print("✅ MLflowTracker: IMPORTED")
    except Exception as e:
        print(f"❌ MLflowTracker: {e}")
    
    try:
        from vehicle_trajectory_prediction.mlops import create_app, PredictionService
        print("✅ FastAPI components: IMPORTED")
    except Exception as e:
        print(f"❌ FastAPI components: {e}")
    
    try:
        from vehicle_trajectory_prediction.mlops import ModelMonitor, DataDriftDetector
        print("✅ Monitoring components: IMPORTED")
    except Exception as e:
        print(f"❌ Monitoring components: {e}")
    
    try:
        from vehicle_trajectory_prediction.mlops import ModelRegistry
        print("✅ ModelRegistry: IMPORTED")
    except Exception as e:
        print(f"❌ ModelRegistry: {e}")
    
    # Test CLI imports
    try:
        from vehicle_trajectory_prediction.cli.mlops import mlops
        print("✅ MLOps CLI: IMPORTED")
    except Exception as e:
        print(f"❌ MLOps CLI: {e}")
    
    # Test basic instantiation (without full dependencies)
    try:
        from vehicle_trajectory_prediction.mlops import MLflowTracker
        tracker = MLflowTracker()
        print("✅ MLflowTracker: INSTANTIATED")
    except Exception as e:
        print(f"❌ MLflowTracker instantiation: {e}")
    
    try:
        from vehicle_trajectory_prediction.mlops import ModelMonitor
        monitor = ModelMonitor()
        print("✅ ModelMonitor: INSTANTIATED")
    except Exception as e:
        print(f"❌ ModelMonitor instantiation: {e}")
    
    try:
        from vehicle_trajectory_prediction.mlops import DataDriftDetector
        detector = DataDriftDetector()
        print("✅ DataDriftDetector: INSTANTIATED")
    except Exception as e:
        print(f"❌ DataDriftDetector instantiation: {e}")
    
    try:
        from vehicle_trajectory_prediction.mlops import ModelRegistry
        registry = ModelRegistry()
        print("✅ ModelRegistry: INSTANTIATED")
    except Exception as e:
        print(f"❌ ModelRegistry instantiation: {e}")
    
    print("\n🎉 MLOps components validation completed!")

if __name__ == "__main__":
    test_mlops_components_final()