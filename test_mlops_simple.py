#!/usr/bin/env python3
"""
Simple test script for MLOps components.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_mlops_components():
    """Test MLOps components individually."""
    print("Testing MLOps components...")
    
    # Test MLflowTracker
    try:
        from vehicle_trajectory_prediction.mlops.tracking import MLflowTracker
        tracker = MLflowTracker()
        print("✅ MLflowTracker: OK")
    except Exception as e:
        print(f"❌ MLflowTracker: {e}")
    
    # Test ModelMonitor
    try:
        from vehicle_trajectory_prediction.mlops.monitoring import ModelMonitor
        monitor = ModelMonitor()
        print("✅ ModelMonitor: OK")
    except Exception as e:
        print(f"❌ ModelMonitor: {e}")
    
    # Test DataDriftDetector
    try:
        from vehicle_trajectory_prediction.mlops.monitoring import DataDriftDetector
        detector = DataDriftDetector()
        print("✅ DataDriftDetector: OK")
    except Exception as e:
        print(f"❌ DataDriftDetector: {e}")
    
    # Test ModelRegistry
    try:
        from vehicle_trajectory_prediction.mlops.registry import ModelRegistry
        registry = ModelRegistry()
        print("✅ ModelRegistry: OK")
    except Exception as e:
        print(f"❌ ModelRegistry: {e}")
    
    # Test PredictionService
    try:
        from vehicle_trajectory_prediction.mlops.serving import PredictionService
        service = PredictionService()
        print("✅ PredictionService: OK")
    except Exception as e:
        print(f"❌ PredictionService: {e}")
    
    # Test FastAPI app creation
    try:
        from vehicle_trajectory_prediction.mlops.serving import create_app
        app = create_app()
        print("✅ FastAPI app creation: OK")
    except Exception as e:
        print(f"❌ FastAPI app creation: {e}")

if __name__ == "__main__":
    test_mlops_components()