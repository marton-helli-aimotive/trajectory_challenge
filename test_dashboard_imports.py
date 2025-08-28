#!/usr/bin/env python3
"""
Test script to check dashboard imports and identify issues.
"""

import sys
from pathlib import Path
import traceback

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.append(str(project_root / "src"))

def test_imports():
    """Test all the imports used in the dashboard."""
    print("Testing dashboard imports...")
    
    try:
        print("✓ Importing streamlit...")
        import streamlit as st
        
        print("✓ Importing plotly...")
        import plotly.graph_objects as go
        import plotly.express as px
        
        print("✓ Importing pandas and numpy...")
        import pandas as pd
        import numpy as np
        
        print("✓ Importing core modules...")
        from vehicle_trajectory_prediction.core.config import ModelConfig
        from vehicle_trajectory_prediction.core.models import TrajectoryData
        
        print("✓ Importing models...")
        from vehicle_trajectory_prediction.models import (
            ConstantVelocityPredictor,
            ConstantAccelerationPredictor,
            PolynomialRegressionPredictor,
            KNearestNeighborsPredictor,
            GaussianProcessPredictor,
            EnsemblePredictor
        )
        
        print("✓ Importing evaluation...")
        from vehicle_trajectory_prediction.evaluation import ComprehensiveEvaluator
        
        print("✓ Importing visualization...")
        from vehicle_trajectory_prediction.visualization.plots import (
            TrajectoryPlotter,
            ModelComparisonPlotter,
            DatasetExplorer
        )
        
        print("✓ All imports successful!")
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        traceback.print_exc()
        return False

def test_dashboard_initialization():
    """Test dashboard initialization."""
    print("\nTesting dashboard initialization...")
    
    try:
        from vehicle_trajectory_prediction.visualization.dashboard import TrajectoryDashboard
        
        print("✓ Creating dashboard instance...")
        dashboard = TrajectoryDashboard()
        
        print("✓ Dashboard created successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Dashboard initialization error: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== Dashboard Import Test ===\n")
    
    imports_ok = test_imports()
    
    if imports_ok:
        init_ok = test_dashboard_initialization()
        
        if init_ok:
            print("\n✅ All tests passed! Dashboard should work correctly.")
        else:
            print("\n❌ Dashboard initialization failed.")
    else:
        print("\n❌ Import tests failed.")