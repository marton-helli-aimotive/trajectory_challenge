#!/usr/bin/env python3
"""
Simple test to check dashboard functionality.
"""

import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.append(str(project_root / "src"))

def test_dashboard_import():
    """Test if dashboard can be imported."""
    try:
        from vehicle_trajectory_prediction.visualization.dashboard import TrajectoryDashboard
        print("✓ Dashboard imported successfully")
        return True
    except Exception as e:
        print(f"✗ Dashboard import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dashboard_creation():
    """Test if dashboard can be created."""
    try:
        from vehicle_trajectory_prediction.visualization.dashboard import TrajectoryDashboard
        dashboard = TrajectoryDashboard()
        print("✓ Dashboard created successfully")
        return True
    except Exception as e:
        print(f"✗ Dashboard creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dashboard_methods():
    """Test if dashboard methods work."""
    try:
        from vehicle_trajectory_prediction.visualization.dashboard import TrajectoryDashboard
        dashboard = TrajectoryDashboard()
        
        # Test if models are initialized
        print(f"✓ Models initialized: {len(dashboard.models)} models")
        
        # Test if sample data is generated
        print(f"✓ Sample data generated: {len(dashboard.sample_trajectories)} trajectories")
        
        # Test if evaluator is created
        print(f"✓ Evaluator created: {type(dashboard.evaluator).__name__}")
        
        return True
    except Exception as e:
        print(f"✗ Dashboard methods test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== Simple Dashboard Test ===\n")
    
    import_ok = test_dashboard_import()
    
    if import_ok:
        creation_ok = test_dashboard_creation()
        
        if creation_ok:
            methods_ok = test_dashboard_methods()
            
            if methods_ok:
                print("\n✅ All tests passed! Dashboard should work correctly.")
            else:
                print("\n❌ Dashboard methods test failed.")
        else:
            print("\n❌ Dashboard creation failed.")
    else:
        print("\n❌ Dashboard import failed.")