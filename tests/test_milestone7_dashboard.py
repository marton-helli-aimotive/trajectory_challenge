"""
Test script for Milestone 7: Interactive Dashboard & Visualization.

This script validates the implementation of the interactive dashboard
and visualization components.
"""

import sys
import os
from pathlib import Path
import importlib
import inspect

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_dashboard_imports():
    """Test that all dashboard components can be imported."""
    print("Testing dashboard imports...")
    
    try:
        # Test main dashboard import
        from src.vehicle_trajectory_prediction.visualization.dashboard import TrajectoryDashboard
        print("✅ TrajectoryDashboard imported successfully")
        
        # Test plotting components
        from src.vehicle_trajectory_prediction.visualization.plots import (
            TrajectoryPlotter,
            ModelComparisonPlotter,
            DatasetExplorer
        )
        print("✅ Plotting components imported successfully")
        
        # Test visualization module
        from src.vehicle_trajectory_prediction.visualization import (
            TrajectoryDashboard,
            TrajectoryPlotter,
            ModelComparisonPlotter,
            DatasetExplorer
        )
        print("✅ Visualization module imports successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_dashboard_class_structure():
    """Test the dashboard class structure and methods."""
    print("\nTesting dashboard class structure...")
    
    try:
        from src.vehicle_trajectory_prediction.visualization.dashboard import TrajectoryDashboard
        
        # Check class exists
        assert hasattr(TrajectoryDashboard, '__init__'), "TrajectoryDashboard missing __init__"
        print("✅ TrajectoryDashboard class exists")
        
        # Check required methods
        required_methods = [
            'run',
            '_show_overview',
            '_show_trajectory_visualization',
            '_show_model_comparison',
            '_show_dataset_exploration',
            '_show_model_explainability',
            '_show_settings'
        ]
        
        for method in required_methods:
            assert hasattr(TrajectoryDashboard, method), f"Missing method: {method}"
        print("✅ All required methods present")
        
        # Check initialization
        dashboard = TrajectoryDashboard()
        assert hasattr(dashboard, 'models'), "Dashboard missing models attribute"
        assert hasattr(dashboard, 'sample_trajectories'), "Dashboard missing sample_trajectories attribute"
        print("✅ Dashboard initialization successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Dashboard class structure error: {e}")
        return False

def test_plotting_components():
    """Test the plotting components structure."""
    print("\nTesting plotting components...")
    
    try:
        from src.vehicle_trajectory_prediction.visualization.plots import (
            TrajectoryPlotter,
            ModelComparisonPlotter,
            DatasetExplorer
        )
        
        # Test TrajectoryPlotter
        plotter = TrajectoryPlotter()
        required_plotter_methods = [
            'plot_2d_trajectory',
            'plot_3d_trajectory',
            'plot_velocity_profile',
            'plot_acceleration_profile',
            'plot_prediction_analysis',
            'plot_prediction_with_uncertainty'
        ]
        
        for method in required_plotter_methods:
            assert hasattr(plotter, method), f"TrajectoryPlotter missing method: {method}"
        print("✅ TrajectoryPlotter methods present")
        
        # Test ModelComparisonPlotter
        comparison_plotter = ModelComparisonPlotter()
        required_comparison_methods = [
            'plot_performance_comparison',
            'plot_safety_comparison',
            'plot_model_ranking',
            'plot_radar_chart'
        ]
        
        for method in required_comparison_methods:
            assert hasattr(comparison_plotter, method), f"ModelComparisonPlotter missing method: {method}"
        print("✅ ModelComparisonPlotter methods present")
        
        # Test DatasetExplorer
        explorer = DatasetExplorer()
        required_explorer_methods = [
            'plot_all_trajectories',
            'plot_trajectory_statistics',
            'plot_feature_correlations',
            'plot_data_quality_metrics'
        ]
        
        for method in required_explorer_methods:
            assert hasattr(explorer, method), f"DatasetExplorer missing method: {method}"
        print("✅ DatasetExplorer methods present")
        
        return True
        
    except Exception as e:
        print(f"❌ Plotting components error: {e}")
        return False

def test_cli_integration():
    """Test CLI integration for dashboard."""
    print("\nTesting CLI integration...")
    
    try:
        from src.vehicle_trajectory_prediction.cli.main import dashboard
        
        # Check that dashboard command exists
        assert callable(dashboard), "Dashboard CLI command not callable"
        print("✅ Dashboard CLI command exists")
        
        # Check command signature
        sig = inspect.signature(dashboard)
        expected_params = ['host', 'port', 'browser', 'config']
        
        for param in expected_params:
            assert param in sig.parameters, f"Missing CLI parameter: {param}"
        print("✅ Dashboard CLI parameters correct")
        
        return True
        
    except Exception as e:
        print(f"❌ CLI integration error: {e}")
        return False

def test_dependencies():
    """Test that required dependencies are available."""
    print("\nTesting dependencies...")
    
    required_packages = [
        'streamlit',
        'plotly',
        'pandas',
        'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package} available")
        except ImportError:
            print(f"❌ {package} not available")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {missing_packages}")
        print("Install with: pip install streamlit plotly pandas numpy")
        return False
    
    return True

def test_file_structure():
    """Test that all required files exist."""
    print("\nTesting file structure...")
    
    required_files = [
        'src/vehicle_trajectory_prediction/visualization/dashboard.py',
        'src/vehicle_trajectory_prediction/visualization/plots.py',
        'src/vehicle_trajectory_prediction/visualization/__init__.py',
        'src/vehicle_trajectory_prediction/cli/dashboard.py'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\nMissing files: {missing_files}")
        return False
    
    return True

def main():
    """Run all tests for milestone 7."""
    print("=" * 60)
    print("MILESTONE 7: Interactive Dashboard & Visualization")
    print("=" * 60)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Dependencies", test_dependencies),
        ("Dashboard Imports", test_dashboard_imports),
        ("Dashboard Class Structure", test_dashboard_class_structure),
        ("Plotting Components", test_plotting_components),
        ("CLI Integration", test_cli_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 MILESTONE 7 IMPLEMENTATION VALIDATION PASSED!")
        print("\nThe interactive dashboard and visualization components are ready.")
        print("\nTo run the dashboard:")
        print("  python -m vehicle_trajectory_prediction.cli dashboard")
        print("  # or")
        print("  streamlit run src/vehicle_trajectory_prediction/visualization/dashboard.py")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please fix the issues above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())