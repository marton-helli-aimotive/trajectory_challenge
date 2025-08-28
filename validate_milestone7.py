#!/usr/bin/env python3
"""
Validation script for Milestone 7: Interactive Dashboard & Visualization.

This script validates that all required components for the interactive dashboard
have been implemented correctly.
"""

import sys
import os
from pathlib import Path
import importlib
import inspect
from typing import List, Tuple, Dict, Any

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def check_file_exists(file_path: str) -> bool:
    """Check if a file exists."""
    return (project_root / file_path).exists()

def check_module_import(module_path: str) -> bool:
    """Check if a module can be imported."""
    try:
        importlib.import_module(module_path)
        return True
    except ImportError:
        return False

def check_class_methods(class_obj: type, required_methods: List[str]) -> List[str]:
    """Check if a class has the required methods."""
    missing_methods = []
    for method in required_methods:
        if not hasattr(class_obj, method):
            missing_methods.append(method)
    return missing_methods

def validate_milestone7() -> Dict[str, Any]:
    """Validate Milestone 7 implementation."""
    results = {
        "overall_status": "PASS",
        "checks": {},
        "errors": [],
        "warnings": []
    }
    
    print("=" * 80)
    print("MILESTONE 7: Interactive Dashboard & Visualization")
    print("=" * 80)
    
    # Check 1: Required files exist
    print("\n1. Checking required files...")
    required_files = [
        "src/vehicle_trajectory_prediction/visualization/dashboard.py",
        "src/vehicle_trajectory_prediction/visualization/plots.py",
        "src/vehicle_trajectory_prediction/visualization/__init__.py",
        "src/vehicle_trajectory_prediction/cli/dashboard.py"
    ]
    
    file_check = {"status": "PASS", "missing": []}
    for file_path in required_files:
        if check_file_exists(file_path):
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} - MISSING")
            file_check["missing"].append(file_path)
            file_check["status"] = "FAIL"
    
    results["checks"]["files"] = file_check
    if file_check["status"] == "FAIL":
        results["overall_status"] = "FAIL"
    
    # Check 2: Dependencies are available
    print("\n2. Checking dependencies...")
    required_deps = ["streamlit", "plotly", "pandas", "numpy"]
    
    dep_check = {"status": "PASS", "missing": []}
    for dep in required_deps:
        if check_module_import(dep):
            print(f"  ✅ {dep}")
        else:
            print(f"  ❌ {dep} - MISSING")
            dep_check["missing"].append(dep)
            dep_check["status"] = "FAIL"
    
    results["checks"]["dependencies"] = dep_check
    if dep_check["status"] == "FAIL":
        results["overall_status"] = "FAIL"
    
    # Check 3: Dashboard module imports
    print("\n3. Checking dashboard module imports...")
    try:
        from src.vehicle_trajectory_prediction.visualization.dashboard import TrajectoryDashboard
        print("  ✅ TrajectoryDashboard imported successfully")
        
        from src.vehicle_trajectory_prediction.visualization.plots import (
            TrajectoryPlotter, ModelComparisonPlotter, DatasetExplorer
        )
        print("  ✅ Plotting components imported successfully")
        
        import_check = {"status": "PASS"}
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        import_check = {"status": "FAIL", "error": str(e)}
        results["overall_status"] = "FAIL"
    
    results["checks"]["imports"] = import_check
    
    # Check 4: Dashboard class structure
    print("\n4. Checking dashboard class structure...")
    try:
        from src.vehicle_trajectory_prediction.visualization.dashboard import TrajectoryDashboard
        
        required_methods = [
            'run',
            '_show_overview',
            '_show_trajectory_visualization',
            '_show_model_comparison',
            '_show_dataset_exploration',
            '_show_model_explainability',
            '_show_settings'
        ]
        
        missing_methods = check_class_methods(TrajectoryDashboard, required_methods)
        
        if not missing_methods:
            print("  ✅ All required dashboard methods present")
            dashboard_check = {"status": "PASS"}
        else:
            print(f"  ❌ Missing dashboard methods: {missing_methods}")
            dashboard_check = {"status": "FAIL", "missing_methods": missing_methods}
            results["overall_status"] = "FAIL"
            
    except Exception as e:
        print(f"  ❌ Dashboard class check error: {e}")
        dashboard_check = {"status": "FAIL", "error": str(e)}
        results["overall_status"] = "FAIL"
    
    results["checks"]["dashboard_class"] = dashboard_check
    
    # Check 5: Plotting components structure
    print("\n5. Checking plotting components...")
    try:
        from src.vehicle_trajectory_prediction.visualization.plots import (
            TrajectoryPlotter, ModelComparisonPlotter, DatasetExplorer
        )
        
        # Check TrajectoryPlotter
        trajectory_methods = [
            'plot_2d_trajectory',
            'plot_3d_trajectory',
            'plot_velocity_profile',
            'plot_acceleration_profile',
            'plot_prediction_analysis',
            'plot_prediction_with_uncertainty'
        ]
        
        missing_trajectory = check_class_methods(TrajectoryPlotter, trajectory_methods)
        
        # Check ModelComparisonPlotter
        comparison_methods = [
            'plot_performance_comparison',
            'plot_safety_comparison',
            'plot_model_ranking',
            'plot_radar_chart'
        ]
        
        missing_comparison = check_class_methods(ModelComparisonPlotter, comparison_methods)
        
        # Check DatasetExplorer
        explorer_methods = [
            'plot_all_trajectories',
            'plot_trajectory_statistics',
            'plot_feature_correlations',
            'plot_data_quality_metrics'
        ]
        
        missing_explorer = check_class_methods(DatasetExplorer, explorer_methods)
        
        all_missing = missing_trajectory + missing_comparison + missing_explorer
        
        if not all_missing:
            print("  ✅ All plotting component methods present")
            plotting_check = {"status": "PASS"}
        else:
            print(f"  ❌ Missing plotting methods: {all_missing}")
            plotting_check = {
                "status": "FAIL", 
                "missing_methods": {
                    "TrajectoryPlotter": missing_trajectory,
                    "ModelComparisonPlotter": missing_comparison,
                    "DatasetExplorer": missing_explorer
                }
            }
            results["overall_status"] = "FAIL"
            
    except Exception as e:
        print(f"  ❌ Plotting components check error: {e}")
        plotting_check = {"status": "FAIL", "error": str(e)}
        results["overall_status"] = "FAIL"
    
    results["checks"]["plotting_components"] = plotting_check
    
    # Check 6: CLI integration
    print("\n6. Checking CLI integration...")
    try:
        from src.vehicle_trajectory_prediction.cli.main import dashboard
        
        # Check command signature
        sig = inspect.signature(dashboard)
        expected_params = ['host', 'port', 'browser', 'config']
        missing_params = [param for param in expected_params if param not in sig.parameters]
        
        if not missing_params:
            print("  ✅ Dashboard CLI command properly configured")
            cli_check = {"status": "PASS"}
        else:
            print(f"  ❌ Missing CLI parameters: {missing_params}")
            cli_check = {"status": "FAIL", "missing_params": missing_params}
            results["overall_status"] = "FAIL"
            
    except Exception as e:
        print(f"  ❌ CLI integration check error: {e}")
        cli_check = {"status": "FAIL", "error": str(e)}
        results["overall_status"] = "FAIL"
    
    results["checks"]["cli_integration"] = cli_check
    
    # Check 7: Dashboard initialization
    print("\n7. Checking dashboard initialization...")
    try:
        from src.vehicle_trajectory_prediction.visualization.dashboard import TrajectoryDashboard
        
        dashboard = TrajectoryDashboard()
        
        required_attrs = ['models', 'sample_trajectories', 'config', 'evaluator']
        missing_attrs = [attr for attr in required_attrs if not hasattr(dashboard, attr)]
        
        if not missing_attrs:
            print("  ✅ Dashboard initialization successful")
            init_check = {"status": "PASS"}
        else:
            print(f"  ❌ Missing dashboard attributes: {missing_attrs}")
            init_check = {"status": "FAIL", "missing_attrs": missing_attrs}
            results["overall_status"] = "FAIL"
            
    except Exception as e:
        print(f"  ❌ Dashboard initialization error: {e}")
        init_check = {"status": "FAIL", "error": str(e)}
        results["overall_status"] = "FAIL"
    
    results["checks"]["dashboard_initialization"] = init_check
    
    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    for check_name, check_result in results["checks"].items():
        status = "✅ PASS" if check_result["status"] == "PASS" else "❌ FAIL"
        print(f"{check_name.replace('_', ' ').title()}: {status}")
        
        if check_result["status"] == "FAIL" and "error" in check_result:
            print(f"  Error: {check_result['error']}")
        elif check_result["status"] == "FAIL" and "missing" in check_result:
            print(f"  Missing: {check_result['missing']}")
    
    print(f"\nOverall Status: {results['overall_status']}")
    
    if results["overall_status"] == "PASS":
        print("\n🎉 MILESTONE 7 VALIDATION PASSED!")
        print("\nThe interactive dashboard and visualization components are ready for use.")
        print("\nTo run the dashboard:")
        print("  python -m vehicle_trajectory_prediction.cli dashboard")
        print("  # or")
        print("  streamlit run src/vehicle_trajectory_prediction/visualization/dashboard.py")
    else:
        print(f"\n⚠️  MILESTONE 7 VALIDATION FAILED!")
        print("Please fix the issues above before proceeding.")
    
    return results

if __name__ == "__main__":
    results = validate_milestone7()
    sys.exit(0 if results["overall_status"] == "PASS" else 1)