#!/usr/bin/env python3
"""
Validation script for Milestone 5 models.
This script checks the syntax and structure of the implemented models.
"""

import ast
import os
import sys
from pathlib import Path

def check_python_syntax(file_path):
    """Check if a Python file has valid syntax."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        ast.parse(content)
        return True, None
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Error reading file: {e}"

def check_file_structure():
    """Check the file structure of Milestone 5 implementation."""
    base_path = Path("src/vehicle_trajectory_prediction/models")
    
    required_files = [
        "gaussian_process.py",
        "tree_ensemble.py", 
        "mixture_density.py",
        "ensemble.py",
        "test_milestone5_models.py"
    ]
    
    print("🔍 Checking Milestone 5 file structure...")
    
    all_files_exist = True
    for file_name in required_files:
        file_path = base_path / file_name
        if file_path.exists():
            print(f"  ✓ {file_name}")
        else:
            print(f"  ✗ {file_name} - MISSING")
            all_files_exist = False
    
    return all_files_exist

def check_syntax():
    """Check syntax of all Milestone 5 Python files."""
    base_path = Path("src/vehicle_trajectory_prediction/models")
    
    files_to_check = [
        "gaussian_process.py",
        "tree_ensemble.py", 
        "mixture_density.py",
        "ensemble.py",
        "test_milestone5_models.py"
    ]
    
    print("\n🔍 Checking Python syntax...")
    
    all_syntax_valid = True
    for file_name in files_to_check:
        file_path = base_path / file_name
        if file_path.exists():
            is_valid, error = check_python_syntax(file_path)
            if is_valid:
                print(f"  ✓ {file_name} - Valid syntax")
            else:
                print(f"  ✗ {file_name} - {error}")
                all_syntax_valid = False
        else:
            print(f"  ✗ {file_name} - File not found")
            all_syntax_valid = False
    
    return all_syntax_valid

def check_imports():
    """Check import structure in __init__.py."""
    init_file = Path("src/vehicle_trajectory_prediction/models/__init__.py")
    
    print("\n🔍 Checking import structure...")
    
    if not init_file.exists():
        print("  ✗ __init__.py not found")
        return False
    
    try:
        with open(init_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for required imports
        required_imports = [
            "GaussianProcessPredictor",
            "TreeEnsemblePredictor", 
            "MixtureDensityPredictor",
            "EnsemblePredictor",
            "EnsembleStrategy",
            "WeightedAverageStrategy",
            "VotingStrategy",
            "DynamicEnsembleStrategy"
        ]
        
        all_imports_found = True
        for import_name in required_imports:
            if import_name in content:
                print(f"  ✓ {import_name} imported")
            else:
                print(f"  ✗ {import_name} not imported")
                all_imports_found = False
        
        return all_imports_found
        
    except Exception as e:
        print(f"  ✗ Error reading __init__.py: {e}")
        return False

def check_class_definitions():
    """Check that required classes are defined in the files."""
    base_path = Path("src/vehicle_trajectory_prediction/models")
    
    expected_classes = {
        "gaussian_process.py": ["GaussianProcessPredictor"],
        "tree_ensemble.py": ["TreeEnsemblePredictor"],
        "mixture_density.py": ["MixtureDensityNetwork", "MixtureDensityPredictor"],
        "ensemble.py": [
            "EnsembleStrategy", 
            "WeightedAverageStrategy", 
            "VotingStrategy", 
            "DynamicEnsembleStrategy",
            "EnsemblePredictor"
        ]
    }
    
    print("\n🔍 Checking class definitions...")
    
    all_classes_found = True
    for file_name, expected_classes_list in expected_classes.items():
        file_path = base_path / file_name
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for class_name in expected_classes_list:
                    if f"class {class_name}" in content:
                        print(f"  ✓ {class_name} in {file_name}")
                    else:
                        print(f"  ✗ {class_name} not found in {file_name}")
                        all_classes_found = False
                        
            except Exception as e:
                print(f"  ✗ Error reading {file_name}: {e}")
                all_classes_found = False
        else:
            print(f"  ✗ {file_name} not found")
            all_classes_found = False
    
    return all_classes_found

def main():
    """Main validation function."""
    print("🚀 Milestone 5 Validation Report")
    print("=" * 50)
    
    # Check file structure
    structure_ok = check_file_structure()
    
    # Check syntax
    syntax_ok = check_syntax()
    
    # Check imports
    imports_ok = check_imports()
    
    # Check class definitions
    classes_ok = check_class_definitions()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Validation Summary")
    print("=" * 50)
    
    print(f"File Structure: {'✓ PASS' if structure_ok else '✗ FAIL'}")
    print(f"Python Syntax:  {'✓ PASS' if syntax_ok else '✗ FAIL'}")
    print(f"Import Structure: {'✓ PASS' if imports_ok else '✗ FAIL'}")
    print(f"Class Definitions: {'✓ PASS' if classes_ok else '✗ FAIL'}")
    
    overall_success = structure_ok and syntax_ok and imports_ok and classes_ok
    
    print(f"\nOverall Status: {'🎉 ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")
    
    if overall_success:
        print("\n✅ Milestone 5 implementation is structurally sound!")
        print("   The models are ready for testing with proper dependencies.")
    else:
        print("\n⚠️  Some issues were found. Please review the errors above.")
    
    return 0 if overall_success else 1

if __name__ == "__main__":
    sys.exit(main())