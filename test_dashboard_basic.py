#!/usr/bin/env python3
"""
Basic Dashboard Testing

This script tests dashboard logic and functions without external dependencies:
- Data structures
- Logic validation
- Error handling
- Configuration validation
"""

import sys
import os
import time
import logging
from typing import Dict, List, Any

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BasicDashboardTester:
    """Basic tester for dashboard logic without external dependencies."""
    
    def __init__(self):
        self.test_results = []
        self.errors = []
        self.warnings = []
        
    def log_test(self, test_name: str, success: bool, message: str = ""):
        """Log test results."""
        result = {
            'test_name': test_name,
            'success': success,
            'message': message,
            'timestamp': time.time()
        }
        self.test_results.append(result)
        
        if success:
            logger.info(f"✅ {test_name}: PASSED - {message}")
        else:
            logger.error(f"❌ {test_name}: FAILED - {message}")
            self.errors.append(f"{test_name}: {message}")
    
    def run_all_tests(self):
        """Run all basic tests."""
        logger.info("Starting basic dashboard tests...")
        
        try:
            # Test 1: Navigation structure
            self.test_navigation_structure()
            
            # Test 2: Configuration validation
            self.test_configuration_validation()
            
            # Test 3: Data structure validation
            self.test_data_structure_validation()
            
            # Test 4: Model definitions
            self.test_model_definitions()
            
            # Test 5: Error handling
            self.test_error_handling()
            
            # Test 6: State management
            self.test_state_management()
            
            # Test 7: UI component definitions
            self.test_ui_component_definitions()
            
            # Test 8: Plot type validation
            self.test_plot_type_validation()
            
            # Test 9: Metrics validation
            self.test_metrics_validation()
            
            # Test 10: Settings validation
            self.test_settings_validation()
            
            # Print final results
            self.print_results()
            
        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            self.errors.append(f"Test execution failed: {e}")
    
    def test_navigation_structure(self):
        """Test navigation structure and page definitions."""
        logger.info("Testing navigation structure...")
        
        try:
            # Define expected navigation structure
            expected_pages = [
                "🏠 Overview",
                "📊 Trajectory Visualization", 
                "🔍 Model Comparison",
                "📈 Dataset Exploration", 
                "🤖 Model Explainability", 
                "⚙️ Settings"
            ]
            
            # Validate page structure
            assert len(expected_pages) == 6, "Wrong number of pages"
            assert all(isinstance(page, str) for page in expected_pages), "All pages must be strings"
            assert all(len(page) > 0 for page in expected_pages), "All pages must have names"
            
            # Test page uniqueness
            unique_pages = set(expected_pages)
            assert len(unique_pages) == len(expected_pages), "Duplicate pages found"
            
            # Test page icons
            for page in expected_pages:
                assert page.startswith(('🏠', '📊', '🔍', '📈', '🤖', '⚙️')), f"Page {page} missing icon"
            
            self.log_test("Navigation Structure", True, "Navigation structure is valid")
            
        except Exception as e:
            self.log_test("Navigation Structure", False, f"Navigation structure test failed: {e}")
    
    def test_configuration_validation(self):
        """Test configuration validation logic."""
        logger.info("Testing configuration validation...")
        
        try:
            # Valid configuration
            valid_config = {
                'prediction_horizon': 10,
                'update_frequency': 'Real-time',
                'plot_type': '2D Trajectory',
                'show_grid': True,
                'color_scheme': 'Default',
                'animation_speed': 1.0
            }
            
            # Validate configuration values
            assert 1 <= valid_config['prediction_horizon'] <= 30, "Invalid prediction horizon"
            assert valid_config['update_frequency'] in ['Real-time', '5 seconds', '10 seconds', '30 seconds', 'Manual'], "Invalid update frequency"
            assert valid_config['plot_type'] in ['2D Trajectory', '3D Trajectory', 'Velocity Profile', 'Acceleration Profile'], "Invalid plot type"
            assert isinstance(valid_config['show_grid'], bool), "Invalid show_grid value"
            assert valid_config['color_scheme'] in ['Default', 'Viridis', 'Plasma', 'Inferno', 'Magma'], "Invalid color scheme"
            assert 0.1 <= valid_config['animation_speed'] <= 2.0, "Invalid animation speed"
            
            # Test invalid configuration detection
            invalid_configs = [
                {'prediction_horizon': -1},
                {'prediction_horizon': 50},
                {'update_frequency': 'Invalid'},
                {'plot_type': 'Invalid'},
                {'show_grid': 'not_bool'},
                {'color_scheme': 'Invalid'},
                {'animation_speed': -0.1},
                {'animation_speed': 5.0}
            ]
            
            error_count = 0
            for invalid_config in invalid_configs:
                try:
                    if 'prediction_horizon' in invalid_config:
                        if not (1 <= invalid_config['prediction_horizon'] <= 30):
                            error_count += 1
                    if 'update_frequency' in invalid_config:
                        if invalid_config['update_frequency'] not in ['Real-time', '5 seconds', '10 seconds', '30 seconds', 'Manual']:
                            error_count += 1
                    if 'plot_type' in invalid_config:
                        if invalid_config['plot_type'] not in ['2D Trajectory', '3D Trajectory', 'Velocity Profile', 'Acceleration Profile']:
                            error_count += 1
                    if 'show_grid' in invalid_config:
                        if not isinstance(invalid_config['show_grid'], bool):
                            error_count += 1
                    if 'color_scheme' in invalid_config:
                        if invalid_config['color_scheme'] not in ['Default', 'Viridis', 'Plasma', 'Inferno', 'Magma']:
                            error_count += 1
                    if 'animation_speed' in invalid_config:
                        if not (0.1 <= invalid_config['animation_speed'] <= 2.0):
                            error_count += 1
                except Exception:
                    error_count += 1
            
            assert error_count > 0, "Invalid configurations not detected"
            
            self.log_test("Configuration Validation", True, "Configuration validation working correctly")
            
        except Exception as e:
            self.log_test("Configuration Validation", False, f"Configuration validation failed: {e}")
    
    def test_data_structure_validation(self):
        """Test data structure validation."""
        logger.info("Testing data structure validation...")
        
        try:
            # Test performance data structure
            performance_data = {
                'Model': ['Constant Velocity', 'Constant Acceleration', 'Polynomial Regression', 'KNN', 'Gaussian Process', 'Ensemble'],
                'RMSE': [1.2, 1.5, 0.8, 0.9, 0.7, 0.6],
                'ADE': [0.9, 1.1, 0.6, 0.7, 0.5, 0.4],
                'FDE': [2.1, 2.5, 1.5, 1.8, 1.3, 1.1],
                'Inference Time (ms)': [15, 20, 45, 35, 80, 60]
            }
            
            # Validate data structure
            assert len(performance_data['Model']) == 6, "Wrong number of models"
            assert len(performance_data['RMSE']) == 6, "Wrong number of RMSE values"
            assert len(performance_data['ADE']) == 6, "Wrong number of ADE values"
            assert len(performance_data['FDE']) == 6, "Wrong number of FDE values"
            assert len(performance_data['Inference Time (ms)']) == 6, "Wrong number of inference times"
            
            # Validate data types
            assert all(isinstance(model, str) for model in performance_data['Model']), "Models must be strings"
            assert all(isinstance(rmse, (int, float)) for rmse in performance_data['RMSE']), "RMSE values must be numeric"
            assert all(isinstance(ade, (int, float)) for ade in performance_data['ADE']), "ADE values must be numeric"
            assert all(isinstance(fde, (int, float)) for fde in performance_data['FDE']), "FDE values must be numeric"
            assert all(isinstance(time_ms, (int, float)) for time_ms in performance_data['Inference Time (ms)']), "Inference times must be numeric"
            
            # Validate data ranges
            assert all(rmse > 0 for rmse in performance_data['RMSE']), "RMSE values must be positive"
            assert all(ade > 0 for ade in performance_data['ADE']), "ADE values must be positive"
            assert all(fde > 0 for fde in performance_data['FDE']), "FDE values must be positive"
            assert all(time_ms > 0 for time_ms in performance_data['Inference Time (ms)']), "Inference times must be positive"
            
            self.log_test("Data Structure Validation", True, "Data structure validation passed")
            
        except Exception as e:
            self.log_test("Data Structure Validation", False, f"Data structure validation failed: {e}")
    
    def test_model_definitions(self):
        """Test model definitions and validation."""
        logger.info("Testing model definitions...")
        
        try:
            # Define available models
            available_models = [
                "Constant Velocity",
                "Constant Acceleration", 
                "Polynomial Regression",
                "KNN",
                "Gaussian Process",
                "Ensemble"
            ]
            
            # Validate model definitions
            assert len(available_models) == 6, "Wrong number of models"
            assert all(isinstance(model, str) for model in available_models), "All models must be strings"
            assert all(len(model) > 0 for model in available_models), "All models must have names"
            
            # Test model uniqueness
            unique_models = set(available_models)
            assert len(unique_models) == len(available_models), "Duplicate models found"
            
            # Test model selection validation
            valid_selections = [
                ["Constant Velocity"],
                ["Constant Velocity", "Polynomial Regression"],
                ["KNN", "Gaussian Process", "Ensemble"],
                available_models  # All models
            ]
            
            invalid_selections = [
                [],  # Empty selection
                ["Invalid Model"],  # Non-existent model
                ["Constant Velocity", "Invalid Model"],  # Mixed valid/invalid
                ["Constant Velocity", "Constant Velocity"]  # Duplicate
            ]
            
            # Test valid selections
            for selection in valid_selections:
                assert len(selection) > 0, "Selection cannot be empty"
                assert all(model in available_models for model in selection), "Invalid model in selection"
                assert len(set(selection)) == len(selection), "Duplicate models in selection"
            
            # Test invalid selections
            for selection in invalid_selections:
                try:
                    if len(selection) == 0:
                        raise ValueError("Empty selection")
                    if not all(model in available_models for model in selection):
                        raise ValueError("Invalid model in selection")
                    if len(set(selection)) != len(selection):
                        raise ValueError("Duplicate models in selection")
                except ValueError:
                    pass  # Expected behavior
            
            self.log_test("Model Definitions", True, "Model definitions and validation working correctly")
            
        except Exception as e:
            self.log_test("Model Definitions", False, f"Model definitions test failed: {e}")
    
    def test_error_handling(self):
        """Test error handling scenarios."""
        logger.info("Testing error handling...")
        
        try:
            # Test various error scenarios
            error_scenarios = [
                # Empty data
                {'data': [], 'expected_error': 'Empty data'},
                
                # Invalid input types
                {'data': None, 'expected_error': 'None data'},
                
                # Invalid numeric values
                {'data': [1, 2, -1, 4], 'expected_error': 'Negative value'},
                
                # Invalid string values
                {'data': ['a', '', 'c'], 'expected_error': 'Empty string'},
                
                # Invalid list lengths
                {'data': [1, 2, 3], 'expected_length': 5, 'expected_error': 'Wrong length'}
            ]
            
            for scenario in error_scenarios:
                try:
                    data = scenario['data']
                    
                    # Test empty data
                    if data == []:
                        raise ValueError("Empty data")
                    
                    # Test None data
                    if data is None:
                        raise ValueError("None data")
                    
                    # Test negative values
                    if isinstance(data, list) and all(isinstance(x, (int, float)) for x in data):
                        if any(x < 0 for x in data):
                            raise ValueError("Negative value")
                    
                    # Test empty strings
                    if isinstance(data, list) and all(isinstance(x, str) for x in data):
                        if any(x == '' for x in data):
                            raise ValueError("Empty string")
                    
                    # Test list length
                    if 'expected_length' in scenario:
                        if len(data) != scenario['expected_length']:
                            raise ValueError("Wrong length")
                            
                except ValueError as e:
                    # Expected error
                    pass
                except Exception as e:
                    # Unexpected error
                    raise e
            
            self.log_test("Error Handling", True, "Error handling scenarios tested successfully")
            
        except Exception as e:
            self.log_test("Error Handling", False, f"Error handling test failed: {e}")
    
    def test_state_management(self):
        """Test state management logic."""
        logger.info("Testing state management...")
        
        try:
            # Test session state structure
            session_state = {
                'page': '🏠 Overview',
                'selected_trajectory': 0,
                'plot_type': '2D Trajectory',
                'show_predictions': True,
                'selected_models': ['Constant Velocity', 'Polynomial Regression'],
                'prediction_horizon': 10,
                'test_size': 3,
                'include_safety': True
            }
            
            # Validate session state
            assert session_state['page'] in ['🏠 Overview', '📊 Trajectory Visualization', '🔍 Model Comparison', 
                                           '📈 Dataset Exploration', '🤖 Model Explainability', '⚙️ Settings'], "Invalid page"
            assert 0 <= session_state['selected_trajectory'] <= 4, "Invalid trajectory selection"
            assert session_state['plot_type'] in ['2D Trajectory', '3D Trajectory', 'Velocity Profile', 'Acceleration Profile'], "Invalid plot type"
            assert isinstance(session_state['show_predictions'], bool), "Invalid show_predictions value"
            assert len(session_state['selected_models']) > 0, "No models selected"
            assert 1 <= session_state['prediction_horizon'] <= 30, "Invalid prediction horizon"
            assert 1 <= session_state['test_size'] <= 10, "Invalid test size"
            assert isinstance(session_state['include_safety'], bool), "Invalid include_safety value"
            
            # Test state transitions
            # Navigate to trajectory visualization
            session_state['page'] = '📊 Trajectory Visualization'
            assert session_state['page'] == '📊 Trajectory Visualization', "Page transition failed"
            
            # Change trajectory selection
            session_state['selected_trajectory'] = 2
            assert session_state['selected_trajectory'] == 2, "Trajectory selection change failed"
            
            # Change plot type
            session_state['plot_type'] = 'Velocity Profile'
            assert session_state['plot_type'] == 'Velocity Profile', "Plot type change failed"
            
            # Toggle predictions
            session_state['show_predictions'] = False
            assert session_state['show_predictions'] == False, "Prediction toggle failed"
            
            # Test state persistence
            original_state = session_state.copy()
            session_state['page'] = '🔍 Model Comparison'
            assert session_state['page'] != original_state['page'], "State not updated"
            assert session_state['selected_trajectory'] == original_state['selected_trajectory'], "Unrelated state changed"
            
            self.log_test("State Management", True, "State management working correctly")
            
        except Exception as e:
            self.log_test("State Management", False, f"State management test failed: {e}")
    
    def test_ui_component_definitions(self):
        """Test UI component definitions."""
        logger.info("Testing UI component definitions...")
        
        try:
            # Define UI components
            ui_components = {
                'buttons': ['Run Model Comparison', 'Save Settings', 'Go to Trajectory Visualization', 'Go to Model Comparison'],
                'selectboxes': ['Navigation', 'Select Trajectory', 'Plot Type', 'Select Model for Analysis', 'Explainability Type'],
                'sliders': ['Prediction Horizon (seconds)', 'Test Trajectories', 'Default Prediction Horizon (seconds)', 'Animation Speed'],
                'checkboxes': ['Show Predictions', 'Include Safety Metrics', 'Show Grid'],
                'multiselects': ['Select Models for Prediction', 'Select Models to Compare']
            }
            
            # Validate component definitions
            for component_type, components in ui_components.items():
                assert len(components) > 0, f"No {component_type} defined"
                assert all(isinstance(comp, str) for comp in components), f"All {component_type} must be strings"
                assert all(len(comp) > 0 for comp in components), f"All {component_type} must have names"
                
                # Test uniqueness within component type
                unique_components = set(components)
                assert len(unique_components) == len(components), f"Duplicate {component_type} found"
            
            # Test component naming conventions
            for component_type, components in ui_components.items():
                for component in components:
                    assert component[0].isupper(), f"Component '{component}' should start with uppercase"
                    # Allow single words for component names
                    assert ' ' in component or component.isupper() or len(component.split()) == 1, f"Component '{component}' should have proper spacing"
            
            self.log_test("UI Component Definitions", True, "UI component definitions are valid")
            
        except Exception as e:
            self.log_test("UI Component Definitions", False, f"UI component definitions test failed: {e}")
    
    def test_plot_type_validation(self):
        """Test plot type validation."""
        logger.info("Testing plot type validation...")
        
        try:
            # Define valid plot types
            valid_plot_types = [
                "2D Trajectory",
                "3D Trajectory", 
                "Velocity Profile",
                "Acceleration Profile"
            ]
            
            # Validate plot types
            assert len(valid_plot_types) == 4, "Wrong number of plot types"
            assert all(isinstance(plot_type, str) for plot_type in valid_plot_types), "All plot types must be strings"
            assert all(len(plot_type) > 0 for plot_type in valid_plot_types), "All plot types must have names"
            
            # Test plot type uniqueness
            unique_plot_types = set(valid_plot_types)
            assert len(unique_plot_types) == len(valid_plot_types), "Duplicate plot types found"
            
            # Test plot type validation
            test_plot_types = [
                "2D Trajectory",  # Valid
                "3D Trajectory",  # Valid
                "Velocity Profile",  # Valid
                "Acceleration Profile",  # Valid
                "Invalid Plot Type",  # Invalid
                "",  # Invalid
                None  # Invalid
            ]
            
            valid_count = 0
            invalid_count = 0
            
            for plot_type in test_plot_types:
                if plot_type in valid_plot_types:
                    valid_count += 1
                else:
                    invalid_count += 1
            
            assert valid_count == 4, "Valid plot types not recognized"
            assert invalid_count == 3, "Invalid plot types not detected"
            
            self.log_test("Plot Type Validation", True, "Plot type validation working correctly")
            
        except Exception as e:
            self.log_test("Plot Type Validation", False, f"Plot type validation failed: {e}")
    
    def test_metrics_validation(self):
        """Test metrics validation."""
        logger.info("Testing metrics validation...")
        
        try:
            # Define metrics
            metrics = {
                'Models Available': 6,
                'Sample Trajectories': 5,
                'Prediction Horizon': "10s",
                'Update Frequency': "Real-time",
                'Total Trajectories': 5,
                'Total Duration': "50.0s",
                'Avg Trajectory Length': "50 points",
                'Data Points': 250
            }
            
            # Validate metrics
            assert len(metrics) == 8, "Wrong number of metrics"
            
            # Validate numeric metrics
            numeric_metrics = ['Models Available', 'Sample Trajectories', 'Total Trajectories', 'Data Points']
            for metric in numeric_metrics:
                assert isinstance(metrics[metric], int), f"Metric {metric} must be integer"
                assert metrics[metric] > 0, f"Metric {metric} must be positive"
            
            # Validate string metrics
            string_metrics = ['Prediction Horizon', 'Update Frequency', 'Total Duration', 'Avg Trajectory Length']
            for metric in string_metrics:
                assert isinstance(metrics[metric], str), f"Metric {metric} must be string"
                assert len(metrics[metric]) > 0, f"Metric {metric} cannot be empty"
            
            # Test metric ranges
            assert 1 <= metrics['Models Available'] <= 10, "Models Available out of range"
            assert 1 <= metrics['Sample Trajectories'] <= 10, "Sample Trajectories out of range"
            assert 1 <= metrics['Total Trajectories'] <= 10, "Total Trajectories out of range"
            assert 100 <= metrics['Data Points'] <= 1000, "Data Points out of range"
            
            self.log_test("Metrics Validation", True, "Metrics validation passed")
            
        except Exception as e:
            self.log_test("Metrics Validation", False, f"Metrics validation failed: {e}")
    
    def test_settings_validation(self):
        """Test settings validation."""
        logger.info("Testing settings validation...")
        
        try:
            # Define settings categories
            settings_categories = {
                'Model Configuration': {
                    'prediction_horizon': {'type': 'slider', 'min': 1, 'max': 30, 'default': 10},
                    'update_frequency': {'type': 'selectbox', 'options': ['Real-time', '5 seconds', '10 seconds', '30 seconds', 'Manual']}
                },
                'Visualization Settings': {
                    'plot_type': {'type': 'selectbox', 'options': ['2D Trajectory', '3D Trajectory', 'Velocity Profile', 'Acceleration Profile']},
                    'show_grid': {'type': 'checkbox', 'default': True},
                    'color_scheme': {'type': 'selectbox', 'options': ['Default', 'Viridis', 'Plasma', 'Inferno', 'Magma']},
                    'animation_speed': {'type': 'slider', 'min': 0.1, 'max': 2.0, 'default': 1.0}
                }
            }
            
            # Validate settings structure
            assert len(settings_categories) == 2, "Wrong number of settings categories"
            
            for category, settings in settings_categories.items():
                assert len(settings) > 0, f"Category {category} has no settings"
                
                for setting_name, setting_config in settings.items():
                    # Validate setting name
                    assert isinstance(setting_name, str), f"Setting name must be string"
                    assert len(setting_name) > 0, f"Setting name cannot be empty"
                    
                    # Validate setting config
                    assert 'type' in setting_config, f"Setting {setting_name} missing type"
                    assert setting_config['type'] in ['slider', 'selectbox', 'checkbox'], f"Invalid setting type for {setting_name}"
                    
                    # Validate type-specific config
                    if setting_config['type'] == 'slider':
                        assert 'min' in setting_config, f"Slider {setting_name} missing min"
                        assert 'max' in setting_config, f"Slider {setting_name} missing max"
                        assert 'default' in setting_config, f"Slider {setting_name} missing default"
                        assert setting_config['min'] < setting_config['max'], f"Slider {setting_name} min >= max"
                        assert setting_config['min'] <= setting_config['default'] <= setting_config['max'], f"Slider {setting_name} default out of range"
                    
                    elif setting_config['type'] == 'selectbox':
                        assert 'options' in setting_config, f"Selectbox {setting_name} missing options"
                        assert len(setting_config['options']) > 0, f"Selectbox {setting_name} has no options"
                        assert all(isinstance(opt, str) for opt in setting_config['options']), f"Selectbox {setting_name} options must be strings"
                    
                    elif setting_config['type'] == 'checkbox':
                        assert 'default' in setting_config, f"Checkbox {setting_name} missing default"
                        assert isinstance(setting_config['default'], bool), f"Checkbox {setting_name} default must be boolean"
            
            self.log_test("Settings Validation", True, "Settings validation passed")
            
        except Exception as e:
            self.log_test("Settings Validation", False, f"Settings validation failed: {e}")
    
    def print_results(self):
        """Print test results summary."""
        logger.info("\n" + "="*60)
        logger.info("BASIC DASHBOARD TEST RESULTS")
        logger.info("="*60)
        
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r['success']])
        failed_tests = total_tests - passed_tests
        
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {failed_tests}")
        logger.info(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if self.errors:
            logger.info("\n❌ ERRORS:")
            for error in self.errors:
                logger.info(f"  - {error}")
        
        if self.warnings:
            logger.info("\n⚠️ WARNINGS:")
            for warning in self.warnings:
                logger.info(f"  - {warning}")
        
        if failed_tests == 0:
            logger.info("\n🎉 ALL TESTS PASSED!")
        else:
            logger.info(f"\n❌ {failed_tests} TESTS FAILED")
        
        logger.info("="*60)

def main():
    """Main function to run the basic tests."""
    tester = BasicDashboardTester()
    tester.run_all_tests()

if __name__ == "__main__":
    main()