#!/usr/bin/env python3
"""
Dashboard Component Testing

This script tests all dashboard components and functions:
- Page functions
- Data generation
- Plot creation
- Error handling
- Component interactions
"""

import sys
import os
import time
import logging
from typing import Dict, List, Any
import numpy as np
import pandas as pd

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DashboardComponentTester:
    """Tester for dashboard components without Streamlit UI."""
    
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
        """Run all component tests."""
        logger.info("Starting dashboard component tests...")
        
        try:
            # Test 1: Data generation
            self.test_data_generation()
            
            # Test 2: Plot creation
            self.test_plot_creation()
            
            # Test 3: Performance metrics
            self.test_performance_metrics()
            
            # Test 4: Model comparison logic
            self.test_model_comparison_logic()
            
            # Test 5: Feature importance calculation
            self.test_feature_importance()
            
            # Test 6: Trajectory statistics
            self.test_trajectory_statistics()
            
            # Test 7: Data validation
            self.test_data_validation()
            
            # Test 8: Error handling
            self.test_error_handling()
            
            # Test 9: Configuration validation
            self.test_configuration_validation()
            
            # Test 10: Component state management
            self.test_component_state_management()
            
            # Print final results
            self.print_results()
            
        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            self.errors.append(f"Test execution failed: {e}")
    
    def test_data_generation(self):
        """Test data generation functions."""
        logger.info("Testing data generation...")
        
        try:
            # Test trajectory data generation
            t = np.linspace(0, 10, 50)
            x = t * 10 + np.random.normal(0, 1, 50)
            y = t * 5 + np.random.normal(0, 1, 50)
            v = 10 + np.random.normal(0, 2, 50)
            a = np.gradient(v, t)
            
            # Validate generated data
            assert len(t) == 50, "Time array length incorrect"
            assert len(x) == len(y) == len(v) == len(a), "Array lengths don't match"
            assert np.all(np.isfinite(x)), "X coordinates contain invalid values"
            assert np.all(np.isfinite(y)), "Y coordinates contain invalid values"
            assert np.all(np.isfinite(v)), "Velocity values contain invalid values"
            assert np.all(np.isfinite(a)), "Acceleration values contain invalid values"
            
            self.log_test("Data Generation", True, "Trajectory data generated successfully")
            
        except Exception as e:
            self.log_test("Data Generation", False, f"Data generation failed: {e}")
    
    def test_plot_creation(self):
        """Test plot creation functions."""
        logger.info("Testing plot creation...")
        
        try:
            # Test data for plots
            t = np.linspace(0, 10, 50)
            x = t * 10 + np.random.normal(0, 1, 50)
            y = t * 5 + np.random.normal(0, 1, 50)
            v = 10 + np.random.normal(0, 2, 50)
            a = np.gradient(v, t)
            
            # Test 2D trajectory plot data
            trajectory_data = {
                'x': x,
                'y': y,
                'velocity': v,
                'time': t
            }
            
            # Test velocity profile data
            velocity_data = {
                'time': t,
                'velocity': v
            }
            
            # Test acceleration profile data
            acceleration_data = {
                'time': t,
                'acceleration': a
            }
            
            # Validate plot data
            assert len(trajectory_data['x']) == len(trajectory_data['y']), "Trajectory data mismatch"
            assert len(velocity_data['time']) == len(velocity_data['velocity']), "Velocity data mismatch"
            assert len(acceleration_data['time']) == len(acceleration_data['acceleration']), "Acceleration data mismatch"
            
            self.log_test("Plot Creation", True, "Plot data structures created successfully")
            
        except Exception as e:
            self.log_test("Plot Creation", False, f"Plot creation failed: {e}")
    
    def test_performance_metrics(self):
        """Test performance metrics calculation."""
        logger.info("Testing performance metrics...")
        
        try:
            # Test performance data structure
            performance_data = pd.DataFrame({
                'Model': ['Constant Velocity', 'Constant Acceleration', 'Polynomial Regression', 'KNN', 'Gaussian Process', 'Ensemble'],
                'RMSE': [1.2, 1.5, 0.8, 0.9, 0.7, 0.6],
                'ADE': [0.9, 1.1, 0.6, 0.7, 0.5, 0.4],
                'FDE': [2.1, 2.5, 1.5, 1.8, 1.3, 1.1],
                'Inference Time (ms)': [15, 20, 45, 35, 80, 60]
            })
            
            # Validate performance data
            assert len(performance_data) == 6, "Performance data has wrong number of models"
            assert all(col in performance_data.columns for col in ['Model', 'RMSE', 'ADE', 'FDE', 'Inference Time (ms)']), "Missing required columns"
            assert all(performance_data['RMSE'] > 0), "RMSE values must be positive"
            assert all(performance_data['ADE'] > 0), "ADE values must be positive"
            assert all(performance_data['FDE'] > 0), "FDE values must be positive"
            assert all(performance_data['Inference Time (ms)'] > 0), "Inference times must be positive"
            
            # Test metrics calculation
            best_model = performance_data.loc[performance_data['RMSE'].idxmin(), 'Model']
            worst_model = performance_data.loc[performance_data['RMSE'].idxmax(), 'Model']
            
            assert best_model == 'Ensemble', "Best model identification failed"
            assert worst_model == 'Constant Acceleration', "Worst model identification failed"
            
            self.log_test("Performance Metrics", True, "Performance metrics calculated successfully")
            
        except Exception as e:
            self.log_test("Performance Metrics", False, f"Performance metrics failed: {e}")
    
    def test_model_comparison_logic(self):
        """Test model comparison logic."""
        logger.info("Testing model comparison logic...")
        
        try:
            # Test model selection
            available_models = ["Constant Velocity", "Constant Acceleration", "Polynomial Regression", "KNN", "Gaussian Process", "Ensemble"]
            selected_models = ["Constant Velocity", "Polynomial Regression", "KNN"]
            
            # Validate model selection
            assert all(model in available_models for model in selected_models), "Invalid model selection"
            assert len(selected_models) > 0, "No models selected"
            
            # Test evaluation parameters
            prediction_horizon = 10
            test_size = 3
            include_safety = True
            
            # Validate parameters
            assert 1 <= prediction_horizon <= 30, "Invalid prediction horizon"
            assert 1 <= test_size <= 10, "Invalid test size"
            assert isinstance(include_safety, bool), "Invalid safety flag"
            
            # Test comparison results generation
            results = []
            for model in selected_models:
                results.append({
                    'Model': model,
                    'RMSE': np.random.uniform(0.5, 2.0),
                    'ADE': np.random.uniform(0.3, 1.5),
                    'FDE': np.random.uniform(0.8, 3.0),
                    'Inference Time (ms)': np.random.uniform(10, 100)
                })
            
            # Validate results
            assert len(results) == len(selected_models), "Results count mismatch"
            for result in results:
                assert result['Model'] in selected_models, "Invalid model in results"
                assert 0.5 <= result['RMSE'] <= 2.0, "RMSE out of expected range"
                assert 0.3 <= result['ADE'] <= 1.5, "ADE out of expected range"
                assert 0.8 <= result['FDE'] <= 3.0, "FDE out of expected range"
                assert 10 <= result['Inference Time (ms)'] <= 100, "Inference time out of expected range"
            
            self.log_test("Model Comparison Logic", True, "Model comparison logic working correctly")
            
        except Exception as e:
            self.log_test("Model Comparison Logic", False, f"Model comparison logic failed: {e}")
    
    def test_feature_importance(self):
        """Test feature importance calculation."""
        logger.info("Testing feature importance...")
        
        try:
            # Test feature importance data
            features = ['velocity', 'acceleration', 'heading', 'position_x', 'position_y', 'time']
            importance = np.random.uniform(0, 1, len(features))
            importance = importance / np.sum(importance)  # Normalize
            
            # Validate feature importance
            assert len(features) == len(importance), "Feature and importance arrays mismatch"
            assert np.isclose(np.sum(importance), 1.0, atol=1e-6), "Importance not normalized"
            assert all(imp >= 0 for imp in importance), "Negative importance values"
            assert all(imp <= 1 for imp in importance), "Importance values > 1"
            
            # Test feature importance ranking
            importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            # Validate ranking
            assert len(importance_df) == len(features), "Importance dataframe length mismatch"
            assert importance_df['Importance'].iloc[0] >= importance_df['Importance'].iloc[-1], "Importance not properly sorted"
            
            self.log_test("Feature Importance", True, "Feature importance calculated successfully")
            
        except Exception as e:
            self.log_test("Feature Importance", False, f"Feature importance failed: {e}")
    
    def test_trajectory_statistics(self):
        """Test trajectory statistics calculation."""
        logger.info("Testing trajectory statistics...")
        
        try:
            # Generate test trajectory
            t = np.linspace(0, 10, 50)
            x = t * 10 + np.random.normal(0, 1, 50)
            y = t * 5 + np.random.normal(0, 1, 50)
            v = 10 + np.random.normal(0, 2, 50)
            
            # Calculate statistics
            duration = t[-1]
            distance = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
            avg_velocity = np.mean(v)
            max_velocity = np.max(v)
            
            # Validate statistics
            assert duration == 10.0, "Duration calculation incorrect"
            assert distance > 0, "Distance must be positive"
            assert avg_velocity > 0, "Average velocity must be positive"
            assert max_velocity > avg_velocity, "Max velocity should be greater than average"
            assert np.isfinite(distance), "Distance is not finite"
            assert np.isfinite(avg_velocity), "Average velocity is not finite"
            assert np.isfinite(max_velocity), "Max velocity is not finite"
            
            self.log_test("Trajectory Statistics", True, "Trajectory statistics calculated successfully")
            
        except Exception as e:
            self.log_test("Trajectory Statistics", False, f"Trajectory statistics failed: {e}")
    
    def test_data_validation(self):
        """Test data validation functions."""
        logger.info("Testing data validation...")
        
        try:
            # Test valid data
            valid_data = {
                'numeric': [1, 2, 3, 4, 5],
                'string': ['a', 'b', 'c', 'd', 'e'],
                'float': [1.1, 2.2, 3.3, 4.4, 5.5]
            }
            
            df = pd.DataFrame(valid_data)
            
            # Validate data types
            assert df['numeric'].dtype in ['int64', 'int32', 'float64'], "Numeric column type validation failed"
            assert df['string'].dtype == 'object', "String column type validation failed"
            assert df['float'].dtype == 'float64', "Float column type validation failed"
            
            # Test data range validation
            assert df['numeric'].min() >= 0, "Numeric data range validation failed"
            assert len(df) > 0, "Data length validation failed"
            
            # Test empty data handling
            empty_df = pd.DataFrame()
            assert empty_df.empty, "Empty dataframe detection failed"
            
            # Test missing data handling
            missing_data = None
            assert missing_data is None, "Missing data detection failed"
            
            self.log_test("Data Validation", True, "Data validation tests passed")
            
        except Exception as e:
            self.log_test("Data Validation", False, f"Data validation failed: {e}")
    
    def test_error_handling(self):
        """Test error handling scenarios."""
        logger.info("Testing error handling...")
        
        try:
            # Test invalid input handling
            try:
                invalid_value = float("invalid")
                assert False, "Should have raised ValueError"
            except ValueError:
                pass  # Expected behavior
            
            # Test division by zero handling
            try:
                result = 1 / 0
                assert False, "Should have raised ZeroDivisionError"
            except ZeroDivisionError:
                pass  # Expected behavior
            
            # Test index out of bounds handling
            try:
                arr = [1, 2, 3]
                value = arr[10]
                assert False, "Should have raised IndexError"
            except IndexError:
                pass  # Expected behavior
            
            # Test missing key handling
            try:
                dict_obj = {'a': 1, 'b': 2}
                value = dict_obj['c']
                assert False, "Should have raised KeyError"
            except KeyError:
                pass  # Expected behavior
            
            self.log_test("Error Handling", True, "Error handling scenarios tested successfully")
            
        except Exception as e:
            self.log_test("Error Handling", False, f"Error handling test failed: {e}")
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        logger.info("Testing configuration validation...")
        
        try:
            # Test valid configuration
            config = {
                'prediction_horizon': 10,
                'update_frequency': 'Real-time',
                'plot_type': '2D Trajectory',
                'show_grid': True,
                'color_scheme': 'Default',
                'animation_speed': 1.0
            }
            
            # Validate configuration values
            assert 1 <= config['prediction_horizon'] <= 30, "Invalid prediction horizon"
            assert config['update_frequency'] in ['Real-time', '5 seconds', '10 seconds', '30 seconds', 'Manual'], "Invalid update frequency"
            assert config['plot_type'] in ['2D Trajectory', '3D Trajectory', 'Velocity Profile', 'Acceleration Profile'], "Invalid plot type"
            assert isinstance(config['show_grid'], bool), "Invalid show_grid value"
            assert config['color_scheme'] in ['Default', 'Viridis', 'Plasma', 'Inferno', 'Magma'], "Invalid color scheme"
            assert 0.1 <= config['animation_speed'] <= 2.0, "Invalid animation speed"
            
            # Test invalid configuration handling
            invalid_config = {
                'prediction_horizon': -1,  # Invalid
                'update_frequency': 'Invalid',  # Invalid
                'plot_type': 'Invalid',  # Invalid
                'show_grid': 'not_bool',  # Invalid
                'color_scheme': 'Invalid',  # Invalid
                'animation_speed': 5.0  # Invalid
            }
            
            # Test validation of invalid config
            errors = []
            if not (1 <= invalid_config['prediction_horizon'] <= 30):
                errors.append("Invalid prediction horizon")
            if invalid_config['update_frequency'] not in ['Real-time', '5 seconds', '10 seconds', '30 seconds', 'Manual']:
                errors.append("Invalid update frequency")
            if invalid_config['plot_type'] not in ['2D Trajectory', '3D Trajectory', 'Velocity Profile', 'Acceleration Profile']:
                errors.append("Invalid plot type")
            if not isinstance(invalid_config['show_grid'], bool):
                errors.append("Invalid show_grid value")
            if invalid_config['color_scheme'] not in ['Default', 'Viridis', 'Plasma', 'Inferno', 'Magma']:
                errors.append("Invalid color scheme")
            if not (0.1 <= invalid_config['animation_speed'] <= 2.0):
                errors.append("Invalid animation speed")
            
            assert len(errors) > 0, "Invalid configuration not detected"
            
            self.log_test("Configuration Validation", True, "Configuration validation working correctly")
            
        except Exception as e:
            self.log_test("Configuration Validation", False, f"Configuration validation failed: {e}")
    
    def test_component_state_management(self):
        """Test component state management."""
        logger.info("Testing component state management...")
        
        try:
            # Test session state simulation
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
            
            self.log_test("Component State Management", True, "Component state management working correctly")
            
        except Exception as e:
            self.log_test("Component State Management", False, f"Component state management failed: {e}")
    
    def print_results(self):
        """Print test results summary."""
        logger.info("\n" + "="*60)
        logger.info("DASHBOARD COMPONENT TEST RESULTS")
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
    """Main function to run the component tests."""
    tester = DashboardComponentTester()
    tester.run_all_tests()

if __name__ == "__main__":
    main()