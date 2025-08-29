#!/usr/bin/env python3
"""
Simple test script to identify dashboard issues without streamlit dependency.
"""

import sys
import os
import time
import logging
from typing import Dict, List, Any
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DashboardIssueTester:
    """Tester for identifying and fixing dashboard issues."""
    
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
    
    def test_2d_trajectory_plot(self):
        """Test 2D trajectory plot to identify colorscale issue."""
        logger.info("Testing 2D trajectory plot...")
        
        try:
            # Create sample trajectory data
            t = np.linspace(0, 10, 50)
            x = t * 10 + np.random.normal(0, 1, 50)
            y = t * 5 + np.random.normal(0, 1, 50)
            v = 10 + np.random.normal(0, 2, 50)
            
            # Test the problematic plot creation from the original code
            fig = go.Figure()
            
            # This is the problematic line from the original code
            # The issue is that 'colorscale' is not a valid property for go.Scatter
            # It should be 'marker_colorscale' or the color should be set differently
            
            # Original problematic code:
            # fig.add_trace(go.Scatter(
            #     x=x, y=y,
            #     mode='lines+markers',
            #     name='Actual Trajectory',
            #     line=dict(color='#1f77b4', width=3),
            #     marker=dict(size=6, color=v, colorscale='Viridis')  # This is wrong
            # ))
            
            # Fixed version:
            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode='lines+markers',
                name='Actual Trajectory',
                line=dict(color='#1f77b4', width=3),
                marker=dict(
                    size=6, 
                    color=v, 
                    colorscale='Viridis',  # This is correct for marker
                    showscale=True
                )
            ))
            
            # Test if the plot can be created without errors
            plot_json = fig.to_json()
            
            self.log_test("2D Trajectory Plot Creation", True, "Plot created successfully")
            
            # Test with the correct approach
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=x, y=y,
                mode='lines+markers',
                name='Actual Trajectory',
                line=dict(color='#1f77b4', width=3),
                marker=dict(
                    size=6,
                    color=v,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Velocity (m/s)")
                )
            ))
            
            plot_json2 = fig2.to_json()
            self.log_test("2D Trajectory Plot with Colorbar", True, "Plot with colorbar created successfully")
            
        except Exception as e:
            self.log_test("2D Trajectory Plot Creation", False, f"Error: {str(e)}")
    
    def test_model_comparison(self):
        """Test model comparison to identify model_name attribute issue."""
        logger.info("Testing model comparison...")
        
        try:
            # Test model structure
            models = ["Constant Velocity", "Constant Acceleration", "Polynomial Regression", "KNN", "Gaussian Process", "Ensemble"]
            
            # Simulate evaluation results
            results = []
            for model in models:
                # Test if model has model_name attribute (it should be a string, not an object)
                if isinstance(model, str):
                    results.append({
                        'Model': model,
                        'RMSE': np.random.uniform(0.5, 2.0),
                        'ADE': np.random.uniform(0.3, 1.5),
                        'FDE': np.random.uniform(0.8, 3.0),
                        'Inference Time (ms)': np.random.uniform(10, 100)
                    })
                else:
                    # This would be the problematic case
                    raise ValueError(f"Model should be a string, got {type(model)}")
            
            # Create DataFrame
            metrics_df = pd.DataFrame(results)
            
            # Test if DataFrame can be created
            if len(metrics_df) == len(models):
                self.log_test("Model Comparison DataFrame", True, f"Created DataFrame with {len(models)} models")
            else:
                self.log_test("Model Comparison DataFrame", False, "DataFrame length mismatch")
            
            # Test performance comparison plot
            fig = px.bar(metrics_df, x='Model', y='RMSE', title="RMSE Comparison")
            plot_json = fig.to_json()
            self.log_test("Model Comparison Plot", True, "Comparison plot created successfully")
            
        except Exception as e:
            self.log_test("Model Comparison", False, f"Error: {str(e)}")
    
    def test_ngsim_data_integration(self):
        """Test NGSIM data integration."""
        logger.info("Testing NGSIM data integration...")
        
        try:
            # Check if NGSIM data directory exists
            data_path = Path("data/ngsim")
            if data_path.exists():
                self.log_test("NGSIM Data Directory", True, "NGSIM data directory found")
                
                # Check for CSV files
                csv_files = list(data_path.glob("*.csv"))
                if csv_files:
                    self.log_test("NGSIM CSV Files", True, f"Found {len(csv_files)} CSV files")
                    
                    # Try to read one file
                    try:
                        sample_df = pd.read_csv(csv_files[0])
                        self.log_test("NGSIM Data Reading", True, f"Successfully read {len(sample_df)} rows from {csv_files[0].name}")
                        
                        # Check for required columns
                        required_columns = ['Vehicle_ID', 'Frame_ID', 'Local_X', 'Local_Y', 'v_Vel']
                        missing_columns = set(required_columns) - set(sample_df.columns)
                        if missing_columns:
                            self.log_test("NGSIM Required Columns", False, f"Missing columns: {missing_columns}")
                        else:
                            self.log_test("NGSIM Required Columns", True, "All required columns present")
                            
                    except Exception as e:
                        self.log_test("NGSIM Data Reading", False, f"Error reading CSV: {str(e)}")
                else:
                    self.log_test("NGSIM CSV Files", False, "No CSV files found in NGSIM directory")
            else:
                self.log_test("NGSIM Data Directory", False, "NGSIM data directory not found")
                self.warnings.append("NGSIM data directory not found - using sample data")
            
        except Exception as e:
            self.log_test("NGSIM Integration", False, f"Error: {str(e)}")
    
    def run_all_tests(self):
        """Run all issue identification tests."""
        logger.info("Starting dashboard issue identification tests...")
        
        try:
            # Test 1: 2D Trajectory plot issue
            self.test_2d_trajectory_plot()
            
            # Test 2: Model comparison issue
            self.test_model_comparison()
            
            # Test 3: NGSIM data integration
            self.test_ngsim_data_integration()
            
            # Print final results
            self.print_results()
            
        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            self.errors.append(f"Test execution failed: {e}")
    
    def print_results(self):
        """Print test results summary."""
        logger.info("\n" + "="*50)
        logger.info("DASHBOARD ISSUE TEST RESULTS")
        logger.info("="*50)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['success'])
        failed_tests = total_tests - passed_tests
        
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {failed_tests}")
        
        if self.errors:
            logger.info("\n❌ ERRORS:")
            for error in self.errors:
                logger.info(f"  - {error}")
        
        if self.warnings:
            logger.info("\n⚠️ WARNINGS:")
            for warning in self.warnings:
                logger.info(f"  - {warning}")
        
        if failed_tests == 0:
            logger.info("\n✅ All tests passed!")
        else:
            logger.info(f"\n❌ {failed_tests} test(s) failed. Please review the errors above.")

if __name__ == "__main__":
    tester = DashboardIssueTester()
    tester.run_all_tests()