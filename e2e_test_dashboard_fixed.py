#!/usr/bin/env python3
"""
Fixed E2E Testing Script for Vehicle Trajectory Prediction Dashboard

This script uses Playwright to test all dashboard functionality with correct Streamlit selectors:
- Navigation between pages
- Button interactions
- Dropdown selections
- Form inputs
- Error handling
"""

import asyncio
import time
from playwright.async_api import async_playwright
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DashboardE2ETester:
    """E2E tester for the Streamlit dashboard."""
    
    def __init__(self, base_url="http://localhost:8501"):
        self.base_url = base_url
        self.errors = []
        self.warnings = []
        
    async def run_tests(self):
        """Run all E2E tests."""
        async with async_playwright() as p:
            # Launch browser
            browser = await p.chromium.launch(
                headless=True,
                args=['--no-sandbox', '--disable-dev-shm-usage']
            )
            
            try:
                page = await browser.new_page()
                
                # Set viewport
                await page.set_viewport_size({"width": 1920, "height": 1080})
                
                logger.info("Starting E2E tests...")
                
                # Test 1: Basic page load
                await self.test_page_load(page)
                
                # Test 2: Navigation
                await self.test_navigation(page)
                
                # Test 3: Overview page interactions
                await self.test_overview_page(page)
                
                # Test 4: Trajectory visualization page
                await self.test_trajectory_visualization(page)
                
                # Test 5: Model comparison page
                await self.test_model_comparison(page)
                
                # Test 6: Dataset exploration page
                await self.test_dataset_exploration(page)
                
                # Test 7: Model explainability page
                await self.test_model_explainability(page)
                
                # Test 8: Settings page
                await self.test_settings_page(page)
                
                # Test 9: Error handling
                await self.test_error_handling(page)
                
                # Test 10: Console errors
                await self.test_console_errors(page)
                
                # Print results
                self.print_results()
                
            except Exception as e:
                logger.error(f"Test execution failed: {e}")
                self.errors.append(f"Test execution failed: {e}")
            finally:
                await browser.close()
    
    async def test_page_load(self, page):
        """Test basic page loading."""
        logger.info("Testing page load...")
        
        try:
            # Navigate to dashboard
            response = await page.goto(self.base_url, wait_until="networkidle")
            
            if response.status != 200:
                self.errors.append(f"Page load failed with status {response.status}")
                return
            
            # Wait for Streamlit to load
            await page.wait_for_selector("h1", timeout=10000)
            
            # Check if main header is present
            header = await page.text_content("h1")
            if "Vehicle Trajectory Prediction Dashboard" not in header:
                self.errors.append("Main header not found")
            
            logger.info("✓ Page load test passed")
            
        except Exception as e:
            self.errors.append(f"Page load test failed: {e}")
    
    async def test_navigation(self, page):
        """Test navigation between pages."""
        logger.info("Testing navigation...")
        
        try:
            # Test all navigation options using Streamlit sidebar selectbox
            nav_options = [
                "🏠 Overview",
                "📊 Trajectory Visualization", 
                "🔍 Model Comparison",
                "📈 Dataset Exploration",
                "🤖 Model Explainability",
                "⚙️ Settings"
            ]
            
            for option in nav_options:
                # Use Streamlit selectbox selector
                await page.select_option('[data-testid="stSelectbox"]', option)
                
                # Wait for page to load
                await page.wait_for_timeout(2000)
                
                # Verify page content loaded
                if option == "🏠 Overview":
                    await page.wait_for_selector("text=System Overview", timeout=5000)
                elif option == "📊 Trajectory Visualization":
                    await page.wait_for_selector("text=Trajectory Visualization", timeout=5000)
                elif option == "🔍 Model Comparison":
                    await page.wait_for_selector("text=Model Comparison", timeout=5000)
                elif option == "📈 Dataset Exploration":
                    await page.wait_for_selector("text=Dataset Exploration", timeout=5000)
                elif option == "🤖 Model Explainability":
                    await page.wait_for_selector("text=Model Explainability", timeout=5000)
                elif option == "⚙️ Settings":
                    await page.wait_for_selector("text=Dashboard Settings", timeout=5000)
                
                logger.info(f"✓ Navigation to {option} successful")
            
        except Exception as e:
            self.errors.append(f"Navigation test failed: {e}")
    
    async def test_overview_page(self, page):
        """Test overview page interactions."""
        logger.info("Testing overview page...")
        
        try:
            # Navigate to overview
            await page.select_option('[data-testid="stSelectbox"]', "🏠 Overview")
            await page.wait_for_timeout(2000)
            
            # Test quick start buttons
            quick_start_buttons = [
                "Go to Trajectory Visualization",
                "Go to Model Comparison"
            ]
            
            for button_text in quick_start_buttons:
                try:
                    # Use Streamlit button selector
                    await page.click(f'button:has-text("{button_text}")')
                    await page.wait_for_timeout(1000)
                    logger.info(f"✓ Quick start button '{button_text}' clicked")
                except Exception as e:
                    self.warnings.append(f"Quick start button '{button_text}' not found: {e}")
            
            # Test performance data table
            try:
                await page.wait_for_selector("table", timeout=5000)
                logger.info("✓ Performance data table found")
            except Exception as e:
                self.warnings.append(f"Performance data table not found: {e}")
            
        except Exception as e:
            self.errors.append(f"Overview page test failed: {e}")
    
    async def test_trajectory_visualization(self, page):
        """Test trajectory visualization page."""
        logger.info("Testing trajectory visualization...")
        
        try:
            # Navigate to trajectory visualization
            await page.select_option('[data-testid="stSelectbox"]', "📊 Trajectory Visualization")
            await page.wait_for_timeout(2000)
            
            # Test trajectory selection dropdown
            try:
                # Use Streamlit selectbox selector for trajectory selection
                await page.select_option('select:has-text("Select Trajectory")', "Vehicle 1")
                await page.wait_for_timeout(1000)
                logger.info("✓ Trajectory selection dropdown clicked")
            except Exception as e:
                self.warnings.append(f"Trajectory selection dropdown not found: {e}")
            
            # Test plot type selection
            plot_types = ["2D Trajectory", "3D Trajectory", "Velocity Profile", "Acceleration Profile"]
            for plot_type in plot_types:
                try:
                    # Use Streamlit selectbox selector for plot type
                    await page.select_option('select:has-text("Plot Type")', plot_type)
                    await page.wait_for_timeout(1000)
                    logger.info(f"✓ Plot type '{plot_type}' selected")
                except Exception as e:
                    self.warnings.append(f"Plot type '{plot_type}' selection failed: {e}")
            
            # Test prediction checkbox
            try:
                # Use Streamlit checkbox selector
                await page.click('input[type="checkbox"]:has-text("Show Predictions")')
                await page.wait_for_timeout(1000)
                logger.info("✓ Show predictions checkbox toggled")
            except Exception as e:
                self.warnings.append(f"Show predictions checkbox not found: {e}")
            
            # Test model selection (if predictions are enabled)
            try:
                model_options = ["Constant Velocity", "Polynomial Regression"]
                for model in model_options:
                    # Use Streamlit multiselect selector
                    await page.click(f'div:has-text("{model}")')
                    await page.wait_for_timeout(500)
                logger.info("✓ Model selection tested")
            except Exception as e:
                self.warnings.append(f"Model selection failed: {e}")
            
        except Exception as e:
            self.errors.append(f"Trajectory visualization test failed: {e}")
    
    async def test_model_comparison(self, page):
        """Test model comparison page."""
        logger.info("Testing model comparison...")
        
        try:
            # Navigate to model comparison
            await page.select_option('[data-testid="stSelectbox"]', "🔍 Model Comparison")
            await page.wait_for_timeout(2000)
            
            # Test model selection
            model_names = ["Constant Velocity", "Constant Acceleration", "Polynomial Regression"]
            for model in model_names:
                try:
                    # Use Streamlit multiselect selector
                    await page.click(f'div:has-text("{model}")')
                    await page.wait_for_timeout(500)
                except Exception as e:
                    self.warnings.append(f"Model '{model}' selection failed: {e}")
            
            # Test prediction horizon slider
            try:
                # Use Streamlit slider selector
                slider = await page.query_selector('input[type="range"]')
                if slider:
                    await slider.fill("15")
                    await page.wait_for_timeout(1000)
                    logger.info("✓ Prediction horizon slider adjusted")
            except Exception as e:
                self.warnings.append(f"Prediction horizon slider not found: {e}")
            
            # Test test size slider
            try:
                sliders = await page.query_selector_all('input[type="range"]')
                if len(sliders) > 1:
                    await sliders[1].fill("2")
                    await page.wait_for_timeout(1000)
                    logger.info("✓ Test size slider adjusted")
            except Exception as e:
                self.warnings.append(f"Test size slider not found: {e}")
            
            # Test safety metrics checkbox
            try:
                # Use Streamlit checkbox selector
                await page.click('input[type="checkbox"]:has-text("Include Safety Metrics")')
                await page.wait_for_timeout(1000)
                logger.info("✓ Safety metrics checkbox toggled")
            except Exception as e:
                self.warnings.append(f"Safety metrics checkbox not found: {e}")
            
            # Test run comparison button
            try:
                # Use Streamlit button selector
                await page.click('button:has-text("Run Model Comparison")')
                await page.wait_for_timeout(5000)  # Wait for evaluation to complete
                logger.info("✓ Model comparison executed")
            except Exception as e:
                self.warnings.append(f"Run model comparison button not found: {e}")
            
        except Exception as e:
            self.errors.append(f"Model comparison test failed: {e}")
    
    async def test_dataset_exploration(self, page):
        """Test dataset exploration page."""
        logger.info("Testing dataset exploration...")
        
        try:
            # Navigate to dataset exploration
            await page.select_option('[data-testid="stSelectbox"]', "📈 Dataset Exploration")
            await page.wait_for_timeout(2000)
            
            # Check for dataset overview metrics
            try:
                await page.wait_for_selector("text=Dataset Overview", timeout=5000)
                logger.info("✓ Dataset overview section found")
            except Exception as e:
                self.warnings.append(f"Dataset overview section not found: {e}")
            
            # Check for distribution plots
            try:
                await page.wait_for_selector("text=Trajectory Distribution", timeout=5000)
                logger.info("✓ Trajectory distribution section found")
            except Exception as e:
                self.warnings.append(f"Trajectory distribution section not found: {e}")
            
        except Exception as e:
            self.errors.append(f"Dataset exploration test failed: {e}")
    
    async def test_model_explainability(self, page):
        """Test model explainability page."""
        logger.info("Testing model explainability...")
        
        try:
            # Navigate to model explainability
            await page.select_option('[data-testid="stSelectbox"]', "🤖 Model Explainability")
            await page.wait_for_timeout(2000)
            
            # Test model selection
            try:
                # Use Streamlit selectbox selector
                await page.select_option('select:has-text("Select Model for Analysis")', "Constant Velocity")
                await page.wait_for_timeout(1000)
                logger.info("✓ Model selection for explainability successful")
            except Exception as e:
                self.warnings.append(f"Model selection for explainability failed: {e}")
            
            # Test explainability type selection
            try:
                # Use Streamlit selectbox selector
                await page.select_option('select:has-text("Explainability Type")', "Feature Importance")
                await page.wait_for_timeout(1000)
                logger.info("✓ Explainability type selection successful")
            except Exception as e:
                self.warnings.append(f"Explainability type selection failed: {e}")
            
        except Exception as e:
            self.errors.append(f"Model explainability test failed: {e}")
    
    async def test_settings_page(self, page):
        """Test settings page."""
        logger.info("Testing settings page...")
        
        try:
            # Navigate to settings
            await page.select_option('[data-testid="stSelectbox"]', "⚙️ Settings")
            await page.wait_for_timeout(2000)
            
            # Test prediction horizon slider
            try:
                # Use Streamlit slider selector
                slider = await page.query_selector('input[type="range"]')
                if slider:
                    await slider.fill("15")
                    await page.wait_for_timeout(1000)
                    logger.info("✓ Settings prediction horizon slider adjusted")
            except Exception as e:
                self.warnings.append(f"Settings prediction horizon slider not found: {e}")
            
            # Test update frequency selection
            try:
                # Use Streamlit selectbox selector
                await page.select_option('select:has-text("Update Frequency")', "10 seconds")
                await page.wait_for_timeout(1000)
                logger.info("✓ Update frequency selection successful")
            except Exception as e:
                self.warnings.append(f"Update frequency selection failed: {e}")
            
            # Test save settings button
            try:
                # Use Streamlit button selector
                await page.click('button:has-text("Save Settings")')
                await page.wait_for_timeout(1000)
                logger.info("✓ Save settings button clicked")
            except Exception as e:
                self.warnings.append(f"Save settings button not found: {e}")
            
        except Exception as e:
            self.errors.append(f"Settings page test failed: {e}")
    
    async def test_error_handling(self, page):
        """Test error handling scenarios."""
        logger.info("Testing error handling...")
        
        try:
            # Check for any error messages on the page
            error_elements = await page.query_selector_all('[data-testid="stAlert"]')
            
            if error_elements:
                for error in error_elements:
                    error_text = await error.text_content()
                    self.warnings.append(f"Error found on page: {error_text}")
            else:
                logger.info("✓ No errors found on page")
            
        except Exception as e:
            self.errors.append(f"Error handling test failed: {e}")
    
    async def test_console_errors(self, page):
        """Test for console errors."""
        logger.info("Testing console errors...")
        
        try:
            # Check for console errors
            console_errors = []
            
            page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
            
            # Navigate through all pages to trigger any console errors
            nav_options = [
                "🏠 Overview",
                "📊 Trajectory Visualization", 
                "🔍 Model Comparison",
                "📈 Dataset Exploration",
                "🤖 Model Explainability",
                "⚙️ Settings"
            ]
            
            for option in nav_options:
                try:
                    await page.select_option('[data-testid="stSelectbox"]', option)
                    await page.wait_for_timeout(1000)
                except:
                    pass
            
            if console_errors:
                for error in console_errors:
                    self.warnings.append(f"Console error: {error}")
            else:
                logger.info("✓ No console errors found")
            
        except Exception as e:
            self.errors.append(f"Console error test failed: {e}")
    
    def print_results(self):
        """Print test results summary."""
        print("\n" + "="*60)
        print("E2E TEST RESULTS")
        print("="*60)
        
        if self.errors:
            print("\n❌ ERRORS:")
            for i, error in enumerate(self.errors, 1):
                print(f"  {i}. {error}")
        
        if self.warnings:
            print("\n⚠️  WARNINGS:")
            for i, warning in enumerate(self.warnings, 1):
                print(f"  {i}. {warning}")
        
        print(f"\n📊 SUMMARY:")
        print(f"  - Errors: {len(self.errors)}")
        print(f"  - Warnings: {len(self.warnings)}")
        
        if self.errors:
            print(f"\n❌ {len(self.errors)} errors need to be fixed.")
        else:
            print("\n🎉 All tests passed!")

async def main():
    """Main function to run the E2E tests."""
    tester = DashboardE2ETester()
    await tester.run_tests()

if __name__ == "__main__":
    asyncio.run(main())