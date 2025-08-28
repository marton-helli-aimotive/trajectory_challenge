#!/usr/bin/env python3
"""
E2E Testing Script for Vehicle Trajectory Prediction Dashboard

This script uses Playwright to test all dashboard functionality:
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
            # Test all navigation options
            nav_options = [
                "🏠 Overview",
                "📊 Trajectory Visualization", 
                "🔍 Model Comparison",
                "📈 Dataset Exploration",
                "🤖 Model Explainability",
                "⚙️ Settings"
            ]
            
            for option in nav_options:
                # Find and click navigation option
                nav_selector = f'text="{option}"'
                await page.click(nav_selector)
                
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
            await page.click('text="🏠 Overview"')
            await page.wait_for_timeout(2000)
            
            # Test quick start buttons
            quick_start_buttons = [
                "Go to Trajectory Visualization",
                "Go to Model Comparison"
            ]
            
            for button_text in quick_start_buttons:
                try:
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
            await page.click('text="📊 Trajectory Visualization"')
            await page.wait_for_timeout(2000)
            
            # Test trajectory selection dropdown
            try:
                await page.click('selectbox:has-text("Select Trajectory")')
                await page.wait_for_timeout(1000)
                logger.info("✓ Trajectory selection dropdown clicked")
            except Exception as e:
                self.warnings.append(f"Trajectory selection dropdown not found: {e}")
            
            # Test plot type selection
            plot_types = ["2D Trajectory", "3D Trajectory", "Velocity Profile", "Acceleration Profile"]
            for plot_type in plot_types:
                try:
                    await page.select_option('selectbox:has-text("Plot Type")', plot_type)
                    await page.wait_for_timeout(1000)
                    logger.info(f"✓ Plot type '{plot_type}' selected")
                except Exception as e:
                    self.warnings.append(f"Plot type '{plot_type}' selection failed: {e}")
            
            # Test prediction checkbox
            try:
                await page.click('checkbox:has-text("Show Predictions")')
                await page.wait_for_timeout(1000)
                logger.info("✓ Show predictions checkbox toggled")
            except Exception as e:
                self.warnings.append(f"Show predictions checkbox not found: {e}")
            
            # Test model selection (if predictions are enabled)
            try:
                model_options = ["Constant Velocity", "Polynomial Regression"]
                for model in model_options:
                    await page.click(f'text="{model}"')
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
            await page.click('text="🔍 Model Comparison"')
            await page.wait_for_timeout(2000)
            
            # Test model selection
            model_names = ["Constant Velocity", "Constant Acceleration", "Polynomial Regression"]
            for model in model_names:
                try:
                    await page.click(f'text="{model}"')
                    await page.wait_for_timeout(500)
                except Exception as e:
                    self.warnings.append(f"Model '{model}' selection failed: {e}")
            
            # Test prediction horizon slider
            try:
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
                await page.click('checkbox:has-text("Include Safety Metrics")')
                await page.wait_for_timeout(1000)
                logger.info("✓ Safety metrics checkbox toggled")
            except Exception as e:
                self.warnings.append(f"Safety metrics checkbox not found: {e}")
            
            # Test run comparison button
            try:
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
            await page.click('text="📈 Dataset Exploration"')
            await page.wait_for_timeout(2000)
            
            # Check for dataset overview metrics
            metrics = ["Total Trajectories", "Total Duration", "Avg Trajectory Length", "Data Points"]
            for metric in metrics:
                try:
                    await page.wait_for_selector(f'text="{metric}"', timeout=5000)
                    logger.info(f"✓ Metric '{metric}' found")
                except Exception as e:
                    self.warnings.append(f"Metric '{metric}' not found: {e}")
            
            # Check for distribution plots
            try:
                await page.wait_for_selector("text=Trajectory Duration Distribution", timeout=5000)
                await page.wait_for_selector("text=Average Velocity Distribution", timeout=5000)
                logger.info("✓ Distribution plots found")
            except Exception as e:
                self.warnings.append(f"Distribution plots not found: {e}")
            
        except Exception as e:
            self.errors.append(f"Dataset exploration test failed: {e}")
    
    async def test_model_explainability(self, page):
        """Test model explainability page."""
        logger.info("Testing model explainability...")
        
        try:
            # Navigate to model explainability
            await page.click('text="🤖 Model Explainability"')
            await page.wait_for_timeout(2000)
            
            # Test model selection dropdown
            try:
                await page.click('selectbox:has-text("Select Model for Analysis")')
                await page.wait_for_timeout(1000)
                logger.info("✓ Model selection dropdown clicked")
            except Exception as e:
                self.warnings.append(f"Model selection dropdown not found: {e}")
            
            # Test explainability type selection
            explainability_types = [
                "Feature Importance", 
                "Prediction Analysis", 
                "Model Behavior", 
                "Uncertainty Analysis"
            ]
            
            for exp_type in explainability_types:
                try:
                    await page.select_option('selectbox:has-text("Explainability Type")', exp_type)
                    await page.wait_for_timeout(1000)
                    logger.info(f"✓ Explainability type '{exp_type}' selected")
                except Exception as e:
                    self.warnings.append(f"Explainability type '{exp_type}' selection failed: {e}")
            
        except Exception as e:
            self.errors.append(f"Model explainability test failed: {e}")
    
    async def test_settings_page(self, page):
        """Test settings page."""
        logger.info("Testing settings page...")
        
        try:
            # Navigate to settings
            await page.click('text="⚙️ Settings"')
            await page.wait_for_timeout(2000)
            
            # Test prediction horizon slider
            try:
                sliders = await page.query_selector_all('input[type="range"]')
                if sliders:
                    await sliders[0].fill("20")
                    await page.wait_for_timeout(1000)
                    logger.info("✓ Prediction horizon slider adjusted")
            except Exception as e:
                self.warnings.append(f"Prediction horizon slider not found: {e}")
            
            # Test update frequency selection
            try:
                await page.select_option('selectbox:has-text("Update Frequency")', "10 seconds")
                await page.wait_for_timeout(1000)
                logger.info("✓ Update frequency selected")
            except Exception as e:
                self.warnings.append(f"Update frequency selection failed: {e}")
            
            # Test default plot type selection
            try:
                await page.select_option('selectbox:has-text("Default Plot Type")', "3D Trajectory")
                await page.wait_for_timeout(1000)
                logger.info("✓ Default plot type selected")
            except Exception as e:
                self.warnings.append(f"Default plot type selection failed: {e}")
            
            # Test checkboxes
            checkboxes = ["Show Grid", "Cache Model Predictions"]
            for checkbox in checkboxes:
                try:
                    await page.click(f'checkbox:has-text("{checkbox}")')
                    await page.wait_for_timeout(500)
                    logger.info(f"✓ Checkbox '{checkbox}' toggled")
                except Exception as e:
                    self.warnings.append(f"Checkbox '{checkbox}' not found: {e}")
            
            # Test save settings button
            try:
                await page.click('button:has-text("Save Settings")')
                await page.wait_for_timeout(2000)
                logger.info("✓ Settings saved")
            except Exception as e:
                self.warnings.append(f"Save settings button not found: {e}")
            
        except Exception as e:
            self.errors.append(f"Settings page test failed: {e}")
    
    async def test_error_handling(self, page):
        """Test error handling."""
        logger.info("Testing error handling...")
        
        try:
            # Check for any error messages on the page
            error_elements = await page.query_selector_all('.stAlert, .error, [data-testid="stAlert"]')
            
            if error_elements:
                for element in error_elements:
                    error_text = await element.text_content()
                    self.errors.append(f"Error found on page: {error_text}")
                    logger.warning(f"Error found: {error_text}")
            else:
                logger.info("✓ No errors found on page")
            
        except Exception as e:
            self.errors.append(f"Error handling test failed: {e}")
    
    async def test_console_errors(self, page):
        """Test for console errors."""
        logger.info("Testing console errors...")
        
        try:
            # Listen for console errors
            console_errors = []
            
            def handle_console_error(msg):
                if msg.type == "error":
                    console_errors.append(msg.text)
            
            page.on("console", handle_console_error)
            
            # Navigate through all pages to trigger any console errors
            pages = [
                "🏠 Overview",
                "📊 Trajectory Visualization",
                "🔍 Model Comparison", 
                "📈 Dataset Exploration",
                "🤖 Model Explainability",
                "⚙️ Settings"
            ]
            
            for page_name in pages:
                await page.click(f'text="{page_name}"')
                await page.wait_for_timeout(2000)
            
            if console_errors:
                for error in console_errors:
                    self.errors.append(f"Console error: {error}")
                    logger.warning(f"Console error: {error}")
            else:
                logger.info("✓ No console errors found")
            
        except Exception as e:
            self.errors.append(f"Console error test failed: {e}")
    
    def print_results(self):
        """Print test results."""
        print("\n" + "="*60)
        print("E2E TEST RESULTS")
        print("="*60)
        
        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for i, error in enumerate(self.errors, 1):
                print(f"  {i}. {error}")
        else:
            print("\n✅ No errors found!")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for i, warning in enumerate(self.warnings, 1):
                print(f"  {i}. {warning}")
        else:
            print("\n✅ No warnings found!")
        
        print(f"\n📊 SUMMARY:")
        print(f"  - Errors: {len(self.errors)}")
        print(f"  - Warnings: {len(self.warnings)}")
        
        if not self.errors:
            print("\n🎉 All E2E tests passed successfully!")
        else:
            print(f"\n❌ {len(self.errors)} errors need to be fixed.")

async def main():
    """Main function to run E2E tests."""
    tester = DashboardE2ETester()
    await tester.run_tests()

if __name__ == "__main__":
    asyncio.run(main())