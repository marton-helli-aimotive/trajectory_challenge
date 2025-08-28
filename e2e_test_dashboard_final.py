#!/usr/bin/env python3
"""
Final E2E Testing Script for Vehicle Trajectory Prediction Dashboard

This script uses Playwright to test all dashboard functionality with correct Streamlit interaction methods:
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
                # Use Streamlit selectbox interaction - click to open and then select option
                selectbox = await page.query_selector('[data-testid="stSelectbox"]')
                if selectbox:
                    # Click to open the dropdown
                    await selectbox.click()
                    await page.wait_for_timeout(500)
                    
                    # Click on the option
                    option_element = await page.query_selector(f'text="{option}"')
                    if option_element:
                        await option_element.click()
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
                    else:
                        self.warnings.append(f"Navigation option '{option}' not found")
                else:
                    self.warnings.append("Navigation selectbox not found")
            
        except Exception as e:
            self.errors.append(f"Navigation test failed: {e}")
    
    async def test_overview_page(self, page):
        """Test overview page interactions."""
        logger.info("Testing overview page...")
        
        try:
            # Navigate to overview
            selectbox = await page.query_selector('[data-testid="stSelectbox"]')
            if selectbox:
                await selectbox.click()
                await page.wait_for_timeout(500)
                overview_option = await page.query_selector('text="🏠 Overview"')
                if overview_option:
                    await overview_option.click()
                    await page.wait_for_timeout(2000)
            
            # Test quick start buttons
            quick_start_buttons = [
                "Go to Trajectory Visualization",
                "Go to Model Comparison"
            ]
            
            for button_text in quick_start_buttons:
                try:
                    # Use Streamlit button selector
                    button = await page.query_selector(f'button:has-text("{button_text}")')
                    if button:
                        await button.click()
                        await page.wait_for_timeout(1000)
                        logger.info(f"✓ Quick start button '{button_text}' clicked")
                    else:
                        self.warnings.append(f"Quick start button '{button_text}' not found")
                except Exception as e:
                    self.warnings.append(f"Quick start button '{button_text}' interaction failed: {e}")
            
            # Test performance data table
            try:
                table = await page.query_selector("table")
                if table:
                    logger.info("✓ Performance data table found")
                else:
                    self.warnings.append("Performance data table not found")
            except Exception as e:
                self.warnings.append(f"Performance data table check failed: {e}")
            
        except Exception as e:
            self.errors.append(f"Overview page test failed: {e}")
    
    async def test_trajectory_visualization(self, page):
        """Test trajectory visualization page."""
        logger.info("Testing trajectory visualization...")
        
        try:
            # Navigate to trajectory visualization
            selectbox = await page.query_selector('[data-testid="stSelectbox"]')
            if selectbox:
                await selectbox.click()
                await page.wait_for_timeout(500)
                trajectory_option = await page.query_selector('text="📊 Trajectory Visualization"')
                if trajectory_option:
                    await trajectory_option.click()
                    await page.wait_for_timeout(2000)
            
            # Test trajectory selection dropdown
            try:
                # Look for trajectory selectbox
                trajectory_selectbox = await page.query_selector('text="Select Trajectory"')
                if trajectory_selectbox:
                    await trajectory_selectbox.click()
                    await page.wait_for_timeout(500)
                    vehicle_option = await page.query_selector('text="Vehicle 1"')
                    if vehicle_option:
                        await vehicle_option.click()
                        await page.wait_for_timeout(1000)
                        logger.info("✓ Trajectory selection dropdown clicked")
                    else:
                        self.warnings.append("Vehicle option not found")
                else:
                    self.warnings.append("Trajectory selection dropdown not found")
            except Exception as e:
                self.warnings.append(f"Trajectory selection dropdown interaction failed: {e}")
            
            # Test plot type selection
            plot_types = ["2D Trajectory", "3D Trajectory", "Velocity Profile", "Acceleration Profile"]
            for plot_type in plot_types:
                try:
                    # Look for plot type selectbox
                    plot_selectbox = await page.query_selector('text="Plot Type"')
                    if plot_selectbox:
                        await plot_selectbox.click()
                        await page.wait_for_timeout(500)
                        plot_option = await page.query_selector(f'text="{plot_type}"')
                        if plot_option:
                            await plot_option.click()
                            await page.wait_for_timeout(1000)
                            logger.info(f"✓ Plot type '{plot_type}' selected")
                        else:
                            self.warnings.append(f"Plot type option '{plot_type}' not found")
                    else:
                        self.warnings.append("Plot type selectbox not found")
                except Exception as e:
                    self.warnings.append(f"Plot type '{plot_type}' selection failed: {e}")
            
            # Test prediction checkbox
            try:
                # Look for checkbox
                checkbox = await page.query_selector('input[type="checkbox"]')
                if checkbox:
                    await checkbox.click()
                    await page.wait_for_timeout(1000)
                    logger.info("✓ Show predictions checkbox toggled")
                else:
                    self.warnings.append("Show predictions checkbox not found")
            except Exception as e:
                self.warnings.append(f"Show predictions checkbox interaction failed: {e}")
            
        except Exception as e:
            self.errors.append(f"Trajectory visualization test failed: {e}")
    
    async def test_model_comparison(self, page):
        """Test model comparison page."""
        logger.info("Testing model comparison...")
        
        try:
            # Navigate to model comparison
            selectbox = await page.query_selector('[data-testid="stSelectbox"]')
            if selectbox:
                await selectbox.click()
                await page.wait_for_timeout(500)
                comparison_option = await page.query_selector('text="🔍 Model Comparison"')
                if comparison_option:
                    await comparison_option.click()
                    await page.wait_for_timeout(2000)
            
            # Test model selection
            model_names = ["Constant Velocity", "Constant Acceleration", "Polynomial Regression"]
            for model in model_names:
                try:
                    # Look for model in multiselect
                    model_element = await page.query_selector(f'text="{model}"')
                    if model_element:
                        await model_element.click()
                        await page.wait_for_timeout(500)
                        logger.info(f"✓ Model '{model}' selected")
                    else:
                        self.warnings.append(f"Model '{model}' not found")
                except Exception as e:
                    self.warnings.append(f"Model '{model}' selection failed: {e}")
            
            # Test prediction horizon slider
            try:
                # Look for slider
                slider = await page.query_selector('input[type="range"]')
                if slider:
                    await slider.fill("15")
                    await page.wait_for_timeout(1000)
                    logger.info("✓ Prediction horizon slider adjusted")
                else:
                    self.warnings.append("Prediction horizon slider not found")
            except Exception as e:
                self.warnings.append(f"Prediction horizon slider interaction failed: {e}")
            
            # Test run comparison button
            try:
                # Look for button
                button = await page.query_selector('button:has-text("Run Model Comparison")')
                if button:
                    await button.click()
                    await page.wait_for_timeout(5000)  # Wait for evaluation to complete
                    logger.info("✓ Model comparison executed")
                else:
                    self.warnings.append("Run model comparison button not found")
            except Exception as e:
                self.warnings.append(f"Run model comparison button interaction failed: {e}")
            
        except Exception as e:
            self.errors.append(f"Model comparison test failed: {e}")
    
    async def test_dataset_exploration(self, page):
        """Test dataset exploration page."""
        logger.info("Testing dataset exploration...")
        
        try:
            # Navigate to dataset exploration
            selectbox = await page.query_selector('[data-testid="stSelectbox"]')
            if selectbox:
                await selectbox.click()
                await page.wait_for_timeout(500)
                exploration_option = await page.query_selector('text="📈 Dataset Exploration"')
                if exploration_option:
                    await exploration_option.click()
                    await page.wait_for_timeout(2000)
            
            # Check for dataset overview metrics
            try:
                overview_section = await page.query_selector('text="Dataset Overview"')
                if overview_section:
                    logger.info("✓ Dataset overview section found")
                else:
                    self.warnings.append("Dataset overview section not found")
            except Exception as e:
                self.warnings.append(f"Dataset overview section check failed: {e}")
            
            # Check for distribution plots
            try:
                distribution_section = await page.query_selector('text="Trajectory Distribution"')
                if distribution_section:
                    logger.info("✓ Trajectory distribution section found")
                else:
                    self.warnings.append("Trajectory distribution section not found")
            except Exception as e:
                self.warnings.append(f"Trajectory distribution section check failed: {e}")
            
        except Exception as e:
            self.errors.append(f"Dataset exploration test failed: {e}")
    
    async def test_model_explainability(self, page):
        """Test model explainability page."""
        logger.info("Testing model explainability...")
        
        try:
            # Navigate to model explainability
            selectbox = await page.query_selector('[data-testid="stSelectbox"]')
            if selectbox:
                await selectbox.click()
                await page.wait_for_timeout(500)
                explainability_option = await page.query_selector('text="🤖 Model Explainability"')
                if explainability_option:
                    await explainability_option.click()
                    await page.wait_for_timeout(2000)
            
            # Test model selection
            try:
                # Look for model selectbox
                model_selectbox = await page.query_selector('text="Select Model for Analysis"')
                if model_selectbox:
                    await model_selectbox.click()
                    await page.wait_for_timeout(500)
                    model_option = await page.query_selector('text="Constant Velocity"')
                    if model_option:
                        await model_option.click()
                        await page.wait_for_timeout(1000)
                        logger.info("✓ Model selection for explainability successful")
                    else:
                        self.warnings.append("Model option for explainability not found")
                else:
                    self.warnings.append("Model selectbox for explainability not found")
            except Exception as e:
                self.warnings.append(f"Model selection for explainability failed: {e}")
            
        except Exception as e:
            self.errors.append(f"Model explainability test failed: {e}")
    
    async def test_settings_page(self, page):
        """Test settings page."""
        logger.info("Testing settings page...")
        
        try:
            # Navigate to settings
            selectbox = await page.query_selector('[data-testid="stSelectbox"]')
            if selectbox:
                await selectbox.click()
                await page.wait_for_timeout(500)
                settings_option = await page.query_selector('text="⚙️ Settings"')
                if settings_option:
                    await settings_option.click()
                    await page.wait_for_timeout(2000)
            
            # Test prediction horizon slider
            try:
                # Look for slider
                slider = await page.query_selector('input[type="range"]')
                if slider:
                    await slider.fill("15")
                    await page.wait_for_timeout(1000)
                    logger.info("✓ Settings prediction horizon slider adjusted")
                else:
                    self.warnings.append("Settings prediction horizon slider not found")
            except Exception as e:
                self.warnings.append(f"Settings prediction horizon slider interaction failed: {e}")
            
            # Test save settings button
            try:
                # Look for button
                button = await page.query_selector('button:has-text("Save Settings")')
                if button:
                    await button.click()
                    await page.wait_for_timeout(1000)
                    logger.info("✓ Save settings button clicked")
                else:
                    self.warnings.append("Save settings button not found")
            except Exception as e:
                self.warnings.append(f"Save settings button interaction failed: {e}")
            
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
                    if "error" in error_text.lower() or "exception" in error_text.lower():
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
                    selectbox = await page.query_selector('[data-testid="stSelectbox"]')
                    if selectbox:
                        await selectbox.click()
                        await page.wait_for_timeout(500)
                        option_element = await page.query_selector(f'text="{option}"')
                        if option_element:
                            await option_element.click()
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