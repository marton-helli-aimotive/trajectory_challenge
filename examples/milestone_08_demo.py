#!/usr/bin/env python3
"""
Milestone 8 Demo: Interactive Dashboard & Visualization

This script demonstrates the interactive dashboard with:
- Real-time model comparison
- Advanced trajectory visualization with animations
- Model explainability and feature importance
- Dataset exploration and analysis tools
- Performance monitoring dashboards
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ruff: noqa: E402
from src.trajectory_prediction.visualization.dashboard import TrajectoryDashboard


def run_milestone_08_demo():
    """Run interactive dashboard demonstration."""
    print("=" * 80)
    print("MILESTONE 8: INTERACTIVE DASHBOARD & VISUALIZATION DEMO")
    print("=" * 80)
    print()

    print("Starting Interactive Dashboard...")
    print("Dashboard Features:")
    print(
        "  - Model Comparison: Compare multiple trajectory prediction models side-by-side"
    )
    print("  - Trajectory Visualization: Interactive 2D/3D plots with animations")
    print("  - Dataset Exploration: Analyze trajectory dataset characteristics")
    print("  - Model Explainability: Feature importance and prediction breakdown")
    print("  - Performance Monitoring: Real-time metrics and evaluation")
    print("  - Real-time Prediction: Interactive parameter tuning")
    print()

    print("To access the dashboard:")
    print("1. Run this script")
    print("2. Open your web browser")
    print("3. Navigate to the URL shown below")
    print("4. Explore the different dashboard pages using the sidebar")
    print()

    # Initialize and run the dashboard
    dashboard = TrajectoryDashboard()

    print("🚀 Launching Streamlit Dashboard...")
    print("   Use Ctrl+C to stop the dashboard")
    print("=" * 80)

    try:
        dashboard.run()
    except KeyboardInterrupt:
        print("\n\nDashboard stopped by user.")
    except Exception as e:
        print(f"\nError running dashboard: {e}")
        print("Make sure you have all required dependencies installed:")
        print("  pip install streamlit plotly pandas numpy")


if __name__ == "__main__":
    run_milestone_08_demo()
