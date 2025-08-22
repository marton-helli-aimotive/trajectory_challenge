#!/usr/bin/env python3
"""
Streamlit app for trajectory prediction dashboard.

Run with: streamlit run trajectory_dashboard_app.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ruff: noqa: E402
from src.trajectory_prediction.visualization.dashboard import TrajectoryDashboard

# Create and run dashboard
if __name__ == "__main__":
    dashboard = TrajectoryDashboard()
    dashboard.run()
