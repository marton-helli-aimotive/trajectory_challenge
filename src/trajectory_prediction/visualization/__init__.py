"""
Visualization and dashboard framework for trajectory prediction.

This module provides:
- Interactive Streamlit dashboard for trajectory visualization
- Real-time prediction visualization
- Model comparison interfaces
- Analysis tools (clustering, anomaly detection)
- Automated reporting system
- Custom dashboard creation
"""

from .dashboard import TrajectoryDashboard, DashboardConfig
from .components import (
    TrajectoryPlotter,
    PredictionVisualizer,
    ModelComparisonChart,
    PerformanceMetricsDisplay
)
from .analysis import (
    TrajectoryClusterAnalyzer,
    AnomalyDetector,
    FeatureImportanceVisualizer,
    ModelInterpretabilityTools
)
from .reports import (
    AutomatedReportGenerator,
    DataQualityReporter,
    ModelComparisonReporter
)

__all__ = [
    "TrajectoryDashboard",
    "DashboardConfig",
    "TrajectoryPlotter",
    "PredictionVisualizer", 
    "ModelComparisonChart",
    "PerformanceMetricsDisplay",
    "TrajectoryClusterAnalyzer",
    "AnomalyDetector",
    "FeatureImportanceVisualizer",
    "ModelInterpretabilityTools",
    "AutomatedReportGenerator",
    "DataQualityReporter",
    "ModelComparisonReporter"
]