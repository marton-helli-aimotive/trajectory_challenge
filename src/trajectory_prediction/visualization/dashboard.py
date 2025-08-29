"""
Interactive Streamlit dashboard for trajectory prediction visualization.

This module provides a comprehensive web-based dashboard for:
- Trajectory data exploration and visualization
- Real-time prediction analysis
- Model comparison and performance monitoring
- Interactive parameter tuning
- System health monitoring
"""

import asyncio
import logging
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

try:
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    st = None

from omegaconf import DictConfig

from .components import TrajectoryPlotter, PredictionVisualizer, ModelComparisonChart
from .analysis import TrajectoryClusterAnalyzer, AnomalyDetector
from ..api.models import TrajectoryRequest, TrajectoryInput, TrajectoryPoint, PredictionConfig
from ..api.examples import TrajectoryPredictionClient
from ..data.schemas import TrajectoryData

logger = logging.getLogger(__name__)


@dataclass
class DashboardConfig:
    """Configuration for the trajectory dashboard."""
    title: str = "Trajectory Prediction Dashboard"
    api_base_url: str = "http://localhost:8000"
    refresh_interval: int = 5  # seconds
    max_trajectories_display: int = 100
    default_prediction_horizon: float = 10.0
    color_scheme: str = "plotly"
    enable_real_time: bool = True
    enable_model_comparison: bool = True
    enable_analysis_tools: bool = True


class TrajectoryDashboard:
    """
    Main Streamlit dashboard for trajectory prediction visualization.
    
    Provides interactive visualization and analysis capabilities.
    """
    
    def __init__(self, config: DashboardConfig):
        if not STREAMLIT_AVAILABLE:
            raise ImportError("Streamlit not available. Install with: pip install streamlit plotly")
        
        self.config = config
        self.api_client = None
        
        # Dashboard state
        self.current_trajectories: List[TrajectoryData] = []
        self.prediction_results: Dict[str, Any] = {}
        self.model_info: List[Dict[str, Any]] = []
        
        # Visualization components
        self.trajectory_plotter = TrajectoryPlotter()
        self.prediction_visualizer = PredictionVisualizer()
        self.model_comparison_chart = ModelComparisonChart()
        
        # Analysis tools
        self.cluster_analyzer = TrajectoryClusterAnalyzer()
        self.anomaly_detector = AnomalyDetector()
        
        logger.info("Trajectory dashboard initialized")
    
    async def initialize_api_client(self) -> bool:
        """Initialize API client connection."""
        
        try:
            self.api_client = TrajectoryPredictionClient(self.config.api_base_url)
            await self.api_client.connect()
            
            # Get model information
            self.model_info = await self.api_client.get_available_models()
            
            return True
            
        except Exception as e:
            st.error(f"Failed to connect to API: {e}")
            logger.error(f"API connection failed: {e}")
            return False
    
    def run_dashboard(self) -> None:
        """Run the main dashboard application."""
        
        # Configure Streamlit page
        st.set_page_config(
            page_title=self.config.title,
            page_icon="🚗",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Main title
        st.title(self.config.title)
        
        # Initialize API connection
        if self.api_client is None:
            with st.spinner("Connecting to API..."):
                try:
                    connected = asyncio.run(self.initialize_api_client())
                    if not connected:
                        st.error("Cannot connect to trajectory prediction API")
                        st.stop()
                except Exception as e:
                    st.error(f"Failed to initialize dashboard: {e}")
                    st.stop()
        
        # Sidebar navigation
        page = st.sidebar.selectbox(
            "Navigate",
            [
                "🏠 Overview",
                "🔮 Live Predictions", 
                "📊 Model Comparison",
                "🔍 Analysis Tools",
                "📈 Performance Monitor",
                "⚙️ Settings"
            ]
        )
        
        # Route to appropriate page
        if page == "🏠 Overview":
            self.render_overview_page()
        elif page == "🔮 Live Predictions":
            self.render_live_predictions_page()
        elif page == "📊 Model Comparison":
            self.render_model_comparison_page()
        elif page == "🔍 Analysis Tools":
            self.render_analysis_tools_page()
        elif page == "📈 Performance Monitor":
            self.render_performance_monitor_page()
        elif page == "⚙️ Settings":
            self.render_settings_page()
    
    def render_overview_page(self) -> None:
        """Render the overview/home page."""
        
        st.header("System Overview")
        
        # API Health Status
        col1, col2, col3, col4 = st.columns(4)
        
        try:
            health_status = asyncio.run(self.api_client.get_health_status())
            
            with col1:
                st.metric(
                    "API Status",
                    health_status.get("status", "Unknown").title(),
                    delta=None
                )
            
            with col2:
                uptime_hours = health_status.get("uptime_seconds", 0) / 3600
                st.metric(
                    "Uptime",
                    f"{uptime_hours:.1f}h",
                    delta=None
                )
            
            with col3:
                st.metric(
                    "Models Loaded",
                    health_status.get("models_loaded", 0),
                    delta=None
                )
            
            with col4:
                avg_response = health_status.get("average_response_time_ms", 0)
                st.metric(
                    "Avg Response",
                    f"{avg_response:.0f}ms",
                    delta=None
                )
            
            # Cache statistics
            if health_status.get("cache_hit_rate", 0) > 0:
                st.subheader("Cache Performance")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        "Cache Hit Rate",
                        f"{health_status.get('cache_hit_rate', 0):.1%}",
                        delta=None
                    )
                
                with col2:
                    st.metric(
                        "Total Predictions",
                        health_status.get("total_predictions", 0),
                        delta=None
                    )
        
        except Exception as e:
            st.error(f"Failed to fetch system status: {e}")
        
        # Model Information
        if self.model_info:
            st.subheader("Available Models")
            
            model_data = []
            for model in self.model_info:
                model_data.append({
                    "Name": model.get("model_name", "Unknown"),
                    "Type": model.get("model_type", "Unknown"),
                    "Version": model.get("model_version", "N/A"),
                    "Status": model.get("status", "Unknown"),
                    "Default": "✓" if model.get("is_default", False) else ""
                })
            
            st.dataframe(pd.DataFrame(model_data), use_container_width=True)
        
        # Recent Activity (placeholder)
        st.subheader("Recent Activity")
        
        # Generate sample activity data
        activity_data = {
            "Time": [datetime.now() - timedelta(minutes=i*5) for i in range(10, 0, -1)],
            "Event": [
                "Model prediction", "Batch processing", "Cache hit",
                "Model comparison", "Health check", "Model prediction",
                "Anomaly detected", "Performance alert", "Model prediction", "Cache miss"
            ],
            "Details": [
                "Vehicle_001 predicted", "5 trajectories processed", "Cache served prediction",
                "Models A vs B compared", "System healthy", "Vehicle_002 predicted",
                "Unusual trajectory pattern", "High latency detected", "Vehicle_003 predicted", "New prediction cached"
            ]
        }
        
        activity_df = pd.DataFrame(activity_data)
        st.dataframe(activity_df, use_container_width=True)
    
    def render_live_predictions_page(self) -> None:
        """Render the live predictions page."""
        
        st.header("Live Trajectory Predictions")
        
        # Prediction input section
        st.subheader("Input Trajectory")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Manual trajectory input
            input_method = st.radio(
                "Input Method",
                ["Draw Trajectory", "Upload Data", "Sample Data"]
            )
            
            if input_method == "Draw Trajectory":
                st.info("Interactive trajectory drawing coming soon!")
                # Placeholder for interactive drawing
                
            elif input_method == "Upload Data":
                uploaded_file = st.file_uploader(
                    "Upload trajectory data",
                    type=['csv', 'json'],
                    help="Upload CSV or JSON file with trajectory points"
                )
                
                if uploaded_file:
                    try:
                        if uploaded_file.name.endswith('.csv'):
                            df = pd.read_csv(uploaded_file)
                            st.write("Uploaded trajectory data:")
                            st.dataframe(df.head())
                        else:
                            data = json.load(uploaded_file)
                            st.write("Uploaded trajectory data:")
                            st.json(data)
                    except Exception as e:
                        st.error(f"Failed to load file: {e}")
            
            else:  # Sample Data
                sample_type = st.selectbox(
                    "Sample Trajectory Type",
                    ["Straight Line", "Curved Path", "Lane Change", "Zigzag"]
                )
                
                trajectory_data = self.generate_sample_trajectory(sample_type)
                self.current_trajectories = [trajectory_data]
                
                # Display trajectory
                fig = self.trajectory_plotter.plot_trajectory(trajectory_data)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Prediction parameters
            st.subheader("Prediction Parameters")
            
            prediction_horizon = st.slider(
                "Prediction Horizon (seconds)",
                min_value=1.0,
                max_value=30.0,
                value=self.config.default_prediction_horizon,
                step=0.5
            )
            
            time_resolution = st.slider(
                "Time Resolution (seconds)",
                min_value=0.01,
                max_value=1.0,
                value=0.1,
                step=0.01
            )
            
            model_name = st.selectbox(
                "Select Model",
                ["Default"] + [m.get("model_name", "") for m in self.model_info]
            )
            
            use_ensemble = st.checkbox(
                "Use Ensemble Prediction",
                value=False,
                help="Combine multiple models for improved accuracy"
            )
            
            uncertainty_type = st.selectbox(
                "Uncertainty Quantification",
                ["gaussian", "ensemble", "quantile", "none"]
            )
            
            # Predict button
            if st.button("🔮 Make Prediction", type="primary"):
                if self.current_trajectories:
                    with st.spinner("Making prediction..."):
                        self.make_prediction(
                            self.current_trajectories[0],
                            prediction_horizon,
                            time_resolution,
                            model_name if model_name != "Default" else None,
                            use_ensemble,
                            uncertainty_type
                        )
        
        # Prediction results
        if self.prediction_results:
            st.subheader("Prediction Results")
            self.display_prediction_results()
    
    def render_model_comparison_page(self) -> None:
        """Render the model comparison page."""
        
        st.header("Model Comparison")
        
        if len(self.model_info) < 2:
            st.warning("At least 2 models are required for comparison")
            return
        
        # Model selection
        col1, col2 = st.columns(2)
        
        with col1:
            models_to_compare = st.multiselect(
                "Select Models to Compare",
                [m.get("model_name", "") for m in self.model_info],
                default=[m.get("model_name", "") for m in self.model_info[:2]]
            )
        
        with col2:
            comparison_metrics = st.multiselect(
                "Comparison Metrics",
                ["ADE", "FDE", "Collision Risk", "Inference Time", "Safety Score"],
                default=["ADE", "FDE", "Collision Risk"]
            )
        
        if len(models_to_compare) >= 2:
            # Test trajectory input
            st.subheader("Test Trajectory")
            
            test_trajectory = self.generate_sample_trajectory("Curved Path")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = self.trajectory_plotter.plot_trajectory(test_trajectory)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if st.button("🔄 Run Comparison", type="primary"):
                    with st.spinner("Comparing models..."):
                        self.run_model_comparison(models_to_compare, test_trajectory)
            
            # Display comparison results
            if hasattr(self, 'comparison_results'):
                st.subheader("Comparison Results")
                self.display_model_comparison_results()
    
    def render_analysis_tools_page(self) -> None:
        """Render the analysis tools page."""
        
        st.header("Trajectory Analysis Tools")
        
        analysis_tool = st.selectbox(
            "Select Analysis Tool",
            ["Trajectory Clustering", "Anomaly Detection", "Feature Importance", "Model Interpretability"]
        )
        
        if analysis_tool == "Trajectory Clustering":
            self.render_clustering_analysis()
        elif analysis_tool == "Anomaly Detection":
            self.render_anomaly_detection()
        elif analysis_tool == "Feature Importance":
            self.render_feature_importance()
        elif analysis_tool == "Model Interpretability":
            self.render_model_interpretability()
    
    def render_clustering_analysis(self) -> None:
        """Render trajectory clustering analysis."""
        
        st.subheader("Trajectory Clustering Analysis")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Clustering parameters
            n_clusters = st.slider("Number of Clusters", 2, 10, 5)
            
            clustering_method = st.selectbox(
                "Clustering Method",
                ["K-Means", "DBSCAN", "Hierarchical"]
            )
            
            distance_metric = st.selectbox(
                "Distance Metric",
                ["DTW", "Euclidean", "Hausdorff"]
            )
            
            if st.button("🔍 Run Clustering"):
                with st.spinner("Performing clustering analysis..."):
                    # Generate sample trajectories for clustering
                    trajectories = self.generate_sample_trajectories_for_clustering()
                    
                    # Run clustering analysis
                    clustering_results = self.cluster_analyzer.analyze_trajectories(
                        trajectories, n_clusters, clustering_method
                    )
                    
                    st.session_state.clustering_results = clustering_results
        
        with col2:
            # Display clustering results
            if 'clustering_results' in st.session_state:
                fig = self.cluster_analyzer.plot_clustering_results(
                    st.session_state.clustering_results
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Cluster statistics
                st.write("Cluster Statistics:")
                stats_df = pd.DataFrame(st.session_state.clustering_results.get('cluster_stats', []))
                st.dataframe(stats_df, use_container_width=True)
    
    def render_anomaly_detection(self) -> None:
        """Render anomaly detection analysis."""
        
        st.subheader("Anomaly Detection")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Anomaly detection parameters
            detection_method = st.selectbox(
                "Detection Method",
                ["Isolation Forest", "One-Class SVM", "Local Outlier Factor"]
            )
            
            contamination_rate = st.slider(
                "Expected Contamination Rate",
                0.01, 0.3, 0.1,
                help="Expected proportion of anomalies"
            )
            
            if st.button("🚨 Detect Anomalies"):
                with st.spinner("Detecting anomalies..."):
                    # Generate mixed trajectories (normal + anomalous)
                    trajectories = self.generate_mixed_trajectories()
                    
                    # Run anomaly detection
                    anomaly_results = self.anomaly_detector.detect_anomalies(
                        trajectories, detection_method, contamination_rate
                    )
                    
                    st.session_state.anomaly_results = anomaly_results
        
        with col2:
            # Display anomaly detection results
            if 'anomaly_results' in st.session_state:
                results = st.session_state.anomaly_results
                
                fig = self.anomaly_detector.plot_anomaly_results(results)
                st.plotly_chart(fig, use_container_width=True)
                
                # Anomaly statistics
                n_anomalies = len([r for r in results.get('predictions', []) if r == -1])
                st.metric("Anomalies Detected", n_anomalies)
                
                # Show anomalous trajectories
                if n_anomalies > 0:
                    st.write("Anomalous Trajectories:")
                    anomaly_indices = [i for i, pred in enumerate(results.get('predictions', [])) if pred == -1]
                    for i, idx in enumerate(anomaly_indices[:5]):  # Show first 5
                        st.write(f"Trajectory {idx}: Anomaly Score = {results.get('scores', [])[idx]:.3f}")
    
    def render_feature_importance(self) -> None:
        """Render feature importance analysis."""
        
        st.subheader("Feature Importance Analysis")
        
        st.info("Feature importance visualization shows which trajectory characteristics are most important for prediction accuracy.")
        
        # Generate mock feature importance data
        features = [
            "Velocity Mean", "Velocity Std", "Acceleration Mean", "Acceleration Std",
            "Curvature Mean", "Heading Change", "Distance Traveled", "Time Duration"
        ]
        
        importance_scores = np.random.rand(len(features))
        importance_scores = importance_scores / importance_scores.sum()  # Normalize
        
        # Create feature importance plot
        fig = go.Figure(go.Bar(
            x=importance_scores,
            y=features,
            orientation='h',
            marker_color='lightblue'
        ))
        
        fig.update_layout(
            title="Feature Importance for Trajectory Prediction",
            xaxis_title="Importance Score",
            yaxis_title="Features",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature correlation matrix
        st.subheader("Feature Correlations")
        
        # Generate mock correlation data
        correlation_matrix = np.random.rand(len(features), len(features))
        correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2  # Make symmetric
        np.fill_diagonal(correlation_matrix, 1)  # Diagonal = 1
        
        fig_corr = go.Figure(data=go.Heatmap(
            z=correlation_matrix,
            x=features,
            y=features,
            colorscale='RdBu',
            zmid=0
        ))
        
        fig_corr.update_layout(
            title="Feature Correlation Matrix",
            height=400
        )
        
        st.plotly_chart(fig_corr, use_container_width=True)
    
    def render_model_interpretability(self) -> None:
        """Render model interpretability tools."""
        
        st.subheader("Model Interpretability")
        
        interpretability_method = st.selectbox(
            "Interpretability Method",
            ["SHAP Values", "LIME", "Attention Weights", "Gradient Analysis"]
        )
        
        if interpretability_method == "SHAP Values":
            st.info("SHAP (SHapley Additive exPlanations) shows how each feature contributes to individual predictions.")
            
            # Mock SHAP visualization
            fig = go.Figure()
            
            # Generate mock SHAP values
            features = ["Velocity", "Acceleration", "Curvature", "Heading", "Position X", "Position Y"]
            shap_values = np.random.randn(len(features))
            colors = ['red' if x < 0 else 'blue' for x in shap_values]
            
            fig.add_trace(go.Bar(
                x=shap_values,
                y=features,
                orientation='h',
                marker_color=colors,
                name='SHAP Values'
            ))
            
            fig.update_layout(
                title="SHAP Feature Importance for Sample Prediction",
                xaxis_title="SHAP Value (Impact on Prediction)",
                yaxis_title="Features",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        elif interpretability_method == "Attention Weights":
            st.info("Attention weights show which parts of the trajectory the model focuses on most.")
            
            # Mock attention visualization
            trajectory_steps = list(range(1, 11))
            attention_weights = np.random.rand(10)
            attention_weights = attention_weights / attention_weights.sum()
            
            fig = go.Figure(go.Scatter(
                x=trajectory_steps,
                y=attention_weights,
                mode='lines+markers',
                line=dict(width=4),
                marker=dict(size=10),
                name='Attention Weight'
            ))
            
            fig.update_layout(
                title="Attention Weights Across Trajectory Steps",
                xaxis_title="Trajectory Step",
                yaxis_title="Attention Weight",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def render_performance_monitor_page(self) -> None:
        """Render the performance monitoring page."""
        
        st.header("Performance Monitor")
        
        # Real-time metrics
        if st.button("🔄 Refresh Metrics"):
            with st.spinner("Fetching metrics..."):
                try:
                    health_status = asyncio.run(self.api_client.get_health_status())
                    st.session_state.health_data = health_status
                except Exception as e:
                    st.error(f"Failed to fetch metrics: {e}")
        
        if 'health_data' in st.session_state:
            health_data = st.session_state.health_data
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Response Time",
                    f"{health_data.get('average_response_time_ms', 0):.0f}ms"
                )
            
            with col2:
                st.metric(
                    "Throughput",
                    f"{health_data.get('total_predictions', 0) / (health_data.get('uptime_seconds', 1) / 3600):.1f}/hr"
                )
            
            with col3:
                st.metric(
                    "Cache Hit Rate",
                    f"{health_data.get('cache_hit_rate', 0):.1%}"
                )
            
            with col4:
                cpu_usage = health_data.get('system_info', {}).get('cpu_usage', 0)
                st.metric(
                    "System Load",
                    f"{cpu_usage:.1f}%" if cpu_usage > 0 else "N/A"
                )
        
        # Performance trends (mock data)
        st.subheader("Performance Trends")
        
        # Generate mock time series data
        timestamps = pd.date_range(
            start=datetime.now() - timedelta(hours=24),
            end=datetime.now(),
            freq='1H'
        )
        
        response_times = 50 + 20 * np.random.randn(len(timestamps)).cumsum()
        throughput = 100 + 10 * np.random.randn(len(timestamps)).cumsum()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_response = go.Figure()
            fig_response.add_trace(go.Scatter(
                x=timestamps,
                y=response_times,
                mode='lines',
                name='Response Time',
                line=dict(color='blue')
            ))
            
            fig_response.update_layout(
                title="Response Time Trend",
                xaxis_title="Time",
                yaxis_title="Response Time (ms)",
                height=300
            )
            
            st.plotly_chart(fig_response, use_container_width=True)
        
        with col2:
            fig_throughput = go.Figure()
            fig_throughput.add_trace(go.Scatter(
                x=timestamps,
                y=throughput,
                mode='lines',
                name='Throughput',
                line=dict(color='green')
            ))
            
            fig_throughput.update_layout(
                title="Throughput Trend",
                xaxis_title="Time", 
                yaxis_title="Requests/Hour",
                height=300
            )
            
            st.plotly_chart(fig_throughput, use_container_width=True)
    
    def render_settings_page(self) -> None:
        """Render the settings configuration page."""
        
        st.header("Dashboard Settings")
        
        # API settings
        st.subheader("API Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            new_api_url = st.text_input(
                "API Base URL",
                value=self.config.api_base_url,
                help="Base URL for the trajectory prediction API"
            )
            
            refresh_interval = st.slider(
                "Refresh Interval (seconds)",
                min_value=1,
                max_value=60,
                value=self.config.refresh_interval
            )
        
        with col2:
            max_trajectories = st.number_input(
                "Max Trajectories Display",
                min_value=10,
                max_value=1000,
                value=self.config.max_trajectories_display
            )
            
            color_scheme = st.selectbox(
                "Color Scheme",
                ["plotly", "plotly_white", "plotly_dark", "ggplot2"]
            )
        
        # Feature toggles
        st.subheader("Feature Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            enable_real_time = st.checkbox(
                "Enable Real-time Updates",
                value=self.config.enable_real_time
            )
            
            enable_model_comparison = st.checkbox(
                "Enable Model Comparison",
                value=self.config.enable_model_comparison
            )
        
        with col2:
            enable_analysis_tools = st.checkbox(
                "Enable Analysis Tools",
                value=self.config.enable_analysis_tools
            )
            
            enable_caching = st.checkbox(
                "Enable Dashboard Caching",
                value=True
            )
        
        # Save settings
        if st.button("💾 Save Settings"):
            # Update configuration
            self.config.api_base_url = new_api_url
            self.config.refresh_interval = refresh_interval
            self.config.max_trajectories_display = max_trajectories
            self.config.color_scheme = color_scheme
            self.config.enable_real_time = enable_real_time
            self.config.enable_model_comparison = enable_model_comparison
            self.config.enable_analysis_tools = enable_analysis_tools
            
            st.success("Settings saved successfully!")
            
            # Reconnect to API if URL changed
            if new_api_url != self.config.api_base_url:
                st.info("Reconnecting to API with new URL...")
                self.api_client = None
                st.experimental_rerun()
    
    def generate_sample_trajectory(self, trajectory_type: str) -> TrajectoryData:
        """Generate sample trajectory data for testing."""
        
        if trajectory_type == "Straight Line":
            positions = [[i * 0.5, 0.0] for i in range(10)]
        elif trajectory_type == "Curved Path":
            positions = [[i * 0.5, 0.1 * i * i] for i in range(10)]
        elif trajectory_type == "Lane Change":
            positions = [[i * 0.5, 2.0 if i > 5 else 0.0] for i in range(10)]
        else:  # Zigzag
            positions = [[i * 0.5, 1.0 * (i % 2)] for i in range(10)]
        
        time_steps = [i * 0.2 for i in range(10)]
        
        return TrajectoryData(
            vehicle_id=f"sample_{trajectory_type.lower().replace(' ', '_')}",
            positions=positions,
            time_steps=time_steps,
            metadata={"type": trajectory_type, "generated": True}
        )
    
    def generate_sample_trajectories_for_clustering(self) -> List[TrajectoryData]:
        """Generate sample trajectories for clustering analysis."""
        
        trajectories = []
        trajectory_types = ["Straight Line", "Curved Path", "Lane Change", "Zigzag"]
        
        for i in range(20):
            traj_type = trajectory_types[i % len(trajectory_types)]
            traj = self.generate_sample_trajectory(traj_type)
            traj.vehicle_id = f"cluster_sample_{i}"
            trajectories.append(traj)
        
        return trajectories
    
    def generate_mixed_trajectories(self) -> List[TrajectoryData]:
        """Generate mixed normal and anomalous trajectories."""
        
        trajectories = []
        
        # Normal trajectories
        for i in range(15):
            if i % 3 == 0:
                traj = self.generate_sample_trajectory("Straight Line")
            elif i % 3 == 1:
                traj = self.generate_sample_trajectory("Curved Path")
            else:
                traj = self.generate_sample_trajectory("Lane Change")
            
            traj.vehicle_id = f"normal_{i}"
            trajectories.append(traj)
        
        # Anomalous trajectories
        for i in range(5):
            # Generate unusual trajectories
            if i % 2 == 0:
                # Erratic movement
                positions = [[j * 0.3 + np.random.randn() * 0.5, np.random.randn() * 2] for j in range(8)]
            else:
                # Sudden direction change
                positions = [[j * 0.5, 0] if j < 4 else [j * 0.5, (j-4) * 1.5] for j in range(8)]
            
            time_steps = [j * 0.2 for j in range(8)]
            
            anomaly_traj = TrajectoryData(
                vehicle_id=f"anomaly_{i}",
                positions=positions,
                time_steps=time_steps,
                metadata={"anomaly": True}
            )
            trajectories.append(anomaly_traj)
        
        return trajectories
    
    def make_prediction(
        self,
        trajectory: TrajectoryData,
        prediction_horizon: float,
        time_resolution: float,
        model_name: Optional[str],
        use_ensemble: bool,
        uncertainty_type: str
    ) -> None:
        """Make trajectory prediction using the API."""
        
        try:
            # Convert to API format
            trajectory_points = [
                TrajectoryPoint(x=pos[0], y=pos[1], timestamp=t)
                for pos, t in zip(trajectory.positions, trajectory.time_steps)
            ]
            
            trajectory_input = TrajectoryInput(
                vehicle_id=trajectory.vehicle_id,
                trajectory_points=trajectory_points,
                metadata=trajectory.metadata
            )
            
            config = PredictionConfig(
                prediction_horizon=prediction_horizon,
                time_resolution=time_resolution,
                uncertainty_quantification=uncertainty_type
            )
            
            request = TrajectoryRequest(
                trajectory=trajectory_input,
                config=config,
                model_name=model_name,
                use_ensemble=use_ensemble
            )
            
            # Make prediction
            response = asyncio.run(
                self.api_client.predict_single_trajectory(
                    trajectory.vehicle_id,
                    trajectory.positions,
                    trajectory.time_steps,
                    prediction_horizon,
                    model_name,
                    use_ensemble
                )
            )
            
            self.prediction_results = {
                "request": request,
                "response": response,
                "original_trajectory": trajectory
            }
            
            st.success("Prediction completed successfully!")
            
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            logger.error(f"Prediction error: {e}")
    
    def display_prediction_results(self) -> None:
        """Display prediction results with visualization."""
        
        if not self.prediction_results:
            return
        
        response = self.prediction_results["response"]
        original_trajectory = self.prediction_results["original_trajectory"]
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Prediction Status",
                response.status.title()
            )
        
        with col2:
            st.metric(
                "Model Used",
                response.metadata.model_name
            )
        
        with col3:
            st.metric(
                "Inference Time",
                f"{response.metadata.inference_time_ms:.1f}ms"
            )
        
        # Visualization
        fig = self.prediction_visualizer.plot_prediction_results(
            original_trajectory,
            response.predicted_trajectory if response.predicted_trajectory else [],
            response.metadata
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional metrics if available
        if hasattr(response.metadata, 'safety_score') and response.metadata.safety_score:
            st.metric(
                "Safety Score",
                f"{response.metadata.safety_score:.2f}"
            )
        
        # Warnings
        if response.warnings:
            st.warning("Warnings:")
            for warning in response.warnings:
                st.write(f"• {warning}")
    
    def run_model_comparison(self, models: List[str], test_trajectory: TrajectoryData) -> None:
        """Run model comparison analysis."""
        
        # This would integrate with the actual model comparison system
        # For now, generate mock comparison results
        
        comparison_results = {
            "models": models,
            "test_trajectory": test_trajectory,
            "results": {}
        }
        
        for model in models:
            # Mock results for each model
            comparison_results["results"][model] = {
                "ade": np.random.uniform(0.5, 2.0),
                "fde": np.random.uniform(1.0, 4.0),
                "collision_risk": np.random.uniform(0.01, 0.1),
                "inference_time_ms": np.random.uniform(20, 100),
                "safety_score": np.random.uniform(0.7, 0.95)
            }
        
        self.comparison_results = comparison_results
    
    def display_model_comparison_results(self) -> None:
        """Display model comparison results."""
        
        if not hasattr(self, 'comparison_results'):
            return
        
        results = self.comparison_results["results"]
        
        # Create comparison dataframe
        comparison_data = []
        for model_name, metrics in results.items():
            comparison_data.append({
                "Model": model_name,
                "ADE": f"{metrics['ade']:.3f}",
                "FDE": f"{metrics['fde']:.3f}",
                "Collision Risk": f"{metrics['collision_risk']:.3f}",
                "Inference Time": f"{metrics['inference_time_ms']:.1f}ms",
                "Safety Score": f"{metrics['safety_score']:.3f}"
            })
        
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
        
        # Comparison charts
        fig = self.model_comparison_chart.create_comparison_chart(results)
        st.plotly_chart(fig, use_container_width=True)


def run_dashboard(config: Optional[DashboardConfig] = None) -> None:
    """Run the Streamlit dashboard application."""
    
    if config is None:
        config = DashboardConfig()
    
    dashboard = TrajectoryDashboard(config)
    dashboard.run_dashboard()


if __name__ == "__main__":
    # Run dashboard with default configuration
    run_dashboard()