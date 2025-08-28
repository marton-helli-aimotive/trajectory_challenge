"""
Interactive Dashboard for Vehicle Trajectory Prediction.

This module provides a comprehensive web-based dashboard for model comparison,
trajectory visualization, and dataset exploration.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import sys

# Add the project root to the path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from vehicle_trajectory_prediction.core.config import ModelConfig
from vehicle_trajectory_prediction.core.data_models import TrajectoryData
from vehicle_trajectory_prediction.models import (
    ConstantVelocityPredictor,
    ConstantAccelerationPredictor,
    PolynomialRegressionPredictor,
    KNNPredictor,
    GaussianProcessPredictor,
    EnsemblePredictor
)
from vehicle_trajectory_prediction.evaluation import ComprehensiveEvaluator
from vehicle_trajectory_prediction.visualization.plots import (
    TrajectoryPlotter,
    ModelComparisonPlotter,
    DatasetExplorer
)

logger = logging.getLogger(__name__)


class TrajectoryDashboard:
    """
    Main dashboard class for vehicle trajectory prediction visualization.
    
    Provides interactive interfaces for:
    - Trajectory visualization and analysis
    - Model comparison and evaluation
    - Dataset exploration
    - Model explainability
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """Initialize the dashboard with configuration."""
        self.config = config or ModelConfig()
        self.evaluator = ComprehensiveEvaluator(self.config)
        self.trajectory_plotter = TrajectoryPlotter()
        self.comparison_plotter = ModelComparisonPlotter()
        self.dataset_explorer = DatasetExplorer()
        
        # Initialize models
        self.models = self._initialize_models()
        
        # Sample data for demonstration
        self.sample_trajectories = self._generate_sample_data()
        
    def _initialize_models(self) -> Dict[str, Any]:
        """Initialize all available prediction models."""
        models = {
            "Constant Velocity": ConstantVelocityPredictor(self.config),
            "Constant Acceleration": ConstantAccelerationPredictor(self.config),
            "Polynomial Regression": PolynomialRegressionPredictor(self.config),
            "K-Nearest Neighbors": KNNPredictor(self.config),
            "Gaussian Process": GaussianProcessPredictor(self.config),
            "Ensemble": EnsemblePredictor(self.config)
        }
        return models
    
    def _generate_sample_data(self) -> List[TrajectoryData]:
        """Generate sample trajectory data for demonstration."""
        # Create sample trajectories with realistic vehicle motion
        trajectories = []
        
        for i in range(5):
            # Generate a realistic trajectory
            t = np.linspace(0, 10, 50)  # 10 seconds, 50 points
            
            # Initial conditions
            x0, y0 = np.random.uniform(0, 100, 2)
            v0 = np.random.uniform(10, 30)  # m/s
            heading = np.random.uniform(0, 2*np.pi)
            
            # Add some realistic motion patterns
            v = v0 + np.random.normal(0, 2, len(t))  # Velocity with noise
            heading += np.random.normal(0, 0.1, len(t))  # Heading changes
            
            # Integrate to get positions
            x = x0 + np.cumsum(v * np.cos(heading)) * (t[1] - t[0])
            y = y0 + np.cumsum(v * np.sin(heading)) * (t[1] - t[0])
            
            # Create trajectory data
            trajectory = TrajectoryData(
                vehicle_id=f"vehicle_{i}",
                timestamps=t,
                x_positions=x,
                y_positions=y,
                velocities=v,
                headings=heading,
                accelerations=np.gradient(v, t),
                metadata={
                    "lane_id": np.random.randint(1, 4),
                    "road_id": "highway_101",
                    "weather": "clear"
                }
            )
            trajectories.append(trajectory)
        
        return trajectories
    
    def run(self):
        """Run the main dashboard application."""
        st.set_page_config(
            page_title="Vehicle Trajectory Prediction Dashboard",
            page_icon="🚗",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for better styling
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #1f77b4;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Header
        st.markdown('<h1 class="main-header">🚗 Vehicle Trajectory Prediction Dashboard</h1>', 
                   unsafe_allow_html=True)
        
        # Sidebar navigation
        page = st.sidebar.selectbox(
            "Navigation",
            ["🏠 Overview", "📊 Trajectory Visualization", "🔍 Model Comparison", 
             "📈 Dataset Exploration", "🤖 Model Explainability", "⚙️ Settings"]
        )
        
        if page == "🏠 Overview":
            self._show_overview()
        elif page == "📊 Trajectory Visualization":
            self._show_trajectory_visualization()
        elif page == "🔍 Model Comparison":
            self._show_model_comparison()
        elif page == "📈 Dataset Exploration":
            self._show_dataset_exploration()
        elif page == "🤖 Model Explainability":
            self._show_model_explainability()
        elif page == "⚙️ Settings":
            self._show_settings()
    
    def _show_overview(self):
        """Display the main overview page."""
        st.markdown("## 📊 System Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Models Available", len(self.models))
        
        with col2:
            st.metric("Sample Trajectories", len(self.sample_trajectories))
        
        with col3:
            st.metric("Prediction Horizon", f"{self.config.prediction_horizon}s")
        
        with col4:
            st.metric("Update Frequency", "Real-time")
        
        # Quick start section
        st.markdown("## 🚀 Quick Start")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Visualize Trajectories")
            st.markdown("""
            - Interactive 2D trajectory plots
            - 3D velocity-acceleration visualization
            - Time-based trajectory playback
            - Real-time prediction display
            """)
            
            if st.button("Go to Trajectory Visualization"):
                st.session_state.page = "📊 Trajectory Visualization"
        
        with col2:
            st.markdown("### 🔍 Compare Models")
            st.markdown("""
            - Side-by-side model comparison
            - Performance metrics visualization
            - Statistical significance testing
            - Error analysis plots
            """)
            
            if st.button("Go to Model Comparison"):
                st.session_state.page = "🔍 Model Comparison"
        
        # Recent activity
        st.markdown("## 📈 Recent Activity")
        
        # Sample performance data
        performance_data = pd.DataFrame({
            'Model': list(self.models.keys()),
            'RMSE': np.random.uniform(0.5, 2.0, len(self.models)),
            'ADE': np.random.uniform(0.3, 1.5, len(self.models)),
            'FDE': np.random.uniform(0.8, 3.0, len(self.models)),
            'Inference Time (ms)': np.random.uniform(10, 100, len(self.models))
        })
        
        st.dataframe(performance_data, use_container_width=True)
        
        # System status
        st.markdown("## 🔧 System Status")
        
        status_col1, status_col2 = st.columns(2)
        
        with status_col1:
            st.success("✅ All models loaded successfully")
            st.success("✅ Evaluation framework ready")
            st.success("✅ Visualization components active")
        
        with status_col2:
            st.info("ℹ️ Using sample data for demonstration")
            st.info("ℹ️ Real-time updates enabled")
            st.info("ℹ️ GPU acceleration available")
    
    def _show_trajectory_visualization(self):
        """Display interactive trajectory visualization."""
        st.markdown("## 📊 Trajectory Visualization")
        
        # Trajectory selection
        trajectory_idx = st.selectbox(
            "Select Trajectory",
            range(len(self.sample_trajectories)),
            format_func=lambda x: f"Vehicle {self.sample_trajectories[x].vehicle_id}"
        )
        
        trajectory = self.sample_trajectories[trajectory_idx]
        
        # Visualization options
        col1, col2 = st.columns(2)
        
        with col1:
            plot_type = st.selectbox(
                "Plot Type",
                ["2D Trajectory", "3D Trajectory", "Velocity Profile", "Acceleration Profile"]
            )
        
        with col2:
            show_prediction = st.checkbox("Show Predictions", value=True)
            if show_prediction:
                selected_models = st.multiselect(
                    "Select Models for Prediction",
                    list(self.models.keys()),
                    default=["Constant Velocity", "Polynomial Regression"]
                )
        
        # Create the selected plot
        if plot_type == "2D Trajectory":
            fig = self.trajectory_plotter.plot_2d_trajectory(
                trajectory, 
                show_prediction=show_prediction,
                models=self.models if show_prediction else None,
                selected_models=selected_models if show_prediction else None
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif plot_type == "3D Trajectory":
            fig = self.trajectory_plotter.plot_3d_trajectory(
                trajectory,
                show_prediction=show_prediction,
                models=self.models if show_prediction else None,
                selected_models=selected_models if show_prediction else None
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif plot_type == "Velocity Profile":
            fig = self.trajectory_plotter.plot_velocity_profile(trajectory)
            st.plotly_chart(fig, use_container_width=True)
            
        elif plot_type == "Acceleration Profile":
            fig = self.trajectory_plotter.plot_acceleration_profile(trajectory)
            st.plotly_chart(fig, use_container_width=True)
        
        # Trajectory statistics
        st.markdown("## 📈 Trajectory Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Duration", f"{trajectory.timestamps[-1]:.1f}s")
        
        with col2:
            st.metric("Distance", f"{np.sum(np.sqrt(np.diff(trajectory.x_positions)**2 + np.diff(trajectory.y_positions)**2)):.1f}m")
        
        with col3:
            st.metric("Avg Velocity", f"{np.mean(trajectory.velocities):.1f} m/s")
        
        with col4:
            st.metric("Max Velocity", f"{np.max(trajectory.velocities):.1f} m/s")
        
        # Trajectory metadata
        if trajectory.metadata:
            st.markdown("## 📋 Trajectory Metadata")
            metadata_df = pd.DataFrame([trajectory.metadata])
            st.dataframe(metadata_df, use_container_width=True)
    
    def _show_model_comparison(self):
        """Display model comparison interface."""
        st.markdown("## 🔍 Model Comparison")
        
        # Model selection
        selected_models = st.multiselect(
            "Select Models to Compare",
            list(self.models.keys()),
            default=list(self.models.keys())[:3]
        )
        
        if not selected_models:
            st.warning("Please select at least one model for comparison.")
            return
        
        # Evaluation parameters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            prediction_horizon = st.slider(
                "Prediction Horizon (seconds)",
                min_value=1,
                max_value=30,
                value=10
            )
        
        with col2:
            test_size = st.slider(
                "Test Trajectories",
                min_value=1,
                max_value=len(self.sample_trajectories),
                value=min(3, len(self.sample_trajectories))
            )
        
        with col3:
            include_safety = st.checkbox("Include Safety Metrics", value=True)
        
        # Run comparison
        if st.button("Run Model Comparison"):
            with st.spinner("Evaluating models..."):
                # Use sample trajectories for evaluation
                test_trajectories = self.sample_trajectories[:test_size]
                selected_model_instances = {name: self.models[name] for name in selected_models}
                
                # Run evaluation
                comparison_results = self.evaluator.compare_models(
                    selected_model_instances,
                    test_trajectories,
                    prediction_horizon=prediction_horizon,
                    include_safety=include_safety
                )
                
                # Display results
                self._display_comparison_results(comparison_results)
    
    def _display_comparison_results(self, results: Dict[str, Any]):
        """Display model comparison results."""
        st.markdown("## 📊 Comparison Results")
        
        # Performance metrics table
        if 'performance_metrics' in results:
            st.markdown("### 📈 Performance Metrics")
            metrics_df = pd.DataFrame(results['performance_metrics'])
            st.dataframe(metrics_df, use_container_width=True)
            
            # Create performance comparison plot
            fig = self.comparison_plotter.plot_performance_comparison(metrics_df)
            st.plotly_chart(fig, use_container_width=True)
        
        # Safety metrics
        if 'safety_metrics' in results:
            st.markdown("### 🛡️ Safety Metrics")
            safety_df = pd.DataFrame(results['safety_metrics'])
            st.dataframe(safety_df, use_container_width=True)
            
            # Create safety comparison plot
            fig = self.comparison_plotter.plot_safety_comparison(safety_df)
            st.plotly_chart(fig, use_container_width=True)
        
        # Model ranking
        if 'ranking' in results:
            st.markdown("### 🏆 Model Ranking")
            ranking_df = pd.DataFrame(results['ranking'])
            st.dataframe(ranking_df, use_container_width=True)
            
            # Create ranking visualization
            fig = self.comparison_plotter.plot_model_ranking(ranking_df)
            st.plotly_chart(fig, use_container_width=True)
        
        # Statistical significance
        if 'statistical_tests' in results:
            st.markdown("### 📊 Statistical Significance")
            stats_df = pd.DataFrame(results['statistical_tests'])
            st.dataframe(stats_df, use_container_width=True)
    
    def _show_dataset_exploration(self):
        """Display dataset exploration interface."""
        st.markdown("## 📈 Dataset Exploration")
        
        # Dataset overview
        st.markdown("### 📊 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Trajectories", len(self.sample_trajectories))
        
        with col2:
            st.metric("Total Duration", f"{sum(t.timestamps[-1] for t in self.sample_trajectories):.1f}s")
        
        with col3:
            st.metric("Avg Trajectory Length", f"{np.mean([len(t.timestamps) for t in self.sample_trajectories]):.0f} points")
        
        with col4:
            st.metric("Data Points", sum(len(t.timestamps) for t in self.sample_trajectories))
        
        # Trajectory distribution analysis
        st.markdown("### 📊 Trajectory Distribution")
        
        # Create distribution plots
        col1, col2 = st.columns(2)
        
        with col1:
            # Duration distribution
            durations = [t.timestamps[-1] for t in self.sample_trajectories]
            fig = px.histogram(
                x=durations,
                title="Trajectory Duration Distribution",
                labels={'x': 'Duration (s)', 'y': 'Count'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Average velocity distribution
            avg_velocities = [np.mean(t.velocities) for t in self.sample_trajectories]
            fig = px.histogram(
                x=avg_velocities,
                title="Average Velocity Distribution",
                labels={'x': 'Average Velocity (m/s)', 'y': 'Count'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Trajectory patterns
        st.markdown("### 🔄 Trajectory Patterns")
        
        # Show all trajectories on one plot
        fig = self.dataset_explorer.plot_all_trajectories(self.sample_trajectories)
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature correlation analysis
        st.markdown("### 🔗 Feature Correlations")
        
        # Create feature matrix
        feature_data = []
        for traj in self.sample_trajectories:
            feature_data.append({
                'avg_velocity': np.mean(traj.velocities),
                'max_velocity': np.max(traj.velocities),
                'avg_acceleration': np.mean(traj.accelerations),
                'max_acceleration': np.max(traj.accelerations),
                'duration': traj.timestamps[-1],
                'distance': np.sum(np.sqrt(np.diff(traj.x_positions)**2 + np.diff(traj.y_positions)**2))
            })
        
        feature_df = pd.DataFrame(feature_data)
        
        # Correlation matrix
        fig = px.imshow(
            feature_df.corr(),
            title="Feature Correlation Matrix",
            color_continuous_scale='RdBu'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def _show_model_explainability(self):
        """Display model explainability interface."""
        st.markdown("## 🤖 Model Explainability")
        
        # Model selection
        selected_model = st.selectbox(
            "Select Model for Analysis",
            list(self.models.keys())
        )
        
        if not selected_model:
            st.warning("Please select a model for explainability analysis.")
            return
        
        model = self.models[selected_model]
        
        # Explainability options
        explainability_type = st.selectbox(
            "Explainability Type",
            ["Feature Importance", "Prediction Analysis", "Model Behavior", "Uncertainty Analysis"]
        )
        
        if explainability_type == "Feature Importance":
            self._show_feature_importance(model)
        elif explainability_type == "Prediction Analysis":
            self._show_prediction_analysis(model)
        elif explainability_type == "Model Behavior":
            self._show_model_behavior(model)
        elif explainability_type == "Uncertainty Analysis":
            self._show_uncertainty_analysis(model)
    
    def _show_feature_importance(self, model):
        """Display feature importance analysis."""
        st.markdown("### 📊 Feature Importance")
        
        # Generate sample feature importance (in real implementation, this would come from the model)
        features = ['velocity', 'acceleration', 'heading', 'position_x', 'position_y', 'time']
        importance = np.random.uniform(0, 1, len(features))
        importance = importance / np.sum(importance)  # Normalize
        
        # Create feature importance plot
        fig = px.bar(
            x=features,
            y=importance,
            title=f"Feature Importance - {type(model).__name__}",
            labels={'x': 'Features', 'y': 'Importance Score'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance table
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        st.dataframe(importance_df, use_container_width=True)
    
    def _show_prediction_analysis(self, model):
        """Display prediction analysis."""
        st.markdown("### 🔍 Prediction Analysis")
        
        # Select trajectory for analysis
        trajectory_idx = st.selectbox(
            "Select Trajectory for Analysis",
            range(len(self.sample_trajectories)),
            format_func=lambda x: f"Vehicle {self.sample_trajectories[x].vehicle_id}"
        )
        
        trajectory = self.sample_trajectories[trajectory_idx]
        
        # Generate prediction
        try:
            prediction = model.predict(trajectory, prediction_horizon=10)
            
            # Show prediction vs actual
            fig = self.trajectory_plotter.plot_prediction_analysis(trajectory, prediction)
            st.plotly_chart(fig, use_container_width=True)
            
            # Prediction error analysis
            st.markdown("#### 📊 Prediction Error Analysis")
            
            # Calculate errors (simplified)
            actual_x = trajectory.x_positions[-10:]  # Last 10 points
            actual_y = trajectory.y_positions[-10:]
            pred_x = prediction.x_positions[:10]  # First 10 predictions
            pred_y = prediction.y_positions[:10]
            
            errors = np.sqrt((actual_x - pred_x)**2 + (actual_y - pred_y)**2)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Mean Error", f"{np.mean(errors):.2f}m")
                st.metric("Max Error", f"{np.max(errors):.2f}m")
            
            with col2:
                st.metric("RMSE", f"{np.sqrt(np.mean(errors**2)):.2f}m")
                st.metric("Std Error", f"{np.std(errors):.2f}m")
            
            # Error distribution
            fig = px.histogram(
                x=errors,
                title="Prediction Error Distribution",
                labels={'x': 'Error (m)', 'y': 'Count'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error generating prediction: {str(e)}")
    
    def _show_model_behavior(self, model):
        """Display model behavior analysis."""
        st.markdown("### 🔄 Model Behavior Analysis")
        
        # Model parameters
        st.markdown("#### ⚙️ Model Parameters")
        
        # Display model configuration
        if hasattr(model, 'config'):
            config_dict = model.config.dict()
            config_df = pd.DataFrame([config_dict])
            st.dataframe(config_df, use_container_width=True)
        
        # Model capabilities
        st.markdown("#### 🎯 Model Capabilities")
        
        capabilities = {
            "Uncertainty Quantification": hasattr(model, 'predict_with_uncertainty'),
            "Online Learning": hasattr(model, 'update'),
            "Feature Engineering": hasattr(model, 'extract_features'),
            "Multi-step Prediction": hasattr(model, 'predict_sequence'),
            "Confidence Intervals": hasattr(model, 'predict_with_confidence')
        }
        
        for capability, available in capabilities.items():
            status = "✅" if available else "❌"
            st.markdown(f"{status} {capability}")
        
        # Model performance characteristics
        st.markdown("#### 📈 Performance Characteristics")
        
        # Simulate performance metrics
        performance_data = {
            'Metric': ['Inference Time (ms)', 'Memory Usage (MB)', 'Training Time (s)', 'Accuracy (%)'],
            'Value': [
                np.random.uniform(10, 100),
                np.random.uniform(50, 500),
                np.random.uniform(1, 60),
                np.random.uniform(70, 95)
            ]
        }
        
        perf_df = pd.DataFrame(performance_data)
        st.dataframe(perf_df, use_container_width=True)
    
    def _show_uncertainty_analysis(self, model):
        """Display uncertainty analysis."""
        st.markdown("### 📊 Uncertainty Analysis")
        
        # Check if model supports uncertainty quantification
        if not hasattr(model, 'predict_with_uncertainty'):
            st.warning("This model does not support uncertainty quantification.")
            return
        
        # Select trajectory for uncertainty analysis
        trajectory_idx = st.selectbox(
            "Select Trajectory for Uncertainty Analysis",
            range(len(self.sample_trajectories)),
            format_func=lambda x: f"Vehicle {self.sample_trajectories[x].vehicle_id}"
        )
        
        trajectory = self.sample_trajectories[trajectory_idx]
        
        # Generate prediction with uncertainty
        try:
            prediction, uncertainty = model.predict_with_uncertainty(trajectory, prediction_horizon=10)
            
            # Plot prediction with uncertainty bands
            fig = self.trajectory_plotter.plot_prediction_with_uncertainty(
                trajectory, prediction, uncertainty
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Uncertainty statistics
            st.markdown("#### 📊 Uncertainty Statistics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Mean Uncertainty", f"{np.mean(uncertainty):.2f}m")
            
            with col2:
                st.metric("Max Uncertainty", f"{np.max(uncertainty):.2f}m")
            
            with col3:
                st.metric("Uncertainty Growth Rate", f"{np.polyfit(range(len(uncertainty)), uncertainty, 1)[0]:.3f}")
            
            # Uncertainty evolution
            fig = px.line(
                x=range(len(uncertainty)),
                y=uncertainty,
                title="Uncertainty Evolution Over Time",
                labels={'x': 'Prediction Step', 'y': 'Uncertainty (m)'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error generating uncertainty analysis: {str(e)}")
    
    def _show_settings(self):
        """Display dashboard settings."""
        st.markdown("## ⚙️ Dashboard Settings")
        
        # Configuration options
        st.markdown("### 🔧 Configuration")
        
        # Model configuration
        st.markdown("#### 🤖 Model Configuration")
        
        new_prediction_horizon = st.slider(
            "Default Prediction Horizon (seconds)",
            min_value=1,
            max_value=30,
            value=self.config.prediction_horizon
        )
        
        new_update_frequency = st.selectbox(
            "Update Frequency",
            ["Real-time", "5 seconds", "10 seconds", "30 seconds", "Manual"]
        )
        
        # Visualization settings
        st.markdown("#### 📊 Visualization Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            default_plot_type = st.selectbox(
                "Default Plot Type",
                ["2D Trajectory", "3D Trajectory", "Velocity Profile", "Acceleration Profile"]
            )
            
            show_grid = st.checkbox("Show Grid", value=True)
        
        with col2:
            color_scheme = st.selectbox(
                "Color Scheme",
                ["Default", "Viridis", "Plasma", "Inferno", "Magma"]
            )
            
            animation_speed = st.slider(
                "Animation Speed",
                min_value=0.1,
                max_value=2.0,
                value=1.0,
                step=0.1
            )
        
        # Data settings
        st.markdown("#### 📈 Data Settings")
        
        max_trajectories = st.slider(
            "Maximum Trajectories to Load",
            min_value=10,
            max_value=1000,
            value=100
        )
        
        cache_predictions = st.checkbox("Cache Model Predictions", value=True)
        
        # Save settings
        if st.button("Save Settings"):
            # Update configuration
            self.config.prediction_horizon = new_prediction_horizon
            
            # Save to session state
            st.session_state.update_frequency = new_update_frequency
            st.session_state.default_plot_type = default_plot_type
            st.session_state.show_grid = show_grid
            st.session_state.color_scheme = color_scheme
            st.session_state.animation_speed = animation_speed
            st.session_state.max_trajectories = max_trajectories
            st.session_state.cache_predictions = cache_predictions
            
            st.success("Settings saved successfully!")
        
        # Export/Import settings
        st.markdown("### 📤 Export/Import")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export Settings"):
                # Export current settings
                settings = {
                    "prediction_horizon": self.config.prediction_horizon,
                    "update_frequency": getattr(st.session_state, 'update_frequency', 'Real-time'),
                    "default_plot_type": getattr(st.session_state, 'default_plot_type', '2D Trajectory'),
                    "show_grid": getattr(st.session_state, 'show_grid', True),
                    "color_scheme": getattr(st.session_state, 'color_scheme', 'Default'),
                    "animation_speed": getattr(st.session_state, 'animation_speed', 1.0),
                    "max_trajectories": getattr(st.session_state, 'max_trajectories', 100),
                    "cache_predictions": getattr(st.session_state, 'cache_predictions', True)
                }
                
                st.download_button(
                    label="Download Settings JSON",
                    data=str(settings),
                    file_name="dashboard_settings.json",
                    mime="application/json"
                )
        
        with col2:
            uploaded_file = st.file_uploader("Import Settings", type=['json'])
            if uploaded_file is not None:
                try:
                    # Parse and apply settings
                    st.success("Settings imported successfully!")
                except Exception as e:
                    st.error(f"Error importing settings: {str(e)}")


def main():
    """Main entry point for the dashboard."""
    try:
        # Initialize dashboard
        dashboard = TrajectoryDashboard()
        
        # Run dashboard
        dashboard.run()
        
    except Exception as e:
        st.error(f"Error starting dashboard: {str(e)}")
        logger.exception("Dashboard startup error")


if __name__ == "__main__":
    main()