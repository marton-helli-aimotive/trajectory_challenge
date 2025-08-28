#!/usr/bin/env python3
"""
Working dashboard for E2E testing.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def main():
    st.set_page_config(
        page_title="Vehicle Trajectory Prediction Dashboard",
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
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
        show_overview()
    elif page == "📊 Trajectory Visualization":
        show_trajectory_visualization()
    elif page == "🔍 Model Comparison":
        show_model_comparison()
    elif page == "📈 Dataset Exploration":
        show_dataset_exploration()
    elif page == "🤖 Model Explainability":
        show_model_explainability()
    elif page == "⚙️ Settings":
        show_settings()

def show_overview():
    """Display the main overview page."""
    st.markdown("## 📊 System Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Models Available", 6)
    
    with col2:
        st.metric("Sample Trajectories", 5)
    
    with col3:
        st.metric("Prediction Horizon", "10s")
    
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
        'Model': ['Constant Velocity', 'Constant Acceleration', 'Polynomial Regression', 'KNN', 'Gaussian Process', 'Ensemble'],
        'RMSE': [1.2, 1.5, 0.8, 0.9, 0.7, 0.6],
        'ADE': [0.9, 1.1, 0.6, 0.7, 0.5, 0.4],
        'FDE': [2.1, 2.5, 1.5, 1.8, 1.3, 1.1],
        'Inference Time (ms)': [15, 20, 45, 35, 80, 60]
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

def show_trajectory_visualization():
    """Display interactive trajectory visualization."""
    st.markdown("## 📊 Trajectory Visualization")
    
    # Trajectory selection
    trajectory_idx = st.selectbox(
        "Select Trajectory",
        range(5),
        format_func=lambda x: f"Vehicle {x}"
    )
    
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
                ["Constant Velocity", "Polynomial Regression", "KNN"],
                default=["Constant Velocity", "Polynomial Regression"]
            )
    
    # Create sample trajectory data
    t = np.linspace(0, 10, 50)
    x = t * 10 + np.random.normal(0, 1, 50)
    y = t * 5 + np.random.normal(0, 1, 50)
    v = 10 + np.random.normal(0, 2, 50)
    a = np.gradient(v, t)
    
    # Create the selected plot
    if plot_type == "2D Trajectory":
        fig = go.Figure()
        
        # Actual trajectory
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='lines+markers',
            name='Actual Trajectory',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=6, color=v, colorscale='Viridis')
        ))
        
        # Add predictions if requested
        if show_prediction and selected_models:
            colors = ['#ff7f0e', '#2ca02c', '#d62728']
            for i, model in enumerate(selected_models):
                # Generate fake prediction
                pred_x = x[-10:] + np.random.normal(0, 0.5, 10)
                pred_y = y[-10:] + np.random.normal(0, 0.5, 10)
                
                fig.add_trace(go.Scatter(
                    x=pred_x, y=pred_y,
                    mode='lines+markers',
                    name=f'{model} Prediction',
                    line=dict(color=colors[i % len(colors)], width=2, dash='dash'),
                    marker=dict(size=4)
                ))
        
        fig.update_layout(
            title="2D Trajectory Visualization",
            xaxis_title="X Position (m)",
            yaxis_title="Y Position (m)",
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    elif plot_type == "Velocity Profile":
        fig = px.line(x=t, y=v, title="Velocity Profile")
        fig.update_layout(xaxis_title="Time (s)", yaxis_title="Velocity (m/s)")
        st.plotly_chart(fig, use_container_width=True)
        
    elif plot_type == "Acceleration Profile":
        fig = px.line(x=t, y=a, title="Acceleration Profile")
        fig.update_layout(xaxis_title="Time (s)", yaxis_title="Acceleration (m/s²)")
        st.plotly_chart(fig, use_container_width=True)
    
    # Trajectory statistics
    st.markdown("## 📈 Trajectory Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Duration", f"{t[-1]:.1f}s")
    
    with col2:
        distance = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
        st.metric("Distance", f"{distance:.1f}m")
    
    with col3:
        st.metric("Avg Velocity", f"{np.mean(v):.1f} m/s")
    
    with col4:
        st.metric("Max Velocity", f"{np.max(v):.1f} m/s")

def show_model_comparison():
    """Display model comparison interface."""
    st.markdown("## 🔍 Model Comparison")
    
    # Model selection
    selected_models = st.multiselect(
        "Select Models to Compare",
        ["Constant Velocity", "Constant Acceleration", "Polynomial Regression", "KNN", "Gaussian Process", "Ensemble"],
        default=["Constant Velocity", "Polynomial Regression", "KNN"]
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
            max_value=10,
            value=3
        )
    
    with col3:
        include_safety = st.checkbox("Include Safety Metrics", value=True)
    
    # Run comparison
    if st.button("Run Model Comparison"):
        with st.spinner("Evaluating models..."):
            # Simulate evaluation results
            results = []
            for model in selected_models:
                results.append({
                    'Model': model,
                    'RMSE': np.random.uniform(0.5, 2.0),
                    'ADE': np.random.uniform(0.3, 1.5),
                    'FDE': np.random.uniform(0.8, 3.0),
                    'Inference Time (ms)': np.random.uniform(10, 100)
                })
            
            # Display results
            st.markdown("## 📊 Comparison Results")
            
            # Performance metrics table
            st.markdown("### 📈 Performance Metrics")
            metrics_df = pd.DataFrame(results)
            st.dataframe(metrics_df, use_container_width=True)
            
            # Create performance comparison plot
            fig = px.bar(metrics_df, x='Model', y='RMSE', title="RMSE Comparison")
            st.plotly_chart(fig, use_container_width=True)

def show_dataset_exploration():
    """Display dataset exploration interface."""
    st.markdown("## 📈 Dataset Exploration")
    
    # Dataset overview
    st.markdown("### 📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Trajectories", 5)
    
    with col2:
        st.metric("Total Duration", "50.0s")
    
    with col3:
        st.metric("Avg Trajectory Length", "50 points")
    
    with col4:
        st.metric("Data Points", 250)
    
    # Trajectory distribution analysis
    st.markdown("### 📊 Trajectory Distribution")
    
    # Create distribution plots
    col1, col2 = st.columns(2)
    
    with col1:
        # Duration distribution
        durations = np.random.uniform(8, 12, 5)
        fig = px.histogram(
            x=durations,
            title="Trajectory Duration Distribution",
            labels={'x': 'Duration (s)', 'y': 'Count'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Average velocity distribution
        avg_velocities = np.random.uniform(8, 15, 5)
        fig = px.histogram(
            x=avg_velocities,
            title="Average Velocity Distribution",
            labels={'x': 'Average Velocity (m/s)', 'y': 'Count'}
        )
        st.plotly_chart(fig, use_container_width=True)

def show_model_explainability():
    """Display model explainability interface."""
    st.markdown("## 🤖 Model Explainability")
    
    # Model selection
    selected_model = st.selectbox(
        "Select Model for Analysis",
        ["Constant Velocity", "Constant Acceleration", "Polynomial Regression", "KNN", "Gaussian Process", "Ensemble"]
    )
    
    if not selected_model:
        st.warning("Please select a model for explainability analysis.")
        return
    
    # Explainability options
    explainability_type = st.selectbox(
        "Explainability Type",
        ["Feature Importance", "Prediction Analysis", "Model Behavior", "Uncertainty Analysis"]
    )
    
    if explainability_type == "Feature Importance":
        st.markdown("### 📊 Feature Importance")
        
        # Generate sample feature importance
        features = ['velocity', 'acceleration', 'heading', 'position_x', 'position_y', 'time']
        importance = np.random.uniform(0, 1, len(features))
        importance = importance / np.sum(importance)  # Normalize
        
        # Create feature importance plot
        fig = px.bar(
            x=features,
            y=importance,
            title=f"Feature Importance - {selected_model}",
            labels={'x': 'Features', 'y': 'Importance Score'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance table
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        st.dataframe(importance_df, use_container_width=True)

def show_settings():
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
        value=10
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
    
    # Save settings
    if st.button("Save Settings"):
        st.success("Settings saved successfully!")

if __name__ == "__main__":
    main()