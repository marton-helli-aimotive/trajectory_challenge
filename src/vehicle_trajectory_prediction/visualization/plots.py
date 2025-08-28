"""
Plotting components for trajectory visualization and analysis.

This module provides comprehensive plotting capabilities for:
- 2D and 3D trajectory visualization
- Model comparison plots
- Dataset exploration visualizations
- Interactive trajectory analysis
"""

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from vehicle_trajectory_prediction.core.models import TrajectoryData


class TrajectoryPlotter:
    """
    Handles trajectory-specific plotting and visualization.
    
    Provides methods for:
    - 2D trajectory plots with predictions
    - 3D trajectory visualization
    - Velocity and acceleration profiles
    - Prediction analysis plots
    """
    
    def __init__(self):
        """Initialize the trajectory plotter."""
        self.color_palette = px.colors.qualitative.Set1
        self.default_colors = {
            'actual': '#1f77b4',
            'prediction': '#ff7f0e',
            'uncertainty': '#2ca02c',
            'background': '#f8f9fa'
        }
    
    def plot_2d_trajectory(
        self, 
        trajectory: TrajectoryData,
        show_prediction: bool = False,
        models: Optional[Dict[str, Any]] = None,
        selected_models: Optional[List[str]] = None
    ) -> go.Figure:
        """
        Create a 2D trajectory plot with optional predictions.
        
        Args:
            trajectory: The trajectory data to plot
            show_prediction: Whether to show model predictions
            models: Dictionary of available models
            selected_models: List of model names to show predictions for
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        # Plot actual trajectory
        fig.add_trace(go.Scatter(
            x=trajectory.x_positions,
            y=trajectory.y_positions,
            mode='lines+markers',
            name='Actual Trajectory',
            line=dict(color=self.default_colors['actual'], width=3),
            marker=dict(size=6),
            hovertemplate='<b>Time:</b> %{customdata:.1f}s<br>' +
                         '<b>Position:</b> (%{x:.1f}, %{y:.1f})<br>' +
                         '<b>Velocity:</b> %{marker.color:.1f} m/s<extra></extra>',
            customdata=trajectory.timestamps,
            marker_color=trajectory.velocities,
            colorscale='Viridis',
            colorbar=dict(title="Velocity (m/s)")
        ))
        
        # Add predictions if requested
        if show_prediction and models and selected_models:
            for i, model_name in enumerate(selected_models):
                if model_name in models:
                    try:
                        model = models[model_name]
                        prediction = model.predict(trajectory, prediction_horizon=10)
                        
                        # Plot prediction
                        fig.add_trace(go.Scatter(
                            x=prediction.x_positions,
                            y=prediction.y_positions,
                            mode='lines+markers',
                            name=f'{model_name} Prediction',
                            line=dict(color=self.color_palette[i % len(self.color_palette)], 
                                     width=2, dash='dash'),
                            marker=dict(size=4),
                            hovertemplate='<b>Prediction:</b> %{customdata:.1f}s<br>' +
                                         '<b>Position:</b> (%{x:.1f}, %{y:.1f})<extra></extra>',
                            customdata=np.arange(len(prediction.x_positions)) * 0.1
                        ))
                    except Exception as e:
                        print(f"Error plotting prediction for {model_name}: {e}")
        
        # Update layout
        fig.update_layout(
            title=f"2D Trajectory - Vehicle {trajectory.vehicle_id}",
            xaxis_title="X Position (m)",
            yaxis_title="Y Position (m)",
            hovermode='closest',
            showlegend=True,
            plot_bgcolor=self.default_colors['background'],
            width=800,
            height=600
        )
        
        # Add grid
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        return fig
    
    def plot_3d_trajectory(
        self,
        trajectory: TrajectoryData,
        show_prediction: bool = False,
        models: Optional[Dict[str, Any]] = None,
        selected_models: Optional[List[str]] = None
    ) -> go.Figure:
        """
        Create a 3D trajectory plot with velocity as the third dimension.
        
        Args:
            trajectory: The trajectory data to plot
            show_prediction: Whether to show model predictions
            models: Dictionary of available models
            selected_models: List of model names to show predictions for
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        # Plot actual trajectory in 3D
        fig.add_trace(go.Scatter3d(
            x=trajectory.x_positions,
            y=trajectory.y_positions,
            z=trajectory.velocities,
            mode='lines+markers',
            name='Actual Trajectory',
            line=dict(color=self.default_colors['actual'], width=4),
            marker=dict(size=4, color=trajectory.velocities, colorscale='Viridis'),
            hovertemplate='<b>Time:</b> %{customdata:.1f}s<br>' +
                         '<b>Position:</b> (%{x:.1f}, %{y:.1f})<br>' +
                         '<b>Velocity:</b> %{z:.1f} m/s<extra></extra>',
            customdata=trajectory.timestamps
        ))
        
        # Add predictions if requested
        if show_prediction and models and selected_models:
            for i, model_name in enumerate(selected_models):
                if model_name in models:
                    try:
                        model = models[model_name]
                        prediction = model.predict(trajectory, prediction_horizon=10)
                        
                        # Plot prediction in 3D
                        fig.add_trace(go.Scatter3d(
                            x=prediction.x_positions,
                            y=prediction.y_positions,
                            z=prediction.velocities,
                            mode='lines+markers',
                            name=f'{model_name} Prediction',
                            line=dict(color=self.color_palette[i % len(self.color_palette)], 
                                     width=3, dash='dash'),
                            marker=dict(size=3),
                            hovertemplate='<b>Prediction:</b> %{customdata:.1f}s<br>' +
                                         '<b>Position:</b> (%{x:.1f}, %{y:.1f})<br>' +
                                         '<b>Velocity:</b> %{z:.1f} m/s<extra></extra>',
                            customdata=np.arange(len(prediction.x_positions)) * 0.1
                        ))
                    except Exception as e:
                        print(f"Error plotting 3D prediction for {model_name}: {e}")
        
        # Update layout
        fig.update_layout(
            title=f"3D Trajectory - Vehicle {trajectory.vehicle_id}",
            scene=dict(
                xaxis_title="X Position (m)",
                yaxis_title="Y Position (m)",
                zaxis_title="Velocity (m/s)",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            showlegend=True,
            width=800,
            height=600
        )
        
        return fig
    
    def plot_velocity_profile(self, trajectory: TrajectoryData) -> go.Figure:
        """
        Create a velocity profile plot.
        
        Args:
            trajectory: The trajectory data to plot
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        # Plot velocity over time
        fig.add_trace(go.Scatter(
            x=trajectory.timestamps,
            y=trajectory.velocities,
            mode='lines+markers',
            name='Velocity',
            line=dict(color=self.default_colors['actual'], width=3),
            marker=dict(size=6),
            hovertemplate='<b>Time:</b> %{x:.1f}s<br>' +
                         '<b>Velocity:</b> %{y:.1f} m/s<extra></extra>'
        ))
        
        # Add acceleration as secondary axis
        fig.add_trace(go.Scatter(
            x=trajectory.timestamps,
            y=trajectory.accelerations,
            mode='lines',
            name='Acceleration',
            line=dict(color='red', width=2),
            yaxis='y2',
            hovertemplate='<b>Time:</b> %{x:.1f}s<br>' +
                         '<b>Acceleration:</b> %{y:.1f} m/s²<extra></extra>'
        ))
        
        # Update layout with dual y-axes
        fig.update_layout(
            title=f"Velocity Profile - Vehicle {trajectory.vehicle_id}",
            xaxis_title="Time (s)",
            yaxis=dict(
                title="Velocity (m/s)",
                titlefont=dict(color=self.default_colors['actual']),
                tickfont=dict(color=self.default_colors['actual'])
            ),
            yaxis2=dict(
                title="Acceleration (m/s²)",
                titlefont=dict(color="red"),
                tickfont=dict(color="red"),
                anchor="x",
                overlaying="y",
                side="right"
            ),
            hovermode='closest',
            showlegend=True,
            plot_bgcolor=self.default_colors['background'],
            width=800,
            height=400
        )
        
        # Add grid
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        return fig
    
    def plot_acceleration_profile(self, trajectory: TrajectoryData) -> go.Figure:
        """
        Create an acceleration profile plot.
        
        Args:
            trajectory: The trajectory data to plot
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        # Plot acceleration over time
        fig.add_trace(go.Scatter(
            x=trajectory.timestamps,
            y=trajectory.accelerations,
            mode='lines+markers',
            name='Acceleration',
            line=dict(color='red', width=3),
            marker=dict(size=6),
            hovertemplate='<b>Time:</b> %{x:.1f}s<br>' +
                         '<b>Acceleration:</b> %{y:.1f} m/s²<extra></extra>'
        ))
        
        # Add zero line for reference
        fig.add_hline(y=0, line_dash="dash", line_color="gray", 
                     annotation_text="Zero Acceleration")
        
        # Update layout
        fig.update_layout(
            title=f"Acceleration Profile - Vehicle {trajectory.vehicle_id}",
            xaxis_title="Time (s)",
            yaxis_title="Acceleration (m/s²)",
            hovermode='closest',
            showlegend=True,
            plot_bgcolor=self.default_colors['background'],
            width=800,
            height=400
        )
        
        # Add grid
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        return fig
    
    def plot_prediction_analysis(
        self, 
        trajectory: TrajectoryData, 
        prediction: TrajectoryData
    ) -> go.Figure:
        """
        Create a prediction analysis plot comparing actual vs predicted trajectories.
        
        Args:
            trajectory: The actual trajectory data
            prediction: The predicted trajectory data
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        # Plot actual trajectory
        fig.add_trace(go.Scatter(
            x=trajectory.x_positions,
            y=trajectory.y_positions,
            mode='lines+markers',
            name='Actual Trajectory',
            line=dict(color=self.default_colors['actual'], width=3),
            marker=dict(size=6),
            hovertemplate='<b>Actual</b><br>' +
                         '<b>Position:</b> (%{x:.1f}, %{y:.1f})<extra></extra>'
        ))
        
        # Plot predicted trajectory
        fig.add_trace(go.Scatter(
            x=prediction.x_positions,
            y=prediction.y_positions,
            mode='lines+markers',
            name='Predicted Trajectory',
            line=dict(color=self.default_colors['prediction'], width=3, dash='dash'),
            marker=dict(size=6),
            hovertemplate='<b>Predicted</b><br>' +
                         '<b>Position:</b> (%{x:.1f}, %{y:.1f})<extra></extra>'
        ))
        
        # Add error vectors (if trajectories have same length)
        min_length = min(len(trajectory.x_positions), len(prediction.x_positions))
        if min_length > 0:
            actual_x = trajectory.x_positions[:min_length]
            actual_y = trajectory.y_positions[:min_length]
            pred_x = prediction.x_positions[:min_length]
            pred_y = prediction.y_positions[:min_length]
            
            # Calculate errors
            errors = np.sqrt((actual_x - pred_x)**2 + (actual_y - pred_y)**2)
            
            # Add error vectors
            for i in range(0, min_length, max(1, min_length // 10)):  # Show every 10th error
                fig.add_trace(go.Scatter(
                    x=[actual_x[i], pred_x[i]],
                    y=[actual_y[i], pred_y[i]],
                    mode='lines',
                    name=f'Error {i}' if i == 0 else None,
                    line=dict(color='red', width=1),
                    showlegend=False,
                    hovertemplate=f'<b>Error:</b> {errors[i]:.2f}m<extra></extra>'
                ))
        
        # Update layout
        fig.update_layout(
            title=f"Prediction Analysis - Vehicle {trajectory.vehicle_id}",
            xaxis_title="X Position (m)",
            yaxis_title="Y Position (m)",
            hovermode='closest',
            showlegend=True,
            plot_bgcolor=self.default_colors['background'],
            width=800,
            height=600
        )
        
        # Add grid
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        return fig
    
    def plot_prediction_with_uncertainty(
        self,
        trajectory: TrajectoryData,
        prediction: TrajectoryData,
        uncertainty: np.ndarray
    ) -> go.Figure:
        """
        Create a prediction plot with uncertainty bands.
        
        Args:
            trajectory: The actual trajectory data
            prediction: The predicted trajectory data
            uncertainty: Uncertainty values for each prediction point
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        # Plot actual trajectory
        fig.add_trace(go.Scatter(
            x=trajectory.x_positions,
            y=trajectory.y_positions,
            mode='lines+markers',
            name='Actual Trajectory',
            line=dict(color=self.default_colors['actual'], width=3),
            marker=dict(size=6),
            hovertemplate='<b>Actual</b><br>' +
                         '<b>Position:</b> (%{x:.1f}, %{y:.1f})<extra></extra>'
        ))
        
        # Plot predicted trajectory
        fig.add_trace(go.Scatter(
            x=prediction.x_positions,
            y=prediction.y_positions,
            mode='lines+markers',
            name='Predicted Trajectory',
            line=dict(color=self.default_colors['prediction'], width=3),
            marker=dict(size=6),
            hovertemplate='<b>Predicted</b><br>' +
                         '<b>Position:</b> (%{x:.1f}, %{y:.1f})<extra></extra>'
        ))
        
        # Add uncertainty bands
        if len(uncertainty) == len(prediction.x_positions):
            # Create uncertainty circles around prediction points
            for i, (x, y, u) in enumerate(zip(prediction.x_positions, prediction.y_positions, uncertainty)):
                # Create circle points
                theta = np.linspace(0, 2*np.pi, 50)
                circle_x = x + u * np.cos(theta)
                circle_y = y + u * np.sin(theta)
                
                fig.add_trace(go.Scatter(
                    x=circle_x,
                    y=circle_y,
                    mode='lines',
                    name=f'Uncertainty {i}' if i == 0 else None,
                    line=dict(color=self.default_colors['uncertainty'], width=1, opacity=0.3),
                    showlegend=False,
                    hovertemplate=f'<b>Uncertainty:</b> {u:.2f}m<extra></extra>'
                ))
        
        # Update layout
        fig.update_layout(
            title=f"Prediction with Uncertainty - Vehicle {trajectory.vehicle_id}",
            xaxis_title="X Position (m)",
            yaxis_title="Y Position (m)",
            hovermode='closest',
            showlegend=True,
            plot_bgcolor=self.default_colors['background'],
            width=800,
            height=600
        )
        
        # Add grid
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        return fig


class ModelComparisonPlotter:
    """
    Handles model comparison plotting and visualization.
    
    Provides methods for:
    - Performance comparison plots
    - Safety metrics visualization
    - Model ranking plots
    - Statistical comparison charts
    """
    
    def __init__(self):
        """Initialize the model comparison plotter."""
        self.color_palette = px.colors.qualitative.Set1
    
    def plot_performance_comparison(self, metrics_df: pd.DataFrame) -> go.Figure:
        """
        Create a performance comparison plot for multiple models.
        
        Args:
            metrics_df: DataFrame with performance metrics for each model
            
        Returns:
            Plotly figure object
        """
        # Melt the dataframe for easier plotting
        metrics_to_plot = ['RMSE', 'ADE', 'FDE']
        available_metrics = [col for col in metrics_to_plot if col in metrics_df.columns]
        
        if not available_metrics:
            return go.Figure()
        
        melted_df = metrics_df.melt(
            id_vars=['Model'], 
            value_vars=available_metrics,
            var_name='Metric', 
            value_name='Value'
        )
        
        fig = px.bar(
            melted_df,
            x='Model',
            y='Value',
            color='Metric',
            title="Model Performance Comparison",
            barmode='group',
            color_discrete_sequence=self.color_palette
        )
        
        fig.update_layout(
            xaxis_title="Model",
            yaxis_title="Error Value",
            showlegend=True,
            width=800,
            height=500
        )
        
        return fig
    
    def plot_safety_comparison(self, safety_df: pd.DataFrame) -> go.Figure:
        """
        Create a safety metrics comparison plot.
        
        Args:
            safety_df: DataFrame with safety metrics for each model
            
        Returns:
            Plotly figure object
        """
        # Melt the dataframe for easier plotting
        safety_metrics = ['Minimum_Distance', 'TTC', 'Lateral_Error', 'Risk_Score']
        available_metrics = [col for col in safety_metrics if col in safety_df.columns]
        
        if not available_metrics:
            return go.Figure()
        
        melted_df = safety_df.melt(
            id_vars=['Model'], 
            value_vars=available_metrics,
            var_name='Safety_Metric', 
            value_name='Value'
        )
        
        fig = px.bar(
            melted_df,
            x='Model',
            y='Value',
            color='Safety_Metric',
            title="Model Safety Metrics Comparison",
            barmode='group',
            color_discrete_sequence=self.color_palette
        )
        
        fig.update_layout(
            xaxis_title="Model",
            yaxis_title="Safety Metric Value",
            showlegend=True,
            width=800,
            height=500
        )
        
        return fig
    
    def plot_model_ranking(self, ranking_df: pd.DataFrame) -> go.Figure:
        """
        Create a model ranking visualization.
        
        Args:
            ranking_df: DataFrame with model rankings
            
        Returns:
            Plotly figure object
        """
        if 'Rank' not in ranking_df.columns:
            return go.Figure()
        
        # Sort by rank
        ranking_df = ranking_df.sort_values('Rank')
        
        fig = px.bar(
            ranking_df,
            x='Model',
            y='Score',
            color='Rank',
            title="Model Performance Ranking",
            color_continuous_scale='RdYlGn_r',  # Red to green (lower rank = better)
            text='Rank'
        )
        
        fig.update_layout(
            xaxis_title="Model",
            yaxis_title="Performance Score",
            showlegend=True,
            width=800,
            height=500
        )
        
        # Update text position
        fig.update_traces(textposition='outside')
        
        return fig
    
    def plot_radar_chart(self, metrics_df: pd.DataFrame) -> go.Figure:
        """
        Create a radar chart for multi-dimensional model comparison.
        
        Args:
            metrics_df: DataFrame with multiple metrics for each model
            
        Returns:
            Plotly figure object
        """
        # Select metrics for radar chart
        metrics_to_plot = ['RMSE', 'ADE', 'FDE', 'Inference_Time']
        available_metrics = [col for col in metrics_to_plot if col in metrics_df.columns]
        
        if len(available_metrics) < 3:
            return go.Figure()
        
        fig = go.Figure()
        
        for i, (_, row) in enumerate(metrics_df.iterrows()):
            values = [row[metric] for metric in available_metrics]
            # Normalize values to 0-1 range for better visualization
            normalized_values = [(v - min(values)) / (max(values) - min(values)) for v in values]
            
            fig.add_trace(go.Scatterpolar(
                r=normalized_values,
                theta=available_metrics,
                fill='toself',
                name=row['Model'],
                line_color=self.color_palette[i % len(self.color_palette)]
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Model Performance Radar Chart",
            width=600,
            height=600
        )
        
        return fig


class DatasetExplorer:
    """
    Handles dataset exploration and visualization.
    
    Provides methods for:
    - Dataset overview plots
    - Trajectory pattern analysis
    - Feature distribution visualization
    - Data quality assessment plots
    """
    
    def __init__(self):
        """Initialize the dataset explorer."""
        self.color_palette = px.colors.qualitative.Set1
    
    def plot_all_trajectories(self, trajectories: List[TrajectoryData]) -> go.Figure:
        """
        Plot all trajectories on a single figure.
        
        Args:
            trajectories: List of trajectory data objects
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        for i, trajectory in enumerate(trajectories):
            fig.add_trace(go.Scatter(
                x=trajectory.x_positions,
                y=trajectory.y_positions,
                mode='lines',
                name=f'Vehicle {trajectory.vehicle_id}',
                line=dict(color=self.color_palette[i % len(self.color_palette)], width=2),
                hovertemplate='<b>Vehicle:</b> %{fullData.name}<br>' +
                             '<b>Position:</b> (%{x:.1f}, %{y:.1f})<extra></extra>'
            ))
        
        fig.update_layout(
            title="All Trajectories Overview",
            xaxis_title="X Position (m)",
            yaxis_title="Y Position (m)",
            hovermode='closest',
            showlegend=True,
            plot_bgcolor='#f8f9fa',
            width=800,
            height=600
        )
        
        # Add grid
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        return fig
    
    def plot_trajectory_statistics(self, trajectories: List[TrajectoryData]) -> go.Figure:
        """
        Create a comprehensive statistics plot for the dataset.
        
        Args:
            trajectories: List of trajectory data objects
            
        Returns:
            Plotly figure object with subplots
        """
        # Calculate statistics
        durations = [t.timestamps[-1] for t in trajectories]
        avg_velocities = [np.mean(t.velocities) for t in trajectories]
        max_velocities = [np.max(t.velocities) for t in trajectories]
        distances = [np.sum(np.sqrt(np.diff(t.x_positions)**2 + np.diff(t.y_positions)**2)) 
                    for t in trajectories]
        
        # Create subplots
        fig = go.Figure()
        
        # Duration distribution
        fig.add_trace(go.Histogram(
            x=durations,
            name='Duration',
            nbinsx=20,
            opacity=0.7
        ))
        
        fig.update_layout(
            title="Trajectory Duration Distribution",
            xaxis_title="Duration (s)",
            yaxis_title="Count",
            showlegend=False,
            width=400,
            height=300
        )
        
        return fig
    
    def plot_feature_correlations(self, trajectories: List[TrajectoryData]) -> go.Figure:
        """
        Create a correlation matrix plot for trajectory features.
        
        Args:
            trajectories: List of trajectory data objects
            
        Returns:
            Plotly figure object
        """
        # Extract features
        feature_data = []
        for traj in trajectories:
            feature_data.append({
                'avg_velocity': np.mean(traj.velocities),
                'max_velocity': np.max(traj.velocities),
                'avg_acceleration': np.mean(traj.accelerations),
                'max_acceleration': np.max(traj.accelerations),
                'duration': traj.timestamps[-1],
                'distance': np.sum(np.sqrt(np.diff(traj.x_positions)**2 + np.diff(traj.y_positions)**2))
            })
        
        feature_df = pd.DataFrame(feature_data)
        
        # Calculate correlation matrix
        corr_matrix = feature_df.corr()
        
        # Create heatmap
        fig = px.imshow(
            corr_matrix,
            title="Feature Correlation Matrix",
            color_continuous_scale='RdBu',
            aspect='auto'
        )
        
        fig.update_layout(
            width=600,
            height=500
        )
        
        return fig
    
    def plot_data_quality_metrics(self, trajectories: List[TrajectoryData]) -> go.Figure:
        """
        Create data quality assessment plots.
        
        Args:
            trajectories: List of trajectory data objects
            
        Returns:
            Plotly figure object
        """
        # Calculate quality metrics
        quality_metrics = []
        
        for traj in trajectories:
            # Completeness (no NaN values)
            completeness = 1.0 - (np.isnan(traj.x_positions).sum() + 
                                 np.isnan(traj.y_positions).sum() + 
                                 np.isnan(traj.velocities).sum()) / (len(traj.timestamps) * 3)
            
            # Consistency (velocity should be positive)
            velocity_consistency = (traj.velocities >= 0).mean()
            
            # Smoothness (no sudden jumps)
            position_jumps = np.sqrt(np.diff(traj.x_positions)**2 + np.diff(traj.y_positions)**2)
            smoothness = 1.0 - (position_jumps > np.percentile(position_jumps, 95)).mean()
            
            quality_metrics.append({
                'vehicle_id': traj.vehicle_id,
                'completeness': completeness,
                'velocity_consistency': velocity_consistency,
                'smoothness': smoothness
            })
        
        quality_df = pd.DataFrame(quality_metrics)
        
        # Create quality metrics plot
        fig = px.bar(
            quality_df,
            x='vehicle_id',
            y=['completeness', 'velocity_consistency', 'smoothness'],
            title="Data Quality Metrics by Vehicle",
            barmode='group',
            color_discrete_sequence=self.color_palette
        )
        
        fig.update_layout(
            xaxis_title="Vehicle ID",
            yaxis_title="Quality Score",
            yaxis_range=[0, 1],
            showlegend=True,
            width=800,
            height=500
        )
        
        return fig