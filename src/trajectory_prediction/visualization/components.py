"""
Core visualization components for trajectory prediction dashboard.

This module provides reusable visualization components including:
- TrajectoryPlotter: Interactive trajectory visualization with Plotly
- PredictionVisualizer: Real-time prediction display with uncertainty
- ModelComparisonChart: Performance comparison across models
- PerformanceMetricsDisplay: Real-time metrics dashboard
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import asyncio
import json
from datetime import datetime, timedelta

from ..data.schemas import TrajectoryData
from ..api.models import TrajectoryRequest, TrajectoryResponse
from ..evaluation.metrics import TrajectoryMetrics, SafetyMetrics


@dataclass
class PlotConfig:
    """Configuration for plot styling and behavior."""
    
    width: int = 800
    height: int = 600
    theme: str = "plotly_white"
    color_palette: List[str] = None
    show_legend: bool = True
    interactive: bool = True
    
    def __post_init__(self):
        if self.color_palette is None:
            self.color_palette = px.colors.qualitative.Set1


class TrajectoryPlotter:
    """Interactive trajectory visualization with Plotly."""
    
    def __init__(self, config: Optional[PlotConfig] = None):
        self.config = config or PlotConfig()
        
    def plot_trajectory(
        self,
        trajectory: Union[TrajectoryData, Dict, pd.DataFrame],
        predictions: Optional[List[TrajectoryResponse]] = None,
        title: str = "Trajectory Visualization",
        show_uncertainty: bool = True,
        highlight_conflicts: bool = True
    ) -> go.Figure:
        """
        Plot trajectory with optional predictions and uncertainty.
        
        Args:
            trajectory: Historical trajectory data
            predictions: List of prediction results from different models
            title: Plot title
            show_uncertainty: Whether to show prediction uncertainty
            highlight_conflicts: Whether to highlight potential conflicts
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        # Extract trajectory data
        if isinstance(trajectory, TrajectoryData):
            x_data = [p.x for p in trajectory.positions]
            y_data = [p.y for p in trajectory.positions]
            timestamps = trajectory.timestamps
        elif isinstance(trajectory, dict):
            x_data = trajectory.get('x', [])
            y_data = trajectory.get('y', [])
            timestamps = trajectory.get('timestamps', list(range(len(x_data))))
        else:  # DataFrame
            x_data = trajectory['x'].tolist()
            y_data = trajectory['y'].tolist()
            timestamps = trajectory.get('timestamp', trajectory.index).tolist()
        
        # Plot historical trajectory
        fig.add_trace(go.Scatter(
            x=x_data,
            y=y_data,
            mode='lines+markers',
            name='Historical Trajectory',
            line=dict(color=self.config.color_palette[0], width=3),
            marker=dict(size=6),
            hovertemplate='<b>Historical</b><br>' +
                         'X: %{x:.2f}<br>' +
                         'Y: %{y:.2f}<br>' +
                         '<extra></extra>'
        ))
        
        # Plot predictions if provided
        if predictions:
            for i, pred in enumerate(predictions):
                if i >= len(self.config.color_palette):
                    color = f'hsl({i * 45 % 360}, 70%, 50%)'
                else:
                    color = self.config.color_palette[i + 1]
                
                # Extract prediction data
                pred_x = [p.x for p in pred.predicted_trajectory.positions]
                pred_y = [p.y for p in pred.predicted_trajectory.positions]
                
                # Main prediction line
                fig.add_trace(go.Scatter(
                    x=pred_x,
                    y=pred_y,
                    mode='lines+markers',
                    name=f'{pred.model_name} Prediction',
                    line=dict(color=color, width=2, dash='dash'),
                    marker=dict(size=4),
                    hovertemplate=f'<b>{pred.model_name}</b><br>' +
                                 'X: %{x:.2f}<br>' +
                                 'Y: %{y:.2f}<br>' +
                                 f'Confidence: {pred.confidence:.2%}<br>' +
                                 '<extra></extra>'
                ))
                
                # Add uncertainty bounds if available and requested
                if show_uncertainty and pred.uncertainty:
                    uncertainty_x = pred_x
                    uncertainty_upper = [p.y + pred.uncertainty.get(f'position_{j}', 0) 
                                       for j, p in enumerate(pred.predicted_trajectory.positions)]
                    uncertainty_lower = [p.y - pred.uncertainty.get(f'position_{j}', 0) 
                                       for j, p in enumerate(pred.predicted_trajectory.positions)]
                    
                    # Upper bound
                    fig.add_trace(go.Scatter(
                        x=uncertainty_x,
                        y=uncertainty_upper,
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                    
                    # Lower bound with fill
                    fig.add_trace(go.Scatter(
                        x=uncertainty_x,
                        y=uncertainty_lower,
                        mode='lines',
                        line=dict(width=0),
                        fill='tonexty',
                        fillcolor=f'rgba({color[4:-1]}, 0.2)',
                        name=f'{pred.model_name} Uncertainty',
                        showlegend=True,
                        hoverinfo='skip'
                    ))
        
        # Highlight potential conflicts
        if highlight_conflicts and predictions:
            conflict_zones = self._detect_conflicts(predictions)
            for zone in conflict_zones:
                fig.add_shape(
                    type="circle",
                    x0=zone['x'] - zone['radius'],
                    y0=zone['y'] - zone['radius'],
                    x1=zone['x'] + zone['radius'],
                    y1=zone['y'] + zone['radius'],
                    line=dict(color="red", width=2, dash="dash"),
                    fillcolor="rgba(255, 0, 0, 0.1)"
                )
        
        # Update layout
        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor='center'),
            xaxis_title="X Position (m)",
            yaxis_title="Y Position (m)",
            width=self.config.width,
            height=self.config.height,
            template=self.config.theme,
            showlegend=self.config.show_legend,
            hovermode='closest',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Equal aspect ratio for realistic trajectory view
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        
        return fig
    
    def plot_velocity_profile(
        self,
        trajectory: Union[TrajectoryData, Dict],
        predictions: Optional[List[TrajectoryResponse]] = None,
        title: str = "Velocity Profile"
    ) -> go.Figure:
        """Plot velocity over time with predictions."""
        fig = go.Figure()
        
        # Extract historical velocity data
        if isinstance(trajectory, TrajectoryData):
            velocities = [v.magnitude for v in trajectory.velocities] if trajectory.velocities else []
            timestamps = trajectory.timestamps
        else:
            velocities = trajectory.get('velocities', [])
            timestamps = trajectory.get('timestamps', list(range(len(velocities))))
        
        if velocities:
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=velocities,
                mode='lines+markers',
                name='Historical Velocity',
                line=dict(color=self.config.color_palette[0], width=3)
            ))
        
        # Add prediction velocities
        if predictions:
            for i, pred in enumerate(predictions):
                pred_velocities = [v.magnitude for v in pred.predicted_trajectory.velocities] if pred.predicted_trajectory.velocities else []
                pred_timestamps = pred.predicted_trajectory.timestamps
                
                if pred_velocities:
                    color = self.config.color_palette[i + 1] if i < len(self.config.color_palette) - 1 else f'hsl({i * 45 % 360}, 70%, 50%)'
                    fig.add_trace(go.Scatter(
                        x=pred_timestamps,
                        y=pred_velocities,
                        mode='lines+markers',
                        name=f'{pred.model_name} Velocity',
                        line=dict(color=color, width=2, dash='dash')
                    ))
        
        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor='center'),
            xaxis_title="Time (s)",
            yaxis_title="Velocity (m/s)",
            template=self.config.theme,
            showlegend=True
        )
        
        return fig
    
    def _detect_conflicts(self, predictions: List[TrajectoryResponse], threshold: float = 2.0) -> List[Dict]:
        """Detect potential conflict zones between predictions."""
        conflicts = []
        
        for i, pred1 in enumerate(predictions):
            for j, pred2 in enumerate(predictions[i+1:], i+1):
                # Compare trajectories at each time step
                min_len = min(
                    len(pred1.predicted_trajectory.positions),
                    len(pred2.predicted_trajectory.positions)
                )
                
                for t in range(min_len):
                    pos1 = pred1.predicted_trajectory.positions[t]
                    pos2 = pred2.predicted_trajectory.positions[t]
                    
                    distance = np.sqrt((pos1.x - pos2.x)**2 + (pos1.y - pos2.y)**2)
                    
                    if distance < threshold:
                        conflicts.append({
                            'x': (pos1.x + pos2.x) / 2,
                            'y': (pos1.y + pos2.y) / 2,
                            'radius': threshold,
                            'models': [pred1.model_name, pred2.model_name],
                            'time_step': t,
                            'distance': distance
                        })
        
        return conflicts


class PredictionVisualizer:
    """Real-time prediction display with uncertainty visualization."""
    
    def __init__(self, config: Optional[PlotConfig] = None):
        self.config = config or PlotConfig()
        
    def create_prediction_dashboard(
        self,
        predictions: List[TrajectoryResponse],
        show_metrics: bool = True,
        show_uncertainty: bool = True
    ) -> Dict[str, go.Figure]:
        """Create comprehensive prediction dashboard."""
        dashboard = {}
        
        # Main trajectory plot
        plotter = TrajectoryPlotter(self.config)
        dashboard['trajectory'] = plotter.plot_trajectory(
            trajectory={'x': [], 'y': []},  # Will be updated with real data
            predictions=predictions,
            title="Real-time Trajectory Predictions",
            show_uncertainty=show_uncertainty
        )
        
        # Confidence comparison
        dashboard['confidence'] = self._create_confidence_plot(predictions)
        
        # Prediction metrics
        if show_metrics:
            dashboard['metrics'] = self._create_metrics_plot(predictions)
        
        # Uncertainty analysis
        if show_uncertainty:
            dashboard['uncertainty'] = self._create_uncertainty_plot(predictions)
        
        return dashboard
    
    def _create_confidence_plot(self, predictions: List[TrajectoryResponse]) -> go.Figure:
        """Create confidence comparison bar chart."""
        model_names = [pred.model_name for pred in predictions]
        confidences = [pred.confidence for pred in predictions]
        
        fig = go.Figure(data=[
            go.Bar(
                x=model_names,
                y=confidences,
                marker_color=self.config.color_palette[:len(model_names)],
                text=[f"{c:.1%}" for c in confidences],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Model Confidence Comparison",
            xaxis_title="Model",
            yaxis_title="Confidence",
            template=self.config.theme,
            yaxis=dict(range=[0, 1], tickformat='.0%')
        )
        
        return fig
    
    def _create_metrics_plot(self, predictions: List[TrajectoryResponse]) -> go.Figure:
        """Create prediction metrics visualization."""
        model_names = [pred.model_name for pred in predictions]
        
        # Extract metrics if available
        rmse_values = []
        mae_values = []
        
        for pred in predictions:
            if pred.metrics:
                rmse_values.append(pred.metrics.get('rmse', 0))
                mae_values.append(pred.metrics.get('mae', 0))
            else:
                rmse_values.append(0)
                mae_values.append(0)
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('RMSE', 'MAE'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # RMSE plot
        fig.add_trace(
            go.Bar(
                x=model_names,
                y=rmse_values,
                name="RMSE",
                marker_color=self.config.color_palette[0]
            ),
            row=1, col=1
        )
        
        # MAE plot
        fig.add_trace(
            go.Bar(
                x=model_names,
                y=mae_values,
                name="MAE",
                marker_color=self.config.color_palette[1]
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title="Prediction Accuracy Metrics",
            template=self.config.theme,
            showlegend=False
        )
        
        return fig
    
    def _create_uncertainty_plot(self, predictions: List[TrajectoryResponse]) -> go.Figure:
        """Create uncertainty analysis visualization."""
        fig = go.Figure()
        
        for i, pred in enumerate(predictions):
            if pred.uncertainty:
                uncertainty_values = list(pred.uncertainty.values())
                time_steps = list(range(len(uncertainty_values)))
                
                color = self.config.color_palette[i] if i < len(self.config.color_palette) else f'hsl({i * 45 % 360}, 70%, 50%)'
                
                fig.add_trace(go.Scatter(
                    x=time_steps,
                    y=uncertainty_values,
                    mode='lines+markers',
                    name=f'{pred.model_name} Uncertainty',
                    line=dict(color=color, width=2)
                ))
        
        fig.update_layout(
            title="Prediction Uncertainty Over Time",
            xaxis_title="Time Step",
            yaxis_title="Uncertainty",
            template=self.config.theme,
            showlegend=True
        )
        
        return fig


class ModelComparisonChart:
    """Performance comparison visualization across models."""
    
    def __init__(self, config: Optional[PlotConfig] = None):
        self.config = config or PlotConfig()
    
    def create_comparison_matrix(
        self,
        model_results: Dict[str, Dict[str, float]],
        metrics: List[str] = None
    ) -> go.Figure:
        """Create heatmap comparison matrix of model performance."""
        if metrics is None:
            metrics = ['rmse', 'mae', 'ade', 'fde']
        
        # Prepare data for heatmap
        model_names = list(model_results.keys())
        matrix_data = []
        
        for metric in metrics:
            row = []
            for model in model_names:
                value = model_results[model].get(metric, np.nan)
                row.append(value)
            matrix_data.append(row)
        
        fig = go.Figure(data=go.Heatmap(
            z=matrix_data,
            x=model_names,
            y=metrics,
            colorscale='RdYlBu_r',
            text=[[f"{val:.3f}" if not np.isnan(val) else "N/A" for val in row] for row in matrix_data],
            texttemplate="%{text}",
            textfont={"size": 12},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Model Performance Comparison Matrix",
            template=self.config.theme,
            width=self.config.width,
            height=self.config.height
        )
        
        return fig
    
    def create_radar_chart(
        self,
        model_results: Dict[str, Dict[str, float]],
        metrics: List[str] = None
    ) -> go.Figure:
        """Create radar chart for multi-dimensional model comparison."""
        if metrics is None:
            metrics = ['accuracy', 'speed', 'confidence', 'stability']
        
        fig = go.Figure()
        
        for i, (model_name, results) in enumerate(model_results.items()):
            values = [results.get(metric, 0) for metric in metrics]
            values.append(values[0])  # Close the polygon
            
            color = self.config.color_palette[i] if i < len(self.config.color_palette) else f'hsl({i * 45 % 360}, 70%, 50%)'
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics + [metrics[0]],
                fill='toself',
                name=model_name,
                line_color=color
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title="Model Performance Radar Chart",
            template=self.config.theme,
            showlegend=True
        )
        
        return fig
    
    def create_performance_timeline(
        self,
        model_history: Dict[str, List[Tuple[datetime, float]]],
        metric: str = 'accuracy'
    ) -> go.Figure:
        """Create performance timeline showing model evolution."""
        fig = go.Figure()
        
        for i, (model_name, history) in enumerate(model_history.items()):
            timestamps = [point[0] for point in history]
            values = [point[1] for point in history]
            
            color = self.config.color_palette[i] if i < len(self.config.color_palette) else f'hsl({i * 45 % 360}, 70%, 50%)'
            
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=values,
                mode='lines+markers',
                name=model_name,
                line=dict(color=color, width=2),
                marker=dict(size=6)
            ))
        
        fig.update_layout(
            title=f"Model {metric.title()} Over Time",
            xaxis_title="Time",
            yaxis_title=metric.title(),
            template=self.config.theme,
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig


class PerformanceMetricsDisplay:
    """Real-time metrics dashboard for system performance."""
    
    def __init__(self, config: Optional[PlotConfig] = None):
        self.config = config or PlotConfig()
    
    def create_metrics_dashboard(
        self,
        current_metrics: Dict[str, float],
        historical_data: Optional[Dict[str, List[float]]] = None,
        thresholds: Optional[Dict[str, Dict[str, float]]] = None
    ) -> Dict[str, Any]:
        """Create comprehensive metrics dashboard."""
        dashboard = {
            'gauges': self._create_gauge_charts(current_metrics, thresholds),
            'trends': self._create_trend_charts(historical_data) if historical_data else None,
            'alerts': self._generate_alerts(current_metrics, thresholds) if thresholds else []
        }
        
        return dashboard
    
    def _create_gauge_charts(
        self,
        metrics: Dict[str, float],
        thresholds: Optional[Dict[str, Dict[str, float]]] = None
    ) -> Dict[str, go.Figure]:
        """Create gauge charts for key metrics."""
        gauges = {}
        
        for metric_name, value in metrics.items():
            threshold_config = thresholds.get(metric_name, {}) if thresholds else {}
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=value,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': metric_name.replace('_', ' ').title()},
                delta={'reference': threshold_config.get('target', value * 0.9)},
                gauge={
                    'axis': {'range': [None, threshold_config.get('max', value * 1.2)]},
                    'bar': {'color': self._get_metric_color(value, threshold_config)},
                    'steps': [
                        {'range': [0, threshold_config.get('good', value * 0.8)], 'color': "lightgray"},
                        {'range': [threshold_config.get('good', value * 0.8), threshold_config.get('warning', value * 1.1)], 'color': "yellow"},
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': threshold_config.get('critical', value * 1.2)
                    }
                }
            ))
            
            fig.update_layout(
                template=self.config.theme,
                height=300,
                font={'color': "darkblue", 'family': "Arial"}
            )
            
            gauges[metric_name] = fig
        
        return gauges
    
    def _create_trend_charts(self, historical_data: Dict[str, List[float]]) -> Dict[str, go.Figure]:
        """Create trend charts for historical metrics."""
        trends = {}
        
        for metric_name, values in historical_data.items():
            x_values = list(range(len(values)))
            
            fig = go.Figure()
            
            # Main trend line
            fig.add_trace(go.Scatter(
                x=x_values,
                y=values,
                mode='lines+markers',
                name=metric_name,
                line=dict(color=self.config.color_palette[0], width=2)
            ))
            
            # Add trend line
            if len(values) > 1:
                z = np.polyfit(x_values, values, 1)
                trend_line = np.poly1d(z)(x_values)
                
                fig.add_trace(go.Scatter(
                    x=x_values,
                    y=trend_line,
                    mode='lines',
                    name='Trend',
                    line=dict(color='red', width=1, dash='dash')
                ))
            
            fig.update_layout(
                title=f"{metric_name.replace('_', ' ').title()} Trend",
                xaxis_title="Time",
                yaxis_title=metric_name.replace('_', ' ').title(),
                template=self.config.theme,
                showlegend=True
            )
            
            trends[metric_name] = fig
        
        return trends
    
    def _get_metric_color(self, value: float, thresholds: Dict[str, float]) -> str:
        """Determine gauge color based on thresholds."""
        if 'critical' in thresholds and value >= thresholds['critical']:
            return "red"
        elif 'warning' in thresholds and value >= thresholds['warning']:
            return "orange"
        elif 'good' in thresholds and value <= thresholds['good']:
            return "green"
        else:
            return "blue"
    
    def _generate_alerts(
        self,
        metrics: Dict[str, float],
        thresholds: Dict[str, Dict[str, float]]
    ) -> List[Dict[str, str]]:
        """Generate alerts based on threshold violations."""
        alerts = []
        
        for metric_name, value in metrics.items():
            threshold_config = thresholds.get(metric_name, {})
            
            if 'critical' in threshold_config and value >= threshold_config['critical']:
                alerts.append({
                    'level': 'critical',
                    'metric': metric_name,
                    'message': f"{metric_name} is critical: {value:.2f} >= {threshold_config['critical']:.2f}",
                    'color': 'red'
                })
            elif 'warning' in threshold_config and value >= threshold_config['warning']:
                alerts.append({
                    'level': 'warning',
                    'metric': metric_name,
                    'message': f"{metric_name} is elevated: {value:.2f} >= {threshold_config['warning']:.2f}",
                    'color': 'orange'
                })
        
        return alerts