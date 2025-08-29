"""
Custom dashboard creation system for trajectory prediction analysis.

This module provides:
- DashboardBuilder: Create custom dashboards with drag-and-drop interface
- DashboardTemplate: Pre-built dashboard templates
- WidgetLibrary: Collection of visualization widgets
- DashboardExporter: Export dashboards to different formats
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict, field
import json
from datetime import datetime
from pathlib import Path
import uuid
import base64
from io import BytesIO

from ..data.schemas import TrajectoryData
from ..api.models import TrajectoryResponse
from .components import (
    TrajectoryPlotter, PredictionVisualizer, ModelComparisonChart, 
    PerformanceMetricsDisplay, PlotConfig
)
from .analysis import (
    TrajectoryClusterAnalyzer, AnomalyDetector, FeatureImportanceVisualizer,
    ModelInterpretabilityTools
)
from .reports import AutomatedReportGenerator, ReportConfig


@dataclass
class WidgetConfig:
    """Configuration for dashboard widgets."""
    
    widget_id: str
    widget_type: str
    title: str
    position: Tuple[int, int] = (0, 0)  # (row, column)
    size: Tuple[int, int] = (1, 1)  # (height, width)
    parameters: Dict[str, Any] = field(default_factory=dict)
    data_source: Optional[str] = None
    refresh_interval: Optional[int] = None
    visible: bool = True


@dataclass
class DashboardTemplate:
    """Template for dashboard layout and configuration."""
    
    template_id: str
    name: str
    description: str
    widgets: List[WidgetConfig] = field(default_factory=list)
    layout: Dict[str, Any] = field(default_factory=dict)
    theme: str = "plotly_white"
    created_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)


class WidgetLibrary:
    """Library of available widgets for dashboard creation."""
    
    def __init__(self):
        self.widgets = self._initialize_widgets()
    
    def _initialize_widgets(self) -> Dict[str, Dict[str, Any]]:
        """Initialize available widgets with metadata."""
        return {
            'trajectory_plot': {
                'name': 'Trajectory Visualization',
                'description': 'Interactive trajectory plot with prediction overlays',
                'category': 'Visualization',
                'parameters': {
                    'show_predictions': {'type': 'boolean', 'default': True},
                    'show_uncertainty': {'type': 'boolean', 'default': False},
                    'color_scheme': {'type': 'select', 'options': ['viridis', 'plasma', 'set1'], 'default': 'set1'},
                    'max_trajectories': {'type': 'number', 'default': 50, 'min': 1, 'max': 500}
                },
                'size_hint': (2, 2),
                'data_requirements': ['trajectories', 'predictions']
            },
            'model_comparison': {
                'name': 'Model Performance Comparison',
                'description': 'Compare performance metrics across models',
                'category': 'Analysis',
                'parameters': {
                    'metrics': {'type': 'multiselect', 'options': ['rmse', 'mae', 'ade', 'fde'], 'default': ['rmse', 'mae']},
                    'chart_type': {'type': 'select', 'options': ['bar', 'radar', 'heatmap'], 'default': 'bar'},
                    'show_confidence': {'type': 'boolean', 'default': True}
                },
                'size_hint': (1, 2),
                'data_requirements': ['evaluation_results']
            },
            'performance_gauges': {
                'name': 'Performance Gauges',
                'description': 'Real-time performance metrics as gauge charts',
                'category': 'Monitoring',
                'parameters': {
                    'metrics': {'type': 'multiselect', 'options': ['accuracy', 'latency', 'throughput'], 'default': ['accuracy']},
                    'thresholds': {'type': 'json', 'default': '{"accuracy": {"warning": 0.8, "critical": 0.6}}'},
                    'update_interval': {'type': 'number', 'default': 5, 'min': 1, 'max': 60}
                },
                'size_hint': (1, 1),
                'data_requirements': ['performance_metrics']
            },
            'cluster_analysis': {
                'name': 'Trajectory Clustering',
                'description': 'Cluster analysis and pattern recognition',
                'category': 'Analysis',
                'parameters': {
                    'method': {'type': 'select', 'options': ['kmeans', 'dbscan', 'hierarchical'], 'default': 'kmeans'},
                    'n_clusters': {'type': 'number', 'default': 5, 'min': 2, 'max': 20},
                    'show_centers': {'type': 'boolean', 'default': True},
                    'color_clusters': {'type': 'boolean', 'default': True}
                },
                'size_hint': (2, 2),
                'data_requirements': ['trajectories']
            },
            'anomaly_detection': {
                'name': 'Anomaly Detection',
                'description': 'Detect and visualize trajectory anomalies',
                'category': 'Analysis',
                'parameters': {
                    'method': {'type': 'select', 'options': ['isolation_forest', 'one_class_svm'], 'default': 'isolation_forest'},
                    'contamination': {'type': 'number', 'default': 0.1, 'min': 0.01, 'max': 0.5},
                    'highlight_anomalies': {'type': 'boolean', 'default': True}
                },
                'size_hint': (2, 1),
                'data_requirements': ['trajectories']
            },
            'feature_importance': {
                'name': 'Feature Importance',
                'description': 'Analyze and visualize feature importance',
                'category': 'Analysis',
                'parameters': {
                    'method': {'type': 'select', 'options': ['random_forest', 'mutual_info'], 'default': 'random_forest'},
                    'top_n': {'type': 'number', 'default': 15, 'min': 5, 'max': 50},
                    'chart_type': {'type': 'select', 'options': ['bar', 'pie', 'radar'], 'default': 'bar'}
                },
                'size_hint': (1, 2),
                'data_requirements': ['features', 'targets']
            },
            'prediction_timeline': {
                'name': 'Prediction Timeline',
                'description': 'Time series view of prediction accuracy',
                'category': 'Monitoring',
                'parameters': {
                    'time_window': {'type': 'select', 'options': ['1h', '6h', '24h', '7d'], 'default': '24h'},
                    'metrics': {'type': 'multiselect', 'options': ['accuracy', 'confidence', 'latency'], 'default': ['accuracy']},
                    'show_trend': {'type': 'boolean', 'default': True}
                },
                'size_hint': (1, 3),
                'data_requirements': ['historical_metrics']
            },
            'data_quality_summary': {
                'name': 'Data Quality Summary',
                'description': 'Overview of data quality metrics',
                'category': 'Monitoring',
                'parameters': {
                    'show_completeness': {'type': 'boolean', 'default': True},
                    'show_consistency': {'type': 'boolean', 'default': True},
                    'show_accuracy': {'type': 'boolean', 'default': True},
                    'alert_threshold': {'type': 'number', 'default': 0.8, 'min': 0.0, 'max': 1.0}
                },
                'size_hint': (1, 1),
                'data_requirements': ['data_quality_metrics']
            },
            'model_interpretability': {
                'name': 'Model Interpretability',
                'description': 'SHAP values and model explanation plots',
                'category': 'Analysis',
                'parameters': {
                    'explanation_type': {'type': 'select', 'options': ['global', 'local'], 'default': 'global'},
                    'max_features': {'type': 'number', 'default': 10, 'min': 5, 'max': 25},
                    'sample_size': {'type': 'number', 'default': 100, 'min': 10, 'max': 1000}
                },
                'size_hint': (2, 1),
                'data_requirements': ['model', 'features']
            }
        }
    
    def get_widget_info(self, widget_type: str) -> Optional[Dict[str, Any]]:
        """Get widget information by type."""
        return self.widgets.get(widget_type)
    
    def get_categories(self) -> List[str]:
        """Get all widget categories."""
        categories = set()
        for widget in self.widgets.values():
            categories.add(widget['category'])
        return sorted(list(categories))
    
    def get_widgets_by_category(self, category: str) -> Dict[str, Dict[str, Any]]:
        """Get widgets filtered by category."""
        return {
            widget_type: widget_info 
            for widget_type, widget_info in self.widgets.items()
            if widget_info['category'] == category
        }


class DashboardBuilder:
    """Interactive dashboard builder with drag-and-drop interface."""
    
    def __init__(self, widget_library: WidgetLibrary):
        self.widget_library = widget_library
        self.current_dashboard = None
        self.available_data_sources = {}
        
        # Initialize session state
        if 'dashboard_widgets' not in st.session_state:
            st.session_state.dashboard_widgets = []
        if 'dashboard_layout' not in st.session_state:
            st.session_state.dashboard_layout = {'rows': 3, 'cols': 3}
        if 'selected_widget' not in st.session_state:
            st.session_state.selected_widget = None
    
    def create_dashboard_interface(self) -> None:
        """Create the main dashboard builder interface."""
        st.title("🛠️ Custom Dashboard Builder")
        
        # Sidebar for widget library and settings
        with st.sidebar:
            self._create_widget_library_sidebar()
            self._create_dashboard_settings_sidebar()
        
        # Main area for dashboard preview and editing
        col1, col2 = st.columns([3, 1])
        
        with col1:
            self._create_dashboard_preview()
        
        with col2:
            self._create_widget_editor()
    
    def _create_widget_library_sidebar(self) -> None:
        """Create widget library in sidebar."""
        st.sidebar.markdown("## 📊 Widget Library")
        
        # Category filter
        categories = self.widget_library.get_categories()
        selected_category = st.sidebar.selectbox(
            "Filter by category",
            ["All"] + categories,
            index=0
        )
        
        # Widget list
        if selected_category == "All":
            widgets = self.widget_library.widgets
        else:
            widgets = self.widget_library.get_widgets_by_category(selected_category)
        
        st.sidebar.markdown("### Available Widgets")
        for widget_type, widget_info in widgets.items():
            with st.sidebar.expander(f"📈 {widget_info['name']}"):
                st.write(widget_info['description'])
                st.write(f"**Category:** {widget_info['category']}")
                st.write(f"**Size:** {widget_info['size_hint'][0]}×{widget_info['size_hint'][1]}")
                
                if st.button(f"Add {widget_info['name']}", key=f"add_{widget_type}"):
                    self._add_widget_to_dashboard(widget_type, widget_info)
    
    def _create_dashboard_settings_sidebar(self) -> None:
        """Create dashboard settings in sidebar."""
        st.sidebar.markdown("## ⚙️ Dashboard Settings")
        
        # Layout settings
        with st.sidebar.expander("Layout Settings"):
            rows = st.number_input("Rows", min_value=1, max_value=10, value=3)
            cols = st.number_input("Columns", min_value=1, max_value=6, value=3)
            
            if st.button("Update Layout"):
                st.session_state.dashboard_layout = {'rows': rows, 'cols': cols}
                st.experimental_rerun()
        
        # Dashboard actions
        with st.sidebar.expander("Dashboard Actions"):
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📥 Load"):
                    self._load_dashboard_dialog()
                
                if st.button("🗑️ Clear"):
                    st.session_state.dashboard_widgets = []
                    st.experimental_rerun()
            
            with col2:
                if st.button("💾 Save"):
                    self._save_dashboard_dialog()
                
                if st.button("📤 Export"):
                    self._export_dashboard_dialog()
    
    def _create_dashboard_preview(self) -> None:
        """Create dashboard preview area."""
        st.markdown("### 📋 Dashboard Preview")
        
        layout = st.session_state.dashboard_layout
        widgets = st.session_state.dashboard_widgets
        
        if not widgets:
            st.info("🎯 Add widgets from the library to start building your dashboard!")
            return
        
        # Create grid layout
        grid_container = st.container()
        
        with grid_container:
            # Create grid based on layout settings
            for row in range(layout['rows']):
                cols = st.columns(layout['cols'])
                
                for col_idx, col in enumerate(cols):
                    # Find widget for this position
                    widget_at_position = None
                    for widget in widgets:
                        if (widget['position'][0] == row and 
                            widget['position'][1] == col_idx):
                            widget_at_position = widget
                            break
                    
                    with col:
                        if widget_at_position:
                            self._render_widget_preview(widget_at_position)
                        else:
                            # Empty slot
                            st.markdown(
                                f"""
                                <div style="
                                    border: 2px dashed #ccc;
                                    padding: 20px;
                                    text-align: center;
                                    color: #999;
                                    border-radius: 5px;
                                    min-height: 100px;
                                ">
                                    Drop widget here<br>
                                    <small>({row}, {col_idx})</small>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
    
    def _create_widget_editor(self) -> None:
        """Create widget editor panel."""
        st.markdown("### 🔧 Widget Editor")
        
        widgets = st.session_state.dashboard_widgets
        
        if not widgets:
            st.info("Select a widget to edit its properties")
            return
        
        # Widget selector
        widget_names = [f"{w['title']} ({w['widget_type']})" for w in widgets]
        if widget_names:
            selected_idx = st.selectbox(
                "Select widget to edit",
                range(len(widget_names)),
                format_func=lambda x: widget_names[x]
            )
            
            selected_widget = widgets[selected_idx]
            st.session_state.selected_widget = selected_widget
            
            # Widget properties editor
            with st.expander("Widget Properties", expanded=True):
                self._create_widget_properties_editor(selected_widget, selected_idx)
    
    def _add_widget_to_dashboard(self, widget_type: str, widget_info: Dict[str, Any]) -> None:
        """Add a widget to the dashboard."""
        widget_id = str(uuid.uuid4())
        
        # Find available position
        layout = st.session_state.dashboard_layout
        position = self._find_available_position(layout)
        
        widget_config = WidgetConfig(
            widget_id=widget_id,
            widget_type=widget_type,
            title=widget_info['name'],
            position=position,
            size=widget_info['size_hint'],
            parameters=self._get_default_parameters(widget_info),
        )
        
        st.session_state.dashboard_widgets.append(asdict(widget_config))
        st.experimental_rerun()
    
    def _find_available_position(self, layout: Dict[str, int]) -> Tuple[int, int]:
        """Find the next available position in the grid."""
        widgets = st.session_state.dashboard_widgets
        occupied_positions = {(w['position'][0], w['position'][1]) for w in widgets}
        
        for row in range(layout['rows']):
            for col in range(layout['cols']):
                if (row, col) not in occupied_positions:
                    return (row, col)
        
        # If no position available, expand grid
        return (layout['rows'], 0)
    
    def _get_default_parameters(self, widget_info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract default parameters from widget info."""
        parameters = {}
        for param_name, param_config in widget_info.get('parameters', {}).items():
            parameters[param_name] = param_config.get('default')
        return parameters
    
    def _render_widget_preview(self, widget_config: Dict[str, Any]) -> None:
        """Render a widget preview."""
        widget_type = widget_config['widget_type']
        widget_info = self.widget_library.get_widget_info(widget_type)
        
        # Widget container with controls
        with st.container():
            # Widget header
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.markdown(f"**{widget_config['title']}**")
            
            with col2:
                if st.button("⚙️", key=f"edit_{widget_config['widget_id']}", 
                           help="Edit widget"):
                    st.session_state.selected_widget = widget_config
            
            with col3:
                if st.button("❌", key=f"delete_{widget_config['widget_id']}", 
                           help="Delete widget"):
                    self._delete_widget(widget_config['widget_id'])
                    st.experimental_rerun()
            
            # Widget content (placeholder)
            if widget_type == 'trajectory_plot':
                self._render_trajectory_plot_preview(widget_config)
            elif widget_type == 'model_comparison':
                self._render_model_comparison_preview(widget_config)
            elif widget_type == 'performance_gauges':
                self._render_performance_gauges_preview(widget_config)
            else:
                # Generic preview
                st.info(f"Preview for {widget_info['name']} widget")
    
    def _render_trajectory_plot_preview(self, widget_config: Dict[str, Any]) -> None:
        """Render trajectory plot widget preview."""
        # Create sample trajectory plot
        fig = go.Figure()
        
        # Sample trajectory data
        x = np.linspace(0, 10, 50) + np.random.normal(0, 0.1, 50)
        y = np.sin(x) + np.random.normal(0, 0.1, 50)
        
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='lines+markers',
            name='Sample Trajectory',
            line=dict(width=2)
        ))
        
        # Add prediction if enabled
        if widget_config['parameters'].get('show_predictions', True):
            pred_x = np.linspace(10, 15, 25)
            pred_y = np.sin(pred_x) + np.random.normal(0, 0.15, 25)
            
            fig.add_trace(go.Scatter(
                x=pred_x, y=pred_y,
                mode='lines+markers',
                name='Prediction',
                line=dict(dash='dash', width=2)
            ))
        
        fig.update_layout(
            title="Trajectory Visualization Preview",
            height=300,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_model_comparison_preview(self, widget_config: Dict[str, Any]) -> None:
        """Render model comparison widget preview."""
        # Sample model performance data
        models = ['Model A', 'Model B', 'Model C']
        metrics = widget_config['parameters'].get('metrics', ['rmse', 'mae'])
        
        if widget_config['parameters'].get('chart_type', 'bar') == 'bar':
            fig = go.Figure()
            
            for i, metric in enumerate(metrics):
                values = np.random.uniform(0.1, 1.0, len(models))
                fig.add_trace(go.Bar(
                    name=metric.upper(),
                    x=models,
                    y=values
                ))
            
            fig.update_layout(
                title="Model Performance Comparison",
                height=300,
                barmode='group'
            )
        
        else:  # radar chart
            fig = go.Figure()
            
            for model in models:
                values = [np.random.uniform(0.1, 1.0) for _ in metrics]
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=metrics,
                    fill='toself',
                    name=model
                ))
            
            fig.update_layout(
                title="Model Performance Radar",
                height=300,
                polar=dict(
                    radialaxis=dict(range=[0, 1])
                )
            )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_performance_gauges_preview(self, widget_config: Dict[str, Any]) -> None:
        """Render performance gauges widget preview."""
        metrics = widget_config['parameters'].get('metrics', ['accuracy'])
        
        cols = st.columns(len(metrics))
        
        for i, metric in enumerate(metrics):
            with cols[i]:
                value = np.random.uniform(0.6, 0.95)
                
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=value,
                    title={'text': metric.title()},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={'axis': {'range': [None, 1]},
                           'bar': {'color': "darkblue"},
                           'steps': [{'range': [0, 0.6], 'color': "lightgray"},
                                   {'range': [0.6, 0.8], 'color': "yellow"}],
                           'threshold': {'line': {'color': "red", 'width': 4},
                                       'thickness': 0.75,
                                       'value': 0.9}}
                ))
                
                fig.update_layout(height=200)
                st.plotly_chart(fig, use_container_width=True)
    
    def _create_widget_properties_editor(self, widget_config: Dict[str, Any], widget_idx: int) -> None:
        """Create widget properties editor."""
        widget_type = widget_config['widget_type']
        widget_info = self.widget_library.get_widget_info(widget_type)
        
        if not widget_info:
            st.error(f"Unknown widget type: {widget_type}")
            return
        
        # Basic properties
        new_title = st.text_input("Title", value=widget_config.get('title', ''))
        
        # Position
        col1, col2 = st.columns(2)
        with col1:
            new_row = st.number_input("Row", min_value=0, value=widget_config['position'][0])
        with col2:
            new_col = st.number_input("Column", min_value=0, value=widget_config['position'][1])
        
        # Widget-specific parameters
        st.markdown("**Widget Parameters**")
        new_parameters = {}
        
        for param_name, param_config in widget_info.get('parameters', {}).items():
            current_value = widget_config['parameters'].get(param_name, param_config.get('default'))
            
            if param_config['type'] == 'boolean':
                new_parameters[param_name] = st.checkbox(
                    param_name.replace('_', ' ').title(),
                    value=current_value
                )
            
            elif param_config['type'] == 'number':
                new_parameters[param_name] = st.number_input(
                    param_name.replace('_', ' ').title(),
                    min_value=param_config.get('min', 0),
                    max_value=param_config.get('max', 100),
                    value=current_value
                )
            
            elif param_config['type'] == 'select':
                options = param_config.get('options', [])
                current_idx = 0
                if current_value in options:
                    current_idx = options.index(current_value)
                
                new_parameters[param_name] = st.selectbox(
                    param_name.replace('_', ' ').title(),
                    options,
                    index=current_idx
                )
            
            elif param_config['type'] == 'multiselect':
                options = param_config.get('options', [])
                default_selection = current_value if isinstance(current_value, list) else [current_value]
                
                new_parameters[param_name] = st.multiselect(
                    param_name.replace('_', ' ').title(),
                    options,
                    default=default_selection
                )
            
            elif param_config['type'] == 'json':
                new_parameters[param_name] = st.text_area(
                    param_name.replace('_', ' ').title(),
                    value=json.dumps(current_value) if current_value else param_config.get('default', '{}'),
                    help="Enter valid JSON"
                )
        
        # Update button
        if st.button("Update Widget", key=f"update_{widget_config['widget_id']}"):
            # Update widget configuration
            st.session_state.dashboard_widgets[widget_idx].update({
                'title': new_title,
                'position': (new_row, new_col),
                'parameters': new_parameters
            })
            st.success("Widget updated successfully!")
            st.experimental_rerun()
    
    def _delete_widget(self, widget_id: str) -> None:
        """Delete a widget from the dashboard."""
        st.session_state.dashboard_widgets = [
            w for w in st.session_state.dashboard_widgets 
            if w['widget_id'] != widget_id
        ]
    
    def _save_dashboard_dialog(self) -> None:
        """Show save dashboard dialog."""
        if not st.session_state.dashboard_widgets:
            st.warning("No widgets to save!")
            return
        
        with st.modal("Save Dashboard"):
            st.markdown("### 💾 Save Dashboard")
            
            name = st.text_input("Dashboard Name", placeholder="My Custom Dashboard")
            description = st.text_area("Description", placeholder="Dashboard description...")
            tags = st.text_input("Tags", placeholder="analysis, monitoring, custom", 
                                help="Comma-separated tags")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Cancel"):
                    st.experimental_rerun()
            
            with col2:
                if st.button("Save") and name:
                    template = DashboardTemplate(
                        template_id=str(uuid.uuid4()),
                        name=name,
                        description=description,
                        widgets=[WidgetConfig(**w) for w in st.session_state.dashboard_widgets],
                        layout=st.session_state.dashboard_layout,
                        tags=[tag.strip() for tag in tags.split(',') if tag.strip()]
                    )
                    
                    # Save to file (in practice, this would save to a database)
                    self._save_template_to_file(template)
                    st.success(f"Dashboard '{name}' saved successfully!")
                    st.experimental_rerun()
    
    def _load_dashboard_dialog(self) -> None:
        """Show load dashboard dialog."""
        with st.modal("Load Dashboard"):
            st.markdown("### 📥 Load Dashboard")
            
            # List available templates (placeholder)
            templates = self._get_available_templates()
            
            if not templates:
                st.info("No saved dashboards found.")
                return
            
            selected_template = st.selectbox(
                "Select dashboard to load",
                templates,
                format_func=lambda x: f"{x.name} - {x.description[:50]}..."
            )
            
            if selected_template:
                st.markdown("**Preview:**")
                st.write(f"Name: {selected_template.name}")
                st.write(f"Description: {selected_template.description}")
                st.write(f"Widgets: {len(selected_template.widgets)}")
                st.write(f"Created: {selected_template.created_at.strftime('%Y-%m-%d %H:%M')}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Cancel"):
                    st.experimental_rerun()
            
            with col2:
                if st.button("Load") and selected_template:
                    self._load_template(selected_template)
                    st.success(f"Dashboard '{selected_template.name}' loaded successfully!")
                    st.experimental_rerun()
    
    def _export_dashboard_dialog(self) -> None:
        """Show export dashboard dialog."""
        if not st.session_state.dashboard_widgets:
            st.warning("No dashboard to export!")
            return
        
        with st.modal("Export Dashboard"):
            st.markdown("### 📤 Export Dashboard")
            
            export_format = st.selectbox(
                "Export Format",
                ["JSON", "HTML Report", "Python Code", "Streamlit App"]
            )
            
            include_data = st.checkbox("Include sample data", value=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Cancel"):
                    st.experimental_rerun()
            
            with col2:
                if st.button("Export"):
                    export_content = self._export_dashboard(export_format, include_data)
                    
                    # Provide download link
                    if export_format == "JSON":
                        st.download_button(
                            "Download JSON",
                            export_content,
                            file_name="dashboard_config.json",
                            mime="application/json"
                        )
                    elif export_format == "HTML Report":
                        st.download_button(
                            "Download HTML",
                            export_content,
                            file_name="dashboard_report.html",
                            mime="text/html"
                        )
                    else:
                        st.code(export_content, language='python')
    
    def _save_template_to_file(self, template: DashboardTemplate) -> None:
        """Save template to file (placeholder implementation)."""
        # In practice, this would save to a database or file system
        templates_dir = Path("dashboards")
        templates_dir.mkdir(exist_ok=True)
        
        template_file = templates_dir / f"{template.template_id}.json"
        
        with open(template_file, 'w') as f:
            json.dump(asdict(template), f, indent=2, default=str)
    
    def _get_available_templates(self) -> List[DashboardTemplate]:
        """Get available dashboard templates (placeholder)."""
        # In practice, this would load from a database or file system
        templates = []
        templates_dir = Path("dashboards")
        
        if templates_dir.exists():
            for template_file in templates_dir.glob("*.json"):
                try:
                    with open(template_file) as f:
                        template_data = json.load(f)
                        template = DashboardTemplate(**template_data)
                        templates.append(template)
                except Exception as e:
                    st.warning(f"Error loading template {template_file}: {e}")
        
        return templates
    
    def _load_template(self, template: DashboardTemplate) -> None:
        """Load a dashboard template."""
        st.session_state.dashboard_widgets = [asdict(w) for w in template.widgets]
        st.session_state.dashboard_layout = template.layout
    
    def _export_dashboard(self, format_type: str, include_data: bool) -> str:
        """Export dashboard in specified format."""
        if format_type == "JSON":
            export_data = {
                'widgets': st.session_state.dashboard_widgets,
                'layout': st.session_state.dashboard_layout,
                'exported_at': datetime.now().isoformat()
            }
            return json.dumps(export_data, indent=2)
        
        elif format_type == "HTML Report":
            return self._generate_html_dashboard(include_data)
        
        elif format_type == "Python Code":
            return self._generate_python_code()
        
        elif format_type == "Streamlit App":
            return self._generate_streamlit_app()
        
        return "Export format not supported"
    
    def _generate_html_dashboard(self, include_data: bool) -> str:
        """Generate HTML dashboard."""
        widgets = st.session_state.dashboard_widgets
        layout = st.session_state.dashboard_layout
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Custom Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .dashboard-grid {{ 
                    display: grid; 
                    grid-template-columns: repeat({layout['cols']}, 1fr);
                    grid-template-rows: repeat({layout['rows']}, auto);
                    gap: 20px; 
                }}
                .widget {{ 
                    border: 1px solid #ddd; 
                    padding: 15px; 
                    border-radius: 5px;
                    background: white;
                }}
            </style>
        </head>
        <body>
            <h1>Custom Dashboard</h1>
            <div class="dashboard-grid">
        """
        
        for widget in widgets:
            html_content += f"""
                <div class="widget">
                    <h3>{widget['title']}</h3>
                    <p>Widget type: {widget['widget_type']}</p>
                    <p>Position: {widget['position']}</p>
                    <div id="widget_{widget['widget_id']}"></div>
                </div>
            """
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        return html_content
    
    def _generate_python_code(self) -> str:
        """Generate Python code for the dashboard."""
        widgets = st.session_state.dashboard_widgets
        
        code = '''
import streamlit as st
import plotly.graph_objects as go
import numpy as np

# Dashboard configuration
st.set_page_config(page_title="Custom Dashboard", layout="wide")
st.title("Custom Dashboard")

# Widget implementations
'''
        
        for widget in widgets:
            widget_type = widget['widget_type']
            title = widget['title']
            
            code += f'''
# {title} Widget
with st.container():
    st.subheader("{title}")
    # Implementation for {widget_type} widget
    st.info("Widget implementation goes here")

'''
        
        return code
    
    def _generate_streamlit_app(self) -> str:
        """Generate complete Streamlit app code."""
        return self._generate_python_code() + """
# Run with: streamlit run dashboard.py
if __name__ == "__main__":
    pass
"""


class CustomDashboardManager:
    """Manager for custom dashboard functionality."""
    
    def __init__(self):
        self.widget_library = WidgetLibrary()
        self.dashboard_builder = DashboardBuilder(self.widget_library)
        self.report_generator = AutomatedReportGenerator()
    
    def create_dashboard_page(self) -> None:
        """Create the main custom dashboard page."""
        st.set_page_config(
            page_title="Custom Dashboard Builder",
            page_icon="🛠️",
            layout="wide"
        )
        
        # Navigation tabs
        tab1, tab2, tab3 = st.tabs(["🛠️ Builder", "📊 Templates", "📋 Gallery"])
        
        with tab1:
            self.dashboard_builder.create_dashboard_interface()
        
        with tab2:
            self._create_templates_page()
        
        with tab3:
            self._create_gallery_page()
    
    def _create_templates_page(self) -> None:
        """Create dashboard templates page."""
        st.markdown("## 📊 Dashboard Templates")
        st.markdown("Choose from pre-built dashboard templates to get started quickly.")
        
        # Template categories
        template_categories = {
            "Monitoring": [
                {
                    "name": "System Health Dashboard",
                    "description": "Real-time monitoring of system performance and health metrics",
                    "widgets": ["performance_gauges", "prediction_timeline", "data_quality_summary"],
                    "preview": "monitoring_preview.png"
                },
                {
                    "name": "Alert Dashboard",
                    "description": "Centralized alert monitoring and incident management",
                    "widgets": ["performance_gauges", "anomaly_detection"],
                    "preview": "alert_preview.png"
                }
            ],
            "Analysis": [
                {
                    "name": "Model Analysis Dashboard",
                    "description": "Comprehensive model performance analysis and comparison",
                    "widgets": ["model_comparison", "feature_importance", "model_interpretability"],
                    "preview": "analysis_preview.png"
                },
                {
                    "name": "Data Exploration Dashboard",
                    "description": "Interactive data exploration and pattern discovery",
                    "widgets": ["trajectory_plot", "cluster_analysis", "anomaly_detection"],
                    "preview": "exploration_preview.png"
                }
            ],
            "Executive": [
                {
                    "name": "Executive Summary Dashboard",
                    "description": "High-level overview for executive reporting",
                    "widgets": ["model_comparison", "performance_gauges", "data_quality_summary"],
                    "preview": "executive_preview.png"
                }
            ]
        }
        
        for category, templates in template_categories.items():
            st.markdown(f"### {category}")
            
            cols = st.columns(len(templates))
            
            for i, template in enumerate(templates):
                with cols[i]:
                    with st.container():
                        st.markdown(f"**{template['name']}**")
                        st.write(template['description'])
                        st.write(f"Widgets: {len(template['widgets'])}")
                        
                        if st.button(f"Use Template", key=f"template_{category}_{i}"):
                            self._apply_template(template)
                            st.success("Template applied successfully!")
                            st.experimental_rerun()
    
    def _create_gallery_page(self) -> None:
        """Create dashboard gallery page."""
        st.markdown("## 📋 Dashboard Gallery")
        st.markdown("Browse dashboards created by the community.")
        
        # Sample gallery items
        gallery_items = [
            {
                "name": "Autonomous Vehicle Monitoring",
                "author": "AI Team",
                "description": "Real-time monitoring dashboard for autonomous vehicle fleet",
                "tags": ["monitoring", "real-time", "fleet"],
                "likes": 42,
                "created": "2024-01-15"
            },
            {
                "name": "Traffic Pattern Analysis",
                "author": "Data Science Team",
                "description": "Comprehensive analysis of traffic patterns and anomalies",
                "tags": ["analysis", "patterns", "traffic"],
                "likes": 38,
                "created": "2024-01-12"
            },
            {
                "name": "Model Performance Tracker",
                "author": "ML Engineering",
                "description": "Track model performance across different scenarios",
                "tags": ["performance", "ml", "tracking"],
                "likes": 35,
                "created": "2024-01-10"
            }
        ]
        
        for item in gallery_items:
            with st.expander(f"📊 {item['name']} - by {item['author']}"):
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.write(item['description'])
                    
                    # Tags
                    tag_html = " ".join([f'<span style="background-color: #e1f5fe; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; margin-right: 4px;">{tag}</span>' for tag in item['tags']])
                    st.markdown(tag_html, unsafe_allow_html=True)
                
                with col2:
                    st.metric("👍 Likes", item['likes'])
                    st.write(f"📅 {item['created']}")
                
                with col3:
                    if st.button("Preview", key=f"preview_{item['name']}"):
                        st.info("Preview functionality coming soon!")
                    
                    if st.button("Use", key=f"use_{item['name']}"):
                        st.info("Import functionality coming soon!")
    
    def _apply_template(self, template: Dict[str, Any]) -> None:
        """Apply a dashboard template."""
        # Clear existing widgets
        st.session_state.dashboard_widgets = []
        
        # Add template widgets
        for i, widget_type in enumerate(template['widgets']):
            widget_info = self.widget_library.get_widget_info(widget_type)
            if widget_info:
                widget_id = str(uuid.uuid4())
                
                widget_config = WidgetConfig(
                    widget_id=widget_id,
                    widget_type=widget_type,
                    title=widget_info['name'],
                    position=(i // 2, i % 2),  # Simple 2-column layout
                    size=widget_info['size_hint'],
                    parameters=self.dashboard_builder._get_default_parameters(widget_info),
                )
                
                st.session_state.dashboard_widgets.append(asdict(widget_config))
        
        # Update layout
        st.session_state.dashboard_layout = {'rows': 3, 'cols': 2}