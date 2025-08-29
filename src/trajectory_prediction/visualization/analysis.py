"""
Advanced analysis and visualization tools for trajectory prediction.

This module provides:
- TrajectoryClusterAnalyzer: Clustering and pattern analysis
- AnomalyDetector: Anomaly detection and visualization
- FeatureImportanceVisualizer: Feature analysis and selection
- ModelInterpretabilityTools: SHAP, LIME, and other interpretability methods
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
from datetime import datetime
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import silhouette_score
import scipy.stats as stats
from scipy.spatial.distance import pdist, squareform
import warnings

from ..data.schemas import TrajectoryData
from ..models.features import TrajectoryFeatureExtractor
from .components import PlotConfig


@dataclass
class ClusteringConfig:
    """Configuration for clustering analysis."""
    
    method: str = "kmeans"  # kmeans, dbscan, hierarchical
    n_clusters: int = 5
    eps: float = 0.5
    min_samples: int = 5
    normalize: bool = True
    reduce_dims: bool = True
    n_components: int = 2


@dataclass
class AnomalyConfig:
    """Configuration for anomaly detection."""
    
    method: str = "isolation_forest"  # isolation_forest, one_class_svm, elliptic_envelope
    contamination: float = 0.1
    nu: float = 0.05
    gamma: str = "scale"
    normalize: bool = True


class TrajectoryClusterAnalyzer:
    """Clustering and pattern analysis for trajectories."""
    
    def __init__(
        self,
        config: Optional[ClusteringConfig] = None,
        plot_config: Optional[PlotConfig] = None
    ):
        self.config = config or ClusteringConfig()
        self.plot_config = plot_config or PlotConfig()
        self.feature_extractor = TrajectoryFeatureExtractor()
        self.scaler = StandardScaler() if self.config.normalize else None
        self.pca = None
        self.tsne = None
        
    def analyze_trajectory_patterns(
        self,
        trajectories: List[TrajectoryData],
        features: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive trajectory pattern analysis.
        
        Args:
            trajectories: List of trajectory data
            features: Pre-computed features (optional)
            
        Returns:
            Dictionary containing clustering results and visualizations
        """
        # Extract features if not provided
        if features is None:
            with st.spinner("Extracting trajectory features..."):
                features = self._extract_features(trajectories)
        
        # Normalize features
        if self.config.normalize and features is not None:
            features = self.scaler.fit_transform(features)
        
        # Dimensionality reduction
        reduced_features = self._reduce_dimensions(features)
        
        # Perform clustering
        with st.spinner(f"Performing {self.config.method} clustering..."):
            cluster_labels, cluster_centers = self._perform_clustering(features)
        
        # Evaluate clustering quality
        quality_metrics = self._evaluate_clustering(features, cluster_labels)
        
        # Create visualizations
        visualizations = self._create_clustering_visualizations(
            trajectories, features, reduced_features, cluster_labels, cluster_centers
        )
        
        # Analyze cluster characteristics
        cluster_analysis = self._analyze_clusters(trajectories, cluster_labels)
        
        return {
            'cluster_labels': cluster_labels,
            'cluster_centers': cluster_centers,
            'quality_metrics': quality_metrics,
            'visualizations': visualizations,
            'cluster_analysis': cluster_analysis,
            'reduced_features': reduced_features
        }
    
    def _extract_features(self, trajectories: List[TrajectoryData]) -> np.ndarray:
        """Extract features from trajectories."""
        features_list = []
        
        for trajectory in trajectories:
            try:
                features = self.feature_extractor.extract_features(trajectory)
                # Convert features dict to array
                feature_array = []
                for category, feature_dict in features.items():
                    if isinstance(feature_dict, dict):
                        feature_array.extend(list(feature_dict.values()))
                    else:
                        feature_array.append(feature_dict)
                features_list.append(feature_array)
            except Exception as e:
                st.warning(f"Error extracting features from trajectory: {e}")
                continue
        
        return np.array(features_list) if features_list else np.array([])
    
    def _reduce_dimensions(self, features: np.ndarray) -> Optional[np.ndarray]:
        """Reduce feature dimensions for visualization."""
        if not self.config.reduce_dims or features.size == 0:
            return None
        
        try:
            # PCA for linear dimensionality reduction
            self.pca = PCA(n_components=min(self.config.n_components, features.shape[1]))
            pca_features = self.pca.fit_transform(features)
            
            # t-SNE for non-linear dimensionality reduction
            if features.shape[0] > 5:  # t-SNE needs at least a few samples
                self.tsne = TSNE(
                    n_components=2,
                    random_state=42,
                    perplexity=min(30, features.shape[0] - 1)
                )
                tsne_features = self.tsne.fit_transform(features)
                
                return {
                    'pca': pca_features,
                    'tsne': tsne_features
                }
            else:
                return {'pca': pca_features}
                
        except Exception as e:
            st.warning(f"Error in dimensionality reduction: {e}")
            return None
    
    def _perform_clustering(self, features: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Perform clustering based on configuration."""
        if features.size == 0:
            return np.array([]), None
        
        try:
            if self.config.method == "kmeans":
                clusterer = KMeans(
                    n_clusters=self.config.n_clusters,
                    random_state=42,
                    n_init=10
                )
                labels = clusterer.fit_predict(features)
                centers = clusterer.cluster_centers_
                
            elif self.config.method == "dbscan":
                clusterer = DBSCAN(
                    eps=self.config.eps,
                    min_samples=self.config.min_samples
                )
                labels = clusterer.fit_predict(features)
                centers = None
                
            elif self.config.method == "hierarchical":
                clusterer = AgglomerativeClustering(
                    n_clusters=self.config.n_clusters
                )
                labels = clusterer.fit_predict(features)
                centers = None
                
            else:
                raise ValueError(f"Unknown clustering method: {self.config.method}")
            
            return labels, centers
            
        except Exception as e:
            st.error(f"Error in clustering: {e}")
            return np.zeros(features.shape[0]), None
    
    def _evaluate_clustering(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Evaluate clustering quality."""
        if features.size == 0 or len(np.unique(labels)) < 2:
            return {}
        
        metrics = {}
        
        try:
            # Silhouette score
            if len(np.unique(labels)) > 1 and len(np.unique(labels)) < features.shape[0]:
                metrics['silhouette_score'] = silhouette_score(features, labels)
            
            # Inertia (for k-means)
            if self.config.method == "kmeans":
                clusterer = KMeans(n_clusters=len(np.unique(labels)), random_state=42)
                clusterer.fit(features)
                metrics['inertia'] = clusterer.inertia_
            
            # Davies-Bouldin index
            from sklearn.metrics import davies_bouldin_score
            if len(np.unique(labels)) > 1:
                metrics['davies_bouldin_score'] = davies_bouldin_score(features, labels)
            
        except Exception as e:
            st.warning(f"Error computing clustering metrics: {e}")
        
        return metrics
    
    def _create_clustering_visualizations(
        self,
        trajectories: List[TrajectoryData],
        features: np.ndarray,
        reduced_features: Optional[Dict],
        labels: np.ndarray,
        centers: Optional[np.ndarray]
    ) -> Dict[str, go.Figure]:
        """Create clustering visualizations."""
        visualizations = {}
        
        # 2D cluster visualization
        if reduced_features and 'pca' in reduced_features:
            visualizations['pca_clusters'] = self._create_2d_cluster_plot(
                reduced_features['pca'], labels, "PCA Cluster Visualization"
            )
        
        if reduced_features and 'tsne' in reduced_features:
            visualizations['tsne_clusters'] = self._create_2d_cluster_plot(
                reduced_features['tsne'], labels, "t-SNE Cluster Visualization"
            )
        
        # Trajectory clusters in original space
        visualizations['trajectory_clusters'] = self._create_trajectory_cluster_plot(
            trajectories, labels
        )
        
        # Cluster size distribution
        visualizations['cluster_distribution'] = self._create_cluster_distribution_plot(labels)
        
        # Feature importance heatmap
        if features.size > 0:
            visualizations['cluster_centers'] = self._create_cluster_centers_heatmap(
                features, labels, centers
            )
        
        return visualizations
    
    def _create_2d_cluster_plot(
        self,
        features_2d: np.ndarray,
        labels: np.ndarray,
        title: str
    ) -> go.Figure:
        """Create 2D cluster scatter plot."""
        fig = go.Figure()
        
        unique_labels = np.unique(labels)
        colors = px.colors.qualitative.Set1[:len(unique_labels)]
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            cluster_points = features_2d[mask]
            
            label_name = f"Cluster {label}" if label >= 0 else "Noise"
            color = colors[i % len(colors)] if label >= 0 else "black"
            
            fig.add_trace(go.Scatter(
                x=cluster_points[:, 0],
                y=cluster_points[:, 1],
                mode='markers',
                name=label_name,
                marker=dict(
                    color=color,
                    size=8,
                    opacity=0.7
                ),
                hovertemplate=f'<b>{label_name}</b><br>' +
                             'X: %{x:.2f}<br>' +
                             'Y: %{y:.2f}<br>' +
                             '<extra></extra>'
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Component 1",
            yaxis_title="Component 2",
            template=self.plot_config.theme,
            showlegend=True,
            hovermode='closest'
        )
        
        return fig
    
    def _create_trajectory_cluster_plot(
        self,
        trajectories: List[TrajectoryData],
        labels: np.ndarray
    ) -> go.Figure:
        """Create trajectory visualization colored by cluster."""
        fig = go.Figure()
        
        unique_labels = np.unique(labels)
        colors = px.colors.qualitative.Set1[:len(unique_labels)]
        
        for i, label in enumerate(unique_labels):
            label_name = f"Cluster {label}" if label >= 0 else "Noise"
            color = colors[i % len(colors)] if label >= 0 else "black"
            
            # Find trajectories in this cluster
            cluster_trajectories = [traj for j, traj in enumerate(trajectories) if labels[j] == label]
            
            # Plot each trajectory in the cluster
            for j, trajectory in enumerate(cluster_trajectories):
                x_coords = [p.x for p in trajectory.positions]
                y_coords = [p.y for p in trajectory.positions]
                
                showlegend = j == 0  # Only show legend for first trajectory in cluster
                
                fig.add_trace(go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode='lines+markers',
                    name=label_name,
                    line=dict(color=color, width=2),
                    marker=dict(size=4),
                    opacity=0.7,
                    showlegend=showlegend,
                    hovertemplate=f'<b>{label_name}</b><br>' +
                                 'X: %{x:.2f}<br>' +
                                 'Y: %{y:.2f}<br>' +
                                 '<extra></extra>'
                ))
        
        fig.update_layout(
            title="Trajectory Clusters in Original Space",
            xaxis_title="X Position (m)",
            yaxis_title="Y Position (m)",
            template=self.plot_config.theme,
            showlegend=True,
            hovermode='closest'
        )
        
        # Equal aspect ratio
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        
        return fig
    
    def _create_cluster_distribution_plot(self, labels: np.ndarray) -> go.Figure:
        """Create cluster size distribution plot."""
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        # Create labels for display
        display_labels = [f"Cluster {label}" if label >= 0 else "Noise" for label in unique_labels]
        
        fig = go.Figure(data=[
            go.Bar(
                x=display_labels,
                y=counts,
                marker_color=px.colors.qualitative.Set1[:len(unique_labels)],
                text=counts,
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Cluster Size Distribution",
            xaxis_title="Cluster",
            yaxis_title="Number of Trajectories",
            template=self.plot_config.theme
        )
        
        return fig
    
    def _create_cluster_centers_heatmap(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        centers: Optional[np.ndarray]
    ) -> go.Figure:
        """Create heatmap of cluster centers or mean features."""
        if centers is None:
            # Compute cluster means
            unique_labels = np.unique(labels)
            centers = []
            for label in unique_labels:
                if label >= 0:  # Skip noise points
                    cluster_features = features[labels == label]
                    centers.append(np.mean(cluster_features, axis=0))
            centers = np.array(centers)
            cluster_names = [f"Cluster {i}" for i in unique_labels if i >= 0]
        else:
            cluster_names = [f"Cluster {i}" for i in range(len(centers))]
        
        # Create feature names (simplified)
        n_features = centers.shape[1] if len(centers) > 0 else 0
        feature_names = [f"Feature {i+1}" for i in range(n_features)]
        
        fig = go.Figure(data=go.Heatmap(
            z=centers,
            x=feature_names,
            y=cluster_names,
            colorscale='Viridis',
            text=np.round(centers, 2),
            texttemplate="%{text}",
            textfont={"size": 8},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Cluster Centers Heatmap",
            template=self.plot_config.theme,
            height=400
        )
        
        return fig
    
    def _analyze_clusters(
        self,
        trajectories: List[TrajectoryData],
        labels: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze cluster characteristics."""
        analysis = {}
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            if label < 0:  # Skip noise
                continue
            
            cluster_trajectories = [traj for i, traj in enumerate(trajectories) if labels[i] == label]
            
            # Basic statistics
            cluster_info = {
                'size': len(cluster_trajectories),
                'avg_length': np.mean([len(traj.positions) for traj in cluster_trajectories]),
                'avg_duration': np.mean([
                    traj.timestamps[-1] - traj.timestamps[0] 
                    for traj in cluster_trajectories if len(traj.timestamps) > 1
                ]) if any(len(traj.timestamps) > 1 for traj in cluster_trajectories) else 0,
            }
            
            # Speed analysis
            if cluster_trajectories and cluster_trajectories[0].velocities:
                speeds = []
                for traj in cluster_trajectories:
                    if traj.velocities:
                        speeds.extend([v.magnitude for v in traj.velocities])
                
                if speeds:
                    cluster_info.update({
                        'avg_speed': np.mean(speeds),
                        'max_speed': np.max(speeds),
                        'speed_std': np.std(speeds)
                    })
            
            analysis[f"cluster_{label}"] = cluster_info
        
        return analysis


class AnomalyDetector:
    """Anomaly detection and visualization for trajectories."""
    
    def __init__(
        self,
        config: Optional[AnomalyConfig] = None,
        plot_config: Optional[PlotConfig] = None
    ):
        self.config = config or AnomalyConfig()
        self.plot_config = plot_config or PlotConfig()
        self.feature_extractor = TrajectoryFeatureExtractor()
        self.scaler = StandardScaler() if self.config.normalize else None
        self.detector = None
        
    def detect_anomalies(
        self,
        trajectories: List[TrajectoryData],
        features: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Detect anomalous trajectories.
        
        Args:
            trajectories: List of trajectory data
            features: Pre-computed features (optional)
            
        Returns:
            Dictionary containing anomaly detection results and visualizations
        """
        # Extract features if not provided
        if features is None:
            with st.spinner("Extracting trajectory features..."):
                features = self._extract_features(trajectories)
        
        if features.size == 0:
            return {'anomaly_labels': [], 'scores': [], 'visualizations': {}}
        
        # Normalize features
        if self.config.normalize:
            features = self.scaler.fit_transform(features)
        
        # Detect anomalies
        with st.spinner(f"Detecting anomalies using {self.config.method}..."):
            anomaly_labels, scores = self._detect_anomalies(features)
        
        # Create visualizations
        visualizations = self._create_anomaly_visualizations(
            trajectories, features, anomaly_labels, scores
        )
        
        # Analyze anomalies
        anomaly_analysis = self._analyze_anomalies(trajectories, anomaly_labels, scores)
        
        return {
            'anomaly_labels': anomaly_labels,
            'scores': scores,
            'visualizations': visualizations,
            'anomaly_analysis': anomaly_analysis
        }
    
    def _extract_features(self, trajectories: List[TrajectoryData]) -> np.ndarray:
        """Extract features from trajectories."""
        features_list = []
        
        for trajectory in trajectories:
            try:
                features = self.feature_extractor.extract_features(trajectory)
                # Convert features dict to array
                feature_array = []
                for category, feature_dict in features.items():
                    if isinstance(feature_dict, dict):
                        feature_array.extend(list(feature_dict.values()))
                    else:
                        feature_array.append(feature_dict)
                features_list.append(feature_array)
            except Exception as e:
                st.warning(f"Error extracting features from trajectory: {e}")
                continue
        
        return np.array(features_list) if features_list else np.array([])
    
    def _detect_anomalies(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect anomalies using specified method."""
        try:
            if self.config.method == "isolation_forest":
                self.detector = IsolationForest(
                    contamination=self.config.contamination,
                    random_state=42
                )
                labels = self.detector.fit_predict(features)
                scores = -self.detector.score_samples(features)  # Invert for anomaly scores
                
            elif self.config.method == "one_class_svm":
                self.detector = OneClassSVM(
                    nu=self.config.nu,
                    gamma=self.config.gamma,
                    kernel='rbf'
                )
                labels = self.detector.fit_predict(features)
                scores = -self.detector.score_samples(features)
                
            elif self.config.method == "elliptic_envelope":
                self.detector = EllipticEnvelope(
                    contamination=self.config.contamination,
                    random_state=42
                )
                labels = self.detector.fit_predict(features)
                scores = -self.detector.score_samples(features)
                
            else:
                raise ValueError(f"Unknown anomaly detection method: {self.config.method}")
            
            # Convert labels: -1 (anomaly) -> 1, 1 (normal) -> 0
            anomaly_labels = (labels == -1).astype(int)
            
            return anomaly_labels, scores
            
        except Exception as e:
            st.error(f"Error in anomaly detection: {e}")
            return np.zeros(features.shape[0]), np.zeros(features.shape[0])
    
    def _create_anomaly_visualizations(
        self,
        trajectories: List[TrajectoryData],
        features: np.ndarray,
        anomaly_labels: np.ndarray,
        scores: np.ndarray
    ) -> Dict[str, go.Figure]:
        """Create anomaly detection visualizations."""
        visualizations = {}
        
        # Trajectory visualization with anomalies highlighted
        visualizations['trajectory_anomalies'] = self._create_trajectory_anomaly_plot(
            trajectories, anomaly_labels
        )
        
        # Anomaly scores distribution
        visualizations['score_distribution'] = self._create_score_distribution_plot(
            scores, anomaly_labels
        )
        
        # Anomaly scores timeline (if trajectories have timestamps)
        if trajectories and hasattr(trajectories[0], 'timestamps'):
            visualizations['anomaly_timeline'] = self._create_anomaly_timeline_plot(
                trajectories, anomaly_labels, scores
            )
        
        # Feature-based anomaly analysis
        if features.size > 0:
            visualizations['feature_anomalies'] = self._create_feature_anomaly_plot(
                features, anomaly_labels
            )
        
        return visualizations
    
    def _create_trajectory_anomaly_plot(
        self,
        trajectories: List[TrajectoryData],
        anomaly_labels: np.ndarray
    ) -> go.Figure:
        """Create trajectory plot highlighting anomalies."""
        fig = go.Figure()
        
        normal_color = self.plot_config.color_palette[0]
        anomaly_color = "red"
        
        # Plot normal trajectories
        normal_indices = np.where(anomaly_labels == 0)[0]
        for idx in normal_indices:
            if idx < len(trajectories):
                trajectory = trajectories[idx]
                x_coords = [p.x for p in trajectory.positions]
                y_coords = [p.y for p in trajectory.positions]
                
                fig.add_trace(go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode='lines+markers',
                    name='Normal' if idx == normal_indices[0] else None,
                    showlegend=idx == normal_indices[0],
                    line=dict(color=normal_color, width=1),
                    marker=dict(size=3),
                    opacity=0.6,
                    hovertemplate='<b>Normal Trajectory</b><br>' +
                                 'X: %{x:.2f}<br>' +
                                 'Y: %{y:.2f}<br>' +
                                 '<extra></extra>'
                ))
        
        # Plot anomalous trajectories
        anomaly_indices = np.where(anomaly_labels == 1)[0]
        for idx in anomaly_indices:
            if idx < len(trajectories):
                trajectory = trajectories[idx]
                x_coords = [p.x for p in trajectory.positions]
                y_coords = [p.y for p in trajectory.positions]
                
                fig.add_trace(go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode='lines+markers',
                    name='Anomaly' if idx == anomaly_indices[0] else None,
                    showlegend=idx == anomaly_indices[0],
                    line=dict(color=anomaly_color, width=3),
                    marker=dict(size=6),
                    opacity=0.9,
                    hovertemplate='<b>Anomalous Trajectory</b><br>' +
                                 'X: %{x:.2f}<br>' +
                                 'Y: %{y:.2f}<br>' +
                                 '<extra></extra>'
                ))
        
        fig.update_layout(
            title="Trajectory Anomaly Detection",
            xaxis_title="X Position (m)",
            yaxis_title="Y Position (m)",
            template=self.plot_config.theme,
            showlegend=True,
            hovermode='closest'
        )
        
        # Equal aspect ratio
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        
        return fig
    
    def _create_score_distribution_plot(
        self,
        scores: np.ndarray,
        anomaly_labels: np.ndarray
    ) -> go.Figure:
        """Create anomaly score distribution plot."""
        fig = go.Figure()
        
        # Normal scores
        normal_scores = scores[anomaly_labels == 0]
        if len(normal_scores) > 0:
            fig.add_trace(go.Histogram(
                x=normal_scores,
                name='Normal',
                marker_color=self.plot_config.color_palette[0],
                opacity=0.7,
                nbinsx=30
            ))
        
        # Anomaly scores
        anomaly_scores = scores[anomaly_labels == 1]
        if len(anomaly_scores) > 0:
            fig.add_trace(go.Histogram(
                x=anomaly_scores,
                name='Anomaly',
                marker_color='red',
                opacity=0.7,
                nbinsx=30
            ))
        
        fig.update_layout(
            title="Anomaly Score Distribution",
            xaxis_title="Anomaly Score",
            yaxis_title="Count",
            template=self.plot_config.theme,
            showlegend=True,
            barmode='overlay'
        )
        
        return fig
    
    def _create_anomaly_timeline_plot(
        self,
        trajectories: List[TrajectoryData],
        anomaly_labels: np.ndarray,
        scores: np.ndarray
    ) -> go.Figure:
        """Create anomaly detection timeline."""
        fig = go.Figure()
        
        # Extract timestamps (use first timestamp of each trajectory)
        timestamps = []
        for i, trajectory in enumerate(trajectories):
            if i < len(anomaly_labels) and trajectory.timestamps:
                timestamps.append(trajectory.timestamps[0])
            else:
                timestamps.append(i)  # Fallback to index
        
        # Plot anomaly scores over time
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=scores,
            mode='markers',
            marker=dict(
                color=['red' if label == 1 else self.plot_config.color_palette[0] 
                       for label in anomaly_labels],
                size=8,
                opacity=0.7
            ),
            name='Anomaly Scores',
            hovertemplate='Time: %{x}<br>' +
                         'Score: %{y:.3f}<br>' +
                         '<extra></extra>'
        ))
        
        fig.update_layout(
            title="Anomaly Detection Timeline",
            xaxis_title="Time",
            yaxis_title="Anomaly Score",
            template=self.plot_config.theme,
            showlegend=False
        )
        
        return fig
    
    def _create_feature_anomaly_plot(
        self,
        features: np.ndarray,
        anomaly_labels: np.ndarray
    ) -> go.Figure:
        """Create feature-based anomaly analysis."""
        # Use PCA for 2D visualization
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            features_2d = pca.fit_transform(features)
            
            fig = go.Figure()
            
            # Normal points
            normal_mask = anomaly_labels == 0
            if np.any(normal_mask):
                fig.add_trace(go.Scatter(
                    x=features_2d[normal_mask, 0],
                    y=features_2d[normal_mask, 1],
                    mode='markers',
                    name='Normal',
                    marker=dict(
                        color=self.plot_config.color_palette[0],
                        size=8,
                        opacity=0.7
                    ),
                    hovertemplate='<b>Normal</b><br>' +
                                 'PC1: %{x:.3f}<br>' +
                                 'PC2: %{y:.3f}<br>' +
                                 '<extra></extra>'
                ))
            
            # Anomaly points
            anomaly_mask = anomaly_labels == 1
            if np.any(anomaly_mask):
                fig.add_trace(go.Scatter(
                    x=features_2d[anomaly_mask, 0],
                    y=features_2d[anomaly_mask, 1],
                    mode='markers',
                    name='Anomaly',
                    marker=dict(
                        color='red',
                        size=10,
                        opacity=0.9,
                        symbol='x'
                    ),
                    hovertemplate='<b>Anomaly</b><br>' +
                                 'PC1: %{x:.3f}<br>' +
                                 'PC2: %{y:.3f}<br>' +
                                 '<extra></extra>'
                ))
            
            fig.update_layout(
                title="Feature-based Anomaly Detection (PCA)",
                xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)",
                yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)",
                template=self.plot_config.theme,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"Error creating feature anomaly plot: {e}")
            return go.Figure().add_annotation(text="Unable to create feature plot")
    
    def _analyze_anomalies(
        self,
        trajectories: List[TrajectoryData],
        anomaly_labels: np.ndarray,
        scores: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze detected anomalies."""
        analysis = {}
        
        # Basic statistics
        n_anomalies = np.sum(anomaly_labels)
        n_total = len(anomaly_labels)
        
        analysis['summary'] = {
            'total_trajectories': n_total,
            'anomalies_detected': int(n_anomalies),
            'anomaly_rate': float(n_anomalies / n_total) if n_total > 0 else 0,
            'avg_anomaly_score': float(np.mean(scores[anomaly_labels == 1])) if n_anomalies > 0 else 0,
            'avg_normal_score': float(np.mean(scores[anomaly_labels == 0])) if (n_total - n_anomalies) > 0 else 0
        }
        
        # Anomaly characteristics
        if n_anomalies > 0:
            anomaly_indices = np.where(anomaly_labels == 1)[0]
            anomaly_trajectories = [trajectories[i] for i in anomaly_indices if i < len(trajectories)]
            
            if anomaly_trajectories:
                analysis['anomaly_characteristics'] = {
                    'avg_length': np.mean([len(traj.positions) for traj in anomaly_trajectories]),
                    'avg_duration': np.mean([
                        traj.timestamps[-1] - traj.timestamps[0] 
                        for traj in anomaly_trajectories if len(traj.timestamps) > 1
                    ]) if any(len(traj.timestamps) > 1 for traj in anomaly_trajectories) else 0,
                }
                
                # Speed analysis for anomalies
                if anomaly_trajectories[0].velocities:
                    speeds = []
                    for traj in anomaly_trajectories:
                        if traj.velocities:
                            speeds.extend([v.magnitude for v in traj.velocities])
                    
                    if speeds:
                        analysis['anomaly_characteristics'].update({
                            'avg_speed': np.mean(speeds),
                            'max_speed': np.max(speeds),
                            'speed_std': np.std(speeds)
                        })
        
        return analysis


class FeatureImportanceVisualizer:
    """Feature importance and selection visualization."""
    
    def __init__(self, plot_config: Optional[PlotConfig] = None):
        self.plot_config = plot_config or PlotConfig()
    
    def analyze_feature_importance(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        feature_names: Optional[List[str]] = None,
        method: str = "random_forest"
    ) -> Dict[str, Any]:
        """
        Analyze feature importance for prediction tasks.
        
        Args:
            features: Feature matrix
            targets: Target values
            feature_names: Names of features
            method: Method for importance calculation
            
        Returns:
            Dictionary containing importance analysis and visualizations
        """
        if feature_names is None:
            feature_names = [f"Feature {i+1}" for i in range(features.shape[1])]
        
        # Calculate feature importance
        importance_scores = self._calculate_importance(features, targets, method)
        
        # Create visualizations
        visualizations = self._create_importance_visualizations(
            importance_scores, feature_names, method
        )
        
        # Feature correlation analysis
        correlation_analysis = self._analyze_feature_correlations(features, feature_names)
        
        return {
            'importance_scores': importance_scores,
            'feature_names': feature_names,
            'visualizations': visualizations,
            'correlation_analysis': correlation_analysis
        }
    
    def _calculate_importance(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        method: str
    ) -> np.ndarray:
        """Calculate feature importance using specified method."""
        try:
            if method == "random_forest":
                from sklearn.ensemble import RandomForestRegressor
                rf = RandomForestRegressor(n_estimators=100, random_state=42)
                rf.fit(features, targets)
                return rf.feature_importances_
            
            elif method == "mutual_info":
                from sklearn.feature_selection import mutual_info_regression
                return mutual_info_regression(features, targets, random_state=42)
            
            elif method == "correlation":
                correlations = []
                for i in range(features.shape[1]):
                    corr = np.corrcoef(features[:, i], targets)[0, 1]
                    correlations.append(abs(corr) if not np.isnan(corr) else 0)
                return np.array(correlations)
            
            else:
                raise ValueError(f"Unknown importance method: {method}")
        
        except Exception as e:
            st.warning(f"Error calculating feature importance: {e}")
            return np.zeros(features.shape[1])
    
    def _create_importance_visualizations(
        self,
        importance_scores: np.ndarray,
        feature_names: List[str],
        method: str
    ) -> Dict[str, go.Figure]:
        """Create feature importance visualizations."""
        visualizations = {}
        
        # Sort features by importance
        sorted_indices = np.argsort(importance_scores)[::-1]
        sorted_scores = importance_scores[sorted_indices]
        sorted_names = [feature_names[i] for i in sorted_indices]
        
        # Feature importance bar chart
        visualizations['importance_bars'] = self._create_importance_bar_chart(
            sorted_scores, sorted_names, method
        )
        
        # Feature importance pie chart (top 10)
        top_n = min(10, len(sorted_scores))
        visualizations['importance_pie'] = self._create_importance_pie_chart(
            sorted_scores[:top_n], sorted_names[:top_n]
        )
        
        # Cumulative importance plot
        visualizations['cumulative_importance'] = self._create_cumulative_importance_plot(
            sorted_scores, sorted_names
        )
        
        return visualizations
    
    def _create_importance_bar_chart(
        self,
        scores: np.ndarray,
        names: List[str],
        method: str
    ) -> go.Figure:
        """Create feature importance bar chart."""
        # Show top 20 features
        top_n = min(20, len(scores))
        
        fig = go.Figure(data=[
            go.Bar(
                x=names[:top_n],
                y=scores[:top_n],
                marker_color=self.plot_config.color_palette[0],
                text=[f"{score:.3f}" for score in scores[:top_n]],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title=f"Feature Importance ({method.replace('_', ' ').title()})",
            xaxis_title="Features",
            yaxis_title="Importance Score",
            template=self.plot_config.theme,
            xaxis_tickangle=-45
        )
        
        return fig
    
    def _create_importance_pie_chart(
        self,
        scores: np.ndarray,
        names: List[str]
    ) -> go.Figure:
        """Create feature importance pie chart."""
        fig = go.Figure(data=[
            go.Pie(
                labels=names,
                values=scores,
                hole=0.3,
                marker_colors=self.plot_config.color_palette[:len(names)]
            )
        ])
        
        fig.update_layout(
            title="Top Features Importance Distribution",
            template=self.plot_config.theme
        )
        
        return fig
    
    def _create_cumulative_importance_plot(
        self,
        scores: np.ndarray,
        names: List[str]
    ) -> go.Figure:
        """Create cumulative importance plot."""
        cumulative_scores = np.cumsum(scores) / np.sum(scores)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=list(range(1, len(cumulative_scores) + 1)),
            y=cumulative_scores,
            mode='lines+markers',
            name='Cumulative Importance',
            line=dict(color=self.plot_config.color_palette[0], width=3),
            marker=dict(size=6)
        ))
        
        # Add 80% and 95% lines
        fig.add_hline(y=0.8, line_dash="dash", line_color="red", 
                     annotation_text="80% Threshold")
        fig.add_hline(y=0.95, line_dash="dash", line_color="orange", 
                     annotation_text="95% Threshold")
        
        fig.update_layout(
            title="Cumulative Feature Importance",
            xaxis_title="Number of Features",
            yaxis_title="Cumulative Importance",
            template=self.plot_config.theme,
            yaxis=dict(range=[0, 1])
        )
        
        return fig
    
    def _analyze_feature_correlations(
        self,
        features: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Analyze feature correlations."""
        try:
            # Calculate correlation matrix
            corr_matrix = np.corrcoef(features.T)
            
            # Find highly correlated features
            high_corr_pairs = []
            n_features = len(feature_names)
            
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    corr_value = corr_matrix[i, j]
                    if abs(corr_value) > 0.8:  # High correlation threshold
                        high_corr_pairs.append({
                            'feature1': feature_names[i],
                            'feature2': feature_names[j],
                            'correlation': corr_value
                        })
            
            return {
                'correlation_matrix': corr_matrix,
                'high_correlations': high_corr_pairs,
                'feature_names': feature_names
            }
        
        except Exception as e:
            st.warning(f"Error analyzing feature correlations: {e}")
            return {}


class ModelInterpretabilityTools:
    """Model interpretability and explanation tools."""
    
    def __init__(self, plot_config: Optional[PlotConfig] = None):
        self.plot_config = plot_config or PlotConfig()
    
    def create_interpretability_dashboard(
        self,
        model: Any,
        features: np.ndarray,
        feature_names: List[str],
        sample_data: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Create comprehensive model interpretability dashboard.
        
        Args:
            model: Trained model with predict method
            features: Feature matrix
            feature_names: Names of features
            sample_data: Sample data for explanations
            
        Returns:
            Dictionary containing interpretability analysis and visualizations
        """
        dashboard = {}
        
        # Global feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            dashboard['global_importance'] = self._create_global_importance_plot(
                model.feature_importances_, feature_names
            )
        
        # Partial dependence plots
        dashboard['partial_dependence'] = self._create_partial_dependence_plots(
            model, features, feature_names[:5]  # Top 5 features
        )
        
        # Feature interaction analysis
        dashboard['feature_interactions'] = self._analyze_feature_interactions(
            model, features, feature_names
        )
        
        return dashboard
    
    def _create_global_importance_plot(
        self,
        importances: np.ndarray,
        feature_names: List[str]
    ) -> go.Figure:
        """Create global feature importance plot."""
        # Sort by importance
        sorted_indices = np.argsort(importances)[::-1]
        sorted_importances = importances[sorted_indices]
        sorted_names = [feature_names[i] for i in sorted_indices]
        
        # Show top 15 features
        top_n = min(15, len(sorted_importances))
        
        fig = go.Figure(data=[
            go.Bar(
                y=sorted_names[:top_n],
                x=sorted_importances[:top_n],
                orientation='h',
                marker_color=self.plot_config.color_palette[0],
                text=[f"{imp:.3f}" for imp in sorted_importances[:top_n]],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Global Feature Importance",
            xaxis_title="Importance Score",
            yaxis_title="Features",
            template=self.plot_config.theme,
            height=400
        )
        
        return fig
    
    def _create_partial_dependence_plots(
        self,
        model: Any,
        features: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, go.Figure]:
        """Create partial dependence plots for key features."""
        plots = {}
        
        for i, feature_name in enumerate(feature_names):
            try:
                # Create feature range
                feature_values = features[:, i]
                feature_range = np.linspace(
                    np.percentile(feature_values, 5),
                    np.percentile(feature_values, 95),
                    50
                )
                
                # Calculate partial dependence
                mean_prediction = np.mean([
                    self._predict_with_feature_value(model, features, i, val)
                    for val in feature_range
                ])
                
                partial_predictions = [
                    self._predict_with_feature_value(model, features, i, val)
                    for val in feature_range
                ]
                
                # Create plot
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=feature_range,
                    y=partial_predictions,
                    mode='lines',
                    name=f'Partial Dependence',
                    line=dict(color=self.plot_config.color_palette[0], width=3)
                ))
                
                fig.add_hline(
                    y=mean_prediction,
                    line_dash="dash",
                    line_color="gray",
                    annotation_text="Mean Prediction"
                )
                
                fig.update_layout(
                    title=f"Partial Dependence: {feature_name}",
                    xaxis_title=feature_name,
                    yaxis_title="Prediction",
                    template=self.plot_config.theme
                )
                
                plots[feature_name] = fig
                
            except Exception as e:
                st.warning(f"Error creating partial dependence plot for {feature_name}: {e}")
        
        return plots
    
    def _predict_with_feature_value(
        self,
        model: Any,
        features: np.ndarray,
        feature_idx: int,
        value: float
    ) -> float:
        """Predict with a specific feature value."""
        # Create modified features with the specified value
        modified_features = features.copy()
        modified_features[:, feature_idx] = value
        
        # Get predictions and return mean
        predictions = model.predict(modified_features)
        return np.mean(predictions)
    
    def _analyze_feature_interactions(
        self,
        model: Any,
        features: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Analyze feature interactions."""
        interactions = {}
        
        try:
            # Simple pairwise interaction analysis
            n_features = min(5, features.shape[1])  # Limit to top 5 features
            
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    feature1_name = feature_names[i]
                    feature2_name = feature_names[j]
                    
                    # Calculate interaction strength (simplified)
                    interaction_strength = self._calculate_interaction_strength(
                        model, features, i, j
                    )
                    
                    interactions[f"{feature1_name}_x_{feature2_name}"] = interaction_strength
        
        except Exception as e:
            st.warning(f"Error analyzing feature interactions: {e}")
        
        return interactions
    
    def _calculate_interaction_strength(
        self,
        model: Any,
        features: np.ndarray,
        idx1: int,
        idx2: int
    ) -> float:
        """Calculate interaction strength between two features."""
        try:
            # Simplified interaction calculation
            # This is a basic approximation - more sophisticated methods exist
            
            # Get feature ranges
            f1_range = np.linspace(
                np.percentile(features[:, idx1], 10),
                np.percentile(features[:, idx1], 90),
                10
            )
            f2_range = np.linspace(
                np.percentile(features[:, idx2], 10),
                np.percentile(features[:, idx2], 90),
                10
            )
            
            # Calculate predictions for different combinations
            predictions = []
            for f1_val in f1_range:
                for f2_val in f2_range:
                    modified_features = features.copy()
                    modified_features[:, idx1] = f1_val
                    modified_features[:, idx2] = f2_val
                    pred = np.mean(model.predict(modified_features))
                    predictions.append(pred)
            
            # Interaction strength as variance in predictions
            return float(np.var(predictions))
        
        except Exception as e:
            return 0.0