"""Advanced feature extraction for vehicle trajectory analysis."""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from scipy import signal
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
import logging

from ..core.config import BaseConfig
from ..core.types import TrajectoryData, TrajectoryPoint

logger = logging.getLogger(__name__)


@dataclass
class FeatureExtractionConfig(BaseConfig):
    """Configuration for feature extraction."""
    
    # Velocity features
    velocity_window_size: int = 5
    velocity_smoothing_factor: float = 0.1
    
    # Acceleration features  
    acceleration_window_size: int = 3
    jerk_window_size: int = 3
    
    # Curvature features
    curvature_window_size: int = 7
    min_curvature_threshold: float = 0.001
    
    # Lane change features
    lane_change_threshold: float = 0.5
    lane_change_min_duration: int = 10
    
    # Spatial-temporal features
    spatial_resolution: float = 1.0
    temporal_resolution: float = 0.1
    context_radius: float = 50.0
    
    # Feature selection
    enable_velocity_features: bool = True
    enable_acceleration_features: bool = True
    enable_curvature_features: bool = True
    enable_lane_change_features: bool = True
    enable_spatial_temporal_features: bool = True
    enable_contextual_features: bool = True


class BaseFeatureExtractor(ABC):
    """Base class for trajectory feature extractors."""
    
    def __init__(self, config: FeatureExtractionConfig):
        self.config = config
        self.feature_names: List[str] = []
        
    @abstractmethod
    def extract_features(self, trajectory: TrajectoryData) -> Dict[str, np.ndarray]:
        """Extract features from trajectory data."""
        pass
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return self.feature_names.copy()


class VelocityFeatureExtractor(BaseFeatureExtractor):
    """Extract velocity-related features from trajectories."""
    
    def __init__(self, config: FeatureExtractionConfig):
        super().__init__(config)
        self.feature_names = [
            'velocity_magnitude', 'velocity_x', 'velocity_y',
            'velocity_smoothness', 'velocity_variability',
            'max_velocity', 'min_velocity', 'mean_velocity',
            'velocity_acceleration_correlation'
        ]
    
    def extract_features(self, trajectory: TrajectoryData) -> Dict[str, np.ndarray]:
        """Extract velocity features from trajectory."""
        features = {}
        
        # Calculate velocity components
        dt = np.diff(trajectory.timestamps)
        dx = np.diff(trajectory.x_positions)
        dy = np.diff(trajectory.y_positions)
        
        # Avoid division by zero
        dt = np.where(dt == 0, 1e-6, dt)
        
        velocity_x = dx / dt
        velocity_y = dy / dt
        velocity_magnitude = np.sqrt(velocity_x**2 + velocity_y**2)
        
        # Smooth velocity using moving average
        window_size = self.config.velocity_window_size
        if len(velocity_magnitude) >= window_size:
            velocity_smooth = signal.convolve(
                velocity_magnitude, 
                np.ones(window_size) / window_size, 
                mode='valid'
            )
        else:
            velocity_smooth = velocity_magnitude
        
        # Velocity statistics
        features['velocity_magnitude'] = velocity_magnitude
        features['velocity_x'] = velocity_x
        features['velocity_y'] = velocity_y
        features['velocity_smoothness'] = velocity_smooth
        features['velocity_variability'] = np.std(velocity_magnitude)
        features['max_velocity'] = np.max(velocity_magnitude)
        features['min_velocity'] = np.min(velocity_magnitude)
        features['mean_velocity'] = np.mean(velocity_magnitude)
        
        # Velocity-acceleration correlation
        if len(velocity_magnitude) > 1:
            acceleration = np.diff(velocity_magnitude) / dt[:-1]
            correlation = np.corrcoef(velocity_magnitude[:-1], acceleration)[0, 1]
            features['velocity_acceleration_correlation'] = correlation if not np.isnan(correlation) else 0.0
        else:
            features['velocity_acceleration_correlation'] = 0.0
            
        return features


class AccelerationFeatureExtractor(BaseFeatureExtractor):
    """Extract acceleration-related features from trajectories."""
    
    def __init__(self, config: FeatureExtractionConfig):
        super().__init__(config)
        self.feature_names = [
            'acceleration_magnitude', 'acceleration_x', 'acceleration_y',
            'jerk_magnitude', 'jerk_x', 'jerk_y',
            'acceleration_smoothness', 'jerk_smoothness',
            'max_acceleration', 'min_acceleration', 'mean_acceleration',
            'max_jerk', 'min_jerk', 'mean_jerk',
            'acceleration_pattern', 'jerk_pattern'
        ]
    
    def extract_features(self, trajectory: TrajectoryData) -> Dict[str, np.ndarray]:
        """Extract acceleration features from trajectory."""
        features = {}
        
        # Calculate velocity first
        dt = np.diff(trajectory.timestamps)
        dx = np.diff(trajectory.x_positions)
        dy = np.diff(trajectory.y_positions)
        dt = np.where(dt == 0, 1e-6, dt)
        
        velocity_x = dx / dt
        velocity_y = dy / dt
        
        # Calculate acceleration
        if len(velocity_x) > 1:
            dt_acc = dt[:-1]
            dt_acc = np.where(dt_acc == 0, 1e-6, dt_acc)
            
            acceleration_x = np.diff(velocity_x) / dt_acc
            acceleration_y = np.diff(velocity_y) / dt_acc
            acceleration_magnitude = np.sqrt(acceleration_x**2 + acceleration_y**2)
            
            # Calculate jerk
            if len(acceleration_x) > 1:
                dt_jerk = dt_acc[:-1]
                dt_jerk = np.where(dt_jerk == 0, 1e-6, dt_jerk)
                
                jerk_x = np.diff(acceleration_x) / dt_jerk
                jerk_y = np.diff(acceleration_y) / dt_jerk
                jerk_magnitude = np.sqrt(jerk_x**2 + jerk_y**2)
            else:
                jerk_x = np.array([])
                jerk_y = np.array([])
                jerk_magnitude = np.array([])
        else:
            acceleration_x = np.array([])
            acceleration_y = np.array([])
            acceleration_magnitude = np.array([])
            jerk_x = np.array([])
            jerk_y = np.array([])
            jerk_magnitude = np.array([])
        
        # Smooth acceleration and jerk
        window_size_acc = self.config.acceleration_window_size
        window_size_jerk = self.config.jerk_window_size
        
        if len(acceleration_magnitude) >= window_size_acc:
            acceleration_smooth = signal.convolve(
                acceleration_magnitude,
                np.ones(window_size_acc) / window_size_acc,
                mode='valid'
            )
        else:
            acceleration_smooth = acceleration_magnitude
            
        if len(jerk_magnitude) >= window_size_jerk:
            jerk_smooth = signal.convolve(
                jerk_magnitude,
                np.ones(window_size_jerk) / window_size_jerk,
                mode='valid'
            )
        else:
            jerk_smooth = jerk_magnitude
        
        # Store features
        features['acceleration_magnitude'] = acceleration_magnitude
        features['acceleration_x'] = acceleration_x
        features['acceleration_y'] = acceleration_y
        features['jerk_magnitude'] = jerk_magnitude
        features['jerk_x'] = jerk_x
        features['jerk_y'] = jerk_y
        features['acceleration_smoothness'] = acceleration_smooth
        features['jerk_smoothness'] = jerk_smooth
        
        # Statistics
        features['max_acceleration'] = np.max(acceleration_magnitude) if len(acceleration_magnitude) > 0 else 0.0
        features['min_acceleration'] = np.min(acceleration_magnitude) if len(acceleration_magnitude) > 0 else 0.0
        features['mean_acceleration'] = np.mean(acceleration_magnitude) if len(acceleration_magnitude) > 0 else 0.0
        features['max_jerk'] = np.max(jerk_magnitude) if len(jerk_magnitude) > 0 else 0.0
        features['min_jerk'] = np.min(jerk_magnitude) if len(jerk_magnitude) > 0 else 0.0
        features['mean_jerk'] = np.mean(jerk_magnitude) if len(jerk_magnitude) > 0 else 0.0
        
        # Pattern analysis
        features['acceleration_pattern'] = self._extract_pattern(acceleration_magnitude)
        features['jerk_pattern'] = self._extract_pattern(jerk_magnitude)
        
        return features
    
    def _extract_pattern(self, data: np.ndarray) -> str:
        """Extract pattern from time series data."""
        if len(data) < 3:
            return 'insufficient_data'
        
        # Simple pattern detection
        diff = np.diff(data)
        positive_count = np.sum(diff > 0)
        negative_count = np.sum(diff < 0)
        total_count = len(diff)
        
        if positive_count / total_count > 0.7:
            return 'increasing'
        elif negative_count / total_count > 0.7:
            return 'decreasing'
        else:
            return 'oscillating'


class CurvatureFeatureExtractor(BaseFeatureExtractor):
    """Extract curvature-related features from trajectories."""
    
    def __init__(self, config: FeatureExtractionConfig):
        super().__init__(config)
        self.feature_names = [
            'curvature', 'curvature_smoothness', 'curvature_variability',
            'max_curvature', 'min_curvature', 'mean_curvature',
            'curvature_rate', 'curvature_acceleration',
            'turning_radius', 'turning_angle',
            'straight_line_ratio', 'curve_complexity'
        ]
    
    def extract_features(self, trajectory: TrajectoryData) -> Dict[str, np.ndarray]:
        """Extract curvature features from trajectory."""
        features = {}
        
        # Calculate curvature using three-point method
        x = trajectory.x_positions
        y = trajectory.y_positions
        
        if len(x) < 3:
            # Not enough points for curvature calculation
            features.update({name: np.array([]) for name in self.feature_names})
            return features
        
        # Calculate curvature for each point using three-point method
        curvature = np.zeros(len(x))
        turning_radius = np.zeros(len(x))
        
        for i in range(1, len(x) - 1):
            # Three consecutive points
            p1 = np.array([x[i-1], y[i-1]])
            p2 = np.array([x[i], y[i]])
            p3 = np.array([x[i+1], y[i+1]])
            
            # Calculate curvature using circumscribed circle
            a = np.linalg.norm(p2 - p3)
            b = np.linalg.norm(p1 - p3)
            c = np.linalg.norm(p1 - p2)
            
            if a * b * c == 0:
                curvature[i] = 0
                turning_radius[i] = np.inf
            else:
                # Semi-perimeter
                s = (a + b + c) / 2
                # Area using Heron's formula
                area = np.sqrt(s * (s - a) * (s - b) * (s - c))
                
                if area > 0:
                    # Radius of circumscribed circle
                    radius = (a * b * c) / (4 * area)
                    curvature[i] = 1 / radius if radius > 0 else 0
                    turning_radius[i] = radius
                else:
                    curvature[i] = 0
                    turning_radius[i] = np.inf
        
        # Smooth curvature
        window_size = self.config.curvature_window_size
        if len(curvature) >= window_size:
            curvature_smooth = signal.convolve(
                curvature,
                np.ones(window_size) / window_size,
                mode='same'
            )
        else:
            curvature_smooth = curvature
        
        # Calculate turning angle
        turning_angle = np.zeros(len(x))
        for i in range(1, len(x)):
            dx = x[i] - x[i-1]
            dy = y[i] - y[i-1]
            angle = np.arctan2(dy, dx)
            if i > 1:
                angle_diff = angle - turning_angle[i-1]
                # Normalize angle difference to [-π, π]
                angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
                turning_angle[i] = turning_angle[i-1] + angle_diff
            else:
                turning_angle[i] = angle
        
        # Calculate curvature rate
        dt = np.diff(trajectory.timestamps)
        dt = np.where(dt == 0, 1e-6, dt)
        curvature_rate = np.diff(curvature) / dt
        
        # Calculate straight line ratio
        total_distance = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
        direct_distance = np.sqrt((x[-1] - x[0])**2 + (y[-1] - y[0])**2)
        straight_line_ratio = direct_distance / total_distance if total_distance > 0 else 1.0
        
        # Calculate curve complexity (number of significant curvature changes)
        curvature_threshold = self.config.min_curvature_threshold
        significant_curves = np.sum(np.abs(curvature) > curvature_threshold)
        curve_complexity = significant_curves / len(curvature) if len(curvature) > 0 else 0
        
        # Store features
        features['curvature'] = curvature
        features['curvature_smoothness'] = curvature_smooth
        features['curvature_variability'] = np.std(curvature)
        features['max_curvature'] = np.max(np.abs(curvature))
        features['min_curvature'] = np.min(curvature)
        features['mean_curvature'] = np.mean(np.abs(curvature))
        features['curvature_rate'] = curvature_rate
        features['curvature_acceleration'] = np.diff(curvature_rate) if len(curvature_rate) > 1 else np.array([])
        features['turning_radius'] = turning_radius
        features['turning_angle'] = turning_angle
        features['straight_line_ratio'] = straight_line_ratio
        features['curve_complexity'] = curve_complexity
        
        return features


class LaneChangeFeatureExtractor(BaseFeatureExtractor):
    """Extract lane change detection features from trajectories."""
    
    def __init__(self, config: FeatureExtractionConfig):
        super().__init__(config)
        self.feature_names = [
            'lane_change_detected', 'lane_change_count',
            'lane_change_duration', 'lane_change_distance',
            'lane_change_velocity', 'lane_change_acceleration',
            'lane_change_angle', 'lane_change_smoothness',
            'lane_position', 'lane_deviation',
            'lane_center_distance', 'lane_boundary_distance'
        ]
    
    def extract_features(self, trajectory: TrajectoryData) -> Dict[str, np.ndarray]:
        """Extract lane change features from trajectory."""
        features = {}
        
        # For NGSIM data, lane information might be available
        # For now, we'll use lateral movement detection
        x = trajectory.x_positions
        y = trajectory.y_positions
        
        if len(x) < 3:
            features.update({name: np.array([]) for name in self.feature_names})
            return features
        
        # Calculate lateral movement (assuming y-axis is lateral)
        lateral_movement = y
        
        # Detect lane changes using lateral movement analysis
        lane_changes = self._detect_lane_changes(lateral_movement, trajectory.timestamps)
        
        # Lane change features
        features['lane_change_detected'] = len(lane_changes) > 0
        features['lane_change_count'] = len(lane_changes)
        
        if len(lane_changes) > 0:
            # Calculate lane change statistics
            durations = []
            distances = []
            velocities = []
            accelerations = []
            angles = []
            
            for start_idx, end_idx in lane_changes:
                # Duration
                duration = trajectory.timestamps[end_idx] - trajectory.timestamps[start_idx]
                durations.append(duration)
                
                # Distance
                distance = np.sqrt((x[end_idx] - x[start_idx])**2 + (y[end_idx] - y[start_idx])**2)
                distances.append(distance)
                
                # Velocity
                velocity = distance / duration if duration > 0 else 0
                velocities.append(velocity)
                
                # Acceleration (simplified)
                if end_idx - start_idx > 1:
                    mid_idx = (start_idx + end_idx) // 2
                    accel = (velocities[-1] - velocity) / duration if duration > 0 else 0
                    accelerations.append(accel)
                else:
                    accelerations.append(0)
                
                # Angle
                angle = np.arctan2(y[end_idx] - y[start_idx], x[end_idx] - x[start_idx])
                angles.append(angle)
            
            features['lane_change_duration'] = np.array(durations)
            features['lane_change_distance'] = np.array(distances)
            features['lane_change_velocity'] = np.array(velocities)
            features['lane_change_acceleration'] = np.array(accelerations)
            features['lane_change_angle'] = np.array(angles)
            features['lane_change_smoothness'] = self._calculate_lane_change_smoothness(lane_changes, lateral_movement)
        else:
            features['lane_change_duration'] = np.array([])
            features['lane_change_distance'] = np.array([])
            features['lane_change_velocity'] = np.array([])
            features['lane_change_acceleration'] = np.array([])
            features['lane_change_angle'] = np.array([])
            features['lane_change_smoothness'] = 0.0
        
        # Lane position features (simplified)
        features['lane_position'] = lateral_movement
        features['lane_deviation'] = np.std(lateral_movement)
        features['lane_center_distance'] = np.abs(lateral_movement - np.mean(lateral_movement))
        features['lane_boundary_distance'] = self._calculate_boundary_distance(lateral_movement)
        
        return features
    
    def _detect_lane_changes(self, lateral_movement: np.ndarray, timestamps: np.ndarray) -> List[Tuple[int, int]]:
        """Detect lane changes in lateral movement."""
        lane_changes = []
        
        # Use rolling window to detect significant lateral movements
        window_size = 5
        threshold = self.config.lane_change_threshold
        min_duration = self.config.lane_change_min_duration
        
        if len(lateral_movement) < window_size:
            return lane_changes
        
        # Calculate rolling statistics
        rolling_mean = pd.Series(lateral_movement).rolling(window=window_size, center=True).mean().fillna(method='bfill').fillna(method='ffill')
        rolling_std = pd.Series(lateral_movement).rolling(window=window_size, center=True).std().fillna(method='bfill').fillna(method='ffill')
        
        # Detect significant deviations
        deviations = np.abs(lateral_movement - rolling_mean)
        significant_deviations = deviations > (threshold * rolling_std)
        
        # Find continuous segments
        in_lane_change = False
        start_idx = 0
        
        for i, is_significant in enumerate(significant_deviations):
            if is_significant and not in_lane_change:
                start_idx = i
                in_lane_change = True
            elif not is_significant and in_lane_change:
                end_idx = i - 1
                duration = end_idx - start_idx + 1
                
                if duration >= min_duration:
                    lane_changes.append((start_idx, end_idx))
                
                in_lane_change = False
        
        # Handle case where lane change continues to end
        if in_lane_change:
            end_idx = len(lateral_movement) - 1
            duration = end_idx - start_idx + 1
            if duration >= min_duration:
                lane_changes.append((start_idx, end_idx))
        
        return lane_changes
    
    def _calculate_lane_change_smoothness(self, lane_changes: List[Tuple[int, int]], lateral_movement: np.ndarray) -> float:
        """Calculate smoothness of lane changes."""
        if not lane_changes:
            return 0.0
        
        smoothness_scores = []
        for start_idx, end_idx in lane_changes:
            segment = lateral_movement[start_idx:end_idx+1]
            if len(segment) > 2:
                # Calculate smoothness as inverse of second derivative
                second_derivative = np.diff(segment, n=2)
                smoothness = 1 / (1 + np.mean(np.abs(second_derivative)))
                smoothness_scores.append(smoothness)
        
        return np.mean(smoothness_scores) if smoothness_scores else 0.0
    
    def _calculate_boundary_distance(self, lateral_movement: np.ndarray) -> np.ndarray:
        """Calculate distance to lane boundaries."""
        # Simplified boundary calculation
        mean_lane = np.mean(lateral_movement)
        std_lane = np.std(lateral_movement)
        
        # Assume lane width is 2 * std
        lane_width = 2 * std_lane
        boundary_distance = np.abs(lateral_movement - mean_lane) - lane_width / 2
        
        return boundary_distance


class SpatialTemporalFeatureExtractor(BaseFeatureExtractor):
    """Extract spatial-temporal features from trajectories."""
    
    def __init__(self, config: FeatureExtractionConfig):
        super().__init__(config)
        self.feature_names = [
            'spatial_density', 'temporal_density',
            'spatial_entropy', 'temporal_entropy',
            'spatial_clustering', 'temporal_clustering',
            'spatial_regularity', 'temporal_regularity',
            'spatial_coverage', 'temporal_coverage',
            'spatial_dispersion', 'temporal_dispersion'
        ]
    
    def extract_features(self, trajectory: TrajectoryData) -> Dict[str, np.ndarray]:
        """Extract spatial-temporal features from trajectory."""
        features = {}
        
        x = trajectory.x_positions
        y = trajectory.y_positions
        timestamps = trajectory.timestamps
        
        if len(x) < 2:
            features.update({name: np.array([]) for name in self.feature_names})
            return features
        
        # Spatial features
        spatial_resolution = self.config.spatial_resolution
        
        # Spatial density (points per unit area)
        spatial_bounds = self._calculate_spatial_bounds(x, y)
        spatial_area = (spatial_bounds[1] - spatial_bounds[0]) * (spatial_bounds[3] - spatial_bounds[2])
        spatial_density = len(x) / spatial_area if spatial_area > 0 else 0
        
        # Spatial entropy (distribution uniformity)
        spatial_entropy = self._calculate_spatial_entropy(x, y, spatial_resolution)
        
        # Spatial clustering
        spatial_clustering = self._calculate_spatial_clustering(x, y)
        
        # Spatial regularity
        spatial_regularity = self._calculate_spatial_regularity(x, y)
        
        # Spatial coverage
        spatial_coverage = self._calculate_spatial_coverage(x, y, spatial_resolution)
        
        # Spatial dispersion
        spatial_dispersion = self._calculate_spatial_dispersion(x, y)
        
        # Temporal features
        temporal_resolution = self.config.temporal_resolution
        
        # Temporal density (points per unit time)
        temporal_duration = timestamps[-1] - timestamps[0]
        temporal_density = len(timestamps) / temporal_duration if temporal_duration > 0 else 0
        
        # Temporal entropy
        temporal_entropy = self._calculate_temporal_entropy(timestamps, temporal_resolution)
        
        # Temporal clustering
        temporal_clustering = self._calculate_temporal_clustering(timestamps)
        
        # Temporal regularity
        temporal_regularity = self._calculate_temporal_regularity(timestamps)
        
        # Temporal coverage
        temporal_coverage = self._calculate_temporal_coverage(timestamps, temporal_resolution)
        
        # Temporal dispersion
        temporal_dispersion = self._calculate_temporal_dispersion(timestamps)
        
        # Store features
        features['spatial_density'] = spatial_density
        features['temporal_density'] = temporal_density
        features['spatial_entropy'] = spatial_entropy
        features['temporal_entropy'] = temporal_entropy
        features['spatial_clustering'] = spatial_clustering
        features['temporal_clustering'] = temporal_clustering
        features['spatial_regularity'] = spatial_regularity
        features['temporal_regularity'] = temporal_regularity
        features['spatial_coverage'] = spatial_coverage
        features['temporal_coverage'] = temporal_coverage
        features['spatial_dispersion'] = spatial_dispersion
        features['temporal_dispersion'] = temporal_dispersion
        
        return features
    
    def _calculate_spatial_bounds(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
        """Calculate spatial bounds of trajectory."""
        return (np.min(x), np.max(x), np.min(y), np.max(y))
    
    def _calculate_spatial_entropy(self, x: np.ndarray, y: np.ndarray, resolution: float) -> float:
        """Calculate spatial entropy of trajectory."""
        # Discretize space into grid
        x_bins = int((np.max(x) - np.min(x)) / resolution) + 1
        y_bins = int((np.max(y) - np.min(y)) / resolution) + 1
        
        if x_bins <= 1 or y_bins <= 1:
            return 0.0
        
        # Create 2D histogram
        hist, _, _ = np.histogram2d(x, y, bins=[x_bins, y_bins])
        
        # Calculate entropy
        hist = hist.flatten()
        hist = hist[hist > 0]  # Remove empty bins
        if len(hist) == 0:
            return 0.0
        
        prob = hist / np.sum(hist)
        entropy = -np.sum(prob * np.log2(prob))
        
        return entropy
    
    def _calculate_spatial_clustering(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate spatial clustering coefficient."""
        if len(x) < 3:
            return 0.0
        
        # Calculate pairwise distances
        points = np.column_stack([x, y])
        distances = cdist(points, points)
        
        # Calculate clustering coefficient
        clustering_scores = []
        for i in range(len(points)):
            # Find neighbors within threshold
            threshold = np.percentile(distances[i], 25)  # 25th percentile as threshold
            neighbors = np.where(distances[i] <= threshold)[0]
            neighbors = neighbors[neighbors != i]
            
            if len(neighbors) >= 2:
                # Calculate local clustering coefficient
                neighbor_distances = distances[np.ix_(neighbors, neighbors)]
                max_possible_edges = len(neighbors) * (len(neighbors) - 1) / 2
                actual_edges = np.sum(neighbor_distances <= threshold) / 2
                clustering = actual_edges / max_possible_edges if max_possible_edges > 0 else 0
                clustering_scores.append(clustering)
        
        return np.mean(clustering_scores) if clustering_scores else 0.0
    
    def _calculate_spatial_regularity(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate spatial regularity of trajectory."""
        if len(x) < 3:
            return 0.0
        
        # Calculate step lengths
        step_lengths = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
        
        # Regularity is inverse of coefficient of variation
        mean_step = np.mean(step_lengths)
        std_step = np.std(step_lengths)
        
        regularity = mean_step / (mean_step + std_step) if (mean_step + std_step) > 0 else 0.0
        
        return regularity
    
    def _calculate_spatial_coverage(self, x: np.ndarray, y: np.ndarray, resolution: float) -> float:
        """Calculate spatial coverage of trajectory."""
        # Calculate convex hull area
        from scipy.spatial import ConvexHull
        
        if len(x) < 3:
            return 0.0
        
        points = np.column_stack([x, y])
        try:
            hull = ConvexHull(points)
            hull_area = hull.volume  # 2D volume is area
        except:
            hull_area = 0.0
        
        # Calculate bounding box area
        bounds = self._calculate_spatial_bounds(x, y)
        bbox_area = (bounds[1] - bounds[0]) * (bounds[3] - bounds[2])
        
        # Coverage is hull area / bbox area
        coverage = hull_area / bbox_area if bbox_area > 0 else 0.0
        
        return coverage
    
    def _calculate_spatial_dispersion(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate spatial dispersion of trajectory."""
        # Calculate centroid
        centroid_x = np.mean(x)
        centroid_y = np.mean(y)
        
        # Calculate distances to centroid
        distances = np.sqrt((x - centroid_x)**2 + (y - centroid_y)**2)
        
        # Dispersion is standard deviation of distances
        dispersion = np.std(distances)
        
        return dispersion
    
    def _calculate_temporal_entropy(self, timestamps: np.ndarray, resolution: float) -> float:
        """Calculate temporal entropy of trajectory."""
        # Discretize time into bins
        time_bins = int((timestamps[-1] - timestamps[0]) / resolution) + 1
        
        if time_bins <= 1:
            return 0.0
        
        # Create histogram
        hist, _ = np.histogram(timestamps, bins=time_bins)
        
        # Calculate entropy
        hist = hist[hist > 0]  # Remove empty bins
        if len(hist) == 0:
            return 0.0
        
        prob = hist / np.sum(hist)
        entropy = -np.sum(prob * np.log2(prob))
        
        return entropy
    
    def _calculate_temporal_clustering(self, timestamps: np.ndarray) -> float:
        """Calculate temporal clustering coefficient."""
        if len(timestamps) < 3:
            return 0.0
        
        # Calculate time intervals
        intervals = np.diff(timestamps)
        
        # Calculate clustering based on interval similarity
        threshold = np.percentile(intervals, 25)
        similar_intervals = np.sum(intervals <= threshold)
        clustering = similar_intervals / len(intervals) if len(intervals) > 0 else 0.0
        
        return clustering
    
    def _calculate_temporal_regularity(self, timestamps: np.ndarray) -> float:
        """Calculate temporal regularity of trajectory."""
        if len(timestamps) < 3:
            return 0.0
        
        # Calculate time intervals
        intervals = np.diff(timestamps)
        
        # Regularity is inverse of coefficient of variation
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        
        regularity = mean_interval / (mean_interval + std_interval) if (mean_interval + std_interval) > 0 else 0.0
        
        return regularity
    
    def _calculate_temporal_coverage(self, timestamps: np.ndarray, resolution: float) -> float:
        """Calculate temporal coverage of trajectory."""
        total_duration = timestamps[-1] - timestamps[0]
        if total_duration <= 0:
            return 0.0
        
        # Calculate number of time bins covered
        time_bins = int(total_duration / resolution) + 1
        covered_bins = len(np.unique(np.floor((timestamps - timestamps[0]) / resolution)))
        
        coverage = covered_bins / time_bins if time_bins > 0 else 0.0
        
        return coverage
    
    def _calculate_temporal_dispersion(self, timestamps: np.ndarray) -> float:
        """Calculate temporal dispersion of trajectory."""
        # Calculate mean time
        mean_time = np.mean(timestamps)
        
        # Calculate dispersion as standard deviation
        dispersion = np.std(timestamps)
        
        return dispersion


class ContextualFeatureExtractor(BaseFeatureExtractor):
    """Extract contextual features from trajectories."""
    
    def __init__(self, config: FeatureExtractionConfig):
        super().__init__(config)
        self.feature_names = [
            'road_geometry_features', 'traffic_density_features',
            'weather_features', 'time_of_day_features',
            'day_of_week_features', 'seasonal_features',
            'intersection_features', 'highway_features',
            'urban_features', 'rural_features'
        ]
    
    def extract_features(self, trajectory: TrajectoryData) -> Dict[str, np.ndarray]:
        """Extract contextual features from trajectory."""
        features = {}
        
        # For now, we'll implement simplified contextual features
        # In a real implementation, these would use external data sources
        
        timestamps = trajectory.timestamps
        x = trajectory.x_positions
        y = trajectory.y_positions
        
        # Time-based contextual features
        if len(timestamps) > 0:
            # Convert timestamps to datetime if they're not already
            if isinstance(timestamps[0], (int, float)):
                # Assume timestamps are Unix timestamps
                from datetime import datetime
                dt_times = [datetime.fromtimestamp(ts) for ts in timestamps]
            else:
                dt_times = timestamps
            
            # Time of day features
            hours = [dt.hour for dt in dt_times]
            features['time_of_day_features'] = {
                'hour_of_day': np.array(hours),
                'is_rush_hour': np.array([6 <= h <= 9 or 16 <= h <= 19 for h in hours]),
                'is_night': np.array([h < 6 or h > 22 for h in hours]),
                'is_day': np.array([6 <= h <= 18 for h in hours])
            }
            
            # Day of week features
            weekdays = [dt.weekday() for dt in dt_times]
            features['day_of_week_features'] = {
                'weekday': np.array(weekdays),
                'is_weekend': np.array([w >= 5 for w in weekdays]),
                'is_weekday': np.array([w < 5 for w in weekdays])
            }
            
            # Seasonal features
            months = [dt.month for dt in dt_times]
            features['seasonal_features'] = {
                'month': np.array(months),
                'is_winter': np.array([m in [12, 1, 2] for m in months]),
                'is_spring': np.array([m in [3, 4, 5] for m in months]),
                'is_summer': np.array([m in [6, 7, 8] for m in months]),
                'is_fall': np.array([m in [9, 10, 11] for m in months])
            }
        else:
            features['time_of_day_features'] = {}
            features['day_of_week_features'] = {}
            features['seasonal_features'] = {}
        
        # Spatial contextual features (simplified)
        if len(x) > 0 and len(y) > 0:
            # Road geometry features (simplified)
            features['road_geometry_features'] = {
                'trajectory_length': np.sqrt((x[-1] - x[0])**2 + (y[-1] - y[0])**2),
                'trajectory_area': self._calculate_trajectory_area(x, y),
                'curvature_variation': np.std(np.diff(np.arctan2(np.diff(y), np.diff(x)))) if len(x) > 2 else 0.0
            }
            
            # Urban/rural classification (simplified)
            trajectory_length = np.sqrt((x[-1] - x[0])**2 + (y[-1] - y[0])**2)
            is_urban = trajectory_length < 1000  # Simplified heuristic
            
            features['urban_features'] = {
                'is_urban': is_urban,
                'urban_density': 1.0 if is_urban else 0.0
            }
            
            features['rural_features'] = {
                'is_rural': not is_urban,
                'rural_density': 0.0 if is_urban else 1.0
            }
            
            # Highway features (simplified)
            avg_velocity = np.mean(np.sqrt(np.diff(x)**2 + np.diff(y)**2) / np.diff(timestamps)) if len(timestamps) > 1 else 0.0
            is_highway = avg_velocity > 20.0  # Simplified heuristic
            
            features['highway_features'] = {
                'is_highway': is_highway,
                'highway_speed': avg_velocity if is_highway else 0.0
            }
            
            # Intersection features (simplified)
            features['intersection_features'] = {
                'intersection_count': self._estimate_intersection_count(x, y),
                'intersection_density': self._estimate_intersection_count(x, y) / max(trajectory_length, 1.0)
            }
        else:
            features['road_geometry_features'] = {}
            features['urban_features'] = {}
            features['rural_features'] = {}
            features['highway_features'] = {}
            features['intersection_features'] = {}
        
        # Placeholder features for external data sources
        features['traffic_density_features'] = {}
        features['weather_features'] = {}
        
        return features
    
    def _calculate_trajectory_area(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate area covered by trajectory."""
        if len(x) < 3:
            return 0.0
        
        # Calculate convex hull area
        from scipy.spatial import ConvexHull
        
        points = np.column_stack([x, y])
        try:
            hull = ConvexHull(points)
            return hull.volume  # 2D volume is area
        except:
            return 0.0
    
    def _estimate_intersection_count(self, x: np.ndarray, y: np.ndarray) -> int:
        """Estimate number of intersections in trajectory."""
        if len(x) < 3:
            return 0
        
        # Simplified intersection detection based on direction changes
        directions = np.arctan2(np.diff(y), np.diff(x))
        direction_changes = np.abs(np.diff(directions))
        
        # Count significant direction changes
        significant_changes = np.sum(direction_changes > np.pi/4)  # 45 degrees
        
        return significant_changes


class TrajectoryFeatureExtractor:
    """Main feature extractor that combines all feature extractors."""
    
    def __init__(self, config: FeatureExtractionConfig):
        self.config = config
        self.extractors = {}
        
        # Initialize feature extractors based on configuration
        if config.enable_velocity_features:
            self.extractors['velocity'] = VelocityFeatureExtractor(config)
        
        if config.enable_acceleration_features:
            self.extractors['acceleration'] = AccelerationFeatureExtractor(config)
        
        if config.enable_curvature_features:
            self.extractors['curvature'] = CurvatureFeatureExtractor(config)
        
        if config.enable_lane_change_features:
            self.extractors['lane_change'] = LaneChangeFeatureExtractor(config)
        
        if config.enable_spatial_temporal_features:
            self.extractors['spatial_temporal'] = SpatialTemporalFeatureExtractor(config)
        
        if config.enable_contextual_features:
            self.extractors['contextual'] = ContextualFeatureExtractor(config)
    
    def extract_features(self, trajectory: TrajectoryData) -> Dict[str, Any]:
        """Extract all features from trajectory."""
        all_features = {}
        
        for extractor_name, extractor in self.extractors.items():
            try:
                features = extractor.extract_features(trajectory)
                all_features[extractor_name] = features
                logger.debug(f"Extracted {len(features)} features from {extractor_name} extractor")
            except Exception as e:
                logger.error(f"Error extracting features from {extractor_name}: {e}")
                all_features[extractor_name] = {}
        
        return all_features
    
    def get_all_feature_names(self) -> Dict[str, List[str]]:
        """Get all feature names organized by extractor."""
        feature_names = {}
        for extractor_name, extractor in self.extractors.items():
            feature_names[extractor_name] = extractor.get_feature_names()
        return feature_names
    
    def get_total_feature_count(self) -> int:
        """Get total number of features."""
        total_count = 0
        for extractor in self.extractors.values():
            total_count += len(extractor.get_feature_names())
        return total_count