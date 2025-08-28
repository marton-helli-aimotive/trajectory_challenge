"""Trajectory quality metrics and validation for vehicle trajectory prediction."""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from scipy import signal
from scipy.stats import zscore
import logging

from ..core.config import BaseConfig
from ..core.types import TrajectoryData, TrajectoryPoint

logger = logging.getLogger(__name__)


@dataclass
class QualityMetricsConfig(BaseConfig):
    """Configuration for trajectory quality metrics."""
    
    # Completeness thresholds
    min_trajectory_length: int = 10
    max_missing_ratio: float = 0.2
    min_time_coverage: float = 0.8
    
    # Smoothness thresholds
    max_velocity_jump: float = 10.0
    max_acceleration_jump: float = 5.0
    max_position_jump: float = 5.0
    smoothness_window_size: int = 5
    
    # Consistency thresholds
    max_velocity: float = 50.0
    max_acceleration: float = 10.0
    max_jerk: float = 5.0
    min_velocity: float = 0.0
    
    # Physics constraints
    max_centripetal_acceleration: float = 15.0
    min_turning_radius: float = 5.0
    max_angular_velocity: float = 2.0
    
    # Quality scoring weights
    completeness_weight: float = 0.3
    smoothness_weight: float = 0.3
    consistency_weight: float = 0.2
    physics_weight: float = 0.2
    
    # Outlier detection
    outlier_z_threshold: float = 3.0
    outlier_iqr_multiplier: float = 1.5
    
    # Report generation
    generate_detailed_reports: bool = True
    save_quality_plots: bool = False
    quality_plot_path: str = "quality_plots"


class BaseQualityMetric(ABC):
    """Base class for trajectory quality metrics."""
    
    def __init__(self, config: QualityMetricsConfig):
        self.config = config
        
    @abstractmethod
    def calculate_metric(self, trajectory: TrajectoryData) -> Dict[str, Any]:
        """Calculate quality metric for trajectory."""
        pass
    
    @abstractmethod
    def get_metric_name(self) -> str:
        """Get name of the quality metric."""
        pass


class CompletenessMetrics(BaseQualityMetric):
    """Calculate completeness metrics for trajectories."""
    
    def __init__(self, config: QualityMetricsConfig):
        super().__init__(config)
    
    def get_metric_name(self) -> str:
        return "completeness"
    
    def calculate_metric(self, trajectory: TrajectoryData) -> Dict[str, Any]:
        """Calculate completeness metrics for trajectory."""
        metrics = {}
        
        # Basic completeness checks
        total_points = len(trajectory.timestamps)
        metrics['total_points'] = total_points
        
        # Check minimum length
        is_sufficient_length = total_points >= self.config.min_trajectory_length
        metrics['is_sufficient_length'] = is_sufficient_length
        metrics['length_score'] = min(1.0, total_points / self.config.min_trajectory_length)
        
        # Check for missing values
        missing_timestamps = np.sum(np.isnan(trajectory.timestamps))
        missing_x = np.sum(np.isnan(trajectory.x_positions))
        missing_y = np.sum(np.isnan(trajectory.y_positions))
        
        total_missing = missing_timestamps + missing_x + missing_y
        missing_ratio = total_missing / (total_points * 3)  # 3 fields per point
        
        metrics['missing_values'] = total_missing
        metrics['missing_ratio'] = missing_ratio
        metrics['is_acceptable_missing'] = missing_ratio <= self.config.max_missing_ratio
        metrics['completeness_score'] = 1.0 - missing_ratio
        
        # Time coverage analysis
        if len(trajectory.timestamps) > 1:
            time_coverage = self._calculate_time_coverage(trajectory.timestamps)
            metrics['time_coverage'] = time_coverage
            metrics['is_acceptable_time_coverage'] = time_coverage >= self.config.min_time_coverage
        else:
            metrics['time_coverage'] = 0.0
            metrics['is_acceptable_time_coverage'] = False
        
        # Data quality indicators
        metrics['has_valid_timestamps'] = not np.any(np.isnan(trajectory.timestamps))
        metrics['has_valid_positions'] = not (np.any(np.isnan(trajectory.x_positions)) or np.any(np.isnan(trajectory.y_positions)))
        
        # Overall completeness score
        completeness_factors = [
            metrics['length_score'],
            metrics['completeness_score'],
            metrics['time_coverage']
        ]
        metrics['overall_completeness_score'] = np.mean(completeness_factors)
        
        return metrics
    
    def _calculate_time_coverage(self, timestamps: np.ndarray) -> float:
        """Calculate time coverage ratio."""
        if len(timestamps) < 2:
            return 0.0
        
        # Remove NaN values
        valid_timestamps = timestamps[~np.isnan(timestamps)]
        
        if len(valid_timestamps) < 2:
            return 0.0
        
        # Calculate expected time intervals
        total_duration = valid_timestamps[-1] - valid_timestamps[0]
        expected_intervals = total_duration / (len(valid_timestamps) - 1)
        
        # Calculate actual intervals
        actual_intervals = np.diff(valid_timestamps)
        
        # Calculate coverage as ratio of consistent intervals
        consistent_intervals = np.sum(np.abs(actual_intervals - expected_intervals) < expected_intervals * 0.1)
        coverage = consistent_intervals / len(actual_intervals) if len(actual_intervals) > 0 else 0.0
        
        return coverage


class SmoothnessMetrics(BaseQualityMetric):
    """Calculate smoothness metrics for trajectories."""
    
    def __init__(self, config: QualityMetricsConfig):
        super().__init__(config)
    
    def get_metric_name(self) -> str:
        return "smoothness"
    
    def calculate_metric(self, trajectory: TrajectoryData) -> Dict[str, Any]:
        """Calculate smoothness metrics for trajectory."""
        metrics = {}
        
        if len(trajectory.timestamps) < 3:
            metrics.update(self._get_empty_smoothness_metrics())
            return metrics
        
        # Calculate velocity and acceleration
        velocity, acceleration = self._calculate_kinematics(trajectory)
        
        # Position smoothness
        position_smoothness = self._calculate_position_smoothness(trajectory)
        metrics.update(position_smoothness)
        
        # Velocity smoothness
        velocity_smoothness = self._calculate_velocity_smoothness(velocity)
        metrics.update(velocity_smoothness)
        
        # Acceleration smoothness
        acceleration_smoothness = self._calculate_acceleration_smoothness(acceleration)
        metrics.update(acceleration_smoothness)
        
        # Jerk analysis
        jerk_smoothness = self._calculate_jerk_smoothness(acceleration)
        metrics.update(jerk_smoothness)
        
        # Overall smoothness score
        smoothness_factors = [
            metrics['position_smoothness_score'],
            metrics['velocity_smoothness_score'],
            metrics['acceleration_smoothness_score'],
            metrics['jerk_smoothness_score']
        ]
        metrics['overall_smoothness_score'] = np.mean(smoothness_factors)
        
        return metrics
    
    def _get_empty_smoothness_metrics(self) -> Dict[str, Any]:
        """Return empty smoothness metrics for insufficient data."""
        return {
            'position_smoothness_score': 0.0,
            'velocity_smoothness_score': 0.0,
            'acceleration_smoothness_score': 0.0,
            'jerk_smoothness_score': 0.0,
            'overall_smoothness_score': 0.0,
            'position_jumps': 0,
            'velocity_jumps': 0,
            'acceleration_jumps': 0,
            'jerk_violations': 0
        }
    
    def _calculate_kinematics(self, trajectory: TrajectoryData) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate velocity and acceleration from trajectory."""
        dt = np.diff(trajectory.timestamps)
        dt = np.where(dt == 0, 1e-6, dt)
        
        dx = np.diff(trajectory.x_positions)
        dy = np.diff(trajectory.y_positions)
        
        velocity_x = dx / dt
        velocity_y = dy / dt
        velocity = np.sqrt(velocity_x**2 + velocity_y**2)
        
        if len(velocity) > 1:
            dt_acc = dt[:-1]
            dt_acc = np.where(dt_acc == 0, 1e-6, dt_acc)
            
            acceleration_x = np.diff(velocity_x) / dt_acc
            acceleration_y = np.diff(velocity_y) / dt_acc
            acceleration = np.sqrt(acceleration_x**2 + acceleration_y**2)
        else:
            acceleration = np.array([])
        
        return velocity, acceleration
    
    def _calculate_position_smoothness(self, trajectory: TrajectoryData) -> Dict[str, Any]:
        """Calculate position smoothness metrics."""
        metrics = {}
        
        # Calculate position jumps
        dx = np.diff(trajectory.x_positions)
        dy = np.diff(trajectory.y_positions)
        position_jumps = np.sqrt(dx**2 + dy**2)
        
        # Count significant jumps
        significant_jumps = np.sum(position_jumps > self.config.max_position_jump)
        metrics['position_jumps'] = significant_jumps
        
        # Calculate smoothness score
        if len(position_jumps) > 0:
            jump_ratio = significant_jumps / len(position_jumps)
            metrics['position_smoothness_score'] = 1.0 - jump_ratio
        else:
            metrics['position_smoothness_score'] = 1.0
        
        return metrics
    
    def _calculate_velocity_smoothness(self, velocity: np.ndarray) -> Dict[str, Any]:
        """Calculate velocity smoothness metrics."""
        metrics = {}
        
        if len(velocity) < 2:
            metrics['velocity_smoothness_score'] = 1.0
            metrics['velocity_jumps'] = 0
            return metrics
        
        # Calculate velocity jumps
        velocity_jumps = np.abs(np.diff(velocity))
        significant_jumps = np.sum(velocity_jumps > self.config.max_velocity_jump)
        metrics['velocity_jumps'] = significant_jumps
        
        # Calculate smoothness score
        jump_ratio = significant_jumps / len(velocity_jumps)
        metrics['velocity_smoothness_score'] = 1.0 - jump_ratio
        
        return metrics
    
    def _calculate_acceleration_smoothness(self, acceleration: np.ndarray) -> Dict[str, Any]:
        """Calculate acceleration smoothness metrics."""
        metrics = {}
        
        if len(acceleration) < 2:
            metrics['acceleration_smoothness_score'] = 1.0
            metrics['acceleration_jumps'] = 0
            return metrics
        
        # Calculate acceleration jumps
        acceleration_jumps = np.abs(np.diff(acceleration))
        significant_jumps = np.sum(acceleration_jumps > self.config.max_acceleration_jump)
        metrics['acceleration_jumps'] = significant_jumps
        
        # Calculate smoothness score
        jump_ratio = significant_jumps / len(acceleration_jumps)
        metrics['acceleration_smoothness_score'] = 1.0 - jump_ratio
        
        return metrics
    
    def _calculate_jerk_smoothness(self, acceleration: np.ndarray) -> Dict[str, Any]:
        """Calculate jerk smoothness metrics."""
        metrics = {}
        
        if len(acceleration) < 2:
            metrics['jerk_smoothness_score'] = 1.0
            metrics['jerk_violations'] = 0
            return metrics
        
        # Calculate jerk
        jerk = np.diff(acceleration)
        
        # Count jerk violations
        jerk_violations = np.sum(np.abs(jerk) > self.config.max_jerk)
        metrics['jerk_violations'] = jerk_violations
        
        # Calculate smoothness score
        violation_ratio = jerk_violations / len(jerk)
        metrics['jerk_smoothness_score'] = 1.0 - violation_ratio
        
        return metrics


class ConsistencyMetrics(BaseQualityMetric):
    """Calculate consistency metrics for trajectories."""
    
    def __init__(self, config: QualityMetricsConfig):
        super().__init__(config)
    
    def get_metric_name(self) -> str:
        return "consistency"
    
    def calculate_metric(self, trajectory: TrajectoryData) -> Dict[str, Any]:
        """Calculate consistency metrics for trajectory."""
        metrics = {}
        
        if len(trajectory.timestamps) < 2:
            metrics.update(self._get_empty_consistency_metrics())
            return metrics
        
        # Calculate kinematics
        velocity, acceleration = self._calculate_kinematics(trajectory)
        
        # Velocity consistency
        velocity_consistency = self._calculate_velocity_consistency(velocity)
        metrics.update(velocity_consistency)
        
        # Acceleration consistency
        acceleration_consistency = self._calculate_acceleration_consistency(acceleration)
        metrics.update(acceleration_consistency)
        
        # Temporal consistency
        temporal_consistency = self._calculate_temporal_consistency(trajectory.timestamps)
        metrics.update(temporal_consistency)
        
        # Spatial consistency
        spatial_consistency = self._calculate_spatial_consistency(trajectory)
        metrics.update(spatial_consistency)
        
        # Overall consistency score
        consistency_factors = [
            metrics['velocity_consistency_score'],
            metrics['acceleration_consistency_score'],
            metrics['temporal_consistency_score'],
            metrics['spatial_consistency_score']
        ]
        metrics['overall_consistency_score'] = np.mean(consistency_factors)
        
        return metrics
    
    def _get_empty_consistency_metrics(self) -> Dict[str, Any]:
        """Return empty consistency metrics for insufficient data."""
        return {
            'velocity_consistency_score': 0.0,
            'acceleration_consistency_score': 0.0,
            'temporal_consistency_score': 0.0,
            'spatial_consistency_score': 0.0,
            'overall_consistency_score': 0.0,
            'velocity_violations': 0,
            'acceleration_violations': 0,
            'temporal_violations': 0,
            'spatial_violations': 0
        }
    
    def _calculate_kinematics(self, trajectory: TrajectoryData) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate velocity and acceleration from trajectory."""
        dt = np.diff(trajectory.timestamps)
        dt = np.where(dt == 0, 1e-6, dt)
        
        dx = np.diff(trajectory.x_positions)
        dy = np.diff(trajectory.y_positions)
        
        velocity_x = dx / dt
        velocity_y = dy / dt
        velocity = np.sqrt(velocity_x**2 + velocity_y**2)
        
        if len(velocity) > 1:
            dt_acc = dt[:-1]
            dt_acc = np.where(dt_acc == 0, 1e-6, dt_acc)
            
            acceleration_x = np.diff(velocity_x) / dt_acc
            acceleration_y = np.diff(velocity_y) / dt_acc
            acceleration = np.sqrt(acceleration_x**2 + acceleration_y**2)
        else:
            acceleration = np.array([])
        
        return velocity, acceleration
    
    def _calculate_velocity_consistency(self, velocity: np.ndarray) -> Dict[str, Any]:
        """Calculate velocity consistency metrics."""
        metrics = {}
        
        if len(velocity) == 0:
            metrics['velocity_consistency_score'] = 1.0
            metrics['velocity_violations'] = 0
            return metrics
        
        # Check velocity bounds
        min_violations = np.sum(velocity < self.config.min_velocity)
        max_violations = np.sum(velocity > self.config.max_velocity)
        total_violations = min_violations + max_violations
        
        metrics['velocity_violations'] = total_violations
        
        # Calculate consistency score
        violation_ratio = total_violations / len(velocity)
        metrics['velocity_consistency_score'] = 1.0 - violation_ratio
        
        return metrics
    
    def _calculate_acceleration_consistency(self, acceleration: np.ndarray) -> Dict[str, Any]:
        """Calculate acceleration consistency metrics."""
        metrics = {}
        
        if len(acceleration) == 0:
            metrics['acceleration_consistency_score'] = 1.0
            metrics['acceleration_violations'] = 0
            return metrics
        
        # Check acceleration bounds
        violations = np.sum(np.abs(acceleration) > self.config.max_acceleration)
        metrics['acceleration_violations'] = violations
        
        # Calculate consistency score
        violation_ratio = violations / len(acceleration)
        metrics['acceleration_consistency_score'] = 1.0 - violation_ratio
        
        return metrics
    
    def _calculate_temporal_consistency(self, timestamps: np.ndarray) -> Dict[str, Any]:
        """Calculate temporal consistency metrics."""
        metrics = {}
        
        if len(timestamps) < 2:
            metrics['temporal_consistency_score'] = 1.0
            metrics['temporal_violations'] = 0
            return metrics
        
        # Check for non-increasing timestamps
        dt = np.diff(timestamps)
        violations = np.sum(dt <= 0)
        metrics['temporal_violations'] = violations
        
        # Calculate consistency score
        violation_ratio = violations / len(dt)
        metrics['temporal_consistency_score'] = 1.0 - violation_ratio
        
        return metrics
    
    def _calculate_spatial_consistency(self, trajectory: TrajectoryData) -> Dict[str, Any]:
        """Calculate spatial consistency metrics."""
        metrics = {}
        
        if len(trajectory.x_positions) < 2:
            metrics['spatial_consistency_score'] = 1.0
            metrics['spatial_violations'] = 0
            return metrics
        
        # Check for duplicate positions
        positions = np.column_stack([trajectory.x_positions, trajectory.y_positions])
        unique_positions = np.unique(positions, axis=0)
        violations = len(positions) - len(unique_positions)
        
        metrics['spatial_violations'] = violations
        
        # Calculate consistency score
        violation_ratio = violations / len(positions)
        metrics['spatial_consistency_score'] = 1.0 - violation_ratio
        
        return metrics


class PhysicsConstraintValidator(BaseQualityMetric):
    """Validate physics constraints for trajectories."""
    
    def __init__(self, config: QualityMetricsConfig):
        super().__init__(config)
    
    def get_metric_name(self) -> str:
        return "physics_constraints"
    
    def calculate_metric(self, trajectory: TrajectoryData) -> Dict[str, Any]:
        """Calculate physics constraint validation metrics."""
        metrics = {}
        
        if len(trajectory.timestamps) < 3:
            metrics.update(self._get_empty_physics_metrics())
            return metrics
        
        # Calculate kinematics
        velocity, acceleration = self._calculate_kinematics(trajectory)
        
        # Centripetal acceleration validation
        centripetal_validation = self._validate_centripetal_acceleration(trajectory, velocity)
        metrics.update(centripetal_validation)
        
        # Turning radius validation
        turning_radius_validation = self._validate_turning_radius(trajectory)
        metrics.update(turning_radius_validation)
        
        # Angular velocity validation
        angular_velocity_validation = self._validate_angular_velocity(trajectory)
        metrics.update(angular_velocity_validation)
        
        # Energy conservation validation
        energy_validation = self._validate_energy_conservation(trajectory, velocity, acceleration)
        metrics.update(energy_validation)
        
        # Overall physics score
        physics_factors = [
            metrics['centripetal_physics_score'],
            metrics['turning_radius_physics_score'],
            metrics['angular_velocity_physics_score'],
            metrics['energy_physics_score']
        ]
        metrics['overall_physics_score'] = np.mean(physics_factors)
        
        return metrics
    
    def _get_empty_physics_metrics(self) -> Dict[str, Any]:
        """Return empty physics metrics for insufficient data."""
        return {
            'centripetal_physics_score': 1.0,
            'turning_radius_physics_score': 1.0,
            'angular_velocity_physics_score': 1.0,
            'energy_physics_score': 1.0,
            'overall_physics_score': 1.0,
            'centripetal_violations': 0,
            'turning_radius_violations': 0,
            'angular_velocity_violations': 0,
            'energy_violations': 0
        }
    
    def _calculate_kinematics(self, trajectory: TrajectoryData) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate velocity and acceleration from trajectory."""
        dt = np.diff(trajectory.timestamps)
        dt = np.where(dt == 0, 1e-6, dt)
        
        dx = np.diff(trajectory.x_positions)
        dy = np.diff(trajectory.y_positions)
        
        velocity_x = dx / dt
        velocity_y = dy / dt
        velocity = np.sqrt(velocity_x**2 + velocity_y**2)
        
        if len(velocity) > 1:
            dt_acc = dt[:-1]
            dt_acc = np.where(dt_acc == 0, 1e-6, dt_acc)
            
            acceleration_x = np.diff(velocity_x) / dt_acc
            acceleration_y = np.diff(velocity_y) / dt_acc
            acceleration = np.sqrt(acceleration_x**2 + acceleration_y**2)
        else:
            acceleration = np.array([])
        
        return velocity, acceleration
    
    def _validate_centripetal_acceleration(self, trajectory: TrajectoryData, velocity: np.ndarray) -> Dict[str, Any]:
        """Validate centripetal acceleration constraints."""
        metrics = {}
        
        if len(velocity) < 2:
            metrics['centripetal_physics_score'] = 1.0
            metrics['centripetal_violations'] = 0
            return metrics
        
        # Calculate curvature and centripetal acceleration
        x = trajectory.x_positions
        y = trajectory.y_positions
        
        # Calculate curvature using three-point method
        curvature = np.zeros(len(x))
        for i in range(1, len(x) - 1):
            p1 = np.array([x[i-1], y[i-1]])
            p2 = np.array([x[i], y[i]])
            p3 = np.array([x[i+1], y[i+1]])
            
            a = np.linalg.norm(p2 - p3)
            b = np.linalg.norm(p1 - p3)
            c = np.linalg.norm(p1 - p2)
            
            if a * b * c > 0:
                s = (a + b + c) / 2
                area = np.sqrt(s * (s - a) * (s - b) * (s - c))
                if area > 0:
                    radius = (a * b * c) / (4 * area)
                    curvature[i] = 1 / radius if radius > 0 else 0
        
        # Calculate centripetal acceleration: a = v^2 / r = v^2 * curvature
        centripetal_acceleration = velocity**2 * curvature[1:-1]  # Align with velocity
        
        # Count violations
        violations = np.sum(np.abs(centripetal_acceleration) > self.config.max_centripetal_acceleration)
        metrics['centripetal_violations'] = violations
        
        # Calculate physics score
        violation_ratio = violations / len(centripetal_acceleration) if len(centripetal_acceleration) > 0 else 0.0
        metrics['centripetal_physics_score'] = 1.0 - violation_ratio
        
        return metrics
    
    def _validate_turning_radius(self, trajectory: TrajectoryData) -> Dict[str, Any]:
        """Validate turning radius constraints."""
        metrics = {}
        
        x = trajectory.x_positions
        y = trajectory.y_positions
        
        if len(x) < 3:
            metrics['turning_radius_physics_score'] = 1.0
            metrics['turning_radius_violations'] = 0
            return metrics
        
        # Calculate turning radius using three-point method
        turning_radii = np.zeros(len(x))
        for i in range(1, len(x) - 1):
            p1 = np.array([x[i-1], y[i-1]])
            p2 = np.array([x[i], y[i]])
            p3 = np.array([x[i+1], y[i+1]])
            
            a = np.linalg.norm(p2 - p3)
            b = np.linalg.norm(p1 - p3)
            c = np.linalg.norm(p1 - p2)
            
            if a * b * c > 0:
                s = (a + b + c) / 2
                area = np.sqrt(s * (s - a) * (s - b) * (s - c))
                if area > 0:
                    radius = (a * b * c) / (4 * area)
                    turning_radii[i] = radius
        
        # Count violations (radius too small)
        violations = np.sum(turning_radii < self.config.min_turning_radius)
        metrics['turning_radius_violations'] = violations
        
        # Calculate physics score
        violation_ratio = violations / len(turning_radii) if len(turning_radii) > 0 else 0.0
        metrics['turning_radius_physics_score'] = 1.0 - violation_ratio
        
        return metrics
    
    def _validate_angular_velocity(self, trajectory: TrajectoryData) -> Dict[str, Any]:
        """Validate angular velocity constraints."""
        metrics = {}
        
        if len(trajectory.timestamps) < 2:
            metrics['angular_velocity_physics_score'] = 1.0
            metrics['angular_velocity_violations'] = 0
            return metrics
        
        # Calculate angular velocity
        x = trajectory.x_positions
        y = trajectory.y_positions
        timestamps = trajectory.timestamps
        
        # Calculate heading angles
        headings = np.arctan2(np.diff(y), np.diff(x))
        
        # Calculate angular velocity
        dt = np.diff(timestamps)
        dt = np.where(dt == 0, 1e-6, dt)
        
        angular_velocity = np.diff(headings) / dt[:-1]
        
        # Normalize angular velocity to [-π, π]
        angular_velocity = np.arctan2(np.sin(angular_velocity), np.cos(angular_velocity))
        
        # Count violations
        violations = np.sum(np.abs(angular_velocity) > self.config.max_angular_velocity)
        metrics['angular_velocity_violations'] = violations
        
        # Calculate physics score
        violation_ratio = violations / len(angular_velocity) if len(angular_velocity) > 0 else 0.0
        metrics['angular_velocity_physics_score'] = 1.0 - violation_ratio
        
        return metrics
    
    def _validate_energy_conservation(self, trajectory: TrajectoryData, velocity: np.ndarray, acceleration: np.ndarray) -> Dict[str, Any]:
        """Validate energy conservation constraints."""
        metrics = {}
        
        if len(velocity) < 2:
            metrics['energy_physics_score'] = 1.0
            metrics['energy_violations'] = 0
            return metrics
        
        # Calculate kinetic energy changes
        kinetic_energy = 0.5 * velocity**2
        energy_changes = np.diff(kinetic_energy)
        
        # Calculate work done by acceleration
        if len(acceleration) > 0:
            # Approximate work as force * distance
            dt = np.diff(trajectory.timestamps)[:-1]  # Align with acceleration
            dt = np.where(dt == 0, 1e-6, dt)
            
            # Work = force * distance = mass * acceleration * velocity * dt
            # Assuming unit mass
            work_done = acceleration * velocity[:-1] * dt
            
            # Energy conservation: ΔKE ≈ Work
            energy_balance = np.abs(energy_changes - work_done)
            
            # Count violations (large energy imbalances)
            threshold = np.std(energy_balance) * 2  # 2 standard deviations
            violations = np.sum(energy_balance > threshold)
            metrics['energy_violations'] = violations
            
            # Calculate physics score
            violation_ratio = violations / len(energy_balance) if len(energy_balance) > 0 else 0.0
            metrics['energy_physics_score'] = 1.0 - violation_ratio
        else:
            metrics['energy_violations'] = 0
            metrics['energy_physics_score'] = 1.0
        
        return metrics


class TrajectoryQualityMetrics:
    """Main quality metrics calculator that combines all quality metrics."""
    
    def __init__(self, config: QualityMetricsConfig):
        self.config = config
        self.metrics = {
            'completeness': CompletenessMetrics(config),
            'smoothness': SmoothnessMetrics(config),
            'consistency': ConsistencyMetrics(config),
            'physics': PhysicsConstraintValidator(config)
        }
    
    def calculate_quality_metrics(self, trajectory: TrajectoryData) -> Dict[str, Any]:
        """Calculate all quality metrics for trajectory."""
        all_metrics = {}
        
        for metric_name, metric_calculator in self.metrics.items():
            try:
                metrics = metric_calculator.calculate_metric(trajectory)
                all_metrics[metric_name] = metrics
                logger.debug(f"Calculated {len(metrics)} metrics for {metric_name}")
            except Exception as e:
                logger.error(f"Error calculating {metric_name} metrics: {e}")
                all_metrics[metric_name] = {}
        
        # Calculate overall quality score
        overall_score = self._calculate_overall_quality_score(all_metrics)
        all_metrics['overall'] = {'quality_score': overall_score}
        
        return all_metrics
    
    def _calculate_overall_quality_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall quality score from individual metrics."""
        scores = []
        weights = []
        
        # Completeness score
        if 'completeness' in metrics and 'overall_completeness_score' in metrics['completeness']:
            scores.append(metrics['completeness']['overall_completeness_score'])
            weights.append(self.config.completeness_weight)
        
        # Smoothness score
        if 'smoothness' in metrics and 'overall_smoothness_score' in metrics['smoothness']:
            scores.append(metrics['smoothness']['overall_smoothness_score'])
            weights.append(self.config.smoothness_weight)
        
        # Consistency score
        if 'consistency' in metrics and 'overall_consistency_score' in metrics['consistency']:
            scores.append(metrics['consistency']['overall_consistency_score'])
            weights.append(self.config.consistency_weight)
        
        # Physics score
        if 'physics' in metrics and 'overall_physics_score' in metrics['physics']:
            scores.append(metrics['physics']['overall_physics_score'])
            weights.append(self.config.physics_weight)
        
        # Calculate weighted average
        if scores and weights:
            total_weight = sum(weights)
            if total_weight > 0:
                weighted_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
                return weighted_score
        
        return 0.0
    
    def get_quality_summary(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Get a summary of quality metrics."""
        summary = {
            'overall_quality_score': metrics.get('overall', {}).get('quality_score', 0.0),
            'completeness_score': metrics.get('completeness', {}).get('overall_completeness_score', 0.0),
            'smoothness_score': metrics.get('smoothness', {}).get('overall_smoothness_score', 0.0),
            'consistency_score': metrics.get('consistency', {}).get('overall_consistency_score', 0.0),
            'physics_score': metrics.get('physics', {}).get('overall_physics_score', 0.0),
            'total_violations': 0,
            'quality_level': 'unknown'
        }
        
        # Count total violations
        for metric_type in ['completeness', 'smoothness', 'consistency', 'physics']:
            if metric_type in metrics:
                for key, value in metrics[metric_type].items():
                    if 'violations' in key and isinstance(value, (int, float)):
                        summary['total_violations'] += value
        
        # Determine quality level
        overall_score = summary['overall_quality_score']
        if overall_score >= 0.9:
            summary['quality_level'] = 'excellent'
        elif overall_score >= 0.8:
            summary['quality_level'] = 'good'
        elif overall_score >= 0.7:
            summary['quality_level'] = 'fair'
        elif overall_score >= 0.6:
            summary['quality_level'] = 'poor'
        else:
            summary['quality_level'] = 'very_poor'
        
        return summary