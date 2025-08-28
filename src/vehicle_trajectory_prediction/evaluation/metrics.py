"""Comprehensive metrics for trajectory prediction evaluation."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
import logging
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

from ..core.models import Trajectory, TrajectoryPoint
from ..models.base import PredictionResult

logger = logging.getLogger(__name__)


class TrajectoryMetrics:
    """Core trajectory prediction accuracy metrics."""
    
    @staticmethod
    def calculate_rmse(true_trajectory: Trajectory, predicted_result: PredictionResult) -> float:
        """Calculate Root Mean Square Error."""
        if len(predicted_result.predicted_points) == 0:
            return np.inf
        
        # Get true positions at prediction timestamps
        true_x, true_y = TrajectoryMetrics._extract_true_positions(true_trajectory, predicted_result)
        predicted_x = predicted_result.x_positions
        predicted_y = predicted_result.y_positions
        
        # Ensure same length
        min_length = min(len(true_x), len(predicted_x))
        if min_length == 0:
            return np.inf
        
        true_x = true_x[:min_length]
        true_y = true_y[:min_length]
        predicted_x = predicted_x[:min_length]
        predicted_y = predicted_y[:min_length]
        
        # Calculate Euclidean distance errors
        errors = np.sqrt((np.array(true_x) - predicted_x)**2 + (np.array(true_y) - predicted_y)**2)
        
        return np.sqrt(np.mean(errors**2))
    
    @staticmethod
    def calculate_ade(true_trajectory: Trajectory, predicted_result: PredictionResult) -> float:
        """Calculate Average Displacement Error."""
        if len(predicted_result.predicted_points) == 0:
            return np.inf
        
        # Get true positions at prediction timestamps
        true_x, true_y = TrajectoryMetrics._extract_true_positions(true_trajectory, predicted_result)
        predicted_x = predicted_result.x_positions
        predicted_y = predicted_result.y_positions
        
        # Ensure same length
        min_length = min(len(true_x), len(predicted_x))
        if min_length == 0:
            return np.inf
        
        true_x = true_x[:min_length]
        true_y = true_y[:min_length]
        predicted_x = predicted_x[:min_length]
        predicted_y = predicted_y[:min_length]
        
        # Calculate Euclidean distance errors
        errors = np.sqrt((np.array(true_x) - predicted_x)**2 + (np.array(true_y) - predicted_y)**2)
        
        return np.mean(errors)
    
    @staticmethod
    def calculate_fde(true_trajectory: Trajectory, predicted_result: PredictionResult) -> float:
        """Calculate Final Displacement Error."""
        if len(predicted_result.predicted_points) == 0:
            return np.inf
        
        # Get true final position
        final_timestamp = predicted_result.timestamps[-1]
        true_final_point = true_trajectory.get_point_at_time(final_timestamp)
        
        if true_final_point is None:
            # Use last known position
            true_final_point = true_trajectory.points[-1]
        
        # Get predicted final position
        predicted_final_x = predicted_result.x_positions[-1]
        predicted_final_y = predicted_result.y_positions[-1]
        
        # Calculate final displacement error
        fde = np.sqrt((true_final_point.x - predicted_final_x)**2 + 
                     (true_final_point.y - predicted_final_y)**2)
        
        return fde
    
    @staticmethod
    def calculate_mae(true_trajectory: Trajectory, predicted_result: PredictionResult) -> float:
        """Calculate Mean Absolute Error."""
        if len(predicted_result.predicted_points) == 0:
            return np.inf
        
        # Get true positions at prediction timestamps
        true_x, true_y = TrajectoryMetrics._extract_true_positions(true_trajectory, predicted_result)
        predicted_x = predicted_result.x_positions
        predicted_y = predicted_result.y_positions
        
        # Ensure same length
        min_length = min(len(true_x), len(predicted_x))
        if min_length == 0:
            return np.inf
        
        true_x = true_x[:min_length]
        true_y = true_y[:min_length]
        predicted_x = predicted_x[:min_length]
        predicted_y = predicted_y[:min_length]
        
        # Calculate Euclidean distance errors
        errors = np.sqrt((np.array(true_x) - predicted_x)**2 + (np.array(true_y) - predicted_y)**2)
        
        return np.mean(np.abs(errors))
    
    @staticmethod
    def calculate_trajectory_similarity(true_trajectory: Trajectory, predicted_result: PredictionResult) -> float:
        """Calculate trajectory similarity using Dynamic Time Warping distance."""
        if len(predicted_result.predicted_points) == 0:
            return np.inf
        
        # Get true positions at prediction timestamps
        true_x, true_y = TrajectoryMetrics._extract_true_positions(true_trajectory, predicted_result)
        predicted_x = predicted_result.x_positions
        predicted_y = predicted_result.y_positions
        
        # Ensure same length
        min_length = min(len(true_x), len(predicted_x))
        if min_length == 0:
            return np.inf
        
        true_x = true_x[:min_length]
        true_y = true_y[:min_length]
        predicted_x = predicted_x[:min_length]
        predicted_y = predicted_y[:min_length]
        
        # Calculate DTW distance
        true_traj = np.column_stack([true_x, true_y])
        pred_traj = np.column_stack([predicted_x, predicted_y])
        
        # Simple DTW implementation
        dtw_distance = TrajectoryMetrics._dtw_distance(true_traj, pred_traj)
        
        return dtw_distance
    
    @staticmethod
    def _extract_true_positions(true_trajectory: Trajectory, predicted_result: PredictionResult) -> Tuple[List[float], List[float]]:
        """Extract true positions at prediction timestamps."""
        true_x = []
        true_y = []
        
        for timestamp in predicted_result.timestamps:
            true_point = true_trajectory.get_point_at_time(timestamp)
            if true_point is not None:
                true_x.append(true_point.x)
                true_y.append(true_point.y)
            else:
                # If no true point, use last known position
                true_x.append(true_trajectory.points[-1].x)
                true_y.append(true_trajectory.points[-1].y)
        
        return true_x, true_y
    
    @staticmethod
    def _dtw_distance(traj1: np.ndarray, traj2: np.ndarray) -> float:
        """Calculate Dynamic Time Warping distance between two trajectories."""
        n, m = len(traj1), len(traj2)
        
        # Initialize DTW matrix
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0
        
        # Fill DTW matrix
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = np.linalg.norm(traj1[i-1] - traj2[j-1])
                dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1])
        
        return dtw_matrix[n, m]


class SafetyMetrics:
    """Safety-critical metrics for trajectory prediction."""
    
    @staticmethod
    def calculate_min_distance(true_trajectory: Trajectory, predicted_result: PredictionResult) -> float:
        """Calculate minimum distance between true and predicted trajectories."""
        if len(predicted_result.predicted_points) == 0:
            return np.inf
        
        # Get true positions at prediction timestamps
        true_x, true_y = TrajectoryMetrics._extract_true_positions(true_trajectory, predicted_result)
        predicted_x = predicted_result.x_positions
        predicted_y = predicted_result.y_positions
        
        # Ensure same length
        min_length = min(len(true_x), len(predicted_x))
        if min_length == 0:
            return np.inf
        
        true_x = true_x[:min_length]
        true_y = true_y[:min_length]
        predicted_x = predicted_x[:min_length]
        predicted_y = predicted_y[:min_length]
        
        # Calculate minimum distance
        true_points = np.column_stack([true_x, true_y])
        pred_points = np.column_stack([predicted_x, predicted_y])
        
        distances = cdist(true_points, pred_points)
        min_dist = np.min(distances)
        
        return min_dist
    
    @staticmethod
    def calculate_ttc(true_trajectory: Trajectory, predicted_result: PredictionResult, 
                     collision_threshold: float = 2.0) -> float:
        """Calculate Time-to-Collision."""
        if len(predicted_result.predicted_points) < 2:
            return np.inf
        
        # Get true positions at prediction timestamps
        true_x, true_y = TrajectoryMetrics._extract_true_positions(true_trajectory, predicted_result)
        predicted_x = predicted_result.x_positions
        predicted_y = predicted_result.y_positions
        
        # Ensure same length
        min_length = min(len(true_x), len(predicted_x))
        if min_length < 2:
            return np.inf
        
        true_x = true_x[:min_length]
        true_y = true_y[:min_length]
        predicted_x = predicted_x[:min_length]
        predicted_y = predicted_y[:min_length]
        
        # Calculate distances
        distances = np.sqrt((np.array(true_x) - predicted_x)**2 + (np.array(true_y) - predicted_y)**2)
        
        # Find collision points (where distance < threshold)
        collision_indices = np.where(distances < collision_threshold)[0]
        
        if len(collision_indices) == 0:
            return np.inf
        
        # Calculate TTC for each collision point
        ttc_values = []
        for idx in collision_indices:
            if idx > 0:
                # Calculate relative velocity
                distance_change = distances[idx] - distances[idx-1]
                time_step = (predicted_result.timestamps[idx] - predicted_result.timestamps[idx-1]).total_seconds()
                
                if time_step > 0 and abs(distance_change) > 1e-6:
                    relative_velocity = abs(distance_change) / time_step
                    if relative_velocity > 1e-6:
                        ttc = distances[idx] / relative_velocity
                        if ttc > 0:
                            ttc_values.append(ttc)
        
        if ttc_values:
            return min(ttc_values)
        
        return np.inf
    
    @staticmethod
    def calculate_lateral_error(true_trajectory: Trajectory, predicted_result: PredictionResult) -> float:
        """Calculate lateral error (perpendicular distance from true trajectory)."""
        if len(predicted_result.predicted_points) == 0:
            return np.inf
        
        # Get true positions at prediction timestamps
        true_x, true_y = TrajectoryMetrics._extract_true_positions(true_trajectory, predicted_result)
        predicted_x = predicted_result.x_positions
        predicted_y = predicted_result.y_positions
        
        # Ensure same length
        min_length = min(len(true_x), len(predicted_x))
        if min_length < 2:
            return np.inf
        
        true_x = true_x[:min_length]
        true_y = true_y[:min_length]
        predicted_x = predicted_x[:min_length]
        predicted_y = predicted_y[:min_length]
        
        # Calculate lateral errors
        lateral_errors = []
        for i in range(1, len(true_x)):
            # Calculate trajectory direction
            dx = true_x[i] - true_x[i-1]
            dy = true_y[i] - true_y[i-1]
            
            if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                continue
            
            # Normalize direction vector
            length = np.sqrt(dx**2 + dy**2)
            dx /= length
            dy /= length
            
            # Calculate perpendicular distance from predicted point to trajectory line
            pred_dx = predicted_x[i] - true_x[i-1]
            pred_dy = predicted_y[i] - true_y[i-1]
            
            # Project onto trajectory direction
            projection = pred_dx * dx + pred_dy * dy
            
            # Calculate perpendicular distance
            lateral_dist = np.sqrt((pred_dx - projection * dx)**2 + (pred_dy - projection * dy)**2)
            lateral_errors.append(lateral_dist)
        
        if lateral_errors:
            return np.mean(lateral_errors)
        else:
            return np.inf
    
    @staticmethod
    def calculate_risk_score(true_trajectory: Trajectory, predicted_result: PredictionResult,
                           distance_weight: float = 0.4, ttc_weight: float = 0.4, 
                           lateral_weight: float = 0.2) -> float:
        """Calculate comprehensive risk score combining multiple safety metrics."""
        min_dist = SafetyMetrics.calculate_min_distance(true_trajectory, predicted_result)
        ttc = SafetyMetrics.calculate_ttc(true_trajectory, predicted_result)
        lateral_error = SafetyMetrics.calculate_lateral_error(true_trajectory, predicted_result)
        
        # Normalize metrics to [0, 1] range (higher = more risky)
        # Distance risk: closer = more risky
        distance_risk = max(0, 1 - min_dist / 10.0) if min_dist != np.inf else 1.0
        
        # TTC risk: lower TTC = more risky
        ttc_risk = max(0, 1 - ttc / 5.0) if ttc != np.inf else 1.0
        
        # Lateral error risk: higher error = more risky
        lateral_risk = min(1.0, lateral_error / 5.0) if lateral_error != np.inf else 1.0
        
        # Calculate weighted risk score
        risk_score = (distance_weight * distance_risk + 
                     ttc_weight * ttc_risk + 
                     lateral_weight * lateral_risk)
        
        return risk_score


class StatisticalMetrics:
    """Statistical metrics for trajectory prediction evaluation."""
    
    @staticmethod
    def calculate_confidence_interval(values: List[float], confidence_level: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for a list of values."""
        if len(values) == 0:
            return np.nan, np.nan
        
        # Remove infinite values
        finite_values = [v for v in values if not np.isinf(v) and not np.isnan(v)]
        
        if len(finite_values) == 0:
            return np.nan, np.nan
        
        # Calculate confidence interval
        mean_val = np.mean(finite_values)
        std_val = np.std(finite_values, ddof=1)
        
        if std_val == 0:
            return mean_val, mean_val
        
        # Calculate t-statistic for confidence interval
        t_value = stats.t.ppf((1 + confidence_level) / 2, len(finite_values) - 1)
        margin_of_error = t_value * std_val / np.sqrt(len(finite_values))
        
        return mean_val - margin_of_error, mean_val + margin_of_error
    
    @staticmethod
    def calculate_percentiles(values: List[float], percentiles: List[float] = None) -> Dict[str, float]:
        """Calculate percentiles for a list of values."""
        if percentiles is None:
            percentiles = [25, 50, 75, 90, 95, 99]
        
        if len(values) == 0:
            return {f'p{p}': np.nan for p in percentiles}
        
        # Remove infinite values
        finite_values = [v for v in values if not np.isinf(v) and not np.isnan(v)]
        
        if len(finite_values) == 0:
            return {f'p{p}': np.nan for p in percentiles}
        
        # Calculate percentiles
        percentile_values = np.percentile(finite_values, percentiles)
        
        return {f'p{p}': float(val) for p, val in zip(percentiles, percentile_values)}
    
    @staticmethod
    def calculate_outlier_rate(values: List[float], method: str = 'iqr') -> float:
        """Calculate outlier rate using specified method."""
        if len(values) == 0:
            return 0.0
        
        # Remove infinite values
        finite_values = [v for v in values if not np.isinf(v) and not np.isnan(v)]
        
        if len(finite_values) == 0:
            return 0.0
        
        if method == 'iqr':
            # IQR method
            q1, q3 = np.percentile(finite_values, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = sum(1 for v in finite_values if v < lower_bound or v > upper_bound)
            return outliers / len(finite_values)
        
        elif method == 'zscore':
            # Z-score method
            mean_val = np.mean(finite_values)
            std_val = np.std(finite_values)
            
            if std_val == 0:
                return 0.0
            
            outliers = sum(1 for v in finite_values if abs(v - mean_val) > 3 * std_val)
            return outliers / len(finite_values)
        
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
    
    @staticmethod
    def calculate_distribution_stats(values: List[float]) -> Dict[str, float]:
        """Calculate comprehensive distribution statistics."""
        if len(values) == 0:
            return {
                'mean': np.nan, 'std': np.nan, 'median': np.nan,
                'skewness': np.nan, 'kurtosis': np.nan,
                'min': np.nan, 'max': np.nan
            }
        
        # Remove infinite values
        finite_values = [v for v in values if not np.isinf(v) and not np.isnan(v)]
        
        if len(finite_values) == 0:
            return {
                'mean': np.nan, 'std': np.nan, 'median': np.nan,
                'skewness': np.nan, 'kurtosis': np.nan,
                'min': np.nan, 'max': np.nan
            }
        
        # Calculate statistics
        mean_val = np.mean(finite_values)
        std_val = np.std(finite_values, ddof=1)
        median_val = np.median(finite_values)
        
        # Calculate skewness and kurtosis
        if std_val > 0:
            skewness = stats.skew(finite_values)
            kurtosis = stats.kurtosis(finite_values)
        else:
            skewness = 0.0
            kurtosis = 0.0
        
        return {
            'mean': float(mean_val),
            'std': float(std_val),
            'median': float(median_val),
            'skewness': float(skewness),
            'kurtosis': float(kurtosis),
            'min': float(min(finite_values)),
            'max': float(max(finite_values))
        }


class PerformanceMetrics:
    """Performance and efficiency metrics for trajectory prediction models."""
    
    @staticmethod
    def measure_inference_time(model, trajectory: Trajectory, 
                             prediction_horizon: int = 30, 
                             num_runs: int = 10) -> Dict[str, float]:
        """Measure inference time for a model."""
        import time
        
        times = []
        
        for _ in range(num_runs):
            start_time = time.time()
            try:
                model.predict(trajectory, prediction_horizon)
                end_time = time.time()
                times.append(end_time - start_time)
            except Exception as e:
                logger.warning(f"Failed to measure inference time: {e}")
                continue
        
        if not times:
            return {
                'mean_time': np.nan,
                'std_time': np.nan,
                'min_time': np.nan,
                'max_time': np.nan
            }
        
        return {
            'mean_time': float(np.mean(times)),
            'std_time': float(np.std(times)),
            'min_time': float(min(times)),
            'max_time': float(max(times))
        }
    
    @staticmethod
    def measure_memory_usage(model, trajectory: Trajectory, 
                           prediction_horizon: int = 30) -> Dict[str, float]:
        """Measure memory usage for a model prediction."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        try:
            model.predict(trajectory, prediction_horizon)
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = final_memory - initial_memory
        except Exception as e:
            logger.warning(f"Failed to measure memory usage: {e}")
            memory_used = np.nan
        
        return {
            'memory_used_mb': float(memory_used) if not np.isnan(memory_used) else np.nan,
            'peak_memory_mb': float(final_memory) if 'final_memory' in locals() else np.nan
        }
    
    @staticmethod
    def calculate_throughput(model, trajectories: List[Trajectory], 
                           prediction_horizon: int = 30) -> Dict[str, float]:
        """Calculate prediction throughput (trajectories per second)."""
        import time
        
        if not trajectories:
            return {'throughput_traj_per_sec': np.nan}
        
        start_time = time.time()
        successful_predictions = 0
        
        for trajectory in trajectories:
            try:
                model.predict(trajectory, prediction_horizon)
                successful_predictions += 1
            except Exception as e:
                logger.warning(f"Failed prediction during throughput test: {e}")
                continue
        
        end_time = time.time()
        total_time = end_time - start_time
        
        if total_time > 0:
            throughput = successful_predictions / total_time
        else:
            throughput = np.nan
        
        return {
            'throughput_traj_per_sec': float(throughput) if not np.isnan(throughput) else np.nan,
            'total_time_sec': float(total_time),
            'successful_predictions': successful_predictions,
            'total_trajectories': len(trajectories)
        }