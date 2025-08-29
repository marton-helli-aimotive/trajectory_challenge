"""
Comprehensive trajectory prediction evaluation metrics.

This module implements safety-critical, standard ML, and probabilistic
metrics for evaluating trajectory prediction models.
"""

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist
from omegaconf import DictConfig

from ..models.base import PredictionResult, TrajectoryPredictor
from ..data.validation.schemas import TrajectoryData


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    model_name: str
    metric_name: str
    value: float
    std_error: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    metadata: Optional[Dict[str, Any]] = None


class SafetyMetrics:
    """
    Safety-critical metrics for trajectory prediction evaluation.
    
    Implements metrics crucial for autonomous vehicle safety:
    - Time-to-Collision (TTC)
    - Minimum distance metrics
    - Collision probability
    - Safety margin analysis
    """
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.vehicle_length = config.get("vehicle_length", 4.5)  # meters
        self.vehicle_width = config.get("vehicle_width", 1.8)   # meters
        self.safety_buffer = config.get("safety_buffer", 2.0)   # meters
        self.collision_threshold = config.get("collision_threshold", 1.0)  # meters
        
    async def time_to_collision(
        self,
        prediction: PredictionResult,
        other_trajectories: List[TrajectoryData]
    ) -> Dict[str, float]:
        """
        Calculate Time-to-Collision (TTC) between predicted and other trajectories.
        
        Args:
            prediction: Predicted trajectory
            other_trajectories: Other vehicle trajectories
            
        Returns:
            Dictionary with TTC metrics
        """
        if not other_trajectories:
            return {"min_ttc": float("inf"), "avg_ttc": float("inf"), "collision_risk": 0.0}
        
        # Convert prediction to trajectory points
        pred_points = list(zip(
            prediction.timestamps / 1000.0,  # Convert to seconds
            prediction.x_coords,
            prediction.y_coords
        ))
        
        ttc_values = []
        
        for other_traj in other_trajectories:
            ttc = await self._calculate_pairwise_ttc(pred_points, other_traj)
            if ttc is not None and ttc > 0:
                ttc_values.append(ttc)
        
        if not ttc_values:
            return {"min_ttc": float("inf"), "avg_ttc": float("inf"), "collision_risk": 0.0}
        
        min_ttc = min(ttc_values)
        avg_ttc = np.mean(ttc_values)
        
        # Collision risk: probability that TTC < safety threshold
        critical_ttc_threshold = self.config.get("critical_ttc_threshold", 5.0)  # seconds
        collision_risk = np.mean([1.0 if ttc < critical_ttc_threshold else 0.0 for ttc in ttc_values])
        
        return {
            "min_ttc": float(min_ttc),
            "avg_ttc": float(avg_ttc),
            "collision_risk": float(collision_risk),
            "num_interactions": len(ttc_values)
        }
    
    async def _calculate_pairwise_ttc(
        self, 
        pred_points: List[Tuple[float, float, float]], 
        other_traj: TrajectoryData
    ) -> Optional[float]:
        """Calculate TTC between predicted trajectory and another vehicle."""
        
        # Find the closest approach
        min_distance = float("inf")
        min_distance_time = None
        
        # Simple approach: check each predicted point
        for pred_time, pred_x, pred_y in pred_points:
            # Find corresponding position of other vehicle at this time
            other_x, other_y = self._interpolate_position(other_traj, pred_time)
            
            if other_x is not None and other_y is not None:
                distance = np.sqrt((pred_x - other_x)**2 + (pred_y - other_y)**2)
                
                if distance < min_distance:
                    min_distance = distance
                    min_distance_time = pred_time
        
        # If minimum distance is below collision threshold, calculate TTC
        if min_distance < self.collision_threshold and min_distance_time is not None:
            # TTC is the time to reach minimum distance
            current_time = pred_points[0][0]  # First timestamp
            ttc = min_distance_time - current_time
            return max(0.0, ttc)  # TTC cannot be negative
        
        return None  # No collision predicted
    
    def _interpolate_position(self, traj: TrajectoryData, target_time: float) -> Tuple[Optional[float], Optional[float]]:
        """Interpolate vehicle position at target time (simplified)."""
        # This is a simplified implementation
        # In practice, you'd have a trajectory with multiple points
        return traj.x, traj.y
    
    async def minimum_distance_metrics(
        self,
        prediction: PredictionResult,
        ground_truth: List[TrajectoryData]
    ) -> Dict[str, float]:
        """
        Calculate minimum distance metrics between prediction and other vehicles.
        
        Args:
            prediction: Predicted trajectory
            ground_truth: Ground truth trajectories of other vehicles
            
        Returns:
            Dictionary with minimum distance metrics
        """
        if not ground_truth:
            return {"min_distance": float("inf"), "avg_min_distance": float("inf"), "safety_violations": 0}
        
        min_distances = []
        safety_violations = 0
        
        # Calculate minimum distance to each other vehicle
        for other_vehicle in ground_truth:
            distances = []
            
            for i, (pred_x, pred_y) in enumerate(zip(prediction.x_coords, prediction.y_coords)):
                # Simple distance calculation (could be enhanced with vehicle dimensions)
                distance = np.sqrt((pred_x - other_vehicle.x)**2 + (pred_y - other_vehicle.y)**2)
                distances.append(distance)
            
            if distances:
                vehicle_min_distance = min(distances)
                min_distances.append(vehicle_min_distance)
                
                # Check for safety violations
                if vehicle_min_distance < self.safety_buffer:
                    safety_violations += 1
        
        if not min_distances:
            return {"min_distance": float("inf"), "avg_min_distance": float("inf"), "safety_violations": 0}
        
        return {
            "min_distance": float(min(min_distances)),
            "avg_min_distance": float(np.mean(min_distances)),
            "safety_violations": safety_violations,
            "total_interactions": len(min_distances)
        }
    
    async def lateral_error(
        self,
        prediction: PredictionResult,
        ground_truth_trajectory: List[TrajectoryData],
        reference_path: Optional[List[Tuple[float, float]]] = None
    ) -> Dict[str, float]:
        """
        Calculate lateral error with respect to reference path or ground truth.
        
        Args:
            prediction: Predicted trajectory
            ground_truth_trajectory: Ground truth trajectory
            reference_path: Optional reference path (e.g., road centerline)
            
        Returns:
            Dictionary with lateral error metrics
        """
        if reference_path is None:
            # Use ground truth as reference path
            reference_path = [(pt.x, pt.y) for pt in ground_truth_trajectory]
        
        if not reference_path:
            return {"mean_lateral_error": 0.0, "max_lateral_error": 0.0, "std_lateral_error": 0.0}
        
        lateral_errors = []
        
        for pred_x, pred_y in zip(prediction.x_coords, prediction.y_coords):
            # Find closest point on reference path
            distances = [
                np.sqrt((pred_x - ref_x)**2 + (pred_y - ref_y)**2)
                for ref_x, ref_y in reference_path
            ]
            
            min_lateral_error = min(distances)
            lateral_errors.append(min_lateral_error)
        
        if not lateral_errors:
            return {"mean_lateral_error": 0.0, "max_lateral_error": 0.0, "std_lateral_error": 0.0}
        
        return {
            "mean_lateral_error": float(np.mean(lateral_errors)),
            "max_lateral_error": float(np.max(lateral_errors)),
            "std_lateral_error": float(np.std(lateral_errors)),
            "median_lateral_error": float(np.median(lateral_errors))
        }


class TrajectoryMetrics:
    """
    Standard trajectory prediction metrics (RMSE, MAE, ADE, FDE).
    
    Implements common metrics used in trajectory prediction literature.
    """
    
    def __init__(self, config: DictConfig):
        self.config = config
    
    async def rmse(
        self, 
        prediction: PredictionResult, 
        ground_truth: List[TrajectoryData]
    ) -> Dict[str, float]:
        """Root Mean Square Error."""
        pred_x, pred_y, gt_x, gt_y = self._align_trajectories(prediction, ground_truth)
        
        if len(pred_x) == 0:
            return {"rmse": float("inf"), "rmse_x": float("inf"), "rmse_y": float("inf")}
        
        rmse_x = np.sqrt(np.mean((pred_x - gt_x)**2))
        rmse_y = np.sqrt(np.mean((pred_y - gt_y)**2))
        rmse_overall = np.sqrt(np.mean((pred_x - gt_x)**2 + (pred_y - gt_y)**2))
        
        return {
            "rmse": float(rmse_overall),
            "rmse_x": float(rmse_x),
            "rmse_y": float(rmse_y)
        }
    
    async def mae(
        self, 
        prediction: PredictionResult, 
        ground_truth: List[TrajectoryData]
    ) -> Dict[str, float]:
        """Mean Absolute Error."""
        pred_x, pred_y, gt_x, gt_y = self._align_trajectories(prediction, ground_truth)
        
        if len(pred_x) == 0:
            return {"mae": float("inf"), "mae_x": float("inf"), "mae_y": float("inf")}
        
        mae_x = np.mean(np.abs(pred_x - gt_x))
        mae_y = np.mean(np.abs(pred_y - gt_y))
        mae_overall = np.mean(np.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2))
        
        return {
            "mae": float(mae_overall),
            "mae_x": float(mae_x),
            "mae_y": float(mae_y)
        }
    
    async def ade(
        self, 
        prediction: PredictionResult, 
        ground_truth: List[TrajectoryData]
    ) -> float:
        """Average Displacement Error."""
        pred_x, pred_y, gt_x, gt_y = self._align_trajectories(prediction, ground_truth)
        
        if len(pred_x) == 0:
            return float("inf")
        
        displacements = np.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
        return float(np.mean(displacements))
    
    async def fde(
        self, 
        prediction: PredictionResult, 
        ground_truth: List[TrajectoryData]
    ) -> float:
        """Final Displacement Error."""
        pred_x, pred_y, gt_x, gt_y = self._align_trajectories(prediction, ground_truth)
        
        if len(pred_x) == 0:
            return float("inf")
        
        # Final displacement error is the error at the last time step
        final_displacement = np.sqrt((pred_x[-1] - gt_x[-1])**2 + (pred_y[-1] - gt_y[-1])**2)
        return float(final_displacement)
    
    async def trajectory_deviation(
        self,
        prediction: PredictionResult,
        ground_truth: List[TrajectoryData]
    ) -> Dict[str, float]:
        """
        Comprehensive trajectory deviation analysis.
        
        Returns multiple deviation metrics including path similarity.
        """
        pred_x, pred_y, gt_x, gt_y = self._align_trajectories(prediction, ground_truth)
        
        if len(pred_x) == 0:
            return {"path_deviation": float("inf"), "direction_error": float("inf")}
        
        # Path deviation (area between curves)
        path_deviation = self._calculate_path_deviation(pred_x, pred_y, gt_x, gt_y)
        
        # Direction error
        direction_error = self._calculate_direction_error(pred_x, pred_y, gt_x, gt_y)
        
        # Speed deviation
        if prediction.velocities is not None:
            gt_velocities = self._calculate_ground_truth_velocities(ground_truth)
            if len(gt_velocities) == len(prediction.velocities):
                speed_deviation = np.mean(np.abs(prediction.velocities - gt_velocities))
            else:
                speed_deviation = float("inf")
        else:
            speed_deviation = float("inf")
        
        return {
            "path_deviation": float(path_deviation),
            "direction_error": float(direction_error),
            "speed_deviation": float(speed_deviation)
        }
    
    def _align_trajectories(
        self, 
        prediction: PredictionResult, 
        ground_truth: List[TrajectoryData]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Align predicted and ground truth trajectories in time."""
        if not ground_truth:
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        # Sort ground truth by timestamp
        gt_sorted = sorted(ground_truth, key=lambda x: x.timestamp.timestamp())
        
        # Simple alignment: take first N points where N is min length
        min_length = min(len(prediction.x_coords), len(gt_sorted))
        
        if min_length == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        pred_x = prediction.x_coords[:min_length]
        pred_y = prediction.y_coords[:min_length]
        gt_x = np.array([pt.x for pt in gt_sorted[:min_length]])
        gt_y = np.array([pt.y for pt in gt_sorted[:min_length]])
        
        return pred_x, pred_y, gt_x, gt_y
    
    def _calculate_path_deviation(
        self, pred_x: np.ndarray, pred_y: np.ndarray, 
        gt_x: np.ndarray, gt_y: np.ndarray
    ) -> float:
        """Calculate path deviation using trapezoidal integration."""
        if len(pred_x) < 2:
            return 0.0
        
        # Calculate the area between the two curves
        distances = np.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
        
        # Simple approximation: mean distance
        return float(np.mean(distances))
    
    def _calculate_direction_error(
        self, pred_x: np.ndarray, pred_y: np.ndarray,
        gt_x: np.ndarray, gt_y: np.ndarray
    ) -> float:
        """Calculate direction error between predicted and ground truth paths."""
        if len(pred_x) < 2:
            return 0.0
        
        # Calculate direction vectors
        pred_directions = np.arctan2(np.diff(pred_y), np.diff(pred_x))
        gt_directions = np.arctan2(np.diff(gt_y), np.diff(gt_x))
        
        # Angular differences
        angle_diffs = np.abs(pred_directions - gt_directions)
        # Wrap to [0, π]
        angle_diffs = np.minimum(angle_diffs, 2*np.pi - angle_diffs)
        
        return float(np.mean(angle_diffs))
    
    def _calculate_ground_truth_velocities(self, ground_truth: List[TrajectoryData]) -> np.ndarray:
        """Calculate velocities from ground truth trajectory."""
        if len(ground_truth) < 2:
            return np.array([])
        
        velocities = []
        gt_sorted = sorted(ground_truth, key=lambda x: x.timestamp.timestamp())
        
        for i in range(1, len(gt_sorted)):
            dt = (gt_sorted[i].timestamp - gt_sorted[i-1].timestamp).total_seconds()
            if dt > 0:
                dx = gt_sorted[i].x - gt_sorted[i-1].x
                dy = gt_sorted[i].y - gt_sorted[i-1].y
                velocity = np.sqrt(dx**2 + dy**2) / dt
                velocities.append(velocity)
        
        return np.array(velocities)


class ProbabilisticMetrics:
    """
    Probabilistic evaluation metrics for uncertainty quantification.
    
    Evaluates the quality of uncertainty estimates in predictions.
    """
    
    def __init__(self, config: DictConfig):
        self.config = config
    
    async def calibration_error(
        self,
        predictions: List[PredictionResult],
        ground_truths: List[List[TrajectoryData]],
        confidence_levels: List[float] = [0.5, 0.8, 0.9, 0.95]
    ) -> Dict[str, float]:
        """
        Calculate calibration error for uncertainty estimates.
        
        A well-calibrated model should have prediction intervals that
        contain the true values at the specified confidence level.
        """
        if len(predictions) != len(ground_truths):
            raise ValueError("Number of predictions must match number of ground truths")
        
        calibration_errors = {}
        
        for confidence_level in confidence_levels:
            within_bounds = []
            
            for pred, gt in zip(predictions, ground_truths):
                if not pred.has_uncertainty or not gt:
                    continue
                
                # Check if ground truth falls within confidence bounds
                pred_x, pred_y, gt_x, gt_y = self._align_for_calibration(pred, gt)
                
                if len(pred_x) > 0:
                    # Calculate confidence bounds
                    z_score = stats.norm.ppf((1 + confidence_level) / 2)
                    
                    x_lower = pred_x - z_score * pred.x_std[:len(pred_x)]
                    x_upper = pred_x + z_score * pred.x_std[:len(pred_x)]
                    y_lower = pred_y - z_score * pred.y_std[:len(pred_y)]
                    y_upper = pred_y + z_score * pred.y_std[:len(pred_y)]
                    
                    # Check if ground truth is within bounds
                    x_within = np.logical_and(gt_x >= x_lower, gt_x <= x_upper)
                    y_within = np.logical_and(gt_y >= y_lower, gt_y <= y_upper)
                    both_within = np.logical_and(x_within, y_within)
                    
                    within_bounds.append(np.mean(both_within))
            
            if within_bounds:
                observed_coverage = np.mean(within_bounds)
                calibration_error = abs(observed_coverage - confidence_level)
                calibration_errors[f"calibration_error_{int(confidence_level*100)}"] = calibration_error
        
        # Overall calibration error
        if calibration_errors:
            calibration_errors["mean_calibration_error"] = np.mean(list(calibration_errors.values()))
        
        return calibration_errors
    
    async def sharpness(self, predictions: List[PredictionResult]) -> Dict[str, float]:
        """
        Calculate prediction sharpness (width of prediction intervals).
        
        Sharpness measures how narrow the prediction intervals are.
        Sharp predictions are more informative.
        """
        if not predictions or not any(pred.has_uncertainty for pred in predictions):
            return {"mean_sharpness": float("inf")}
        
        sharpnesses_x = []
        sharpnesses_y = []
        
        for pred in predictions:
            if pred.has_uncertainty:
                # Average standard deviation as sharpness measure
                mean_std_x = np.mean(pred.x_std)
                mean_std_y = np.mean(pred.y_std)
                
                sharpnesses_x.append(mean_std_x)
                sharpnesses_y.append(mean_std_y)
        
        return {
            "mean_sharpness_x": float(np.mean(sharpnesses_x)) if sharpnesses_x else float("inf"),
            "mean_sharpness_y": float(np.mean(sharpnesses_y)) if sharpnesses_y else float("inf"),
            "mean_sharpness": float(np.mean(sharpnesses_x + sharpnesses_y)) if (sharpnesses_x or sharpnesses_y) else float("inf")
        }
    
    async def prediction_interval_coverage(
        self,
        predictions: List[PredictionResult],
        ground_truths: List[List[TrajectoryData]],
        nominal_coverage: float = 0.95
    ) -> Dict[str, float]:
        """Calculate prediction interval coverage probability."""
        if len(predictions) != len(ground_truths):
            return {"coverage": 0.0}
        
        coverage_scores = []
        
        for pred, gt in zip(predictions, ground_truths):
            if not pred.has_uncertainty or not gt:
                continue
            
            pred_x, pred_y, gt_x, gt_y = self._align_for_calibration(pred, gt)
            
            if len(pred_x) > 0:
                # Calculate prediction intervals
                z_score = stats.norm.ppf((1 + nominal_coverage) / 2)
                
                x_within = np.abs(gt_x - pred_x) <= z_score * pred.x_std[:len(pred_x)]
                y_within = np.abs(gt_y - pred_y) <= z_score * pred.y_std[:len(pred_y)]
                
                # Both coordinates must be within bounds
                both_within = np.logical_and(x_within, y_within)
                coverage_scores.append(np.mean(both_within))
        
        if coverage_scores:
            mean_coverage = np.mean(coverage_scores)
            return {
                "coverage": float(mean_coverage),
                "coverage_deviation": float(abs(mean_coverage - nominal_coverage))
            }
        
        return {"coverage": 0.0, "coverage_deviation": float(nominal_coverage)}
    
    def _align_for_calibration(
        self, 
        prediction: PredictionResult, 
        ground_truth: List[TrajectoryData]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Align prediction and ground truth for calibration analysis."""
        if not ground_truth:
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        # Simple alignment - take minimum length
        gt_sorted = sorted(ground_truth, key=lambda x: x.timestamp.timestamp())
        min_length = min(len(prediction.x_coords), len(gt_sorted))
        
        if min_length == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        pred_x = prediction.x_coords[:min_length]
        pred_y = prediction.y_coords[:min_length]
        gt_x = np.array([pt.x for pt in gt_sorted[:min_length]])
        gt_y = np.array([pt.y for pt in gt_sorted[:min_length]])
        
        return pred_x, pred_y, gt_x, gt_y


class MetricCalculator:
    """
    Unified interface for calculating all trajectory prediction metrics.
    
    Coordinates the calculation of safety, trajectory, and probabilistic metrics.
    """
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.safety_metrics = SafetyMetrics(config)
        self.trajectory_metrics = TrajectoryMetrics(config)
        self.probabilistic_metrics = ProbabilisticMetrics(config)
    
    async def calculate_all_metrics(
        self,
        model_name: str,
        predictions: List[PredictionResult],
        ground_truths: List[List[TrajectoryData]],
        other_vehicles: Optional[List[List[TrajectoryData]]] = None
    ) -> Dict[str, Any]:
        """
        Calculate all available metrics for a set of predictions.
        
        Args:
            model_name: Name of the model being evaluated
            predictions: List of prediction results
            ground_truths: List of ground truth trajectories
            other_vehicles: Optional list of other vehicle trajectories for safety metrics
            
        Returns:
            Dictionary containing all calculated metrics
        """
        if len(predictions) != len(ground_truths):
            raise ValueError("Number of predictions must match number of ground truths")
        
        all_metrics = {
            "model_name": model_name,
            "num_predictions": len(predictions),
            "safety_metrics": {},
            "trajectory_metrics": {},
            "probabilistic_metrics": {}
        }
        
        # Calculate trajectory metrics for each prediction
        trajectory_results = []
        safety_results = []
        
        for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
            # Trajectory metrics
            rmse = await self.trajectory_metrics.rmse(pred, gt)
            mae = await self.trajectory_metrics.mae(pred, gt)
            ade = await self.trajectory_metrics.ade(pred, gt)
            fde = await self.trajectory_metrics.fde(pred, gt)
            deviation = await self.trajectory_metrics.trajectory_deviation(pred, gt)
            lateral = await self.safety_metrics.lateral_error(pred, gt)
            
            trajectory_result = {
                **rmse, **mae, "ade": ade, "fde": fde, 
                **deviation, **lateral
            }
            trajectory_results.append(trajectory_result)
            
            # Safety metrics (if other vehicles provided)
            if other_vehicles and i < len(other_vehicles):
                ttc = await self.safety_metrics.time_to_collision(pred, other_vehicles[i])
                min_dist = await self.safety_metrics.minimum_distance_metrics(pred, other_vehicles[i])
                
                safety_result = {**ttc, **min_dist}
                safety_results.append(safety_result)
        
        # Aggregate trajectory metrics
        if trajectory_results:
            all_metrics["trajectory_metrics"] = self._aggregate_metrics(trajectory_results)
        
        # Aggregate safety metrics
        if safety_results:
            all_metrics["safety_metrics"] = self._aggregate_metrics(safety_results)
        
        # Probabilistic metrics
        if any(pred.has_uncertainty for pred in predictions):
            calibration = await self.probabilistic_metrics.calibration_error(predictions, ground_truths)
            sharpness = await self.probabilistic_metrics.sharpness(predictions)
            coverage = await self.probabilistic_metrics.prediction_interval_coverage(predictions, ground_truths)
            
            all_metrics["probabilistic_metrics"] = {
                **calibration, **sharpness, **coverage
            }
        
        return all_metrics
    
    def _aggregate_metrics(self, metric_results: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Aggregate individual metric results into summary statistics."""
        if not metric_results:
            return {}
        
        aggregated = {}
        
        # Get all metric names
        all_keys = set()
        for result in metric_results:
            all_keys.update(result.keys())
        
        # Calculate statistics for each metric
        for key in all_keys:
            values = [result.get(key, float("inf")) for result in metric_results]
            # Filter out infinite values for statistics
            finite_values = [v for v in values if np.isfinite(v)]
            
            if finite_values:
                aggregated[key] = {
                    "mean": float(np.mean(finite_values)),
                    "std": float(np.std(finite_values)),
                    "median": float(np.median(finite_values)),
                    "min": float(np.min(finite_values)),
                    "max": float(np.max(finite_values)),
                    "count": len(finite_values)
                }
            else:
                aggregated[key] = {
                    "mean": float("inf"),
                    "std": 0.0,
                    "median": float("inf"),
                    "min": float("inf"),
                    "max": float("inf"),
                    "count": 0
                }
        
        return aggregated