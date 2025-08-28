"""Basic model evaluation framework for trajectory prediction."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

from .base import BaseTrajectoryPredictor, PredictionResult
from ..core.models import Trajectory, TrajectoryPoint
from ..core.config import ModelConfig

logger = logging.getLogger(__name__)


class TrajectoryEvaluator:
    """Evaluator for trajectory prediction models."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.metrics = {}
        
    def calculate_rmse(self, true_trajectory: Trajectory, predicted_result: PredictionResult) -> float:
        """Calculate Root Mean Square Error."""
        if len(predicted_result.predicted_points) == 0:
            return np.inf
        
        # Get true positions at prediction timestamps
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
        
        # Calculate RMSE
        predicted_x = predicted_result.x_positions
        predicted_y = predicted_result.y_positions
        
        # Ensure same length
        min_length = min(len(true_x), len(predicted_x))
        true_x = true_x[:min_length]
        true_y = true_y[:min_length]
        predicted_x = predicted_x[:min_length]
        predicted_y = predicted_y[:min_length]
        
        # Calculate Euclidean distance errors
        errors = np.sqrt((np.array(true_x) - predicted_x)**2 + (np.array(true_y) - predicted_y)**2)
        
        return np.sqrt(np.mean(errors**2))
    
    def calculate_ade(self, true_trajectory: Trajectory, predicted_result: PredictionResult) -> float:
        """Calculate Average Displacement Error."""
        if len(predicted_result.predicted_points) == 0:
            return np.inf
        
        # Get true positions at prediction timestamps
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
        
        # Calculate ADE
        predicted_x = predicted_result.x_positions
        predicted_y = predicted_result.y_positions
        
        # Ensure same length
        min_length = min(len(true_x), len(predicted_x))
        true_x = true_x[:min_length]
        true_y = true_y[:min_length]
        predicted_x = predicted_x[:min_length]
        predicted_y = predicted_y[:min_length]
        
        # Calculate Euclidean distance errors
        errors = np.sqrt((np.array(true_x) - predicted_x)**2 + (np.array(true_y) - predicted_y)**2)
        
        return np.mean(errors)
    
    def calculate_fde(self, true_trajectory: Trajectory, predicted_result: PredictionResult) -> float:
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
    
    def calculate_min_distance(self, true_trajectory: Trajectory, predicted_result: PredictionResult) -> float:
        """Calculate minimum distance between true and predicted trajectories."""
        if len(predicted_result.predicted_points) == 0:
            return np.inf
        
        # Get true positions at prediction timestamps
        true_points = []
        for timestamp in predicted_result.timestamps:
            true_point = true_trajectory.get_point_at_time(timestamp)
            if true_point is not None:
                true_points.append((true_point.x, true_point.y))
            else:
                # If no true point, use last known position
                true_points.append((true_trajectory.points[-1].x, true_trajectory.points[-1].y))
        
        # Get predicted positions
        predicted_points = list(zip(predicted_result.x_positions, predicted_result.y_positions))
        
        # Calculate minimum distance
        min_dist = np.inf
        for true_point in true_points:
            for pred_point in predicted_points:
                dist = np.sqrt((true_point[0] - pred_point[0])**2 + (true_point[1] - pred_point[1])**2)
                min_dist = min(min_dist, dist)
        
        return min_dist
    
    def calculate_ttc(self, true_trajectory: Trajectory, predicted_result: PredictionResult) -> float:
        """Calculate Time-to-Collision (simplified version)."""
        if len(predicted_result.predicted_points) < 2:
            return np.inf
        
        # Get true positions at prediction timestamps
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
        
        # Calculate relative velocity and distance
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
        
        # Calculate distances and velocities
        distances = np.sqrt((np.array(true_x) - predicted_x)**2 + (np.array(true_y) - predicted_y)**2)
        
        # Calculate relative velocity (simplified)
        if len(distances) >= 2:
            relative_velocity = np.abs(distances[1:] - distances[:-1]) / self.config.prediction_frequency
            
            # Find minimum TTC (avoid division by zero)
            ttc_values = []
            for i in range(len(relative_velocity)):
                if relative_velocity[i] > 1e-6:  # Avoid division by zero
                    ttc = distances[i] / relative_velocity[i]
                    if ttc > 0:  # Only positive TTC
                        ttc_values.append(ttc)
            
            if ttc_values:
                return min(ttc_values)
        
        return np.inf
    
    def calculate_lateral_error(self, true_trajectory: Trajectory, predicted_result: PredictionResult) -> float:
        """Calculate lateral error (perpendicular distance from true trajectory)."""
        if len(predicted_result.predicted_points) == 0:
            return np.inf
        
        # Get true positions at prediction timestamps
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
        
        # Calculate lateral errors
        predicted_x = predicted_result.x_positions
        predicted_y = predicted_result.y_positions
        
        # Ensure same length
        min_length = min(len(true_x), len(predicted_x))
        true_x = true_x[:min_length]
        true_y = true_y[:min_length]
        predicted_x = predicted_x[:min_length]
        predicted_y = predicted_y[:min_length]
        
        # Calculate lateral errors (simplified as perpendicular distance)
        lateral_errors = []
        for i in range(1, len(true_x)):
            # Calculate trajectory direction
            dx = true_x[i] - true_x[i-1]
            dy = true_y[i] - true_y[i-1]
            
            if dx == 0 and dy == 0:
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
    
    def evaluate_prediction(
        self, 
        true_trajectory: Trajectory, 
        predicted_result: PredictionResult
    ) -> Dict[str, float]:
        """Evaluate a single prediction."""
        metrics = {}
        
        # Calculate all metrics
        metrics['rmse'] = self.calculate_rmse(true_trajectory, predicted_result)
        metrics['ade'] = self.calculate_ade(true_trajectory, predicted_result)
        metrics['fde'] = self.calculate_fde(true_trajectory, predicted_result)
        metrics['min_distance'] = self.calculate_min_distance(true_trajectory, predicted_result)
        metrics['ttc'] = self.calculate_ttc(true_trajectory, predicted_result)
        metrics['lateral_error'] = self.calculate_lateral_error(true_trajectory, predicted_result)
        
        return metrics
    
    def evaluate_model(
        self, 
        model: BaseTrajectoryPredictor,
        test_trajectories: List[Trajectory],
        prediction_horizon: Optional[int] = None,
        prediction_frequency: Optional[float] = None
    ) -> Dict[str, Any]:
        """Evaluate a model on test trajectories."""
        logger.info(f"Evaluating {model.model_name} on {len(test_trajectories)} test trajectories")
        
        all_metrics = []
        failed_predictions = 0
        
        for i, trajectory in enumerate(test_trajectories):
            try:
                # Make prediction
                prediction = model.predict(trajectory, prediction_horizon, prediction_frequency)
                
                # Calculate metrics
                metrics = self.evaluate_prediction(trajectory, prediction)
                metrics['trajectory_id'] = trajectory.vehicle_id
                metrics['trajectory_length'] = trajectory.length
                all_metrics.append(metrics)
                
            except Exception as e:
                logger.warning(f"Failed to evaluate trajectory {trajectory.vehicle_id}: {e}")
                failed_predictions += 1
                continue
        
        if not all_metrics:
            raise ValueError("No successful predictions for evaluation")
        
        # Calculate summary statistics
        metrics_df = pd.DataFrame(all_metrics)
        summary = {}
        
        for metric in ['rmse', 'ade', 'fde', 'min_distance', 'ttc', 'lateral_error']:
            if metric in metrics_df.columns:
                values = metrics_df[metric].replace([np.inf, -np.inf], np.nan).dropna()
                if len(values) > 0:
                    summary[f'{metric}_mean'] = float(values.mean())
                    summary[f'{metric}_std'] = float(values.std())
                    summary[f'{metric}_median'] = float(values.median())
                    summary[f'{metric}_min'] = float(values.min())
                    summary[f'{metric}_max'] = float(values.max())
                else:
                    summary[f'{metric}_mean'] = np.nan
                    summary[f'{metric}_std'] = np.nan
                    summary[f'{metric}_median'] = np.nan
                    summary[f'{metric}_min'] = np.nan
                    summary[f'{metric}_max'] = np.nan
        
        summary['total_trajectories'] = len(test_trajectories)
        summary['successful_predictions'] = len(all_metrics)
        summary['failed_predictions'] = failed_predictions
        summary['success_rate'] = len(all_metrics) / len(test_trajectories)
        
        logger.info(f"Evaluation completed. Success rate: {summary['success_rate']:.2%}")
        
        return {
            'summary': summary,
            'detailed_metrics': all_metrics,
            'metrics_dataframe': metrics_df
        }
    
    def compare_models(
        self, 
        models: List[BaseTrajectoryPredictor],
        test_trajectories: List[Trajectory],
        prediction_horizon: Optional[int] = None,
        prediction_frequency: Optional[float] = None
    ) -> Dict[str, Any]:
        """Compare multiple models on the same test set."""
        logger.info(f"Comparing {len(models)} models on {len(test_trajectories)} test trajectories")
        
        results = {}
        
        for model in models:
            try:
                model_results = self.evaluate_model(
                    model, test_trajectories, prediction_horizon, prediction_frequency
                )
                results[model.model_name] = model_results
            except Exception as e:
                logger.error(f"Failed to evaluate model {model.model_name}: {e}")
                results[model.model_name] = {'error': str(e)}
        
        # Create comparison summary
        comparison_summary = {}
        for model_name, result in results.items():
            if 'summary' in result:
                comparison_summary[model_name] = result['summary']
        
        return {
            'model_results': results,
            'comparison_summary': comparison_summary
        }
    
    def plot_comparison(
        self, 
        comparison_results: Dict[str, Any],
        metrics: List[str] = None,
        save_path: Optional[str] = None
    ):
        """Plot comparison of model performances."""
        if metrics is None:
            metrics = ['rmse', 'ade', 'fde']
        
        comparison_summary = comparison_results['comparison_summary']
        
        if not comparison_summary:
            logger.warning("No comparison data to plot")
            return
        
        # Prepare data for plotting
        plot_data = []
        for model_name, summary in comparison_summary.items():
            for metric in metrics:
                mean_key = f'{metric}_mean'
                if mean_key in summary and not np.isnan(summary[mean_key]):
                    plot_data.append({
                        'Model': model_name,
                        'Metric': metric.upper(),
                        'Value': summary[mean_key]
                    })
        
        if not plot_data:
            logger.warning("No valid data for plotting")
            return
        
        # Create plot
        df = pd.DataFrame(plot_data)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x='Metric', y='Value', hue='Model')
        plt.title('Model Performance Comparison')
        plt.ylabel('Error Value')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Comparison plot saved to {save_path}")
        
        plt.show()