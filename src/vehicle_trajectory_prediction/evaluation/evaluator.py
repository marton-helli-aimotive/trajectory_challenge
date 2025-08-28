"""Comprehensive evaluator for trajectory prediction models."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
import logging
import warnings
from tqdm import tqdm

from ..core.models import Trajectory, TrajectoryPoint
from ..models.base import BaseTrajectoryPredictor, PredictionResult
from ..core.config import ModelConfig
from .metrics import TrajectoryMetrics, SafetyMetrics, StatisticalMetrics, PerformanceMetrics

logger = logging.getLogger(__name__)


class ComprehensiveEvaluator:
    """Comprehensive evaluator for trajectory prediction models."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.metrics_history = []
        
    def evaluate_prediction(
        self, 
        true_trajectory: Trajectory, 
        predicted_result: PredictionResult,
        include_safety: bool = True,
        include_performance: bool = False
    ) -> Dict[str, float]:
        """Evaluate a single prediction with comprehensive metrics."""
        metrics = {}
        
        # Core trajectory metrics
        metrics['rmse'] = TrajectoryMetrics.calculate_rmse(true_trajectory, predicted_result)
        metrics['ade'] = TrajectoryMetrics.calculate_ade(true_trajectory, predicted_result)
        metrics['fde'] = TrajectoryMetrics.calculate_fde(true_trajectory, predicted_result)
        metrics['mae'] = TrajectoryMetrics.calculate_mae(true_trajectory, predicted_result)
        metrics['trajectory_similarity'] = TrajectoryMetrics.calculate_trajectory_similarity(
            true_trajectory, predicted_result
        )
        
        # Safety metrics
        if include_safety:
            metrics['min_distance'] = SafetyMetrics.calculate_min_distance(true_trajectory, predicted_result)
            metrics['ttc'] = SafetyMetrics.calculate_ttc(true_trajectory, predicted_result)
            metrics['lateral_error'] = SafetyMetrics.calculate_lateral_error(true_trajectory, predicted_result)
            metrics['risk_score'] = SafetyMetrics.calculate_risk_score(true_trajectory, predicted_result)
        
        return metrics
    
    def evaluate_model(
        self, 
        model: BaseTrajectoryPredictor,
        test_trajectories: List[Trajectory],
        prediction_horizon: Optional[int] = None,
        prediction_frequency: Optional[float] = None,
        include_safety: bool = True,
        include_performance: bool = True,
        include_statistical: bool = True,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Evaluate a model on test trajectories with comprehensive analysis."""
        logger.info(f"Evaluating {model.model_name} on {len(test_trajectories)} test trajectories")
        
        if prediction_horizon is None:
            prediction_horizon = self.config.prediction_horizon
        if prediction_frequency is None:
            prediction_frequency = self.config.prediction_frequency
        
        all_metrics = []
        failed_predictions = 0
        
        # Evaluate predictions
        iterator = tqdm(test_trajectories, desc=f"Evaluating {model.model_name}") if verbose else test_trajectories
        
        for trajectory in iterator:
            try:
                # Make prediction
                prediction = model.predict(trajectory, prediction_horizon, prediction_frequency)
                
                # Calculate metrics
                metrics = self.evaluate_prediction(
                    trajectory, prediction, include_safety, include_performance
                )
                metrics['trajectory_id'] = trajectory.vehicle_id
                metrics['trajectory_length'] = trajectory.length
                all_metrics.append(metrics)
                
            except Exception as e:
                logger.warning(f"Failed to evaluate trajectory {trajectory.vehicle_id}: {e}")
                failed_predictions += 1
                continue
        
        if not all_metrics:
            raise ValueError("No successful predictions for evaluation")
        
        # Create results dictionary
        results = {
            'model_name': model.model_name,
            'total_trajectories': len(test_trajectories),
            'successful_predictions': len(all_metrics),
            'failed_predictions': failed_predictions,
            'success_rate': len(all_metrics) / len(test_trajectories),
            'detailed_metrics': all_metrics
        }
        
        # Calculate summary statistics
        metrics_df = pd.DataFrame(all_metrics)
        summary = self._calculate_summary_statistics(metrics_df, include_statistical)
        results['summary'] = summary
        results['metrics_dataframe'] = metrics_df
        
        # Performance metrics
        if include_performance:
            performance_metrics = self._calculate_performance_metrics(
                model, test_trajectories[:min(10, len(test_trajectories))], prediction_horizon
            )
            results['performance_metrics'] = performance_metrics
        
        logger.info(f"Evaluation completed. Success rate: {results['success_rate']:.2%}")
        
        return results
    
    def compare_models(
        self, 
        models: List[BaseTrajectoryPredictor],
        test_trajectories: List[Trajectory],
        prediction_horizon: Optional[int] = None,
        prediction_frequency: Optional[float] = None,
        include_safety: bool = True,
        include_performance: bool = True,
        include_statistical: bool = True,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Compare multiple models on the same test set."""
        logger.info(f"Comparing {len(models)} models on {len(test_trajectories)} test trajectories")
        
        results = {}
        
        for model in models:
            try:
                model_results = self.evaluate_model(
                    model, test_trajectories, prediction_horizon, prediction_frequency,
                    include_safety, include_performance, include_statistical, verbose
                )
                results[model.model_name] = model_results
            except Exception as e:
                logger.error(f"Failed to evaluate model {model.model_name}: {e}")
                results[model.model_name] = {'error': str(e)}
        
        # Create comparison summary
        comparison_summary = self._create_comparison_summary(results)
        
        # Statistical significance testing
        if include_statistical and len(models) > 1:
            significance_tests = self._perform_significance_tests(results)
            comparison_summary['significance_tests'] = significance_tests
        
        return {
            'model_results': results,
            'comparison_summary': comparison_summary
        }
    
    def _calculate_summary_statistics(self, metrics_df: pd.DataFrame, include_statistical: bool = True) -> Dict[str, Any]:
        """Calculate comprehensive summary statistics."""
        summary = {}
        
        # Core metrics
        core_metrics = ['rmse', 'ade', 'fde', 'mae', 'trajectory_similarity']
        safety_metrics = ['min_distance', 'ttc', 'lateral_error', 'risk_score']
        
        all_metrics = core_metrics + safety_metrics
        
        for metric in all_metrics:
            if metric in metrics_df.columns:
                values = metrics_df[metric].replace([np.inf, -np.inf], np.nan).dropna()
                if len(values) > 0:
                    # Basic statistics
                    summary[f'{metric}_mean'] = float(values.mean())
                    summary[f'{metric}_std'] = float(values.std())
                    summary[f'{metric}_median'] = float(values.median())
                    summary[f'{metric}_min'] = float(values.min())
                    summary[f'{metric}_max'] = float(values.max())
                    
                    # Statistical metrics
                    if include_statistical:
                        # Confidence intervals
                        ci_lower, ci_upper = StatisticalMetrics.calculate_confidence_interval(values.tolist())
                        summary[f'{metric}_ci_lower'] = float(ci_lower) if not np.isnan(ci_lower) else np.nan
                        summary[f'{metric}_ci_upper'] = float(ci_upper) if not np.isnan(ci_upper) else np.nan
                        
                        # Percentiles
                        percentiles = StatisticalMetrics.calculate_percentiles(values.tolist())
                        for p_key, p_value in percentiles.items():
                            summary[f'{metric}_{p_key}'] = p_value
                        
                        # Distribution statistics
                        dist_stats = StatisticalMetrics.calculate_distribution_stats(values.tolist())
                        for stat_key, stat_value in dist_stats.items():
                            summary[f'{metric}_{stat_key}'] = stat_value
                        
                        # Outlier rate
                        outlier_rate = StatisticalMetrics.calculate_outlier_rate(values.tolist())
                        summary[f'{metric}_outlier_rate'] = float(outlier_rate)
                else:
                    # Set NaN for all statistics if no valid values
                    summary[f'{metric}_mean'] = np.nan
                    summary[f'{metric}_std'] = np.nan
                    summary[f'{metric}_median'] = np.nan
                    summary[f'{metric}_min'] = np.nan
                    summary[f'{metric}_max'] = np.nan
                    if include_statistical:
                        summary[f'{metric}_ci_lower'] = np.nan
                        summary[f'{metric}_ci_upper'] = np.nan
                        summary[f'{metric}_outlier_rate'] = np.nan
        
        return summary
    
    def _calculate_performance_metrics(self, model: BaseTrajectoryPredictor, 
                                     trajectories: List[Trajectory], 
                                     prediction_horizon: int) -> Dict[str, Any]:
        """Calculate performance metrics for a model."""
        performance_metrics = {}
        
        if trajectories:
            # Measure inference time
            inference_time = PerformanceMetrics.measure_inference_time(
                model, trajectories[0], prediction_horizon
            )
            performance_metrics['inference_time'] = inference_time
            
            # Measure memory usage
            memory_usage = PerformanceMetrics.measure_memory_usage(
                model, trajectories[0], prediction_horizon
            )
            performance_metrics['memory_usage'] = memory_usage
            
            # Calculate throughput
            throughput = PerformanceMetrics.calculate_throughput(
                model, trajectories, prediction_horizon
            )
            performance_metrics['throughput'] = throughput
        
        return performance_metrics
    
    def _create_comparison_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create comparison summary across models."""
        comparison_summary = {}
        
        for model_name, result in results.items():
            if 'summary' in result:
                comparison_summary[model_name] = result['summary']
        
        # Create ranking tables
        ranking_tables = {}
        if comparison_summary:
            metrics_to_rank = ['rmse', 'ade', 'fde', 'mae', 'risk_score']
            
            for metric in metrics_to_rank:
                ranking_data = []
                for model_name, summary in comparison_summary.items():
                    mean_key = f'{metric}_mean'
                    if mean_key in summary and not np.isnan(summary[mean_key]):
                        ranking_data.append({
                            'model': model_name,
                            'value': summary[mean_key]
                        })
                
                if ranking_data:
                    # Sort by value (lower is better for error metrics)
                    ranking_data.sort(key=lambda x: x['value'])
                    ranking_tables[metric] = ranking_data
        
        comparison_summary['rankings'] = ranking_tables
        
        return comparison_summary
    
    def _perform_significance_tests(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform statistical significance tests between models."""
        from scipy import stats
        
        significance_tests = {}
        
        # Get all model names with successful results
        model_names = [name for name, result in results.items() if 'detailed_metrics' in result]
        
        if len(model_names) < 2:
            return significance_tests
        
        # Compare each pair of models
        for i, model1 in enumerate(model_names):
            for model2 in model_names[i+1:]:
                comparison_key = f"{model1}_vs_{model2}"
                
                # Extract metrics for comparison
                metrics1 = pd.DataFrame(results[model1]['detailed_metrics'])
                metrics2 = pd.DataFrame(results[model2]['detailed_metrics'])
                
                significance_tests[comparison_key] = {}
                
                # Test each metric
                for metric in ['rmse', 'ade', 'fde', 'mae']:
                    if metric in metrics1.columns and metric in metrics2.columns:
                        values1 = metrics1[metric].replace([np.inf, -np.inf], np.nan).dropna()
                        values2 = metrics2[metric].replace([np.inf, -np.inf], np.nan).dropna()
                        
                        if len(values1) > 0 and len(values2) > 0:
                            try:
                                # Perform t-test
                                t_stat, p_value = stats.ttest_ind(values1, values2)
                                
                                significance_tests[comparison_key][metric] = {
                                    't_statistic': float(t_stat),
                                    'p_value': float(p_value),
                                    'significant': p_value < 0.05,
                                    'model1_mean': float(values1.mean()),
                                    'model2_mean': float(values2.mean()),
                                    'model1_std': float(values1.std()),
                                    'model2_std': float(values2.std())
                                }
                            except Exception as e:
                                logger.warning(f"Failed to perform t-test for {metric}: {e}")
                                significance_tests[comparison_key][metric] = {'error': str(e)}
        
        return significance_tests
    
    def generate_evaluation_report(self, evaluation_results: Dict[str, Any], 
                                 output_path: Optional[str] = None) -> str:
        """Generate a comprehensive evaluation report."""
        report_lines = []
        
        # Header
        report_lines.append("=" * 80)
        report_lines.append("TRAJECTORY PREDICTION MODEL EVALUATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Model information
        if 'model_name' in evaluation_results:
            report_lines.append(f"Model: {evaluation_results['model_name']}")
            report_lines.append(f"Total Trajectories: {evaluation_results['total_trajectories']}")
            report_lines.append(f"Successful Predictions: {evaluation_results['successful_predictions']}")
            report_lines.append(f"Success Rate: {evaluation_results['success_rate']:.2%}")
            report_lines.append("")
        
        # Summary statistics
        if 'summary' in evaluation_results:
            report_lines.append("SUMMARY STATISTICS")
            report_lines.append("-" * 40)
            
            summary = evaluation_results['summary']
            metrics = ['rmse', 'ade', 'fde', 'mae', 'risk_score']
            
            for metric in metrics:
                mean_key = f'{metric}_mean'
                if mean_key in summary and not np.isnan(summary[mean_key]):
                    report_lines.append(f"{metric.upper()}:")
                    report_lines.append(f"  Mean: {summary[mean_key]:.4f}")
                    report_lines.append(f"  Std:  {summary[f'{metric}_std']:.4f}")
                    report_lines.append(f"  Median: {summary[f'{metric}_median']:.4f}")
                    if f'{metric}_ci_lower' in summary:
                        report_lines.append(f"  95% CI: [{summary[f'{metric}_ci_lower']:.4f}, {summary[f'{metric}_ci_upper']:.4f}]")
                    report_lines.append("")
        
        # Performance metrics
        if 'performance_metrics' in evaluation_results:
            report_lines.append("PERFORMANCE METRICS")
            report_lines.append("-" * 40)
            
            perf = evaluation_results['performance_metrics']
            if 'inference_time' in perf:
                it = perf['inference_time']
                report_lines.append(f"Inference Time: {it['mean_time']:.4f} ± {it['std_time']:.4f} seconds")
            
            if 'throughput' in perf:
                tp = perf['throughput']
                report_lines.append(f"Throughput: {tp['throughput_traj_per_sec']:.2f} trajectories/second")
            
            report_lines.append("")
        
        report = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Evaluation report saved to {output_path}")
        
        return report