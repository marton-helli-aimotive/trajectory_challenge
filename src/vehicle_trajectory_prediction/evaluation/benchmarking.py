"""Performance benchmarking for trajectory prediction models."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import time
import psutil
import os
from datetime import datetime
from tqdm import tqdm
import warnings

from ..core.models import Trajectory, TrajectoryPoint
from ..models.base import BaseTrajectoryPredictor
from ..core.config import ModelConfig

logger = logging.getLogger(__name__)


class ModelBenchmarker:
    """Comprehensive benchmarking for trajectory prediction models."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.benchmark_results = {}
        
    def benchmark_single_model(
        self, 
        model: BaseTrajectoryPredictor,
        test_trajectories: List[Trajectory],
        prediction_horizon: Optional[int] = None,
        prediction_frequency: Optional[float] = None,
        num_runs: int = 5,
        warmup_runs: int = 2
    ) -> Dict[str, Any]:
        """Benchmark a single model's performance."""
        
        if prediction_horizon is None:
            prediction_horizon = self.config.prediction_horizon
        if prediction_frequency is None:
            prediction_frequency = self.config.prediction_frequency
        
        logger.info(f"Benchmarking {model.model_name} with {len(test_trajectories)} trajectories")
        
        # Warmup runs
        logger.info("Performing warmup runs...")
        for _ in range(warmup_runs):
            for trajectory in test_trajectories[:min(5, len(test_trajectories))]:
                try:
                    model.predict(trajectory, prediction_horizon, prediction_frequency)
                except Exception as e:
                    logger.warning(f"Warmup failed: {e}")
        
        # Performance measurements
        inference_times = []
        memory_usage = []
        throughput_measurements = []
        success_rates = []
        
        for run in range(num_runs):
            logger.info(f"Benchmark run {run + 1}/{num_runs}")
            
            # Measure inference time
            run_times = []
            run_memory = []
            successful_predictions = 0
            
            start_time = time.time()
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            for trajectory in test_trajectories:
                try:
                    pred_start = time.time()
                    model.predict(trajectory, prediction_horizon, prediction_frequency)
                    pred_end = time.time()
                    
                    run_times.append(pred_end - pred_start)
                    successful_predictions += 1
                    
                except Exception as e:
                    logger.warning(f"Prediction failed: {e}")
                    continue
            
            end_time = time.time()
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Calculate metrics for this run
            if run_times:
                inference_times.append(np.mean(run_times))
                memory_usage.append(final_memory - initial_memory)
                throughput = successful_predictions / (end_time - start_time)
                throughput_measurements.append(throughput)
                success_rate = successful_predictions / len(test_trajectories)
                success_rates.append(success_rate)
        
        # Calculate summary statistics
        benchmark_results = {
            'model_name': model.model_name,
            'num_trajectories': len(test_trajectories),
            'num_runs': num_runs,
            'prediction_horizon': prediction_horizon,
            'prediction_frequency': prediction_frequency,
            'timestamp': datetime.now().isoformat()
        }
        
        if inference_times:
            benchmark_results['inference_time'] = {
                'mean': float(np.mean(inference_times)),
                'std': float(np.std(inference_times)),
                'min': float(np.min(inference_times)),
                'max': float(np.max(inference_times)),
                'median': float(np.median(inference_times)),
                'all_values': [float(t) for t in inference_times]
            }
        
        if memory_usage:
            benchmark_results['memory_usage'] = {
                'mean': float(np.mean(memory_usage)),
                'std': float(np.std(memory_usage)),
                'min': float(np.min(memory_usage)),
                'max': float(np.max(memory_usage)),
                'all_values': [float(m) for m in memory_usage]
            }
        
        if throughput_measurements:
            benchmark_results['throughput'] = {
                'mean': float(np.mean(throughput_measurements)),
                'std': float(np.std(throughput_measurements)),
                'min': float(np.min(throughput_measurements)),
                'max': float(np.max(throughput_measurements)),
                'all_values': [float(t) for t in throughput_measurements]
            }
        
        if success_rates:
            benchmark_results['success_rate'] = {
                'mean': float(np.mean(success_rates)),
                'std': float(np.std(success_rates)),
                'min': float(np.min(success_rates)),
                'max': float(np.max(success_rates)),
                'all_values': [float(s) for s in success_rates]
            }
        
        return benchmark_results
    
    def benchmark_multiple_models(
        self, 
        models: List[BaseTrajectoryPredictor],
        test_trajectories: List[Trajectory],
        prediction_horizon: Optional[int] = None,
        prediction_frequency: Optional[float] = None,
        num_runs: int = 5,
        warmup_runs: int = 2
    ) -> Dict[str, Any]:
        """Benchmark multiple models and compare their performance."""
        
        logger.info(f"Benchmarking {len(models)} models")
        
        all_results = {}
        
        for model in models:
            try:
                model_results = self.benchmark_single_model(
                    model, test_trajectories, prediction_horizon, 
                    prediction_frequency, num_runs, warmup_runs
                )
                all_results[model.model_name] = model_results
            except Exception as e:
                logger.error(f"Failed to benchmark {model.model_name}: {e}")
                all_results[model.model_name] = {'error': str(e)}
        
        # Create comparison summary
        comparison_summary = self._create_benchmark_comparison(all_results)
        
        return {
            'model_results': all_results,
            'comparison_summary': comparison_summary,
            'benchmark_config': {
                'num_trajectories': len(test_trajectories),
                'num_runs': num_runs,
                'warmup_runs': warmup_runs,
                'prediction_horizon': prediction_horizon,
                'prediction_frequency': prediction_frequency,
                'timestamp': datetime.now().isoformat()
            }
        }
    
    def _create_benchmark_comparison(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create comparison summary for benchmark results."""
        
        comparison = {
            'rankings': {},
            'performance_ratios': {},
            'statistical_comparison': {}
        }
        
        # Create rankings for each metric
        metrics = ['inference_time', 'memory_usage', 'throughput', 'success_rate']
        
        for metric in metrics:
            rankings = []
            for model_name, results in all_results.items():
                if 'error' not in results and metric in results:
                    mean_value = results[metric]['mean']
                    rankings.append({
                        'model': model_name,
                        'value': mean_value,
                        'std': results[metric]['std']
                    })
            
            if rankings:
                # Sort by value (lower is better for time/memory, higher is better for throughput/success)
                if metric in ['inference_time', 'memory_usage']:
                    rankings.sort(key=lambda x: x['value'])
                else:
                    rankings.sort(key=lambda x: x['value'], reverse=True)
                
                comparison['rankings'][metric] = rankings
        
        # Calculate performance ratios
        if 'inference_time' in comparison['rankings'] and len(comparison['rankings']['inference_time']) > 1:
            fastest_model = comparison['rankings']['inference_time'][0]
            performance_ratios = {}
            
            for ranking in comparison['rankings']['inference_time']:
                if ranking['model'] != fastest_model['model']:
                    ratio = ranking['value'] / fastest_model['value']
                    performance_ratios[ranking['model']] = {
                        'ratio_to_fastest': ratio,
                        'times_slower': ratio
                    }
            
            comparison['performance_ratios']['inference_time'] = performance_ratios
        
        return comparison
    
    def benchmark_scalability(
        self, 
        model: BaseTrajectoryPredictor,
        test_trajectories: List[Trajectory],
        trajectory_sizes: List[int] = None,
        prediction_horizon: Optional[int] = None,
        prediction_frequency: Optional[float] = None
    ) -> Dict[str, Any]:
        """Benchmark model scalability with different trajectory sizes."""
        
        if trajectory_sizes is None:
            trajectory_sizes = [10, 25, 50, 100, 200]
        
        if prediction_horizon is None:
            prediction_horizon = self.config.prediction_horizon
        if prediction_frequency is None:
            prediction_frequency = self.config.prediction_frequency
        
        logger.info(f"Benchmarking scalability for {model.model_name}")
        
        scalability_results = {
            'model_name': model.model_name,
            'trajectory_sizes': trajectory_sizes,
            'results': {}
        }
        
        for size in trajectory_sizes:
            if size > len(test_trajectories):
                logger.warning(f"Skipping size {size} (only {len(test_trajectories)} trajectories available)")
                continue
            
            # Sample trajectories
            sampled_trajectories = test_trajectories[:size]
            
            # Benchmark this size
            size_results = self.benchmark_single_model(
                model, sampled_trajectories, prediction_horizon, 
                prediction_frequency, num_runs=3, warmup_runs=1
            )
            
            scalability_results['results'][size] = size_results
        
        # Calculate scalability metrics
        scalability_metrics = self._calculate_scalability_metrics(scalability_results)
        scalability_results['scalability_metrics'] = scalability_metrics
        
        return scalability_results
    
    def _calculate_scalability_metrics(self, scalability_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate scalability metrics from benchmark results."""
        
        metrics = {}
        sizes = list(scalability_results['results'].keys())
        
        if len(sizes) < 2:
            return metrics
        
        # Calculate scaling factors for inference time
        inference_times = []
        for size in sizes:
            if 'inference_time' in scalability_results['results'][size]:
                inference_times.append(scalability_results['results'][size]['inference_time']['mean'])
        
        if len(inference_times) >= 2:
            # Calculate scaling factor (how much time increases with data size)
            scaling_factors = []
            for i in range(1, len(sizes)):
                size_ratio = sizes[i] / sizes[i-1]
                time_ratio = inference_times[i] / inference_times[i-1]
                scaling_factor = time_ratio / size_ratio
                scaling_factors.append(scaling_factor)
            
            metrics['inference_time_scaling'] = {
                'mean_scaling_factor': float(np.mean(scaling_factors)),
                'std_scaling_factor': float(np.std(scaling_factors)),
                'scaling_factors': [float(f) for f in scaling_factors],
                'is_linear': all(0.8 <= f <= 1.2 for f in scaling_factors),
                'is_sublinear': all(f < 0.8 for f in scaling_factors),
                'is_superlinear': all(f > 1.2 for f in scaling_factors)
            }
        
        return metrics
    
    def benchmark_memory_profiling(
        self, 
        model: BaseTrajectoryPredictor,
        test_trajectories: List[Trajectory],
        prediction_horizon: Optional[int] = None,
        prediction_frequency: Optional[float] = None
    ) -> Dict[str, Any]:
        """Detailed memory profiling for a model."""
        
        if prediction_horizon is None:
            prediction_horizon = self.config.prediction_horizon
        if prediction_frequency is None:
            prediction_frequency = self.config.prediction_frequency
        
        logger.info(f"Memory profiling {model.model_name}")
        
        process = psutil.Process(os.getpid())
        
        # Initial memory state
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        initial_virtual = process.memory_info().vms / 1024 / 1024  # MB
        
        memory_profile = {
            'model_name': model.model_name,
            'initial_memory_mb': float(initial_memory),
            'initial_virtual_mb': float(initial_virtual),
            'prediction_memory': [],
            'peak_memory': 0.0
        }
        
        # Profile memory during predictions
        for i, trajectory in enumerate(test_trajectories):
            try:
                # Memory before prediction
                memory_before = process.memory_info().rss / 1024 / 1024
                
                # Make prediction
                model.predict(trajectory, prediction_horizon, prediction_frequency)
                
                # Memory after prediction
                memory_after = process.memory_info().rss / 1024 / 1024
                memory_used = memory_after - memory_before
                
                memory_profile['prediction_memory'].append({
                    'trajectory_id': trajectory.vehicle_id,
                    'memory_before_mb': float(memory_before),
                    'memory_after_mb': float(memory_after),
                    'memory_used_mb': float(memory_used)
                })
                
                # Track peak memory
                memory_profile['peak_memory'] = max(memory_profile['peak_memory'], memory_after)
                
            except Exception as e:
                logger.warning(f"Memory profiling failed for trajectory {trajectory.vehicle_id}: {e}")
                continue
        
        # Calculate memory statistics
        if memory_profile['prediction_memory']:
            memory_used_values = [m['memory_used_mb'] for m in memory_profile['prediction_memory']]
            memory_profile['memory_statistics'] = {
                'mean_memory_per_prediction_mb': float(np.mean(memory_used_values)),
                'std_memory_per_prediction_mb': float(np.std(memory_used_values)),
                'max_memory_per_prediction_mb': float(np.max(memory_used_values)),
                'min_memory_per_prediction_mb': float(np.min(memory_used_values)),
                'total_memory_increase_mb': float(memory_profile['peak_memory'] - initial_memory)
            }
        
        return memory_profile
    
    def generate_benchmark_report(
        self, 
        benchmark_results: Dict[str, Any], 
        output_path: Optional[str] = None
    ) -> str:
        """Generate a comprehensive benchmark report."""
        
        report_lines = []
        
        # Header
        report_lines.append("=" * 80)
        report_lines.append("TRAJECTORY PREDICTION MODEL BENCHMARK REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Benchmark configuration
        if 'benchmark_config' in benchmark_results:
            config = benchmark_results['benchmark_config']
            report_lines.append("BENCHMARK CONFIGURATION")
            report_lines.append("-" * 40)
            report_lines.append(f"Number of Trajectories: {config['num_trajectories']}")
            report_lines.append(f"Number of Runs: {config['num_runs']}")
            report_lines.append(f"Warmup Runs: {config['warmup_runs']}")
            report_lines.append(f"Prediction Horizon: {config['prediction_horizon']}")
            report_lines.append(f"Prediction Frequency: {config['prediction_frequency']}")
            report_lines.append(f"Timestamp: {config['timestamp']}")
            report_lines.append("")
        
        # Model results
        if 'model_results' in benchmark_results:
            report_lines.append("MODEL PERFORMANCE RESULTS")
            report_lines.append("-" * 40)
            
            for model_name, results in benchmark_results['model_results'].items():
                if 'error' in results:
                    report_lines.append(f"{model_name}: ERROR - {results['error']}")
                    continue
                
                report_lines.append(f"{model_name}:")
                
                if 'inference_time' in results:
                    it = results['inference_time']
                    report_lines.append(f"  Inference Time: {it['mean']:.4f} ± {it['std']:.4f} seconds")
                
                if 'throughput' in results:
                    tp = results['throughput']
                    report_lines.append(f"  Throughput: {tp['mean']:.2f} ± {tp['std']:.2f} trajectories/second")
                
                if 'memory_usage' in results:
                    mem = results['memory_usage']
                    report_lines.append(f"  Memory Usage: {mem['mean']:.2f} ± {mem['std']:.2f} MB")
                
                if 'success_rate' in results:
                    sr = results['success_rate']
                    report_lines.append(f"  Success Rate: {sr['mean']:.2%} ± {sr['std']:.2%}")
                
                report_lines.append("")
        
        # Rankings
        if 'comparison_summary' in benchmark_results:
            comparison = benchmark_results['comparison_summary']
            
            if 'rankings' in comparison:
                report_lines.append("PERFORMANCE RANKINGS")
                report_lines.append("-" * 40)
                
                for metric, rankings in comparison['rankings'].items():
                    report_lines.append(f"{metric.replace('_', ' ').title()}:")
                    for i, ranking in enumerate(rankings):
                        report_lines.append(f"  {i+1}. {ranking['model']}: {ranking['value']:.4f} ± {ranking['std']:.4f}")
                    report_lines.append("")
        
        report = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Benchmark report saved to {output_path}")
        
        return report