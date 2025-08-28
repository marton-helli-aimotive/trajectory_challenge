"""Confidence interval estimation for trajectory prediction metrics."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from scipy import stats
from scipy.stats import bootstrap
import warnings

logger = logging.getLogger(__name__)


class ConfidenceIntervalEstimator:
    """Estimator for confidence intervals of trajectory prediction metrics."""
    
    def __init__(self, confidence_level: float = 0.95, n_bootstrap: int = 1000):
        self.confidence_level = confidence_level
        self.n_bootstrap = n_bootstrap
        self.alpha = 1 - confidence_level
    
    def calculate_parametric_ci(
        self, 
        values: List[float], 
        method: str = "t_distribution"
    ) -> Dict[str, float]:
        """Calculate parametric confidence intervals."""
        
        # Remove infinite and NaN values
        finite_values = [v for v in values if not np.isinf(v) and not np.isnan(v)]
        
        if len(finite_values) == 0:
            return {
                'lower': np.nan,
                'upper': np.nan,
                'mean': np.nan,
                'std': np.nan,
                'method': method,
                'n_samples': 0
            }
        
        mean_val = np.mean(finite_values)
        std_val = np.std(finite_values, ddof=1)
        n_samples = len(finite_values)
        
        if method == "t_distribution":
            # Student's t-distribution (recommended for small samples)
            if std_val == 0:
                ci_lower = ci_upper = mean_val
            else:
                t_value = stats.t.ppf(1 - self.alpha / 2, n_samples - 1)
                margin_of_error = t_value * std_val / np.sqrt(n_samples)
                ci_lower = mean_val - margin_of_error
                ci_upper = mean_val + margin_of_error
        
        elif method == "normal":
            # Normal distribution (for large samples)
            if std_val == 0:
                ci_lower = ci_upper = mean_val
            else:
                z_value = stats.norm.ppf(1 - self.alpha / 2)
                margin_of_error = z_value * std_val / np.sqrt(n_samples)
                ci_lower = mean_val - margin_of_error
                ci_upper = mean_val + margin_of_error
        
        else:
            raise ValueError(f"Unknown parametric method: {method}")
        
        return {
            'lower': float(ci_lower),
            'upper': float(ci_upper),
            'mean': float(mean_val),
            'std': float(std_val),
            'method': method,
            'n_samples': n_samples,
            'confidence_level': self.confidence_level
        }
    
    def calculate_bootstrap_ci(
        self, 
        values: List[float], 
        method: str = "percentile"
    ) -> Dict[str, float]:
        """Calculate bootstrap confidence intervals."""
        
        # Remove infinite and NaN values
        finite_values = [v for v in values if not np.isinf(v) and not np.isnan(v)]
        
        if len(finite_values) == 0:
            return {
                'lower': np.nan,
                'upper': np.nan,
                'mean': np.nan,
                'method': method,
                'n_samples': 0
            }
        
        if method == "percentile":
            # Percentile bootstrap
            bootstrap_means = []
            
            for _ in range(self.n_bootstrap):
                # Sample with replacement
                bootstrap_sample = np.random.choice(finite_values, size=len(finite_values), replace=True)
                bootstrap_means.append(np.mean(bootstrap_sample))
            
            # Calculate percentiles
            lower_percentile = (self.alpha / 2) * 100
            upper_percentile = (1 - self.alpha / 2) * 100
            
            ci_lower = np.percentile(bootstrap_means, lower_percentile)
            ci_upper = np.percentile(bootstrap_means, upper_percentile)
            
        elif method == "bca":
            # Bias-corrected and accelerated bootstrap
            try:
                # Use scipy's bootstrap function
                bootstrap_result = bootstrap(
                    (finite_values,), 
                    np.mean, 
                    n_resamples=self.n_bootstrap,
                    confidence_level=self.confidence_level,
                    method='BCa'
                )
                ci_lower = bootstrap_result.confidence_interval.low
                ci_upper = bootstrap_result.confidence_interval.high
            except Exception as e:
                logger.warning(f"BCa bootstrap failed, falling back to percentile: {e}")
                return self.calculate_bootstrap_ci(values, "percentile")
        
        else:
            raise ValueError(f"Unknown bootstrap method: {method}")
        
        return {
            'lower': float(ci_lower),
            'upper': float(ci_upper),
            'mean': float(np.mean(finite_values)),
            'method': f"bootstrap_{method}",
            'n_samples': len(finite_values),
            'n_bootstrap': self.n_bootstrap,
            'confidence_level': self.confidence_level
        }
    
    def calculate_ci_for_metric(
        self, 
        values: List[float], 
        methods: List[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """Calculate confidence intervals using multiple methods."""
        
        if methods is None:
            methods = ["t_distribution", "bootstrap_percentile"]
        
        results = {}
        
        for method in methods:
            if method.startswith("bootstrap_"):
                bootstrap_method = method.split("_", 1)[1]
                results[method] = self.calculate_bootstrap_ci(values, bootstrap_method)
            else:
                results[method] = self.calculate_parametric_ci(values, method)
        
        return results
    
    def calculate_ci_for_multiple_metrics(
        self, 
        metrics_dict: Dict[str, List[float]], 
        methods: List[str] = None
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Calculate confidence intervals for multiple metrics."""
        
        results = {}
        
        for metric_name, values in metrics_dict.items():
            results[metric_name] = self.calculate_ci_for_metric(values, methods)
        
        return results
    
    def calculate_prediction_ci(
        self, 
        true_values: List[float], 
        predicted_values: List[float],
        method: str = "residual_bootstrap"
    ) -> Dict[str, float]:
        """Calculate confidence intervals for prediction errors."""
        
        if len(true_values) != len(predicted_values):
            raise ValueError("True and predicted values must have the same length")
        
        # Calculate residuals
        residuals = [t - p for t, p in zip(true_values, predicted_values)]
        
        if method == "residual_bootstrap":
            # Bootstrap residuals
            bootstrap_means = []
            
            for _ in range(self.n_bootstrap):
                # Sample residuals with replacement
                bootstrap_residuals = np.random.choice(residuals, size=len(residuals), replace=True)
                bootstrap_means.append(np.mean(bootstrap_residuals))
            
            # Calculate percentiles
            lower_percentile = (self.alpha / 2) * 100
            upper_percentile = (1 - self.alpha / 2) * 100
            
            ci_lower = np.percentile(bootstrap_means, lower_percentile)
            ci_upper = np.percentile(bootstrap_means, upper_percentile)
            
            return {
                'lower': float(ci_lower),
                'upper': float(ci_upper),
                'mean_error': float(np.mean(residuals)),
                'method': method,
                'n_samples': len(residuals),
                'confidence_level': self.confidence_level
            }
        
        else:
            raise ValueError(f"Unknown prediction CI method: {method}")
    
    def calculate_trajectory_ci(
        self, 
        trajectory_errors: List[List[float]], 
        method: str = "trajectory_bootstrap"
    ) -> Dict[str, Any]:
        """Calculate confidence intervals for trajectory-level metrics."""
        
        if method == "trajectory_bootstrap":
            # Bootstrap entire trajectories
            bootstrap_means = []
            
            for _ in range(self.n_bootstrap):
                # Sample trajectories with replacement
                bootstrap_trajectories = np.random.choice(
                    trajectory_errors, 
                    size=len(trajectory_errors), 
                    replace=True
                )
                
                # Calculate mean error across all trajectories
                all_errors = []
                for traj_errors in bootstrap_trajectories:
                    all_errors.extend(traj_errors)
                
                if all_errors:
                    bootstrap_means.append(np.mean(all_errors))
            
            if not bootstrap_means:
                return {
                    'lower': np.nan,
                    'upper': np.nan,
                    'mean': np.nan,
                    'method': method,
                    'n_trajectories': len(trajectory_errors)
                }
            
            # Calculate percentiles
            lower_percentile = (self.alpha / 2) * 100
            upper_percentile = (1 - self.alpha / 2) * 100
            
            ci_lower = np.percentile(bootstrap_means, lower_percentile)
            ci_upper = np.percentile(bootstrap_means, upper_percentile)
            
            # Calculate overall mean
            all_errors = []
            for traj_errors in trajectory_errors:
                all_errors.extend(traj_errors)
            overall_mean = np.mean(all_errors) if all_errors else np.nan
            
            return {
                'lower': float(ci_lower),
                'upper': float(ci_upper),
                'mean': float(overall_mean) if not np.isnan(overall_mean) else np.nan,
                'method': method,
                'n_trajectories': len(trajectory_errors),
                'n_total_errors': len(all_errors),
                'confidence_level': self.confidence_level
            }
        
        else:
            raise ValueError(f"Unknown trajectory CI method: {method}")
    
    def calculate_model_comparison_ci(
        self, 
        model1_metrics: List[float], 
        model2_metrics: List[float],
        method: str = "bootstrap_difference"
    ) -> Dict[str, float]:
        """Calculate confidence intervals for the difference between two models."""
        
        if method == "bootstrap_difference":
            # Bootstrap the difference in means
            bootstrap_differences = []
            
            for _ in range(self.n_bootstrap):
                # Sample from each model's metrics
                sample1 = np.random.choice(model1_metrics, size=len(model1_metrics), replace=True)
                sample2 = np.random.choice(model2_metrics, size=len(model2_metrics), replace=True)
                
                # Calculate difference in means
                diff = np.mean(sample1) - np.mean(sample2)
                bootstrap_differences.append(diff)
            
            # Calculate percentiles
            lower_percentile = (self.alpha / 2) * 100
            upper_percentile = (1 - self.alpha / 2) * 100
            
            ci_lower = np.percentile(bootstrap_differences, lower_percentile)
            ci_upper = np.percentile(bootstrap_differences, upper_percentile)
            
            # Calculate observed difference
            observed_diff = np.mean(model1_metrics) - np.mean(model2_metrics)
            
            return {
                'lower': float(ci_lower),
                'upper': float(ci_upper),
                'observed_difference': float(observed_diff),
                'method': method,
                'model1_n': len(model1_metrics),
                'model2_n': len(model2_metrics),
                'confidence_level': self.confidence_level
            }
        
        else:
            raise ValueError(f"Unknown model comparison CI method: {method}")
    
    def generate_ci_report(
        self, 
        ci_results: Dict[str, Any], 
        output_path: Optional[str] = None
    ) -> str:
        """Generate a confidence interval report."""
        
        report_lines = []
        
        # Header
        report_lines.append("=" * 80)
        report_lines.append("CONFIDENCE INTERVAL ESTIMATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        report_lines.append(f"Confidence Level: {self.confidence_level:.1%}")
        report_lines.append(f"Alpha Level: {self.alpha:.3f}")
        report_lines.append("")
        
        # Process results
        if isinstance(ci_results, dict):
            for metric_name, metric_results in ci_results.items():
                report_lines.append(f"METRIC: {metric_name.upper()}")
                report_lines.append("-" * 40)
                
                if isinstance(metric_results, dict):
                    for method_name, method_results in metric_results.items():
                        if isinstance(method_results, dict) and 'lower' in method_results:
                            report_lines.append(f"Method: {method_name}")
                            report_lines.append(f"  Mean: {method_results.get('mean', 'N/A'):.4f}")
                            report_lines.append(f"  CI: [{method_results['lower']:.4f}, {method_results['upper']:.4f}]")
                            report_lines.append(f"  Width: {method_results['upper'] - method_results['lower']:.4f}")
                            if 'n_samples' in method_results:
                                report_lines.append(f"  Sample Size: {method_results['n_samples']}")
                            report_lines.append("")
                
                report_lines.append("")
        
        report = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Confidence interval report saved to {output_path}")
        
        return report