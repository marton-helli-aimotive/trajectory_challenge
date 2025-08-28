"""Statistical significance testing for trajectory prediction models."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from scipy import stats
from scipy.stats import mannwhitneyu, wilcoxon, friedmanchisquare
import warnings

logger = logging.getLogger(__name__)


class StatisticalTestSuite:
    """Suite of statistical tests for comparing trajectory prediction models."""
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.test_results = {}
    
    def compare_two_models(
        self, 
        model1_metrics: List[float], 
        model2_metrics: List[float],
        test_name: str = "model1_vs_model2",
        test_type: str = "paired"
    ) -> Dict[str, Any]:
        """Compare two models using appropriate statistical tests."""
        
        # Remove infinite and NaN values
        metrics1 = [m for m in model1_metrics if not np.isinf(m) and not np.isnan(m)]
        metrics2 = [m for m in model2_metrics if not np.isinf(m) and not np.isnan(m)]
        
        if len(metrics1) == 0 or len(metrics2) == 0:
            return {
                'error': 'No valid metrics for comparison',
                'test_name': test_name
            }
        
        results = {
            'test_name': test_name,
            'test_type': test_type,
            'model1_n': len(metrics1),
            'model2_n': len(metrics2),
            'model1_mean': float(np.mean(metrics1)),
            'model2_mean': float(np.mean(metrics2)),
            'model1_std': float(np.std(metrics1)),
            'model2_std': float(np.std(metrics2)),
            'mean_difference': float(np.mean(metrics1) - np.mean(metrics2))
        }
        
        # Perform appropriate statistical test
        if test_type == "paired":
            if len(metrics1) == len(metrics2):
                # Paired t-test
                t_stat, p_value = stats.ttest_rel(metrics1, metrics2)
                results['test_method'] = 'paired_t_test'
                results['t_statistic'] = float(t_stat)
                results['p_value'] = float(p_value)
                results['significant'] = p_value < self.alpha
                
                # Wilcoxon signed-rank test (non-parametric)
                try:
                    w_stat, w_p_value = wilcoxon(metrics1, metrics2)
                    results['wilcoxon_statistic'] = float(w_stat)
                    results['wilcoxon_p_value'] = float(w_p_value)
                    results['wilcoxon_significant'] = w_p_value < self.alpha
                except Exception as e:
                    logger.warning(f"Wilcoxon test failed: {e}")
                    results['wilcoxon_error'] = str(e)
            else:
                results['error'] = 'Paired test requires equal sample sizes'
        
        elif test_type == "independent":
            # Independent t-test
            t_stat, p_value = stats.ttest_ind(metrics1, metrics2)
            results['test_method'] = 'independent_t_test'
            results['t_statistic'] = float(t_stat)
            results['p_value'] = float(p_value)
            results['significant'] = p_value < self.alpha
            
            # Mann-Whitney U test (non-parametric)
            try:
                u_stat, u_p_value = mannwhitneyu(metrics1, metrics2, alternative='two-sided')
                results['mann_whitney_statistic'] = float(u_stat)
                results['mann_whitney_p_value'] = float(u_p_value)
                results['mann_whitney_significant'] = u_p_value < self.alpha
            except Exception as e:
                logger.warning(f"Mann-Whitney test failed: {e}")
                results['mann_whitney_error'] = str(e)
        
        else:
            results['error'] = f'Unknown test type: {test_type}'
        
        # Calculate effect size (Cohen's d)
        if 'error' not in results:
            pooled_std = np.sqrt(((len(metrics1) - 1) * np.var(metrics1) + 
                                 (len(metrics2) - 1) * np.var(metrics2)) / 
                                (len(metrics1) + len(metrics2) - 2))
            if pooled_std > 0:
                cohens_d = (np.mean(metrics1) - np.mean(metrics2)) / pooled_std
                results['cohens_d'] = float(cohens_d)
                results['effect_size_interpretation'] = self._interpret_cohens_d(cohens_d)
        
        return results
    
    def compare_multiple_models(
        self, 
        model_metrics: Dict[str, List[float]],
        test_type: str = "independent"
    ) -> Dict[str, Any]:
        """Compare multiple models using appropriate statistical tests."""
        
        model_names = list(model_metrics.keys())
        if len(model_names) < 2:
            return {'error': 'Need at least 2 models for comparison'}
        
        # Clean metrics
        cleaned_metrics = {}
        for name, metrics in model_metrics.items():
            cleaned_metrics[name] = [m for m in metrics if not np.isinf(m) and not np.isnan(m)]
        
        results = {
            'test_type': test_type,
            'num_models': len(model_names),
            'model_names': model_names,
            'pairwise_comparisons': {},
            'overall_tests': {}
        }
        
        # Pairwise comparisons
        for i, model1 in enumerate(model_names):
            for model2 in model_names[i+1:]:
                comparison_key = f"{model1}_vs_{model2}"
                comparison_result = self.compare_two_models(
                    cleaned_metrics[model1], 
                    cleaned_metrics[model2],
                    comparison_key,
                    test_type
                )
                results['pairwise_comparisons'][comparison_key] = comparison_result
        
        # Overall tests for multiple models
        if len(model_names) > 2:
            overall_results = self._perform_overall_tests(cleaned_metrics, test_type)
            results['overall_tests'] = overall_results
        
        return results
    
    def _perform_overall_tests(
        self, 
        model_metrics: Dict[str, List[float]], 
        test_type: str
    ) -> Dict[str, Any]:
        """Perform overall statistical tests for multiple models."""
        
        results = {}
        
        # One-way ANOVA (parametric)
        try:
            # Ensure all groups have the same length by truncating to minimum
            min_length = min(len(metrics) for metrics in model_metrics.values())
            if min_length > 0:
                equal_length_metrics = [metrics[:min_length] for metrics in model_metrics.values()]
                f_stat, p_value = stats.f_oneway(*equal_length_metrics)
                
                results['anova'] = {
                    'f_statistic': float(f_stat),
                    'p_value': float(p_value),
                    'significant': p_value < self.alpha
                }
        except Exception as e:
            logger.warning(f"ANOVA test failed: {e}")
            results['anova_error'] = str(e)
        
        # Kruskal-Wallis test (non-parametric)
        try:
            # Handle different sample sizes
            h_stat, p_value = stats.kruskal(*model_metrics.values())
            results['kruskal_wallis'] = {
                'h_statistic': float(h_stat),
                'p_value': float(p_value),
                'significant': p_value < self.alpha
            }
        except Exception as e:
            logger.warning(f"Kruskal-Wallis test failed: {e}")
            results['kruskal_wallis_error'] = str(e)
        
        # Friedman test (for repeated measures/paired data)
        if test_type == "paired":
            try:
                # Ensure all groups have the same length
                min_length = min(len(metrics) for metrics in model_metrics.values())
                if min_length > 0:
                    equal_length_metrics = [metrics[:min_length] for metrics in model_metrics.values()]
                    # Transpose to get subjects as rows and models as columns
                    data_matrix = np.array(equal_length_metrics).T
                    h_stat, p_value = friedmanchisquare(*data_matrix.T)
                    
                    results['friedman'] = {
                        'h_statistic': float(h_stat),
                        'p_value': float(p_value),
                        'significant': p_value < self.alpha
                    }
            except Exception as e:
                logger.warning(f"Friedman test failed: {e}")
                results['friedman_error'] = str(e)
        
        return results
    
    def _interpret_cohens_d(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def perform_bootstrap_test(
        self, 
        model1_metrics: List[float], 
        model2_metrics: List[float],
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """Perform bootstrap test for comparing two models."""
        
        # Remove infinite and NaN values
        metrics1 = [m for m in model1_metrics if not np.isinf(m) and not np.isnan(m)]
        metrics2 = [m for m in model2_metrics if not np.isinf(m) and not np.isnan(m)]
        
        if len(metrics1) == 0 or len(metrics2) == 0:
            return {'error': 'No valid metrics for bootstrap test'}
        
        # Bootstrap sampling
        bootstrap_differences = []
        
        for _ in range(n_bootstrap):
            # Sample with replacement
            sample1 = np.random.choice(metrics1, size=len(metrics1), replace=True)
            sample2 = np.random.choice(metrics2, size=len(metrics2), replace=True)
            
            # Calculate difference in means
            diff = np.mean(sample1) - np.mean(sample2)
            bootstrap_differences.append(diff)
        
        # Calculate confidence interval
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_differences, lower_percentile)
        ci_upper = np.percentile(bootstrap_differences, upper_percentile)
        
        # Calculate p-value (proportion of bootstrap samples where difference is in opposite direction)
        observed_diff = np.mean(metrics1) - np.mean(metrics2)
        if observed_diff > 0:
            p_value = np.mean(np.array(bootstrap_differences) <= 0)
        else:
            p_value = np.mean(np.array(bootstrap_differences) >= 0)
        
        # Two-sided p-value
        p_value = 2 * min(p_value, 1 - p_value)
        
        return {
            'bootstrap_confidence_interval': [float(ci_lower), float(ci_upper)],
            'bootstrap_p_value': float(p_value),
            'bootstrap_significant': p_value < self.alpha,
            'observed_difference': float(observed_diff),
            'n_bootstrap': n_bootstrap,
            'confidence_level': confidence_level
        }
    
    def perform_permutation_test(
        self, 
        model1_metrics: List[float], 
        model2_metrics: List[float],
        n_permutations: int = 1000
    ) -> Dict[str, Any]:
        """Perform permutation test for comparing two models."""
        
        # Remove infinite and NaN values
        metrics1 = [m for m in model1_metrics if not np.isinf(m) and not np.isnan(m)]
        metrics2 = [m for m in model2_metrics if not np.isinf(m) and not np.isnan(m)]
        
        if len(metrics1) == 0 or len(metrics2) == 0:
            return {'error': 'No valid metrics for permutation test'}
        
        # Combine all metrics
        all_metrics = metrics1 + metrics2
        observed_diff = np.mean(metrics1) - np.mean(metrics2)
        
        # Permutation test
        extreme_count = 0
        
        for _ in range(n_permutations):
            # Shuffle the combined data
            shuffled = np.random.permutation(all_metrics)
            
            # Split into two groups
            n1 = len(metrics1)
            perm_group1 = shuffled[:n1]
            perm_group2 = shuffled[n1:]
            
            # Calculate difference
            perm_diff = np.mean(perm_group1) - np.mean(perm_group2)
            
            # Count extreme values
            if abs(perm_diff) >= abs(observed_diff):
                extreme_count += 1
        
        # Calculate p-value
        p_value = extreme_count / n_permutations
        
        return {
            'permutation_p_value': float(p_value),
            'permutation_significant': p_value < self.alpha,
            'observed_difference': float(observed_diff),
            'n_permutations': n_permutations,
            'extreme_count': extreme_count
        }
    
    def generate_statistical_report(
        self, 
        test_results: Dict[str, Any], 
        output_path: Optional[str] = None
    ) -> str:
        """Generate a comprehensive statistical test report."""
        
        report_lines = []
        
        # Header
        report_lines.append("=" * 80)
        report_lines.append("STATISTICAL SIGNIFICANCE TEST REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Test parameters
        report_lines.append(f"Significance Level (alpha): {self.alpha}")
        report_lines.append("")
        
        # Overall tests
        if 'overall_tests' in test_results and test_results['overall_tests']:
            report_lines.append("OVERALL TESTS")
            report_lines.append("-" * 40)
            
            overall = test_results['overall_tests']
            
            if 'anova' in overall:
                anova = overall['anova']
                report_lines.append("One-way ANOVA:")
                report_lines.append(f"  F-statistic: {anova['f_statistic']:.4f}")
                report_lines.append(f"  p-value: {anova['p_value']:.4f}")
                report_lines.append(f"  Significant: {anova['significant']}")
                report_lines.append("")
            
            if 'kruskal_wallis' in overall:
                kw = overall['kruskal_wallis']
                report_lines.append("Kruskal-Wallis Test:")
                report_lines.append(f"  H-statistic: {kw['h_statistic']:.4f}")
                report_lines.append(f"  p-value: {kw['p_value']:.4f}")
                report_lines.append(f"  Significant: {kw['significant']}")
                report_lines.append("")
        
        # Pairwise comparisons
        if 'pairwise_comparisons' in test_results:
            report_lines.append("PAIRWISE COMPARISONS")
            report_lines.append("-" * 40)
            
            for comparison_key, result in test_results['pairwise_comparisons'].items():
                if 'error' in result:
                    report_lines.append(f"{comparison_key}: {result['error']}")
                    continue
                
                report_lines.append(f"{comparison_key}:")
                report_lines.append(f"  Model 1 Mean: {result['model1_mean']:.4f} ± {result['model1_std']:.4f}")
                report_lines.append(f"  Model 2 Mean: {result['model2_mean']:.4f} ± {result['model2_std']:.4f}")
                report_lines.append(f"  Mean Difference: {result['mean_difference']:.4f}")
                
                if 'p_value' in result:
                    report_lines.append(f"  p-value: {result['p_value']:.4f}")
                    report_lines.append(f"  Significant: {result['significant']}")
                
                if 'cohens_d' in result:
                    report_lines.append(f"  Cohen's d: {result['cohens_d']:.4f} ({result['effect_size_interpretation']})")
                
                report_lines.append("")
        
        report = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Statistical report saved to {output_path}")
        
        return report