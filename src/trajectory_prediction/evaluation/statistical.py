"""
Statistical testing framework for trajectory prediction model evaluation.

This module provides comprehensive statistical analysis including:
- Significance testing for model comparisons
- Confidence intervals and bootstrapping
- Effect size calculations
- Multiple comparison corrections
"""

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ttest_ind, wilcoxon, mannwhitneyu, friedmanchisquare
from sklearn.utils import resample
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@dataclass
class SignificanceTest:
    """Results of a statistical significance test."""
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    alpha: float = 0.05
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    interpretation: str = ""


class StatisticalTester:
    """
    Statistical testing framework for model comparison.
    
    Provides rigorous statistical analysis of model performance differences.
    """
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.alpha = config.get("significance_alpha", 0.05)
        self.bootstrap_n_samples = config.get("bootstrap_samples", 1000)
        self.random_state = config.get("random_state", 42)
    
    async def compare_models_statistical(
        self,
        model_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform comprehensive statistical comparison of models.
        
        Args:
            model_results: Dictionary of model evaluation results
            
        Returns:
            Statistical comparison results
        """
        if len(model_results) < 2:
            return {"error": "Need at least 2 models for statistical comparison"}
        
        logger.info(f"Performing statistical comparison of {len(model_results)} models")
        
        comparison_results = {
            "models": list(model_results.keys()),
            "alpha": self.alpha,
            "pairwise_tests": {},
            "overall_tests": {},
            "effect_sizes": {},
            "confidence_intervals": {}
        }
        
        # Extract metric values for all models
        metric_data = self._extract_metric_data(model_results)
        
        # Pairwise comparisons
        model_names = list(model_results.keys())
        for i, model1 in enumerate(model_names):
            for model2 in model_names[i+1:]:
                comparison_key = f"{model1}_vs_{model2}"
                
                pairwise_results = await self._pairwise_comparison(
                    model1, model2, metric_data
                )
                comparison_results["pairwise_tests"][comparison_key] = pairwise_results
        
        # Overall significance tests (if more than 2 models)
        if len(model_names) > 2:
            overall_results = await self._overall_comparison(metric_data)
            comparison_results["overall_tests"] = overall_results
        
        # Multiple comparison correction
        comparison_results = self._apply_multiple_comparison_correction(comparison_results)
        
        return comparison_results
    
    async def _pairwise_comparison(
        self,
        model1: str,
        model2: str,
        metric_data: Dict[str, Dict[str, List[float]]]
    ) -> Dict[str, SignificanceTest]:
        """Perform pairwise statistical comparison between two models."""
        
        pairwise_results = {}
        
        for metric_name, model_data in metric_data.items():
            if model1 in model_data and model2 in model_data:
                values1 = model_data[model1]
                values2 = model_data[model2]
                
                if len(values1) > 1 and len(values2) > 1:
                    # Perform multiple tests
                    tests = await self._run_statistical_tests(values1, values2, metric_name)
                    pairwise_results[metric_name] = tests
        
        return pairwise_results
    
    async def _run_statistical_tests(
        self,
        values1: List[float],
        values2: List[float],
        metric_name: str
    ) -> Dict[str, SignificanceTest]:
        """Run multiple statistical tests on two sets of values."""
        
        tests = {}
        
        # Remove any infinite or NaN values
        clean_values1 = [v for v in values1 if np.isfinite(v)]
        clean_values2 = [v for v in values2 if np.isfinite(v)]
        
        if len(clean_values1) < 2 or len(clean_values2) < 2:
            return {"error": "Insufficient data for statistical testing"}
        
        # Convert to numpy arrays
        arr1 = np.array(clean_values1)
        arr2 = np.array(clean_values2)
        
        # T-test (assumes normality)
        try:
            t_stat, t_p = ttest_ind(arr1, arr2, equal_var=False)  # Welch's t-test
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt((np.var(arr1, ddof=1) + np.var(arr2, ddof=1)) / 2)
            cohens_d = (np.mean(arr1) - np.mean(arr2)) / pooled_std if pooled_std > 0 else 0
            
            tests["t_test"] = SignificanceTest(
                test_name="Welch's t-test",
                statistic=float(t_stat),
                p_value=float(t_p),
                significant=t_p < self.alpha,
                alpha=self.alpha,
                effect_size=float(abs(cohens_d)),
                interpretation=self._interpret_cohens_d(abs(cohens_d))
            )
            
        except Exception as e:
            logger.warning(f"T-test failed for {metric_name}: {e}")
        
        # Mann-Whitney U test (non-parametric)
        try:
            u_stat, u_p = mannwhitneyu(arr1, arr2, alternative='two-sided')
            
            # Effect size for Mann-Whitney (r = Z / sqrt(N))
            n1, n2 = len(arr1), len(arr2)
            z_score = stats.norm.ppf(u_p/2) * -1  # Approximate z-score
            r_effect = abs(z_score) / np.sqrt(n1 + n2)
            
            tests["mann_whitney"] = SignificanceTest(
                test_name="Mann-Whitney U test",
                statistic=float(u_stat),
                p_value=float(u_p),
                significant=u_p < self.alpha,
                alpha=self.alpha,
                effect_size=float(r_effect),
                interpretation=self._interpret_r_effect(r_effect)
            )
            
        except Exception as e:
            logger.warning(f"Mann-Whitney test failed for {metric_name}: {e}")
        
        # Bootstrap confidence interval for difference in means
        try:
            ci = await self._bootstrap_confidence_interval(arr1, arr2)
            if "t_test" in tests:
                tests["t_test"].confidence_interval = ci
            
        except Exception as e:
            logger.warning(f"Bootstrap CI failed for {metric_name}: {e}")
        
        return tests
    
    async def _bootstrap_confidence_interval(
        self,
        values1: np.ndarray,
        values2: np.ndarray,
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval for difference in means."""
        
        np.random.seed(self.random_state)
        
        differences = []
        
        for _ in range(self.bootstrap_n_samples):
            # Bootstrap samples
            boot1 = resample(values1, n_samples=len(values1), random_state=None)
            boot2 = resample(values2, n_samples=len(values2), random_state=None)
            
            # Calculate difference in means
            diff = np.mean(boot1) - np.mean(boot2)
            differences.append(diff)
        
        # Calculate confidence interval
        alpha_level = 1 - confidence_level
        lower_percentile = (alpha_level / 2) * 100
        upper_percentile = (1 - alpha_level / 2) * 100
        
        ci_lower = np.percentile(differences, lower_percentile)
        ci_upper = np.percentile(differences, upper_percentile)
        
        return float(ci_lower), float(ci_upper)
    
    async def _overall_comparison(
        self,
        metric_data: Dict[str, Dict[str, List[float]]]
    ) -> Dict[str, SignificanceTest]:
        """Perform overall comparison tests across all models."""
        
        overall_tests = {}
        
        for metric_name, model_data in metric_data.items():
            # Get values for all models
            all_values = []
            for model_name, values in model_data.items():
                clean_values = [v for v in values if np.isfinite(v)]
                if len(clean_values) > 1:
                    all_values.append(clean_values)
            
            if len(all_values) >= 3:  # Need at least 3 groups
                try:
                    # Kruskal-Wallis test (non-parametric ANOVA)
                    h_stat, h_p = stats.kruskal(*all_values)
                    
                    overall_tests[f"{metric_name}_kruskal_wallis"] = SignificanceTest(
                        test_name="Kruskal-Wallis H test",
                        statistic=float(h_stat),
                        p_value=float(h_p),
                        significant=h_p < self.alpha,
                        alpha=self.alpha,
                        interpretation="Significant differences exist between models" if h_p < self.alpha else "No significant differences between models"
                    )
                    
                except Exception as e:
                    logger.warning(f"Kruskal-Wallis test failed for {metric_name}: {e}")
        
        return overall_tests
    
    def _extract_metric_data(
        self,
        model_results: Dict[str, Any]
    ) -> Dict[str, Dict[str, List[float]]]:
        """Extract metric values from model results for statistical analysis."""
        
        metric_data = {}
        
        # Common metrics to extract
        metrics_to_extract = [
            ("trajectory_metrics", "ade"),
            ("trajectory_metrics", "fde"), 
            ("trajectory_metrics", "rmse"),
            ("safety_metrics", "min_ttc"),
            ("safety_metrics", "collision_risk"),
            ("probabilistic_metrics", "mean_calibration_error")
        ]
        
        for category, metric in metrics_to_extract:
            metric_key = f"{category}_{metric}"
            metric_data[metric_key] = {}
            
            for model_name, results in model_results.items():
                if (category in results and 
                    metric in results[category]):
                    
                    metric_info = results[category][metric]
                    
                    # Extract values (could be single value or dict with stats)
                    if isinstance(metric_info, dict) and "mean" in metric_info:
                        value = metric_info["mean"]
                        # If we have individual fold results, extract those
                        if "cross_validation" in results:
                            cv_results = results["cross_validation"]
                            if "fold_results" in cv_results:
                                fold_values = []
                                for fold in cv_results["fold_results"]:
                                    if (category in fold and 
                                        metric in fold[category] and
                                        isinstance(fold[category][metric], dict) and
                                        "mean" in fold[category][metric]):
                                        fold_val = fold[category][metric]["mean"]
                                        if np.isfinite(fold_val):
                                            fold_values.append(fold_val)
                                
                                if fold_values:
                                    metric_data[metric_key][model_name] = fold_values
                                else:
                                    metric_data[metric_key][model_name] = [value]
                            else:
                                metric_data[metric_key][model_name] = [value]
                        else:
                            metric_data[metric_key][model_name] = [value]
                    
                    elif isinstance(metric_info, (int, float)) and np.isfinite(metric_info):
                        metric_data[metric_key][model_name] = [metric_info]
        
        # Filter out metrics with insufficient data
        filtered_data = {}
        for metric_key, model_data in metric_data.items():
            if len(model_data) >= 2:  # Need at least 2 models
                filtered_data[metric_key] = model_data
        
        return filtered_data
    
    def _apply_multiple_comparison_correction(
        self,
        comparison_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply multiple comparison correction to p-values."""
        
        # Collect all p-values
        all_p_values = []
        p_value_info = []  # (comparison_key, metric, test_type, original_index)
        
        for comparison_key, pairwise_data in comparison_results["pairwise_tests"].items():
            for metric_name, tests in pairwise_data.items():
                if isinstance(tests, dict):
                    for test_type, test_result in tests.items():
                        if isinstance(test_result, SignificanceTest):
                            all_p_values.append(test_result.p_value)
                            p_value_info.append((comparison_key, metric_name, test_type, len(all_p_values)-1))
        
        if len(all_p_values) > 1:
            # Apply Benjamini-Hochberg FDR correction
            try:
                from statsmodels.stats.multitest import multipletests
                
                rejected, corrected_p_values, _, _ = multipletests(
                    all_p_values, alpha=self.alpha, method='fdr_bh'
                )
                
                # Update significance tests with corrected p-values
                for i, (comparison_key, metric_name, test_type, orig_idx) in enumerate(p_value_info):
                    test_obj = comparison_results["pairwise_tests"][comparison_key][metric_name][test_type]
                    test_obj.p_value_corrected = float(corrected_p_values[i])
                    test_obj.significant_corrected = bool(rejected[i])
                
                comparison_results["multiple_comparison_correction"] = {
                    "method": "Benjamini-Hochberg FDR",
                    "total_tests": len(all_p_values),
                    "rejected_hypotheses": int(np.sum(rejected))
                }
                
            except ImportError:
                logger.warning("statsmodels not available for multiple comparison correction")
        
        return comparison_results
    
    def _interpret_cohens_d(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size."""
        if cohens_d < 0.2:
            return "negligible effect"
        elif cohens_d < 0.5:
            return "small effect"
        elif cohens_d < 0.8:
            return "medium effect"
        else:
            return "large effect"
    
    def _interpret_r_effect(self, r: float) -> str:
        """Interpret r effect size."""
        if r < 0.1:
            return "negligible effect"
        elif r < 0.3:
            return "small effect"
        elif r < 0.5:
            return "medium effect"
        else:
            return "large effect"
    
    async def confidence_interval_mean(
        self,
        values: List[float],
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate confidence interval for the mean."""
        
        clean_values = [v for v in values if np.isfinite(v)]
        if len(clean_values) < 2:
            return float("nan"), float("nan")
        
        arr = np.array(clean_values)
        n = len(arr)
        mean = np.mean(arr)
        sem = stats.sem(arr)  # Standard error of the mean
        
        # T-distribution critical value
        alpha = 1 - confidence_level
        t_critical = stats.t.ppf(1 - alpha/2, n - 1)
        
        margin_error = t_critical * sem
        
        return float(mean - margin_error), float(mean + margin_error)
    
    async def effect_size_analysis(
        self,
        model_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Comprehensive effect size analysis between models."""
        
        if len(model_results) < 2:
            return {"error": "Need at least 2 models for effect size analysis"}
        
        metric_data = self._extract_metric_data(model_results)
        effect_sizes = {}
        
        model_names = list(model_results.keys())
        
        for i, model1 in enumerate(model_names):
            for model2 in model_names[i+1:]:
                comparison_key = f"{model1}_vs_{model2}"
                effect_sizes[comparison_key] = {}
                
                for metric_name, model_data in metric_data.items():
                    if model1 in model_data and model2 in model_data:
                        values1 = [v for v in model_data[model1] if np.isfinite(v)]
                        values2 = [v for v in model_data[model2] if np.isfinite(v)]
                        
                        if len(values1) > 1 and len(values2) > 1:
                            arr1, arr2 = np.array(values1), np.array(values2)
                            
                            # Cohen's d
                            pooled_std = np.sqrt((np.var(arr1, ddof=1) + np.var(arr2, ddof=1)) / 2)
                            cohens_d = (np.mean(arr1) - np.mean(arr2)) / pooled_std if pooled_std > 0 else 0
                            
                            # Glass's delta (using control group std)
                            glass_delta = (np.mean(arr1) - np.mean(arr2)) / np.std(arr2, ddof=1) if np.std(arr2, ddof=1) > 0 else 0
                            
                            effect_sizes[comparison_key][metric_name] = {
                                "cohens_d": float(cohens_d),
                                "cohens_d_interpretation": self._interpret_cohens_d(abs(cohens_d)),
                                "glass_delta": float(glass_delta),
                                "mean_difference": float(np.mean(arr1) - np.mean(arr2)),
                                "relative_improvement": float((np.mean(arr2) - np.mean(arr1)) / np.mean(arr2)) if np.mean(arr2) != 0 else 0
                            }
        
        return effect_sizes