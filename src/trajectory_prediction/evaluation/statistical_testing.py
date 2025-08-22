"""Statistical significance testing for trajectory prediction model evaluation."""

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy import stats


def paired_t_test(
    results_a: list[float], results_b: list[float], alpha: float = 0.05
) -> dict[str, Any]:
    """Perform paired t-test to compare two model performances.

    Args:
        results_a: Performance metrics from model A
        results_b: Performance metrics from model B
        alpha: Significance level

    Returns:
        Dictionary with test results
    """
    if len(results_a) != len(results_b):
        raise ValueError("Both result lists must have the same length")

    if len(results_a) < 2:
        return {
            "test_type": "paired_t_test",
            "p_value": float("nan"),
            "statistic": float("nan"),
            "significant": False,
            "effect_size": float("nan"),
            "confidence_interval": (float("nan"), float("nan")),
            "error": "Insufficient data points",
        }

    # Filter out infinite values
    finite_mask = np.isfinite(results_a) & np.isfinite(results_b)
    if not finite_mask.any():
        return {
            "test_type": "paired_t_test",
            "p_value": float("nan"),
            "statistic": float("nan"),
            "significant": False,
            "effect_size": float("nan"),
            "confidence_interval": (float("nan"), float("nan")),
            "error": "No finite values",
        }

    filtered_a = np.array(results_a)[finite_mask]
    filtered_b = np.array(results_b)[finite_mask]

    # Perform paired t-test
    try:
        result = stats.ttest_rel(filtered_a, filtered_b)
        # Handle both old and new scipy versions
        statistic = result[0] if isinstance(result, tuple) else result.statistic
        p_value = result[1] if isinstance(result, tuple) else result.pvalue
    except Exception as e:
        return {
            "test_type": "paired_t_test",
            "p_value": float("nan"),
            "statistic": float("nan"),
            "significant": False,
            "effect_size": float("nan"),
            "confidence_interval": (float("nan"), float("nan")),
            "error": str(e),
        }

    # Calculate effect size (Cohen's d for paired samples)
    differences = filtered_a - filtered_b
    effect_size = (
        np.mean(differences) / np.std(differences, ddof=1)
        if len(differences) > 1
        else 0.0
    )

    # Calculate confidence interval for the difference
    n = len(differences)
    if n > 1:
        sem = stats.sem(differences)
        t_critical = stats.t.ppf(1 - alpha / 2, n - 1)
        margin_of_error = t_critical * sem
        mean_diff = np.mean(differences)
        confidence_interval = (mean_diff - margin_of_error, mean_diff + margin_of_error)
    else:
        confidence_interval = (float("nan"), float("nan"))

    return {
        "test_type": "paired_t_test",
        "p_value": p_value,
        "statistic": statistic,
        "significant": p_value < alpha,
        "effect_size": float(effect_size),
        "confidence_interval": confidence_interval,
        "n_samples": len(filtered_a),
        "alpha": alpha,
    }


def wilcoxon_signed_rank_test(
    results_a: list[float], results_b: list[float], alpha: float = 0.05
) -> dict[str, Any]:
    """Perform Wilcoxon signed-rank test (non-parametric alternative to paired t-test).

    Args:
        results_a: Performance metrics from model A
        results_b: Performance metrics from model B
        alpha: Significance level

    Returns:
        Dictionary with test results
    """
    if len(results_a) != len(results_b):
        raise ValueError("Both result lists must have the same length")

    if len(results_a) < 3:
        return {
            "test_type": "wilcoxon_signed_rank",
            "p_value": float("nan"),
            "statistic": float("nan"),
            "significant": False,
            "error": "Insufficient data points (need at least 3)",
        }

    # Filter out infinite values
    finite_mask = np.isfinite(results_a) & np.isfinite(results_b)
    if not finite_mask.any():
        return {
            "test_type": "wilcoxon_signed_rank",
            "p_value": float("nan"),
            "statistic": float("nan"),
            "significant": False,
            "error": "No finite values",
        }

    filtered_a = np.array(results_a)[finite_mask]
    filtered_b = np.array(results_b)[finite_mask]

    # Remove ties (identical values)
    differences = filtered_a - filtered_b
    non_zero_mask = differences != 0

    if not non_zero_mask.any():
        return {
            "test_type": "wilcoxon_signed_rank",
            "p_value": 1.0,
            "statistic": 0.0,
            "significant": False,
            "n_samples": len(filtered_a),
            "alpha": alpha,
            "note": "All differences are zero",
        }

    filtered_differences = differences[non_zero_mask]

    if len(filtered_differences) < 3:
        return {
            "test_type": "wilcoxon_signed_rank",
            "p_value": float("nan"),
            "statistic": float("nan"),
            "significant": False,
            "error": "Insufficient non-zero differences",
        }

    # Perform Wilcoxon signed-rank test
    try:
        statistic, p_value = stats.wilcoxon(filtered_differences)

        return {
            "test_type": "wilcoxon_signed_rank",
            "p_value": float(p_value),
            "statistic": float(statistic),
            "significant": p_value < alpha,
            "n_samples": len(filtered_a),
            "n_non_zero_differences": len(filtered_differences),
            "alpha": alpha,
        }
    except Exception as e:
        return {
            "test_type": "wilcoxon_signed_rank",
            "p_value": float("nan"),
            "statistic": float("nan"),
            "significant": False,
            "error": str(e),
        }


def multiple_comparison_correction(
    p_values: list[float], method: str = "bonferroni", alpha: float = 0.05
) -> dict[str, Any]:
    """Apply multiple comparison correction to p-values.

    Args:
        p_values: List of p-values to correct
        method: Correction method ('bonferroni', 'holm', 'fdr_bh')
        alpha: Family-wise error rate

    Returns:
        Dictionary with corrected p-values and significance
    """
    p_array = np.array(p_values)
    finite_mask = np.isfinite(p_array)

    if not finite_mask.any():
        return {
            "method": method,
            "corrected_p_values": p_values,
            "significant": [False] * len(p_values),
            "alpha": alpha,
            "error": "No finite p-values",
        }

    if method == "bonferroni":
        corrected_p = np.minimum(p_array * len(p_values), 1.0)
        significant = corrected_p < alpha

    elif method == "holm":
        # Holm-Bonferroni method
        sorted_indices = np.argsort(p_array)
        corrected_p = np.zeros_like(p_array)
        significant = np.zeros(len(p_array), dtype=bool)

        for i, idx in enumerate(sorted_indices):
            if np.isfinite(p_array[idx]):
                corrected_p[idx] = min(p_array[idx] * (len(p_values) - i), 1.0)
                significant[idx] = corrected_p[idx] < alpha

                # If this hypothesis is not rejected, stop
                if not significant[idx]:
                    break

    elif method == "fdr_bh":
        # Benjamini-Hochberg FDR correction
        from scipy.stats import false_discovery_control

        try:
            significant = false_discovery_control(p_array, alpha=alpha)
            corrected_p = (
                p_array.copy()
            )  # FDR doesn't adjust p-values, just determines significance
        except Exception:
            # Fallback implementation
            sorted_indices = np.argsort(p_array)
            corrected_p = np.zeros_like(p_array)
            significant = np.zeros(len(p_array), dtype=bool)

            for i, idx in enumerate(sorted_indices[::-1]):  # Start from largest p-value
                if np.isfinite(p_array[idx]):
                    threshold = (i + 1) / len(p_values) * alpha
                    if p_array[idx] <= threshold:
                        significant[sorted_indices[-(i + 1) :]] = True
                        break

            corrected_p = p_array.copy()
    else:
        raise ValueError(f"Unknown correction method: {method}")

    return {
        "method": method,
        "corrected_p_values": corrected_p.tolist(),
        "significant": significant.tolist(),
        "alpha": alpha,
        "n_hypotheses": len(p_values),
    }


def bootstrap_confidence_interval(
    data: list[float],
    statistic_func: Callable[[Any], float] = np.mean,
    confidence_level: float = 0.95,
    n_bootstrap: int = 10000,
) -> dict[str, Any]:
    """Calculate bootstrap confidence interval for a statistic.

    Args:
        data: Data to bootstrap
        statistic_func: Function to calculate statistic (default: mean)
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        n_bootstrap: Number of bootstrap samples

    Returns:
        Dictionary with confidence interval and bootstrap statistics
    """
    finite_data = np.array([x for x in data if np.isfinite(x)])

    if len(finite_data) < 2:
        return {
            "confidence_interval": (float("nan"), float("nan")),
            "mean": float("nan"),
            "std": float("nan"),
            "confidence_level": confidence_level,
            "error": "Insufficient finite data",
        }

    # Original statistic
    original_stat = statistic_func(finite_data)

    # Bootstrap sampling
    bootstrap_stats_list = []
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(
            finite_data, size=len(finite_data), replace=True
        )
        bootstrap_stats_list.append(statistic_func(bootstrap_sample))

    bootstrap_stats = np.array(bootstrap_stats_list)

    # Calculate confidence interval using percentile method
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    confidence_interval = (
        np.percentile(bootstrap_stats, lower_percentile),
        np.percentile(bootstrap_stats, upper_percentile),
    )

    return {
        "confidence_interval": confidence_interval,
        "original_statistic": float(original_stat),
        "bootstrap_mean": float(np.mean(bootstrap_stats)),
        "bootstrap_std": float(np.std(bootstrap_stats)),
        "confidence_level": confidence_level,
        "n_bootstrap": n_bootstrap,
        "n_data_points": len(finite_data),
    }


def effect_size_cohens_d(group1: list[float], group2: list[float]) -> float:
    """Calculate Cohen's d effect size between two groups.

    Args:
        group1: First group of values
        group2: Second group of values

    Returns:
        Cohen's d effect size
    """
    # Filter finite values
    g1 = np.array([x for x in group1 if np.isfinite(x)])
    g2 = np.array([x for x in group2 if np.isfinite(x)])

    if len(g1) < 2 or len(g2) < 2:
        return float("nan")

    # Calculate pooled standard deviation
    n1, n2 = len(g1), len(g2)
    s1, s2 = np.std(g1, ddof=1), np.std(g2, ddof=1)

    pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    # Calculate Cohen's d
    return float((np.mean(g1) - np.mean(g2)) / pooled_std)
