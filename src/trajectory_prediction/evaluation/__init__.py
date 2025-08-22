"""Evaluation metrics and validation framework."""

from .comprehensive_evaluator import ComprehensiveEvaluator
from .cross_validation import CrossValidationRunner, TimeSeriesCrossValidator
from .metrics import ModelEvaluator, ade, evaluate_single_prediction, fde, rmse
from .safety_metrics import (
    collision_probability,
    lateral_error,
    minimum_distance,
    safety_critical_scenarios,
    time_to_collision,
)
from .statistical_testing import (
    bootstrap_confidence_interval,
    effect_size_cohens_d,
    multiple_comparison_correction,
    paired_t_test,
    wilcoxon_signed_rank_test,
)
from .uncertainty_metrics import (
    adaptive_uncertainty_score,
    calibration_error,
    continuous_ranked_probability_score,
    epistemic_aleatoric_decomposition,
    mean_prediction_interval_width,
    prediction_consistency_score,
    prediction_interval_coverage_probability,
    uncertainty_quality_metrics,
)

__all__ = [
    # Comprehensive evaluation
    "ComprehensiveEvaluator",
    # Basic metrics
    "ModelEvaluator",
    "rmse",
    "ade",
    "fde",
    "evaluate_single_prediction",
    # Cross validation
    "CrossValidationRunner",
    "TimeSeriesCrossValidator",
    # Safety metrics
    "minimum_distance",
    "time_to_collision",
    "lateral_error",
    "collision_probability",
    "safety_critical_scenarios",
    # Statistical testing
    "paired_t_test",
    "wilcoxon_signed_rank_test",
    "multiple_comparison_correction",
    "bootstrap_confidence_interval",
    "effect_size_cohens_d",
    # Uncertainty quantification metrics
    "prediction_interval_coverage_probability",
    "mean_prediction_interval_width",
    "continuous_ranked_probability_score",
    "calibration_error",
    "epistemic_aleatoric_decomposition",
    "uncertainty_quality_metrics",
    "prediction_consistency_score",
    "adaptive_uncertainty_score",
]
