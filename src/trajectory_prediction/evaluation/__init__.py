"""Evaluation metrics and validation framework."""

from .cross_validation import CrossValidationRunner, TimeSeriesCrossValidator
from .metrics import ModelEvaluator, ade, evaluate_single_prediction, fde, rmse
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
    # Basic metrics
    "ModelEvaluator",
    "rmse",
    "ade",
    "fde",
    "evaluate_single_prediction",
    # Cross validation
    "CrossValidationRunner",
    "TimeSeriesCrossValidator",
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
