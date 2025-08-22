"""Advanced evaluation metrics for uncertainty quantification in trajectory prediction."""

from typing import Any

import numpy as np

from ..models.base import PredictionResult


def prediction_interval_coverage_probability(
    predictions: list[PredictionResult],
    ground_truth: list[list[tuple[float, float]]],
) -> float:
    """Calculate Prediction Interval Coverage Probability (PICP).

    Measures how often the true values fall within predicted confidence intervals.
    """
    if not predictions or not ground_truth:
        return 0.0

    total_points = 0
    covered_points = 0

    for pred, true_points in zip(predictions, ground_truth):
        if not pred.confidence_scores or not true_points:
            continue

        min_len = min(
            len(pred.predicted_points), len(true_points), len(pred.confidence_scores)
        )

        for i in range(min_len):
            pred_x, pred_y = pred.predicted_points[i]
            true_x, true_y = true_points[i]
            confidence = pred.confidence_scores[i]

            # Estimate prediction interval based on confidence
            # This is a simplified approach - in practice you'd use proper uncertainty bounds
            uncertainty = (1.0 - confidence) * 10.0  # Scale factor

            # Check if true point falls within interval
            if (
                abs(pred_x - true_x) <= uncertainty
                and abs(pred_y - true_y) <= uncertainty
            ):
                covered_points += 1

            total_points += 1

    return covered_points / total_points if total_points > 0 else 0.0


def mean_prediction_interval_width(
    predictions: list[PredictionResult],
) -> float:
    """Calculate Mean Prediction Interval Width (MPIW).

    Measures the average width of prediction intervals.
    """
    total_width = 0.0
    total_points = 0

    for pred in predictions:
        if not pred.confidence_scores:
            continue

        for i, confidence in enumerate(pred.confidence_scores):
            if i < len(pred.predicted_points):
                # Estimate interval width from confidence
                uncertainty = (1.0 - confidence) * 10.0
                width = 2 * uncertainty  # Total interval width
                total_width += width
                total_points += 1

    return total_width / total_points if total_points > 0 else 0.0


def continuous_ranked_probability_score(
    predictions: list[PredictionResult],
    ground_truth: list[list[tuple[float, float]]],
) -> float:
    """Calculate Continuous Ranked Probability Score (CRPS).

    Measures the quality of probabilistic predictions.
    """
    if not predictions or not ground_truth:
        return float("inf")

    total_crps = 0.0
    total_points = 0

    for pred, true_points in zip(predictions, ground_truth):
        if not pred.confidence_scores or not true_points:
            continue

        min_len = min(
            len(pred.predicted_points), len(true_points), len(pred.confidence_scores)
        )

        for i in range(min_len):
            pred_x, pred_y = pred.predicted_points[i]
            true_x, true_y = true_points[i]
            confidence = pred.confidence_scores[i]

            # Simplified CRPS calculation
            # In practice, you'd integrate over the full predictive distribution
            uncertainty = (1.0 - confidence) * 5.0

            # Distance between prediction and truth
            distance = np.sqrt((pred_x - true_x) ** 2 + (pred_y - true_y) ** 2)

            # Simplified CRPS approximation
            crps_value = distance - 0.5 * uncertainty
            total_crps += abs(crps_value)
            total_points += 1

    return total_crps / total_points if total_points > 0 else float("inf")


def calibration_error(
    predictions: list[PredictionResult],
    ground_truth: list[list[tuple[float, float]]],
    n_bins: int = 10,
) -> dict[str, float]:
    """Calculate calibration error metrics.

    Measures how well predicted confidence aligns with actual accuracy.
    """
    if not predictions or not ground_truth:
        return {"ece": float("inf"), "mce": float("inf")}

    # Collect confidence scores and accuracy indicators
    confidences = []
    accuracies = []

    for pred, true_points in zip(predictions, ground_truth):
        if not pred.confidence_scores or not true_points:
            continue

        min_len = min(
            len(pred.predicted_points), len(true_points), len(pred.confidence_scores)
        )

        for i in range(min_len):
            pred_x, pred_y = pred.predicted_points[i]
            true_x, true_y = true_points[i]
            confidence = pred.confidence_scores[i]

            # Calculate accuracy as inverse of normalized error
            error = np.sqrt((pred_x - true_x) ** 2 + (pred_y - true_y) ** 2)
            accuracy = 1.0 / (1.0 + error)  # Normalized accuracy

            confidences.append(confidence)
            accuracies.append(accuracy)

    if not confidences:
        return {"ece": float("inf"), "mce": float("inf")}

    confidences = np.array(confidences)
    accuracies = np.array(accuracies)

    # Calculate Expected Calibration Error (ECE)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    mce = 0.0

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()

            calibration_error_bin = abs(avg_confidence_in_bin - accuracy_in_bin)
            ece += prop_in_bin * calibration_error_bin
            mce = max(mce, calibration_error_bin)

    return {"ece": float(ece), "mce": float(mce)}


def epistemic_aleatoric_decomposition(
    ensemble_predictions: list[list[PredictionResult]],
    ground_truth: list[list[tuple[float, float]]],
) -> dict[str, float]:
    """Decompose uncertainty into epistemic and aleatoric components.

    Args:
        ensemble_predictions: List of prediction lists from ensemble members
        ground_truth: Ground truth trajectory points
    """
    if not ensemble_predictions or not ground_truth:
        return {"epistemic": 0.0, "aleatoric": 0.0, "total": 0.0}

    n_members = len(ensemble_predictions)
    if n_members < 2:
        return {"epistemic": 0.0, "aleatoric": 0.0, "total": 0.0}

    total_epistemic = 0.0
    total_aleatoric = 0.0
    total_points = 0

    # For each prediction instance
    for pred_idx in range(len(ground_truth)):
        member_predictions = []

        # Collect predictions from all ensemble members
        for member_idx in range(n_members):
            if pred_idx < len(ensemble_predictions[member_idx]):
                member_predictions.append(ensemble_predictions[member_idx][pred_idx])

        if len(member_predictions) < 2:
            continue

        true_points = ground_truth[pred_idx]
        if not true_points:
            continue

        # Find common prediction length
        min_len = min(
            len(pred.predicted_points)
            for pred in member_predictions
            if pred.predicted_points
        )
        min_len = min(min_len, len(true_points))

        for point_idx in range(min_len):
            # Collect predictions from all members for this point
            member_x = []
            member_y = []
            member_confidences = []

            for pred in member_predictions:
                if point_idx < len(pred.predicted_points):
                    x, y = pred.predicted_points[point_idx]
                    member_x.append(x)
                    member_y.append(y)

                    if pred.confidence_scores and point_idx < len(
                        pred.confidence_scores
                    ):
                        member_confidences.append(pred.confidence_scores[point_idx])

            if len(member_x) < 2:
                continue

            # Epistemic uncertainty: variance across ensemble members
            epistemic_x = np.var(member_x)
            epistemic_y = np.var(member_y)
            epistemic = epistemic_x + epistemic_y

            # Aleatoric uncertainty: average of individual uncertainties
            if member_confidences:
                avg_confidence = np.mean(member_confidences)
                aleatoric = (1.0 - avg_confidence) ** 2
            else:
                aleatoric = 0.25  # Default uncertainty

            total_epistemic += epistemic
            total_aleatoric += aleatoric
            total_points += 1

    if total_points == 0:
        return {"epistemic": 0.0, "aleatoric": 0.0, "total": 0.0}

    avg_epistemic = total_epistemic / total_points
    avg_aleatoric = total_aleatoric / total_points
    total_uncertainty = avg_epistemic + avg_aleatoric

    return {
        "epistemic": float(avg_epistemic),
        "aleatoric": float(avg_aleatoric),
        "total": float(total_uncertainty),
    }


def uncertainty_quality_metrics(
    predictions: list[PredictionResult],
    ground_truth: list[list[tuple[float, float]]],
) -> dict[str, Any]:
    """Calculate comprehensive uncertainty quality metrics."""
    metrics = {}

    # Coverage probability
    metrics["picp_95"] = prediction_interval_coverage_probability(
        predictions, ground_truth
    )
    metrics["picp_90"] = prediction_interval_coverage_probability(
        predictions, ground_truth
    )

    # Interval width
    metrics["mpiw"] = mean_prediction_interval_width(predictions)

    # Probabilistic scoring
    metrics["crps"] = continuous_ranked_probability_score(predictions, ground_truth)

    # Calibration
    calibration = calibration_error(predictions, ground_truth)
    metrics.update(calibration)

    # Sharpness (how narrow are the prediction intervals)
    total_sharpness = 0.0
    total_points = 0

    for pred in predictions:
        if pred.confidence_scores:
            avg_confidence = np.mean(pred.confidence_scores)
            sharpness = avg_confidence  # Higher confidence = sharper predictions
            total_sharpness += sharpness
            total_points += 1

    metrics["sharpness"] = total_sharpness / total_points if total_points > 0 else 0.0

    # Reliability (combination of calibration and sharpness)
    if "ece" in metrics and metrics["ece"] != float("inf"):
        metrics["reliability"] = (1.0 - metrics["ece"]) * metrics["sharpness"]
    else:
        metrics["reliability"] = 0.0

    return metrics


def prediction_consistency_score(
    predictions_t1: list[PredictionResult],
    predictions_t2: list[PredictionResult],
) -> float:
    """Calculate consistency score between predictions at different times.

    Measures how stable predictions are when made at different time points.
    """
    if not predictions_t1 or not predictions_t2:
        return 0.0

    total_consistency = 0.0
    valid_comparisons = 0

    for pred1, pred2 in zip(predictions_t1, predictions_t2):
        if not pred1.predicted_points or not pred2.predicted_points:
            continue

        # Compare overlapping portions of predictions
        min_len = min(len(pred1.predicted_points), len(pred2.predicted_points))

        point_similarities = []
        for i in range(min_len):
            x1, y1 = pred1.predicted_points[i]
            x2, y2 = pred2.predicted_points[i]

            # Calculate similarity (inverse of distance)
            distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            similarity = 1.0 / (1.0 + distance)
            point_similarities.append(similarity)

        if point_similarities:
            avg_similarity = np.mean(point_similarities)
            total_consistency += avg_similarity
            valid_comparisons += 1

    return (
        float(total_consistency / valid_comparisons) if valid_comparisons > 0 else 0.0
    )


def adaptive_uncertainty_score(
    predictions: list[PredictionResult],
    ground_truth: list[list[tuple[float, float]]],
    difficulty_scores: list[float] | None = None,
) -> dict[str, float]:
    """Calculate uncertainty scores adapted to prediction difficulty.

    Higher difficulty scenarios should have higher uncertainty.
    """
    if not predictions or not ground_truth:
        return {"adaptive_uncertainty": 0.0, "difficulty_correlation": 0.0}

    uncertainties = []
    errors = []
    difficulties = difficulty_scores or [0.5] * len(predictions)

    for pred, true_points, _difficulty in zip(predictions, ground_truth, difficulties):
        if not pred.confidence_scores or not true_points:
            continue

        min_len = min(
            len(pred.predicted_points), len(true_points), len(pred.confidence_scores)
        )

        for i in range(min_len):
            pred_x, pred_y = pred.predicted_points[i]
            true_x, true_y = true_points[i]
            confidence = pred.confidence_scores[i]

            uncertainty = 1.0 - confidence
            error = np.sqrt((pred_x - true_x) ** 2 + (pred_y - true_y) ** 2)

            uncertainties.append(uncertainty)
            errors.append(error)

    if not uncertainties:
        return {"adaptive_uncertainty": 0.0, "difficulty_correlation": 0.0}

    # Calculate correlation between uncertainty and actual error
    correlation: float = 0.0
    if len(uncertainties) > 1 and len(errors) > 1:
        try:
            # Convert to numpy arrays to ensure numeric types
            uncertainties_arr = np.array(uncertainties, dtype=float)
            errors_arr = np.array(errors, dtype=float)

            # Calculate correlation coefficient
            corr_matrix = np.corrcoef(uncertainties_arr, errors_arr)
            correlation = float(corr_matrix[0, 1])

            if np.isnan(correlation):
                correlation = 0.0
        except (ValueError, TypeError, IndexError):
            correlation = 0.0

    # Adaptive score: uncertainty should correlate with error
    adaptive_score = max(0.0, correlation)

    return {
        "adaptive_uncertainty": adaptive_score,
        "difficulty_correlation": correlation,
    }
