"""Initial evaluation metrics for trajectory prediction models."""

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from ..data.models import Trajectory
from ..models.base import ModelPerformance, PredictionResult, TrajectoryPredictor


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Square Error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def ade(
    predicted_points: list[tuple[float, float]],
    ground_truth_points: list[tuple[float, float]],
) -> float:
    """Calculate Average Displacement Error.

    Args:
        predicted_points: List of (x, y) predicted coordinates
        ground_truth_points: List of (x, y) ground truth coordinates

    Returns:
        Average displacement error over all points
    """
    if not predicted_points or not ground_truth_points:
        return float("inf")

    # Take minimum length to handle different prediction horizons
    min_len = min(len(predicted_points), len(ground_truth_points))

    total_displacement = 0.0
    for i in range(min_len):
        pred_x, pred_y = predicted_points[i]
        true_x, true_y = ground_truth_points[i]
        displacement = np.sqrt((pred_x - true_x) ** 2 + (pred_y - true_y) ** 2)
        total_displacement += displacement

    return total_displacement / min_len if min_len > 0 else float("inf")


def fde(
    predicted_points: list[tuple[float, float]],
    ground_truth_points: list[tuple[float, float]],
) -> float:
    """Calculate Final Displacement Error.

    Args:
        predicted_points: List of (x, y) predicted coordinates
        ground_truth_points: List of (x, y) ground truth coordinates

    Returns:
        Displacement error at the final prediction point
    """
    if not predicted_points or not ground_truth_points:
        return float("inf")

    # Compare final points
    pred_x, pred_y = predicted_points[-1]
    true_x, true_y = ground_truth_points[-1]

    return float(np.sqrt((pred_x - true_x) ** 2 + (pred_y - true_y) ** 2))


def evaluate_single_prediction(
    prediction: PredictionResult, ground_truth: list[tuple[float, float]]
) -> dict[str, float]:
    """Evaluate a single prediction against ground truth.

    Args:
        prediction: Model prediction result
        ground_truth: Ground truth trajectory points

    Returns:
        Dictionary of evaluation metrics
    """
    metrics = {}

    predicted_points = prediction.predicted_points

    if not predicted_points or not ground_truth:
        return {
            "rmse": float("inf"),
            "ade": float("inf"),
            "fde": float("inf"),
            "num_predicted_points": len(predicted_points),
            "num_ground_truth_points": len(ground_truth),
        }

    # Calculate ADE and FDE
    metrics["ade"] = ade(predicted_points, ground_truth)
    metrics["fde"] = fde(predicted_points, ground_truth)

    # Calculate RMSE (coordinate-wise)
    min_len = min(len(predicted_points), len(ground_truth))
    if min_len > 0:
        pred_coords = np.array(predicted_points[:min_len])
        true_coords = np.array(ground_truth[:min_len])
        metrics["rmse"] = rmse(true_coords.flatten(), pred_coords.flatten())
    else:
        metrics["rmse"] = float("inf")

    # Additional metrics
    metrics["num_predicted_points"] = len(predicted_points)
    metrics["num_ground_truth_points"] = len(ground_truth)
    metrics["prediction_horizon"] = prediction.prediction_horizon

    # Include timing information if available
    if "inference_time" in prediction.metadata:
        metrics["inference_time"] = prediction.metadata["inference_time"]

    return metrics


class ModelEvaluator:
    """Evaluator for trajectory prediction models."""

    def __init__(self) -> None:
        self.results_cache: dict[str, dict[str, Any]] = {}

    def prepare_test_data(
        self, trajectories: list[Trajectory], prediction_horizon: float
    ) -> list[tuple[Trajectory, list[tuple[float, float]]]]:
        """Prepare test data by splitting trajectories into input and ground truth.

        Args:
            trajectories: List of complete trajectories
            prediction_horizon: How far into future to predict (seconds)

        Returns:
            List of (input_trajectory, ground_truth_points) tuples
        """
        test_cases = []

        for trajectory in trajectories:
            points = trajectory.points
            if len(points) < 5:  # Need minimum points for meaningful evaluation
                continue

            # Find split point based on prediction horizon
            last_timestamp = points[-1].timestamp
            split_timestamp = last_timestamp - prediction_horizon

            # Find the split index
            split_idx = None
            for i, point in enumerate(points):
                if point.timestamp >= split_timestamp:
                    split_idx = i
                    break

            if split_idx is None or split_idx < 2:
                continue  # Not enough input points

            # Create input trajectory (up to split point)
            input_points = points[:split_idx]
            input_trajectory = Trajectory(
                trajectory_id=f"{trajectory.trajectory_id}_test",
                vehicle=trajectory.vehicle,
                points=input_points,
                dataset_name=trajectory.dataset_name,
                completeness_score=None,
                temporal_consistency_score=None,
                spatial_accuracy_score=None,
                smoothness_score=None,
            )

            # Create ground truth (after split point)
            ground_truth_points = [(p.x, p.y) for p in points[split_idx:]]

            if ground_truth_points:  # Only add if we have ground truth
                test_cases.append((input_trajectory, ground_truth_points))

        return test_cases

    def evaluate_model(
        self,
        model: TrajectoryPredictor,
        test_cases: list[tuple[Trajectory, list[tuple[float, float]]]],
        prediction_horizon: float,
    ) -> ModelPerformance:
        """Evaluate a model on test cases.

        Args:
            model: Trained model to evaluate
            test_cases: List of (input_trajectory, ground_truth) pairs
            prediction_horizon: Prediction horizon in seconds

        Returns:
            Model performance metrics
        """
        if not model.is_fitted:
            raise ValueError("Model must be fitted before evaluation")

        start_time = time.time()

        all_metrics = []
        total_inference_time = 0.0

        for input_trajectory, ground_truth in test_cases:
            try:
                # Make prediction
                prediction = model.predict(input_trajectory, prediction_horizon)

                # Evaluate prediction
                metrics = evaluate_single_prediction(prediction, ground_truth)
                all_metrics.append(metrics)

                # Track inference time
                if "inference_time" in metrics:
                    total_inference_time += metrics["inference_time"]

            except Exception as e:
                # Handle prediction failures gracefully
                print(
                    f"Prediction failed for trajectory {input_trajectory.trajectory_id}: {e}"
                )
                all_metrics.append(
                    {
                        "rmse": float("inf"),
                        "ade": float("inf"),
                        "fde": float("inf"),
                        "inference_time": 0.0,
                    }
                )

        if not all_metrics:
            # No successful predictions
            return ModelPerformance(
                rmse=float("inf"),
                ade=float("inf"),
                fde=float("inf"),
                training_time=getattr(model, "_training_time", 0.0),
                inference_time=0.0,
                metadata={
                    "n_test_cases": len(test_cases),
                    "n_successful_predictions": 0,
                    "model_name": model.model_name,
                },
            )

        # Aggregate metrics (ignore infinite values)
        finite_rmse = [m["rmse"] for m in all_metrics if np.isfinite(m["rmse"])]
        finite_ade = [m["ade"] for m in all_metrics if np.isfinite(m["ade"])]
        finite_fde = [m["fde"] for m in all_metrics if np.isfinite(m["fde"])]

        avg_rmse = float(np.mean(finite_rmse)) if finite_rmse else float("inf")
        avg_ade = float(np.mean(finite_ade)) if finite_ade else float("inf")
        avg_fde = float(np.mean(finite_fde)) if finite_fde else float("inf")

        evaluation_time = time.time() - start_time

        return ModelPerformance(
            rmse=avg_rmse,
            ade=avg_ade,
            fde=avg_fde,
            training_time=getattr(model, "_training_time", 0.0),
            inference_time=total_inference_time / len(test_cases),
            metadata={
                "n_test_cases": len(test_cases),
                "n_successful_predictions": len(finite_rmse),
                "evaluation_time": evaluation_time,
                "model_name": model.model_name,
                "rmse_std": float(np.std(finite_rmse)) if finite_rmse else 0.0,
                "ade_std": float(np.std(finite_ade)) if finite_ade else 0.0,
                "fde_std": float(np.std(finite_fde)) if finite_fde else 0.0,
            },
        )

    def compare_models(
        self,
        models: dict[str, TrajectoryPredictor],
        test_cases: list[tuple[Trajectory, list[tuple[float, float]]]],
        prediction_horizon: float,
    ) -> pd.DataFrame:
        """Compare multiple models on the same test cases.

        Args:
            models: Dictionary mapping model names to model instances
            test_cases: List of test cases
            prediction_horizon: Prediction horizon in seconds

        Returns:
            DataFrame with comparison results
        """
        results = []

        for model_name, model in models.items():
            print(f"Evaluating {model_name}...")

            try:
                performance = self.evaluate_model(model, test_cases, prediction_horizon)

                result = {
                    "model_name": model_name,
                    "model_type": model.config.model_type,
                    "rmse": performance.rmse,
                    "ade": performance.ade,
                    "fde": performance.fde,
                    "training_time": performance.training_time,
                    "inference_time": performance.inference_time,
                    "n_test_cases": performance.metadata.get("n_test_cases", 0),
                    "n_successful": performance.metadata.get(
                        "n_successful_predictions", 0
                    ),
                    "success_rate": performance.metadata.get(
                        "n_successful_predictions", 0
                    )
                    / max(1, performance.metadata.get("n_test_cases", 1)),
                }

                results.append(result)

            except Exception as e:
                print(f"Evaluation failed for {model_name}: {e}")
                # Add failed result
                results.append(
                    {
                        "model_name": model_name,
                        "model_type": getattr(model.config, "model_type", "unknown"),
                        "rmse": float("inf"),
                        "ade": float("inf"),
                        "fde": float("inf"),
                        "training_time": 0.0,
                        "inference_time": 0.0,
                        "n_test_cases": len(test_cases),
                        "n_successful": 0,
                        "success_rate": 0.0,
                    }
                )

        df = pd.DataFrame(results)

        # Sort by ADE (best first, excluding infinite values)
        finite_mask = np.isfinite(df["ade"])
        if finite_mask.any():
            df_finite = df[finite_mask].sort_values("ade")
            df_infinite = df[~finite_mask]
            df = pd.concat([df_finite, df_infinite], ignore_index=True)

        return df
