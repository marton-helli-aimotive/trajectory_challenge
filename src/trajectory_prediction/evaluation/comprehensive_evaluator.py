"""Comprehensive evaluation framework for trajectory prediction models."""

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..data.models import Trajectory
from ..models.base import PredictionResult, TrajectoryPredictor
from .metrics import ModelEvaluator, ade, fde
from .safety_metrics import (
    collision_probability,
    lateral_error,
    minimum_distance,
    safety_critical_scenarios,
    time_to_collision,
)


class ComprehensiveEvaluator:
    """Comprehensive evaluation framework for trajectory prediction models."""

    def __init__(self) -> None:
        self.base_evaluator = ModelEvaluator()
        self.results_cache: dict[str, Any] = {}

    def evaluate_model_comprehensive(
        self,
        model: TrajectoryPredictor,
        test_cases: list[tuple[Trajectory, list[tuple[float, float]]]],
        prediction_horizon: float,
        include_safety: bool = True,
        include_uncertainty: bool = True,
    ) -> dict[str, Any]:
        """Perform comprehensive evaluation of a trajectory prediction model.

        Args:
            model: Trained model to evaluate
            test_cases: List of (input_trajectory, ground_truth) pairs
            prediction_horizon: Prediction horizon in seconds
            include_safety: Whether to include safety-critical metrics
            include_uncertainty: Whether to include uncertainty metrics

        Returns:
            Comprehensive evaluation results
        """
        if not model.is_fitted:
            raise ValueError("Model must be fitted before evaluation")

        print(f"Comprehensive evaluation of {model.model_name}...")
        start_time = time.time()

        # Basic evaluation metrics
        base_performance = self.base_evaluator.evaluate_model(
            model, test_cases, prediction_horizon
        )

        # Initialize results with base metrics
        results = {
            "model_name": model.model_name,
            "model_type": model.config.model_type,
            "basic_metrics": {
                "rmse": base_performance.rmse,
                "ade": base_performance.ade,
                "fde": base_performance.fde,
                "training_time": base_performance.training_time,
                "inference_time": base_performance.inference_time,
            },
            "n_test_cases": len(test_cases),
            "n_successful": base_performance.metadata.get(
                "n_successful_predictions", 0
            ),
            "success_rate": base_performance.metadata.get("n_successful_predictions", 0)
            / max(1, len(test_cases)),
        }

        # Detailed per-prediction analysis
        detailed_results = self._evaluate_detailed_predictions(
            model, test_cases, prediction_horizon, include_safety, include_uncertainty
        )

        results.update(detailed_results)

        # Multi-horizon evaluation
        if prediction_horizon > 2.0:  # Only if horizon is long enough
            multi_horizon_results = self._evaluate_multi_horizon(
                model, test_cases, prediction_horizon
            )
            results["multi_horizon"] = multi_horizon_results

        # Scenario-based evaluation
        scenario_results = self._evaluate_by_scenario(
            model, test_cases, prediction_horizon
        )
        results["scenario_analysis"] = scenario_results

        # Overall evaluation time
        results["evaluation_time"] = time.time() - start_time

        return results

    def _evaluate_detailed_predictions(
        self,
        model: TrajectoryPredictor,
        test_cases: list[tuple[Trajectory, list[tuple[float, float]]]],
        prediction_horizon: float,
        include_safety: bool,
        include_uncertainty: bool,
    ) -> dict[str, Any]:
        """Evaluate individual predictions with detailed metrics."""
        all_predictions = []
        safety_metrics = []
        uncertainty_metrics = []

        for input_trajectory, ground_truth in test_cases:
            try:
                # Make prediction
                prediction = model.predict(input_trajectory, prediction_horizon)
                predicted_points = prediction.predicted_points

                if not predicted_points or not ground_truth:
                    continue

                # Basic metrics for this prediction
                pred_ade = ade(predicted_points, ground_truth)
                pred_fde = fde(predicted_points, ground_truth)

                # Track successful prediction
                pred_result = {
                    "trajectory_id": input_trajectory.trajectory_id,
                    "ade": pred_ade,
                    "fde": pred_fde,
                    "n_predicted_points": len(predicted_points),
                    "n_ground_truth_points": len(ground_truth),
                }

                all_predictions.append(pred_result)

                # Safety-critical metrics
                if include_safety:
                    safety_result = self._evaluate_safety_metrics(
                        predicted_points, ground_truth
                    )
                    safety_result["trajectory_id"] = input_trajectory.trajectory_id
                    safety_metrics.append(safety_result)

                # Uncertainty metrics
                if include_uncertainty and hasattr(prediction, "uncertainty"):
                    uncertainty_result = self._evaluate_uncertainty_metrics(
                        prediction, ground_truth
                    )
                    uncertainty_result["trajectory_id"] = input_trajectory.trajectory_id
                    uncertainty_metrics.append(uncertainty_result)

            except Exception as e:
                print(
                    f"Failed to evaluate trajectory {input_trajectory.trajectory_id}: {e}"
                )
                continue

        # Aggregate results
        results = {
            "detailed_predictions": all_predictions,
            "aggregated_metrics": self._aggregate_prediction_metrics(all_predictions),
        }

        if safety_metrics:
            results["safety_metrics"] = safety_metrics
            results["aggregated_safety"] = self._aggregate_safety_metrics(
                safety_metrics
            )

        if uncertainty_metrics:
            results["uncertainty_metrics"] = uncertainty_metrics
            results["aggregated_uncertainty"] = self._aggregate_uncertainty_metrics(
                uncertainty_metrics
            )

        return results

    def _evaluate_safety_metrics(
        self,
        predicted_points: list[tuple[float, float]],
        ground_truth: list[tuple[float, float]],
    ) -> dict[str, Any]:
        """Evaluate safety-critical metrics for a single prediction."""
        safety_result: dict[str, Any] = {}

        # Minimum distance
        pred_array = np.array(predicted_points)
        gt_array = np.array(ground_truth)
        min_dist = minimum_distance(pred_array, gt_array)
        safety_result["minimum_distance"] = min_dist

        # Time to collision
        ttc = time_to_collision(predicted_points, ground_truth)
        safety_result["time_to_collision"] = ttc if ttc is not None else float("inf")

        # Lateral error
        lateral_errors = lateral_error(predicted_points, ground_truth)
        safety_result["lateral_errors"] = lateral_errors
        safety_result["max_lateral_error"] = (
            max(lateral_errors) if lateral_errors else float("inf")
        )
        safety_result["mean_lateral_error"] = (
            float(np.mean(lateral_errors)) if lateral_errors else float("inf")
        )

        # Collision probability
        collision_prob = collision_probability(predicted_points, ground_truth)
        safety_result["collision_probability"] = collision_prob

        # Safety scenario analysis
        scenario_analysis = safety_critical_scenarios(predicted_points, ground_truth)
        safety_result["scenario_analysis"] = scenario_analysis

        return safety_result

    def _evaluate_uncertainty_metrics(
        self, _prediction: PredictionResult, _ground_truth: list[tuple[float, float]]
    ) -> dict[str, Any]:
        """Evaluate uncertainty quantification metrics for a single prediction."""
        uncertainty_result: dict[str, Any] = {}

        # For now, return basic uncertainty metrics
        # Uncertainty evaluation pending prediction structure clarification
        uncertainty_result["uncertainty_available"] = False
        uncertainty_result["note"] = (
            "Uncertainty evaluation pending prediction structure clarification"
        )

        return uncertainty_result

    def _evaluate_multi_horizon(
        self,
        model: TrajectoryPredictor,
        test_cases: list[tuple[Trajectory, list[tuple[float, float]]]],
        max_horizon: float,
    ) -> dict[str, Any]:
        """Evaluate model performance across different prediction horizons."""
        horizons = [1.0, 2.0, 3.0, 5.0]
        horizons = [h for h in horizons if h <= max_horizon]

        multi_horizon_results = {}

        for horizon in horizons:
            try:
                # Create test cases for this horizon
                horizon_test_cases = []
                for input_traj, ground_truth in test_cases:
                    # Truncate ground truth to match horizon
                    points_per_second = len(ground_truth) / max_horizon
                    horizon_points = int(horizon * points_per_second)
                    if horizon_points > 0:
                        truncated_gt = ground_truth[:horizon_points]
                        horizon_test_cases.append((input_traj, truncated_gt))

                if not horizon_test_cases:
                    continue

                # Evaluate at this horizon
                performance = self.base_evaluator.evaluate_model(
                    model, horizon_test_cases, horizon
                )

                multi_horizon_results[f"horizon_{horizon}s"] = {
                    "rmse": performance.rmse,
                    "ade": performance.ade,
                    "fde": performance.fde,
                    "n_test_cases": len(horizon_test_cases),
                }

            except Exception as e:
                print(f"Failed to evaluate horizon {horizon}s: {e}")
                continue

        return multi_horizon_results

    def _evaluate_by_scenario(
        self,
        model: TrajectoryPredictor,
        test_cases: list[tuple[Trajectory, list[tuple[float, float]]]],
        prediction_horizon: float,
    ) -> dict[str, Any]:
        """Evaluate model performance by different driving scenarios."""
        scenario_results = {}

        # Group trajectories by scenario characteristics
        scenarios: dict[str, list[tuple[Trajectory, list[tuple[float, float]]]]] = {
            "straight_driving": [],
            "lane_changes": [],
            "turns": [],
            "complex_maneuvers": [],
        }

        for input_traj, ground_truth in test_cases:
            # Simple scenario classification based on trajectory characteristics
            scenario_type = self._classify_trajectory_scenario(ground_truth)
            if scenario_type in scenarios:
                scenarios[scenario_type].append((input_traj, ground_truth))

        # Evaluate each scenario
        for scenario_name, scenario_cases in scenarios.items():
            if len(scenario_cases) >= 3:  # Need minimum cases for meaningful evaluation
                try:
                    performance = self.base_evaluator.evaluate_model(
                        model, scenario_cases, prediction_horizon
                    )
                    scenario_results[scenario_name] = {
                        "rmse": performance.rmse,
                        "ade": performance.ade,
                        "fde": performance.fde,
                        "n_test_cases": len(scenario_cases),
                        "success_rate": performance.metadata.get(
                            "n_successful_predictions", 0
                        )
                        / max(1, len(scenario_cases)),
                    }
                except Exception as e:
                    print(f"Failed to evaluate scenario {scenario_name}: {e}")

        return scenario_results

    def _classify_trajectory_scenario(
        self,
        ground_truth: list[tuple[float, float]],
    ) -> str:
        """Classify trajectory into driving scenario categories."""
        if len(ground_truth) < 3:
            return "complex_maneuvers"

        # Calculate trajectory characteristics
        points = np.array(ground_truth)

        # Calculate velocity changes
        velocities = np.diff(points, axis=0)
        speed_changes = np.linalg.norm(velocities, axis=1)

        # Calculate direction changes
        if len(velocities) > 1:
            direction_changes = []
            for i in range(len(velocities) - 1):
                v1, v2 = velocities[i], velocities[i + 1]
                if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                    cos_angle = np.dot(v1, v2) / (
                        np.linalg.norm(v1) * np.linalg.norm(v2)
                    )
                    cos_angle = np.clip(cos_angle, -1, 1)
                    angle = np.arccos(cos_angle)
                    direction_changes.append(angle)

            avg_direction_change = (
                np.mean(direction_changes) if direction_changes else 0
            )
        else:
            avg_direction_change = 0

        # Calculate lateral displacement
        lateral_displacement = np.std(points[:, 1]) if len(points) > 2 else 0

        # Classification logic
        if avg_direction_change > 0.3:  # ~17 degrees
            return "turns"
        elif lateral_displacement > 2.0:  # 2 meter lateral movement
            return "lane_changes"
        elif np.std(speed_changes) < 1.0:  # Consistent speed
            return "straight_driving"
        else:
            return "complex_maneuvers"

    def _aggregate_prediction_metrics(
        self, predictions: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Aggregate metrics across all predictions."""
        if not predictions:
            return {}

        # Extract metrics
        ades = [p["ade"] for p in predictions if np.isfinite(p["ade"])]
        fdes = [p["fde"] for p in predictions if np.isfinite(p["fde"])]

        return {
            "mean_ade": float(np.mean(ades)) if ades else float("inf"),
            "std_ade": float(np.std(ades)) if len(ades) > 1 else 0.0,
            "mean_fde": float(np.mean(fdes)) if fdes else float("inf"),
            "std_fde": float(np.std(fdes)) if len(fdes) > 1 else 0.0,
            "median_ade": float(np.median(ades)) if ades else float("inf"),
            "median_fde": float(np.median(fdes)) if fdes else float("inf"),
            "percentile_90_ade": float(np.percentile(ades, 90))
            if ades
            else float("inf"),
            "percentile_90_fde": float(np.percentile(fdes, 90))
            if fdes
            else float("inf"),
        }

    def _aggregate_safety_metrics(
        self, safety_metrics: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Aggregate safety metrics across all predictions."""
        if not safety_metrics:
            return {}

        # Extract safety metrics
        min_distances = [
            s["minimum_distance"]
            for s in safety_metrics
            if np.isfinite(s.get("minimum_distance", float("inf")))
        ]
        collision_probs = [
            s["collision_probability"]
            for s in safety_metrics
            if np.isfinite(s.get("collision_probability", 0))
        ]
        max_lateral_errors = [
            s["max_lateral_error"]
            for s in safety_metrics
            if np.isfinite(s.get("max_lateral_error", float("inf")))
        ]

        # Count critical scenarios
        critical_scenarios = sum(
            1
            for s in safety_metrics
            if s.get("scenario_analysis", {}).get("is_critical", False)
        )

        return {
            "mean_minimum_distance": float(np.mean(min_distances))
            if min_distances
            else float("inf"),
            "min_minimum_distance": float(np.min(min_distances))
            if min_distances
            else float("inf"),
            "mean_collision_probability": float(np.mean(collision_probs))
            if collision_probs
            else 0.0,
            "max_collision_probability": float(np.max(collision_probs))
            if collision_probs
            else 0.0,
            "mean_max_lateral_error": float(np.mean(max_lateral_errors))
            if max_lateral_errors
            else float("inf"),
            "critical_scenario_rate": critical_scenarios / len(safety_metrics)
            if safety_metrics
            else 0.0,
            "n_critical_scenarios": critical_scenarios,
        }

    def _aggregate_uncertainty_metrics(
        self, uncertainty_metrics: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Aggregate uncertainty metrics across all predictions."""
        if not uncertainty_metrics:
            return {}

        # Extract uncertainty metrics
        picp_values = [
            u["picp_95"]
            for u in uncertainty_metrics
            if "picp_95" in u and np.isfinite(u["picp_95"])
        ]
        calibration_errors = [
            u["calibration_error"]
            for u in uncertainty_metrics
            if "calibration_error" in u and np.isfinite(u["calibration_error"])
        ]

        return {
            "mean_picp_95": float(np.mean(picp_values))
            if picp_values
            else float("nan"),
            "mean_calibration_error": float(np.mean(calibration_errors))
            if calibration_errors
            else float("nan"),
            "n_with_uncertainty": len(
                [u for u in uncertainty_metrics if "error" not in u]
            ),
        }

    def compare_models_comprehensive(
        self,
        models: dict[str, TrajectoryPredictor],
        test_cases: list[tuple[Trajectory, list[tuple[float, float]]]],
        prediction_horizon: float,
    ) -> dict[str, Any]:
        """Comprehensive comparison of multiple models."""
        print("Comprehensive model comparison...")

        model_results = {}
        for model_name, model in models.items():
            try:
                results = self.evaluate_model_comprehensive(
                    model, test_cases, prediction_horizon
                )
                model_results[model_name] = results
            except Exception as e:
                print(f"Failed to evaluate {model_name}: {e}")
                model_results[model_name] = {"error": str(e)}

        # Create comparison summary
        comparison_summary = self._create_comparison_summary(model_results)

        return {
            "individual_results": model_results,
            "comparison_summary": comparison_summary,
            "n_test_cases": len(test_cases),
            "prediction_horizon": prediction_horizon,
        }

    def _create_comparison_summary(
        self, model_results: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        """Create a summary comparison of model results."""
        # Extract key metrics for comparison
        comparison_data = []

        for model_name, results in model_results.items():
            if "error" in results:
                continue

            basic_metrics = results.get("basic_metrics", {})
            safety = results.get("aggregated_safety", {})

            comparison_data.append(
                {
                    "model_name": model_name,
                    "model_type": results.get("model_type", "unknown"),
                    "ade": basic_metrics.get("ade", float("inf")),
                    "fde": basic_metrics.get("fde", float("inf")),
                    "rmse": basic_metrics.get("rmse", float("inf")),
                    "success_rate": results.get("success_rate", 0.0),
                    "min_distance": safety.get("mean_minimum_distance", float("inf")),
                    "collision_prob": safety.get("mean_collision_probability", 0.0),
                    "critical_rate": safety.get("critical_scenario_rate", 0.0),
                    "training_time": basic_metrics.get("training_time", 0.0),
                    "inference_time": basic_metrics.get("inference_time", 0.0),
                }
            )

        if not comparison_data:
            return {"error": "No successful model evaluations"}

        # Create DataFrame for easier analysis
        df = pd.DataFrame(comparison_data)

        # Rank models by different criteria
        rankings = {}

        # Accuracy ranking (lower is better for error metrics)
        df_finite = df[np.isfinite(df["ade"])]
        if not df_finite.empty:
            rankings["accuracy"] = df_finite.nsmallest(3, "ade")["model_name"].tolist()

        # Safety ranking (higher min_distance is better, lower collision_prob is better)
        df_safe = df[np.isfinite(df["min_distance"])]
        if not df_safe.empty:
            # Combine safety scores (normalize and weight)
            safety_score = (
                df_safe["min_distance"] / df_safe["min_distance"].max()
                - df_safe["collision_prob"]
                - df_safe["critical_rate"]
            )
            df_safe = df_safe.copy()
            df_safe["safety_score"] = safety_score
            rankings["safety"] = df_safe.nlargest(3, "safety_score")[
                "model_name"
            ].tolist()

        # Speed ranking (lower inference time is better)
        df_speed = df[df["inference_time"] > 0]
        if not df_speed.empty:
            rankings["speed"] = df_speed.nsmallest(3, "inference_time")[
                "model_name"
            ].tolist()

        # Overall ranking (balanced score)
        df_overall = df[np.isfinite(df["ade"]) & (df["success_rate"] > 0)]
        if not df_overall.empty:
            # Normalize metrics and create composite score
            ade_norm = 1 - (df_overall["ade"] - df_overall["ade"].min()) / (
                df_overall["ade"].max() - df_overall["ade"].min() + 1e-8
            )
            success_norm = df_overall["success_rate"]

            # Weighted composite score
            composite_score = (
                0.4 * ade_norm
                + 0.3 * success_norm
                + 0.3 * (1 - df_overall["collision_prob"])
            )
            df_overall = df_overall.copy()
            df_overall["composite_score"] = composite_score
            rankings["overall"] = df_overall.nlargest(3, "composite_score")[
                "model_name"
            ].tolist()

        return {
            "model_comparison_table": df.to_dict("records"),
            "rankings": rankings,
            "best_models": {
                "accuracy": rankings.get("accuracy", [None])[0],
                "safety": rankings.get("safety", [None])[0],
                "speed": rankings.get("speed", [None])[0],
                "overall": rankings.get("overall", [None])[0],
            },
            "summary_statistics": {
                "n_models_evaluated": len(comparison_data),
                "mean_ade": float(df["ade"][np.isfinite(df["ade"])].mean())
                if np.isfinite(df["ade"]).any()
                else float("nan"),
                "best_ade": float(df["ade"][np.isfinite(df["ade"])].min())
                if np.isfinite(df["ade"]).any()
                else float("nan"),
                "mean_success_rate": float(df["success_rate"].mean()),
            },
        }

    def generate_evaluation_report(
        self, comprehensive_results: dict[str, Any], output_file: str | None = None
    ) -> str:
        """Generate a human-readable evaluation report."""
        report_lines = []

        report_lines.append("=" * 80)
        report_lines.append("COMPREHENSIVE TRAJECTORY PREDICTION EVALUATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")

        if "individual_results" in comprehensive_results:
            # Multi-model comparison report
            comparison = comprehensive_results.get("comparison_summary", {})

            report_lines.append("MULTI-MODEL COMPARISON SUMMARY")
            report_lines.append("-" * 40)

            best_models = comparison.get("best_models", {})
            if best_models:
                report_lines.append("Best Models by Category:")
                for category, model in best_models.items():
                    if model:
                        report_lines.append(f"  {category.title()}: {model}")
                report_lines.append("")

            # Model comparison table
            model_table = comparison.get("model_comparison_table", [])
            if model_table:
                report_lines.append("Model Performance Comparison:")
                report_lines.append(
                    f"{'Model':<20} {'ADE':<8} {'FDE':<8} {'Success%':<10} {'Safety':<10}"
                )
                report_lines.append("-" * 60)

                for model_data in model_table:
                    name = model_data["model_name"][:19]
                    ade = (
                        f"{model_data['ade']:.3f}"
                        if np.isfinite(model_data["ade"])
                        else "inf"
                    )
                    fde = (
                        f"{model_data['fde']:.3f}"
                        if np.isfinite(model_data["fde"])
                        else "inf"
                    )
                    success = f"{model_data['success_rate'] * 100:.1f}%"
                    safety = (
                        f"{model_data['min_distance']:.2f}m"
                        if np.isfinite(model_data["min_distance"])
                        else "inf"
                    )

                    report_lines.append(
                        f"{name:<20} {ade:<8} {fde:<8} {success:<10} {safety:<10}"
                    )
        else:
            # Single model report
            model_name = comprehensive_results.get("model_name", "Unknown Model")
            report_lines.append(f"MODEL EVALUATION: {model_name}")
            report_lines.append("-" * 40)

            basic_metrics = comprehensive_results.get("basic_metrics", {})
            if basic_metrics:
                report_lines.append("Basic Performance Metrics:")
                report_lines.append(f"  RMSE: {basic_metrics.get('rmse', 'N/A'):.4f}")
                report_lines.append(f"  ADE:  {basic_metrics.get('ade', 'N/A'):.4f}")
                report_lines.append(f"  FDE:  {basic_metrics.get('fde', 'N/A'):.4f}")
                report_lines.append(
                    f"  Success Rate: {comprehensive_results.get('success_rate', 0) * 100:.1f}%"
                )
                report_lines.append("")

            safety_metrics = comprehensive_results.get("aggregated_safety", {})
            if safety_metrics:
                report_lines.append("Safety-Critical Metrics:")
                report_lines.append(
                    f"  Mean Minimum Distance: {safety_metrics.get('mean_minimum_distance', 'N/A'):.3f}m"
                )
                report_lines.append(
                    f"  Mean Collision Probability: {safety_metrics.get('mean_collision_probability', 'N/A'):.4f}"
                )
                report_lines.append(
                    f"  Critical Scenario Rate: {safety_metrics.get('critical_scenario_rate', 'N/A') * 100:.1f}%"
                )
                report_lines.append("")

        report_lines.append("=" * 80)

        report_text = "\n".join(report_lines)

        if output_file:
            with Path(output_file).open("w") as f:
                f.write(report_text)
            print(f"Evaluation report saved to {output_file}")

        return report_text
