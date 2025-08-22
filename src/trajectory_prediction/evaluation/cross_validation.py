"""Time-series cross-validation for trajectory prediction models."""

import time
from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd

from ..data.models import Trajectory
from ..models.base import ModelPerformance, TrajectoryPredictor
from .metrics import ModelEvaluator


class TimeSeriesCrossValidator:
    """Cross-validator that respects temporal ordering of trajectory data."""

    def __init__(
        self,
        n_splits: int = 5,
        test_size: float = 0.2,
        gap: float = 0.0,
        prediction_horizon: float = 2.0,
    ):
        """Initialize time series cross-validator.

        Args:
            n_splits: Number of cross-validation splits
            test_size: Fraction of data to use for testing in each split
            gap: Time gap between train and test sets (seconds)
            prediction_horizon: How far into future to predict (seconds)
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.prediction_horizon = prediction_horizon

    def split(
        self, trajectories: list[Trajectory]
    ) -> Iterator[tuple[list[Trajectory], list[Trajectory]]]:
        """Generate train/test splits respecting temporal order.

        Args:
            trajectories: List of trajectories to split

        Yields:
            Tuples of (train_trajectories, test_trajectories)
        """
        if not trajectories:
            return

        # Sort trajectories by their start time
        sorted_trajectories = sorted(trajectories, key=lambda t: t.points[0].timestamp)

        # Get time range
        start_time = sorted_trajectories[0].points[0].timestamp
        end_time = sorted_trajectories[-1].points[-1].timestamp
        total_duration = end_time - start_time

        # Calculate split parameters
        test_duration = total_duration * self.test_size
        step_size = (total_duration - test_duration) / self.n_splits

        for i in range(self.n_splits):
            # Calculate time boundaries for this split
            train_start = start_time + i * step_size
            train_end = train_start + (total_duration - test_duration - i * step_size)
            test_start = train_end + self.gap
            test_end = test_start + test_duration

            # Split trajectories based on time
            train_trajectories = []
            test_trajectories = []

            for trajectory in sorted_trajectories:
                traj_start = trajectory.points[0].timestamp
                traj_end = trajectory.points[-1].timestamp

                # Check if trajectory overlaps with train period
                if traj_start < train_end and traj_end > train_start:
                    # Filter points within train period
                    train_points = [
                        p
                        for p in trajectory.points
                        if train_start <= p.timestamp <= train_end
                    ]
                    if len(train_points) >= 2:
                        train_traj = Trajectory(
                            trajectory_id=f"{trajectory.trajectory_id}_train_{i}",
                            vehicle=trajectory.vehicle,
                            points=train_points,
                            dataset_name=trajectory.dataset_name,
                            completeness_score=None,
                            temporal_consistency_score=None,
                            spatial_accuracy_score=None,
                            smoothness_score=None,
                        )
                        train_trajectories.append(train_traj)

                # Check if trajectory overlaps with test period
                if traj_start < test_end and traj_end > test_start:
                    # Filter points within test period
                    test_points = [
                        p
                        for p in trajectory.points
                        if test_start <= p.timestamp <= test_end
                    ]
                    if len(test_points) >= 2:
                        test_traj = Trajectory(
                            trajectory_id=f"{trajectory.trajectory_id}_test_{i}",
                            vehicle=trajectory.vehicle,
                            points=test_points,
                            dataset_name=trajectory.dataset_name,
                            completeness_score=None,
                            temporal_consistency_score=None,
                            spatial_accuracy_score=None,
                            smoothness_score=None,
                        )
                        test_trajectories.append(test_traj)

            if train_trajectories and test_trajectories:
                yield train_trajectories, test_trajectories


class CrossValidationRunner:
    """Runner for cross-validation experiments."""

    def __init__(self, cv: TimeSeriesCrossValidator | None = None):
        """Initialize cross-validation runner.

        Args:
            cv: Cross-validator instance. If None, creates default.
        """
        self.cv = cv if cv is not None else TimeSeriesCrossValidator()
        self.evaluator = ModelEvaluator()

    def run_cross_validation(
        self,
        model: TrajectoryPredictor,
        trajectories: list[Trajectory],
        refit_per_fold: bool = True,
    ) -> dict[str, Any]:
        """Run cross-validation for a single model.

        Args:
            model: Model to evaluate
            trajectories: Dataset trajectories
            refit_per_fold: Whether to refit model for each fold

        Returns:
            Cross-validation results
        """
        fold_results = []
        fold_performances = []

        print(
            f"Running {self.cv.n_splits}-fold cross-validation for {model.model_name}..."
        )

        for fold_idx, (train_trajs, test_trajs) in enumerate(
            self.cv.split(trajectories)
        ):
            print(
                f"  Fold {fold_idx + 1}/{self.cv.n_splits} - Train: {len(train_trajs)}, Test: {len(test_trajs)}"
            )

            start_time = time.time()

            try:
                # Create a fresh model instance for this fold if requested
                if refit_per_fold:
                    # Create new model with same config
                    from ..models.factory import create_model

                    # Use the original model_type from config instead of deriving from class name
                    fold_model = create_model(model.config.model_type, model.config)
                else:
                    fold_model = model

                # Train model on fold training data
                if not fold_model.is_fitted or refit_per_fold:
                    fold_model.fit(train_trajs)

                # Prepare test cases
                test_cases = self.evaluator.prepare_test_data(
                    test_trajs, self.cv.prediction_horizon
                )

                if not test_cases:
                    print(f"    No valid test cases for fold {fold_idx + 1}")
                    continue

                # Evaluate model
                performance = self.evaluator.evaluate_model(
                    fold_model, test_cases, self.cv.prediction_horizon
                )

                fold_time = time.time() - start_time

                fold_result = {
                    "fold": fold_idx,
                    "n_train": len(train_trajs),
                    "n_test": len(test_trajs),
                    "n_test_cases": len(test_cases),
                    "performance": performance,
                    "fold_time": fold_time,
                }

                fold_results.append(fold_result)
                fold_performances.append(performance)

                print(
                    f"    ADE: {performance.ade:.3f}, FDE: {performance.fde:.3f}, RMSE: {performance.rmse:.3f}"
                )

            except Exception as e:
                print(f"    Fold {fold_idx + 1} failed: {e}")
                continue

        if not fold_performances:
            raise ValueError("All cross-validation folds failed")

        # Aggregate results across folds
        aggregated_results = self._aggregate_cv_results(fold_performances)

        return {
            "model_name": model.model_name,
            "model_type": model.config.model_type,
            "cv_config": {
                "n_splits": self.cv.n_splits,
                "test_size": self.cv.test_size,
                "prediction_horizon": self.cv.prediction_horizon,
                "refit_per_fold": refit_per_fold,
            },
            "aggregated_performance": aggregated_results,
            "fold_results": fold_results,
        }

    def _aggregate_cv_results(
        self, performances: list[ModelPerformance]
    ) -> dict[str, float]:
        """Aggregate cross-validation results across folds."""
        # Extract finite metrics
        rmse_scores = [p.rmse for p in performances if np.isfinite(p.rmse)]
        ade_scores = [p.ade for p in performances if np.isfinite(p.ade)]
        fde_scores = [p.fde for p in performances if np.isfinite(p.fde)]
        training_times = [
            p.training_time for p in performances if np.isfinite(p.training_time)
        ]
        inference_times = [
            p.inference_time for p in performances if np.isfinite(p.inference_time)
        ]

        return {
            "mean_rmse": float(np.mean(rmse_scores)) if rmse_scores else float("inf"),
            "std_rmse": float(np.std(rmse_scores)) if rmse_scores else 0.0,
            "mean_ade": float(np.mean(ade_scores)) if ade_scores else float("inf"),
            "std_ade": float(np.std(ade_scores)) if ade_scores else 0.0,
            "mean_fde": float(np.mean(fde_scores)) if fde_scores else float("inf"),
            "std_fde": float(np.std(fde_scores)) if fde_scores else 0.0,
            "mean_training_time": float(np.mean(training_times))
            if training_times
            else 0.0,
            "mean_inference_time": float(np.mean(inference_times))
            if inference_times
            else 0.0,
            "n_successful_folds": len(rmse_scores),
            "total_folds": len(performances),
        }

    def compare_models_cv(
        self,
        models: dict[str, TrajectoryPredictor],
        trajectories: list[Trajectory],
        refit_per_fold: bool = True,
    ) -> pd.DataFrame:
        """Compare multiple models using cross-validation.

        Args:
            models: Dictionary of models to compare
            trajectories: Dataset trajectories
            refit_per_fold: Whether to refit models for each fold

        Returns:
            DataFrame with comparison results
        """
        cv_results = []

        for model_name, model in models.items():
            try:
                result = self.run_cross_validation(model, trajectories, refit_per_fold)
                cv_results.append(result)
            except Exception as e:
                print(f"Cross-validation failed for {model_name}: {e}")
                # Add failed result
                cv_results.append(
                    {
                        "model_name": model_name,
                        "model_type": getattr(model.config, "model_type", "unknown"),
                        "aggregated_performance": {
                            "mean_rmse": float("inf"),
                            "std_rmse": 0.0,
                            "mean_ade": float("inf"),
                            "std_ade": 0.0,
                            "mean_fde": float("inf"),
                            "std_fde": 0.0,
                            "mean_training_time": 0.0,
                            "mean_inference_time": 0.0,
                            "n_successful_folds": 0,
                            "total_folds": self.cv.n_splits,
                        },
                    }
                )

        # Convert to DataFrame for easy analysis
        comparison_data = []
        for result in cv_results:
            perf = result["aggregated_performance"]
            comparison_data.append(
                {
                    "model_name": result["model_name"],
                    "model_type": result.get("model_type", "unknown"),
                    "mean_rmse": perf["mean_rmse"],
                    "std_rmse": perf["std_rmse"],
                    "mean_ade": perf["mean_ade"],
                    "std_ade": perf["std_ade"],
                    "mean_fde": perf["mean_fde"],
                    "std_fde": perf["std_fde"],
                    "mean_training_time": perf["mean_training_time"],
                    "mean_inference_time": perf["mean_inference_time"],
                    "success_rate": perf["n_successful_folds"]
                    / max(1, perf["total_folds"]),
                }
            )

        df = pd.DataFrame(comparison_data)

        # Sort by mean ADE (best first, excluding infinite values)
        finite_mask = np.isfinite(df["mean_ade"])
        if finite_mask.any():
            df_finite = df[finite_mask].sort_values("mean_ade")
            df_infinite = df[~finite_mask]
            df = pd.concat([df_finite, df_infinite], ignore_index=True)

        return df
