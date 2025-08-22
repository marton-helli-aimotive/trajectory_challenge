"""Ensemble methods for combining multiple trajectory predictors."""

from typing import Any

import numpy as np
import pandas as pd

from ..data.models import Trajectory
from .base import ModelConfig, PredictionResult, TrajectoryPredictor


class EnsemblePredictor(TrajectoryPredictor):
    """Ensemble predictor that combines multiple models with uncertainty quantification."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._predictors: list[TrajectoryPredictor] = []
        self._weights: list[float] = []
        self._predictor_configs: list[ModelConfig] = []

        # Extract ensemble configuration
        params = config.hyperparameters
        self.combination_method = params.get("combination_method", "weighted_average")
        self.weight_update_method = params.get(
            "weight_update_method", "performance_based"
        )

    def add_predictor(
        self, predictor: TrajectoryPredictor, weight: float = 1.0
    ) -> None:
        """Add a predictor to the ensemble."""
        self._predictors.append(predictor)
        self._weights.append(weight)
        self._predictor_configs.append(predictor.config)

    def fit(
        self, trajectories: list[Trajectory], features: pd.DataFrame | None = None
    ) -> "EnsemblePredictor":
        """Train all predictors in the ensemble."""
        if not self._predictors:
            raise ValueError("No predictors added to ensemble")

        # Train each predictor
        for predictor in self._predictors:
            if not predictor.is_fitted:
                predictor.fit(trajectories, features)

        # Update weights based on validation performance
        if self.weight_update_method == "performance_based":
            self._update_weights_by_performance(trajectories)

        self.is_fitted = True
        return self

    def predict(
        self,
        trajectory: Trajectory,
        prediction_horizon: float,
        features: pd.DataFrame | None = None,
    ) -> PredictionResult:
        """Predict using ensemble of models."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")

        if not self._predictors:
            raise ValueError("No predictors in ensemble")

        # Get predictions from all models
        predictions = []
        confidences = []

        for predictor in self._predictors:
            pred_result = predictor.predict(trajectory, prediction_horizon, features)
            predictions.append(pred_result)
            if pred_result.confidence_scores:
                confidences.append(np.mean(pred_result.confidence_scores))
            else:
                confidences.append(0.5)  # Default confidence

        # Combine predictions based on method
        if self.combination_method == "weighted_average":
            combined_result = self._weighted_average_combination(
                predictions, trajectory.trajectory_id, prediction_horizon
            )
        elif self.combination_method == "dynamic_weighting":
            combined_result = self._dynamic_weighting_combination(
                predictions, confidences, trajectory.trajectory_id, prediction_horizon
            )
        elif self.combination_method == "uncertainty_weighting":
            combined_result = self._uncertainty_weighting_combination(
                predictions, trajectory.trajectory_id, prediction_horizon
            )
        else:
            combined_result = self._simple_average_combination(
                predictions, trajectory.trajectory_id, prediction_horizon
            )

        # Add ensemble metadata
        combined_result.metadata.update(
            {
                "ensemble_method": self.combination_method,
                "n_predictors": len(self._predictors),
                "predictor_types": [
                    pred.config.model_type for pred in self._predictors
                ],
                "individual_confidences": confidences,
            }
        )

        return combined_result

    def _weighted_average_combination(
        self,
        predictions: list[PredictionResult],
        trajectory_id: str,
        prediction_horizon: float,
    ) -> PredictionResult:
        """Combine predictions using weighted average."""
        if not predictions:
            return PredictionResult(
                trajectory_id=trajectory_id,
                predicted_points=[],
                prediction_horizon=prediction_horizon,
                confidence_scores=[],
            )

        # Normalize weights
        total_weight = sum(self._weights)
        normalized_weights = [w / total_weight for w in self._weights]

        # Find maximum trajectory length
        max_length = max(len(pred.predicted_points) for pred in predictions)

        combined_points = []
        combined_confidences = []

        for i in range(max_length):
            weighted_x = 0.0
            weighted_y = 0.0
            weighted_confidence = 0.0
            total_available_weight = 0.0

            for _, (pred, weight) in enumerate(zip(predictions, normalized_weights)):
                if i < len(pred.predicted_points):
                    x, y = pred.predicted_points[i]
                    weighted_x += weight * x
                    weighted_y += weight * y

                    if pred.confidence_scores and i < len(pred.confidence_scores):
                        weighted_confidence += weight * pred.confidence_scores[i]
                    else:
                        weighted_confidence += weight * 0.5

                    total_available_weight += weight

            if total_available_weight > 0:
                combined_points.append(
                    (
                        weighted_x / total_available_weight,
                        weighted_y / total_available_weight,
                    )
                )
                combined_confidences.append(
                    weighted_confidence / total_available_weight
                )

        return PredictionResult(
            trajectory_id=trajectory_id,
            predicted_points=combined_points,
            prediction_horizon=prediction_horizon,
            confidence_scores=combined_confidences,
            metadata={"combination_weights": normalized_weights},
        )

    def _dynamic_weighting_combination(
        self,
        predictions: list[PredictionResult],
        confidences: list[float],
        trajectory_id: str,
        prediction_horizon: float,
    ) -> PredictionResult:
        """Combine predictions using dynamic confidence-based weighting."""
        # Use confidence scores as weights
        dynamic_weights = [
            max(conf, 0.1) for conf in confidences
        ]  # Minimum weight of 0.1
        total_weight = sum(dynamic_weights)
        dynamic_weights = [w / total_weight for w in dynamic_weights]

        # Temporarily override weights for this prediction
        original_weights = self._weights.copy()
        self._weights = dynamic_weights

        result = self._weighted_average_combination(
            predictions, trajectory_id, prediction_horizon
        )
        result.metadata["dynamic_weights"] = dynamic_weights

        # Restore original weights
        self._weights = original_weights

        return result

    def _uncertainty_weighting_combination(
        self,
        predictions: list[PredictionResult],
        trajectory_id: str,
        prediction_horizon: float,
    ) -> PredictionResult:
        """Combine predictions by weighting inversely to uncertainty."""
        uncertainties = []

        for pred in predictions:
            if pred.confidence_scores:
                # Higher confidence = lower uncertainty
                avg_confidence = np.mean(pred.confidence_scores)
                uncertainty = 1.0 - avg_confidence
            else:
                uncertainty = 0.5  # Default uncertainty

            uncertainties.append(max(uncertainty, 0.1))  # Minimum uncertainty

        # Invert uncertainties to get weights (lower uncertainty = higher weight)
        inv_uncertainties = [1.0 / u for u in uncertainties]
        total_weight = sum(inv_uncertainties)
        uncertainty_weights = [w / total_weight for w in inv_uncertainties]

        # Temporarily override weights
        original_weights = self._weights.copy()
        self._weights = uncertainty_weights

        result = self._weighted_average_combination(
            predictions, trajectory_id, prediction_horizon
        )
        result.metadata["uncertainty_weights"] = uncertainty_weights

        # Restore original weights
        self._weights = original_weights

        return result

    def _simple_average_combination(
        self,
        predictions: list[PredictionResult],
        trajectory_id: str,
        prediction_horizon: float,
    ) -> PredictionResult:
        """Simple unweighted average of predictions."""
        equal_weights = [1.0 / len(predictions)] * len(predictions)

        # Temporarily override weights
        original_weights = self._weights.copy()
        self._weights = equal_weights

        result = self._weighted_average_combination(
            predictions, trajectory_id, prediction_horizon
        )

        # Restore original weights
        self._weights = original_weights

        return result

    def _update_weights_by_performance(
        self, validation_trajectories: list[Trajectory]
    ) -> None:
        """Update predictor weights based on validation performance."""
        if len(validation_trajectories) < 2:
            return

        # Use a subset for validation
        val_size = min(10, len(validation_trajectories) // 2)
        val_trajectories = validation_trajectories[:val_size]

        predictor_errors = []

        for predictor in self._predictors:
            total_error = 0.0
            valid_predictions = 0

            for traj in val_trajectories:
                if len(traj.points) < 2:
                    continue

                # Use first part for prediction, last part for validation
                split_idx = len(traj.points) // 2
                input_points = traj.points[:split_idx]
                true_points = traj.points[split_idx:]

                if not input_points or not true_points:
                    continue

                # Create input trajectory
                input_traj = Trajectory(
                    trajectory_id=traj.trajectory_id + "_input",
                    vehicle=traj.vehicle,
                    points=input_points,
                    dataset_name=traj.dataset_name,
                    completeness_score=traj.completeness_score,
                    temporal_consistency_score=traj.temporal_consistency_score,
                    spatial_accuracy_score=traj.spatial_accuracy_score,
                    smoothness_score=traj.smoothness_score,
                )

                # Predict
                try:
                    pred_result = predictor.predict(input_traj, prediction_horizon=1.0)

                    if pred_result.predicted_points and true_points:
                        # Calculate error against first true point
                        pred_x, pred_y = pred_result.predicted_points[0]
                        true_x, true_y = true_points[0].x, true_points[0].y
                        error = np.sqrt((pred_x - true_x) ** 2 + (pred_y - true_y) ** 2)
                        total_error += error
                        valid_predictions += 1

                except Exception:
                    # Skip failed predictions
                    continue

            avg_error = (
                total_error / valid_predictions
                if valid_predictions > 0
                else float("inf")
            )
            predictor_errors.append(avg_error)

        # Update weights inversely proportional to error
        if all(error != float("inf") for error in predictor_errors):
            inv_errors = [1.0 / (error + 1e-6) for error in predictor_errors]
            total_inv_error = sum(inv_errors)
            self._weights = [w / total_inv_error for w in inv_errors]

    def get_feature_importance(self) -> dict[str, float] | None:
        """Get aggregated feature importance from ensemble."""
        importance_dicts = []

        for predictor in self._predictors:
            importance = predictor.get_feature_importance()
            if importance:
                importance_dicts.append(importance)

        if not importance_dicts:
            return None

        # Aggregate feature importances
        all_features: set[str] = set()
        for imp_dict in importance_dicts:
            all_features.update(imp_dict.keys())

        aggregated_importance = {}
        for feature in all_features:
            weighted_importance = 0.0
            total_weight = 0.0

            for i, imp_dict in enumerate(importance_dicts):
                if feature in imp_dict:
                    weight = self._weights[i] if i < len(self._weights) else 1.0
                    weighted_importance += weight * imp_dict[feature]
                    total_weight += weight

            if total_weight > 0:
                aggregated_importance[feature] = weighted_importance / total_weight

        return aggregated_importance

    def predict_with_individual_results(
        self,
        trajectory: Trajectory,
        prediction_horizon: float,
        features: pd.DataFrame | None = None,
    ) -> tuple[PredictionResult, list[PredictionResult]]:
        """Get both ensemble prediction and individual predictor results."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")

        # Get individual predictions
        individual_predictions = []
        for predictor in self._predictors:
            pred_result = predictor.predict(trajectory, prediction_horizon, features)
            individual_predictions.append(pred_result)

        # Get ensemble prediction
        ensemble_prediction = self.predict(trajectory, prediction_horizon, features)

        return ensemble_prediction, individual_predictions

    def get_prediction_uncertainty(
        self,
        trajectory: Trajectory,
        prediction_horizon: float,
        features: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Calculate detailed uncertainty metrics from ensemble disagreement."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")

        individual_predictions = []
        for predictor in self._predictors:
            pred_result = predictor.predict(trajectory, prediction_horizon, features)
            individual_predictions.append(pred_result)

        if not individual_predictions or not individual_predictions[0].predicted_points:
            return {}

        # Calculate prediction variance across ensemble
        max_length = max(len(pred.predicted_points) for pred in individual_predictions)

        position_variances = []
        confidence_variances = []

        for i in range(max_length):
            x_values = []
            y_values = []
            conf_values = []

            for pred in individual_predictions:
                if i < len(pred.predicted_points):
                    x, y = pred.predicted_points[i]
                    x_values.append(x)
                    y_values.append(y)

                    if pred.confidence_scores and i < len(pred.confidence_scores):
                        conf_values.append(pred.confidence_scores[i])

            if x_values:
                pos_var = np.var(x_values) + np.var(y_values)
                position_variances.append(pos_var)

            if conf_values:
                conf_var = np.var(conf_values)
                confidence_variances.append(conf_var)

        return {
            "mean_position_variance": float(np.mean(position_variances))
            if position_variances
            else 0.0,
            "max_position_variance": float(np.max(position_variances))
            if position_variances
            else 0.0,
            "mean_confidence_variance": float(np.mean(confidence_variances))
            if confidence_variances
            else 0.0,
            "ensemble_disagreement": float(np.mean(position_variances))
            if position_variances
            else 0.0,
            "n_predictors": len(self._predictors),
        }
