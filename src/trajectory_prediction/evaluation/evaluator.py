"""
Model evaluation, cross-validation, and comparison framework.

This module provides comprehensive evaluation capabilities including:
- Cross-validation with different strategies
- Model comparison and benchmarking
- Performance analysis and reporting
"""

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

import numpy as np
from sklearn.model_selection import KFold, TimeSeriesSplit
from omegaconf import DictConfig

from .metrics import MetricCalculator, EvaluationResult
from ..models.base import TrajectoryPredictor, PredictionResult
from ..data.validation.schemas import TrajectoryData

logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""
    cv_folds: int = 5
    test_size: float = 0.2
    random_state: int = 42
    cv_strategy: str = "kfold"  # kfold, time_series
    parallel_evaluation: bool = True
    max_workers: int = 4


class ModelEvaluator:
    """
    Comprehensive model evaluation framework.
    
    Provides single-model evaluation with detailed analysis.
    """
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.eval_config = EvaluationConfig()
        self.metric_calculator = MetricCalculator(config)
        
    async def evaluate_model(
        self,
        model: TrajectoryPredictor,
        test_trajectories: List[TrajectoryData],
        prediction_horizon: float = 10.0,
        other_vehicles: Optional[List[List[TrajectoryData]]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation of a single model.
        
        Args:
            model: Trained trajectory prediction model
            test_trajectories: Test trajectory data
            prediction_horizon: Prediction horizon in seconds
            other_vehicles: Other vehicle trajectories for safety metrics
            
        Returns:
            Comprehensive evaluation results
        """
        if not model.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        logger.info(f"Evaluating model {model.name} on {len(test_trajectories)} trajectories")
        
        # Group trajectories by vehicle for evaluation
        vehicle_trajectories = self._group_by_vehicle(test_trajectories)
        
        predictions = []
        ground_truths = []
        
        # Generate predictions for each vehicle
        for vehicle_id, traj_points in vehicle_trajectories.items():
            if len(traj_points) < model.min_history_length + 5:  # Need history + future
                continue
            
            # Sort by timestamp
            traj_points = sorted(traj_points, key=lambda x: x.timestamp)
            
            # Split into history and ground truth
            split_point = len(traj_points) - max(5, int(prediction_horizon * 10))  # Rough estimate
            history = traj_points[:split_point]
            future_gt = traj_points[split_point:]
            
            if len(history) >= model.min_history_length and future_gt:
                try:
                    # Generate prediction
                    prediction = await model.predict(history, prediction_horizon)
                    predictions.append(prediction)
                    ground_truths.append(future_gt)
                    
                except Exception as e:
                    logger.warning(f"Failed to predict for vehicle {vehicle_id}: {e}")
                    continue
        
        if not predictions:
            logger.error("No predictions generated during evaluation")
            return {"error": "No predictions generated", "model_name": model.name}
        
        # Calculate all metrics
        metrics = await self.metric_calculator.calculate_all_metrics(
            model.name, predictions, ground_truths, other_vehicles
        )
        
        # Add evaluation metadata
        metrics.update({
            "evaluation_config": {
                "prediction_horizon": prediction_horizon,
                "test_vehicles": len(vehicle_trajectories),
                "successful_predictions": len(predictions),
                "evaluation_timestamp": str(pd.Timestamp.now())
            },
            "model_info": model.get_model_info()
        })
        
        logger.info(f"Evaluation complete for {model.name}: {len(predictions)} predictions analyzed")
        
        return metrics
    
    async def evaluate_prediction_horizons(
        self,
        model: TrajectoryPredictor,
        test_trajectories: List[TrajectoryData],
        horizons: List[float] = [1.0, 3.0, 5.0, 10.0, 15.0]
    ) -> Dict[float, Dict[str, Any]]:
        """
        Evaluate model performance across different prediction horizons.
        
        Args:
            model: Trained model to evaluate
            test_trajectories: Test data
            horizons: List of prediction horizons to evaluate
            
        Returns:
            Results for each horizon
        """
        horizon_results = {}
        
        for horizon in horizons:
            logger.info(f"Evaluating horizon {horizon}s")
            
            try:
                results = await self.evaluate_model(
                    model, test_trajectories, horizon
                )
                horizon_results[horizon] = results
                
            except Exception as e:
                logger.error(f"Failed to evaluate horizon {horizon}: {e}")
                horizon_results[horizon] = {"error": str(e)}
        
        return horizon_results
    
    def _group_by_vehicle(self, trajectories: List[TrajectoryData]) -> Dict[int, List[TrajectoryData]]:
        """Group trajectory points by vehicle ID."""
        vehicle_groups = {}
        
        for point in trajectories:
            vehicle_id = point.vehicle_id
            if vehicle_id not in vehicle_groups:
                vehicle_groups[vehicle_id] = []
            vehicle_groups[vehicle_id].append(point)
        
        return vehicle_groups


class CrossValidator:
    """
    Cross-validation framework for trajectory prediction models.
    
    Supports different CV strategies including time-series aware splits.
    """
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.eval_config = EvaluationConfig()
        self.metric_calculator = MetricCalculator(config)
    
    async def cross_validate_model(
        self,
        model: TrajectoryPredictor,
        trajectories: List[TrajectoryData],
        prediction_horizon: float = 10.0,
        cv_folds: int = 5
    ) -> Dict[str, Any]:
        """
        Perform cross-validation on a trajectory prediction model.
        
        Args:
            model: Model to cross-validate
            trajectories: All available trajectory data
            prediction_horizon: Prediction horizon for evaluation
            cv_folds: Number of cross-validation folds
            
        Returns:
            Cross-validation results with performance statistics
        """
        logger.info(f"Starting {cv_folds}-fold cross-validation for {model.name}")
        
        # Group trajectories by vehicle for proper splitting
        vehicle_trajectories = self._group_by_vehicle(trajectories)
        vehicle_ids = list(vehicle_trajectories.keys())
        
        if len(vehicle_ids) < cv_folds:
            raise ValueError(f"Not enough vehicles ({len(vehicle_ids)}) for {cv_folds}-fold CV")
        
        # Create cross-validation splits
        cv_splitter = self._create_cv_splitter(cv_folds)
        fold_results = []
        
        for fold_idx, (train_indices, test_indices) in enumerate(cv_splitter.split(vehicle_ids)):
            logger.info(f"Processing fold {fold_idx + 1}/{cv_folds}")
            
            # Split data
            train_vehicle_ids = [vehicle_ids[i] for i in train_indices]
            test_vehicle_ids = [vehicle_ids[i] for i in test_indices]
            
            train_data = []
            test_data = []
            
            for vid in train_vehicle_ids:
                train_data.extend(vehicle_trajectories[vid])
            
            for vid in test_vehicle_ids:
                test_data.extend(vehicle_trajectories[vid])
            
            try:
                # Train model on fold data
                logger.info(f"Training on {len(train_data)} points, testing on {len(test_data)} points")
                await model.fit(train_data)
                
                # Evaluate on test data
                evaluator = ModelEvaluator(self.config)
                fold_metrics = await evaluator.evaluate_model(model, test_data, prediction_horizon)
                
                fold_metrics["fold"] = fold_idx
                fold_metrics["train_size"] = len(train_data)
                fold_metrics["test_size"] = len(test_data)
                
                fold_results.append(fold_metrics)
                
            except Exception as e:
                logger.error(f"Fold {fold_idx} failed: {e}")
                fold_results.append({
                    "fold": fold_idx,
                    "error": str(e),
                    "train_size": len(train_data),
                    "test_size": len(test_data)
                })
        
        # Aggregate results across folds
        cv_summary = self._aggregate_cv_results(fold_results, model.name)
        
        return cv_summary
    
    def _create_cv_splitter(self, cv_folds: int):
        """Create appropriate cross-validation splitter."""
        if self.eval_config.cv_strategy == "time_series":
            return TimeSeriesSplit(n_splits=cv_folds)
        else:
            return KFold(n_splits=cv_folds, shuffle=True, random_state=self.eval_config.random_state)
    
    def _aggregate_cv_results(self, fold_results: List[Dict[str, Any]], model_name: str) -> Dict[str, Any]:
        """Aggregate cross-validation results across folds."""
        
        # Filter out failed folds
        successful_folds = [fold for fold in fold_results if "error" not in fold]
        failed_folds = [fold for fold in fold_results if "error" in fold]
        
        if not successful_folds:
            return {
                "model_name": model_name,
                "status": "failed",
                "successful_folds": 0,
                "failed_folds": len(failed_folds),
                "errors": [fold["error"] for fold in failed_folds]
            }
        
        # Aggregate metrics from successful folds
        aggregated_metrics = {}
        
        # Aggregate trajectory metrics
        trajectory_metrics = []
        safety_metrics = []
        probabilistic_metrics = []
        
        for fold in successful_folds:
            if "trajectory_metrics" in fold:
                trajectory_metrics.append(fold["trajectory_metrics"])
            if "safety_metrics" in fold:
                safety_metrics.append(fold["safety_metrics"])
            if "probabilistic_metrics" in fold:
                probabilistic_metrics.append(fold["probabilistic_metrics"])
        
        # Calculate cross-fold statistics
        if trajectory_metrics:
            aggregated_metrics["trajectory_metrics"] = self._calculate_cross_fold_stats(trajectory_metrics)
        
        if safety_metrics:
            aggregated_metrics["safety_metrics"] = self._calculate_cross_fold_stats(safety_metrics)
        
        if probabilistic_metrics:
            aggregated_metrics["probabilistic_metrics"] = self._calculate_cross_fold_stats(probabilistic_metrics)
        
        # Overall summary
        cv_summary = {
            "model_name": model_name,
            "cv_folds": len(fold_results),
            "successful_folds": len(successful_folds),
            "failed_folds": len(failed_folds),
            "fold_results": fold_results,
            **aggregated_metrics
        }
        
        if failed_folds:
            cv_summary["errors"] = [fold["error"] for fold in failed_folds]
        
        return cv_summary
    
    def _calculate_cross_fold_stats(self, fold_metrics: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Calculate statistics across CV folds for each metric."""
        if not fold_metrics:
            return {}
        
        # Get all metric names
        all_metrics = set()
        for fold in fold_metrics:
            for metric_category in fold.values():
                if isinstance(metric_category, dict):
                    all_metrics.update(metric_category.keys())
        
        cross_fold_stats = {}
        
        for metric_name in all_metrics:
            values = []
            
            # Collect values across folds
            for fold in fold_metrics:
                for category in fold.values():
                    if isinstance(category, dict) and metric_name in category:
                        if isinstance(category[metric_name], dict) and "mean" in category[metric_name]:
                            values.append(category[metric_name]["mean"])
                        elif isinstance(category[metric_name], (int, float)):
                            values.append(category[metric_name])
            
            # Calculate cross-fold statistics
            if values and all(np.isfinite(v) for v in values):
                cross_fold_stats[metric_name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "median": float(np.median(values)),
                    "cv": float(np.std(values) / np.mean(values)) if np.mean(values) != 0 else float("inf")
                }
        
        return cross_fold_stats
    
    def _group_by_vehicle(self, trajectories: List[TrajectoryData]) -> Dict[int, List[TrajectoryData]]:
        """Group trajectory points by vehicle ID."""
        vehicle_groups = {}
        
        for point in trajectories:
            vehicle_id = point.vehicle_id
            if vehicle_id not in vehicle_groups:
                vehicle_groups[vehicle_id] = []
            vehicle_groups[vehicle_id].append(point)
        
        return vehicle_groups


class ModelComparator:
    """
    Framework for comparing multiple trajectory prediction models.
    
    Provides statistical comparison and ranking of models.
    """
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.eval_config = EvaluationConfig()
    
    async def compare_models(
        self,
        models: List[TrajectoryPredictor],
        test_trajectories: List[TrajectoryData],
        prediction_horizon: float = 10.0,
        include_cross_validation: bool = True
    ) -> Dict[str, Any]:
        """
        Comprehensive comparison of multiple models.
        
        Args:
            models: List of trained models to compare
            test_trajectories: Test data for evaluation
            prediction_horizon: Prediction horizon for comparison
            include_cross_validation: Whether to include CV results
            
        Returns:
            Detailed comparison results with rankings
        """
        logger.info(f"Comparing {len(models)} models")
        
        comparison_results = {
            "models": [model.name for model in models],
            "prediction_horizon": prediction_horizon,
            "test_size": len(test_trajectories),
            "individual_results": {},
            "rankings": {},
            "statistical_tests": {}
        }
        
        # Evaluate each model
        evaluator = ModelEvaluator(self.config)
        
        for model in models:
            logger.info(f"Evaluating {model.name}")
            
            try:
                # Single evaluation
                model_results = await evaluator.evaluate_model(
                    model, test_trajectories, prediction_horizon
                )
                
                comparison_results["individual_results"][model.name] = model_results
                
                # Cross-validation if requested
                if include_cross_validation:
                    cv_validator = CrossValidator(self.config)
                    cv_results = await cv_validator.cross_validate_model(
                        model, test_trajectories, prediction_horizon
                    )
                    comparison_results["individual_results"][model.name]["cross_validation"] = cv_results
                
            except Exception as e:
                logger.error(f"Failed to evaluate {model.name}: {e}")
                comparison_results["individual_results"][model.name] = {
                    "error": str(e),
                    "status": "failed"
                }
        
        # Generate rankings
        comparison_results["rankings"] = self._generate_rankings(
            comparison_results["individual_results"]
        )
        
        # Statistical significance tests (if we have multiple successful evaluations)
        successful_models = [
            name for name, results in comparison_results["individual_results"].items()
            if "error" not in results
        ]
        
        if len(successful_models) >= 2:
            from .statistical import StatisticalTester
            tester = StatisticalTester(self.config)
            
            comparison_results["statistical_tests"] = await tester.compare_models_statistical(
                {name: comparison_results["individual_results"][name] for name in successful_models}
            )
        
        return comparison_results
    
    def _generate_rankings(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate model rankings based on various metrics."""
        
        # Filter successful models
        successful_models = {
            name: results for name, results in model_results.items()
            if "error" not in results and "trajectory_metrics" in results
        }
        
        if len(successful_models) < 2:
            return {"message": "Not enough successful models for ranking"}
        
        rankings = {}
        
        # Key metrics for ranking
        key_metrics = [
            ("trajectory_metrics", "ade", "mean", False),  # Lower is better
            ("trajectory_metrics", "fde", "mean", False),
            ("trajectory_metrics", "rmse", "mean", False),
            ("safety_metrics", "min_ttc", "mean", True),   # Higher is better
            ("safety_metrics", "collision_risk", "mean", False)
        ]
        
        for category, metric, stat, higher_better in key_metrics:
            metric_values = {}
            
            for model_name, results in successful_models.items():
                if (category in results and 
                    metric in results[category] and 
                    stat in results[category][metric]):
                    
                    value = results[category][metric][stat]
                    if np.isfinite(value):
                        metric_values[model_name] = value
            
            if len(metric_values) >= 2:
                # Rank models for this metric
                sorted_models = sorted(
                    metric_values.items(), 
                    key=lambda x: x[1], 
                    reverse=higher_better
                )
                
                rankings[f"{category}_{metric}"] = {
                    "ranking": [{"model": name, "value": value, "rank": i+1} 
                              for i, (name, value) in enumerate(sorted_models)],
                    "best_model": sorted_models[0][0],
                    "best_value": sorted_models[0][1]
                }
        
        # Overall ranking (simple average of ranks)
        if rankings:
            overall_scores = {name: [] for name in successful_models.keys()}
            
            for ranking_info in rankings.values():
                for entry in ranking_info["ranking"]:
                    overall_scores[entry["model"]].append(entry["rank"])
            
            # Calculate average rank
            avg_ranks = {
                name: np.mean(ranks) for name, ranks in overall_scores.items()
                if ranks
            }
            
            overall_ranking = sorted(avg_ranks.items(), key=lambda x: x[1])
            
            rankings["overall"] = {
                "ranking": [{"model": name, "avg_rank": rank, "rank": i+1}
                          for i, (name, rank) in enumerate(overall_ranking)],
                "best_model": overall_ranking[0][0]
            }
        
        return rankings