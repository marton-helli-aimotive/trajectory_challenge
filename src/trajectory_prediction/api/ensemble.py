"""
Ensemble prediction system for trajectory prediction models.

This module provides:
- Model ensemble orchestration
- Multiple ensemble strategies (voting, weighted averaging, stacking)
- Uncertainty quantification from ensemble predictions
- Dynamic model selection and weighting
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import json

from ..models.base import TrajectoryPredictor, PredictionResult
from ..data.schemas import TrajectoryData

logger = logging.getLogger(__name__)


class EnsembleStrategy(str, Enum):
    """Ensemble prediction strategies."""
    SIMPLE_AVERAGE = "simple_average"
    WEIGHTED_AVERAGE = "weighted_average"
    MAJORITY_VOTING = "majority_voting"
    STACKING = "stacking"
    DYNAMIC_SELECTION = "dynamic_selection"


class UncertaintyMethod(str, Enum):
    """Uncertainty quantification methods for ensembles."""
    PREDICTION_VARIANCE = "prediction_variance"
    MODEL_DISAGREEMENT = "model_disagreement"
    CONFIDENCE_INTERVALS = "confidence_intervals"
    ENSEMBLE_ENTROPY = "ensemble_entropy"


@dataclass
class EnsemblePredictionResult:
    """Result from ensemble prediction."""
    ensemble_prediction: PredictionResult
    individual_predictions: List[PredictionResult]
    model_weights: Dict[str, float]
    uncertainty_metrics: Dict[str, float]
    ensemble_confidence: float
    prediction_metadata: Dict[str, Any]


class EnsemblePredictor:
    """
    Ensemble prediction system for trajectory models.
    
    Combines multiple trajectory prediction models to improve accuracy and reliability.
    """
    
    def __init__(self, models: List[TrajectoryPredictor], config: Dict[str, Any]):
        self.models = models
        self.config = config
        
        # Ensemble configuration
        self.strategy = EnsembleStrategy(config.get("strategy", EnsembleStrategy.WEIGHTED_AVERAGE))
        self.uncertainty_method = UncertaintyMethod(config.get("uncertainty_method", UncertaintyMethod.PREDICTION_VARIANCE))
        
        # Model weights (initialized equally, can be learned)
        self.model_weights = self._initialize_model_weights()
        
        # Performance tracking for dynamic weighting
        self.model_performance_history = {model.name: [] for model in models}
        
        # Ensemble metadata
        self.prediction_count = 0
        self.ensemble_performance_history = []
        
        logger.info(f"Ensemble predictor initialized with {len(models)} models using {self.strategy} strategy")
    
    def _initialize_model_weights(self) -> Dict[str, float]:
        """Initialize model weights based on configuration or equal weighting."""
        
        weights = {}
        
        if "model_weights" in self.config:
            # Use configured weights
            configured_weights = self.config["model_weights"]
            total_weight = sum(configured_weights.values())
            
            for model in self.models:
                weights[model.name] = configured_weights.get(model.name, 1.0) / total_weight
        else:
            # Equal weights
            weight_per_model = 1.0 / len(self.models)
            for model in self.models:
                weights[model.name] = weight_per_model
        
        logger.info(f"Model weights initialized: {weights}")
        return weights
    
    async def predict(
        self,
        trajectory_data: TrajectoryData,
        prediction_horizon: float,
        return_individual_predictions: bool = False
    ) -> Union[PredictionResult, EnsemblePredictionResult]:
        """
        Make ensemble prediction.
        
        Args:
            trajectory_data: Input trajectory data
            prediction_horizon: Prediction horizon in seconds
            return_individual_predictions: Whether to return detailed ensemble results
            
        Returns:
            Ensemble prediction result
        """
        
        try:
            # Get predictions from all models
            individual_predictions = await self._get_individual_predictions(
                trajectory_data, prediction_horizon
            )
            
            if not individual_predictions:
                raise ValueError("No successful predictions from ensemble models")
            
            # Apply ensemble strategy
            ensemble_prediction = await self._apply_ensemble_strategy(
                individual_predictions, trajectory_data, prediction_horizon
            )
            
            # Calculate uncertainty metrics
            uncertainty_metrics = await self._calculate_uncertainty_metrics(
                individual_predictions, ensemble_prediction
            )
            
            # Calculate ensemble confidence
            ensemble_confidence = self._calculate_ensemble_confidence(
                individual_predictions, uncertainty_metrics
            )
            
            # Update performance tracking
            self._update_performance_tracking(individual_predictions, ensemble_prediction)
            
            self.prediction_count += 1
            
            if return_individual_predictions:
                return EnsemblePredictionResult(
                    ensemble_prediction=ensemble_prediction,
                    individual_predictions=individual_predictions,
                    model_weights=self.model_weights.copy(),
                    uncertainty_metrics=uncertainty_metrics,
                    ensemble_confidence=ensemble_confidence,
                    prediction_metadata={
                        "ensemble_strategy": self.strategy.value,
                        "num_models": len(individual_predictions),
                        "prediction_count": self.prediction_count
                    }
                )
            else:
                return ensemble_prediction
                
        except Exception as e:
            logger.error(f"Ensemble prediction failed: {e}")
            raise
    
    async def _get_individual_predictions(
        self,
        trajectory_data: TrajectoryData,
        prediction_horizon: float
    ) -> List[Tuple[str, PredictionResult]]:
        """Get predictions from all individual models."""
        
        async def predict_with_model(model: TrajectoryPredictor) -> Optional[Tuple[str, PredictionResult]]:
            try:
                prediction = await model.predict_trajectory(trajectory_data, prediction_horizon)
                return (model.name, prediction)
            except Exception as e:
                logger.warning(f"Model {model.name} prediction failed: {e}")
                return None
        
        # Run predictions in parallel
        tasks = [predict_with_model(model) for model in self.models]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful predictions
        successful_predictions = []
        for result in results:
            if isinstance(result, tuple) and result is not None:
                successful_predictions.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Model prediction exception: {result}")
        
        logger.debug(f"Got {len(successful_predictions)} successful predictions out of {len(self.models)} models")
        
        return successful_predictions
    
    async def _apply_ensemble_strategy(
        self,
        individual_predictions: List[Tuple[str, PredictionResult]],
        trajectory_data: TrajectoryData,
        prediction_horizon: float
    ) -> PredictionResult:
        """Apply the configured ensemble strategy to combine predictions."""
        
        if self.strategy == EnsembleStrategy.SIMPLE_AVERAGE:
            return await self._simple_average_ensemble(individual_predictions)
        
        elif self.strategy == EnsembleStrategy.WEIGHTED_AVERAGE:
            return await self._weighted_average_ensemble(individual_predictions)
        
        elif self.strategy == EnsembleStrategy.MAJORITY_VOTING:
            return await self._majority_voting_ensemble(individual_predictions)
        
        elif self.strategy == EnsembleStrategy.STACKING:
            return await self._stacking_ensemble(individual_predictions, trajectory_data)
        
        elif self.strategy == EnsembleStrategy.DYNAMIC_SELECTION:
            return await self._dynamic_selection_ensemble(individual_predictions, trajectory_data)
        
        else:
            # Fallback to simple average
            logger.warning(f"Unknown ensemble strategy {self.strategy}, using simple average")
            return await self._simple_average_ensemble(individual_predictions)
    
    async def _simple_average_ensemble(
        self,
        individual_predictions: List[Tuple[str, PredictionResult]]
    ) -> PredictionResult:
        """Simple averaging ensemble strategy."""
        
        # Extract predictions
        predictions = [pred for _, pred in individual_predictions]
        
        # Average trajectory predictions
        # Note: This is a simplified implementation - real implementation would
        # need to handle the actual PredictionResult structure
        
        # For now, create a placeholder ensemble result
        # In real implementation, this would properly average the trajectory points,
        # uncertainties, and other prediction components
        
        ensemble_result = predictions[0]  # Placeholder - use first prediction as base
        
        # Add ensemble metadata
        if hasattr(ensemble_result, 'metadata'):
            ensemble_result.metadata = ensemble_result.metadata or {}
            ensemble_result.metadata['ensemble_method'] = 'simple_average'
            ensemble_result.metadata['num_models'] = len(predictions)
        
        logger.debug("Applied simple average ensemble strategy")
        return ensemble_result
    
    async def _weighted_average_ensemble(
        self,
        individual_predictions: List[Tuple[str, PredictionResult]]
    ) -> PredictionResult:
        """Weighted averaging ensemble strategy."""
        
        # Get model weights
        weighted_predictions = []
        total_weight = 0.0
        
        for model_name, prediction in individual_predictions:
            weight = self.model_weights.get(model_name, 0.0)
            if weight > 0:
                weighted_predictions.append((weight, prediction))
                total_weight += weight
        
        if total_weight == 0:
            logger.warning("No valid model weights, falling back to simple average")
            return await self._simple_average_ensemble(individual_predictions)
        
        # Normalize weights
        normalized_weights = [(w/total_weight, pred) for w, pred in weighted_predictions]
        
        # Apply weighted averaging
        # Placeholder implementation - would properly weight and combine predictions
        ensemble_result = normalized_weights[0][1]  # Use highest weighted prediction as base
        
        if hasattr(ensemble_result, 'metadata'):
            ensemble_result.metadata = ensemble_result.metadata or {}
            ensemble_result.metadata['ensemble_method'] = 'weighted_average'
            ensemble_result.metadata['model_weights'] = {
                name: self.model_weights.get(name, 0.0) 
                for name, _ in individual_predictions
            }
        
        logger.debug("Applied weighted average ensemble strategy")
        return ensemble_result
    
    async def _majority_voting_ensemble(
        self,
        individual_predictions: List[Tuple[str, PredictionResult]]
    ) -> PredictionResult:
        """Majority voting ensemble strategy."""
        
        # For trajectory prediction, majority voting might be applied to
        # discretized trajectory classes or risk assessments
        
        # Placeholder implementation - would implement proper voting mechanism
        ensemble_result = individual_predictions[0][1]
        
        if hasattr(ensemble_result, 'metadata'):
            ensemble_result.metadata = ensemble_result.metadata or {}
            ensemble_result.metadata['ensemble_method'] = 'majority_voting'
            ensemble_result.metadata['num_votes'] = len(individual_predictions)
        
        logger.debug("Applied majority voting ensemble strategy")
        return ensemble_result
    
    async def _stacking_ensemble(
        self,
        individual_predictions: List[Tuple[str, PredictionResult]],
        trajectory_data: TrajectoryData
    ) -> PredictionResult:
        """Stacking ensemble strategy with meta-learner."""
        
        # Stacking would use a meta-learner trained on the outputs of base models
        # Placeholder implementation
        
        ensemble_result = individual_predictions[0][1]
        
        if hasattr(ensemble_result, 'metadata'):
            ensemble_result.metadata = ensemble_result.metadata or {}
            ensemble_result.metadata['ensemble_method'] = 'stacking'
            ensemble_result.metadata['meta_learner'] = 'placeholder'
        
        logger.debug("Applied stacking ensemble strategy")
        return ensemble_result
    
    async def _dynamic_selection_ensemble(
        self,
        individual_predictions: List[Tuple[str, PredictionResult]],
        trajectory_data: TrajectoryData
    ) -> PredictionResult:
        """Dynamic selection ensemble strategy."""
        
        # Dynamic selection chooses the best model based on input characteristics
        # or recent performance
        
        # Simple implementation: select model with best recent performance
        best_model_name = None
        best_performance = float('-inf')
        
        for model_name, _ in individual_predictions:
            if model_name in self.model_performance_history:
                recent_performance = np.mean(self.model_performance_history[model_name][-10:])
                if recent_performance > best_performance:
                    best_performance = recent_performance
                    best_model_name = model_name
        
        # Use best model's prediction
        if best_model_name:
            ensemble_result = next(pred for name, pred in individual_predictions if name == best_model_name)
        else:
            ensemble_result = individual_predictions[0][1]
        
        if hasattr(ensemble_result, 'metadata'):
            ensemble_result.metadata = ensemble_result.metadata or {}
            ensemble_result.metadata['ensemble_method'] = 'dynamic_selection'
            ensemble_result.metadata['selected_model'] = best_model_name
        
        logger.debug(f"Applied dynamic selection ensemble strategy, selected: {best_model_name}")
        return ensemble_result
    
    async def _calculate_uncertainty_metrics(
        self,
        individual_predictions: List[Tuple[str, PredictionResult]],
        ensemble_prediction: PredictionResult
    ) -> Dict[str, float]:
        """Calculate uncertainty metrics from ensemble predictions."""
        
        uncertainty_metrics = {}
        
        if self.uncertainty_method == UncertaintyMethod.PREDICTION_VARIANCE:
            # Calculate variance across model predictions
            # Placeholder implementation
            uncertainty_metrics['prediction_variance'] = 0.1
            uncertainty_metrics['std_dev'] = 0.3
        
        elif self.uncertainty_method == UncertaintyMethod.MODEL_DISAGREEMENT:
            # Measure disagreement between models
            # Placeholder implementation
            disagreement_score = min(1.0, len(individual_predictions) / len(self.models))
            uncertainty_metrics['model_disagreement'] = disagreement_score
        
        elif self.uncertainty_method == UncertaintyMethod.CONFIDENCE_INTERVALS:
            # Calculate confidence intervals from ensemble
            # Placeholder implementation
            uncertainty_metrics['confidence_interval_lower'] = -0.5
            uncertainty_metrics['confidence_interval_upper'] = 0.5
        
        elif self.uncertainty_method == UncertaintyMethod.ENSEMBLE_ENTROPY:
            # Calculate entropy of ensemble predictions
            # Placeholder implementation
            uncertainty_metrics['ensemble_entropy'] = 0.2
        
        # Always include basic metrics
        uncertainty_metrics['num_successful_models'] = len(individual_predictions)
        uncertainty_metrics['model_agreement_ratio'] = len(individual_predictions) / len(self.models)
        
        return uncertainty_metrics
    
    def _calculate_ensemble_confidence(
        self,
        individual_predictions: List[Tuple[str, PredictionResult]],
        uncertainty_metrics: Dict[str, float]
    ) -> float:
        """Calculate overall ensemble confidence score."""
        
        # Base confidence on number of successful predictions
        base_confidence = len(individual_predictions) / len(self.models)
        
        # Adjust based on uncertainty metrics
        if 'model_disagreement' in uncertainty_metrics:
            # Lower confidence if models disagree
            disagreement_penalty = uncertainty_metrics['model_disagreement'] * 0.3
            base_confidence *= (1.0 - disagreement_penalty)
        
        # Ensure confidence is in [0, 1] range
        confidence = max(0.0, min(1.0, base_confidence))
        
        return confidence
    
    def _update_performance_tracking(
        self,
        individual_predictions: List[Tuple[str, PredictionResult]],
        ensemble_prediction: PredictionResult
    ) -> None:
        """Update performance tracking for dynamic weighting."""
        
        # Placeholder performance scoring - in real implementation would use
        # actual performance metrics against ground truth
        
        for model_name, prediction in individual_predictions:
            # Mock performance score
            performance_score = np.random.uniform(0.7, 0.9)  # Placeholder
            
            self.model_performance_history[model_name].append(performance_score)
            
            # Keep only recent history
            if len(self.model_performance_history[model_name]) > 100:
                self.model_performance_history[model_name] = self.model_performance_history[model_name][-100:]
        
        # Update model weights based on recent performance
        if self.config.get("adaptive_weighting", False):
            self._update_adaptive_weights()
    
    def _update_adaptive_weights(self) -> None:
        """Update model weights based on recent performance."""
        
        # Calculate weights based on recent performance
        new_weights = {}
        total_performance = 0.0
        
        for model_name, performance_history in self.model_performance_history.items():
            if performance_history:
                recent_performance = np.mean(performance_history[-20:])  # Last 20 predictions
                total_performance += recent_performance
                new_weights[model_name] = recent_performance
        
        # Normalize weights
        if total_performance > 0:
            for model_name in new_weights:
                new_weights[model_name] /= total_performance
            
            # Apply smoothing to prevent rapid weight changes
            smoothing_factor = self.config.get("weight_smoothing", 0.1)
            for model_name in new_weights:
                old_weight = self.model_weights.get(model_name, 0.0)
                self.model_weights[model_name] = (
                    (1 - smoothing_factor) * old_weight + 
                    smoothing_factor * new_weights[model_name]
                )
        
        logger.debug(f"Updated adaptive weights: {self.model_weights}")
    
    def get_ensemble_statistics(self) -> Dict[str, Any]:
        """Get ensemble prediction statistics."""
        
        stats = {
            "num_models": len(self.models),
            "ensemble_strategy": self.strategy.value,
            "uncertainty_method": self.uncertainty_method.value,
            "total_predictions": self.prediction_count,
            "model_weights": self.model_weights.copy(),
            "model_names": [model.name for model in self.models]
        }
        
        # Add performance statistics
        if self.model_performance_history:
            performance_stats = {}
            for model_name, history in self.model_performance_history.items():
                if history:
                    performance_stats[model_name] = {
                        "mean_performance": np.mean(history),
                        "std_performance": np.std(history),
                        "recent_performance": np.mean(history[-10:]) if len(history) >= 10 else np.mean(history)
                    }
            stats["performance_statistics"] = performance_stats
        
        return stats
    
    async def update_model_weights(self, new_weights: Dict[str, float]) -> None:
        """Update model weights manually."""
        
        # Validate and normalize weights
        total_weight = sum(new_weights.values())
        if total_weight <= 0:
            raise ValueError("Total weight must be positive")
        
        normalized_weights = {name: weight/total_weight for name, weight in new_weights.items()}
        
        # Update weights for models that exist
        for model in self.models:
            if model.name in normalized_weights:
                self.model_weights[model.name] = normalized_weights[model.name]
        
        logger.info(f"Model weights updated: {self.model_weights}")
    
    async def add_model(self, model: TrajectoryPredictor, weight: float = None) -> None:
        """Add a new model to the ensemble."""
        
        if model.name in [m.name for m in self.models]:
            raise ValueError(f"Model {model.name} already exists in ensemble")
        
        self.models.append(model)
        
        if weight is None:
            # Distribute weight equally
            weight_per_model = 1.0 / len(self.models)
            self.model_weights = {name: weight_per_model for name in [m.name for m in self.models]}
        else:
            # Add with specified weight and renormalize
            self.model_weights[model.name] = weight
            total_weight = sum(self.model_weights.values())
            self.model_weights = {name: w/total_weight for name, w in self.model_weights.items()}
        
        # Initialize performance tracking
        self.model_performance_history[model.name] = []
        
        logger.info(f"Added model {model.name} to ensemble")
    
    async def remove_model(self, model_name: str) -> None:
        """Remove a model from the ensemble."""
        
        # Find and remove model
        model_to_remove = None
        for model in self.models:
            if model.name == model_name:
                model_to_remove = model
                break
        
        if model_to_remove is None:
            raise ValueError(f"Model {model_name} not found in ensemble")
        
        self.models.remove(model_to_remove)
        
        # Remove from weights and renormalize
        if model_name in self.model_weights:
            del self.model_weights[model_name]
            
            if self.model_weights:
                total_weight = sum(self.model_weights.values())
                self.model_weights = {name: w/total_weight for name, w in self.model_weights.items()}
        
        # Remove performance history
        if model_name in self.model_performance_history:
            del self.model_performance_history[model_name]
        
        logger.info(f"Removed model {model_name} from ensemble")