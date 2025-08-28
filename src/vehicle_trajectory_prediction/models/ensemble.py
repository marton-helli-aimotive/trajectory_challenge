"""Ensemble methods for combining multiple trajectory prediction models."""

from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

from .base import BaseTrajectoryPredictor, PredictionResult
from ..core.models import Trajectory, TrajectoryPoint
from ..core.config import ModelConfig

logger = logging.getLogger(__name__)


class EnsembleStrategy(ABC):
    """Abstract base class for ensemble combination strategies."""
    
    @abstractmethod
    def combine_predictions(self, predictions: List[PredictionResult], 
                          weights: Optional[List[float]] = None) -> PredictionResult:
        """
        Combine multiple predictions into a single prediction.
        
        Args:
            predictions: List of predictions from different models
            weights: Optional weights for each model
            
        Returns:
            Combined prediction result
        """
        pass


class WeightedAverageStrategy(EnsembleStrategy):
    """Weighted average combination strategy."""
    
    def combine_predictions(self, predictions: List[PredictionResult], 
                          weights: Optional[List[float]] = None) -> PredictionResult:
        """Combine predictions using weighted average."""
        if not predictions:
            raise ValueError("No predictions provided")
        
        if weights is None:
            weights = [1.0 / len(predictions)] * len(predictions)
        
        if len(weights) != len(predictions):
            raise ValueError("Number of weights must match number of predictions")
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Combine x and y positions
        x_positions = np.zeros(len(predictions[0].x_positions))
        y_positions = np.zeros(len(predictions[0].y_positions))
        
        for pred, weight in zip(predictions, weights):
            x_positions += weight * np.array(pred.x_positions)
            y_positions += weight * np.array(pred.y_positions)
        
        # Use timestamps from first prediction
        timestamps = predictions[0].timestamps
        
        # Create combined trajectory points
        predicted_points = []
        for i in range(len(x_positions)):
            point = TrajectoryPoint(
                x=x_positions[i],
                y=y_positions[i],
                timestamp=timestamps[i],
                velocity=None,
                acceleration=None,
                heading=None
            )
            predicted_points.append(point)
        
        # Combine uncertainties if available
        uncertainty = None
        if any(p.uncertainty is not None for p in predictions):
            uncertainty = self._combine_uncertainties(predictions, weights)
        
        return PredictionResult(
            predicted_points=predicted_points,
            timestamps=timestamps,
            x_positions=x_positions,
            y_positions=y_positions,
            velocities=None,
            accelerations=None,
            headings=None,
            confidence_scores=weights,
            uncertainty=uncertainty
        )
    
    def _combine_uncertainties(self, predictions: List[PredictionResult], 
                             weights: List[float]) -> Dict[str, Any]:
        """Combine uncertainty estimates from multiple models."""
        combined_uncertainty = {}
        
        # Combine standard deviations
        if any('x_std' in p.uncertainty for p in predictions if p.uncertainty):
            x_stds = []
            y_stds = []
            for pred in predictions:
                if pred.uncertainty and 'x_std' in pred.uncertainty:
                    x_stds.append(pred.uncertainty['x_std'])
                    y_stds.append(pred.uncertainty['y_std'])
            
            if x_stds:
                # Weighted average of variances
                x_var = np.zeros(len(x_stds[0]))
                y_var = np.zeros(len(y_stds[0]))
                
                for std_x, std_y, weight in zip(x_stds, y_stds, weights):
                    x_var += weight * np.array(std_x) ** 2
                    y_var += weight * np.array(std_y) ** 2
                
                combined_uncertainty['x_std'] = np.sqrt(x_var).tolist()
                combined_uncertainty['y_std'] = np.sqrt(y_var).tolist()
        
        return combined_uncertainty


class VotingStrategy(EnsembleStrategy):
    """Voting-based combination strategy."""
    
    def __init__(self, voting_method: str = 'median'):
        """
        Initialize voting strategy.
        
        Args:
            voting_method: 'median' or 'mode'
        """
        self.voting_method = voting_method
    
    def combine_predictions(self, predictions: List[PredictionResult], 
                          weights: Optional[List[float]] = None) -> PredictionResult:
        """Combine predictions using voting."""
        if not predictions:
            raise ValueError("No predictions provided")
        
        # Stack all predictions
        all_x = np.array([pred.x_positions for pred in predictions])
        all_y = np.array([pred.y_positions for pred in predictions])
        
        if self.voting_method == 'median':
            x_positions = np.median(all_x, axis=0)
            y_positions = np.median(all_y, axis=0)
        elif self.voting_method == 'mode':
            # For continuous values, use histogram-based mode approximation
            x_positions = self._approximate_mode(all_x, axis=0)
            y_positions = self._approximate_mode(all_y, axis=0)
        else:
            raise ValueError(f"Unknown voting method: {self.voting_method}")
        
        # Use timestamps from first prediction
        timestamps = predictions[0].timestamps
        
        # Create combined trajectory points
        predicted_points = []
        for i in range(len(x_positions)):
            point = TrajectoryPoint(
                x=x_positions[i],
                y=y_positions[i],
                timestamp=timestamps[i],
                velocity=None,
                acceleration=None,
                heading=None
            )
            predicted_points.append(point)
        
        return PredictionResult(
            predicted_points=predicted_points,
            timestamps=timestamps,
            x_positions=x_positions,
            y_positions=y_positions,
            velocities=None,
            accelerations=None,
            headings=None,
            confidence_scores=None,
            uncertainty=None
        )
    
    def _approximate_mode(self, data: np.ndarray, axis: int = 0) -> np.ndarray:
        """Approximate mode for continuous data using histogram."""
        if axis == 0:
            result = np.zeros(data.shape[1])
            for i in range(data.shape[1]):
                hist, bins = np.histogram(data[:, i], bins=10)
                bin_centers = (bins[:-1] + bins[1:]) / 2
                result[i] = bin_centers[np.argmax(hist)]
            return result
        else:
            return np.median(data, axis=axis)


class DynamicEnsembleStrategy(EnsembleStrategy):
    """Dynamic ensemble selection based on recent performance."""
    
    def __init__(self, window_size: int = 10, performance_metric: str = 'rmse'):
        """
        Initialize dynamic ensemble strategy.
        
        Args:
            window_size: Number of recent predictions to consider
            performance_metric: Metric to use for performance evaluation
        """
        self.window_size = window_size
        self.performance_metric = performance_metric
        self.recent_errors = []  # List of error lists for each model
    
    def combine_predictions(self, predictions: List[PredictionResult], 
                          weights: Optional[List[float]] = None) -> PredictionResult:
        """Combine predictions using dynamic weights based on recent performance."""
        if not predictions:
            raise ValueError("No predictions provided")
        
        # Calculate dynamic weights based on recent performance
        if self.recent_errors and len(self.recent_errors) > 0:
            # Calculate inverse of recent errors (lower error = higher weight)
            recent_performance = []
            for errors in self.recent_errors:
                if errors:
                    avg_error = np.mean(errors[-self.window_size:])
                    recent_performance.append(1.0 / (avg_error + 1e-8))
                else:
                    recent_performance.append(1.0)
            
            # Normalize weights
            total_performance = sum(recent_performance)
            weights = [p / total_performance for p in recent_performance]
        else:
            # Use uniform weights initially
            weights = [1.0 / len(predictions)] * len(predictions)
        
        # Use weighted average strategy
        weighted_strategy = WeightedAverageStrategy()
        return weighted_strategy.combine_predictions(predictions, weights)
    
    def update_performance(self, model_errors: List[float]):
        """Update performance history with new errors."""
        if len(model_errors) != len(self.recent_errors):
            # Initialize error lists for new models
            self.recent_errors = [[] for _ in range(len(model_errors))]
        
        for i, error in enumerate(model_errors):
            self.recent_errors[i].append(error)


class EnsemblePredictor(BaseTrajectoryPredictor):
    """
    Ensemble model that combines multiple trajectory prediction models.
    
    This model uses different combination strategies to improve prediction accuracy
    and provides online learning capabilities for dynamic weight adjustment.
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """Initialize the Ensemble predictor."""
        super().__init__(config)
        
        # Ensemble-specific configuration
        self.models = getattr(self.config, 'models', [])
        self.strategy_type = getattr(self.config, 'strategy_type', 'weighted_average')
        self.initial_weights = getattr(self.config, 'initial_weights', None)
        self.enable_online_learning = getattr(self.config, 'enable_online_learning', True)
        self.learning_rate = getattr(self.config, 'learning_rate', 0.01)
        self.performance_window = getattr(self.config, 'performance_window', 10)
        
        # Model components
        self.base_models = []
        self.weights = None
        self.strategy = None
        self.performance_history = []
        self.online_learner = None
        
        # Initialize ensemble strategy
        self._initialize_strategy()
        
        # Initialize weights
        if self.initial_weights is not None:
            self.weights = self.initial_weights
        else:
            self.weights = [1.0 / len(self.models)] * len(self.models) if self.models else []
        
        # Initialize online learning if enabled
        if self.enable_online_learning:
            self.online_learner = DynamicEnsembleStrategy(
                window_size=self.performance_window,
                performance_metric='rmse'
            )
    
    def _initialize_strategy(self):
        """Initialize the ensemble combination strategy."""
        if self.strategy_type == 'weighted_average':
            self.strategy = WeightedAverageStrategy()
        elif self.strategy_type == 'voting_median':
            self.strategy = VotingStrategy(voting_method='median')
        elif self.strategy_type == 'voting_mode':
            self.strategy = VotingStrategy(voting_method='mode')
        elif self.strategy_type == 'dynamic':
            self.strategy = DynamicEnsembleStrategy(
                window_size=self.performance_window,
                performance_metric='rmse'
            )
        else:
            raise ValueError(f"Unknown strategy type: {self.strategy_type}")
    
    def add_model(self, model: BaseTrajectoryPredictor, weight: float = 1.0):
        """
        Add a model to the ensemble.
        
        Args:
            model: Trajectory prediction model
            weight: Initial weight for the model
        """
        self.base_models.append(model)
        self.weights.append(weight)
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        
        logger.info(f"Added model to ensemble. Total models: {len(self.base_models)}")
    
    def remove_model(self, index: int):
        """
        Remove a model from the ensemble.
        
        Args:
            index: Index of the model to remove
        """
        if 0 <= index < len(self.base_models):
            del self.base_models[index]
            del self.weights[index]
            
            # Normalize weights
            if self.weights:
                total_weight = sum(self.weights)
                self.weights = [w / total_weight for w in self.weights]
            
            logger.info(f"Removed model from ensemble. Total models: {len(self.base_models)}")
        else:
            raise ValueError(f"Invalid model index: {index}")
    
    def train(self, trajectories: List[Trajectory]) -> None:
        """
        Train all models in the ensemble.
        
        Args:
            trajectories: List of training trajectories
        """
        if not self.base_models:
            raise ValueError("No models in ensemble")
        
        logger.info(f"Training ensemble with {len(self.base_models)} models")
        
        # Train each model
        for i, model in enumerate(self.base_models):
            try:
                logger.info(f"Training model {i+1}/{len(self.base_models)}")
                model.train(trajectories)
            except Exception as e:
                logger.error(f"Error training model {i+1}: {e}")
                # Remove failed model
                self.remove_model(i)
        
        if not self.base_models:
            raise ValueError("All models failed to train")
        
        self.is_trained = True
        logger.info("Ensemble training completed")
    
    def predict(self, trajectory: Trajectory) -> PredictionResult:
        """
        Predict future trajectory using ensemble of models.
        
        Args:
            trajectory: Input trajectory for prediction
            
        Returns:
            PredictionResult with combined prediction
        """
        if not self.is_trained:
            raise RuntimeError("Ensemble must be trained before prediction")
        
        if not self.base_models:
            raise ValueError("No models in ensemble")
        
        # Get predictions from all models
        predictions = []
        for model in self.base_models:
            try:
                pred = model.predict(trajectory)
                predictions.append(pred)
            except Exception as e:
                logger.warning(f"Model prediction failed: {e}")
                continue
        
        if not predictions:
            raise RuntimeError("All models failed to predict")
        
        # Combine predictions using the selected strategy
        if self.strategy_type == 'dynamic' and self.online_learner:
            combined_pred = self.online_learner.combine_predictions(predictions)
        else:
            combined_pred = self.strategy.combine_predictions(predictions, self.weights)
        
        return combined_pred
    
    def update_weights_online(self, true_trajectory: Trajectory, 
                            predicted_trajectory: PredictionResult):
        """
        Update model weights based on prediction accuracy.
        
        Args:
            true_trajectory: Ground truth trajectory
            predicted_trajectory: Predicted trajectory from ensemble
        """
        if not self.enable_online_learning or not self.online_learner:
            return
        
        # Calculate errors for each model
        model_errors = []
        for model in self.base_models:
            try:
                pred = model.predict(true_trajectory)
                error = self._calculate_prediction_error(true_trajectory, pred)
                model_errors.append(error)
            except Exception as e:
                logger.warning(f"Error calculating model error: {e}")
                model_errors.append(float('inf'))
        
        # Update online learner
        self.online_learner.update_performance(model_errors)
        
        logger.info(f"Updated online weights. Errors: {model_errors}")
    
    def _calculate_prediction_error(self, true_trajectory: Trajectory, 
                                  predicted_trajectory: PredictionResult) -> float:
        """Calculate prediction error using RMSE."""
        # Extract true future points
        true_points = true_trajectory.points[-self.prediction_horizon:]
        true_x = [p.x for p in true_points]
        true_y = [p.y for p in true_points]
        
        # Extract predicted points
        pred_x = predicted_trajectory.x_positions
        pred_y = predicted_trajectory.y_positions
        
        # Calculate RMSE
        x_errors = np.array(true_x) - np.array(pred_x)
        y_errors = np.array(true_y) - np.array(pred_y)
        
        rmse = np.sqrt(np.mean(x_errors**2 + y_errors**2))
        return rmse
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get performance information for each model in the ensemble."""
        if not self.is_trained:
            return {}
        
        performance = {
            'strategy_type': self.strategy_type,
            'n_models': len(self.base_models),
            'weights': self.weights.copy(),
            'models': []
        }
        
        for i, model in enumerate(self.base_models):
            model_info = {
                'index': i,
                'model_type': model.get_model_info().get('model_type', 'Unknown'),
                'weight': self.weights[i] if i < len(self.weights) else 0.0,
                'is_trained': model.is_trained
            }
            performance['models'].append(model_info)
        
        return performance
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the ensemble model."""
        info = super().get_model_info()
        info.update({
            'model_type': f'Ensemble ({self.strategy_type})',
            'n_models': len(self.base_models),
            'strategy_type': self.strategy_type,
            'enable_online_learning': self.enable_online_learning,
            'learning_rate': self.learning_rate,
            'performance_window': self.performance_window,
            'weights': self.weights,
            'model_types': [model.get_model_info().get('model_type', 'Unknown') 
                           for model in self.base_models]
        })
        return info
    
    def save_model(self, filepath: str) -> None:
        """Save the ensemble model to disk."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before saving")
        
        import joblib
        
        # Save individual models
        model_paths = []
        for i, model in enumerate(self.base_models):
            model_path = f"{filepath}_model_{i}.joblib"
            model.save_model(model_path)
            model_paths.append(model_path)
        
        # Save ensemble configuration
        ensemble_data = {
            'config': self.config,
            'strategy_type': self.strategy_type,
            'weights': self.weights,
            'enable_online_learning': self.enable_online_learning,
            'learning_rate': self.learning_rate,
            'performance_window': self.performance_window,
            'model_paths': model_paths,
            'is_trained': self.is_trained
        }
        
        joblib.dump(ensemble_data, filepath)
        logger.info(f"Ensemble model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load the ensemble model from disk."""
        import joblib
        
        ensemble_data = joblib.load(filepath)
        
        # Restore configuration
        self.config = ensemble_data['config']
        self.strategy_type = ensemble_data['strategy_type']
        self.weights = ensemble_data['weights']
        self.enable_online_learning = ensemble_data['enable_online_learning']
        self.learning_rate = ensemble_data['learning_rate']
        self.performance_window = ensemble_data['performance_window']
        self.is_trained = ensemble_data['is_trained']
        
        # Reinitialize strategy
        self._initialize_strategy()
        
        # Load individual models
        self.base_models = []
        for model_path in ensemble_data['model_paths']:
            try:
                # This would require knowing the model type to recreate
                # For now, we'll just store the paths
                logger.warning("Individual model loading not implemented yet")
                break
            except Exception as e:
                logger.error(f"Error loading model from {model_path}: {e}")
        
        logger.info(f"Ensemble model loaded from {filepath}")