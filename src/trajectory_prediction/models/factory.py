"""
Model factory for creating trajectory prediction models.

Implements factory pattern for easy model instantiation and ensemble creation.
"""

from typing import Dict, List, Type, Union

from omegaconf import DictConfig

from .base import TrajectoryPredictor
from .baseline import ConstantVelocityPredictor, ConstantAccelerationPredictor
from .polynomial import PolynomialTrajectoryPredictor
from .knn import KNNTrajectoryPredictor
from .gaussian_process import GaussianProcessPredictor
from .ensemble import EnsembleTrajectoryPredictor


class ModelFactory:
    """
    Factory for creating trajectory prediction models.
    
    Supports all implemented model types and ensemble combinations.
    """
    
    _models: Dict[str, Type[TrajectoryPredictor]] = {
        "constant_velocity": ConstantVelocityPredictor,
        "constant_acceleration": ConstantAccelerationPredictor,
        "polynomial": PolynomialTrajectoryPredictor,
        "knn": KNNTrajectoryPredictor,
        "gaussian_process": GaussianProcessPredictor,
        "ensemble": EnsembleTrajectoryPredictor,
    }
    
    @classmethod
    def create(
        self, 
        model_type: str, 
        config: DictConfig,
        **kwargs
    ) -> TrajectoryPredictor:
        """
        Create a trajectory prediction model.
        
        Args:
            model_type: Type of model to create
            config: Model configuration
            **kwargs: Additional model parameters
            
        Returns:
            Configured model instance
            
        Raises:
            ValueError: If model_type is not supported
        """
        if model_type not in self._models:
            available = ", ".join(self._models.keys())
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Available: {available}"
            )
        
        model_class = self._models[model_type]
        
        # Handle ensemble model creation
        if model_type == "ensemble":
            base_models = self._create_ensemble_models(config, **kwargs)
            return model_class(config, base_models)
        
        return model_class(config, **kwargs)
    
    @classmethod
    def create_ensemble(
        self,
        model_types: List[str],
        config: DictConfig,
        **kwargs
    ) -> TrajectoryPredictor:
        """
        Create ensemble model from list of base model types.
        
        Args:
            model_types: List of base model type names
            config: Configuration
            **kwargs: Additional parameters
            
        Returns:
            Ensemble model instance
        """
        base_models = []
        
        for model_type in model_types:
            if model_type == "ensemble":
                raise ValueError("Cannot create ensemble of ensemble models")
            
            model = self.create(model_type, config, **kwargs)
            base_models.append(model)
        
        ensemble_config = config.get("ensemble", {})
        return EnsembleTrajectoryPredictor(ensemble_config, base_models)
    
    @classmethod
    def _create_ensemble_models(
        self, 
        config: DictConfig, 
        **kwargs
    ) -> List[TrajectoryPredictor]:
        """Create base models for ensemble from configuration."""
        ensemble_config = config.get("ensemble", {})
        base_model_types = ensemble_config.get("base_models", [
            "constant_velocity", 
            "polynomial", 
            "knn"
        ])
        
        base_models = []
        for model_type in base_model_types:
            model = self.create(model_type, config, **kwargs)
            base_models.append(model)
        
        return base_models
    
    @classmethod
    def get_available_models(self) -> Dict[str, Type[TrajectoryPredictor]]:
        """Get all available model types."""
        return self._models.copy()
    
    @classmethod
    def register(self, model_type: str, model_class: Type[TrajectoryPredictor]) -> None:
        """
        Register a new model type.
        
        Args:
            model_type: Name for the model type
            model_class: TrajectoryPredictor implementation
        """
        self._models[model_type] = model_class
    
    @classmethod
    def is_supported(self, model_type: str) -> bool:
        """Check if model type is supported."""
        return model_type in self._models