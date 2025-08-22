"""Model factory for creating trajectory prediction models."""

from typing import Any

from .base import ModelConfig, TrajectoryPredictor
from .baseline import ConstantAccelerationPredictor, ConstantVelocityPredictor
from .ensemble import EnsemblePredictor
from .gaussian_process import GaussianProcessPredictor
from .knn import KNearestNeighborsPredictor
from .mixture_density import SimplifiedMDNPredictor
from .polynomial import PolynomialRegressionPredictor
from .tree_ensemble import TreeEnsemblePredictor


class ModelRegistry:
    """Registry of available trajectory prediction models."""

    _models: dict[str, type[TrajectoryPredictor]] = {
        # Baseline models
        "constant_velocity": ConstantVelocityPredictor,
        "constant_acceleration": ConstantAccelerationPredictor,
        # Classical ML models
        "polynomial_regression": PolynomialRegressionPredictor,
        "polynomial": PolynomialRegressionPredictor,  # Alias for convenience
        "knn": KNearestNeighborsPredictor,
        # Advanced ML models with uncertainty quantification
        "gaussian_process": GaussianProcessPredictor,
        "gp": GaussianProcessPredictor,  # Alias
        "tree_ensemble": TreeEnsemblePredictor,
        "random_forest": TreeEnsemblePredictor,  # Alias
        "mixture_density": SimplifiedMDNPredictor,
        "mdn": SimplifiedMDNPredictor,  # Alias
        # Ensemble methods
        "ensemble": EnsemblePredictor,
    }

    @classmethod
    def register(cls, name: str, model_class: type[TrajectoryPredictor]) -> None:
        """Register a new model type.

        Args:
            name: Model name for factory lookup
            model_class: Model class to register
        """
        cls._models[name] = model_class

    @classmethod
    def get_available_models(cls) -> list[str]:
        """Get list of available model names."""
        return list(cls._models.keys())

    @classmethod
    def create_model(
        cls, model_name: str, config: ModelConfig | None = None
    ) -> TrajectoryPredictor:
        """Create a model instance.

        Args:
            model_name: Name of the model to create
            config: Optional model configuration

        Returns:
            Instantiated model

        Raises:
            ValueError: If model name is not registered
        """
        if model_name not in cls._models:
            available = ", ".join(cls.get_available_models())
            raise ValueError(
                f"Unknown model '{model_name}'. Available models: {available}"
            )

        model_class = cls._models[model_name]
        return model_class(config)  # type: ignore[arg-type]


class ModelFactory:
    """Factory for creating and managing trajectory prediction models."""

    def __init__(self) -> None:
        self.registry = ModelRegistry()

    def create_model(
        self, model_name: str, config: ModelConfig | None = None
    ) -> TrajectoryPredictor:
        """Create a model using the registry.

        Args:
            model_name: Name of the model to create
            config: Optional model configuration

        Returns:
            Instantiated model
        """
        return self.registry.create_model(model_name, config)

    def create_baseline_models(self) -> dict[str, TrajectoryPredictor]:
        """Create all baseline models with default configurations.

        Returns:
            Dictionary mapping model names to instances
        """
        models = {}

        # Constant Velocity
        models["constant_velocity"] = self.create_model(
            "constant_velocity",
            ModelConfig(name="ConstantVelocity", model_type="baseline"),
        )

        # Constant Acceleration
        models["constant_acceleration"] = self.create_model(
            "constant_acceleration",
            ModelConfig(name="ConstantAcceleration", model_type="baseline"),
        )

        return models

    def create_advanced_models(self) -> dict[str, TrajectoryPredictor]:
        """Create all advanced ML models with default configurations.

        Returns:
            Dictionary mapping model names to instances
        """
        models = {}

        # Gaussian Process
        models["gaussian_process"] = self.create_model(
            "gaussian_process",
            ModelConfig(
                name="GaussianProcess",
                model_type="advanced",
                hyperparameters={
                    "kernel": "rbf",
                    "length_scale": 1.0,
                    "noise_level": 1e-5,
                },
            ),
        )

        # Tree Ensemble (Random Forest)
        models["tree_ensemble"] = self.create_model(
            "tree_ensemble",
            ModelConfig(
                name="TreeEnsemble",
                model_type="advanced",
                hyperparameters={
                    "model_type": "random_forest",
                    "n_estimators": 100,
                    "max_depth": 10,
                },
            ),
        )

        # Mixture Density Network
        models["mixture_density"] = self.create_model(
            "mixture_density",
            ModelConfig(
                name="MixtureDensity",
                model_type="advanced",
                hyperparameters={
                    "n_components": 3,
                    "hidden_size": (64, 32),
                    "max_iter": 500,
                    "learning_rate_init": 0.001,
                },
            ),
        )

        return models

    def create_ensemble_model(
        self, predictor_configs: list[dict[str, Any]] | None = None
    ) -> EnsemblePredictor:
        """Create an ensemble model with specified predictors.

        Args:
            predictor_configs: List of predictor configurations

        Returns:
            Configured ensemble model
        """
        if predictor_configs is None:
            # Default ensemble with diverse models
            predictor_configs = [
                {"model": "gaussian_process", "weight": 1.0},
                {"model": "tree_ensemble", "weight": 1.0},
                {"model": "polynomial_regression", "weight": 0.8},
            ]

        ensemble_model = self.create_model(
            "ensemble",
            ModelConfig(
                name="EnsembleModel",
                model_type="ensemble",
                hyperparameters={
                    "combination_method": "uncertainty_weighting",
                    "weight_update_method": "performance_based",
                },
            ),
        )

        # Type cast to ensure we have the right type
        if not isinstance(ensemble_model, EnsemblePredictor):
            raise TypeError("Expected EnsemblePredictor from factory")

        # Add predictors to ensemble
        for pred_config in predictor_configs:
            model_name = pred_config["model"]
            weight = pred_config.get("weight", 1.0)
            hyperparams = pred_config.get("hyperparameters", {})

            predictor = self.create_model(
                model_name,
                ModelConfig(
                    name=f"{model_name}_ensemble_member",
                    model_type="ensemble_member",
                    hyperparameters=hyperparams,
                ),
            )

            ensemble_model.add_predictor(predictor, weight)

        return ensemble_model

    def create_classical_models(self) -> dict[str, TrajectoryPredictor]:
        """Create all classical ML models with default configurations.

        Returns:
            Dictionary mapping model names to instances
        """
        models = {}

        # Polynomial Regression
        models["polynomial_regression"] = self.create_model(
            "polynomial_regression",
            ModelConfig(
                name="PolynomialRegression",
                model_type="classical",
                hyperparameters={
                    "degree": 3,
                    "alpha": 0.01,
                    "include_bias": True,
                    "interaction_only": False,
                },
            ),
        )

        # K-Nearest Neighbors
        models["knn"] = self.create_model(
            "knn",
            ModelConfig(
                name="KNearestNeighbors",
                model_type="classical",
                hyperparameters={
                    "n_neighbors": 5,
                    "normalize_length": True,
                    "target_length": 10,
                    "distance_metric": "dtw",
                    "use_velocity": True,
                },
            ),
        )

        return models

    def create_all_models(self) -> dict[str, TrajectoryPredictor]:
        """Create all available models with default configurations.

        Returns:
            Dictionary mapping model names to instances
        """
        models = {}
        models.update(self.create_baseline_models())
        models.update(self.create_classical_models())
        models.update(self.create_advanced_models())
        return models

    def get_available_models(self) -> list[str]:
        """Get list of available model names."""
        return self.registry.get_available_models()

    def create_from_config_dict(
        self, config_dict: dict[str, Any]
    ) -> dict[str, TrajectoryPredictor]:
        """Create models from a configuration dictionary.

        Args:
            config_dict: Dictionary with model configurations

        Returns:
            Dictionary mapping model names to instances

        Example:
            config = {
                "constant_velocity": {
                    "name": "CV_Model",
                    "model_type": "baseline"
                },
                "polynomial_regression": {
                    "name": "Poly_Model",
                    "model_type": "classical",
                    "hyperparameters": {
                        "degree": 2,
                        "alpha": 0.1
                    }
                }
            }
        """
        models = {}

        for model_name, model_config in config_dict.items():
            if model_name in self.get_available_models():
                config = ModelConfig(**model_config)
                models[model_name] = self.create_model(model_name, config)
            else:
                print(f"Warning: Unknown model '{model_name}' in configuration")

        return models


# Global factory instance
model_factory = ModelFactory()


def create_model(
    model_name: str, config: ModelConfig | None = None
) -> TrajectoryPredictor:
    """Convenience function to create a model.

    Args:
        model_name: Name of the model to create
        config: Optional model configuration

    Returns:
        Instantiated model
    """
    return model_factory.create_model(model_name, config)


def get_available_models() -> list[str]:
    """Get list of available model names."""
    return model_factory.get_available_models()
