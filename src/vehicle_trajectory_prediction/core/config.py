"""Configuration management for the vehicle trajectory prediction system."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Try to import optional dependencies
try:
    import hydra
    from hydra.core.config_store import ConfigStore
    from omegaconf import DictConfig, OmegaConf
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False

try:
    from pydantic import BaseModel, Field, validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False


class DataConfig:
    """Configuration for data processing."""
    
    def __init__(
        self,
        raw_data_path: str = "data/raw",
        processed_data_path: str = "data/processed",
        features_data_path: str = "data/features",
        cache_dir: str = "data/cache",
        dataset_name: str = "ngsim",
        dataset_url: Optional[str] = None,
        dataset_format: str = "csv",
        batch_size: int = 1000,
        num_workers: int = 4,
        chunk_size: int = 10000,
        storage_format: str = "parquet",
        compression: str = "snappy",
        partitioning: List[str] = None
    ):
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path
        self.features_data_path = features_data_path
        self.cache_dir = cache_dir
        self.dataset_name = dataset_name
        self.dataset_url = dataset_url
        self.dataset_format = dataset_format
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.chunk_size = chunk_size
        self.storage_format = storage_format
        self.compression = compression
        self.partitioning = partitioning or ["date"]
        
        # Ensure paths exist
        for path in [self.raw_data_path, self.processed_data_path, self.features_data_path, self.cache_dir]:
            Path(path).mkdir(parents=True, exist_ok=True)


class ModelConfig:
    """Configuration for model training and prediction."""
    
    def __init__(
        self,
        models: List[str] = None,
        train_test_split: float = 0.8,
        validation_split: float = 0.2,
        random_state: int = 42,
        prediction_horizon: int = 30,
        prediction_frequency: float = 0.1,
        cv_config: Dict[str, Any] = None,
        ca_config: Dict[str, Any] = None,
        polynomial_config: Dict[str, Any] = None,
        knn_config: Dict[str, Any] = None,
        gaussian_process_config: Dict[str, Any] = None,
        ensemble_config: Dict[str, Any] = None
    ):
        self.models = models or ["cv", "ca", "polynomial", "knn", "gaussian_process", "ensemble"]
        self.train_test_split = train_test_split
        self.validation_split = validation_split
        self.random_state = random_state
        self.prediction_horizon = prediction_horizon
        self.prediction_frequency = prediction_frequency
        self.cv_config = cv_config or {}
        self.ca_config = ca_config or {}
        self.polynomial_config = polynomial_config or {"degree": 3, "features": ["position", "velocity", "acceleration"]}
        self.knn_config = knn_config or {"n_neighbors": 5, "weights": "uniform"}
        self.gaussian_process_config = gaussian_process_config or {"kernel": "rbf", "alpha": 1e-6}
        self.ensemble_config = ensemble_config or {"method": "voting", "weights": None}


class FeatureConfig:
    """Configuration for feature engineering."""
    
    def __init__(
        self,
        velocity_features: bool = True,
        acceleration_features: bool = True,
        curvature_features: bool = True,
        lane_change_features: bool = True,
        spatial_features: bool = True,
        temporal_features: bool = True,
        window_size: int = 10,
        min_trajectory_length: int = 20,
        max_trajectory_length: int = 1000,
        enable_augmentation: bool = True,
        noise_std: float = 0.1,
        interpolation_method: str = "linear",
        feature_store_enabled: bool = True,
        feature_cache_size: int = 1000
    ):
        self.velocity_features = velocity_features
        self.acceleration_features = acceleration_features
        self.curvature_features = curvature_features
        self.lane_change_features = lane_change_features
        self.spatial_features = spatial_features
        self.temporal_features = temporal_features
        self.window_size = window_size
        self.min_trajectory_length = min_trajectory_length
        self.max_trajectory_length = max_trajectory_length
        self.enable_augmentation = enable_augmentation
        self.noise_std = noise_std
        self.interpolation_method = interpolation_method
        self.feature_store_enabled = feature_store_enabled
        self.feature_cache_size = feature_cache_size


class EvaluationConfig:
    """Configuration for model evaluation."""
    
    def __init__(
        self,
        metrics: List[str] = None,
        cv_folds: int = 5,
        cv_strategy: str = "time_series",
        statistical_tests: List[str] = None,
        confidence_level: float = 0.95,
        benchmark_inference_speed: bool = True,
        benchmark_memory_usage: bool = True,
        num_benchmark_runs: int = 100
    ):
        self.metrics = metrics or ["rmse", "ade", "fde", "min_distance", "ttc", "lateral_error"]
        self.cv_folds = cv_folds
        self.cv_strategy = cv_strategy
        self.statistical_tests = statistical_tests or ["wilcoxon", "mann_whitney"]
        self.confidence_level = confidence_level
        self.benchmark_inference_speed = benchmark_inference_speed
        self.benchmark_memory_usage = benchmark_memory_usage
        self.num_benchmark_runs = num_benchmark_runs


class LoggingConfig:
    """Configuration for logging."""
    
    def __init__(
        self,
        level: str = "INFO",
        format: str = "json",
        include_timestamp: bool = True,
        include_process_id: bool = True,
        include_thread_id: bool = True,
        log_to_file: bool = True,
        log_file_path: str = "logs/app.log",
        max_file_size: int = 100 * 1024 * 1024,
        backup_count: int = 5
    ):
        self.level = level
        self.format = format
        self.include_timestamp = include_timestamp
        self.include_process_id = include_process_id
        self.include_thread_id = include_thread_id
        self.log_to_file = log_to_file
        self.log_file_path = log_file_path
        self.max_file_size = max_file_size
        self.backup_count = backup_count


class MLOpsConfig:
    """Configuration for MLOps infrastructure."""
    
    def __init__(
        self,
        mlflow_enabled: bool = True,
        mlflow_tracking_uri: str = "sqlite:///mlflow.db",
        mlflow_registry_uri: str = "sqlite:///mlflow.db",
        mlflow_experiment_name: str = "trajectory_prediction",
        model_serving_enabled: bool = True,
        serving_host: str = "0.0.0.0",
        serving_port: int = 8000,
        serving_workers: int = 4,
        monitoring_enabled: bool = True,
        drift_detection_enabled: bool = True,
        monitoring_interval: int = 3600
    ):
        self.mlflow_enabled = mlflow_enabled
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.mlflow_registry_uri = mlflow_registry_uri
        self.mlflow_experiment_name = mlflow_experiment_name
        self.model_serving_enabled = model_serving_enabled
        self.serving_host = serving_host
        self.serving_port = serving_port
        self.serving_workers = serving_workers
        self.monitoring_enabled = monitoring_enabled
        self.drift_detection_enabled = drift_detection_enabled
        self.monitoring_interval = monitoring_interval


class Config:
    """Main configuration class."""
    
    def __init__(
        self,
        environment: str = "development",
        debug: bool = False,
        data: DataConfig = None,
        model: ModelConfig = None,
        features: FeatureConfig = None,
        evaluation: EvaluationConfig = None,
        logging: LoggingConfig = None,
        mlops: MLOpsConfig = None
    ):
        # Validate environment
        valid_environments = ["development", "staging", "production"]
        if environment not in valid_environments:
            raise ValueError(f"Environment must be one of {valid_environments}")
        
        self.environment = environment
        self.debug = debug
        self.data = data or DataConfig()
        self.model = model or ModelConfig()
        self.features = features or FeatureConfig()
        self.evaluation = evaluation or EvaluationConfig()
        self.logging = logging or LoggingConfig()
        self.mlops = mlops or MLOpsConfig()


# Register configuration with Hydra if available
# Temporarily commented out to avoid Hydra Config class conflicts
# if HYDRA_AVAILABLE:
#     cs = ConfigStore.instance()
#     cs.store(name="config", node=Config)


def get_config(cfg=None):
    """Get configuration from Hydra or return default."""
    if HYDRA_AVAILABLE and cfg is not None:
        # Convert DictConfig to dict and create Config object
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        return Config(**config_dict)
    else:
        return Config()


def get_config_from_dict(config_dict: Dict[str, Any]) -> Config:
    """Get configuration from dictionary."""
    return Config(**config_dict)


def save_config(config: Config, path: Union[str, Path]) -> None:
    """Save configuration to file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Simple JSON serialization for basic config
    import json
    config_dict = {
        "environment": config.environment,
        "debug": config.debug,
        "data": {
            "raw_data_path": config.data.raw_data_path,
            "processed_data_path": config.data.processed_data_path,
            "features_data_path": config.data.features_data_path,
            "cache_dir": config.data.cache_dir,
            "dataset_name": config.data.dataset_name,
            "batch_size": config.data.batch_size,
            "storage_format": config.data.storage_format
        },
        "model": {
            "models": config.model.models,
            "train_test_split": config.model.train_test_split,
            "prediction_horizon": config.model.prediction_horizon
        }
    }
    
    with open(path, "w") as f:
        json.dump(config_dict, f, indent=2)


def load_config(path: Union[str, Path]) -> Config:
    """Load configuration from file."""
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    
    # Simple JSON loading for basic config
    import json
    with open(path, "r") as f:
        config_dict = json.load(f)
    
    return Config(**config_dict)