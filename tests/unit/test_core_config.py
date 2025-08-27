"""Unit tests for configuration management."""

import pytest
from pathlib import Path
from typing import Dict, Any

from vehicle_trajectory_prediction.core.config import (
    Config,
    DataConfig,
    ModelConfig,
    FeatureConfig,
    EvaluationConfig,
    LoggingConfig,
    MLOpsConfig,
    get_config_from_dict,
    save_config,
    load_config,
)
from vehicle_trajectory_prediction.core.exceptions import ConfigurationError


class TestDataConfig:
    """Test DataConfig model."""
    
    def test_valid_data_config(self) -> None:
        """Test creating a valid data configuration."""
        config = DataConfig(
            raw_data_path="data/raw",
            processed_data_path="data/processed",
            features_data_path="data/features",
            cache_dir="data/cache",
            dataset_name="ngsim",
            batch_size=1000,
            num_workers=4,
            chunk_size=10000,
            storage_format="parquet",
            compression="snappy",
            partitioning=["date"]
        )
        
        assert config.raw_data_path == "data/raw"
        assert config.processed_data_path == "data/processed"
        assert config.dataset_name == "ngsim"
        assert config.batch_size == 1000
        assert config.storage_format == "parquet"
    
    def test_data_config_defaults(self) -> None:
        """Test data configuration with defaults."""
        config = DataConfig()
        
        assert config.raw_data_path == "data/raw"
        assert config.processed_data_path == "data/processed"
        assert config.dataset_name == "ngsim"
        assert config.batch_size == 1000
        assert config.storage_format == "parquet"


class TestModelConfig:
    """Test ModelConfig model."""
    
    def test_valid_model_config(self) -> None:
        """Test creating a valid model configuration."""
        config = ModelConfig(
            models=["cv", "ca", "polynomial"],
            train_test_split=0.8,
            validation_split=0.2,
            random_state=42,
            prediction_horizon=30,
            prediction_frequency=0.1
        )
        
        assert config.models == ["cv", "ca", "polynomial"]
        assert config.train_test_split == 0.8
        assert config.validation_split == 0.2
        assert config.random_state == 42
        assert config.prediction_horizon == 30
        assert config.prediction_frequency == 0.1
    
    def test_model_config_defaults(self) -> None:
        """Test model configuration with defaults."""
        config = ModelConfig()
        
        assert "cv" in config.models
        assert "ca" in config.models
        assert "polynomial" in config.models
        assert config.train_test_split == 0.8
        assert config.prediction_horizon == 30


class TestFeatureConfig:
    """Test FeatureConfig model."""
    
    def test_valid_feature_config(self) -> None:
        """Test creating a valid feature configuration."""
        config = FeatureConfig(
            velocity_features=True,
            acceleration_features=True,
            curvature_features=True,
            lane_change_features=True,
            window_size=10,
            min_trajectory_length=20,
            max_trajectory_length=1000,
            enable_augmentation=True,
            noise_std=0.1,
            interpolation_method="linear"
        )
        
        assert config.velocity_features is True
        assert config.acceleration_features is True
        assert config.curvature_features is True
        assert config.window_size == 10
        assert config.min_trajectory_length == 20
        assert config.max_trajectory_length == 1000
        assert config.enable_augmentation is True
        assert config.noise_std == 0.1
        assert config.interpolation_method == "linear"


class TestEvaluationConfig:
    """Test EvaluationConfig model."""
    
    def test_valid_evaluation_config(self) -> None:
        """Test creating a valid evaluation configuration."""
        config = EvaluationConfig(
            metrics=["rmse", "ade", "fde"],
            cv_folds=5,
            cv_strategy="time_series",
            statistical_tests=["wilcoxon"],
            confidence_level=0.95,
            benchmark_inference_speed=True,
            benchmark_memory_usage=True,
            num_benchmark_runs=100
        )
        
        assert "rmse" in config.metrics
        assert "ade" in config.metrics
        assert "fde" in config.metrics
        assert config.cv_folds == 5
        assert config.cv_strategy == "time_series"
        assert config.confidence_level == 0.95
        assert config.benchmark_inference_speed is True
        assert config.num_benchmark_runs == 100


class TestLoggingConfig:
    """Test LoggingConfig model."""
    
    def test_valid_logging_config(self) -> None:
        """Test creating a valid logging configuration."""
        config = LoggingConfig(
            level="INFO",
            format="json",
            include_timestamp=True,
            include_process_id=True,
            include_thread_id=True,
            log_to_file=True,
            log_file_path="logs/app.log",
            max_file_size=104857600,
            backup_count=5
        )
        
        assert config.level == "INFO"
        assert config.format == "json"
        assert config.include_timestamp is True
        assert config.include_process_id is True
        assert config.include_thread_id is True
        assert config.log_to_file is True
        assert config.log_file_path == "logs/app.log"
        assert config.max_file_size == 104857600
        assert config.backup_count == 5


class TestMLOpsConfig:
    """Test MLOpsConfig model."""
    
    def test_valid_mlops_config(self) -> None:
        """Test creating a valid MLOps configuration."""
        config = MLOpsConfig(
            mlflow_enabled=True,
            mlflow_tracking_uri="sqlite:///mlflow.db",
            mlflow_registry_uri="sqlite:///mlflow.db",
            mlflow_experiment_name="trajectory_prediction",
            model_serving_enabled=True,
            serving_host="0.0.0.0",
            serving_port=8000,
            serving_workers=4,
            monitoring_enabled=True,
            drift_detection_enabled=True,
            monitoring_interval=3600
        )
        
        assert config.mlflow_enabled is True
        assert config.mlflow_tracking_uri == "sqlite:///mlflow.db"
        assert config.mlflow_experiment_name == "trajectory_prediction"
        assert config.model_serving_enabled is True
        assert config.serving_host == "0.0.0.0"
        assert config.serving_port == 8000
        assert config.serving_workers == 4
        assert config.monitoring_enabled is True
        assert config.drift_detection_enabled is True
        assert config.monitoring_interval == 3600


class TestConfig:
    """Test main Config model."""
    
    def test_valid_config(self) -> None:
        """Test creating a valid main configuration."""
        config = Config(
            environment="development",
            debug=False,
            data=DataConfig(),
            model=ModelConfig(),
            features=FeatureConfig(),
            evaluation=EvaluationConfig(),
            logging=LoggingConfig(),
            mlops=MLOpsConfig()
        )
        
        assert config.environment == "development"
        assert config.debug is False
        assert isinstance(config.data, DataConfig)
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.features, FeatureConfig)
        assert isinstance(config.evaluation, EvaluationConfig)
        assert isinstance(config.logging, LoggingConfig)
        assert isinstance(config.mlops, MLOpsConfig)
    
    def test_invalid_environment_raises_error(self) -> None:
        """Test that invalid environment raises error."""
        with pytest.raises(ValueError, match="Environment must be one of"):
            Config(
                environment="invalid",
                debug=False,
                data=DataConfig(),
                model=ModelConfig(),
                features=FeatureConfig(),
                evaluation=EvaluationConfig(),
                logging=LoggingConfig(),
                mlops=MLOpsConfig()
            )
    
    def test_config_defaults(self) -> None:
        """Test configuration with defaults."""
        config = Config()
        
        assert config.environment == "development"
        assert config.debug is False
        assert isinstance(config.data, DataConfig)
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.features, FeatureConfig)
        assert isinstance(config.evaluation, EvaluationConfig)
        assert isinstance(config.logging, LoggingConfig)
        assert isinstance(config.mlops, MLOpsConfig)


class TestConfigFunctions:
    """Test configuration utility functions."""
    
    def test_get_config_from_dict(self) -> None:
        """Test creating config from dictionary."""
        config_dict = {
            "environment": "development",
            "debug": False,
            "data": {
                "raw_data_path": "custom/raw",
                "batch_size": 500
            },
            "model": {
                "models": ["cv", "ca"],
                "prediction_horizon": 20
            }
        }
        
        config = get_config_from_dict(config_dict)
        
        assert config.environment == "development"
        assert config.debug is False
        assert config.data.raw_data_path == "custom/raw"
        assert config.data.batch_size == 500
        assert config.model.models == ["cv", "ca"]
        assert config.model.prediction_horizon == 20
    
    def test_save_and_load_config(self, tmp_path: Path) -> None:
        """Test saving and loading configuration."""
        config = Config()
        config_path = tmp_path / "test_config.json"
        
        # Save configuration
        save_config(config, config_path)
        assert config_path.exists()
        
        # Load configuration
        loaded_config = load_config(config_path)
        
        assert loaded_config.environment == config.environment
        assert loaded_config.debug == config.debug
        assert isinstance(loaded_config.data, DataConfig)
        assert isinstance(loaded_config.model, ModelConfig)
    
    def test_load_nonexistent_config_raises_error(self, tmp_path: Path) -> None:
        """Test that loading nonexistent config raises error."""
        config_path = tmp_path / "nonexistent_config.json"
        
        with pytest.raises(FileNotFoundError):
            load_config(config_path)