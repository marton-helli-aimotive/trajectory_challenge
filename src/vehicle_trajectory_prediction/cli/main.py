"""Main CLI interface for the vehicle trajectory prediction system."""

import asyncio
import sys
from pathlib import Path
from typing import Optional

import click
import hydra
from omegaconf import DictConfig

from ..core.config import get_config
from ..core.logging import setup_logging, get_logger


@click.group()
@click.option("--config", "-c", type=str, help="Configuration file path")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--debug", is_flag=True, help="Enable debug mode")
def cli(config: Optional[str], verbose: bool, debug: bool) -> None:
    """Vehicle Trajectory Prediction System CLI.
    
    A comprehensive system for predicting vehicle trajectories using advanced ML techniques.
    """
    # Set up logging
    log_level = "DEBUG" if debug else ("INFO" if verbose else "WARNING")
    setup_logging(level=log_level, json_format=False)
    
    # Store config path for subcommands
    cli.config_path = config


@cli.command()
@click.option("--host", default="127.0.0.1", help="Host to bind to")
@click.option("--port", default=8000, type=int, help="Port to bind to")
@click.option("--workers", default=4, type=int, help="Number of worker processes")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development")
def serve(host: str, port: int, workers: int, reload: bool) -> None:
    """Start the model serving API server."""
    logger = get_logger(__name__)
    logger.info("Starting model serving API server", host=host, port=port, workers=workers)
    
    try:
        import uvicorn
        from ..mlops.serving import create_app
        
        app = create_app()
        
        if reload:
            uvicorn.run(
                "vehicle_trajectory_prediction.mlops.serving:app",
                host=host,
                port=port,
                reload=True,
                log_level="info"
            )
        else:
            uvicorn.run(
                app,
                host=host,
                port=port,
                workers=workers,
                log_level="info"
            )
    except ImportError as e:
        logger.error("Failed to import serving dependencies", error=str(e))
        sys.exit(1)
    except Exception as e:
        logger.error("Failed to start server", error=str(e))
        sys.exit(1)


@cli.command()
@click.option("--dataset", "-d", required=True, help="Dataset name or path")
@click.option("--output", "-o", required=True, help="Output directory for processed data")
@click.option("--format", "-f", default="parquet", help="Output format (parquet, csv)")
@click.option("--batch-size", default=1000, type=int, help="Batch size for processing")
def process_data(dataset: str, output: str, format: str, batch_size: int) -> None:
    """Process and prepare trajectory data."""
    logger = get_logger(__name__)
    logger.info("Processing trajectory data", dataset=dataset, output=output, format=format)
    
    try:
        from ..data.processor import DataProcessor
        
        processor = DataProcessor()
        processor.process_dataset(dataset, output, format=format, batch_size=batch_size)
        logger.info("Data processing completed successfully")
    except Exception as e:
        logger.error("Data processing failed", error=str(e))
        sys.exit(1)


@cli.command()
@click.option("--data-path", "-d", required=True, help="Path to processed data")
@click.option("--output", "-o", required=True, help="Output directory for features")
@click.option("--config", "-c", help="Feature engineering configuration file")
def extract_features(data_path: str, output: str, config: Optional[str]) -> None:
    """Extract features from trajectory data."""
    logger = get_logger(__name__)
    logger.info("Extracting features", data_path=data_path, output=output)
    
    try:
        from ..features.extractor import FeatureExtractor
        
        extractor = FeatureExtractor()
        extractor.extract_features(data_path, output, config_path=config)
        logger.info("Feature extraction completed successfully")
    except Exception as e:
        logger.error("Feature extraction failed", error=str(e))
        sys.exit(1)


@cli.command()
@click.option("--data-path", "-d", required=True, help="Path to feature data")
@click.option("--output", "-o", required=True, help="Output directory for models")
@click.option("--models", "-m", multiple=True, help="Models to train")
@click.option("--config", "-c", help="Training configuration file")
def train(data_path: str, output: str, models: tuple, config: Optional[str]) -> None:
    """Train trajectory prediction models."""
    logger = get_logger(__name__)
    model_list = list(models) if models else None
    logger.info("Training models", data_path=data_path, output=output, models=model_list)
    
    try:
        from ..models.trainer import ModelTrainer
        
        trainer = ModelTrainer()
        trainer.train_models(data_path, output, models=model_list, config_path=config)
        logger.info("Model training completed successfully")
    except Exception as e:
        logger.error("Model training failed", error=str(e))
        sys.exit(1)


@cli.command()
@click.option("--data-path", "-d", required=True, help="Path to test data")
@click.option("--models-path", "-m", required=True, help="Path to trained models")
@click.option("--output", "-o", required=True, help="Output directory for evaluation results")
@click.option("--metrics", multiple=True, help="Metrics to compute")
def evaluate(data_path: str, models_path: str, output: str, metrics: tuple) -> None:
    """Evaluate trained models."""
    logger = get_logger(__name__)
    metric_list = list(metrics) if metrics else None
    logger.info("Evaluating models", data_path=data_path, models_path=models_path, output=output)
    
    try:
        from ..evaluation.evaluator import ModelEvaluator
        
        evaluator = ModelEvaluator()
        evaluator.evaluate_models(data_path, models_path, output, metrics=metric_list)
        logger.info("Model evaluation completed successfully")
    except Exception as e:
        logger.error("Model evaluation failed", error=str(e))
        sys.exit(1)


@cli.command()
@click.option("--host", default="127.0.0.1", help="Host to bind to")
@click.option("--port", default=8050, type=int, help="Port to bind to")
@click.option("--debug", is_flag=True, help="Enable debug mode")
def dashboard(host: str, port: int, debug: bool) -> None:
    """Start the interactive dashboard."""
    logger = get_logger(__name__)
    logger.info("Starting dashboard", host=host, port=port, debug=debug)
    
    try:
        from ..visualization.dashboard import create_dashboard
        
        app = create_dashboard()
        app.run_server(host=host, port=port, debug=debug)
    except ImportError as e:
        logger.error("Failed to import dashboard dependencies", error=str(e))
        sys.exit(1)
    except Exception as e:
        logger.error("Failed to start dashboard", error=str(e))
        sys.exit(1)


@cli.command()
@click.option("--config", "-c", help="Configuration file path")
def test(config: Optional[str]) -> None:
    """Run tests."""
    logger = get_logger(__name__)
    logger.info("Running tests")
    
    try:
        import pytest
        
        # Add src to Python path
        src_path = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(src_path))
        
        # Run pytest
        pytest.main(["-v", "tests/"])
        logger.info("Tests completed successfully")
    except ImportError as e:
        logger.error("Failed to import pytest", error=str(e))
        sys.exit(1)
    except Exception as e:
        logger.error("Tests failed", error=str(e))
        sys.exit(1)


@cli.command()
@click.option("--config", "-c", help="Configuration file path")
def validate(config: Optional[str]) -> None:
    """Validate configuration and data."""
    logger = get_logger(__name__)
    logger.info("Validating configuration and data")
    
    try:
        # Load configuration
        if config:
            cfg = hydra.compose(config_name=config)
        else:
            cfg = hydra.compose(config_name="config")
        
        # Validate configuration
        config_obj = get_config(cfg)
        logger.info("Configuration validation passed")
        
        # Validate data paths
        data_paths = [
            config_obj.data.raw_data_path,
            config_obj.data.processed_data_path,
            config_obj.data.features_data_path,
            config_obj.data.cache_dir
        ]
        
        for path in data_paths:
            path_obj = Path(path)
            if not path_obj.exists():
                logger.warning(f"Data path does not exist: {path}")
            else:
                logger.info(f"Data path exists: {path}")
        
        logger.info("Validation completed successfully")
    except Exception as e:
        logger.error("Validation failed", error=str(e))
        sys.exit(1)


@cli.command()
@click.option("--config", "-c", help="Configuration file path")
def setup(config: Optional[str]) -> None:
    """Set up the development environment."""
    logger = get_logger(__name__)
    logger.info("Setting up development environment")
    
    try:
        # Create necessary directories
        directories = [
            "data/raw",
            "data/processed", 
            "data/features",
            "data/cache",
            "logs",
            "notebooks",
            "models",
            "results"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
        
        # Install pre-commit hooks
        try:
            import subprocess
            subprocess.run(["pre-commit", "install"], check=True)
            logger.info("Pre-commit hooks installed")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("Failed to install pre-commit hooks")
        
        logger.info("Development environment setup completed")
    except Exception as e:
        logger.error("Setup failed", error=str(e))
        sys.exit(1)


def main() -> None:
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()