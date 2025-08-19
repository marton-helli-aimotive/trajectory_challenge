"""Command line interface for trajectory prediction."""

from pathlib import Path

import typer
from hydra import compose, initialize_config_dir

from trajectory_prediction.utils.logging import get_logger, setup_logging

app = typer.Typer(
    name="trajectory-predict",
    help="Advanced vehicle trajectory prediction pipeline",
)


@app.command()
def train(
    config_path: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file or directory",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory for results",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Enable debug mode",
    ),
) -> None:
    """Train trajectory prediction models."""
    logger = get_logger(__name__)

    # Setup configuration
    if config_path and config_path.is_dir():
        with initialize_config_dir(
            config_dir=str(config_path.absolute()), version_base=None
        ):
            cfg = compose(config_name="config")
    else:
        # Use default config
        config_dir = Path(__file__).parent.parent.parent.parent / "configs"
        with initialize_config_dir(
            config_dir=str(config_dir.absolute()), version_base=None
        ):
            cfg = compose(config_name="config")

    # Setup logging
    log_level = "DEBUG" if debug else cfg.logging.level
    log_file = Path(cfg.log_dir) / cfg.logging.file_path if cfg.logging.file else None
    if output_dir:
        log_file = output_dir / "trajectory_prediction.log"
    setup_logging(
        log_level=log_level, log_file=log_file, console_output=cfg.logging.console
    )

    logger.info("Starting trajectory prediction training")
    logger.info(f"Configuration: {cfg}")

    # TODO: Implement training pipeline
    logger.info("Training pipeline not yet implemented")


@app.command()
def predict(
    model_path: Path = typer.Option(
        ...,
        "--model",
        "-m",
        help="Path to trained model",
    ),
    data_path: Path = typer.Option(
        ...,
        "--data",
        "-d",
        help="Path to input data",
    ),
    output_path: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Path for prediction output",
    ),
) -> None:
    """Generate trajectory predictions."""
    logger = get_logger(__name__)

    logger.info("Starting trajectory prediction")
    logger.info(f"Model: {model_path}")
    logger.info(f"Data: {data_path}")
    logger.info(f"Output: {output_path}")

    # TODO: Implement prediction pipeline
    logger.info("Prediction pipeline not yet implemented")


@app.command()
def evaluate(
    predictions_path: Path = typer.Option(
        ...,
        "--predictions",
        "-p",
        help="Path to predictions file",
    ),
    ground_truth_path: Path = typer.Option(
        ...,
        "--ground-truth",
        "-g",
        help="Path to ground truth file",
    ),
    output_dir: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output directory for evaluation results",
    ),
) -> None:
    """Evaluate trajectory predictions."""
    logger = get_logger(__name__)

    logger.info("Starting trajectory evaluation")
    logger.info(f"Predictions: {predictions_path}")
    logger.info(f"Ground truth: {ground_truth_path}")
    logger.info(f"Output: {output_dir}")

    # TODO: Implement evaluation pipeline
    logger.info("Evaluation pipeline not yet implemented")


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
