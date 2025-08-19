"""Test the CLI module."""

from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from trajectory_prediction.cli import app


@pytest.fixture
def runner() -> CliRunner:
    """Create a CLI test runner."""
    return CliRunner()


def test_cli_help(runner: CliRunner) -> None:
    """Test CLI help command."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "trajectory prediction pipeline" in result.stdout.lower()


def test_train_command_help(runner: CliRunner) -> None:
    """Test train command help."""
    result = runner.invoke(app, ["train", "--help"])
    assert result.exit_code == 0
    assert "Train trajectory prediction models" in result.stdout


def test_predict_command_help(runner: CliRunner) -> None:
    """Test predict command help."""
    result = runner.invoke(app, ["predict", "--help"])
    assert result.exit_code == 0
    assert "Generate trajectory predictions" in result.stdout


def test_evaluate_command_help(runner: CliRunner) -> None:
    """Test evaluate command help."""
    result = runner.invoke(app, ["evaluate", "--help"])
    assert result.exit_code == 0
    assert "Evaluate trajectory predictions" in result.stdout


@patch("trajectory_prediction.cli.get_logger")
@patch("trajectory_prediction.cli.setup_logging")
@patch("trajectory_prediction.cli.initialize_config_dir")
@patch("trajectory_prediction.cli.compose")
def test_train_command_basic(
    mock_compose,
    mock_init_config,
    mock_setup_logging,
    mock_get_logger,
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    """Test basic train command execution."""
    # Mock logger and config
    mock_logger = mock_get_logger.return_value
    mock_cfg = mock_compose.return_value
    mock_cfg.logging.level = "DEBUG"
    mock_cfg.logging.console = True
    mock_cfg.logging.file = False
    mock_cfg.log_dir = str(tmp_path)

    result = runner.invoke(app, ["train", "--debug"])

    # Should exit successfully even though pipeline is not implemented
    assert result.exit_code == 0
    mock_setup_logging.assert_called_once()
    mock_logger.info.assert_called()


@patch("trajectory_prediction.cli.get_logger")
def test_predict_command_basic(
    mock_get_logger, runner: CliRunner, tmp_path: Path
) -> None:
    """Test basic predict command execution."""
    mock_logger = mock_get_logger.return_value

    # Create dummy files
    model_path = tmp_path / "model.pkl"
    data_path = tmp_path / "data.csv"
    output_path = tmp_path / "output.csv"

    model_path.touch()
    data_path.touch()

    result = runner.invoke(
        app,
        [
            "predict",
            "--model",
            str(model_path),
            "--data",
            str(data_path),
            "--output",
            str(output_path),
        ],
    )

    # Should exit successfully even though pipeline is not implemented
    assert result.exit_code == 0
    mock_logger.info.assert_called()


@patch("trajectory_prediction.cli.get_logger")
def test_evaluate_command_basic(
    mock_get_logger, runner: CliRunner, tmp_path: Path
) -> None:
    """Test basic evaluate command execution."""
    mock_logger = mock_get_logger.return_value

    # Create dummy files
    predictions_path = tmp_path / "predictions.csv"
    ground_truth_path = tmp_path / "ground_truth.csv"
    output_dir = tmp_path / "output"

    predictions_path.touch()
    ground_truth_path.touch()
    output_dir.mkdir()

    result = runner.invoke(
        app,
        [
            "evaluate",
            "--predictions",
            str(predictions_path),
            "--ground-truth",
            str(ground_truth_path),
            "--output",
            str(output_dir),
        ],
    )

    # Should exit successfully even though pipeline is not implemented
    assert result.exit_code == 0
    mock_logger.info.assert_called()
