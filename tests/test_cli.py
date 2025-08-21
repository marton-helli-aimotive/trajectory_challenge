"""Test the CLI module."""

from pathlib import Path

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
    result = runner.invoke(app, ["train-command", "--help"])
    assert result.exit_code == 0
    assert "Train trajectory prediction models" in result.stdout


def test_predict_command_help(runner: CliRunner) -> None:
    """Test predict command help."""
    result = runner.invoke(app, ["predict-command", "--help"])
    assert result.exit_code == 0
    assert "Make trajectory predictions" in result.stdout


def test_evaluate_command_help(runner: CliRunner) -> None:
    """Test evaluate command help."""
    result = runner.invoke(app, ["evaluate-command", "--help"])
    assert result.exit_code == 0
    assert "Evaluate trajectory prediction results" in result.stdout


def test_train_command_basic(runner: CliRunner, tmp_path: Path) -> None:
    """Test basic train command execution."""
    # Create dummy config file
    config_file = tmp_path / "config.yaml"
    config_file.touch()

    result = runner.invoke(app, ["train-command", str(config_file)])

    # Should exit successfully
    assert result.exit_code == 0
    assert "Training models with config" in result.stdout


def test_predict_command_basic(runner: CliRunner, tmp_path: Path) -> None:
    """Test basic predict command execution."""
    # Create dummy files
    model_path = tmp_path / "model.pkl"
    data_path = tmp_path / "data.csv"
    output_path = tmp_path / "output.csv"

    model_path.touch()
    data_path.touch()

    result = runner.invoke(
        app,
        [
            "predict-command",
            str(model_path),
            str(data_path),
            str(output_path),
        ],
    )

    # Should exit successfully
    assert result.exit_code == 0
    assert "Making predictions using model" in result.stdout


def test_evaluate_command_basic(runner: CliRunner, tmp_path: Path) -> None:
    """Test basic evaluate command execution."""
    # Create dummy files
    results_path = tmp_path / "results.csv"
    ground_truth_path = tmp_path / "ground_truth.csv"
    output_path = tmp_path / "output"

    results_path.touch()
    ground_truth_path.touch()
    output_path.mkdir()

    result = runner.invoke(
        app,
        [
            "evaluate-command",
            str(results_path),
            str(ground_truth_path),
            str(output_path),
        ],
    )

    # Should exit successfully
    assert result.exit_code == 0
    assert "Evaluating results from" in result.stdout
