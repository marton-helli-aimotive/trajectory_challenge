"""Test the logging utilities."""

from pathlib import Path
from unittest.mock import patch

from trajectory_prediction.utils.logging import (
    DEFAULT_LOG_LEVEL,
    get_logger,
    get_logging_config,
    setup_logging,
)


def test_get_logging_config_default() -> None:
    """Test default logging configuration."""
    config = get_logging_config()

    assert config["version"] == 1
    assert not config["disable_existing_loggers"]
    assert "formatters" in config
    assert "handlers" in config
    assert "loggers" in config

    # Check that console handler is included by default
    assert "console" in config["handlers"]
    assert config["loggers"]["trajectory_prediction"]["level"] == DEFAULT_LOG_LEVEL


def test_get_logging_config_with_file() -> None:
    """Test logging configuration with file output."""
    log_file = Path("/tmp/test.log")
    config = get_logging_config(log_file=log_file)

    assert "file" in config["handlers"]
    assert str(log_file) in config["handlers"]["file"]["filename"]


def test_get_logging_config_no_console() -> None:
    """Test logging configuration without console output."""
    config = get_logging_config(console_output=False)

    assert "console" not in config["handlers"]


@patch("logging.config.dictConfig")
def test_setup_logging(mock_dict_config) -> None:
    """Test logging setup function."""
    setup_logging()
    mock_dict_config.assert_called_once()


def test_get_logger() -> None:
    """Test logger retrieval."""
    logger = get_logger("test_logger")
    assert logger.name == "test_logger"
