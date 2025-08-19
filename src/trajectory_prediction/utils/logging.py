"""Logging configuration for trajectory prediction package."""

import logging
import logging.config
from pathlib import Path
from typing import Any

DEFAULT_LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logging_config(
    log_level: str = DEFAULT_LOG_LEVEL,
    log_file: Path | None = None,
    console_output: bool = True,
) -> dict[str, Any]:
    """Get logging configuration dictionary.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for file logging
        console_output: Whether to output to console

    Returns:
        Logging configuration dictionary
    """
    config: dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": LOG_FORMAT,
                "datefmt": LOG_DATE_FORMAT,
            },
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
                "datefmt": LOG_DATE_FORMAT,
            },
        },
        "handlers": {},
        "loggers": {
            "trajectory_prediction": {
                "level": log_level,
                "handlers": [],
                "propagate": False,
            },
            "root": {
                "level": log_level,
                "handlers": [],
            },
        },
    }

    # Add console handler if requested
    if console_output:
        config["handlers"]["console"] = {
            "class": "logging.StreamHandler",
            "level": log_level,
            "formatter": "standard",
            "stream": "ext://sys.stdout",
        }
        config["loggers"]["trajectory_prediction"]["handlers"].append("console")
        config["loggers"]["root"]["handlers"].append("console")

    # Add file handler if log file specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        config["handlers"]["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": log_level,
            "formatter": "detailed",
            "filename": str(log_file),
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
        }
        config["loggers"]["trajectory_prediction"]["handlers"].append("file")
        config["loggers"]["root"]["handlers"].append("file")

    return config


def setup_logging(
    log_level: str = DEFAULT_LOG_LEVEL,
    log_file: Path | None = None,
    console_output: bool = True,
) -> None:
    """Setup logging configuration.

    Args:
        log_level: Logging level
        log_file: Optional file path for file logging
        console_output: Whether to output to console
    """
    config = get_logging_config(log_level, log_file, console_output)
    logging.config.dictConfig(config)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name.

    Args:
        name: Logger name

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)
