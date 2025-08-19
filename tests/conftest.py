"""Test configuration and fixtures."""

from pathlib import Path

import pytest


@pytest.fixture
def test_data_dir() -> Path:
    """Return path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def sample_config() -> dict:
    """Return sample configuration for testing."""
    return {
        "data": {
            "name": "test",
            "sample_rate": 10,
            "min_trajectory_length": 10,
        },
        "models": {"baseline": {"constant_velocity": {"enabled": True}}},
        "evaluation": {"metrics": {"rmse": {"enabled": True}}},
    }
