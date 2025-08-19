"""Test the package initialization."""

import trajectory_prediction


def test_package_version() -> None:
    """Test that package version is defined."""
    assert hasattr(trajectory_prediction, "__version__")
    assert trajectory_prediction.__version__ == "0.1.0"


def test_package_author() -> None:
    """Test that package author is defined."""
    assert hasattr(trajectory_prediction, "__author__")
    assert isinstance(trajectory_prediction.__author__, str)


def test_package_description() -> None:
    """Test that package description is defined."""
    assert hasattr(trajectory_prediction, "__description__")
    assert isinstance(trajectory_prediction.__description__, str)
