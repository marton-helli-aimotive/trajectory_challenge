"""trajectory_challenge package root.

Shortcuts exposed:
- add_numbers, divide: simple utilities used by tests.
- ngsim: data loading API
- trajectory: processing helpers
"""
from __future__ import annotations

from . import ngsim  # re-export module
from .utils import add_numbers, divide  # small helpers for tests

__all__ = [
	"ngsim",
	"add_numbers",
	"divide",
]
