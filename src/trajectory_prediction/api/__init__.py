"""
FastAPI-based prediction API for trajectory prediction models.

This module provides:
- RESTful prediction endpoints
- Batch processing capabilities
- Model ensemble serving
- Request/response validation
- Performance optimization (caching, batching)
- Auto-generated OpenAPI documentation
"""

from .server import app, TrajectoryPredictionAPI
from .models import (
    TrajectoryRequest,
    TrajectoryResponse,
    BatchTrajectoryRequest,
    BatchTrajectoryResponse,
    PredictionMetadata,
    HealthStatus
)
from .ensemble import EnsemblePredictor
from .caching import PredictionCache
from .optimization import RequestBatcher, InferenceOptimizer

__all__ = [
    "app",
    "TrajectoryPredictionAPI",
    "TrajectoryRequest",
    "TrajectoryResponse", 
    "BatchTrajectoryRequest",
    "BatchTrajectoryResponse",
    "PredictionMetadata",
    "HealthStatus",
    "EnsemblePredictor",
    "PredictionCache",
    "RequestBatcher",
    "InferenceOptimizer"
]