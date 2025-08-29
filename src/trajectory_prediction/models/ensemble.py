"""Ensemble trajectory prediction model."""

from typing import Any, Dict, List, Optional
import numpy as np
from omegaconf import DictConfig

from ..data.validation.schemas import TrajectoryData
from .base import TrajectoryPredictor, PredictionResult, EnsemblePredictor


class EnsembleTrajectoryPredictor(EnsemblePredictor):
    """Ensemble trajectory prediction model combining multiple base models."""
    
    def __init__(self, config: DictConfig, base_models: List[TrajectoryPredictor]):
        super().__init__(config, base_models)
        self.name = "Ensemble"