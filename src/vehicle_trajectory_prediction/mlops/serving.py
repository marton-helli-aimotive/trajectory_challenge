"""FastAPI model serving for vehicle trajectory prediction."""

import os
import time
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# from ..core.config import Config, get_config  # Removed for now
from ..core.logging import get_logger, setup_logging
from ..core.models import TrajectoryPoint, Trajectory
from ..models.base import BaseTrajectoryPredictor
# from ..models.factory import ModelFactory  # Removed for now
from .tracking import MLflowTracker


class PredictionRequest(BaseModel):
    """Request model for trajectory prediction."""
    
    vehicle_id: str = Field(..., description="Vehicle identifier")
    current_state: Dict[str, float] = Field(..., description="Current vehicle state")
    prediction_horizon: int = Field(default=30, ge=1, le=100, description="Prediction horizon in seconds")
    model_name: Optional[str] = Field(default=None, description="Specific model to use")
    confidence_level: float = Field(default=0.95, ge=0.5, le=0.99, description="Confidence level for uncertainty")
    
    class Config:
        json_schema_extra = {
            "example": {
                "vehicle_id": "vehicle_001",
                "current_state": {
                    "x": 100.0,
                    "y": 200.0,
                    "velocity": 25.0,
                    "acceleration": 0.5,
                    "heading": 1.57,
                    "timestamp": "2024-01-01T12:00:00Z"
                },
                "prediction_horizon": 30,
                "model_name": "cv",
                "confidence_level": 0.95
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request model for batch trajectory prediction."""
    
    predictions: List[PredictionRequest] = Field(..., description="List of prediction requests")
    model_name: Optional[str] = Field(default=None, description="Model to use for all predictions")
    
    class Config:
        json_schema_extra = {
            "example": {
                "predictions": [
                    {
                        "vehicle_id": "vehicle_001",
                        "current_state": {
                            "x": 100.0,
                            "y": 200.0,
                            "velocity": 25.0,
                            "acceleration": 0.5,
                            "heading": 1.57
                        },
                        "prediction_horizon": 30
                    }
                ],
                "model_name": "cv"
            }
        }


class PredictionResponse(BaseModel):
    """Response model for trajectory prediction."""
    
    vehicle_id: str
    model_name: str
    prediction_horizon: int
    predicted_trajectory: List[Dict[str, float]]
    confidence_intervals: Optional[List[Dict[str, float]]] = None
    uncertainty_metrics: Optional[Dict[str, float]] = None
    inference_time: float
    timestamp: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "vehicle_id": "vehicle_001",
                "model_name": "cv",
                "prediction_horizon": 30,
                "predicted_trajectory": [
                    {"x": 100.0, "y": 200.0, "velocity": 25.0, "timestamp": 0.0},
                    {"x": 125.0, "y": 200.0, "velocity": 25.0, "timestamp": 1.0}
                ],
                "confidence_intervals": [
                    {"lower": 95.0, "upper": 105.0, "lower_y": 195.0, "upper_y": 205.0}
                ],
                "uncertainty_metrics": {"position_std": 2.5, "velocity_std": 1.0},
                "inference_time": 0.015,
                "timestamp": "2024-01-01T12:00:00Z"
            }
        }


class ModelInfo(BaseModel):
    """Model information response."""
    
    model_name: str
    model_type: str
    version: str
    description: str
    metrics: Dict[str, float]
    last_updated: str
    status: str


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str
    timestamp: str
    version: str
    models_loaded: int
    uptime: float


class PredictionService:
    """Service for handling trajectory predictions."""
    
    def __init__(self, config: Optional[Any] = None):
        """Initialize prediction service.
        
        Args:
            config: Configuration object
        """
        self.logger = get_logger(__name__)
        self.config = config or {}
        self.models: Dict[str, BaseTrajectoryPredictor] = {}
        # self.model_factory = ModelFactory()  # Removed for now
        self.tracker = MLflowTracker(config)
        self.start_time = time.time()
        
        # Load models
        # self._load_models()  # Removed for now
        
        self.logger.info("Prediction service initialized", models_loaded=len(self.models))
    
    def _load_models(self) -> None:
        """Load trained models."""
        try:
            models_path = Path(self.config.get("data", {}).get("models_path", "models"))
            if not models_path.exists():
                self.logger.warning("Models path does not exist", path=str(models_path))
                return
            
            # Load models from MLflow or local files
            for model_file in models_path.glob("*.pkl"):
                model_name = model_file.stem
                try:
                    with open(model_file, "rb") as f:
                        model = pickle.load(f)
                    self.models[model_name] = model
                    self.logger.info("Loaded model", model_name=model_name)
                except Exception as e:
                    self.logger.error("Failed to load model", model_name=model_name, error=str(e))
            
            # Try to load from MLflow if no local models
            if not self.models:
                self._load_models_from_mlflow()
                
        except Exception as e:
            self.logger.error("Failed to load models", error=str(e))
    
    def _load_models_from_mlflow(self) -> None:
        """Load models from MLflow registry."""
        try:
            # Get best model for each type
            model_types = ["cv", "ca", "polynomial", "knn", "gaussian_process", "ensemble"]
            
            for model_type in model_types:
                run_id = self.tracker.get_best_model("rmse")
                if run_id:
                    try:
                        model = self.tracker.load_model(run_id, model_type)
                        self.models[model_type] = model
                        self.logger.info("Loaded model from MLflow", model_type=model_type, run_id=run_id)
                    except Exception as e:
                        self.logger.error("Failed to load model from MLflow", model_type=model_type, error=str(e))
                        
        except Exception as e:
            self.logger.error("Failed to load models from MLflow", error=str(e))
    
    def predict_trajectory(self, request: PredictionRequest) -> PredictionResponse:
        """Predict trajectory for a single vehicle.
        
        Args:
            request: Prediction request
            
        Returns:
            Prediction response
        """
        start_time = time.time()
        
        try:
            # Determine model to use
            model_name = request.model_name or list(self.models.keys())[0]
            if model_name not in self.models:
                raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
            
            model = self.models[model_name]
            
            # Convert current state to trajectory point
            current_point = TrajectoryPoint(
                x=request.current_state["x"],
                y=request.current_state["y"],
                velocity=request.current_state.get("velocity", 0.0),
                acceleration=request.current_state.get("acceleration", 0.0),
                heading=request.current_state.get("heading", 0.0),
                timestamp=datetime.fromisoformat(request.current_state.get("timestamp", datetime.now().isoformat())),
                vehicle_id=request.vehicle_id
            )
            
            # Make prediction
            predicted_trajectory = model.predict(
                current_point, 
                horizon=request.prediction_horizon
            )
            
            # Convert to response format
            trajectory_points = []
            for i, point in enumerate(predicted_trajectory.points):
                trajectory_points.append({
                    "x": float(point.x),
                    "y": float(point.y),
                    "velocity": float(point.velocity),
                    "acceleration": float(point.acceleration),
                    "heading": float(point.heading),
                    "timestamp": float(i)  # Relative time from start
                })
            
            # Calculate confidence intervals if model supports it
            confidence_intervals = None
            uncertainty_metrics = None
            
            if hasattr(model, 'predict_with_uncertainty'):
                try:
                    mean_pred, std_pred = model.predict_with_uncertainty(
                        current_point, 
                        horizon=request.prediction_horizon
                    )
                    
                    confidence_intervals = []
                    for i, (mean, std) in enumerate(zip(mean_pred, std_pred)):
                        confidence_intervals.append({
                            "lower": float(mean[0] - 2 * std[0]),
                            "upper": float(mean[0] + 2 * std[0]),
                            "lower_y": float(mean[1] - 2 * std[1]),
                            "upper_y": float(mean[1] + 2 * std[1])
                        })
                    
                    uncertainty_metrics = {
                        "position_std": float(np.mean(std_pred[:, :2])),
                        "velocity_std": float(np.mean(std_pred[:, 2:]))
                    }
                except Exception as e:
                    self.logger.warning("Failed to calculate uncertainty", error=str(e))
            
            inference_time = time.time() - start_time
            
            return PredictionResponse(
                vehicle_id=request.vehicle_id,
                model_name=model_name,
                prediction_horizon=request.prediction_horizon,
                predicted_trajectory=trajectory_points,
                confidence_intervals=confidence_intervals,
                uncertainty_metrics=uncertainty_metrics,
                inference_time=inference_time,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            self.logger.error("Prediction failed", error=str(e), vehicle_id=request.vehicle_id)
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    def predict_batch(self, request: BatchPredictionRequest) -> List[PredictionResponse]:
        """Predict trajectories for multiple vehicles.
        
        Args:
            request: Batch prediction request
            
        Returns:
            List of prediction responses
        """
        responses = []
        
        for pred_request in request.predictions:
            # Override model name if specified in batch request
            if request.model_name:
                pred_request.model_name = request.model_name
            
            response = self.predict_trajectory(pred_request)
            responses.append(response)
        
        return responses
    
    def get_model_info(self) -> List[ModelInfo]:
        """Get information about loaded models.
        
        Returns:
            List of model information
        """
        model_info = []
        
        for model_name, model in self.models.items():
            info = ModelInfo(
                model_name=model_name,
                model_type=type(model).__name__,
                version="1.0.0",
                description=f"{model_name} trajectory prediction model",
                metrics={},  # Could be populated from evaluation results
                last_updated=datetime.now().isoformat(),
                status="loaded"
            )
            model_info.append(info)
        
        return model_info
    
    def get_health(self) -> HealthResponse:
        """Get service health information.
        
        Returns:
            Health response
        """
        uptime = time.time() - self.start_time
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            version="1.0.0",
            models_loaded=len(self.models),
            uptime=uptime
        )


# Global prediction service instance
prediction_service: Optional[PredictionService] = None


def get_prediction_service() -> PredictionService:
    """Get prediction service instance."""
    global prediction_service
    if prediction_service is None:
        prediction_service = PredictionService()
    return prediction_service


def create_app() -> FastAPI:
    """Create FastAPI application."""
    # Set up logging
    setup_logging(json_format=True)
    logger = get_logger(__name__)
    
    # Create FastAPI app
    app = FastAPI(
        title="Vehicle Trajectory Prediction API",
        description="Production API for vehicle trajectory prediction using advanced ML models",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.on_event("startup")
    async def startup_event():
        """Initialize services on startup."""
        logger.info("Starting Vehicle Trajectory Prediction API")
        get_prediction_service()  # Initialize prediction service
    
    @app.get("/", response_model=Dict[str, str])
    async def root():
        """Root endpoint."""
        return {
            "message": "Vehicle Trajectory Prediction API",
            "version": "1.0.0",
            "docs": "/docs"
        }
    
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint."""
        service = get_prediction_service()
        return service.get_health()
    
    @app.post("/predict", response_model=PredictionResponse)
    async def predict_trajectory(
        request: PredictionRequest,
        service: PredictionService = Depends(get_prediction_service)
    ):
        """Predict trajectory for a single vehicle."""
        return service.predict_trajectory(request)
    
    @app.post("/predict/batch", response_model=List[PredictionResponse])
    async def predict_batch(
        request: BatchPredictionRequest,
        service: PredictionService = Depends(get_prediction_service)
    ):
        """Predict trajectories for multiple vehicles."""
        return service.predict_batch(request)
    
    @app.get("/models", response_model=List[ModelInfo])
    async def list_models(
        service: PredictionService = Depends(get_prediction_service)
    ):
        """List available models."""
        return service.get_model_info()
    
    @app.get("/models/{model_name}", response_model=ModelInfo)
    async def get_model_info(
        model_name: str,
        service: PredictionService = Depends(get_prediction_service)
    ):
        """Get information about a specific model."""
        models = service.get_model_info()
        for model in models:
            if model.model_name == model_name:
                return model
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    @app.post("/models/reload")
    async def reload_models(
        background_tasks: BackgroundTasks,
        service: PredictionService = Depends(get_prediction_service)
    ):
        """Reload models from storage."""
        def reload():
            service._load_models()
        
        background_tasks.add_task(reload)
        return {"message": "Model reload started"}
    
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        """Global exception handler."""
        logger.error("Unhandled exception", error=str(exc), path=request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"}
        )
    
    logger.info("FastAPI application created")
    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)