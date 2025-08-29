"""
FastAPI server implementation for trajectory prediction API.

This module provides:
- RESTful prediction endpoints (single and batch)
- Model management and serving
- Health checks and monitoring
- Request validation and error handling
- Performance optimization middleware
"""

import asyncio
import time
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import traceback
import uuid
import psutil
import os

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from omegaconf import DictConfig

from .models import (
    TrajectoryRequest, TrajectoryResponse, BatchTrajectoryRequest, 
    BatchTrajectoryResponse, PredictionStatus, HealthStatus, ModelInfo,
    ErrorResponse, CacheStats, ServerMetrics, LoadTestConfig, LoadTestResult
)
from .ensemble import EnsemblePredictor
from .caching import PredictionCache
from .optimization import RequestBatcher, InferenceOptimizer
from ..models.base import TrajectoryPredictor
from ..data.schemas import TrajectoryData
from ..mlops.versioning import ModelRegistry

logger = logging.getLogger(__name__)


class TrajectoryPredictionAPI:
    """
    Main trajectory prediction API class.
    
    Orchestrates model serving, caching, batching, and optimization.
    """
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.api_config = config.get("api", {})
        
        # API metadata
        self.version = self.api_config.get("version", "1.0.0")
        self.start_time = time.time()
        
        # Model management
        self.models: Dict[str, TrajectoryPredictor] = {}
        self.default_model_name = None
        self.model_registry = None
        
        # Performance optimization
        self.prediction_cache = None
        self.request_batcher = None
        self.inference_optimizer = None
        self.ensemble_predictor = None
        
        # Metrics and monitoring
        self.total_predictions = 0
        self.total_batch_predictions = 0
        self.response_times = []
        self.error_count = 0
        
        logger.info("Trajectory Prediction API initialized")
    
    async def initialize(self) -> None:
        """Initialize API components."""
        
        try:
            # Initialize caching
            cache_config = self.api_config.get("caching", {})
            if cache_config.get("enabled", True):
                self.prediction_cache = PredictionCache(cache_config)
                logger.info("Prediction cache initialized")
            
            # Initialize request batching
            batch_config = self.api_config.get("batching", {})
            if batch_config.get("enabled", True):
                self.request_batcher = RequestBatcher(batch_config)
                logger.info("Request batcher initialized")
            
            # Initialize inference optimizer
            optimization_config = self.api_config.get("optimization", {})
            self.inference_optimizer = InferenceOptimizer(optimization_config)
            logger.info("Inference optimizer initialized")
            
            # Initialize model registry
            if "model_registry" in self.config:
                from ..mlops.versioning import ModelRegistry
                self.model_registry = ModelRegistry(self.config)
                logger.info("Model registry connected")
            
            # Load models
            await self._load_models()
            
            # Initialize ensemble predictor if multiple models loaded
            if len(self.models) > 1:
                ensemble_config = self.api_config.get("ensemble", {})
                if ensemble_config.get("enabled", False):
                    self.ensemble_predictor = EnsemblePredictor(
                        list(self.models.values()), ensemble_config
                    )
                    logger.info("Ensemble predictor initialized")
            
        except Exception as e:
            logger.error(f"API initialization failed: {e}")
            raise
    
    async def _load_models(self) -> None:
        """Load models from registry or configuration."""
        
        model_config = self.api_config.get("models", {})
        
        # Load from model registry if available
        if self.model_registry:
            try:
                production_models = self.model_registry.get_production_models()
                
                for model_name, model_version in production_models.items():
                    model = await self.model_registry.get_model(model_name, model_version.version)
                    self.models[model_name] = model
                    
                    if not self.default_model_name:
                        self.default_model_name = model_name
                    
                    logger.info(f"Loaded model from registry: {model_name} v{model_version.version}")
                
            except Exception as e:
                logger.warning(f"Failed to load models from registry: {e}")
        
        # Load from configuration as fallback
        if not self.models and "model_paths" in model_config:
            for model_name, model_path in model_config["model_paths"].items():
                try:
                    # Placeholder for model loading - would load from file
                    # model = load_model(model_path)
                    # self.models[model_name] = model
                    logger.info(f"Model configuration found for: {model_name}")
                except Exception as e:
                    logger.error(f"Failed to load model {model_name}: {e}")
        
        if not self.models:
            logger.warning("No models loaded - API will run in demo mode")
        
        logger.info(f"Total models loaded: {len(self.models)}")
    
    async def predict_single(self, request: TrajectoryRequest) -> TrajectoryResponse:
        """Process single prediction request."""
        
        start_time = time.time()
        request_id = request.request_id or str(uuid.uuid4())
        
        try:
            # Convert request to internal format
            trajectory_data = await self._convert_request_to_trajectory_data(request)
            
            # Check cache first
            cache_key = None
            if self.prediction_cache:
                cache_key = self.prediction_cache.generate_cache_key(
                    trajectory_data, request.config.prediction_horizon
                )
                cached_result = await self.prediction_cache.get(cache_key)
                if cached_result:
                    logger.debug(f"Cache hit for request {request_id}")
                    return self._create_cached_response(request_id, cached_result, start_time)
            
            # Select model
            model = await self._select_model(request)
            
            # Make prediction
            if request.use_ensemble and self.ensemble_predictor:
                prediction_result = await self.ensemble_predictor.predict(
                    trajectory_data, request.config.prediction_horizon
                )
                model_name = "ensemble"
            else:
                prediction_result = await model.predict_trajectory(
                    trajectory_data, request.config.prediction_horizon
                )
                model_name = model.name
            
            # Convert result to response format
            response = await self._convert_prediction_to_response(
                request_id, prediction_result, model_name, start_time
            )
            
            # Cache result
            if self.prediction_cache and cache_key:
                await self.prediction_cache.set(cache_key, response, ttl=300)  # 5 minutes
            
            # Update metrics
            self.total_predictions += 1
            inference_time = (time.time() - start_time) * 1000
            self.response_times.append(inference_time)
            
            # Keep only last 1000 response times
            if len(self.response_times) > 1000:
                self.response_times = self.response_times[-1000:]
            
            return response
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Prediction failed for request {request_id}: {e}")
            
            return TrajectoryResponse(
                request_id=request_id,
                status=PredictionStatus.FAILED,
                predicted_trajectory=None,
                metadata={
                    "model_name": "unknown",
                    "inference_time_ms": (time.time() - start_time) * 1000,
                    "prediction_quality": None,
                    "safety_score": None
                },
                error_message=str(e),
                warnings=[],
                processed_at=datetime.now()
            )
    
    async def predict_batch(self, request: BatchTrajectoryRequest) -> BatchTrajectoryResponse:
        """Process batch prediction request."""
        
        start_time = time.time()
        batch_id = request.batch_id or str(uuid.uuid4())
        
        try:
            if request.parallel_processing:
                # Process requests in parallel
                tasks = [self.predict_single(req) for req in request.trajectories]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Convert exceptions to error responses
                processed_results = []
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        error_response = TrajectoryResponse(
                            request_id=request.trajectories[i].request_id,
                            status=PredictionStatus.FAILED,
                            predicted_trajectory=None,
                            metadata={
                                "model_name": "unknown",
                                "inference_time_ms": 0.0,
                                "prediction_quality": None,
                                "safety_score": None
                            },
                            error_message=str(result),
                            processed_at=datetime.now()
                        )
                        processed_results.append(error_response)
                    else:
                        processed_results.append(result)
            else:
                # Process requests sequentially
                processed_results = []
                for req in request.trajectories:
                    result = await self.predict_single(req)
                    processed_results.append(result)
            
            # Generate batch summary
            successful_count = len([r for r in processed_results if r.status == PredictionStatus.SUCCESS])
            failed_count = len(processed_results) - successful_count
            
            batch_status = PredictionStatus.SUCCESS
            if failed_count > 0:
                batch_status = PredictionStatus.PARTIAL if successful_count > 0 else PredictionStatus.FAILED
            
            total_time = (time.time() - start_time) * 1000
            
            summary = {
                "total_requests": len(request.trajectories),
                "successful_predictions": successful_count,
                "failed_predictions": failed_count,
                "success_rate": successful_count / len(request.trajectories),
                "parallel_processing": request.parallel_processing,
                "average_processing_time_ms": total_time / len(request.trajectories)
            }
            
            self.total_batch_predictions += len(request.trajectories)
            
            return BatchTrajectoryResponse(
                batch_id=batch_id,
                status=batch_status,
                results=processed_results,
                summary=summary,
                total_processing_time_ms=total_time,
                processed_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Batch prediction failed for batch {batch_id}: {e}")
            
            return BatchTrajectoryResponse(
                batch_id=batch_id,
                status=PredictionStatus.FAILED,
                results=[],
                summary={
                    "total_requests": len(request.trajectories),
                    "error": str(e)
                },
                total_processing_time_ms=(time.time() - start_time) * 1000,
                processed_at=datetime.now()
            )
    
    async def _convert_request_to_trajectory_data(self, request: TrajectoryRequest) -> TrajectoryData:
        """Convert API request to internal TrajectoryData format."""
        
        positions = [[point.x, point.y] for point in request.trajectory.trajectory_points]
        time_steps = [point.timestamp for point in request.trajectory.trajectory_points]
        
        return TrajectoryData(
            vehicle_id=request.trajectory.vehicle_id,
            positions=positions,
            time_steps=time_steps,
            metadata=request.trajectory.metadata
        )
    
    async def _select_model(self, request: TrajectoryRequest) -> TrajectoryPredictor:
        """Select appropriate model for prediction."""
        
        if request.model_name and request.model_name in self.models:
            return self.models[request.model_name]
        
        if self.default_model_name and self.default_model_name in self.models:
            return self.models[self.default_model_name]
        
        if self.models:
            return next(iter(self.models.values()))
        
        raise ValueError("No models available for prediction")
    
    async def _convert_prediction_to_response(
        self,
        request_id: str,
        prediction_result: Any,
        model_name: str,
        start_time: float
    ) -> TrajectoryResponse:
        """Convert prediction result to API response format."""
        
        # This would be implemented based on the actual PredictionResult structure
        # For now, creating a placeholder response
        
        from .models import PredictedTrajectoryPoint, PredictionMetadata
        
        inference_time = (time.time() - start_time) * 1000
        
        # Placeholder predicted trajectory
        predicted_points = []
        # In real implementation, this would extract from prediction_result
        
        metadata = PredictionMetadata(
            model_name=model_name,
            model_version="1.0.0",  # Would get from model
            inference_time_ms=inference_time,
            prediction_quality=0.85,  # Would calculate from prediction
            safety_score=0.92  # Would calculate safety metrics
        )
        
        return TrajectoryResponse(
            request_id=request_id,
            status=PredictionStatus.SUCCESS,
            predicted_trajectory=predicted_points,
            metadata=metadata,
            warnings=[],
            processed_at=datetime.now()
        )
    
    def _create_cached_response(
        self, 
        request_id: str, 
        cached_result: TrajectoryResponse, 
        start_time: float
    ) -> TrajectoryResponse:
        """Create response from cached result."""
        
        cached_result.request_id = request_id
        cached_result.status = PredictionStatus.CACHED
        cached_result.metadata.inference_time_ms = (time.time() - start_time) * 1000
        cached_result.processed_at = datetime.now()
        
        return cached_result
    
    def get_health_status(self) -> HealthStatus:
        """Get API health status."""
        
        uptime = time.time() - self.start_time
        
        # Calculate cache hit rate
        cache_hit_rate = 0.0
        if self.prediction_cache:
            stats = self.prediction_cache.get_stats()
            cache_hit_rate = stats.get("hit_rate", 0.0)
        
        # Calculate average response time
        avg_response_time = (
            sum(self.response_times) / len(self.response_times) 
            if self.response_times else 0.0
        )
        
        # System information
        system_info = {
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "memory_available_gb": psutil.virtual_memory().available / (1024**3),
            "disk_usage_percent": psutil.disk_usage('/').percent,
            "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}"
        }
        
        return HealthStatus(
            status="healthy" if len(self.models) > 0 else "degraded",
            version=self.version,
            uptime_seconds=uptime,
            models_loaded=len(self.models),
            total_predictions=self.total_predictions + self.total_batch_predictions,
            cache_hit_rate=cache_hit_rate,
            average_response_time_ms=avg_response_time,
            system_info=system_info,
            last_health_check=datetime.now()
        )
    
    def get_model_info(self) -> List[ModelInfo]:
        """Get information about loaded models."""
        
        model_infos = []
        
        for model_name, model in self.models.items():
            model_info = ModelInfo(
                model_name=model_name,
                model_version=getattr(model, 'version', "1.0.0"),
                model_type=model.__class__.__name__,
                description=getattr(model, 'description', f"{model.__class__.__name__} trajectory predictor"),
                supported_features=getattr(model, 'supported_features', ["trajectory_prediction"]),
                performance_metrics=getattr(model, 'performance_metrics', {}),
                is_default=(model_name == self.default_model_name),
                status="active"
            )
            model_infos.append(model_info)
        
        return model_infos
    
    def get_cache_stats(self) -> Optional[CacheStats]:
        """Get prediction cache statistics."""
        
        if not self.prediction_cache:
            return None
        
        stats = self.prediction_cache.get_stats()
        
        return CacheStats(
            cache_size=stats.get("cache_size", 0),
            max_cache_size=stats.get("max_cache_size", 0),
            hit_count=stats.get("hit_count", 0),
            miss_count=stats.get("miss_count", 0),
            hit_rate=stats.get("hit_rate", 0.0),
            total_requests=stats.get("total_requests", 0),
            eviction_count=stats.get("eviction_count", 0),
            average_lookup_time_ms=stats.get("average_lookup_time_ms", 0.0)
        )
    
    def get_server_metrics(self) -> ServerMetrics:
        """Get server performance metrics."""
        
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        network = psutil.net_io_counters()
        
        return ServerMetrics(
            cpu_usage_percent=cpu_usage,
            memory_usage_percent=memory.percent,
            disk_usage_percent=disk.percent,
            network_io={
                "bytes_sent": float(network.bytes_sent),
                "bytes_recv": float(network.bytes_recv),
                "packets_sent": float(network.packets_sent),
                "packets_recv": float(network.packets_recv)
            },
            active_connections=len(psutil.net_connections()),
            request_queue_size=0,  # Would need to track this
            thread_pool_size=os.cpu_count() or 4,
            garbage_collection_time=0.0  # Would need to track this
        )


# Global API instance
api_instance: Optional[TrajectoryPredictionAPI] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global api_instance
    
    # Startup
    logger.info("Starting Trajectory Prediction API...")
    
    # Initialize API instance (would get config from environment)
    from omegaconf import DictConfig
    config = DictConfig({
        "api": {
            "version": "1.0.0",
            "caching": {"enabled": True},
            "batching": {"enabled": True},
            "optimization": {"enabled": True},
            "ensemble": {"enabled": False}
        }
    })
    
    api_instance = TrajectoryPredictionAPI(config)
    await api_instance.initialize()
    
    logger.info("Trajectory Prediction API started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Trajectory Prediction API...")


# Create FastAPI application
app = FastAPI(
    title="Trajectory Prediction API",
    description="High-performance API for trajectory prediction in autonomous vehicles",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


# Dependency to get API instance
async def get_api_instance() -> TrajectoryPredictionAPI:
    global api_instance
    if api_instance is None:
        raise HTTPException(status_code=503, detail="API not initialized")
    return api_instance


# Exception handlers
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error_code="VALIDATION_ERROR",
            error_message=str(exc),
            timestamp=datetime.now()
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error_code="INTERNAL_ERROR",
            error_message="Internal server error",
            details={"exception_type": type(exc).__name__},
            timestamp=datetime.now()
        ).dict()
    )


# API Routes
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Trajectory Prediction API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }


@app.post("/predict", response_model=TrajectoryResponse)
async def predict_trajectory(
    request: TrajectoryRequest,
    api: TrajectoryPredictionAPI = Depends(get_api_instance)
):
    """Predict single trajectory."""
    return await api.predict_single(request)


@app.post("/predict/batch", response_model=BatchTrajectoryResponse)
async def predict_trajectory_batch(
    request: BatchTrajectoryRequest,
    api: TrajectoryPredictionAPI = Depends(get_api_instance)
):
    """Predict multiple trajectories in batch."""
    return await api.predict_batch(request)


@app.get("/health", response_model=HealthStatus)
async def health_check(api: TrajectoryPredictionAPI = Depends(get_api_instance)):
    """Get API health status."""
    return api.get_health_status()


@app.get("/models", response_model=List[ModelInfo])
async def list_models(api: TrajectoryPredictionAPI = Depends(get_api_instance)):
    """Get information about available models."""
    return api.get_model_info()


@app.get("/cache/stats", response_model=Optional[CacheStats])
async def get_cache_statistics(api: TrajectoryPredictionAPI = Depends(get_api_instance)):
    """Get prediction cache statistics."""
    return api.get_cache_stats()


@app.delete("/cache")
async def clear_cache(api: TrajectoryPredictionAPI = Depends(get_api_instance)):
    """Clear prediction cache."""
    if api.prediction_cache:
        await api.prediction_cache.clear()
        return {"message": "Cache cleared successfully"}
    else:
        raise HTTPException(status_code=404, detail="Cache not enabled")


@app.get("/metrics", response_model=ServerMetrics)
async def get_server_metrics(api: TrajectoryPredictionAPI = Depends(get_api_instance)):
    """Get server performance metrics."""
    return api.get_server_metrics()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "trajectory_prediction.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )