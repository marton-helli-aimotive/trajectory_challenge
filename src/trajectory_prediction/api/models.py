"""
Pydantic models for API request/response validation.

This module defines the data models used for:
- Trajectory prediction requests and responses
- Batch processing
- API metadata and health checks
- Input validation and output formatting
"""

from typing import List, Optional, Dict, Any, Union
from datetime import datetime
import numpy as np
from pydantic import BaseModel, Field, validator, root_validator
from enum import Enum


class PredictionStatus(str, Enum):
    """Prediction status enumeration."""
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    CACHED = "cached"


class UncertaintyType(str, Enum):
    """Uncertainty quantification type."""
    NONE = "none"
    GAUSSIAN = "gaussian"
    ENSEMBLE = "ensemble"
    QUANTILE = "quantile"


class TrajectoryPoint(BaseModel):
    """Individual trajectory point with position and time."""
    x: float = Field(..., description="X coordinate in meters")
    y: float = Field(..., description="Y coordinate in meters")
    timestamp: float = Field(..., description="Time in seconds")
    
    @validator('timestamp')
    def validate_timestamp(cls, v):
        if v < 0:
            raise ValueError('Timestamp must be non-negative')
        return v


class TrajectoryInput(BaseModel):
    """Input trajectory for prediction."""
    vehicle_id: str = Field(..., description="Unique vehicle identifier")
    trajectory_points: List[TrajectoryPoint] = Field(
        ..., 
        min_items=2, 
        max_items=1000,
        description="Historical trajectory points (chronologically ordered)"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Additional trajectory metadata"
    )
    
    @root_validator
    def validate_chronological_order(cls, values):
        """Ensure trajectory points are in chronological order."""
        points = values.get('trajectory_points', [])
        if len(points) > 1:
            timestamps = [p.timestamp for p in points]
            if timestamps != sorted(timestamps):
                raise ValueError('Trajectory points must be in chronological order')
        return values
    
    @validator('trajectory_points')
    def validate_minimum_duration(cls, v):
        """Ensure minimum trajectory duration."""
        if len(v) >= 2:
            duration = v[-1].timestamp - v[0].timestamp
            if duration < 0.1:  # Minimum 0.1 seconds
                raise ValueError('Trajectory duration must be at least 0.1 seconds')
        return v


class PredictionConfig(BaseModel):
    """Configuration for prediction request."""
    prediction_horizon: float = Field(
        default=10.0,
        ge=0.1,
        le=30.0,
        description="Prediction horizon in seconds"
    )
    time_resolution: float = Field(
        default=0.1,
        ge=0.01,
        le=1.0,
        description="Time resolution for prediction output in seconds"
    )
    uncertainty_quantification: UncertaintyType = Field(
        default=UncertaintyType.GAUSSIAN,
        description="Type of uncertainty quantification"
    )
    confidence_level: float = Field(
        default=0.95,
        ge=0.5,
        le=0.99,
        description="Confidence level for uncertainty bounds"
    )
    return_intermediate_predictions: bool = Field(
        default=False,
        description="Whether to return intermediate prediction steps"
    )


class TrajectoryRequest(BaseModel):
    """Single trajectory prediction request."""
    request_id: Optional[str] = Field(
        default=None,
        description="Optional request identifier for tracking"
    )
    trajectory: TrajectoryInput = Field(..., description="Input trajectory data")
    config: PredictionConfig = Field(
        default_factory=PredictionConfig,
        description="Prediction configuration"
    )
    model_name: Optional[str] = Field(
        default=None,
        description="Specific model to use (uses default if not specified)"
    )
    use_ensemble: bool = Field(
        default=False,
        description="Whether to use ensemble prediction"
    )


class PredictedTrajectoryPoint(BaseModel):
    """Predicted trajectory point with uncertainty."""
    x: float = Field(..., description="Predicted X coordinate in meters")
    y: float = Field(..., description="Predicted Y coordinate in meters")
    timestamp: float = Field(..., description="Prediction time in seconds")
    uncertainty_x: Optional[float] = Field(
        default=None,
        description="Uncertainty in X coordinate (standard deviation)"
    )
    uncertainty_y: Optional[float] = Field(
        default=None,
        description="Uncertainty in Y coordinate (standard deviation)"
    )
    confidence_bounds: Optional[Dict[str, List[float]]] = Field(
        default=None,
        description="Confidence bounds for position"
    )


class PredictionMetadata(BaseModel):
    """Metadata about the prediction."""
    model_name: str = Field(..., description="Name of the model used")
    model_version: Optional[str] = Field(None, description="Version of the model")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")
    prediction_quality: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Quality score of the prediction (0-1)"
    )
    safety_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Safety assessment score (0-1)"
    )
    ensemble_details: Optional[Dict[str, Any]] = Field(
        None,
        description="Details about ensemble prediction if applicable"
    )


class TrajectoryResponse(BaseModel):
    """Single trajectory prediction response."""
    request_id: Optional[str] = Field(None, description="Request identifier")
    status: PredictionStatus = Field(..., description="Prediction status")
    predicted_trajectory: Optional[List[PredictedTrajectoryPoint]] = Field(
        None,
        description="Predicted trajectory points"
    )
    metadata: PredictionMetadata = Field(..., description="Prediction metadata")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    warnings: List[str] = Field(
        default_factory=list,
        description="Warning messages"
    )
    processed_at: datetime = Field(
        default_factory=datetime.now,
        description="When the prediction was processed"
    )


class BatchTrajectoryRequest(BaseModel):
    """Batch trajectory prediction request."""
    batch_id: Optional[str] = Field(
        default=None,
        description="Optional batch identifier"
    )
    trajectories: List[TrajectoryRequest] = Field(
        ...,
        min_items=1,
        max_items=100,
        description="List of trajectory prediction requests"
    )
    parallel_processing: bool = Field(
        default=True,
        description="Whether to process trajectories in parallel"
    )
    timeout_seconds: Optional[float] = Field(
        default=60.0,
        ge=1.0,
        le=300.0,
        description="Timeout for batch processing"
    )


class BatchTrajectoryResponse(BaseModel):
    """Batch trajectory prediction response."""
    batch_id: Optional[str] = Field(None, description="Batch identifier")
    status: PredictionStatus = Field(..., description="Overall batch status")
    results: List[TrajectoryResponse] = Field(
        ...,
        description="Individual prediction results"
    )
    summary: Dict[str, Any] = Field(
        ...,
        description="Batch processing summary"
    )
    total_processing_time_ms: float = Field(
        ...,
        description="Total batch processing time in milliseconds"
    )
    processed_at: datetime = Field(
        default_factory=datetime.now,
        description="When the batch was processed"
    )


class ModelInfo(BaseModel):
    """Information about available models."""
    model_name: str = Field(..., description="Model name")
    model_version: str = Field(..., description="Model version")
    model_type: str = Field(..., description="Type of model")
    description: str = Field(..., description="Model description")
    supported_features: List[str] = Field(
        ...,
        description="List of supported features"
    )
    performance_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Model performance metrics"
    )
    is_default: bool = Field(
        default=False,
        description="Whether this is the default model"
    )
    status: str = Field(..., description="Model status (active, maintenance, etc.)")


class HealthStatus(BaseModel):
    """API health status."""
    status: str = Field(..., description="Overall health status")
    version: str = Field(..., description="API version")
    uptime_seconds: float = Field(..., description="Uptime in seconds")
    models_loaded: int = Field(..., description="Number of loaded models")
    total_predictions: int = Field(..., description="Total predictions served")
    cache_hit_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Cache hit rate"
    )
    average_response_time_ms: float = Field(
        ...,
        description="Average response time in milliseconds"
    )
    system_info: Dict[str, Any] = Field(
        ...,
        description="System information"
    )
    last_health_check: datetime = Field(
        default_factory=datetime.now,
        description="Last health check timestamp"
    )


class ErrorResponse(BaseModel):
    """Standard error response."""
    error_code: str = Field(..., description="Error code")
    error_message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional error details"
    )
    request_id: Optional[str] = Field(None, description="Request identifier")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Error timestamp"
    )


class CacheStats(BaseModel):
    """Prediction cache statistics."""
    cache_size: int = Field(..., description="Current cache size")
    max_cache_size: int = Field(..., description="Maximum cache size")
    hit_count: int = Field(..., description="Number of cache hits")
    miss_count: int = Field(..., description="Number of cache misses")
    hit_rate: float = Field(..., description="Cache hit rate")
    total_requests: int = Field(..., description="Total requests processed")
    eviction_count: int = Field(..., description="Number of cache evictions")
    average_lookup_time_ms: float = Field(
        ...,
        description="Average cache lookup time"
    )


class LoadTestConfig(BaseModel):
    """Load testing configuration."""
    concurrent_users: int = Field(
        ...,
        ge=1,
        le=1000,
        description="Number of concurrent users"
    )
    requests_per_user: int = Field(
        ...,
        ge=1,
        le=1000,
        description="Number of requests per user"
    )
    ramp_up_time_seconds: float = Field(
        ...,
        ge=1.0,
        le=300.0,
        description="Ramp-up time in seconds"
    )
    test_data_size: int = Field(
        default=100,
        ge=10,
        le=10000,
        description="Size of test dataset"
    )
    target_percentile: float = Field(
        default=95.0,
        ge=50.0,
        le=99.9,
        description="Target percentile for response time"
    )


class LoadTestResult(BaseModel):
    """Load testing results."""
    test_id: str = Field(..., description="Test identifier")
    config: LoadTestConfig = Field(..., description="Test configuration")
    total_requests: int = Field(..., description="Total requests sent")
    successful_requests: int = Field(..., description="Successful requests")
    failed_requests: int = Field(..., description="Failed requests")
    success_rate: float = Field(..., description="Success rate")
    average_response_time_ms: float = Field(
        ...,
        description="Average response time"
    )
    percentile_response_times: Dict[str, float] = Field(
        ...,
        description="Percentile response times"
    )
    max_response_time_ms: float = Field(..., description="Maximum response time")
    min_response_time_ms: float = Field(..., description="Minimum response time")
    requests_per_second: float = Field(..., description="Requests per second")
    test_duration_seconds: float = Field(..., description="Total test duration")
    errors: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of errors encountered"
    )
    started_at: datetime = Field(..., description="Test start time")
    completed_at: datetime = Field(..., description="Test completion time")


class ServerMetrics(BaseModel):
    """Server performance metrics."""
    cpu_usage_percent: float = Field(..., description="CPU usage percentage")
    memory_usage_percent: float = Field(..., description="Memory usage percentage")
    disk_usage_percent: float = Field(..., description="Disk usage percentage")
    network_io: Dict[str, float] = Field(..., description="Network I/O statistics")
    active_connections: int = Field(..., description="Active connections")
    request_queue_size: int = Field(..., description="Request queue size")
    thread_pool_size: int = Field(..., description="Thread pool size")
    garbage_collection_time: float = Field(
        ...,
        description="Garbage collection time"
    )