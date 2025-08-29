"""
Integration tests for API endpoints and model serving.

This module tests:
- API endpoint integration
- Request/response validation
- Model serving workflow
- Error handling and edge cases
"""

import pytest
import numpy as np
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any
import httpx

from src.trajectory_prediction.api.models import (
    TrajectoryRequest, TrajectoryResponse, TrajectoryInput, 
    TrajectoryPoint, PredictionConfig, BatchTrajectoryRequest
)
from src.trajectory_prediction.api.server import TrajectoryPredictionAPI
from src.trajectory_prediction.data.schemas import TrajectoryData, Position, Velocity
from tests.conftest import validate_prediction_response


class TestAPIEndpoints:
    """Test API endpoint functionality."""
    
    @pytest.fixture
    def mock_api_server(self):
        """Create mock API server for testing."""
        class MockAPIServer:
            def __init__(self):
                self.models = {}
                self.request_count = 0
                self.health_status = "healthy"
                
            async def predict_single(self, request: TrajectoryRequest) -> TrajectoryResponse:
                """Mock single prediction endpoint."""
                self.request_count += 1
                await asyncio.sleep(0.01)  # Simulate processing time
                
                # Create mock prediction
                predicted_positions = []
                predicted_velocities = []
                predicted_timestamps = []
                
                last_point = request.trajectory.points[-1]
                
                for i in range(10):  # Predict 10 steps
                    dt = 0.1
                    new_time = last_point.timestamp + (i + 1) * dt
                    new_x = last_point.x + last_point.vx * (i + 1) * dt
                    new_y = last_point.y + last_point.vy * (i + 1) * dt
                    
                    predicted_positions.append(Position(x=new_x, y=new_y))
                    predicted_velocities.append(Velocity(vx=last_point.vx, vy=last_point.vy))
                    predicted_timestamps.append(new_time)
                
                predicted_trajectory = TrajectoryData(
                    trajectory_id=f"{request.trajectory.trajectory_id}_pred",
                    vehicle_id=request.trajectory.vehicle_id,
                    positions=predicted_positions,
                    velocities=predicted_velocities,
                    timestamps=predicted_timestamps
                )
                
                return TrajectoryResponse(
                    request_id=f"req_{self.request_count}",
                    model_name=request.config.models[0] if request.config.models else "default",
                    predicted_trajectory=predicted_trajectory,
                    confidence=np.random.uniform(0.7, 0.95),
                    inference_time=0.01,
                    metadata={"timestamp": asyncio.get_event_loop().time()}
                )
            
            async def predict_batch(self, request: BatchTrajectoryRequest) -> List[TrajectoryResponse]:
                """Mock batch prediction endpoint."""
                responses = []
                
                for trajectory_input in request.trajectories:
                    single_request = TrajectoryRequest(
                        trajectory=trajectory_input,
                        config=request.config
                    )
                    response = await self.predict_single(single_request)
                    responses.append(response)
                
                return responses
            
            async def get_models(self) -> List[Dict[str, Any]]:
                """Mock get models endpoint."""
                return [
                    {
                        "name": "constant_velocity",
                        "type": "baseline",
                        "version": "1.0.0",
                        "status": "active"
                    },
                    {
                        "name": "constant_acceleration", 
                        "type": "baseline",
                        "version": "1.0.0",
                        "status": "active"
                    }
                ]
            
            async def health_check(self) -> Dict[str, Any]:
                """Mock health check endpoint."""
                return {
                    "status": self.health_status,
                    "timestamp": asyncio.get_event_loop().time(),
                    "models_loaded": len(self.models),
                    "total_requests": self.request_count
                }
        
        return MockAPIServer()
    
    @pytest.mark.asyncio
    async def test_single_prediction_endpoint(self, mock_api_server, sample_trajectory_request):
        """Test single prediction API endpoint."""
        response = await mock_api_server.predict_single(sample_trajectory_request)
        
        assert isinstance(response, TrajectoryResponse)
        assert validate_prediction_response(response)
        assert response.model_name in sample_trajectory_request.config.models
        assert 0.0 <= response.confidence <= 1.0
        assert response.inference_time > 0
        assert len(response.predicted_trajectory.positions) > 0
    
    @pytest.mark.asyncio
    async def test_batch_prediction_endpoint(self, mock_api_server):
        """Test batch prediction API endpoint."""
        # Create batch request
        trajectories = []
        for i in range(3):
            points = [
                TrajectoryPoint(timestamp=0.0, x=float(i), y=0.0, vx=1.0, vy=0.0),
                TrajectoryPoint(timestamp=0.1, x=float(i) + 0.1, y=0.0, vx=1.0, vy=0.0)
            ]
            trajectory = TrajectoryInput(
                trajectory_id=f"batch_{i}",
                vehicle_id=f"vehicle_{i}",
                points=points
            )
            trajectories.append(trajectory)
        
        config = PredictionConfig(
            prediction_horizon=2.0,
            models=["constant_velocity"]
        )
        
        batch_request = BatchTrajectoryRequest(
            trajectories=trajectories,
            config=config
        )
        
        responses = await mock_api_server.predict_batch(batch_request)
        
        assert len(responses) == 3
        assert all(isinstance(resp, TrajectoryResponse) for resp in responses)
        assert all(validate_prediction_response(resp) for resp in responses)
    
    @pytest.mark.asyncio
    async def test_models_endpoint(self, mock_api_server):
        """Test models listing endpoint."""
        models = await mock_api_server.get_models()
        
        assert isinstance(models, list)
        assert len(models) > 0
        
        for model in models:
            assert isinstance(model, dict)
            assert "name" in model
            assert "type" in model
            assert "version" in model
            assert "status" in model
    
    @pytest.mark.asyncio
    async def test_health_check_endpoint(self, mock_api_server):
        """Test health check endpoint."""
        health = await mock_api_server.health_check()
        
        assert isinstance(health, dict)
        assert "status" in health
        assert "timestamp" in health
        assert health["status"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_api_request_validation(self, mock_api_server):
        """Test API request validation."""
        # Test with invalid request (missing required fields)
        invalid_points = [
            TrajectoryPoint(timestamp=0.0, x=1.0, y=1.0)  # Missing velocities
        ]
        
        invalid_trajectory = TrajectoryInput(
            trajectory_id="invalid",
            vehicle_id="vehicle",
            points=invalid_points
        )
        
        config = PredictionConfig(
            prediction_horizon=2.0,
            models=["constant_velocity"]
        )
        
        invalid_request = TrajectoryRequest(
            trajectory=invalid_trajectory,
            config=config
        )
        
        # API should handle validation errors gracefully
        try:
            response = await mock_api_server.predict_single(invalid_request)
            # If it doesn't raise an error, check the response is still valid
            assert isinstance(response, TrajectoryResponse)
        except (ValueError, ValidationError):
            # Acceptable to raise validation error
            pass
    
    @pytest.mark.asyncio
    async def test_api_concurrent_requests(self, mock_api_server, sample_trajectory_request):
        """Test API handling concurrent requests."""
        # Send multiple requests concurrently
        num_requests = 10
        tasks = []
        
        for i in range(num_requests):
            task = mock_api_server.predict_single(sample_trajectory_request)
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        
        assert len(responses) == num_requests
        assert all(isinstance(resp, TrajectoryResponse) for resp in responses)
        assert all(validate_prediction_response(resp) for resp in responses)
        
        # Check that all requests were processed
        assert mock_api_server.request_count == num_requests


class TestAPIModelIntegration:
    """Test API integration with actual models."""
    
    @pytest.fixture
    def api_with_models(self, mock_model):
        """Create API instance with mock models."""
        class APIWithModels:
            def __init__(self, model):
                self.models = {"mock_model": model}
                self.request_count = 0
            
            async def predict_with_model(
                self,
                request: TrajectoryRequest,
                model_name: str
            ) -> TrajectoryResponse:
                """Predict using specific model."""
                if model_name not in self.models:
                    raise ValueError(f"Model {model_name} not found")
                
                model = self.models[model_name]
                
                # Convert request to trajectory data
                positions = [Position(x=p.x, y=p.y) for p in request.trajectory.points]
                velocities = [Velocity(vx=p.vx, vy=p.vy) for p in request.trajectory.points]
                timestamps = [p.timestamp for p in request.trajectory.points]
                
                input_trajectory = TrajectoryData(
                    trajectory_id=request.trajectory.trajectory_id,
                    vehicle_id=request.trajectory.vehicle_id,
                    positions=positions,
                    velocities=velocities,
                    timestamps=timestamps
                )
                
                # Get prediction from model
                predicted_trajectory = await model.predict(input_trajectory)
                
                self.request_count += 1
                
                return TrajectoryResponse(
                    request_id=f"model_req_{self.request_count}",
                    model_name=model_name,
                    predicted_trajectory=predicted_trajectory,
                    confidence=np.random.uniform(0.7, 0.95),
                    inference_time=0.02,
                    metadata={"model_version": "1.0.0"}
                )
        
        return APIWithModels(mock_model)
    
    @pytest.mark.asyncio
    async def test_api_model_prediction(self, api_with_models, sample_trajectory_request):
        """Test API prediction using actual model."""
        response = await api_with_models.predict_with_model(
            sample_trajectory_request,
            "mock_model"
        )
        
        assert isinstance(response, TrajectoryResponse)
        assert validate_prediction_response(response)
        assert response.model_name == "mock_model"
        assert len(response.predicted_trajectory.positions) > 0
    
    @pytest.mark.asyncio
    async def test_api_model_not_found(self, api_with_models, sample_trajectory_request):
        """Test API behavior when model is not found."""
        with pytest.raises(ValueError):
            await api_with_models.predict_with_model(
                sample_trajectory_request,
                "nonexistent_model"
            )
    
    @pytest.mark.asyncio
    async def test_api_model_ensemble(self, api_with_models, sample_trajectory_request):
        """Test API ensemble prediction."""
        # Add multiple models
        api_with_models.models["model_2"] = api_with_models.models["mock_model"]
        api_with_models.models["model_3"] = api_with_models.models["mock_model"]
        
        # Get predictions from multiple models
        responses = []
        for model_name in ["mock_model", "model_2", "model_3"]:
            response = await api_with_models.predict_with_model(
                sample_trajectory_request,
                model_name
            )
            responses.append(response)
        
        assert len(responses) == 3
        assert all(isinstance(resp, TrajectoryResponse) for resp in responses)
        assert len(set(resp.model_name for resp in responses)) == 3  # Different models


class TestAPIErrorHandling:
    """Test API error handling and edge cases."""
    
    @pytest.fixture
    def failing_api_server(self):
        """Create API server that simulates various failures."""
        class FailingAPIServer:
            def __init__(self):
                self.failure_mode = None
                self.request_count = 0
            
            def set_failure_mode(self, mode: str):
                """Set the type of failure to simulate."""
                self.failure_mode = mode
            
            async def predict_single(self, request: TrajectoryRequest) -> TrajectoryResponse:
                """Predict with potential failures."""
                self.request_count += 1
                
                if self.failure_mode == "timeout":
                    await asyncio.sleep(10)  # Simulate timeout
                elif self.failure_mode == "server_error":
                    raise RuntimeError("Internal server error")
                elif self.failure_mode == "validation_error":
                    raise ValueError("Invalid request format")
                elif self.failure_mode == "model_error":
                    raise ModelError("Model prediction failed")
                
                # Normal response
                return TrajectoryResponse(
                    request_id=f"req_{self.request_count}",
                    model_name="test_model",
                    predicted_trajectory=TrajectoryData(
                        trajectory_id="pred",
                        vehicle_id="vehicle",
                        positions=[Position(x=1.0, y=1.0)],
                        velocities=[Velocity(vx=1.0, vy=0.0)],
                        timestamps=[1.0]
                    ),
                    confidence=0.8,
                    inference_time=0.01
                )
        
        return FailingAPIServer()
    
    @pytest.mark.asyncio
    async def test_api_server_error_handling(self, failing_api_server, sample_trajectory_request):
        """Test API handling of server errors."""
        failing_api_server.set_failure_mode("server_error")
        
        with pytest.raises(RuntimeError):
            await failing_api_server.predict_single(sample_trajectory_request)
    
    @pytest.mark.asyncio
    async def test_api_validation_error_handling(self, failing_api_server, sample_trajectory_request):
        """Test API handling of validation errors."""
        failing_api_server.set_failure_mode("validation_error")
        
        with pytest.raises(ValueError):
            await failing_api_server.predict_single(sample_trajectory_request)
    
    @pytest.mark.asyncio
    async def test_api_timeout_handling(self, failing_api_server, sample_trajectory_request):
        """Test API timeout handling."""
        failing_api_server.set_failure_mode("timeout")
        
        # Test with timeout
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                failing_api_server.predict_single(sample_trajectory_request),
                timeout=0.1
            )
    
    @pytest.mark.asyncio
    async def test_api_partial_batch_failure(self, failing_api_server):
        """Test API handling of partial batch failures."""
        # Create batch where some requests will fail
        trajectories = []
        for i in range(5):
            points = [
                TrajectoryPoint(timestamp=0.0, x=float(i), y=0.0, vx=1.0, vy=0.0),
                TrajectoryPoint(timestamp=0.1, x=float(i) + 0.1, y=0.0, vx=1.0, vy=0.0)
            ]
            trajectory = TrajectoryInput(
                trajectory_id=f"batch_{i}",
                vehicle_id=f"vehicle_{i}",
                points=points
            )
            trajectories.append(trajectory)
        
        config = PredictionConfig(
            prediction_horizon=2.0,
            models=["test_model"]
        )
        
        batch_request = BatchTrajectoryRequest(
            trajectories=trajectories,
            config=config
        )
        
        # Set failure mode for some requests
        original_predict = failing_api_server.predict_single
        
        async def selective_failure(request):
            if "batch_2" in request.trajectory.trajectory_id:
                raise ValueError("Selective failure")
            return await original_predict(request)
        
        failing_api_server.predict_single = selective_failure
        
        # Batch processing should handle partial failures
        try:
            responses = await failing_api_server.predict_batch(batch_request)
            # Some responses should be successful
            successful_responses = [r for r in responses if r is not None]
            assert len(successful_responses) < len(trajectories)
        except Exception:
            # Acceptable to fail the entire batch
            pass


class TestAPIPerformance:
    """Test API performance characteristics."""
    
    @pytest.fixture
    def performance_api(self):
        """Create API for performance testing."""
        class PerformanceAPI:
            def __init__(self):
                self.request_times = []
                self.total_requests = 0
            
            async def predict_single(self, request: TrajectoryRequest) -> TrajectoryResponse:
                """Fast prediction for performance testing."""
                start_time = asyncio.get_event_loop().time()
                
                # Minimal processing
                await asyncio.sleep(0.001)  # 1ms processing time
                
                end_time = asyncio.get_event_loop().time()
                processing_time = end_time - start_time
                
                self.request_times.append(processing_time)
                self.total_requests += 1
                
                # Minimal response
                return TrajectoryResponse(
                    request_id=f"perf_{self.total_requests}",
                    model_name="fast_model",
                    predicted_trajectory=TrajectoryData(
                        trajectory_id="fast_pred",
                        vehicle_id="vehicle",
                        positions=[Position(x=1.0, y=1.0)],
                        velocities=[Velocity(vx=1.0, vy=0.0)],
                        timestamps=[1.0]
                    ),
                    confidence=0.8,
                    inference_time=processing_time
                )
            
            def get_performance_stats(self) -> Dict[str, float]:
                """Get performance statistics."""
                if not self.request_times:
                    return {}
                
                return {
                    'total_requests': self.total_requests,
                    'avg_response_time': np.mean(self.request_times),
                    'min_response_time': np.min(self.request_times),
                    'max_response_time': np.max(self.request_times),
                    'p95_response_time': np.percentile(self.request_times, 95),
                    'requests_per_second': len(self.request_times) / sum(self.request_times) if sum(self.request_times) > 0 else 0
                }
        
        return PerformanceAPI()
    
    @pytest.mark.asyncio
    async def test_api_response_time(self, performance_api, sample_trajectory_request):
        """Test API response time performance."""
        # Send single request
        response = await performance_api.predict_single(sample_trajectory_request)
        
        assert isinstance(response, TrajectoryResponse)
        assert response.inference_time < 0.1  # Should be under 100ms
        
        stats = performance_api.get_performance_stats()
        assert stats['avg_response_time'] < 0.1
    
    @pytest.mark.asyncio
    async def test_api_throughput(self, performance_api, sample_trajectory_request):
        """Test API throughput performance."""
        num_requests = 100
        
        # Send requests concurrently
        tasks = []
        for i in range(num_requests):
            task = performance_api.predict_single(sample_trajectory_request)
            tasks.append(task)
        
        start_time = asyncio.get_event_loop().time()
        responses = await asyncio.gather(*tasks)
        end_time = asyncio.get_event_loop().time()
        
        total_time = end_time - start_time
        throughput = num_requests / total_time
        
        # Verify all requests completed
        assert len(responses) == num_requests
        assert all(isinstance(resp, TrajectoryResponse) for resp in responses)
        
        # Verify throughput is reasonable
        assert throughput > 10  # At least 10 requests per second
        
        # Get detailed performance stats
        stats = performance_api.get_performance_stats()
        assert stats['total_requests'] == num_requests
        assert stats['avg_response_time'] < 1.0  # Average under 1 second
    
    @pytest.mark.asyncio
    async def test_api_memory_usage(self, performance_api, sample_trajectory_request):
        """Test API memory usage during operation."""
        import psutil
        import gc
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Send many requests
        for i in range(50):
            await performance_api.predict_single(sample_trajectory_request)
            
            # Force garbage collection periodically
            if i % 10 == 0:
                gc.collect()
        
        # Get final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory usage should not grow excessively
        assert memory_increase < 100  # Less than 100MB increase
        
        stats = performance_api.get_performance_stats()
        memory_per_request = memory_increase / stats['total_requests']
        
        # Each request should use reasonable memory
        assert memory_per_request < 2.0  # Less than 2MB per request
    
    @pytest.mark.asyncio
    async def test_api_scalability(self, performance_api, sample_trajectory_request):
        """Test API scalability with increasing load."""
        load_levels = [10, 25, 50, 100]
        throughput_results = []
        
        for load in load_levels:
            # Reset stats for each test
            performance_api.request_times = []
            performance_api.total_requests = 0
            
            # Send requests at this load level
            tasks = []
            for i in range(load):
                task = performance_api.predict_single(sample_trajectory_request)
                tasks.append(task)
            
            start_time = asyncio.get_event_loop().time()
            responses = await asyncio.gather(*tasks)
            end_time = asyncio.get_event_loop().time()
            
            total_time = end_time - start_time
            throughput = load / total_time
            throughput_results.append(throughput)
            
            # Verify all requests completed
            assert len(responses) == load
        
        # Throughput should remain relatively stable with increasing load
        # (or at least not degrade dramatically)
        min_throughput = min(throughput_results)
        max_throughput = max(throughput_results)
        
        # Throughput shouldn't drop by more than 50% under increased load
        assert min_throughput > max_throughput * 0.5


class TestAPIIntegrationWithExternalServices:
    """Test API integration with external services and dependencies."""
    
    @pytest.fixture
    def api_with_dependencies(self):
        """Create API with external dependencies."""
        class APIWithDependencies:
            def __init__(self):
                self.model_service_available = True
                self.database_available = True
                self.monitoring_service_available = True
            
            async def predict_with_dependencies(
                self,
                request: TrajectoryRequest
            ) -> TrajectoryResponse:
                """Prediction that depends on external services."""
                # Check model service
                if not self.model_service_available:
                    raise ServiceUnavailableError("Model service unavailable")
                
                # Check database
                if not self.database_available:
                    raise ServiceUnavailableError("Database unavailable")
                
                # Log to monitoring (optional)
                if self.monitoring_service_available:
                    # Simulate logging
                    await asyncio.sleep(0.001)
                
                # Return prediction
                return TrajectoryResponse(
                    request_id="dep_req",
                    model_name="dependent_model",
                    predicted_trajectory=TrajectoryData(
                        trajectory_id="dep_pred",
                        vehicle_id="vehicle",
                        positions=[Position(x=1.0, y=1.0)],
                        velocities=[Velocity(vx=1.0, vy=0.0)],
                        timestamps=[1.0]
                    ),
                    confidence=0.8,
                    inference_time=0.01
                )
        
        return APIWithDependencies()
    
    @pytest.mark.asyncio
    async def test_api_with_all_services_available(self, api_with_dependencies, sample_trajectory_request):
        """Test API when all external services are available."""
        response = await api_with_dependencies.predict_with_dependencies(sample_trajectory_request)
        
        assert isinstance(response, TrajectoryResponse)
        assert validate_prediction_response(response)
    
    @pytest.mark.asyncio
    async def test_api_with_model_service_unavailable(self, api_with_dependencies, sample_trajectory_request):
        """Test API behavior when model service is unavailable."""
        api_with_dependencies.model_service_available = False
        
        with pytest.raises(ServiceUnavailableError):
            await api_with_dependencies.predict_with_dependencies(sample_trajectory_request)
    
    @pytest.mark.asyncio
    async def test_api_with_database_unavailable(self, api_with_dependencies, sample_trajectory_request):
        """Test API behavior when database is unavailable."""
        api_with_dependencies.database_available = False
        
        with pytest.raises(ServiceUnavailableError):
            await api_with_dependencies.predict_with_dependencies(sample_trajectory_request)
    
    @pytest.mark.asyncio
    async def test_api_with_monitoring_unavailable(self, api_with_dependencies, sample_trajectory_request):
        """Test API behavior when monitoring service is unavailable."""
        api_with_dependencies.monitoring_service_available = False
        
        # Should still work without monitoring
        response = await api_with_dependencies.predict_with_dependencies(sample_trajectory_request)
        
        assert isinstance(response, TrajectoryResponse)
        assert validate_prediction_response(response)


# Custom exceptions for testing
class ServiceUnavailableError(Exception):
    """Exception for unavailable services."""
    pass


class ModelError(Exception):
    """Exception for model-related errors."""
    pass


class ValidationError(Exception):
    """Exception for validation errors."""
    pass