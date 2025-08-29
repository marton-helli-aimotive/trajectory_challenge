"""
Comprehensive API testing framework.

This module provides:
- Unit tests for API endpoints
- Integration tests for full workflows
- Load testing capabilities
- Performance benchmarking
- API contract validation
"""

import asyncio
import time
import json
import logging
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import statistics

import httpx
import pytest
from fastapi.testclient import TestClient

from .models import (
    TrajectoryRequest, TrajectoryResponse, BatchTrajectoryRequest,
    BatchTrajectoryResponse, TrajectoryInput, TrajectoryPoint,
    PredictionConfig, PredictionStatus, HealthStatus, LoadTestConfig,
    LoadTestResult
)

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Individual test result."""
    test_name: str
    status: str  # "passed", "failed", "error"
    duration_ms: float
    error_message: Optional[str] = None
    response_data: Optional[Dict[str, Any]] = None


@dataclass
class LoadTestMetrics:
    """Load test metrics."""
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time_ms: float
    min_response_time_ms: float
    max_response_time_ms: float
    p50_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    throughput_rps: float
    error_rate: float
    start_time: datetime
    end_time: datetime


class APITester:
    """
    Comprehensive API testing framework.
    
    Provides unit tests, integration tests, and performance validation.
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = None
        self.test_results: List[TestResult] = []
        
    async def setup(self) -> None:
        """Setup test environment."""
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=30.0)
        logger.info(f"API tester setup complete for {self.base_url}")
    
    async def cleanup(self) -> None:
        """Cleanup test environment."""
        if self.client:
            await self.client.aclose()
        logger.info("API tester cleanup complete")
    
    def create_test_trajectory(
        self,
        vehicle_id: str = "test_vehicle",
        num_points: int = 10,
        duration: float = 1.0
    ) -> TrajectoryInput:
        """Create test trajectory data."""
        
        points = []
        for i in range(num_points):
            t = i * duration / (num_points - 1)
            x = i * 0.5  # Simple linear trajectory
            y = 0.0
            
            points.append(TrajectoryPoint(x=x, y=y, timestamp=t))
        
        return TrajectoryInput(
            vehicle_id=vehicle_id,
            trajectory_points=points,
            metadata={"test": True}
        )
    
    async def test_health_endpoint(self) -> TestResult:
        """Test health check endpoint."""
        
        start_time = time.time()
        
        try:
            response = await self.client.get("/health")
            duration_ms = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                health_data = response.json()
                
                # Validate health response structure
                required_fields = ["status", "version", "uptime_seconds"]
                missing_fields = [field for field in required_fields if field not in health_data]
                
                if missing_fields:
                    return TestResult(
                        test_name="health_endpoint",
                        status="failed",
                        duration_ms=duration_ms,
                        error_message=f"Missing fields: {missing_fields}",
                        response_data=health_data
                    )
                
                return TestResult(
                    test_name="health_endpoint",
                    status="passed",
                    duration_ms=duration_ms,
                    response_data=health_data
                )
            else:
                return TestResult(
                    test_name="health_endpoint",
                    status="failed",
                    duration_ms=duration_ms,
                    error_message=f"HTTP {response.status_code}",
                    response_data=response.json() if response.content else None
                )
                
        except Exception as e:
            return TestResult(
                test_name="health_endpoint",
                status="error",
                duration_ms=(time.time() - start_time) * 1000,
                error_message=str(e)
            )
    
    async def test_single_prediction(self) -> TestResult:
        """Test single trajectory prediction endpoint."""
        
        start_time = time.time()
        
        try:
            # Create test request
            trajectory = self.create_test_trajectory()
            config = PredictionConfig(prediction_horizon=5.0)
            request = TrajectoryRequest(
                request_id="test_single_001",
                trajectory=trajectory,
                config=config
            )
            
            response = await self.client.post("/predict", json=request.dict())
            duration_ms = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                prediction_data = response.json()
                
                # Validate prediction response structure
                required_fields = ["status", "metadata", "processed_at"]
                missing_fields = [field for field in required_fields if field not in prediction_data]
                
                if missing_fields:
                    return TestResult(
                        test_name="single_prediction",
                        status="failed",
                        duration_ms=duration_ms,
                        error_message=f"Missing fields: {missing_fields}",
                        response_data=prediction_data
                    )
                
                # Check if prediction was successful
                if prediction_data["status"] not in [PredictionStatus.SUCCESS, PredictionStatus.CACHED]:
                    return TestResult(
                        test_name="single_prediction",
                        status="failed",
                        duration_ms=duration_ms,
                        error_message=f"Prediction failed: {prediction_data.get('error_message', 'Unknown error')}",
                        response_data=prediction_data
                    )
                
                return TestResult(
                    test_name="single_prediction",
                    status="passed",
                    duration_ms=duration_ms,
                    response_data=prediction_data
                )
            else:
                return TestResult(
                    test_name="single_prediction",
                    status="failed",
                    duration_ms=duration_ms,
                    error_message=f"HTTP {response.status_code}",
                    response_data=response.json() if response.content else None
                )
                
        except Exception as e:
            return TestResult(
                test_name="single_prediction",
                status="error",
                duration_ms=(time.time() - start_time) * 1000,
                error_message=str(e)
            )
    
    async def test_batch_prediction(self) -> TestResult:
        """Test batch trajectory prediction endpoint."""
        
        start_time = time.time()
        
        try:
            # Create test batch request
            trajectories = []
            for i in range(3):
                trajectory = self.create_test_trajectory(vehicle_id=f"test_batch_{i}")
                config = PredictionConfig(prediction_horizon=5.0)
                request = TrajectoryRequest(
                    request_id=f"test_batch_{i}",
                    trajectory=trajectory,
                    config=config
                )
                trajectories.append(request)
            
            batch_request = BatchTrajectoryRequest(
                batch_id="test_batch_001",
                trajectories=trajectories,
                parallel_processing=True
            )
            
            response = await self.client.post("/predict/batch", json=batch_request.dict())
            duration_ms = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                batch_data = response.json()
                
                # Validate batch response structure
                required_fields = ["status", "results", "summary", "total_processing_time_ms"]
                missing_fields = [field for field in required_fields if field not in batch_data]
                
                if missing_fields:
                    return TestResult(
                        test_name="batch_prediction",
                        status="failed",
                        duration_ms=duration_ms,
                        error_message=f"Missing fields: {missing_fields}",
                        response_data=batch_data
                    )
                
                # Check if batch contained expected number of results
                if len(batch_data["results"]) != len(trajectories):
                    return TestResult(
                        test_name="batch_prediction",
                        status="failed",
                        duration_ms=duration_ms,
                        error_message=f"Expected {len(trajectories)} results, got {len(batch_data['results'])}",
                        response_data=batch_data
                    )
                
                return TestResult(
                    test_name="batch_prediction",
                    status="passed",
                    duration_ms=duration_ms,
                    response_data=batch_data
                )
            else:
                return TestResult(
                    test_name="batch_prediction",
                    status="failed",
                    duration_ms=duration_ms,
                    error_message=f"HTTP {response.status_code}",
                    response_data=response.json() if response.content else None
                )
                
        except Exception as e:
            return TestResult(
                test_name="batch_prediction",
                status="error",
                duration_ms=(time.time() - start_time) * 1000,
                error_message=str(e)
            )
    
    async def test_models_endpoint(self) -> TestResult:
        """Test models information endpoint."""
        
        start_time = time.time()
        
        try:
            response = await self.client.get("/models")
            duration_ms = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                models_data = response.json()
                
                # Validate models response structure
                if not isinstance(models_data, list):
                    return TestResult(
                        test_name="models_endpoint",
                        status="failed",
                        duration_ms=duration_ms,
                        error_message="Expected list of models",
                        response_data=models_data
                    )
                
                # Check model structure if any models are available
                if models_data:
                    required_fields = ["model_name", "model_type", "status"]
                    model = models_data[0]
                    missing_fields = [field for field in required_fields if field not in model]
                    
                    if missing_fields:
                        return TestResult(
                            test_name="models_endpoint",
                            status="failed",
                            duration_ms=duration_ms,
                            error_message=f"Missing model fields: {missing_fields}",
                            response_data=models_data
                        )
                
                return TestResult(
                    test_name="models_endpoint",
                    status="passed",
                    duration_ms=duration_ms,
                    response_data=models_data
                )
            else:
                return TestResult(
                    test_name="models_endpoint",
                    status="failed",
                    duration_ms=duration_ms,
                    error_message=f"HTTP {response.status_code}",
                    response_data=response.json() if response.content else None
                )
                
        except Exception as e:
            return TestResult(
                test_name="models_endpoint",
                status="error",
                duration_ms=(time.time() - start_time) * 1000,
                error_message=str(e)
            )
    
    async def test_cache_endpoints(self) -> TestResult:
        """Test cache statistics and management endpoints."""
        
        start_time = time.time()
        
        try:
            # Test cache stats endpoint
            stats_response = await self.client.get("/cache/stats")
            
            if stats_response.status_code != 200:
                return TestResult(
                    test_name="cache_endpoints",
                    status="failed",
                    duration_ms=(time.time() - start_time) * 1000,
                    error_message=f"Cache stats HTTP {stats_response.status_code}"
                )
            
            # Test cache clear endpoint (if cache is enabled)
            stats_data = stats_response.json()
            if stats_data:  # Cache is enabled
                clear_response = await self.client.delete("/cache")
                
                if clear_response.status_code != 200:
                    return TestResult(
                        test_name="cache_endpoints",
                        status="failed",
                        duration_ms=(time.time() - start_time) * 1000,
                        error_message=f"Cache clear HTTP {clear_response.status_code}"
                    )
            
            return TestResult(
                test_name="cache_endpoints",
                status="passed",
                duration_ms=(time.time() - start_time) * 1000,
                response_data={"cache_stats": stats_data}
            )
                
        except Exception as e:
            return TestResult(
                test_name="cache_endpoints",
                status="error",
                duration_ms=(time.time() - start_time) * 1000,
                error_message=str(e)
            )
    
    async def test_input_validation(self) -> TestResult:
        """Test input validation for various invalid requests."""
        
        start_time = time.time()
        
        try:
            test_cases = [
                {
                    "name": "empty_trajectory",
                    "data": {
                        "trajectory": {
                            "vehicle_id": "test",
                            "trajectory_points": [],
                            "metadata": {}
                        }
                    },
                    "expected_status": 422  # Validation error
                },
                {
                    "name": "invalid_prediction_horizon",
                    "data": {
                        "trajectory": {
                            "vehicle_id": "test", 
                            "trajectory_points": [
                                {"x": 0, "y": 0, "timestamp": 0},
                                {"x": 1, "y": 1, "timestamp": 1}
                            ],
                            "metadata": {}
                        },
                        "config": {
                            "prediction_horizon": -1  # Invalid negative horizon
                        }
                    },
                    "expected_status": 422
                },
                {
                    "name": "malformed_json",
                    "data": "invalid json",
                    "expected_status": 422
                }
            ]
            
            failed_cases = []
            
            for test_case in test_cases:
                try:
                    if isinstance(test_case["data"], str):
                        response = await self.client.post(
                            "/predict",
                            content=test_case["data"],
                            headers={"Content-Type": "application/json"}
                        )
                    else:
                        response = await self.client.post("/predict", json=test_case["data"])
                    
                    if response.status_code != test_case["expected_status"]:
                        failed_cases.append(
                            f"{test_case['name']}: expected {test_case['expected_status']}, "
                            f"got {response.status_code}"
                        )
                        
                except Exception as e:
                    failed_cases.append(f"{test_case['name']}: exception {str(e)}")
            
            duration_ms = (time.time() - start_time) * 1000
            
            if failed_cases:
                return TestResult(
                    test_name="input_validation",
                    status="failed",
                    duration_ms=duration_ms,
                    error_message="; ".join(failed_cases)
                )
            else:
                return TestResult(
                    test_name="input_validation",
                    status="passed",
                    duration_ms=duration_ms
                )
                
        except Exception as e:
            return TestResult(
                test_name="input_validation",
                status="error",
                duration_ms=(time.time() - start_time) * 1000,
                error_message=str(e)
            )
    
    async def run_all_tests(self) -> List[TestResult]:
        """Run all API tests."""
        
        test_methods = [
            self.test_health_endpoint,
            self.test_single_prediction,
            self.test_batch_prediction,
            self.test_models_endpoint,
            self.test_cache_endpoints,
            self.test_input_validation
        ]
        
        results = []
        
        for test_method in test_methods:
            try:
                result = await test_method()
                results.append(result)
                logger.info(f"Test {result.test_name}: {result.status} ({result.duration_ms:.1f}ms)")
            except Exception as e:
                logger.error(f"Test {test_method.__name__} crashed: {e}")
                results.append(TestResult(
                    test_name=test_method.__name__,
                    status="error",
                    duration_ms=0.0,
                    error_message=str(e)
                ))
        
        self.test_results.extend(results)
        return results
    
    def get_test_summary(self) -> Dict[str, Any]:
        """Get test summary statistics."""
        
        if not self.test_results:
            return {"message": "No tests run"}
        
        passed = len([r for r in self.test_results if r.status == "passed"])
        failed = len([r for r in self.test_results if r.status == "failed"])
        errors = len([r for r in self.test_results if r.status == "error"])
        
        total_duration = sum(r.duration_ms for r in self.test_results)
        avg_duration = total_duration / len(self.test_results)
        
        return {
            "total_tests": len(self.test_results),
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "success_rate": passed / len(self.test_results),
            "total_duration_ms": total_duration,
            "average_duration_ms": avg_duration,
            "failed_tests": [
                {"name": r.test_name, "error": r.error_message}
                for r in self.test_results if r.status != "passed"
            ]
        }


class LoadTester:
    """
    Load testing framework for API performance validation.
    
    Supports concurrent users, various load patterns, and detailed metrics.
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.test_data: List[TrajectoryRequest] = []
        
    def generate_test_data(self, size: int = 100) -> None:
        """Generate test data for load testing."""
        
        self.test_data = []
        
        for i in range(size):
            # Create varied test trajectories
            num_points = 5 + (i % 10)  # 5-14 points
            duration = 1.0 + (i % 5) * 0.5  # 1.0-3.0 seconds
            
            points = []
            for j in range(num_points):
                t = j * duration / (num_points - 1)
                # Create varied trajectories (straight, curved, etc.)
                if i % 3 == 0:
                    # Straight line
                    x, y = j * 0.5, 0.0
                elif i % 3 == 1:
                    # Curved trajectory
                    x, y = j * 0.5, 0.1 * j * j
                else:
                    # Zigzag trajectory
                    x, y = j * 0.5, 0.2 * (j % 2)
                
                points.append(TrajectoryPoint(x=x, y=y, timestamp=t))
            
            trajectory = TrajectoryInput(
                vehicle_id=f"load_test_{i}",
                trajectory_points=points,
                metadata={"load_test": True, "test_id": i}
            )
            
            config = PredictionConfig(
                prediction_horizon=5.0 + (i % 3) * 2.5,  # 5.0-10.0 seconds
                time_resolution=0.1,
                uncertainty_quantification="gaussian"
            )
            
            request = TrajectoryRequest(
                request_id=f"load_test_req_{i}",
                trajectory=trajectory,
                config=config
            )
            
            self.test_data.append(request)
        
        logger.info(f"Generated {len(self.test_data)} test requests for load testing")
    
    async def run_load_test(self, config: LoadTestConfig) -> LoadTestResult:
        """Run load test with specified configuration."""
        
        if not self.test_data:
            self.generate_test_data(config.test_data_size)
        
        logger.info(f"Starting load test: {config.concurrent_users} users, "
                   f"{config.requests_per_user} requests each")
        
        start_time = datetime.now()
        response_times = []
        errors = []
        successful_requests = 0
        total_requests = config.concurrent_users * config.requests_per_user
        
        # Create semaphore for controlled ramp-up
        ramp_up_delay = config.ramp_up_time_seconds / config.concurrent_users
        
        async def user_session(user_id: int) -> List[float]:
            """Simulate user session with multiple requests."""
            
            session_times = []
            session_errors = []
            
            # Ramp-up delay
            await asyncio.sleep(user_id * ramp_up_delay)
            
            async with httpx.AsyncClient(base_url=self.base_url, timeout=30.0) as client:
                for request_num in range(config.requests_per_user):
                    # Select test data (cycle through available data)
                    test_request = self.test_data[
                        (user_id * config.requests_per_user + request_num) % len(self.test_data)
                    ]
                    
                    request_start = time.time()
                    
                    try:
                        response = await client.post("/predict", json=test_request.dict())
                        response_time = (time.time() - request_start) * 1000
                        session_times.append(response_time)
                        
                        if response.status_code != 200:
                            session_errors.append({
                                "user_id": user_id,
                                "request_num": request_num,
                                "status_code": response.status_code,
                                "error": response.text
                            })
                            
                    except Exception as e:
                        response_time = (time.time() - request_start) * 1000
                        session_times.append(response_time)
                        session_errors.append({
                            "user_id": user_id,
                            "request_num": request_num,
                            "exception": str(e)
                        })
            
            return session_times, session_errors
        
        # Run concurrent user sessions
        tasks = [user_session(user_id) for user_id in range(config.concurrent_users)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for result in results:
            if isinstance(result, tuple):
                session_times, session_errors = result
                response_times.extend(session_times)
                errors.extend(session_errors)
                successful_requests += len([t for t in session_times if t > 0])  # Assume positive time means success
            elif isinstance(result, Exception):
                errors.append({"exception": str(result)})
        
        end_time = datetime.now()
        
        # Calculate metrics
        total_duration = (end_time - start_time).total_seconds()
        failed_requests = len(errors)
        successful_requests = total_requests - failed_requests
        
        # Response time percentiles
        if response_times:
            percentiles = {
                "p50": statistics.median(response_times),
                "p95": statistics.quantiles(response_times, n=20)[18] if len(response_times) > 20 else max(response_times),
                "p99": statistics.quantiles(response_times, n=100)[98] if len(response_times) > 100 else max(response_times)
            }
        else:
            percentiles = {"p50": 0.0, "p95": 0.0, "p99": 0.0}
        
        return LoadTestResult(
            test_id=f"load_test_{int(time.time())}",
            config=config,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            success_rate=successful_requests / total_requests if total_requests > 0 else 0.0,
            average_response_time_ms=statistics.mean(response_times) if response_times else 0.0,
            percentile_response_times=percentiles,
            max_response_time_ms=max(response_times) if response_times else 0.0,
            min_response_time_ms=min(response_times) if response_times else 0.0,
            requests_per_second=total_requests / total_duration if total_duration > 0 else 0.0,
            test_duration_seconds=total_duration,
            errors=errors,
            started_at=start_time,
            completed_at=end_time
        )
    
    async def run_stress_test(
        self,
        max_users: int = 100,
        step_size: int = 10,
        step_duration: int = 30
    ) -> List[LoadTestResult]:
        """Run stress test with increasing load."""
        
        stress_results = []
        
        for users in range(step_size, max_users + 1, step_size):
            logger.info(f"Stress test step: {users} concurrent users")
            
            config = LoadTestConfig(
                concurrent_users=users,
                requests_per_user=10,
                ramp_up_time_seconds=min(step_duration / 2, 30),
                test_data_size=min(users * 10, 1000)
            )
            
            result = await self.run_load_test(config)
            stress_results.append(result)
            
            # Check if system is struggling (high error rate or very slow responses)
            if result.success_rate < 0.8 or result.average_response_time_ms > 5000:
                logger.warning(f"System stressed at {users} users, stopping stress test")
                break
            
            # Brief pause between stress steps
            await asyncio.sleep(5)
        
        return stress_results
    
    def analyze_load_test_results(
        self,
        results: Union[LoadTestResult, List[LoadTestResult]]
    ) -> Dict[str, Any]:
        """Analyze load test results and provide insights."""
        
        if isinstance(results, LoadTestResult):
            results = [results]
        
        analysis = {
            "total_tests": len(results),
            "test_summary": [],
            "performance_trends": {},
            "recommendations": []
        }
        
        for result in results:
            test_summary = {
                "concurrent_users": result.config.concurrent_users,
                "total_requests": result.total_requests,
                "success_rate": result.success_rate,
                "average_response_time_ms": result.average_response_time_ms,
                "throughput_rps": result.requests_per_second,
                "p95_response_time_ms": result.percentile_response_times.get("p95", 0.0)
            }
            analysis["test_summary"].append(test_summary)
        
        # Performance trends
        if len(results) > 1:
            users = [r.config.concurrent_users for r in results]
            response_times = [r.average_response_time_ms for r in results]
            success_rates = [r.success_rate for r in results]
            
            analysis["performance_trends"] = {
                "response_time_slope": self._calculate_slope(users, response_times),
                "success_rate_trend": self._calculate_slope(users, success_rates),
                "throughput_peak": max(r.requests_per_second for r in results),
                "breaking_point": self._find_breaking_point(results)
            }
        
        # Generate recommendations
        latest_result = results[-1]
        
        if latest_result.success_rate < 0.95:
            analysis["recommendations"].append(
                f"High error rate ({(1-latest_result.success_rate)*100:.1f}%). "
                "Consider scaling up or optimizing error handling."
            )
        
        if latest_result.average_response_time_ms > 1000:
            analysis["recommendations"].append(
                f"High response times ({latest_result.average_response_time_ms:.0f}ms). "
                "Consider performance optimization or caching."
            )
        
        target_p95 = latest_result.config.target_percentile
        p95_time = latest_result.percentile_response_times.get("p95", 0.0)
        if p95_time > 2000:  # 2 second P95 threshold
            analysis["recommendations"].append(
                f"P95 response time ({p95_time:.0f}ms) exceeds recommended threshold. "
                "Review system capacity and optimization opportunities."
            )
        
        if not analysis["recommendations"]:
            analysis["recommendations"].append(
                "✅ Performance looks good! System handling load within acceptable parameters."
            )
        
        return analysis
    
    def _calculate_slope(self, x_values: List[float], y_values: List[float]) -> float:
        """Calculate slope of linear trend."""
        if len(x_values) < 2:
            return 0.0
        
        n = len(x_values)
        x_mean = sum(x_values) / n
        y_mean = sum(y_values) / n
        
        numerator = sum((x_values[i] - x_mean) * (y_values[i] - y_mean) for i in range(n))
        denominator = sum((x_values[i] - x_mean) ** 2 for i in range(n))
        
        return numerator / denominator if denominator != 0 else 0.0
    
    def _find_breaking_point(self, results: List[LoadTestResult]) -> Optional[int]:
        """Find the breaking point where performance significantly degrades."""
        
        for i, result in enumerate(results):
            if result.success_rate < 0.9 or result.average_response_time_ms > 3000:
                return result.config.concurrent_users
        
        return None