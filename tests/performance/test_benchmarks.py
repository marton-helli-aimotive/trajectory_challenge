"""
Performance benchmarks and optimization tests.

This module tests:
- Model inference speed benchmarks
- API endpoint load testing
- Memory usage profiling
- System bottleneck identification
"""

import pytest
import numpy as np
import asyncio
import time
import psutil
import gc
from typing import List, Dict, Any, Tuple
from unittest.mock import Mock, AsyncMock
from dataclasses import dataclass
from pathlib import Path
import concurrent.futures
import threading

from src.trajectory_prediction.data.schemas import TrajectoryData, Position, Velocity
from src.trajectory_prediction.models.base import TrajectoryPredictor
from src.trajectory_prediction.api.models import TrajectoryRequest, TrajectoryResponse
from tests.conftest import validate_trajectory_data, performance_thresholds


@dataclass
class BenchmarkResult:
    """Results from performance benchmark."""
    
    test_name: str
    total_time: float
    throughput: float  # operations per second
    memory_usage_mb: float
    cpu_usage_percent: float
    p50_latency: float
    p95_latency: float
    p99_latency: float
    success_rate: float
    error_count: int
    metadata: Dict[str, Any]


class PerformanceBenchmark:
    """Base class for performance benchmarks."""
    
    def __init__(self):
        self.results = []
        self.start_time = None
        self.end_time = None
        self.operation_times = []
        self.memory_samples = []
        self.error_count = 0
    
    def start_benchmark(self):
        """Start benchmark timing and monitoring."""
        self.start_time = time.time()
        self.operation_times = []
        self.memory_samples = []
        self.error_count = 0
        gc.collect()  # Clean up before benchmark
    
    def record_operation(self, operation_time: float, success: bool = True):
        """Record timing for a single operation."""
        self.operation_times.append(operation_time)
        if not success:
            self.error_count += 1
    
    def sample_memory(self):
        """Sample current memory usage."""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        self.memory_samples.append(memory_mb)
    
    def finish_benchmark(self, test_name: str, metadata: Dict[str, Any] = None) -> BenchmarkResult:
        """Finish benchmark and calculate results."""
        self.end_time = time.time()
        
        total_time = self.end_time - self.start_time
        num_operations = len(self.operation_times)
        throughput = num_operations / total_time if total_time > 0 else 0
        
        # Calculate latency percentiles
        if self.operation_times:
            p50_latency = np.percentile(self.operation_times, 50)
            p95_latency = np.percentile(self.operation_times, 95)
            p99_latency = np.percentile(self.operation_times, 99)
        else:
            p50_latency = p95_latency = p99_latency = 0
        
        # Calculate memory usage
        avg_memory = np.mean(self.memory_samples) if self.memory_samples else 0
        
        # Calculate success rate
        success_rate = (num_operations - self.error_count) / num_operations if num_operations > 0 else 0
        
        # Get CPU usage (approximate)
        process = psutil.Process()
        cpu_percent = process.cpu_percent()
        
        result = BenchmarkResult(
            test_name=test_name,
            total_time=total_time,
            throughput=throughput,
            memory_usage_mb=avg_memory,
            cpu_usage_percent=cpu_percent,
            p50_latency=p50_latency,
            p95_latency=p95_latency,
            p99_latency=p99_latency,
            success_rate=success_rate,
            error_count=self.error_count,
            metadata=metadata or {}
        )
        
        self.results.append(result)
        return result


class TestModelInferenceBenchmarks:
    """Benchmark model inference performance."""
    
    @pytest.fixture
    def benchmark(self):
        """Create performance benchmark instance."""
        return PerformanceBenchmark()
    
    @pytest.fixture
    def fast_mock_model(self):
        """Create a fast mock model for benchmarking."""
        class FastMockModel(TrajectoryPredictor):
            def __init__(self):
                super().__init__()
                self.model_name = "fast_mock"
                self.is_trained = True
                self.prediction_cache = {}
            
            async def predict(self, trajectory: TrajectoryData, **kwargs) -> TrajectoryData:
                """Fast prediction with minimal processing."""
                # Simple cache key
                cache_key = f"{len(trajectory.positions)}_{trajectory.positions[0].x}_{trajectory.positions[-1].x}"
                
                if cache_key in self.prediction_cache:
                    return self.prediction_cache[cache_key]
                
                # Minimal prediction computation
                last_pos = trajectory.positions[-1]
                last_vel = trajectory.velocities[-1] if trajectory.velocities else Velocity(vx=1.0, vy=0.0)
                
                # Predict 5 steps ahead
                pred_positions = []
                pred_velocities = []
                pred_timestamps = []
                
                for i in range(5):
                    dt = 0.1
                    new_time = trajectory.timestamps[-1] + (i + 1) * dt
                    new_x = last_pos.x + last_vel.vx * (i + 1) * dt
                    new_y = last_pos.y + last_vel.vy * (i + 1) * dt
                    
                    pred_positions.append(Position(x=new_x, y=new_y))
                    pred_velocities.append(last_vel)
                    pred_timestamps.append(new_time)
                
                prediction = TrajectoryData(
                    trajectory_id=f"{trajectory.trajectory_id}_pred",
                    vehicle_id=trajectory.vehicle_id,
                    positions=pred_positions,
                    velocities=pred_velocities,
                    timestamps=pred_timestamps
                )
                
                # Cache result
                self.prediction_cache[cache_key] = prediction
                return prediction
        
        return FastMockModel()
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_single_prediction_benchmark(
        self,
        benchmark: PerformanceBenchmark,
        fast_mock_model,
        sample_trajectory,
        performance_thresholds
    ):
        """Benchmark single model prediction performance."""
        benchmark.start_benchmark()
        
        # Run single predictions
        num_predictions = 1000
        
        for i in range(num_predictions):
            benchmark.sample_memory()
            
            start_time = time.time()
            try:
                prediction = await fast_mock_model.predict(sample_trajectory)
                success = validate_trajectory_data(prediction)
                end_time = time.time()
                
                benchmark.record_operation(end_time - start_time, success)
                
            except Exception as e:
                end_time = time.time()
                benchmark.record_operation(end_time - start_time, False)
        
        result = benchmark.finish_benchmark(
            "single_prediction_benchmark",
            {"model": fast_mock_model.model_name, "num_predictions": num_predictions}
        )
        
        # Verify performance meets thresholds
        inference_thresholds = performance_thresholds["performance"]["inference_time"]
        
        assert result.success_rate > 0.95  # At least 95% success rate
        assert result.p95_latency < inference_thresholds["acceptable"]  # P95 under acceptable threshold
        assert result.throughput > 100  # At least 100 predictions/sec
        
        print(f"\nSingle Prediction Benchmark Results:")
        print(f"Throughput: {result.throughput:.1f} predictions/sec")
        print(f"P50 Latency: {result.p50_latency*1000:.1f}ms")
        print(f"P95 Latency: {result.p95_latency*1000:.1f}ms")
        print(f"Success Rate: {result.success_rate:.1%}")
        print(f"Memory Usage: {result.memory_usage_mb:.1f}MB")
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_batch_prediction_benchmark(
        self,
        benchmark: PerformanceBenchmark,
        fast_mock_model,
        sample_trajectories,
        performance_thresholds
    ):
        """Benchmark batch prediction performance."""
        benchmark.start_benchmark()
        
        batch_sizes = [1, 5, 10, 25, 50, 100]
        
        for batch_size in batch_sizes:
            batch_trajectories = sample_trajectories[:batch_size]
            
            benchmark.sample_memory()
            
            start_time = time.time()
            try:
                # Predict batch concurrently
                tasks = [fast_mock_model.predict(traj) for traj in batch_trajectories]
                predictions = await asyncio.gather(*tasks)
                
                success = all(validate_trajectory_data(pred) for pred in predictions)
                end_time = time.time()
                
                batch_time = end_time - start_time
                benchmark.record_operation(batch_time, success)
                
            except Exception as e:
                end_time = time.time()
                benchmark.record_operation(end_time - start_time, False)
        
        result = benchmark.finish_benchmark(
            "batch_prediction_benchmark",
            {"model": fast_mock_model.model_name, "batch_sizes": batch_sizes}
        )
        
        # Verify batch performance
        assert result.success_rate > 0.95
        assert result.throughput > 10  # At least 10 batches/sec
        
        print(f"\nBatch Prediction Benchmark Results:")
        print(f"Throughput: {result.throughput:.1f} batches/sec")
        print(f"P95 Latency: {result.p95_latency*1000:.1f}ms")
        print(f"Success Rate: {result.success_rate:.1%}")
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_prediction_benchmark(
        self,
        benchmark: PerformanceBenchmark,
        fast_mock_model,
        sample_trajectory
    ):
        """Benchmark concurrent prediction performance."""
        benchmark.start_benchmark()
        
        concurrency_levels = [1, 5, 10, 20, 50]
        
        for concurrency in concurrency_levels:
            benchmark.sample_memory()
            
            # Create concurrent prediction tasks
            tasks = [fast_mock_model.predict(sample_trajectory) for _ in range(concurrency)]
            
            start_time = time.time()
            try:
                predictions = await asyncio.gather(*tasks)
                success = all(validate_trajectory_data(pred) for pred in predictions)
                end_time = time.time()
                
                benchmark.record_operation(end_time - start_time, success)
                
            except Exception as e:
                end_time = time.time()
                benchmark.record_operation(end_time - start_time, False)
        
        result = benchmark.finish_benchmark(
            "concurrent_prediction_benchmark",
            {"model": fast_mock_model.model_name, "concurrency_levels": concurrency_levels}
        )
        
        print(f"\nConcurrent Prediction Benchmark Results:")
        print(f"Throughput: {result.throughput:.1f} concurrent batches/sec")
        print(f"P95 Latency: {result.p95_latency*1000:.1f}ms")
        print(f"Success Rate: {result.success_rate:.1%}")
    
    @pytest.mark.performance
    def test_model_memory_usage_benchmark(
        self,
        benchmark: PerformanceBenchmark,
        fast_mock_model,
        sample_trajectories
    ):
        """Benchmark model memory usage patterns."""
        benchmark.start_benchmark()
        
        # Test memory usage with increasing load
        for batch_size in [10, 50, 100, 200, 500]:
            benchmark.sample_memory()
            
            # Create many trajectories
            trajectories = sample_trajectories * (batch_size // len(sample_trajectories) + 1)
            trajectories = trajectories[:batch_size]
            
            start_time = time.time()
            try:
                # Load trajectories into memory and process
                for i, traj in enumerate(trajectories):
                    # Simulate processing
                    _ = len(traj.positions)
                    if i % 50 == 0:  # Sample memory periodically
                        benchmark.sample_memory()
                
                end_time = time.time()
                benchmark.record_operation(end_time - start_time, True)
                
            except Exception as e:
                end_time = time.time()
                benchmark.record_operation(end_time - start_time, False)
            
            # Force garbage collection
            gc.collect()
        
        result = benchmark.finish_benchmark(
            "memory_usage_benchmark",
            {"model": fast_mock_model.model_name}
        )
        
        # Verify memory usage is reasonable
        memory_threshold = 1000  # 1GB max
        assert result.memory_usage_mb < memory_threshold
        
        print(f"\nMemory Usage Benchmark Results:")
        print(f"Average Memory: {result.memory_usage_mb:.1f}MB")
        print(f"Peak Memory: {max(benchmark.memory_samples):.1f}MB" if benchmark.memory_samples else "N/A")


class TestAPILoadBenchmarks:
    """Benchmark API endpoint performance under load."""
    
    @pytest.fixture
    def load_test_api(self):
        """Create API for load testing."""
        class LoadTestAPI:
            def __init__(self):
                self.request_count = 0
                self.active_requests = 0
                self.max_concurrent = 0
                self.request_times = []
                self.errors = []
            
            async def predict_single(self, request: TrajectoryRequest) -> TrajectoryResponse:
                """Fast prediction for load testing."""
                self.active_requests += 1
                self.max_concurrent = max(self.max_concurrent, self.active_requests)
                
                start_time = time.time()
                
                try:
                    # Simulate processing time with some variance
                    processing_time = np.random.uniform(0.005, 0.020)  # 5-20ms
                    await asyncio.sleep(processing_time)
                    
                    # Create response
                    response = TrajectoryResponse(
                        request_id=f"load_{self.request_count}",
                        model_name="load_test_model",
                        predicted_trajectory=TrajectoryData(
                            trajectory_id="load_pred",
                            vehicle_id="vehicle",
                            positions=[Position(x=1.0, y=1.0)],
                            velocities=[Velocity(vx=1.0, vy=0.0)],
                            timestamps=[1.0]
                        ),
                        confidence=0.8,
                        inference_time=processing_time
                    )
                    
                    end_time = time.time()
                    self.request_times.append(end_time - start_time)
                    self.request_count += 1
                    
                    return response
                    
                except Exception as e:
                    self.errors.append(str(e))
                    raise
                
                finally:
                    self.active_requests -= 1
            
            def get_stats(self) -> Dict[str, Any]:
                """Get load testing statistics."""
                return {
                    'total_requests': self.request_count,
                    'active_requests': self.active_requests,
                    'max_concurrent': self.max_concurrent,
                    'avg_response_time': np.mean(self.request_times) if self.request_times else 0,
                    'p95_response_time': np.percentile(self.request_times, 95) if self.request_times else 0,
                    'error_count': len(self.errors),
                    'error_rate': len(self.errors) / max(self.request_count, 1)
                }
        
        return LoadTestAPI()
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_api_load_test_ramp_up(
        self,
        benchmark: PerformanceBenchmark,
        load_test_api,
        sample_trajectory_request
    ):
        """Test API performance with ramping load."""
        benchmark.start_benchmark()
        
        # Ramp up load levels
        load_levels = [10, 25, 50, 100, 200]
        
        for load in load_levels:
            benchmark.sample_memory()
            
            # Create concurrent requests
            tasks = []
            for i in range(load):
                task = load_test_api.predict_single(sample_trajectory_request)
                tasks.append(task)
            
            start_time = time.time()
            try:
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                success_count = sum(1 for resp in responses if not isinstance(resp, Exception))
                success_rate = success_count / len(responses)
                
                end_time = time.time()
                benchmark.record_operation(end_time - start_time, success_rate > 0.95)
                
            except Exception as e:
                end_time = time.time()
                benchmark.record_operation(end_time - start_time, False)
        
        result = benchmark.finish_benchmark(
            "api_load_ramp_test",
            {"load_levels": load_levels, "api_stats": load_test_api.get_stats()}
        )
        
        # Verify load test performance
        api_stats = load_test_api.get_stats()
        
        assert api_stats['error_rate'] < 0.05  # Less than 5% errors
        assert api_stats['avg_response_time'] < 1.0  # Average under 1 second
        assert result.success_rate > 0.90  # At least 90% success
        
        print(f"\nAPI Load Ramp Test Results:")
        print(f"Total Requests: {api_stats['total_requests']}")
        print(f"Error Rate: {api_stats['error_rate']:.1%}")
        print(f"Max Concurrent: {api_stats['max_concurrent']}")
        print(f"Avg Response Time: {api_stats['avg_response_time']*1000:.1f}ms")
        print(f"P95 Response Time: {api_stats['p95_response_time']*1000:.1f}ms")
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_api_sustained_load_test(
        self,
        benchmark: PerformanceBenchmark,
        load_test_api,
        sample_trajectory_request
    ):
        """Test API performance under sustained load."""
        benchmark.start_benchmark()
        
        # Sustained load parameters
        requests_per_second = 50
        duration_seconds = 10
        total_requests = requests_per_second * duration_seconds
        
        # Create request schedule
        async def send_request_batch():
            """Send a batch of requests every second."""
            for second in range(duration_seconds):
                benchmark.sample_memory()
                
                # Send requests for this second
                tasks = []
                for i in range(requests_per_second):
                    task = load_test_api.predict_single(sample_trajectory_request)
                    tasks.append(task)
                
                start_time = time.time()
                try:
                    responses = await asyncio.gather(*tasks, return_exceptions=True)
                    success_count = sum(1 for resp in responses if not isinstance(resp, Exception))
                    success_rate = success_count / len(responses)
                    
                    end_time = time.time()
                    benchmark.record_operation(end_time - start_time, success_rate > 0.95)
                    
                    # Wait for next second (if we finished early)
                    elapsed = end_time - start_time
                    if elapsed < 1.0:
                        await asyncio.sleep(1.0 - elapsed)
                        
                except Exception as e:
                    end_time = time.time()
                    benchmark.record_operation(end_time - start_time, False)
        
        await send_request_batch()
        
        result = benchmark.finish_benchmark(
            "api_sustained_load_test",
            {
                "requests_per_second": requests_per_second,
                "duration_seconds": duration_seconds,
                "total_requests": total_requests,
                "api_stats": load_test_api.get_stats()
            }
        )
        
        api_stats = load_test_api.get_stats()
        
        # Verify sustained performance
        assert api_stats['error_rate'] < 0.05
        assert api_stats['total_requests'] >= total_requests * 0.95  # At least 95% completed
        assert result.success_rate > 0.90
        
        print(f"\nAPI Sustained Load Test Results:")
        print(f"Target RPS: {requests_per_second}")
        print(f"Actual RPS: {api_stats['total_requests'] / duration_seconds:.1f}")
        print(f"Error Rate: {api_stats['error_rate']:.1%}")
        print(f"Avg Response Time: {api_stats['avg_response_time']*1000:.1f}ms")
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_api_stress_test(
        self,
        benchmark: PerformanceBenchmark,
        load_test_api,
        sample_trajectory_request
    ):
        """Test API behavior under extreme stress."""
        benchmark.start_benchmark()
        
        # Extreme load to find breaking point
        stress_levels = [100, 200, 500, 1000, 2000]
        breaking_point = None
        
        for stress_load in stress_levels:
            benchmark.sample_memory()
            
            # Create extreme concurrent load
            tasks = []
            for i in range(stress_load):
                task = load_test_api.predict_single(sample_trajectory_request)
                tasks.append(task)
            
            start_time = time.time()
            try:
                # Set timeout to prevent hanging
                responses = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=30.0
                )
                
                success_count = sum(1 for resp in responses if not isinstance(resp, Exception))
                success_rate = success_count / len(responses)
                
                end_time = time.time()
                benchmark.record_operation(end_time - start_time, success_rate > 0.50)
                
                # If success rate drops below 50%, we found breaking point
                if success_rate < 0.50:
                    breaking_point = stress_load
                    break
                    
            except asyncio.TimeoutError:
                end_time = time.time()
                benchmark.record_operation(end_time - start_time, False)
                breaking_point = stress_load
                break
            except Exception as e:
                end_time = time.time()
                benchmark.record_operation(end_time - start_time, False)
        
        result = benchmark.finish_benchmark(
            "api_stress_test",
            {
                "stress_levels": stress_levels,
                "breaking_point": breaking_point,
                "api_stats": load_test_api.get_stats()
            }
        )
        
        api_stats = load_test_api.get_stats()
        
        print(f"\nAPI Stress Test Results:")
        print(f"Breaking Point: {breaking_point} concurrent requests" if breaking_point else "No breaking point found")
        print(f"Max Concurrent: {api_stats['max_concurrent']}")
        print(f"Total Requests Processed: {api_stats['total_requests']}")
        print(f"Final Error Rate: {api_stats['error_rate']:.1%}")


class TestMemoryProfilingBenchmarks:
    """Profile memory usage patterns and identify leaks."""
    
    @pytest.fixture
    def memory_profiler(self):
        """Create memory profiling utilities."""
        class MemoryProfiler:
            def __init__(self):
                self.snapshots = []
                self.baseline = None
            
            def take_snapshot(self, label: str = ""):
                """Take memory snapshot."""
                process = psutil.Process()
                memory_info = process.memory_info()
                
                snapshot = {
                    'label': label,
                    'timestamp': time.time(),
                    'rss_mb': memory_info.rss / 1024 / 1024,
                    'vms_mb': memory_info.vms / 1024 / 1024,
                    'percent': process.memory_percent()
                }
                
                self.snapshots.append(snapshot)
                
                if self.baseline is None:
                    self.baseline = snapshot
                
                return snapshot
            
            def get_memory_growth(self) -> float:
                """Get memory growth since baseline."""
                if not self.snapshots or self.baseline is None:
                    return 0.0
                
                return self.snapshots[-1]['rss_mb'] - self.baseline['rss_mb']
            
            def detect_leaks(self, threshold_mb: float = 10.0) -> List[Dict]:
                """Detect potential memory leaks."""
                leaks = []
                
                if len(self.snapshots) < 2:
                    return leaks
                
                # Look for consistent growth patterns
                growth_rates = []
                for i in range(1, len(self.snapshots)):
                    prev = self.snapshots[i-1]
                    curr = self.snapshots[i]
                    
                    time_diff = curr['timestamp'] - prev['timestamp']
                    memory_diff = curr['rss_mb'] - prev['rss_mb']
                    
                    if time_diff > 0:
                        growth_rate = memory_diff / time_diff  # MB per second
                        growth_rates.append(growth_rate)
                
                # If average growth rate is positive and significant
                if growth_rates:
                    avg_growth_rate = np.mean(growth_rates)
                    if avg_growth_rate > threshold_mb / 60:  # threshold per minute
                        leaks.append({
                            'type': 'consistent_growth',
                            'growth_rate_mb_per_sec': avg_growth_rate,
                            'total_growth_mb': self.get_memory_growth(),
                            'snapshots': len(self.snapshots)
                        })
                
                return leaks
        
        return MemoryProfiler()
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_model_memory_profile(
        self,
        memory_profiler,
        fast_mock_model,
        sample_trajectory
    ):
        """Profile model memory usage during operation."""
        memory_profiler.take_snapshot("baseline")
        
        # Test memory usage with repeated predictions
        for iteration in range(100):
            # Predict
            await fast_mock_model.predict(sample_trajectory)
            
            # Take periodic snapshots
            if iteration % 20 == 0:
                memory_profiler.take_snapshot(f"iteration_{iteration}")
                gc.collect()  # Force GC to see if memory is freed
        
        memory_profiler.take_snapshot("final")
        
        # Analyze memory growth
        memory_growth = memory_profiler.get_memory_growth()
        leaks = memory_profiler.detect_leaks()
        
        # Verify no significant memory leaks
        assert memory_growth < 50.0  # Less than 50MB growth
        assert len(leaks) == 0  # No leaks detected
        
        print(f"\nModel Memory Profile:")
        print(f"Memory Growth: {memory_growth:.1f}MB")
        print(f"Leaks Detected: {len(leaks)}")
        
        for leak in leaks:
            print(f"  - {leak['type']}: {leak['growth_rate_mb_per_sec']:.3f} MB/sec")
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_api_memory_profile_under_load(
        self,
        memory_profiler,
        load_test_api,
        sample_trajectory_request
    ):
        """Profile API memory usage under load."""
        memory_profiler.take_snapshot("api_baseline")
        
        # Gradually increase load and monitor memory
        load_levels = [10, 50, 100, 200]
        
        for load in load_levels:
            # Send concurrent requests
            tasks = [load_test_api.predict_single(sample_trajectory_request) for _ in range(load)]
            await asyncio.gather(*tasks, return_exceptions=True)
            
            memory_profiler.take_snapshot(f"load_{load}")
            
            # Brief pause to allow memory cleanup
            await asyncio.sleep(0.5)
        
        # Final cleanup and snapshot
        gc.collect()
        await asyncio.sleep(1.0)
        memory_profiler.take_snapshot("api_final")
        
        # Analyze results
        memory_growth = memory_profiler.get_memory_growth()
        leaks = memory_profiler.detect_leaks()
        
        # Verify reasonable memory usage
        assert memory_growth < 100.0  # Less than 100MB growth
        
        print(f"\nAPI Memory Profile Under Load:")
        print(f"Memory Growth: {memory_growth:.1f}MB")
        print(f"Leaks Detected: {len(leaks)}")
        
        # Print snapshot details
        for snapshot in memory_profiler.snapshots:
            print(f"  {snapshot['label']}: {snapshot['rss_mb']:.1f}MB")


class TestBottleneckIdentification:
    """Identify and analyze system bottlenecks."""
    
    @pytest.mark.performance
    def test_cpu_bottleneck_detection(self, sample_trajectory):
        """Test for CPU bottlenecks during computation."""
        import threading
        import queue
        
        def cpu_intensive_task(trajectory, result_queue):
            """Simulate CPU-intensive trajectory processing."""
            start_time = time.time()
            
            # Simulate complex computation
            for _ in range(1000):
                # Complex trajectory analysis
                total_distance = 0
                for i in range(1, len(trajectory.positions)):
                    dx = trajectory.positions[i].x - trajectory.positions[i-1].x
                    dy = trajectory.positions[i].y - trajectory.positions[i-1].y
                    total_distance += np.sqrt(dx**2 + dy**2)
                
                # Simulate feature extraction
                features = np.random.randn(100)
                features = features / np.linalg.norm(features)
            
            end_time = time.time()
            result_queue.put(end_time - start_time)
        
        # Test with different numbers of threads
        thread_counts = [1, 2, 4, 8]
        results = {}
        
        for num_threads in thread_counts:
            threads = []
            result_queue = queue.Queue()
            
            start_time = time.time()
            
            # Start threads
            for i in range(num_threads):
                thread = threading.Thread(
                    target=cpu_intensive_task,
                    args=(sample_trajectory, result_queue)
                )
                threads.append(thread)
                thread.start()
            
            # Wait for completion
            for thread in threads:
                thread.join()
            
            total_time = time.time() - start_time
            
            # Collect individual thread times
            thread_times = []
            while not result_queue.empty():
                thread_times.append(result_queue.get())
            
            results[num_threads] = {
                'total_time': total_time,
                'avg_thread_time': np.mean(thread_times),
                'efficiency': thread_times[0] / total_time if thread_times else 0  # Compared to single thread
            }
        
        # Analyze scalability
        single_thread_time = results[1]['total_time']
        
        print(f"\nCPU Bottleneck Analysis:")
        for threads, result in results.items():
            speedup = single_thread_time / result['total_time']
            efficiency = speedup / threads
            
            print(f"  {threads} threads: {result['total_time']:.2f}s (speedup: {speedup:.2f}x, efficiency: {efficiency:.1%})")
        
        # Check for CPU bottlenecks (poor scaling)
        four_thread_efficiency = results[4]['efficiency']
        assert four_thread_efficiency > 0.25  # At least 25% efficiency with 4 threads
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_io_bottleneck_detection(self):
        """Test for I/O bottlenecks during async operations."""
        async def io_intensive_task(delay: float) -> float:
            """Simulate I/O-intensive operation."""
            start_time = time.time()
            await asyncio.sleep(delay)  # Simulate I/O wait
            return time.time() - start_time
        
        # Test with different concurrency levels
        concurrency_levels = [1, 5, 10, 20, 50]
        io_delay = 0.01  # 10ms I/O delay per operation
        results = {}
        
        for concurrency in concurrency_levels:
            start_time = time.time()
            
            # Run concurrent I/O operations
            tasks = [io_intensive_task(io_delay) for _ in range(concurrency)]
            task_times = await asyncio.gather(*tasks)
            
            total_time = time.time() - start_time
            
            results[concurrency] = {
                'total_time': total_time,
                'avg_task_time': np.mean(task_times),
                'theoretical_min': io_delay,  # Best case if fully concurrent
                'efficiency': io_delay / total_time  # How close to theoretical minimum
            }
        
        print(f"\nI/O Bottleneck Analysis:")
        for concurrency, result in results.items():
            print(f"  {concurrency} tasks: {result['total_time']:.3f}s (efficiency: {result['efficiency']:.1%})")
        
        # Check for good I/O concurrency
        high_concurrency_efficiency = results[50]['efficiency']
        assert high_concurrency_efficiency > 0.8  # Should be close to theoretical minimum
    
    @pytest.mark.performance
    def test_memory_bottleneck_detection(self, sample_trajectories):
        """Test for memory bottlenecks and allocation patterns."""
        import tracemalloc
        
        # Start memory tracing
        tracemalloc.start()
        
        # Simulate memory-intensive operations
        def memory_intensive_processing(trajectories, batch_size):
            """Simulate processing that uses significant memory."""
            large_matrices = []
            processed_data = []
            
            for i in range(0, len(trajectories), batch_size):
                batch = trajectories[i:i+batch_size]
                
                # Create large temporary matrices
                for traj in batch:
                    matrix = np.random.randn(1000, 1000)  # 8MB matrix
                    large_matrices.append(matrix)
                    
                    # Process trajectory
                    positions = np.array([[p.x, p.y] for p in traj.positions])
                    processed_data.append(positions)
                
                # Simulate cleanup (but keep some data)
                if len(large_matrices) > 50:  # Keep only recent matrices
                    large_matrices = large_matrices[-25:]
            
            return processed_data, large_matrices
        
        # Test with different batch sizes
        batch_sizes = [1, 5, 10, 20]
        memory_results = {}
        
        for batch_size in batch_sizes:
            # Take memory snapshot before
            snapshot_before = tracemalloc.take_snapshot()
            
            # Process data
            processed, matrices = memory_intensive_processing(sample_trajectories, batch_size)
            
            # Take memory snapshot after
            snapshot_after = tracemalloc.take_snapshot()
            
            # Calculate memory difference
            top_stats = snapshot_after.compare_to(snapshot_before, 'lineno')
            
            total_memory_diff = sum(stat.size_diff for stat in top_stats)
            peak_memory = max(stat.size for stat in top_stats) if top_stats else 0
            
            memory_results[batch_size] = {
                'memory_diff_mb': total_memory_diff / 1024 / 1024,
                'peak_memory_mb': peak_memory / 1024 / 1024,
                'num_objects': len(processed) + len(matrices)
            }
            
            # Cleanup
            del processed, matrices
            gc.collect()
        
        # Stop tracing
        tracemalloc.stop()
        
        print(f"\nMemory Bottleneck Analysis:")
        for batch_size, result in memory_results.items():
            print(f"  Batch size {batch_size}: {result['memory_diff_mb']:.1f}MB allocated, peak: {result['peak_memory_mb']:.1f}MB")
        
        # Verify memory usage is reasonable
        max_memory_diff = max(result['memory_diff_mb'] for result in memory_results.values())
        assert max_memory_diff < 500.0  # Less than 500MB allocated


@pytest.mark.performance
class TestPerformanceRegression:
    """Test for performance regressions between versions."""
    
    def test_performance_baseline_comparison(self, benchmark: PerformanceBenchmark):
        """Compare current performance against baseline."""
        # This would typically load baseline results from file
        baseline_results = {
            'single_prediction_throughput': 1000,  # predictions/sec
            'single_prediction_p95_latency': 0.01,  # 10ms
            'api_load_test_error_rate': 0.02,  # 2%
            'memory_usage_mb': 200,  # 200MB
        }
        
        # Simulate current results (would come from actual benchmarks)
        current_results = {
            'single_prediction_throughput': 950,   # Slight regression
            'single_prediction_p95_latency': 0.012,  # Slight regression
            'api_load_test_error_rate': 0.015,  # Improvement
            'memory_usage_mb': 210,  # Slight regression
        }
        
        # Compare results and identify regressions
        regressions = []
        improvements = []
        
        for metric, baseline_value in baseline_results.items():
            current_value = current_results.get(metric, 0)
            
            # Calculate percentage change
            if baseline_value != 0:
                pct_change = (current_value - baseline_value) / baseline_value
                
                # For throughput, higher is better
                if 'throughput' in metric:
                    if pct_change < -0.05:  # 5% regression threshold
                        regressions.append((metric, pct_change, baseline_value, current_value))
                    elif pct_change > 0.05:
                        improvements.append((metric, pct_change, baseline_value, current_value))
                
                # For latency, memory, error rate - lower is better
                else:
                    if pct_change > 0.05:  # 5% regression threshold
                        regressions.append((metric, pct_change, baseline_value, current_value))
                    elif pct_change < -0.05:
                        improvements.append((metric, pct_change, baseline_value, current_value))
        
        print(f"\nPerformance Regression Analysis:")
        
        if regressions:
            print(f"Regressions detected:")
            for metric, pct_change, baseline, current in regressions:
                print(f"  {metric}: {pct_change:+.1%} ({baseline} -> {current})")
        
        if improvements:
            print(f"Improvements detected:")
            for metric, pct_change, baseline, current in improvements:
                print(f"  {metric}: {pct_change:+.1%} ({baseline} -> {current})")
        
        if not regressions and not improvements:
            print("No significant performance changes detected.")
        
        # Fail test if critical regressions detected
        critical_regressions = [
            r for r in regressions 
            if abs(r[1]) > 0.1 and ('throughput' in r[0] or 'latency' in r[0])
        ]
        
        assert len(critical_regressions) == 0, f"Critical performance regressions detected: {critical_regressions}"