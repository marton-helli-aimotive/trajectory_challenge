"""
Performance optimization for trajectory prediction API.

This module provides:
- Request batching for improved throughput
- Model inference optimization
- Connection pooling and resource management
- Performance monitoring and auto-scaling
"""

import asyncio
import time
import logging
from typing import List, Dict, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime, timedelta
import numpy as np
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor

from ..models.base import TrajectoryPredictor
from ..data.schemas import TrajectoryData

logger = logging.getLogger(__name__)


@dataclass
class BatchedRequest:
    """Container for batched prediction request."""
    request_id: str
    trajectory_data: TrajectoryData
    prediction_horizon: float
    model_name: Optional[str]
    timestamp: float
    future: asyncio.Future
    priority: int = 0
    timeout: float = 30.0


@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    throughput_rps: float = 0.0
    batch_utilization: float = 0.0
    queue_size: int = 0
    active_workers: int = 0
    cpu_usage_percent: float = 0.0
    memory_usage_percent: float = 0.0
    timestamp: float = field(default_factory=time.time)


class RequestBatcher:
    """
    Request batching system for improved throughput.
    
    Collects individual requests into batches for efficient processing.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Batching configuration
        self.max_batch_size = config.get("max_batch_size", 32)
        self.batch_timeout_ms = config.get("batch_timeout_ms", 50)  # 50ms
        self.max_queue_size = config.get("max_queue_size", 1000)
        
        # Request queue
        self.request_queue: deque = deque()
        self.queue_lock = asyncio.Lock()
        
        # Batching state
        self.current_batch: List[BatchedRequest] = []
        self.batch_future: Optional[asyncio.Future] = None
        self.batch_timer: Optional[asyncio.Handle] = None
        
        # Performance tracking
        self.batches_processed = 0
        self.total_requests_batched = 0
        self.batch_sizes: List[int] = []
        
        # Background task
        self.batching_task: Optional[asyncio.Task] = None
        self.running = False
        
        logger.info(f"Request batcher initialized with max_batch_size={self.max_batch_size}")
    
    async def start(self) -> None:
        """Start the batching system."""
        if self.running:
            return
        
        self.running = True
        self.batching_task = asyncio.create_task(self._batching_loop())
        logger.info("Request batcher started")
    
    async def stop(self) -> None:
        """Stop the batching system."""
        if not self.running:
            return
        
        self.running = False
        
        if self.batching_task:
            self.batching_task.cancel()
            try:
                await self.batching_task
            except asyncio.CancelledError:
                pass
        
        # Process remaining requests
        await self._process_remaining_requests()
        
        logger.info("Request batcher stopped")
    
    async def add_request(
        self,
        request_id: str,
        trajectory_data: TrajectoryData,
        prediction_horizon: float,
        model_name: Optional[str] = None,
        priority: int = 0,
        timeout: float = 30.0
    ) -> Any:
        """
        Add request to batch queue.
        
        Returns a future that will be resolved when the request is processed.
        """
        
        if not self.running:
            await self.start()
        
        async with self.queue_lock:
            if len(self.request_queue) >= self.max_queue_size:
                raise RuntimeError("Request queue is full")
            
            # Create batched request
            future = asyncio.Future()
            batched_request = BatchedRequest(
                request_id=request_id,
                trajectory_data=trajectory_data,
                prediction_horizon=prediction_horizon,
                model_name=model_name,
                timestamp=time.time(),
                future=future,
                priority=priority,
                timeout=timeout
            )
            
            self.request_queue.append(batched_request)
        
        return await future
    
    async def _batching_loop(self) -> None:
        """Main batching loop."""
        
        while self.running:
            try:
                # Wait for requests or timeout
                await asyncio.sleep(self.batch_timeout_ms / 1000)
                
                # Collect batch
                batch = await self._collect_batch()
                
                if batch:
                    # Process batch
                    await self._process_batch(batch)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Batching loop error: {e}")
                await asyncio.sleep(0.1)
    
    async def _collect_batch(self) -> List[BatchedRequest]:
        """Collect requests into a batch."""
        
        batch = []
        current_time = time.time()
        
        async with self.queue_lock:
            # Collect up to max_batch_size requests
            while len(batch) < self.max_batch_size and self.request_queue:
                request = self.request_queue.popleft()
                
                # Check if request has timed out
                if current_time - request.timestamp > request.timeout:
                    request.future.set_exception(TimeoutError("Request timed out in queue"))
                    continue
                
                batch.append(request)
            
            # Sort by priority if needed
            if batch and any(req.priority != 0 for req in batch):
                batch.sort(key=lambda req: req.priority, reverse=True)
        
        return batch
    
    async def _process_batch(self, batch: List[BatchedRequest]) -> None:
        """Process a batch of requests."""
        
        if not batch:
            return
        
        start_time = time.time()
        
        try:
            # Group by model for efficient batching
            model_batches = {}
            for request in batch:
                model_name = request.model_name or "default"
                if model_name not in model_batches:
                    model_batches[model_name] = []
                model_batches[model_name].append(request)
            
            # Process each model batch
            tasks = []
            for model_name, model_requests in model_batches.items():
                task = asyncio.create_task(
                    self._process_model_batch(model_name, model_requests)
                )
                tasks.append(task)
            
            # Wait for all model batches to complete
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Update statistics
            self.batches_processed += 1
            self.total_requests_batched += len(batch)
            self.batch_sizes.append(len(batch))
            
            # Keep only recent batch sizes for statistics
            if len(self.batch_sizes) > 1000:
                self.batch_sizes = self.batch_sizes[-1000:]
            
            processing_time = (time.time() - start_time) * 1000
            logger.debug(f"Processed batch of {len(batch)} requests in {processing_time:.1f}ms")
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            
            # Set exception on all futures in the batch
            for request in batch:
                if not request.future.done():
                    request.future.set_exception(e)
    
    async def _process_model_batch(
        self,
        model_name: str,
        requests: List[BatchedRequest]
    ) -> None:
        """Process batch of requests for a specific model."""
        
        # This would integrate with the actual model serving system
        # For now, we'll simulate batch processing
        
        try:
            # Simulate batch prediction
            await asyncio.sleep(0.01)  # Simulate processing time
            
            # Set results on futures
            for request in requests:
                if not request.future.done():
                    # Create mock prediction result
                    result = f"prediction_for_{request.request_id}"
                    request.future.set_result(result)
                    
        except Exception as e:
            # Set exception on all futures in this model batch
            for request in requests:
                if not request.future.done():
                    request.future.set_exception(e)
    
    async def _process_remaining_requests(self) -> None:
        """Process any remaining requests during shutdown."""
        
        remaining_requests = []
        
        async with self.queue_lock:
            remaining_requests = list(self.request_queue)
            self.request_queue.clear()
        
        if remaining_requests:
            logger.info(f"Processing {len(remaining_requests)} remaining requests")
            await self._process_batch(remaining_requests)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get batching statistics."""
        
        avg_batch_size = np.mean(self.batch_sizes) if self.batch_sizes else 0.0
        batch_utilization = avg_batch_size / self.max_batch_size if self.max_batch_size > 0 else 0.0
        
        return {
            "batches_processed": self.batches_processed,
            "total_requests_batched": self.total_requests_batched,
            "average_batch_size": avg_batch_size,
            "batch_utilization": batch_utilization,
            "queue_size": len(self.request_queue),
            "max_batch_size": self.max_batch_size,
            "batch_timeout_ms": self.batch_timeout_ms,
            "running": self.running
        }


class InferenceOptimizer:
    """
    Model inference optimization system.
    
    Provides model warm-up, connection pooling, and resource optimization.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Optimization configuration
        self.enable_model_warmup = config.get("enable_model_warmup", True)
        self.warmup_iterations = config.get("warmup_iterations", 10)
        self.enable_threading = config.get("enable_threading", True)
        self.max_workers = config.get("max_workers", min(32, (os.cpu_count() or 1) + 4))
        
        # Thread pool for CPU-bound operations
        self.thread_pool = None
        if self.enable_threading:
            self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Model optimization state
        self.warmed_models: Dict[str, bool] = {}
        self.model_inference_times: Dict[str, List[float]] = {}
        
        # Resource monitoring
        self.performance_history: List[PerformanceMetrics] = []
        self.monitoring_task: Optional[asyncio.Task] = None
        
        logger.info(f"Inference optimizer initialized with {self.max_workers} workers")
    
    async def start_monitoring(self) -> None:
        """Start performance monitoring."""
        if self.monitoring_task is None:
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
    
    async def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            self.monitoring_task = None
    
    async def warm_up_model(self, model: TrajectoryPredictor) -> None:
        """Warm up model with dummy predictions."""
        
        if not self.enable_model_warmup or self.warmed_models.get(model.name, False):
            return
        
        logger.info(f"Warming up model: {model.name}")
        
        start_time = time.time()
        
        try:
            # Create dummy trajectory data for warmup
            dummy_trajectory = TrajectoryData(
                vehicle_id="warmup",
                positions=[[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]],
                time_steps=[0, 0.1, 0.2, 0.3, 0.4],
                metadata={}
            )
            
            # Run warmup iterations
            warmup_times = []
            for i in range(self.warmup_iterations):
                iter_start = time.time()
                
                try:
                    await model.predict_trajectory(dummy_trajectory, prediction_horizon=5.0)
                except Exception as e:
                    logger.warning(f"Warmup iteration {i} failed for {model.name}: {e}")
                
                iter_time = (time.time() - iter_start) * 1000
                warmup_times.append(iter_time)
            
            # Record warmup completion
            self.warmed_models[model.name] = True
            self.model_inference_times[model.name] = warmup_times
            
            total_warmup_time = (time.time() - start_time) * 1000
            avg_inference_time = np.mean(warmup_times) if warmup_times else 0.0
            
            logger.info(
                f"Model {model.name} warmed up in {total_warmup_time:.1f}ms "
                f"(avg inference: {avg_inference_time:.1f}ms)"
            )
            
        except Exception as e:
            logger.error(f"Model warmup failed for {model.name}: {e}")
    
    async def optimize_prediction(
        self,
        model: TrajectoryPredictor,
        trajectory_data: TrajectoryData,
        prediction_horizon: float
    ) -> Any:
        """Optimize prediction with performance enhancements."""
        
        start_time = time.time()
        
        try:
            # Ensure model is warmed up
            if not self.warmed_models.get(model.name, False):
                await self.warm_up_model(model)
            
            # Run prediction
            if self.enable_threading and self.thread_pool:
                # Run in thread pool for CPU-bound operations
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self.thread_pool,
                    self._sync_predict,
                    model,
                    trajectory_data,
                    prediction_horizon
                )
            else:
                # Run directly
                result = await model.predict_trajectory(trajectory_data, prediction_horizon)
            
            # Track inference time
            inference_time = (time.time() - start_time) * 1000
            if model.name not in self.model_inference_times:
                self.model_inference_times[model.name] = []
            
            self.model_inference_times[model.name].append(inference_time)
            
            # Keep only recent inference times
            if len(self.model_inference_times[model.name]) > 1000:
                self.model_inference_times[model.name] = self.model_inference_times[model.name][-1000:]
            
            return result
            
        except Exception as e:
            logger.error(f"Optimized prediction failed: {e}")
            raise
    
    def _sync_predict(
        self,
        model: TrajectoryPredictor,
        trajectory_data: TrajectoryData,
        prediction_horizon: float
    ) -> Any:
        """Synchronous prediction wrapper for thread pool."""
        
        # This would run the synchronous version of prediction
        # For now, we'll simulate it
        import time
        time.sleep(0.01)  # Simulate processing time
        return f"prediction_result_{trajectory_data.vehicle_id}"
    
    async def _monitoring_loop(self) -> None:
        """Performance monitoring loop."""
        
        while True:
            try:
                # Collect performance metrics
                metrics = self._collect_performance_metrics()
                self.performance_history.append(metrics)
                
                # Keep only recent history
                if len(self.performance_history) > 1000:
                    self.performance_history = self.performance_history[-1000:]
                
                # Log performance warnings
                if metrics.cpu_usage_percent > 80:
                    logger.warning(f"High CPU usage: {metrics.cpu_usage_percent:.1f}%")
                
                if metrics.memory_usage_percent > 80:
                    logger.warning(f"High memory usage: {metrics.memory_usage_percent:.1f}%")
                
                if metrics.queue_size > 100:
                    logger.warning(f"Large queue size: {metrics.queue_size}")
                
                # Wait before next monitoring cycle
                await asyncio.sleep(10)  # Monitor every 10 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(10)
    
    def _collect_performance_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics."""
        
        # System metrics
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # Inference time statistics
        all_inference_times = []
        for times in self.model_inference_times.values():
            all_inference_times.extend(times[-100:])  # Recent times
        
        avg_latency = np.mean(all_inference_times) if all_inference_times else 0.0
        p95_latency = np.percentile(all_inference_times, 95) if all_inference_times else 0.0
        p99_latency = np.percentile(all_inference_times, 99) if all_inference_times else 0.0
        
        # Calculate throughput (requests per second)
        throughput = 0.0
        if len(self.performance_history) > 1:
            recent_metrics = self.performance_history[-10:]  # Last 10 measurements
            time_span = recent_metrics[-1].timestamp - recent_metrics[0].timestamp
            request_diff = recent_metrics[-1].total_requests - recent_metrics[0].total_requests
            
            if time_span > 0:
                throughput = request_diff / time_span
        
        return PerformanceMetrics(
            total_requests=sum(len(times) for times in self.model_inference_times.values()),
            successful_requests=sum(len(times) for times in self.model_inference_times.values()),  # Placeholder
            failed_requests=0,  # Placeholder
            average_latency_ms=avg_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            throughput_rps=throughput,
            batch_utilization=0.0,  # Would get from batcher
            queue_size=0,  # Would get from batcher
            active_workers=self.thread_pool._threads if self.thread_pool else 0,
            cpu_usage_percent=cpu_usage,
            memory_usage_percent=memory.percent,
            timestamp=time.time()
        )
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        
        stats = {
            "warmed_models": list(self.warmed_models.keys()),
            "enable_threading": self.enable_threading,
            "max_workers": self.max_workers,
            "thread_pool_active": self.thread_pool is not None
        }
        
        # Add inference time statistics
        inference_stats = {}
        for model_name, times in self.model_inference_times.items():
            if times:
                inference_stats[model_name] = {
                    "count": len(times),
                    "average_ms": np.mean(times),
                    "p95_ms": np.percentile(times, 95),
                    "p99_ms": np.percentile(times, 99),
                    "min_ms": np.min(times),
                    "max_ms": np.max(times)
                }
        
        stats["inference_statistics"] = inference_stats
        
        # Add recent performance metrics
        if self.performance_history:
            recent_metrics = self.performance_history[-1]
            stats["current_performance"] = {
                "cpu_usage_percent": recent_metrics.cpu_usage_percent,
                "memory_usage_percent": recent_metrics.memory_usage_percent,
                "average_latency_ms": recent_metrics.average_latency_ms,
                "throughput_rps": recent_metrics.throughput_rps
            }
        
        return stats
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
            self.thread_pool = None
        
        logger.info("Inference optimizer cleaned up")


class HorizontalScaler:
    """
    Horizontal scaling system for API instances.
    
    Monitors load and provides scaling recommendations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Scaling configuration
        self.scale_up_threshold = config.get("scale_up_threshold", 80)  # CPU %
        self.scale_down_threshold = config.get("scale_down_threshold", 20)  # CPU %
        self.min_instances = config.get("min_instances", 1)
        self.max_instances = config.get("max_instances", 10)
        
        # Scaling state
        self.current_instances = 1
        self.scaling_history: List[Dict[str, Any]] = []
        
        logger.info("Horizontal scaler initialized")
    
    def should_scale_up(self, metrics: PerformanceMetrics) -> bool:
        """Determine if scaling up is needed."""
        
        conditions = [
            metrics.cpu_usage_percent > self.scale_up_threshold,
            metrics.queue_size > 50,  # Long queue
            metrics.average_latency_ms > 1000,  # High latency
            self.current_instances < self.max_instances
        ]
        
        return all(conditions[:3]) and conditions[3]  # All performance conditions + not at max
    
    def should_scale_down(self, metrics: PerformanceMetrics) -> bool:
        """Determine if scaling down is possible."""
        
        conditions = [
            metrics.cpu_usage_percent < self.scale_down_threshold,
            metrics.queue_size < 10,  # Short queue
            metrics.average_latency_ms < 200,  # Low latency
            self.current_instances > self.min_instances
        ]
        
        return all(conditions)
    
    def get_scaling_recommendation(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Get scaling recommendation based on current metrics."""
        
        recommendation = {
            "action": "none",
            "current_instances": self.current_instances,
            "recommended_instances": self.current_instances,
            "reason": "No scaling needed",
            "metrics": {
                "cpu_usage": metrics.cpu_usage_percent,
                "queue_size": metrics.queue_size,
                "latency_ms": metrics.average_latency_ms
            }
        }
        
        if self.should_scale_up(metrics):
            new_instances = min(self.current_instances + 1, self.max_instances)
            recommendation.update({
                "action": "scale_up",
                "recommended_instances": new_instances,
                "reason": "High load detected"
            })
            
        elif self.should_scale_down(metrics):
            new_instances = max(self.current_instances - 1, self.min_instances)
            recommendation.update({
                "action": "scale_down", 
                "recommended_instances": new_instances,
                "reason": "Low load detected"
            })
        
        return recommendation
    
    def record_scaling_event(self, action: str, old_instances: int, new_instances: int) -> None:
        """Record scaling event for history tracking."""
        
        event = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "old_instances": old_instances,
            "new_instances": new_instances
        }
        
        self.scaling_history.append(event)
        self.current_instances = new_instances
        
        # Keep only recent history
        if len(self.scaling_history) > 100:
            self.scaling_history = self.scaling_history[-100:]
        
        logger.info(f"Scaling event: {action} from {old_instances} to {new_instances} instances")
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get scaling statistics."""
        
        return {
            "current_instances": self.current_instances,
            "min_instances": self.min_instances,
            "max_instances": self.max_instances,
            "scale_up_threshold": self.scale_up_threshold,
            "scale_down_threshold": self.scale_down_threshold,
            "total_scaling_events": len(self.scaling_history),
            "recent_scaling_history": self.scaling_history[-10:] if self.scaling_history else []
        }