"""
API usage examples and code snippets.

This module provides:
- Complete usage examples for all API endpoints
- Integration examples for common use cases
- Best practices and optimization tips
- Sample client implementations
"""

import asyncio
import json
import time
from typing import List, Dict, Any, Optional
from datetime import datetime

import httpx
import numpy as np

from .models import (
    TrajectoryRequest, TrajectoryResponse, BatchTrajectoryRequest,
    TrajectoryInput, TrajectoryPoint, PredictionConfig,
    LoadTestConfig, LoadTestResult
)


class TrajectoryPredictionClient:
    """
    Example client for the Trajectory Prediction API.
    
    Demonstrates best practices for interacting with the API.
    """
    
    def __init__(self, base_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        self.base_url = base_url
        self.api_key = api_key
        self.client = None
        
        # Client configuration
        self.timeout = 30.0
        self.retry_attempts = 3
        self.retry_delay = 1.0
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
    
    async def connect(self) -> None:
        """Establish connection to API."""
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=self.timeout
        )
        
        # Test connection
        response = await self.client.get("/health")
        if response.status_code != 200:
            raise ConnectionError(f"Failed to connect to API: {response.status_code}")
    
    async def disconnect(self) -> None:
        """Close connection to API."""
        if self.client:
            await self.client.aclose()
    
    async def predict_single_trajectory(
        self,
        vehicle_id: str,
        positions: List[List[float]],
        time_steps: List[float],
        prediction_horizon: float = 10.0,
        model_name: Optional[str] = None,
        use_ensemble: bool = False
    ) -> TrajectoryResponse:
        """
        Predict a single trajectory.
        
        Args:
            vehicle_id: Unique vehicle identifier
            positions: List of [x, y] position coordinates
            time_steps: Corresponding time stamps
            prediction_horizon: How far to predict (seconds)
            model_name: Specific model to use
            use_ensemble: Whether to use ensemble prediction
            
        Returns:
            Trajectory prediction response
        """
        
        # Create trajectory points
        trajectory_points = [
            TrajectoryPoint(x=pos[0], y=pos[1], timestamp=t)
            for pos, t in zip(positions, time_steps)
        ]
        
        # Create request
        trajectory_input = TrajectoryInput(
            vehicle_id=vehicle_id,
            trajectory_points=trajectory_points,
            metadata={}
        )
        
        config = PredictionConfig(
            prediction_horizon=prediction_horizon,
            time_resolution=0.1,
            uncertainty_quantification="gaussian",
            confidence_level=0.95
        )
        
        request = TrajectoryRequest(
            trajectory=trajectory_input,
            config=config,
            model_name=model_name,
            use_ensemble=use_ensemble
        )
        
        # Make request with retries
        for attempt in range(self.retry_attempts):
            try:
                response = await self.client.post("/predict", json=request.dict())
                
                if response.status_code == 200:
                    return TrajectoryResponse(**response.json())
                else:
                    error_msg = f"API error: {response.status_code} - {response.text}"
                    if attempt < self.retry_attempts - 1:
                        await asyncio.sleep(self.retry_delay * (attempt + 1))
                        continue
                    else:
                        raise RuntimeError(error_msg)
                        
            except httpx.RequestError as e:
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                    continue
                else:
                    raise ConnectionError(f"Request failed: {e}")
    
    async def predict_batch_trajectories(
        self,
        trajectories: List[Dict[str, Any]],
        prediction_horizon: float = 10.0,
        parallel_processing: bool = True
    ) -> BatchTrajectoryResponse:
        """
        Predict multiple trajectories in batch.
        
        Args:
            trajectories: List of trajectory data dictionaries
            prediction_horizon: How far to predict (seconds)
            parallel_processing: Whether to process in parallel
            
        Returns:
            Batch prediction response
        """
        
        # Convert to request format
        trajectory_requests = []
        
        for i, traj_data in enumerate(trajectories):
            trajectory_points = [
                TrajectoryPoint(x=pos[0], y=pos[1], timestamp=t)
                for pos, t in zip(traj_data["positions"], traj_data["time_steps"])
            ]
            
            trajectory_input = TrajectoryInput(
                vehicle_id=traj_data.get("vehicle_id", f"vehicle_{i}"),
                trajectory_points=trajectory_points,
                metadata=traj_data.get("metadata", {})
            )
            
            config = PredictionConfig(prediction_horizon=prediction_horizon)
            
            request = TrajectoryRequest(
                request_id=f"batch_req_{i}",
                trajectory=trajectory_input,
                config=config
            )
            
            trajectory_requests.append(request)
        
        # Create batch request
        batch_request = BatchTrajectoryRequest(
            trajectories=trajectory_requests,
            parallel_processing=parallel_processing,
            timeout_seconds=60.0
        )
        
        # Make request
        response = await self.client.post("/predict/batch", json=batch_request.dict())
        
        if response.status_code == 200:
            return BatchTrajectoryResponse(**response.json())
        else:
            raise RuntimeError(f"Batch prediction failed: {response.status_code} - {response.text}")
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get API health status."""
        response = await self.client.get("/health")
        return response.json()
    
    async def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models."""
        response = await self.client.get("/models")
        return response.json()
    
    async def get_cache_statistics(self) -> Optional[Dict[str, Any]]:
        """Get cache statistics if available."""
        response = await self.client.get("/cache/stats")
        if response.status_code == 200:
            return response.json()
        return None
    
    async def clear_cache(self) -> bool:
        """Clear prediction cache."""
        response = await self.client.delete("/cache")
        return response.status_code == 200


async def example_single_prediction():
    """Example: Single trajectory prediction."""
    
    print("=== Single Trajectory Prediction Example ===")
    
    async with TrajectoryPredictionClient() as client:
        # Create sample trajectory (vehicle moving in straight line)
        positions = [[i * 0.5, 0.0] for i in range(10)]
        time_steps = [i * 0.1 for i in range(10)]
        
        try:
            response = await client.predict_single_trajectory(
                vehicle_id="example_vehicle_1",
                positions=positions,
                time_steps=time_steps,
                prediction_horizon=5.0
            )
            
            print(f"Prediction Status: {response.status}")
            print(f"Model Used: {response.metadata.model_name}")
            print(f"Inference Time: {response.metadata.inference_time_ms:.1f}ms")
            
            if response.predicted_trajectory:
                print(f"Predicted {len(response.predicted_trajectory)} future points")
                
                # Show first few predicted points
                for i, point in enumerate(response.predicted_trajectory[:3]):
                    print(f"  Point {i+1}: ({point.x:.2f}, {point.y:.2f}) at t={point.timestamp:.2f}s")
            
            if response.warnings:
                print(f"Warnings: {response.warnings}")
                
        except Exception as e:
            print(f"Prediction failed: {e}")


async def example_batch_prediction():
    """Example: Batch trajectory prediction."""
    
    print("\n=== Batch Trajectory Prediction Example ===")
    
    async with TrajectoryPredictionClient() as client:
        # Create multiple sample trajectories
        trajectories = []
        
        for i in range(5):
            # Create varied trajectories
            if i % 2 == 0:
                # Straight line trajectory
                positions = [[j * 0.3, 0.0] for j in range(8)]
            else:
                # Curved trajectory
                positions = [[j * 0.3, 0.1 * j * j] for j in range(8)]
            
            time_steps = [j * 0.2 for j in range(8)]
            
            trajectories.append({
                "vehicle_id": f"batch_vehicle_{i}",
                "positions": positions,
                "time_steps": time_steps,
                "metadata": {"batch_example": True, "trajectory_type": "straight" if i % 2 == 0 else "curved"}
            })
        
        try:
            response = await client.predict_batch_trajectories(
                trajectories=trajectories,
                prediction_horizon=3.0,
                parallel_processing=True
            )
            
            print(f"Batch Status: {response.status}")
            print(f"Total Processing Time: {response.total_processing_time_ms:.1f}ms")
            print(f"Success Rate: {response.summary['success_rate']:.1%}")
            print(f"Processed {len(response.results)} trajectories")
            
            # Show results summary
            successful = len([r for r in response.results if r.status == "success"])
            failed = len(response.results) - successful
            
            print(f"Results: {successful} successful, {failed} failed")
            
        except Exception as e:
            print(f"Batch prediction failed: {e}")


async def example_model_information():
    """Example: Get model information."""
    
    print("\n=== Model Information Example ===")
    
    async with TrajectoryPredictionClient() as client:
        try:
            models = await client.get_available_models()
            
            print(f"Available models: {len(models)}")
            
            for model in models:
                print(f"\nModel: {model['model_name']}")
                print(f"  Type: {model['model_type']}")
                print(f"  Version: {model.get('model_version', 'N/A')}")
                print(f"  Status: {model['status']}")
                print(f"  Default: {model.get('is_default', False)}")
                
                if 'performance_metrics' in model and model['performance_metrics']:
                    print("  Performance Metrics:")
                    for metric, value in model['performance_metrics'].items():
                        print(f"    {metric}: {value}")
        
        except Exception as e:
            print(f"Failed to get model information: {e}")


async def example_health_monitoring():
    """Example: Health monitoring and system information."""
    
    print("\n=== Health Monitoring Example ===")
    
    async with TrajectoryPredictionClient() as client:
        try:
            health = await client.get_health_status()
            
            print(f"API Status: {health['status']}")
            print(f"Version: {health['version']}")
            print(f"Uptime: {health['uptime_seconds']:.1f} seconds")
            print(f"Models Loaded: {health['models_loaded']}")
            print(f"Total Predictions: {health['total_predictions']}")
            print(f"Cache Hit Rate: {health['cache_hit_rate']:.1%}")
            print(f"Average Response Time: {health['average_response_time_ms']:.1f}ms")
            
            if 'system_info' in health:
                sys_info = health['system_info']
                print(f"\nSystem Information:")
                print(f"  CPU Cores: {sys_info.get('cpu_count', 'N/A')}")
                print(f"  Memory Total: {sys_info.get('memory_total_gb', 0):.1f} GB")
                print(f"  Memory Available: {sys_info.get('memory_available_gb', 0):.1f} GB")
                print(f"  Disk Usage: {sys_info.get('disk_usage_percent', 0):.1f}%")
        
        except Exception as e:
            print(f"Failed to get health status: {e}")


async def example_cache_management():
    """Example: Cache management and statistics."""
    
    print("\n=== Cache Management Example ===")
    
    async with TrajectoryPredictionClient() as client:
        try:
            cache_stats = await client.get_cache_statistics()
            
            if cache_stats:
                print("Cache Statistics:")
                print(f"  Cache Type: {cache_stats.get('cache_type', 'N/A')}")
                print(f"  Current Size: {cache_stats.get('cache_size', 0)}")
                print(f"  Max Size: {cache_stats.get('max_cache_size', 0)}")
                print(f"  Hit Rate: {cache_stats.get('hit_rate', 0):.1%}")
                print(f"  Total Requests: {cache_stats.get('total_requests', 0)}")
                
                # Clear cache example
                print("\nClearing cache...")
                success = await client.clear_cache()
                if success:
                    print("Cache cleared successfully")
                else:
                    print("Failed to clear cache")
            else:
                print("Cache not enabled or not available")
        
        except Exception as e:
            print(f"Cache management failed: {e}")


async def example_error_handling():
    """Example: Proper error handling and retry logic."""
    
    print("\n=== Error Handling Example ===")
    
    async with TrajectoryPredictionClient() as client:
        # Test with invalid data to demonstrate error handling
        try:
            # Invalid trajectory (too few points)
            positions = [[0, 0]]  # Only one point
            time_steps = [0]
            
            response = await client.predict_single_trajectory(
                vehicle_id="error_test",
                positions=positions,
                time_steps=time_steps,
                prediction_horizon=5.0
            )
            
            print("Unexpected success with invalid data")
            
        except RuntimeError as e:
            print(f"Handled API error: {e}")
        except ConnectionError as e:
            print(f"Handled connection error: {e}")
        except Exception as e:
            print(f"Handled unexpected error: {e}")


def example_data_preparation():
    """Example: How to prepare trajectory data."""
    
    print("\n=== Data Preparation Example ===")
    
    # Example 1: From GPS coordinates
    gps_data = [
        {"lat": 37.7749, "lon": -122.4194, "timestamp": 1643723400.0},
        {"lat": 37.7750, "lon": -122.4193, "timestamp": 1643723401.0},
        {"lat": 37.7751, "lon": -122.4192, "timestamp": 1643723402.0},
    ]
    
    print("Converting GPS data to trajectory format:")
    
    # Convert to local coordinates (simplified - in practice would use proper projection)
    base_lat, base_lon = gps_data[0]["lat"], gps_data[0]["lon"]
    
    positions = []
    time_steps = []
    
    for point in gps_data:
        # Simple conversion (not accurate for long distances)
        x = (point["lon"] - base_lon) * 111320 * np.cos(np.radians(base_lat))  # meters
        y = (point["lat"] - base_lat) * 110540  # meters
        t = point["timestamp"] - gps_data[0]["timestamp"]  # relative time
        
        positions.append([x, y])
        time_steps.append(t)
    
    print(f"Converted {len(positions)} GPS points to trajectory")
    for i, (pos, t) in enumerate(zip(positions, time_steps)):
        print(f"  Point {i+1}: ({pos[0]:.2f}, {pos[1]:.2f}) at t={t:.1f}s")
    
    # Example 2: From vehicle sensor data
    print("\nFrom vehicle sensor data:")
    sensor_data = {
        "speed_mps": [10.0, 9.8, 9.5, 9.2, 9.0],  # Speed in m/s
        "heading_deg": [0, 2, 5, 8, 12],  # Heading in degrees
        "timestamps": [0, 0.5, 1.0, 1.5, 2.0]  # Time in seconds
    }
    
    # Convert to positions using dead reckoning
    positions = [[0, 0]]  # Start at origin
    
    for i in range(1, len(sensor_data["speed_mps"])):
        dt = sensor_data["timestamps"][i] - sensor_data["timestamps"][i-1]
        speed = sensor_data["speed_mps"][i-1]
        heading_rad = np.radians(sensor_data["heading_deg"][i-1])
        
        dx = speed * dt * np.cos(heading_rad)
        dy = speed * dt * np.sin(heading_rad)
        
        new_pos = [positions[-1][0] + dx, positions[-1][1] + dy]
        positions.append(new_pos)
    
    print(f"Generated {len(positions)} position points from sensor data")


async def example_performance_optimization():
    """Example: Performance optimization techniques."""
    
    print("\n=== Performance Optimization Example ===")
    
    async with TrajectoryPredictionClient() as client:
        # Technique 1: Batch similar requests
        print("1. Batching similar requests:")
        
        start_time = time.time()
        
        # Individual requests (slower)
        individual_times = []
        for i in range(3):
            positions = [[j * 0.5, 0] for j in range(5)]
            time_steps = [j * 0.1 for j in range(5)]
            
            pred_start = time.time()
            response = await client.predict_single_trajectory(
                vehicle_id=f"individual_{i}",
                positions=positions,
                time_steps=time_steps,
                prediction_horizon=2.0
            )
            individual_times.append((time.time() - pred_start) * 1000)
        
        individual_total = time.time() - start_time
        
        # Batch request (faster)
        batch_start = time.time()
        
        trajectories = []
        for i in range(3):
            trajectories.append({
                "vehicle_id": f"batch_{i}",
                "positions": [[j * 0.5, 0] for j in range(5)],
                "time_steps": [j * 0.1 for j in range(5)]
            })
        
        batch_response = await client.predict_batch_trajectories(
            trajectories=trajectories,
            prediction_horizon=2.0
        )
        
        batch_total = time.time() - batch_start
        
        print(f"  Individual requests: {individual_total*1000:.1f}ms total")
        print(f"  Batch request: {batch_total*1000:.1f}ms total")
        print(f"  Speedup: {individual_total/batch_total:.1f}x")
        
        # Technique 2: Cache awareness
        print("\n2. Cache-aware predictions:")
        
        # Make same prediction twice to test caching
        positions = [[0, 0], [1, 0], [2, 0]]
        time_steps = [0, 1, 2]
        
        # First prediction (cache miss)
        start = time.time()
        response1 = await client.predict_single_trajectory(
            vehicle_id="cache_test",
            positions=positions,
            time_steps=time_steps,
            prediction_horizon=5.0
        )
        first_time = (time.time() - start) * 1000
        
        # Second prediction (cache hit)
        start = time.time()
        response2 = await client.predict_single_trajectory(
            vehicle_id="cache_test",
            positions=positions,
            time_steps=time_steps,
            prediction_horizon=5.0
        )
        second_time = (time.time() - start) * 1000
        
        print(f"  First prediction: {first_time:.1f}ms ({response1.status})")
        print(f"  Second prediction: {second_time:.1f}ms ({response2.status})")
        if second_time < first_time * 0.5:  # Significantly faster
            print(f"  Cache speedup: {first_time/second_time:.1f}x")


async def run_all_examples():
    """Run all API usage examples."""
    
    print("🚗 Trajectory Prediction API - Usage Examples")
    print("=" * 50)
    
    try:
        await example_single_prediction()
        await example_batch_prediction()
        await example_model_information()
        await example_health_monitoring()
        await example_cache_management()
        await example_error_handling()
        example_data_preparation()
        await example_performance_optimization()
        
        print("\n✅ All examples completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        print("Make sure the API server is running at http://localhost:8000")


if __name__ == "__main__":
    # Run examples
    asyncio.run(run_all_examples())