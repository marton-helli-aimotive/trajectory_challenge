#!/usr/bin/env python3
"""
REST API Client Example

This example demonstrates how to use the Trajectory Prediction REST API:
1. Starting the API server
2. Making single prediction requests
3. Batch prediction requests
4. Error handling and best practices
5. Performance monitoring

Prerequisites:
    Start the API server in another terminal:
    python -m trajectory_prediction.api.server

Usage:
    python examples/api_client_example.py [--api-url URL] [--batch-size N]

Examples:
    python examples/api_client_example.py
    python examples/api_client_example.py --api-url http://localhost:8000 --batch-size 5
"""

import asyncio
import aiohttp
import argparse
import json
import time
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass

# For generating test data
from trajectory_prediction.data.schemas import TrajectoryData, Position, Velocity


@dataclass
class APIResponse:
    """Structure for API response tracking."""
    status_code: int
    response_data: Dict[str, Any]
    response_time: float
    error_message: str = None


class TrajectoryAPIClient:
    """
    Client for interacting with the Trajectory Prediction API.
    
    This client provides methods for making predictions, handling errors,
    and monitoring performance.
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the API client.
        
        Args:
            base_url: Base URL of the API server
        """
        self.base_url = base_url.rstrip('/')
        self.session = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            connector=aiohttp.TCPConnector(limit=100)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def health_check(self) -> APIResponse:
        """Check API server health."""
        start_time = time.time()
        
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                response_time = time.time() - start_time
                data = await response.json()
                
                return APIResponse(
                    status_code=response.status,
                    response_data=data,
                    response_time=response_time
                )
        
        except Exception as e:
            return APIResponse(
                status_code=0,
                response_data={},
                response_time=time.time() - start_time,
                error_message=str(e)
            )
    
    async def predict_single(self, trajectory_data: Dict[str, Any]) -> APIResponse:
        """
        Make a single trajectory prediction.
        
        Args:
            trajectory_data: Trajectory data in API format
            
        Returns:
            APIResponse: Response with prediction results
        """
        start_time = time.time()
        
        try:
            async with self.session.post(
                f"{self.base_url}/predict",
                json=trajectory_data,
                headers={"Content-Type": "application/json"}
            ) as response:
                response_time = time.time() - start_time
                data = await response.json()
                
                return APIResponse(
                    status_code=response.status,
                    response_data=data,
                    response_time=response_time
                )
        
        except Exception as e:
            return APIResponse(
                status_code=0,
                response_data={},
                response_time=time.time() - start_time,
                error_message=str(e)
            )
    
    async def predict_batch(self, batch_data: Dict[str, Any]) -> APIResponse:
        """
        Make batch trajectory predictions.
        
        Args:
            batch_data: Batch trajectory data in API format
            
        Returns:
            APIResponse: Response with batch prediction results
        """
        start_time = time.time()
        
        try:
            async with self.session.post(
                f"{self.base_url}/predict/batch",
                json=batch_data,
                headers={"Content-Type": "application/json"}
            ) as response:
                response_time = time.time() - start_time
                data = await response.json()
                
                return APIResponse(
                    status_code=response.status,
                    response_data=data,
                    response_time=response_time
                )
        
        except Exception as e:
            return APIResponse(
                status_code=0,
                response_data={},
                response_time=time.time() - start_time,
                error_message=str(e)
            )
    
    async def get_models(self) -> APIResponse:
        """Get list of available models."""
        start_time = time.time()
        
        try:
            async with self.session.get(f"{self.base_url}/models") as response:
                response_time = time.time() - start_time
                data = await response.json()
                
                return APIResponse(
                    status_code=response.status,
                    response_data=data,
                    response_time=response_time
                )
        
        except Exception as e:
            return APIResponse(
                status_code=0,
                response_data={},
                response_time=time.time() - start_time,
                error_message=str(e)
            )


def create_sample_trajectory_data(trajectory_id: str, scenario: str = "straight") -> Dict[str, Any]:
    """
    Create sample trajectory data for API requests.
    
    Args:
        trajectory_id: Unique identifier for trajectory
        scenario: Type of scenario ('straight', 'curved')
        
    Returns:
        dict: Trajectory data in API format
    """
    if scenario == "straight":
        # Straight line trajectory
        points = []
        for i in range(5):
            timestamp = i * 0.1
            x = 15.0 * timestamp  # 15 m/s forward
            y = 1.0 * timestamp   # 1 m/s sideways
            
            points.append({
                "timestamp": timestamp,
                "x": x,
                "y": y,
                "vx": 15.0,
                "vy": 1.0
            })
    
    elif scenario == "curved":
        # Curved trajectory (partial circle)
        points = []
        radius = 20.0
        angular_velocity = 0.1
        
        for i in range(5):
            timestamp = i * 0.1
            angle = angular_velocity * timestamp
            
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            vx = -radius * angular_velocity * np.sin(angle)
            vy = radius * angular_velocity * np.cos(angle)
            
            points.append({
                "timestamp": timestamp,
                "x": x,
                "y": y,
                "vx": vx,
                "vy": vy
            })
    
    return {
        "trajectory": {
            "trajectory_id": trajectory_id,
            "vehicle_id": f"vehicle_{trajectory_id}",
            "points": points
        },
        "config": {
            "prediction_horizon": 3.0,
            "time_step": 0.1,
            "models": ["constant_velocity"],
            "include_uncertainty": True
        }
    }


async def demonstrate_health_check(client: TrajectoryAPIClient):
    """Demonstrate API health check."""
    print("\n🏥 API Health Check")
    print("=" * 30)
    
    response = await client.health_check()
    
    if response.status_code == 200:
        print(f"✅ API is healthy!")
        print(f"   Response time: {response.response_time*1000:.1f}ms")
        print(f"   Server status: {response.response_data.get('status', 'unknown')}")
        print(f"   Version: {response.response_data.get('version', 'unknown')}")
        
        # Show system info if available
        if 'system_info' in response.response_data:
            sys_info = response.response_data['system_info']
            print(f"   Models loaded: {sys_info.get('models_loaded', 'unknown')}")
            print(f"   Total requests: {sys_info.get('total_requests', 'unknown')}")
    else:
        print(f"❌ API health check failed!")
        print(f"   Status code: {response.status_code}")
        print(f"   Error: {response.error_message}")
        return False
    
    return True


async def demonstrate_single_prediction(client: TrajectoryAPIClient):
    """Demonstrate single trajectory prediction."""
    print("\n🎯 Single Trajectory Prediction")
    print("=" * 35)
    
    # Create sample trajectory data
    trajectory_data = create_sample_trajectory_data("demo_001", scenario="straight")
    
    print(f"📤 Sending prediction request...")
    print(f"   Trajectory ID: {trajectory_data['trajectory']['trajectory_id']}")
    print(f"   Input points: {len(trajectory_data['trajectory']['points'])}")
    print(f"   Prediction horizon: {trajectory_data['config']['prediction_horizon']}s")
    print(f"   Models: {', '.join(trajectory_data['config']['models'])}")
    
    # Make prediction request
    response = await client.predict_single(trajectory_data)
    
    if response.status_code == 200:
        print(f"✅ Prediction successful!")
        print(f"   Response time: {response.response_time*1000:.1f}ms")
        
        data = response.response_data
        print(f"   Request ID: {data.get('request_id', 'unknown')}")
        print(f"   Model used: {data.get('model_name', 'unknown')}")
        print(f"   Confidence: {data.get('confidence', 0):.3f}")
        print(f"   Inference time: {data.get('inference_time', 0)*1000:.1f}ms")
        
        # Show prediction details
        if 'predicted_trajectory' in data:
            pred_traj = data['predicted_trajectory']
            num_positions = len(pred_traj.get('positions', []))
            print(f"   Predicted points: {num_positions}")
            
            if num_positions > 0:
                final_pos = pred_traj['positions'][-1]
                print(f"   Final position: ({final_pos['x']:.2f}, {final_pos['y']:.2f})")
        
        # Show uncertainty if available
        if 'uncertainty' in data:
            uncertainty = data['uncertainty']
            avg_uncertainty = np.mean([v for v in uncertainty.values() if isinstance(v, (int, float))])
            print(f"   Average uncertainty: {avg_uncertainty:.3f}m")
    
    else:
        print(f"❌ Prediction failed!")
        print(f"   Status code: {response.status_code}")
        print(f"   Error: {response.error_message}")
        
        # Show detailed error if available
        if response.response_data:
            error_detail = response.response_data.get('detail', 'No details available')
            print(f"   Details: {error_detail}")
    
    return response.status_code == 200


async def demonstrate_batch_prediction(client: TrajectoryAPIClient, batch_size: int = 3):
    """Demonstrate batch trajectory prediction."""
    print(f"\n📦 Batch Trajectory Prediction (size: {batch_size})")
    print("=" * 50)
    
    # Create batch of trajectory data
    trajectories = []
    scenarios = ["straight", "curved"]
    
    for i in range(batch_size):
        scenario = scenarios[i % len(scenarios)]
        trajectory_data = create_sample_trajectory_data(f"batch_{i:03d}", scenario=scenario)
        trajectories.append(trajectory_data['trajectory'])
    
    batch_data = {
        "trajectories": trajectories,
        "config": {
            "prediction_horizon": 2.5,
            "time_step": 0.1,
            "models": ["constant_velocity", "constant_acceleration"]
        }
    }
    
    print(f"📤 Sending batch prediction request...")
    print(f"   Number of trajectories: {len(trajectories)}")
    print(f"   Models: {', '.join(batch_data['config']['models'])}")
    
    # Make batch prediction request
    response = await client.predict_batch(batch_data)
    
    if response.status_code == 200:
        print(f"✅ Batch prediction successful!")
        print(f"   Total response time: {response.response_time*1000:.1f}ms")
        
        data = response.response_data
        predictions = data.get('predictions', [])
        print(f"   Predictions received: {len(predictions)}")
        
        # Analyze batch results
        if predictions:
            total_inference_time = sum(pred.get('inference_time', 0) for pred in predictions)
            avg_confidence = np.mean([pred.get('confidence', 0) for pred in predictions])
            
            print(f"   Total inference time: {total_inference_time*1000:.1f}ms")
            print(f"   Average confidence: {avg_confidence:.3f}")
            print(f"   Throughput: {len(predictions)/response.response_time:.1f} pred/sec")
            
            # Show individual results
            print(f"\n   Individual Results:")
            for i, pred in enumerate(predictions[:3]):  # Show first 3
                model = pred.get('model_name', 'unknown')
                conf = pred.get('confidence', 0)
                inf_time = pred.get('inference_time', 0) * 1000
                print(f"     #{i+1}: {model}, confidence={conf:.3f}, time={inf_time:.1f}ms")
                
            if len(predictions) > 3:
                print(f"     ... and {len(predictions)-3} more")
        
        # Batch metadata
        if 'batch_metadata' in data:
            metadata = data['batch_metadata']
            print(f"   Total predictions: {metadata.get('total_predictions', 0)}")
            print(f"   Batch processing time: {metadata.get('total_inference_time', 0)*1000:.1f}ms")
    
    else:
        print(f"❌ Batch prediction failed!")
        print(f"   Status code: {response.status_code}")
        print(f"   Error: {response.error_message}")
    
    return response.status_code == 200


async def demonstrate_model_info(client: TrajectoryAPIClient):
    """Demonstrate retrieving model information."""
    print("\n🤖 Available Models Information")
    print("=" * 35)
    
    response = await client.get_models()
    
    if response.status_code == 200:
        print(f"✅ Models retrieved successfully!")
        print(f"   Response time: {response.response_time*1000:.1f}ms")
        
        data = response.response_data
        models = data.get('models', [])
        print(f"   Available models: {len(models)}")
        
        # Show model details
        for i, model in enumerate(models, 1):
            print(f"\n   Model #{i}: {model.get('name', 'unknown')}")
            print(f"     Type: {model.get('type', 'unknown')}")
            print(f"     Version: {model.get('version', 'unknown')}")
            print(f"     Status: {model.get('status', 'unknown')}")
            print(f"     Description: {model.get('description', 'No description')}")
            
            # Performance metrics
            if 'metrics' in model:
                metrics = model['metrics']
                rmse = metrics.get('rmse', 0)
                inference_time = metrics.get('inference_time_ms', 0)
                print(f"     RMSE: {rmse:.3f}m")
                print(f"     Inference time: {inference_time:.1f}ms")
    
    else:
        print(f"❌ Failed to retrieve models!")
        print(f"   Status code: {response.status_code}")
        print(f"   Error: {response.error_message}")
    
    return response.status_code == 200


async def demonstrate_error_handling(client: TrajectoryAPIClient):
    """Demonstrate error handling with invalid requests."""
    print("\n⚠️  Error Handling Demonstration")
    print("=" * 35)
    
    # Test 1: Invalid trajectory data
    print("🔸 Test 1: Invalid trajectory data")
    invalid_data = {
        "trajectory": {
            "trajectory_id": "invalid_test",
            "vehicle_id": "test_vehicle",
            "points": []  # Empty points - should cause error
        },
        "config": {
            "prediction_horizon": 3.0,
            "models": ["constant_velocity"]
        }
    }
    
    response = await client.predict_single(invalid_data)
    if response.status_code != 200:
        print(f"   ✓ Correctly caught error: {response.status_code}")
        if response.response_data:
            error_detail = response.response_data.get('detail', 'No details')
            print(f"   Error details: {error_detail}")
    else:
        print(f"   ⚠️  Expected error but got success")
    
    # Test 2: Non-existent model
    print("\n🔸 Test 2: Non-existent model")
    invalid_model_data = {
        "trajectory": {
            "trajectory_id": "model_test",
            "vehicle_id": "test_vehicle",
            "points": [
                {"timestamp": 0.0, "x": 0.0, "y": 0.0, "vx": 10.0, "vy": 0.0},
                {"timestamp": 0.1, "x": 1.0, "y": 0.0, "vx": 10.0, "vy": 0.0}
            ]
        },
        "config": {
            "prediction_horizon": 2.0,
            "models": ["non_existent_model"]  # This model doesn't exist
        }
    }
    
    response = await client.predict_single(invalid_model_data)
    if response.status_code != 200:
        print(f"   ✓ Correctly caught error: {response.status_code}")
        if response.response_data:
            error_detail = response.response_data.get('detail', 'No details')
            print(f"   Error details: {error_detail}")
    else:
        print(f"   ⚠️  Expected error but got success")


async def run_performance_benchmark(client: TrajectoryAPIClient, num_requests: int = 10):
    """Run a simple performance benchmark."""
    print(f"\n⚡ Performance Benchmark ({num_requests} requests)")
    print("=" * 45)
    
    # Prepare test data
    trajectory_data = create_sample_trajectory_data("benchmark_001", scenario="straight")
    
    print("🚀 Running benchmark...")
    response_times = []
    successful_requests = 0
    
    start_time = time.time()
    
    # Make requests
    tasks = []
    for i in range(num_requests):
        task = client.predict_single(trajectory_data)
        tasks.append(task)
    
    # Wait for all requests to complete
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    
    total_time = time.time() - start_time
    
    # Analyze results
    for response in responses:
        if isinstance(response, APIResponse) and response.status_code == 200:
            response_times.append(response.response_time)
            successful_requests += 1
    
    if response_times:
        avg_response_time = np.mean(response_times) * 1000
        min_response_time = np.min(response_times) * 1000
        max_response_time = np.max(response_times) * 1000
        std_response_time = np.std(response_times) * 1000
        throughput = successful_requests / total_time
        
        print(f"📊 Benchmark Results:")
        print(f"   Total requests: {num_requests}")
        print(f"   Successful: {successful_requests}")
        print(f"   Success rate: {successful_requests/num_requests*100:.1f}%")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Throughput: {throughput:.1f} req/sec")
        print(f"   Response times:")
        print(f"     Average: {avg_response_time:.1f}ms")
        print(f"     Min: {min_response_time:.1f}ms")
        print(f"     Max: {max_response_time:.1f}ms")
        print(f"     Std Dev: {std_response_time:.1f}ms")
    else:
        print(f"❌ No successful responses received")


async def run_api_client_example(api_url: str, batch_size: int):
    """
    Run the complete API client example.
    
    Args:
        api_url: API server URL
        batch_size: Size of batch for batch prediction demo
    """
    print("🌐 Trajectory Prediction API Client Example")
    print("=" * 50)
    print(f"API URL: {api_url}")
    
    async with TrajectoryAPIClient(api_url) as client:
        # 1. Health check
        health_ok = await demonstrate_health_check(client)
        if not health_ok:
            print("\n❌ API server is not available. Please ensure it's running:")
            print("   python -m trajectory_prediction.api.server")
            return
        
        # 2. Model information
        await demonstrate_model_info(client)
        
        # 3. Single prediction
        single_ok = await demonstrate_single_prediction(client)
        
        # 4. Batch prediction
        if single_ok:
            await demonstrate_batch_prediction(client, batch_size)
        
        # 5. Error handling
        await demonstrate_error_handling(client)
        
        # 6. Performance benchmark
        if single_ok:
            await run_performance_benchmark(client, num_requests=5)
    
    print("\n✅ API client example completed!")
    print("\nNext steps:")
    print("  • Try the interactive dashboard: streamlit run src/trajectory_prediction/visualization/dashboard.py")
    print("  • Explore more examples: python examples/model_comparison.py")
    print("  • Check the API documentation: http://localhost:8000/docs")


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="REST API Client Example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--api-url',
        type=str,
        default='http://localhost:8000',
        help='API server URL'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=3,
        help='Batch size for batch prediction demo'
    )
    
    args = parser.parse_args()
    
    # Run the example
    try:
        asyncio.run(run_api_client_example(args.api_url, args.batch_size))
    except KeyboardInterrupt:
        print("\n⚠️  Example interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running example: {str(e)}")
        print("\nMake sure the API server is running:")
        print("  python -m trajectory_prediction.api.server")
        raise


if __name__ == '__main__':
    main()