"""
Pytest configuration and shared fixtures for trajectory prediction tests.

This module provides:
- Common test fixtures for data and models
- Test configuration and setup
- Mock objects and test utilities
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
from pathlib import Path
import asyncio
import tempfile
import shutil

# Import system components for testing
from src.trajectory_prediction.data.schemas import TrajectoryData, Position, Velocity
from src.trajectory_prediction.models.base import TrajectoryPredictor
from src.trajectory_prediction.api.models import TrajectoryRequest, TrajectoryResponse
from src.trajectory_prediction.evaluation.metrics import TrajectoryMetrics


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_positions() -> List[Position]:
    """Generate sample trajectory positions."""
    positions = []
    for i in range(20):
        x = float(i * 2.0 + np.random.normal(0, 0.1))
        y = float(np.sin(i * 0.3) * 5 + np.random.normal(0, 0.1))
        positions.append(Position(x=x, y=y))
    return positions


@pytest.fixture
def sample_velocities() -> List[Velocity]:
    """Generate sample trajectory velocities."""
    velocities = []
    for i in range(20):
        vx = float(2.0 + np.random.normal(0, 0.2))
        vy = float(np.cos(i * 0.3) * 1.5 + np.random.normal(0, 0.2))
        velocities.append(Velocity(vx=vx, vy=vy))
    return velocities


@pytest.fixture
def sample_timestamps() -> List[float]:
    """Generate sample trajectory timestamps."""
    start_time = datetime.now().timestamp()
    return [start_time + i * 0.1 for i in range(20)]


@pytest.fixture
def sample_trajectory(sample_positions, sample_velocities, sample_timestamps) -> TrajectoryData:
    """Generate a complete sample trajectory."""
    return TrajectoryData(
        trajectory_id="test_001",
        vehicle_id="vehicle_001",
        positions=sample_positions,
        velocities=sample_velocities,
        timestamps=sample_timestamps,
        metadata={"source": "test", "scenario": "highway"}
    )


@pytest.fixture
def sample_trajectories(sample_trajectory) -> List[TrajectoryData]:
    """Generate multiple sample trajectories."""
    trajectories = []
    
    for i in range(5):
        # Create variations of the base trajectory
        positions = []
        velocities = []
        timestamps = []
        
        for j in range(20):
            # Add some variation to each trajectory
            offset_x = i * 10.0
            offset_y = i * 2.0
            
            x = float(j * 2.0 + offset_x + np.random.normal(0, 0.2))
            y = float(np.sin(j * 0.3) * 5 + offset_y + np.random.normal(0, 0.2))
            positions.append(Position(x=x, y=y))
            
            vx = float(2.0 + np.random.normal(0, 0.3))
            vy = float(np.cos(j * 0.3) * 1.5 + np.random.normal(0, 0.3))
            velocities.append(Velocity(vx=vx, vy=vy))
            
            timestamp = sample_trajectory.timestamps[0] + j * 0.1 + i * 100
            timestamps.append(timestamp)
        
        trajectory = TrajectoryData(
            trajectory_id=f"test_{i:03d}",
            vehicle_id=f"vehicle_{i:03d}",
            positions=positions,
            velocities=velocities,
            timestamps=timestamps,
            metadata={"source": "test", "scenario": "highway", "variant": i}
        )
        trajectories.append(trajectory)
    
    return trajectories


@pytest.fixture
def sample_trajectory_request(sample_trajectory) -> TrajectoryRequest:
    """Generate a sample trajectory prediction request."""
    from src.trajectory_prediction.api.models import TrajectoryInput, TrajectoryPoint, PredictionConfig
    
    # Convert trajectory to request format
    trajectory_points = []
    for i, (pos, vel) in enumerate(zip(sample_trajectory.positions, sample_trajectory.velocities)):
        point = TrajectoryPoint(
            timestamp=sample_trajectory.timestamps[i],
            x=pos.x,
            y=pos.y,
            vx=vel.vx,
            vy=vel.vy
        )
        trajectory_points.append(point)
    
    trajectory_input = TrajectoryInput(
        trajectory_id=sample_trajectory.trajectory_id,
        vehicle_id=sample_trajectory.vehicle_id,
        points=trajectory_points[-10:]  # Use last 10 points as input
    )
    
    config = PredictionConfig(
        prediction_horizon=5.0,
        time_step=0.1,
        models=["constant_velocity", "constant_acceleration"]
    )
    
    return TrajectoryRequest(
        trajectory=trajectory_input,
        config=config
    )


@pytest.fixture
def mock_model() -> TrajectoryPredictor:
    """Create a mock trajectory predictor for testing."""
    
    class MockPredictor(TrajectoryPredictor):
        def __init__(self):
            super().__init__()
            self.model_name = "mock_predictor"
            self.is_trained = True
        
        async def predict(self, trajectory: TrajectoryData, **kwargs) -> TrajectoryData:
            """Mock prediction that extrapolates the trajectory."""
            if not trajectory.positions:
                raise ValueError("Empty trajectory")
            
            # Simple extrapolation
            last_pos = trajectory.positions[-1]
            last_vel = trajectory.velocities[-1] if trajectory.velocities else Velocity(vx=1.0, vy=0.0)
            last_time = trajectory.timestamps[-1] if trajectory.timestamps else 0.0
            
            pred_positions = []
            pred_velocities = []
            pred_timestamps = []
            
            for i in range(10):  # Predict 10 steps
                dt = 0.1
                new_time = last_time + (i + 1) * dt
                new_x = last_pos.x + last_vel.vx * (i + 1) * dt
                new_y = last_pos.y + last_vel.vy * (i + 1) * dt
                
                pred_positions.append(Position(x=new_x, y=new_y))
                pred_velocities.append(last_vel)  # Constant velocity
                pred_timestamps.append(new_time)
            
            return TrajectoryData(
                trajectory_id=f"{trajectory.trajectory_id}_pred",
                vehicle_id=trajectory.vehicle_id,
                positions=pred_positions,
                velocities=pred_velocities,
                timestamps=pred_timestamps,
                metadata={"source": "mock_prediction", "base_trajectory": trajectory.trajectory_id}
            )
        
        async def train(self, trajectories: List[TrajectoryData], **kwargs) -> Dict[str, Any]:
            """Mock training that returns dummy metrics."""
            return {
                "training_samples": len(trajectories),
                "training_time": 1.0,
                "loss": 0.1
            }
        
        async def evaluate(self, trajectories: List[TrajectoryData], **kwargs) -> Dict[str, float]:
            """Mock evaluation that returns dummy metrics."""
            return {
                "rmse": 0.5,
                "mae": 0.3,
                "ade": 0.4,
                "fde": 0.6
            }
    
    return MockPredictor()


@pytest.fixture
def temp_directory():
    """Create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Generate sample configuration for tests."""
    return {
        "data": {
            "batch_size": 32,
            "sequence_length": 20,
            "prediction_horizon": 10,
            "features": ["position", "velocity", "acceleration"]
        },
        "model": {
            "type": "constant_velocity",
            "parameters": {
                "learning_rate": 0.001,
                "epochs": 100
            }
        },
        "evaluation": {
            "metrics": ["rmse", "mae", "ade", "fde"],
            "cross_validation_folds": 5
        },
        "api": {
            "host": "localhost",
            "port": 8000,
            "timeout": 30
        }
    }


@pytest.fixture
def performance_thresholds() -> Dict[str, Dict[str, float]]:
    """Define performance thresholds for testing."""
    return {
        "accuracy": {
            "rmse": {"excellent": 0.1, "good": 0.5, "acceptable": 1.0},
            "mae": {"excellent": 0.08, "good": 0.4, "acceptable": 0.8},
            "ade": {"excellent": 0.1, "good": 0.5, "acceptable": 1.0},
            "fde": {"excellent": 0.2, "good": 1.0, "acceptable": 2.0}
        },
        "performance": {
            "inference_time": {"excellent": 0.01, "good": 0.1, "acceptable": 1.0},  # seconds
            "memory_usage": {"excellent": 100, "good": 500, "acceptable": 1000},  # MB
            "throughput": {"excellent": 1000, "good": 100, "acceptable": 10}  # requests/sec
        }
    }


@pytest.fixture
def ml_metrics():
    """Create trajectory metrics calculator."""
    return TrajectoryMetrics()


# Property-based testing helpers
class TrajectoryGenerator:
    """Generator for property-based trajectory testing."""
    
    @staticmethod
    def generate_valid_trajectory(
        min_length: int = 5,
        max_length: int = 100,
        min_speed: float = 0.1,
        max_speed: float = 30.0
    ) -> TrajectoryData:
        """Generate a valid trajectory for property testing."""
        length = np.random.randint(min_length, max_length + 1)
        
        positions = []
        velocities = []
        timestamps = []
        
        # Start position and velocity
        x, y = 0.0, 0.0
        vx = np.random.uniform(-max_speed, max_speed)
        vy = np.random.uniform(-max_speed, max_speed)
        
        # Ensure minimum speed
        speed = np.sqrt(vx**2 + vy**2)
        if speed < min_speed:
            vx = vx * min_speed / speed if speed > 0 else min_speed
            vy = vy * min_speed / speed if speed > 0 else 0.0
        
        start_time = datetime.now().timestamp()
        
        for i in range(length):
            positions.append(Position(x=x, y=y))
            velocities.append(Velocity(vx=vx, vy=vy))
            timestamps.append(start_time + i * 0.1)
            
            # Update position
            x += vx * 0.1
            y += vy * 0.1
            
            # Add some random variation to velocity
            vx += np.random.normal(0, 0.1)
            vy += np.random.normal(0, 0.1)
            
            # Constrain speed
            speed = np.sqrt(vx**2 + vy**2)
            if speed > max_speed:
                vx = vx * max_speed / speed
                vy = vy * max_speed / speed
            elif speed < min_speed:
                vx = vx * min_speed / speed if speed > 0 else min_speed
                vy = vy * min_speed / speed if speed > 0 else 0.0
        
        return TrajectoryData(
            trajectory_id=f"generated_{np.random.randint(1000, 9999)}",
            vehicle_id=f"vehicle_{np.random.randint(100, 999)}",
            positions=positions,
            velocities=velocities,
            timestamps=timestamps,
            metadata={"source": "property_test"}
        )


@pytest.fixture
def trajectory_generator():
    """Provide trajectory generator for property-based tests."""
    return TrajectoryGenerator()


# Performance testing fixtures
@pytest.fixture
def performance_monitor():
    """Create performance monitoring context manager."""
    
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.memory_usage = {}
        
        def __enter__(self):
            import psutil
            import time
            
            self.start_time = time.time()
            process = psutil.Process()
            self.memory_usage['start'] = process.memory_info().rss / 1024 / 1024  # MB
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            import psutil
            import time
            
            self.end_time = time.time()
            process = psutil.Process()
            self.memory_usage['end'] = process.memory_info().rss / 1024 / 1024  # MB
        
        @property
        def elapsed_time(self) -> float:
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return 0.0
        
        @property
        def memory_delta(self) -> float:
            if 'start' in self.memory_usage and 'end' in self.memory_usage:
                return self.memory_usage['end'] - self.memory_usage['start']
            return 0.0
    
    return PerformanceMonitor


# Mock external services for integration testing
@pytest.fixture
def mock_api_server():
    """Create a mock API server for integration testing."""
    
    class MockAPIServer:
        def __init__(self):
            self.responses = {}
            self.request_count = 0
        
        def set_response(self, endpoint: str, response: Any):
            self.responses[endpoint] = response
        
        async def handle_request(self, endpoint: str, data: Any = None) -> Any:
            self.request_count += 1
            return self.responses.get(endpoint, {"error": "endpoint not found"})
        
        def reset(self):
            self.responses = {}
            self.request_count = 0
    
    return MockAPIServer()


# Test data validation helpers
def validate_trajectory_data(trajectory: TrajectoryData) -> bool:
    """Validate trajectory data structure."""
    if not trajectory.positions:
        return False
    
    if trajectory.velocities and len(trajectory.velocities) != len(trajectory.positions):
        return False
    
    if trajectory.timestamps and len(trajectory.timestamps) != len(trajectory.positions):
        return False
    
    # Check timestamp ordering
    if trajectory.timestamps and len(trajectory.timestamps) > 1:
        for i in range(1, len(trajectory.timestamps)):
            if trajectory.timestamps[i] <= trajectory.timestamps[i-1]:
                return False
    
    return True


def validate_prediction_response(response: TrajectoryResponse) -> bool:
    """Validate prediction response structure."""
    if not hasattr(response, 'predicted_trajectory'):
        return False
    
    if not response.predicted_trajectory.positions:
        return False
    
    if not (0.0 <= response.confidence <= 1.0):
        return False
    
    return validate_trajectory_data(response.predicted_trajectory)


# Pytest markers for test categorization
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "api: mark test as API test"
    )
    config.addinivalue_line(
        "markers", "model: mark test as model test"
    )
    config.addinivalue_line(
        "markers", "data: mark test as data test"
    )