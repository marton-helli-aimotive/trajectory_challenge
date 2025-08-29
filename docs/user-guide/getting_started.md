# Getting Started Guide

Welcome to the Trajectory Prediction System! This guide will help you get up and running quickly with trajectory prediction for autonomous vehicles.

## Prerequisites

- **Python**: 3.8 or higher
- **System Memory**: At least 4GB RAM
- **Storage**: 2GB free space for models and data
- **Operating System**: Linux, macOS, or Windows

## Installation

### Option 1: pip Install (Recommended)

```bash
# Install the package
pip install trajectory-prediction

# Verify installation
python -c "import trajectory_prediction; print('Installation successful!')"
```

### Option 2: From Source

```bash
# Clone the repository
git clone https://github.com/trajectory-prediction/trajectory-prediction.git
cd trajectory-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# Run tests to verify installation
python -m pytest tests/ -v
```

### Option 3: Docker

```bash
# Pull the Docker image
docker pull trajectory-prediction:latest

# Run the container
docker run -p 8000:8000 trajectory-prediction:latest

# The API will be available at http://localhost:8000
```

## Quick Start Examples

### 1. Basic Trajectory Prediction

```python
import numpy as np
from trajectory_prediction.data.schemas import TrajectoryData, Position, Velocity
from trajectory_prediction.models.baseline.constant_velocity import ConstantVelocityPredictor

# Create sample trajectory data
positions = [
    Position(x=0.0, y=0.0),
    Position(x=1.0, y=0.0),
    Position(x=2.0, y=0.0),
    Position(x=3.0, y=0.0)
]

velocities = [
    Velocity(vx=1.0, vy=0.0),
    Velocity(vx=1.0, vy=0.0),
    Velocity(vx=1.0, vy=0.0),
    Velocity(vx=1.0, vy=0.0)
]

timestamps = [0.0, 1.0, 2.0, 3.0]

# Create trajectory
trajectory = TrajectoryData(
    trajectory_id="example_001",
    vehicle_id="vehicle_001",
    positions=positions,
    velocities=velocities,
    timestamps=timestamps
)

# Create and use predictor
predictor = ConstantVelocityPredictor()
prediction = await predictor.predict(trajectory, prediction_horizon=5.0)

print(f"Predicted {len(prediction.positions)} future positions:")
for i, pos in enumerate(prediction.positions):
    print(f"  t={prediction.timestamps[i]:.1f}s: ({pos.x:.1f}, {pos.y:.1f})")
```

**Output:**
```
Predicted 5 future positions:
  t=4.0s: (4.0, 0.0)
  t=5.0s: (5.0, 0.0)
  t=6.0s: (6.0, 0.0)
  t=7.0s: (7.0, 0.0)
  t=8.0s: (8.0, 0.0)
```

### 2. Using the REST API

Start the API server:

```bash
# Start the server
python -m trajectory_prediction.api.server

# Server will start at http://localhost:8000
# API documentation at http://localhost:8000/docs
```

Make a prediction request:

```python
import requests

# Prepare trajectory data
trajectory_data = {
    "trajectory": {
        "trajectory_id": "api_example",
        "vehicle_id": "car_001",
        "points": [
            {"timestamp": 0.0, "x": 0.0, "y": 0.0, "vx": 15.0, "vy": 0.0},
            {"timestamp": 0.1, "x": 1.5, "y": 0.0, "vx": 15.0, "vy": 0.0},
            {"timestamp": 0.2, "x": 3.0, "y": 0.0, "vx": 15.0, "vy": 0.0}
        ]
    },
    "config": {
        "prediction_horizon": 2.0,
        "time_step": 0.1,
        "models": ["constant_velocity"]
    }
}

# Make prediction request
response = requests.post('http://localhost:8000/predict', json=trajectory_data)
result = response.json()

print(f"Prediction confidence: {result['confidence']:.2f}")
print(f"Number of predicted points: {len(result['predicted_trajectory']['positions'])}")
```

### 3. Interactive Dashboard

Launch the Streamlit dashboard:

```bash
# Start the dashboard
streamlit run src/trajectory_prediction/visualization/dashboard.py

# Dashboard will open at http://localhost:8501
```

The dashboard provides:
- **Real-time prediction visualization**
- **Model comparison interface**
- **Performance monitoring**
- **Interactive parameter tuning**

### 4. Batch Processing

```python
from trajectory_prediction.models.factory import ModelFactory
from trajectory_prediction.data.etl.pipeline import TrajectoryETLPipeline

# Load multiple models
models = {
    'cv': ModelFactory.create_model('constant_velocity'),
    'ca': ModelFactory.create_model('constant_acceleration'),
    'poly': ModelFactory.create_model('polynomial')
}

# Process multiple trajectories
async def batch_predict(trajectories, models):
    results = {}
    
    for model_name, model in models.items():
        predictions = []
        for trajectory in trajectories:
            pred = await model.predict(trajectory)
            predictions.append(pred)
        results[model_name] = predictions
    
    return results

# Run batch processing
trajectories = load_trajectory_data("data/sample_trajectories.json")
batch_results = await batch_predict(trajectories, models)

print(f"Processed {len(trajectories)} trajectories with {len(models)} models")
```

## Core Concepts

### 1. Trajectory Data Structure

```python
@dataclass
class TrajectoryData:
    trajectory_id: str          # Unique identifier
    vehicle_id: str            # Vehicle identifier
    positions: List[Position]   # (x, y) coordinates
    velocities: List[Velocity]  # (vx, vy) velocities
    timestamps: List[float]     # Time stamps
    metadata: Dict[str, Any]    # Additional information
```

### 2. Model Types

| Model Type | Use Case | Strengths | Limitations |
|------------|----------|-----------|-------------|
| **Constant Velocity** | Highway driving | Fast, interpretable | Assumes no acceleration |
| **Constant Acceleration** | City traffic | Handles acceleration | Constant acceleration assumption |
| **Polynomial** | Curved paths | Flexible trajectories | Can overfit |
| **K-Nearest Neighbors** | Complex patterns | Non-parametric | Requires large dataset |
| **Gaussian Process** | Uncertainty critical | Principled uncertainty | Computational complexity |

### 3. Prediction Configuration

```python
@dataclass
class PredictionConfig:
    prediction_horizon: float = 5.0    # How far to predict (seconds)
    time_step: float = 0.1             # Time resolution (seconds)
    models: List[str] = ["constant_velocity"]  # Models to use
    include_uncertainty: bool = True    # Include uncertainty estimates
    max_speed: float = 50.0            # Physical constraints (m/s)
    max_acceleration: float = 10.0     # Physical constraints (m/s²)
```

## Common Use Cases

### 1. Real-Time Prediction Service

```python
from trajectory_prediction.api.client import TrajectoryPredictionClient

# Initialize client
client = TrajectoryPredictionClient("http://localhost:8000")

# Real-time prediction loop
async def real_time_prediction():
    while True:
        # Get current vehicle state
        current_trajectory = get_vehicle_trajectory()
        
        # Predict future trajectory
        prediction = await client.predict(current_trajectory)
        
        # Use prediction for planning/control
        plan_vehicle_path(prediction)
        
        # Wait for next cycle
        await asyncio.sleep(0.1)  # 10 Hz
```

### 2. Offline Batch Analysis

```python
import pandas as pd
from trajectory_prediction.evaluation.evaluator import ModelEvaluator

# Load historical data
trajectories = load_historical_trajectories("data/ngsim_dataset.parquet")

# Split into train/test
train_data = trajectories[:8000]
test_data = trajectories[8000:]

# Train multiple models
models = ['constant_velocity', 'polynomial', 'gaussian_process']
trained_models = {}

for model_name in models:
    model = ModelFactory.create_model(model_name)
    await model.train(train_data)
    trained_models[model_name] = model

# Evaluate models
evaluator = ModelEvaluator()
results = await evaluator.compare_models(trained_models, test_data)

print("Model Comparison Results:")
for model_name, metrics in results.items():
    print(f"  {model_name}: RMSE={metrics['rmse']:.3f}, MAE={metrics['mae']:.3f}")
```

### 3. Custom Model Development

```python
from trajectory_prediction.models.base import TrajectoryPredictor

class CustomPredictor(TrajectoryPredictor):
    def __init__(self, custom_parameter: float = 1.0):
        super().__init__()
        self.model_name = "custom_predictor"
        self.custom_parameter = custom_parameter
    
    async def predict(self, trajectory: TrajectoryData, **kwargs) -> TrajectoryData:
        # Implement custom prediction logic
        prediction_horizon = kwargs.get('prediction_horizon', 5.0)
        time_step = kwargs.get('time_step', 0.1)
        
        # Your custom prediction algorithm here
        predicted_positions = self.my_prediction_algorithm(
            trajectory, prediction_horizon, time_step
        )
        
        # Return prediction in standard format
        return TrajectoryData(
            trajectory_id=f"{trajectory.trajectory_id}_pred",
            vehicle_id=trajectory.vehicle_id,
            positions=predicted_positions,
            velocities=predicted_velocities,
            timestamps=predicted_timestamps
        )
    
    def my_prediction_algorithm(self, trajectory, horizon, dt):
        # Your implementation here
        pass

# Register and use custom model
ModelFactory.register("custom_predictor", CustomPredictor)
model = ModelFactory.create_model("custom_predictor", custom_parameter=2.0)
```

## Configuration

### Environment Variables

```bash
# API Configuration
export TRAJECTORY_API_HOST=0.0.0.0
export TRAJECTORY_API_PORT=8000
export TRAJECTORY_API_WORKERS=4

# Model Configuration  
export TRAJECTORY_MODEL_CACHE_SIZE=100
export TRAJECTORY_PREDICTION_TIMEOUT=30

# Data Configuration
export TRAJECTORY_DATA_PATH="/data/trajectories"
export TRAJECTORY_MODEL_PATH="/models"

# Monitoring
export TRAJECTORY_LOG_LEVEL=INFO
export TRAJECTORY_METRICS_ENABLED=true
```

### Configuration File

Create `config.yaml`:

```yaml
api:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  timeout: 30
  cors_origins: ["*"]

models:
  cache_size: 100
  default_horizon: 5.0
  default_time_step: 0.1
  available_models:
    - constant_velocity
    - constant_acceleration
    - polynomial
    - knn
    - gaussian_process

data:
  data_path: "/data/trajectories"
  model_path: "/models"
  batch_size: 32
  max_trajectory_length: 200

monitoring:
  log_level: "INFO"
  metrics_enabled: true
  health_check_interval: 30
  performance_logging: true

caching:
  redis_url: "redis://localhost:6379"
  cache_ttl: 300
  max_cache_size: "1GB"
```

Load configuration:

```python
from trajectory_prediction.config import load_config

config = load_config("config.yaml")
```

## Next Steps

### Learn More
- **[Model Documentation](../models/)**: Deep dive into prediction models
- **[API Reference](../api/)**: Complete API documentation  
- **[Tutorials](../tutorials/)**: Step-by-step tutorials and examples
- **[Deployment Guide](../deployment/)**: Production deployment options

### Try Examples
- **Basic Prediction**: `examples/basic_prediction.py`
- **Model Comparison**: `examples/model_comparison.py`
- **Real-time Streaming**: `examples/streaming_prediction.py`
- **Custom Models**: `examples/custom_model.py`

### Get Help
- **Documentation**: Browse the complete documentation
- **Issues**: Report bugs or request features on GitHub
- **Discussions**: Join the community discussions
- **Support**: Contact support for enterprise needs

## Troubleshooting

### Common Issues

#### Installation Problems
```bash
# Update pip and setuptools
pip install --upgrade pip setuptools

# Install with verbose output
pip install -v trajectory-prediction

# Force reinstall
pip install --force-reinstall trajectory-prediction
```

#### Import Errors
```python
# Check installation
python -c "import trajectory_prediction; print(trajectory_prediction.__version__)"

# Check dependencies
pip list | grep -E "(numpy|pandas|fastapi|streamlit)"

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

#### API Connection Issues
```bash
# Check if server is running
curl http://localhost:8000/health

# Check port availability
netstat -an | grep 8000

# Run server with debug logging
TRAJECTORY_LOG_LEVEL=DEBUG python -m trajectory_prediction.api.server
```

#### Memory Issues
```python
# Monitor memory usage
import psutil
print(f"Memory usage: {psutil.virtual_memory().percent}%")

# Reduce batch size
config.batch_size = 16  # Instead of 32

# Enable garbage collection
import gc
gc.collect()
```

### Performance Tips

1. **Use appropriate models**: Start with baseline models for simple scenarios
2. **Enable caching**: Configure Redis for production deployments  
3. **Batch requests**: Group multiple predictions for better throughput
4. **Monitor resources**: Track CPU, memory, and response times
5. **Optimize data**: Use efficient data formats (Parquet, Arrow)

Happy predicting! 🚗💨