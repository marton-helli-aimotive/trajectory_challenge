# Common Issues and Solutions

This guide provides solutions to frequently encountered issues when working with the Trajectory Prediction System.

## Installation and Setup Issues

### 1. Package Installation Failures

#### Problem: pip install fails with compilation errors
```bash
ERROR: Failed building wheel for some-package
ERROR: Could not build wheels for some-package which use PEP 517
```

**Solution:**
```bash
# Update pip and setuptools
pip install --upgrade pip setuptools wheel

# Install with verbose output to see specific error
pip install -v trajectory-prediction

# Install system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install build-essential python3-dev

# Install system dependencies (CentOS/RHEL)
sudo yum groupinstall "Development Tools"
sudo yum install python3-devel

# Force reinstall with no cache
pip install --no-cache-dir --force-reinstall trajectory-prediction
```

#### Problem: Import errors after installation
```python
ImportError: No module named 'trajectory_prediction'
```

**Solution:**
```bash
# Check if package is installed
pip list | grep trajectory

# Install in development mode
pip install -e .

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"

# Verify installation
python -c "import trajectory_prediction; print(trajectory_prediction.__version__)"

# If using virtual environment, ensure it's activated
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows
```

#### Problem: Dependency conflicts
```bash
ERROR: pip's dependency resolver does not currently consider all the packages that are installed
```

**Solution:**
```bash
# Check for conflicts
pip check

# Create fresh virtual environment
deactivate  # if in virtual env
rm -rf venv/
python -m venv venv
source venv/bin/activate
pip install trajectory-prediction

# Use pip-tools for dependency resolution
pip install pip-tools
pip-compile requirements.in
pip install -r requirements.txt
```

### 2. Data Loading Issues

#### Problem: File not found errors
```python
FileNotFoundError: [Errno 2] No such file or directory: 'data/trajectories.parquet'
```

**Solution:**
```bash
# Check current working directory
pwd

# Verify file exists
ls -la data/

# Use absolute paths
python -c "
import os
print('Current directory:', os.getcwd())
print('Data files:', os.listdir('data/') if os.path.exists('data/') else 'data/ not found')
"

# Create data directory if missing
mkdir -p data/samples
mkdir -p data/models
mkdir -p data/cache

# Download sample data
python scripts/download_sample_data.py
```

#### Problem: Data format/schema errors
```python
ValidationError: 2 validation errors for TrajectoryData
```

**Solution:**
```python
# Check data schema
from trajectory_prediction.data.schemas import TrajectoryData
import pandas as pd

# Load and inspect data
df = pd.read_parquet('data/trajectories.parquet')
print(df.head())
print(df.dtypes)
print(df.columns)

# Validate specific trajectory
try:
    trajectory_data = TrajectoryData(
        trajectory_id="test_001",
        vehicle_id="vehicle_001",
        positions=[{"x": 0.0, "y": 0.0}],  # This should be Position objects
        velocities=[{"vx": 1.0, "vy": 0.0}],  # This should be Velocity objects
        timestamps=[0.0]
    )
except Exception as e:
    print(f"Validation error: {e}")

# Correct format
from trajectory_prediction.data.schemas import Position, Velocity

trajectory_data = TrajectoryData(
    trajectory_id="test_001",
    vehicle_id="vehicle_001",
    positions=[Position(x=0.0, y=0.0)],
    velocities=[Velocity(vx=1.0, vy=0.0)],
    timestamps=[0.0]
)
```

## API Server Issues

### 3. Server Startup Problems

#### Problem: Port already in use
```bash
OSError: [Errno 98] Address already in use
```

**Solution:**
```bash
# Find process using port 8000
lsof -i :8000
# or
netstat -tulpn | grep :8000

# Kill process (replace PID with actual PID)
kill -9 <PID>

# Use different port
export TRAJECTORY_API_PORT=8001
python -m trajectory_prediction.api.server

# Or specify port in command
python -m trajectory_prediction.api.server --port 8001
```

#### Problem: Permission denied on port 80 or 443
```bash
PermissionError: [Errno 13] Permission denied
```

**Solution:**
```bash
# Use non-privileged port (1024+)
python -m trajectory_prediction.api.server --port 8000

# Or run with sudo (not recommended for development)
sudo python -m trajectory_prediction.api.server --port 80

# Better: Use reverse proxy
# Start API on port 8000, configure nginx/apache to proxy
```

#### Problem: Server starts but endpoints return 500 errors
```bash
curl http://localhost:8000/health
# Returns: 500 Internal Server Error
```

**Solution:**
```bash
# Check logs
python -m trajectory_prediction.api.server --log-level DEBUG

# Enable debug mode
export TRAJECTORY_DEBUG=true
python -m trajectory_prediction.api.server

# Test specific components
python -c "
from trajectory_prediction.api.server import create_app
app = create_app()
print('App created successfully')
"

# Check dependencies
python -c "
import fastapi
import uvicorn
import trajectory_prediction
print('All imports successful')
"
```

### 4. API Request Issues

#### Problem: 422 Validation Error on prediction requests
```json
{
  "detail": [
    {
      "loc": ["body", "trajectory", "points"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

**Solution:**
```bash
# Check request format
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "trajectory": {
      "trajectory_id": "test_001",
      "vehicle_id": "vehicle_001",
      "points": [
        {"timestamp": 0.0, "x": 0.0, "y": 0.0, "vx": 10.0, "vy": 0.0},
        {"timestamp": 0.1, "x": 1.0, "y": 0.0, "vx": 10.0, "vy": 0.0}
      ]
    },
    "config": {
      "prediction_horizon": 2.0,
      "models": ["constant_velocity"]
    }
  }'

# Use Python client
python -c "
import requests
response = requests.post('http://localhost:8000/predict', json={
    'trajectory': {
        'trajectory_id': 'test_001',
        'vehicle_id': 'vehicle_001',
        'points': [
            {'timestamp': 0.0, 'x': 0.0, 'y': 0.0, 'vx': 10.0, 'vy': 0.0},
            {'timestamp': 0.1, 'x': 1.0, 'y': 0.0, 'vx': 10.0, 'vy': 0.0}
        ]
    },
    'config': {
        'prediction_horizon': 2.0,
        'models': ['constant_velocity']
    }
})
print(f'Status: {response.status_code}')
print(f'Response: {response.json()}')
"
```

#### Problem: Slow API responses or timeouts
```bash
# Request takes > 30 seconds or times out
```

**Solution:**
```bash
# Check server performance
curl -w "@curl-format.txt" http://localhost:8000/health

# Create curl-format.txt
cat > curl-format.txt << EOF
     time_namelookup:  %{time_namelookup}\n
        time_connect:  %{time_connect}\n
     time_appconnect:  %{time_appconnect}\n
    time_pretransfer:  %{time_pretransfer}\n
       time_redirect:  %{time_redirect}\n
  time_starttransfer:  %{time_starttransfer}\n
                     ----------\n
          time_total:  %{time_total}\n
EOF

# Enable performance monitoring
export TRAJECTORY_PERFORMANCE_LOGGING=true
python -m trajectory_prediction.api.server

# Check resource usage
top -p $(pgrep -f "python.*trajectory")
```

## Model Training and Prediction Issues

### 5. Model Training Problems

#### Problem: Training fails with insufficient data
```python
ValueError: Not enough data points for training
```

**Solution:**
```python
# Check data size
import pandas as pd
df = pd.read_parquet('data/training_data.parquet')
print(f"Training samples: {len(df)}")
print(f"Unique trajectories: {df['trajectory_id'].nunique()}")

# Generate synthetic data if needed
from trajectory_prediction.data.generators import SyntheticDataGenerator

generator = SyntheticDataGenerator()
synthetic_trajectories = generator.generate_trajectories(
    num_trajectories=1000,
    scenario="highway"
)

# Combine with existing data
all_trajectories = existing_trajectories + synthetic_trajectories
```

#### Problem: Training takes too long or runs out of memory
```python
MemoryError: Unable to allocate array with shape (1000000, 100)
```

**Solution:**
```python
# Reduce batch size
from trajectory_prediction.config import TrainingConfig

config = TrainingConfig(
    batch_size=16,  # Reduce from default 32
    max_trajectory_length=100,  # Limit trajectory length
    num_workers=2  # Reduce parallel workers
)

# Use data streaming
from trajectory_prediction.data.loaders import StreamingDataLoader

loader = StreamingDataLoader(
    data_path='data/large_dataset.parquet',
    batch_size=16,
    streaming=True
)

# Monitor memory usage
import psutil
print(f"Memory usage: {psutil.virtual_memory().percent}%")

# Enable garbage collection
import gc
gc.collect()
```

### 6. Model Prediction Issues

#### Problem: Models return NaN or infinite values
```python
RuntimeWarning: invalid value encountered in prediction
```

**Solution:**
```python
# Check input data for NaN values
import numpy as np
trajectory_data = load_trajectory()

positions = np.array([[p.x, p.y] for p in trajectory_data.positions])
velocities = np.array([[v.vx, v.vy] for v in trajectory_data.velocities])
timestamps = np.array(trajectory_data.timestamps)

print(f"NaN in positions: {np.isnan(positions).any()}")
print(f"NaN in velocities: {np.isnan(velocities).any()}")
print(f"NaN in timestamps: {np.isnan(timestamps).any()}")

# Clean data
def clean_trajectory_data(trajectory):
    # Remove NaN values
    valid_indices = ~(
        np.isnan([p.x for p in trajectory.positions]) |
        np.isnan([p.y for p in trajectory.positions]) |
        np.isnan([v.vx for v in trajectory.velocities]) |
        np.isnan([v.vy for v in trajectory.velocities])
    )
    
    trajectory.positions = [p for i, p in enumerate(trajectory.positions) if valid_indices[i]]
    trajectory.velocities = [v for i, v in enumerate(trajectory.velocities) if valid_indices[i]]
    trajectory.timestamps = [t for i, t in enumerate(trajectory.timestamps) if valid_indices[i]]
    
    return trajectory

# Enable debug mode for detailed error info
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### Problem: Poor prediction accuracy
```python
# RMSE > 10.0, predictions don't match expected behavior
```

**Solution:**
```python
# Check model configuration
from trajectory_prediction.models.factory import ModelFactory

model = ModelFactory.create_model('constant_velocity')
print(f"Model config: {model.config}")

# Validate against simple cases
import matplotlib.pyplot as plt

# Create simple test case
test_trajectory = create_straight_line_trajectory(
    start_pos=(0, 0),
    velocity=(10, 0),  # 10 m/s in x direction
    duration=2.0
)

prediction = model.predict(test_trajectory, prediction_horizon=3.0)

# Plot for visual inspection
plt.figure(figsize=(10, 6))
plt.plot([p.x for p in test_trajectory.positions], 
         [p.y for p in test_trajectory.positions], 'bo-', label='Input')
plt.plot([p.x for p in prediction.positions], 
         [p.y for p in prediction.positions], 'ro--', label='Prediction')
plt.legend()
plt.title('Trajectory Prediction Validation')
plt.show()

# Check against physics
expected_final_x = test_trajectory.positions[-1].x + 10.0 * 3.0  # 30m forward
actual_final_x = prediction.positions[-1].x
print(f"Expected final x: {expected_final_x}")
print(f"Actual final x: {actual_final_x}")
print(f"Error: {abs(expected_final_x - actual_final_x)}")
```

## Dashboard and Visualization Issues

### 7. Dashboard Problems

#### Problem: Streamlit dashboard won't start
```bash
FileNotFoundError: [Errno 2] No such file or directory: 'dashboard.py'
```

**Solution:**
```bash
# Check file location
find . -name "dashboard.py" -type f

# Run from correct directory
cd src/trajectory_prediction/visualization/
streamlit run dashboard.py

# Or use full path
streamlit run src/trajectory_prediction/visualization/dashboard.py

# Check Streamlit installation
pip list | grep streamlit
pip install streamlit
```

#### Problem: Dashboard shows connection errors to API
```bash
ConnectionError: Failed to connect to API server
```

**Solution:**
```python
# Check API server status
import requests
try:
    response = requests.get('http://localhost:8000/health')
    print(f"API Status: {response.status_code}")
except Exception as e:
    print(f"API Connection Error: {e}")

# Update API URL in dashboard
# In dashboard.py, check:
API_BASE_URL = "http://localhost:8000"  # Ensure this matches your API server

# Or set environment variable
export TRAJECTORY_API_URL=http://localhost:8000
streamlit run dashboard.py
```

#### Problem: Plots not displaying or show errors
```python
AttributeError: module 'plotly' has no attribute 'graph_objects'
```

**Solution:**
```bash
# Install required plotting libraries
pip install plotly matplotlib seaborn

# Check versions
python -c "
import plotly
import matplotlib
import seaborn as sns
print(f'Plotly: {plotly.__version__}')
print(f'Matplotlib: {matplotlib.__version__}')
print(f'Seaborn: {sns.__version__}')
"

# Clear Streamlit cache
streamlit cache clear
```

## Performance Issues

### 8. Slow Performance

#### Problem: API responses are slow (>1000ms)
```bash
# API taking 2-5 seconds per prediction
```

**Solution:**
```python
# Enable caching
from trajectory_prediction.api.cache import PredictionCache

cache = PredictionCache(backend='memory', max_size=1000)

# Use async/await properly
import asyncio

async def make_prediction(trajectory):
    model = ModelFactory.create_model('constant_velocity')
    return await model.predict(trajectory)

# Profile bottlenecks
import cProfile
import pstats

cProfile.run('make_prediction(trajectory)', 'profile_stats')
stats = pstats.Stats('profile_stats')
stats.sort_stats('cumulative').print_stats(10)

# Monitor system resources
import psutil
print(f"CPU: {psutil.cpu_percent()}%")
print(f"Memory: {psutil.virtual_memory().percent}%")
```

#### Problem: High memory usage
```bash
# Memory usage grows over time, eventually crashes
```

**Solution:**
```python
# Enable garbage collection
import gc
gc.enable()
gc.set_threshold(700, 10, 10)

# Check for memory leaks
import tracemalloc

tracemalloc.start()
# ... run your code ...
current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage: {current / 1024 / 1024:.2f} MB")
print(f"Peak memory usage: {peak / 1024 / 1024:.2f} MB")
tracemalloc.stop()

# Limit cache sizes
cache_config = {
    'model_cache_size': 10,  # Reduce from 100
    'prediction_cache_size': 1000  # Reduce from 10000
}

# Clear caches periodically
def clear_caches():
    model_cache.clear()
    prediction_cache.clear()
    gc.collect()
```

## Testing Issues

### 9. Test Failures

#### Problem: Tests fail in CI but pass locally
```bash
# Tests pass on local machine but fail in GitHub Actions
```

**Solution:**
```bash
# Check Python version consistency
python --version

# Pin dependency versions
pip freeze > requirements-test.txt

# Use same test environment locally
python -m venv test-env
source test-env/bin/activate
pip install -r requirements-test.txt
python -m pytest tests/

# Check for race conditions in async tests
pytest tests/ -v --tb=short --disable-warnings

# Use deterministic random seeds
pytest tests/ --randomly-seed=42
```

#### Problem: Flaky tests that randomly fail
```python
# Test passes sometimes, fails other times
```

**Solution:**
```python
# Fix random seeds in tests
import numpy as np
import random

def test_model_prediction():
    np.random.seed(42)
    random.seed(42)
    # ... test code ...

# Add proper cleanup
def test_with_cleanup():
    model = create_model()
    try:
        result = model.predict(data)
        assert result is not None
    finally:
        model.cleanup()

# Use fixtures for consistent test data
@pytest.fixture
def sample_trajectory():
    return TrajectoryData(
        trajectory_id="test_001",
        vehicle_id="vehicle_001",
        positions=[Position(x=0.0, y=0.0), Position(x=1.0, y=0.0)],
        velocities=[Velocity(vx=1.0, vy=0.0), Velocity(vx=1.0, vy=0.0)],
        timestamps=[0.0, 1.0]
    )
```

## Docker and Deployment Issues

### 10. Container Problems

#### Problem: Docker build failures
```bash
ERROR: failed to solve: process "/bin/sh -c pip install -r requirements.txt" did not complete successfully
```

**Solution:**
```bash
# Check Docker version
docker --version

# Build with verbose output
docker build --progress=plain --no-cache .

# Check Dockerfile syntax
docker build --dry-run .

# Test individual layers
docker run -it python:3.9-slim /bin/bash
# Then run each command manually

# Check disk space
docker system df
docker system prune -f
```

#### Problem: Container starts but services are unhealthy
```bash
# Health checks failing
```

**Solution:**
```bash
# Check container logs
docker logs container_name

# Exec into container
docker exec -it container_name /bin/bash

# Test health check manually
docker exec container_name curl -f http://localhost:8000/health

# Check port bindings
docker port container_name

# Verify environment variables
docker exec container_name env | grep TRAJECTORY
```

## Getting Help

### 11. Debug Information Collection

When reporting issues, collect this information:

```bash
#!/bin/bash
# debug_info.sh - Collect system information

echo "=== System Information ==="
uname -a
python --version
pip --version

echo "=== Package Information ==="
pip list | grep -E "(trajectory|numpy|pandas|fastapi|streamlit)"

echo "=== Environment Variables ==="
env | grep TRAJECTORY

echo "=== Log Files ==="
tail -n 50 logs/trajectory_prediction.log

echo "=== Resource Usage ==="
df -h
free -h
ps aux | grep python

echo "=== Network Status ==="
netstat -tulpn | grep -E "(8000|8501|6379)"
```

Run and include output when seeking help:
```bash
chmod +x debug_info.sh
./debug_info.sh > debug_info.txt
```

### 12. Support Channels

- **Documentation**: Check the complete [documentation](../README.md)
- **GitHub Issues**: Report bugs at [GitHub Issues](https://github.com/trajectory-prediction/issues)
- **Discussions**: Community help at [GitHub Discussions](https://github.com/trajectory-prediction/discussions)
- **Stack Overflow**: Use tags `trajectory-prediction` and `autonomous-vehicles`

This troubleshooting guide covers the most common issues encountered when working with the Trajectory Prediction System. For issues not covered here, please check the other documentation sections or reach out through the support channels.