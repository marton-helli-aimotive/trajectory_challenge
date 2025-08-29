# Local Development Setup

This guide walks through setting up the Trajectory Prediction System for local development, testing, and experimentation.

## Prerequisites

### System Requirements
- **Python**: 3.8 or higher (3.9+ recommended)
- **Memory**: Minimum 4GB RAM (8GB+ recommended)
- **Storage**: 2GB free space for dependencies and data
- **OS**: Linux, macOS, or Windows (WSL recommended for Windows)

### Development Tools
- **Git**: For version control
- **Docker**: Optional, for containerized development
- **IDE**: VS Code, PyCharm, or similar with Python support

## Installation Methods

### Method 1: Standard Installation (Recommended)

#### 1. Clone the Repository
```bash
# Clone the repository
git clone https://github.com/trajectory-prediction/trajectory-prediction.git
cd trajectory-prediction

# Verify repository structure
ls -la
```

#### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Verify activation
which python  # Should show path within venv directory
```

#### 3. Install Dependencies
```bash
# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install package in development mode
pip install -e .

# Install development dependencies
pip install -e ".[dev,test,docs]"

# Verify installation
python -c "import trajectory_prediction; print(trajectory_prediction.__version__)"
```

#### 4. Install Pre-commit Hooks
```bash
# Install pre-commit hooks
pre-commit install

# Test hooks (optional)
pre-commit run --all-files
```

#### 5. Download Sample Data
```bash
# Create data directory
mkdir -p data/samples

# Download sample NGSIM data (if available)
python scripts/download_sample_data.py

# Or use synthetic data for testing
python scripts/generate_synthetic_data.py --output data/samples/synthetic.parquet --num-trajectories 1000
```

### Method 2: Docker Development Environment

#### 1. Build Development Container
```bash
# Build development image
docker build -f docker/Dockerfile.dev -t trajectory-prediction:dev .

# Run development container
docker run -it \
  -v $(pwd):/workspace \
  -p 8000:8000 \
  -p 8501:8501 \
  --name trajectory-dev \
  trajectory-prediction:dev
```

#### 2. Container Development Workflow
```bash
# Start existing container
docker start -i trajectory-dev

# Run commands inside container
docker exec -it trajectory-dev python -m pytest tests/

# Stop container
docker stop trajectory-dev
```

### Method 3: Poetry Installation (Alternative)

If you prefer Poetry for dependency management:

```bash
# Install Poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies with Poetry
poetry install

# Activate Poetry shell
poetry shell

# Run commands
poetry run python -m trajectory_prediction.api.server
```

## Configuration

### 1. Environment Variables
Create a `.env` file in the project root:

```bash
# Copy example environment file
cp .env.example .env

# Edit configuration
nano .env
```

Example `.env` file:
```bash
# API Configuration
TRAJECTORY_API_HOST=localhost
TRAJECTORY_API_PORT=8000
TRAJECTORY_API_WORKERS=1
TRAJECTORY_API_DEBUG=true

# Data Configuration
TRAJECTORY_DATA_PATH=./data
TRAJECTORY_MODEL_PATH=./models
TRAJECTORY_CACHE_PATH=./cache

# Database Configuration (optional)
TRAJECTORY_DB_URL=sqlite:///./data/trajectories.db

# Redis Configuration (optional)
TRAJECTORY_REDIS_URL=redis://localhost:6379

# MLflow Configuration
MLFLOW_TRACKING_URI=./mlruns
MLFLOW_EXPERIMENT_NAME=trajectory_prediction

# Logging Configuration
TRAJECTORY_LOG_LEVEL=INFO
TRAJECTORY_LOG_FILE=./logs/trajectory_prediction.log

# Development Configuration
TRAJECTORY_DEBUG=true
TRAJECTORY_HOT_RELOAD=true
```

### 2. Configuration File
Create `config/local.yaml`:

```yaml
# Local development configuration
api:
  host: "localhost"
  port: 8000
  debug: true
  hot_reload: true
  workers: 1

models:
  cache_size: 10
  default_horizon: 5.0
  default_time_step: 0.1
  available_models:
    - constant_velocity
    - constant_acceleration
    - polynomial
    - knn
    - gaussian_process

data:
  data_path: "./data"
  model_path: "./models"
  cache_path: "./cache"
  batch_size: 16
  max_trajectory_length: 200

monitoring:
  log_level: "DEBUG"
  metrics_enabled: true
  performance_logging: true

development:
  auto_reload: true
  debug_mode: true
  test_mode: false
```

## Verification

### 1. Run Unit Tests
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/unit/ -v                    # Unit tests
python -m pytest tests/integration/ -v             # Integration tests
python -m pytest tests/performance/ -v             # Performance tests

# Run with coverage
python -m pytest tests/ --cov=trajectory_prediction --cov-report=html

# View coverage report
open htmlcov/index.html
```

### 2. Test API Server
```bash
# Start API server
python -m trajectory_prediction.api.server

# In another terminal, test endpoints
curl http://localhost:8000/health

# Test prediction endpoint
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
```

### 3. Launch Dashboard
```bash
# Start Streamlit dashboard
streamlit run src/trajectory_prediction/visualization/dashboard.py

# Access dashboard at http://localhost:8501
```

### 4. Run Example Scripts
```bash
# Basic prediction example
python examples/basic_prediction.py

# Model comparison example
python examples/model_comparison.py

# Streaming prediction example
python examples/streaming_prediction.py
```

## Development Workflow

### 1. Code Quality Tools

#### Linting and Formatting
```bash
# Run linting
ruff check src/ tests/

# Auto-fix linting issues
ruff check --fix src/ tests/

# Format code
black src/ tests/

# Check imports
isort --check-only src/ tests/

# Fix import order
isort src/ tests/
```

#### Type Checking
```bash
# Run type checking
mypy src/trajectory_prediction/

# Run with strict mode
mypy --strict src/trajectory_prediction/models/
```

#### Security Scanning
```bash
# Scan for security issues
bandit -r src/trajectory_prediction/

# Check for known vulnerabilities
safety check

# Audit dependencies
pip-audit
```

### 2. Testing Strategy

#### Unit Tests
```bash
# Run specific model tests
python -m pytest tests/unit/test_models.py -v

# Run with debugging
python -m pytest tests/unit/test_models.py::TestConstantVelocityPredictor::test_basic_prediction -vv -s

# Run performance tests
python -m pytest tests/performance/ -v --benchmark-only
```

#### Integration Tests
```bash
# Run API integration tests
python -m pytest tests/integration/test_api_integration.py -v

# Run end-to-end pipeline tests
python -m pytest tests/integration/test_end_to_end_pipeline.py -v
```

#### Property-based Testing
```bash
# Run property-based tests with Hypothesis
python -m pytest tests/unit/test_data_schemas.py::test_trajectory_properties -v

# Run with more examples
python -m pytest tests/unit/test_data_schemas.py::test_trajectory_properties -v --hypothesis-max-examples=1000
```

### 3. Debugging

#### Logging Configuration
```python
# Enable debug logging in development
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

#### Profiling
```bash
# Profile model inference
python -m cProfile -o profile_output.prof examples/basic_prediction.py

# View profiling results
python -m pstats profile_output.prof

# Memory profiling with memory_profiler
pip install memory_profiler
python -m memory_profiler examples/basic_prediction.py
```

#### Interactive Debugging
```python
# Add breakpoint in code
import pdb; pdb.set_trace()

# Or use modern debugger
import ipdb; ipdb.set_trace()

# Run with debugger
python -m pdb examples/basic_prediction.py
```

### 4. Data Management

#### Synthetic Data Generation
```bash
# Generate test trajectories
python scripts/generate_synthetic_data.py \
  --num-trajectories 5000 \
  --scenario highway \
  --output data/test/highway_5k.parquet

python scripts/generate_synthetic_data.py \
  --num-trajectories 3000 \
  --scenario urban \
  --output data/test/urban_3k.parquet
```

#### Data Validation
```bash
# Validate data quality
python scripts/validate_data.py data/samples/

# Check data schema
python -c "
from trajectory_prediction.data.schemas import TrajectoryData
from trajectory_prediction.data.etl.extractors import ParquetExtractor

extractor = ParquetExtractor()
trajectories = extractor.extract('data/samples/synthetic.parquet')
print(f'Loaded {len(trajectories)} trajectories')
print(f'First trajectory: {trajectories[0]}')
"
```

### 5. Model Development

#### Training Models Locally
```bash
# Train baseline models
python scripts/train_models.py \
  --config config/local.yaml \
  --data data/samples/synthetic.parquet \
  --models constant_velocity constant_acceleration

# Train advanced models
python scripts/train_models.py \
  --config config/local.yaml \
  --data data/samples/ \
  --models polynomial knn gaussian_process \
  --validate
```

#### Model Evaluation
```bash
# Evaluate model performance
python scripts/evaluate_models.py \
  --models-dir models/ \
  --test-data data/test/ \
  --output results/evaluation_report.html

# Compare models
python scripts/compare_models.py \
  --baseline-models constant_velocity constant_acceleration \
  --advanced-models polynomial knn \
  --test-data data/test/
```

### 6. Experiment Tracking

#### MLflow Setup
```bash
# Start MLflow UI
mlflow ui --backend-store-uri ./mlruns --port 5000

# Access MLflow at http://localhost:5000
```

#### Experiment Logging
```python
# Example experiment
import mlflow
from trajectory_prediction.models.baseline.constant_velocity import ConstantVelocityPredictor

with mlflow.start_run(experiment_name="local_development"):
    model = ConstantVelocityPredictor(noise_std=0.1)
    
    # Log parameters
    mlflow.log_param("model_type", "constant_velocity")
    mlflow.log_param("noise_std", 0.1)
    
    # Train and evaluate
    metrics = evaluate_model(model, test_data)
    
    # Log metrics
    mlflow.log_metrics(metrics)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
```

## Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Check Python path
python -c "import sys; print('\n'.join(sys.path))"

# Reinstall in development mode
pip uninstall trajectory-prediction
pip install -e .
```

#### 2. Port Already in Use
```bash
# Find process using port 8000
lsof -i :8000

# Kill process (replace PID)
kill -9 <PID>

# Or use different port
TRAJECTORY_API_PORT=8001 python -m trajectory_prediction.api.server
```

#### 3. Memory Issues
```bash
# Monitor memory usage
top -p $(pgrep -f "python.*trajectory")

# Reduce batch size
export TRAJECTORY_BATCH_SIZE=8

# Use memory profiling
python -m memory_profiler examples/basic_prediction.py
```

#### 4. Permission Issues
```bash
# Fix data directory permissions
chmod -R 755 data/
chmod -R 755 models/
chmod -R 755 cache/

# Create necessary directories
mkdir -p data models cache logs
```

#### 5. Dependency Conflicts
```bash
# Check for conflicts
pip check

# Create fresh environment
deactivate
rm -rf venv/
python -m venv venv
source venv/bin/activate
pip install -e .
```

### Performance Optimization

#### 1. Enable JIT Compilation
```bash
# Install Numba for JIT compilation
pip install numba

# Set environment variables
export NUMBA_CACHE_DIR=./cache/numba
export NUMBA_NUM_THREADS=4
```

#### 2. Optimize Data Loading
```bash
# Use memory mapping for large files
export TRAJECTORY_USE_MEMORY_MAP=true

# Enable parallel data loading
export TRAJECTORY_NUM_WORKERS=4
```

#### 3. Enable Caching
```bash
# Install Redis for caching
pip install redis

# Start Redis (if installed via package manager)
redis-server

# Or use in-memory caching
export TRAJECTORY_CACHE_BACKEND=memory
```

### Development Tips

1. **Use Hot Reload**: Set `TRAJECTORY_HOT_RELOAD=true` for faster development iterations
2. **Enable Debug Mode**: Set `TRAJECTORY_DEBUG=true` for detailed error messages
3. **Use Small Datasets**: Start with synthetic data (1000-5000 trajectories) for faster testing
4. **Monitor Resources**: Keep an eye on memory and CPU usage during development
5. **Version Control**: Commit frequently and use meaningful commit messages
6. **Test Early**: Run tests after each significant change

### IDE Configuration

#### VS Code Setup
Create `.vscode/settings.json`:
```json
{
    "python.defaultInterpreter": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.ruffEnabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/"],
    "files.exclude": {
        "**/__pycache__": true,
        "**/.pytest_cache": true,
        "**/htmlcov": true
    }
}
```

#### PyCharm Setup
1. Set interpreter to `./venv/bin/python`
2. Enable pytest as test runner
3. Configure code style to use Black
4. Enable pre-commit integration

Now you have a fully configured local development environment for the Trajectory Prediction System!