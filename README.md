# Vehicle Trajectory Prediction System

A comprehensive system for predicting vehicle trajectories using advanced machine learning techniques and modern Python engineering practices.

## 🚀 Features

- **Async ETL Pipeline**: Scalable data ingestion with `asyncio` and `aiohttp`
- **Advanced Feature Engineering**: Velocity profiles, acceleration patterns, curvature analysis
- **Multiple ML Models**: CV, CA, Polynomial Regression, KNN, Gaussian Process, Ensemble methods
- **Comprehensive Evaluation**: Safety-critical metrics, statistical validation, performance benchmarking
- **Interactive Dashboard**: Real-time model comparison and trajectory visualization
- **Production MLOps**: MLflow tracking, model serving, monitoring, and drift detection
- **Containerized Deployment**: Multi-stage Docker builds with development and production environments

## 📋 Requirements

- Python 3.9+
- 16GB RAM minimum (recommended)
- GPU support for deep learning models (optional)
- Container runtime (Podman/Docker)

## 🏗️ Architecture

The project follows clean architecture principles with the following structure:

```
src/vehicle_trajectory_prediction/
├── core/           # Core functionality (config, logging, models)
├── data/           # ETL pipeline and data processing
├── features/       # Feature engineering and extraction
├── models/         # ML model implementations
├── evaluation/     # Model evaluation and metrics
├── visualization/  # Dashboard and plotting
├── mlops/          # MLOps infrastructure
├── utils/          # Utility functions
└── cli/            # Command-line interface
```

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd vehicle-trajectory-prediction

# Install dependencies
pip install -e .[dev]

# Setup development environment
python -m vehicle_trajectory_prediction.cli setup
```

### 2. Using Docker (Recommended)

```bash
# Build and start all services
podman-compose up --build

# Or build specific stage
podman build --target development -t trajectory-dev .
```

### 3. Basic Usage

```bash
# Validate configuration
python -m vehicle_trajectory_prediction.cli validate

# Process data
python -m vehicle_trajectory_prediction.cli process-data -d ngsim -o data/processed

# Extract features
python -m vehicle_trajectory_prediction.cli extract-features -d data/processed -o data/features

# Train models
python -m vehicle_trajectory_prediction.cli train -d data/features -o models/

# Evaluate models
python -m vehicle_trajectory_prediction.cli evaluate -d data/test -m models/ -o results/

# Start dashboard
python -m vehicle_trajectory_prediction.cli dashboard

# Start API server
python -m vehicle_trajectory_prediction.cli serve
```

## 📊 Data Models

The system uses Pydantic models for robust data validation:

### TrajectoryPoint
```python
from vehicle_trajectory_prediction.core.models import TrajectoryPoint

point = TrajectoryPoint(
    x=100.0,
    y=200.0,
    timestamp=datetime.now(),
    velocity=25.0,
    acceleration=0.5,
    heading=1.57,
    vehicle_id="vehicle_001"
)
```

### Trajectory
```python
from vehicle_trajectory_prediction.core.models import Trajectory

trajectory = Trajectory(
    vehicle_id="vehicle_001",
    points=[point1, point2, ...],
    start_time=datetime.now(),
    end_time=datetime.now() + timedelta(seconds=10),
    duration=10.0,
    total_distance=100.0
)
```

## ⚙️ Configuration

The system uses Hydra for hierarchical configuration management:

```yaml
# configs/default/config.yaml
environment: development
debug: false

data:
  raw_data_path: data/raw
  processed_data_path: data/processed
  batch_size: 1000
  storage_format: parquet

model:
  models: [cv, ca, polynomial, knn, gaussian_process, ensemble]
  prediction_horizon: 30
  train_test_split: 0.8
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/e2e/

# Run with coverage
pytest --cov=src/vehicle_trajectory_prediction

# Run property-based tests
pytest tests/ -m "property"
```

## 📈 Model Performance

The system implements multiple evaluation metrics:

- **RMSE**: Root Mean Square Error
- **ADE**: Average Displacement Error
- **FDE**: Final Displacement Error
- **Min Distance**: Minimum distance between trajectories
- **TTC**: Time-to-Collision analysis
- **Lateral Error**: Cross-track deviation

## 🏭 Production Deployment

### Using Docker Compose

```bash
# Production deployment
podman-compose -f podman-compose.prod.yml up -d

# Scale services
podman-compose up -d --scale trajectory-prediction=3
```

### Kubernetes Deployment

```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/

# Monitor deployment
kubectl get pods -l app=trajectory-prediction
```

## 📚 API Documentation

The system provides a FastAPI-based REST API:

```bash
# Start API server
python -m vehicle_trajectory_prediction.cli serve

# Access API documentation
open http://localhost:8000/docs
```

### Example API Usage

```python
import requests

# Predict trajectory
response = requests.post("http://localhost:8000/predict", json={
    "vehicle_id": "vehicle_001",
    "current_state": {
        "x": 100.0,
        "y": 200.0,
        "velocity": 25.0,
        "acceleration": 0.5,
        "heading": 1.57,
        "timestamp": "2024-01-01T12:00:00Z"
    },
    "prediction_horizon": 30,
    "model_name": "cv"
})

prediction = response.json()
```

## 🔧 Development

### Code Quality

```bash
# Format code
black src/ tests/
isort src/ tests/

# Type checking
mypy src/

# Linting
flake8 src/ tests/

# Pre-commit hooks
pre-commit install
pre-commit run --all-files
```

### Adding New Models

1. Create model implementation in `src/vehicle_trajectory_prediction/models/`
2. Implement the `BaseModel` interface
3. Add configuration in `configs/`
4. Add tests in `tests/unit/test_models.py`
5. Update CLI commands

### Adding New Features

1. Create feature extractor in `src/vehicle_trajectory_prediction/features/`
2. Add feature configuration
3. Implement validation logic
4. Add tests
5. Update documentation

## 📊 Monitoring and Observability

The system includes comprehensive monitoring:

- **MLflow**: Experiment tracking and model registry
- **Evidently**: Data drift detection and model monitoring
- **Structured Logging**: JSON-formatted logs with correlation IDs
- **Health Checks**: API health endpoints
- **Metrics**: Prometheus-compatible metrics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run the test suite
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- NGSIM dataset for trajectory data
- Open source ML libraries (scikit-learn, XGBoost, GPy)
- MLOps tools (MLflow, Evidently, Hydra)

## 📞 Support

For questions and support:

- Create an issue on GitHub
- Check the documentation
- Review the examples in `notebooks/`

---

**Note**: This is a research and development project. For production use, ensure proper testing, validation, and compliance with relevant safety standards.