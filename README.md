# Advanced Vehicle Trajectory Prediction Pipeline

A comprehensive machine learning engineering project for predicting vehicle trajectories using modern Python practices, advanced ML models, and production-ready MLOps infrastructure.

## Features

- **Async ETL Pipeline**: Scalable data ingestion with Apache Parquet storage
- **5 Prediction Models**: From baseline to advanced ensemble approaches  
- **Comprehensive Evaluation**: Safety-critical metrics (TTC, minimum distance, etc.)
- **Interactive Dashboard**: Real-time model comparison and visualization
- **MLOps Ready**: Experiment tracking, monitoring, and model serving
- **Production Infrastructure**: Containerized deployment with horizontal scaling

## Quick Start

```bash
# Install dependencies
pip install -e .

# Initialize data pipeline
trajectory-predict init

# Train models
trajectory-predict train --config configs/train.yaml

# Launch dashboard
trajectory-dashboard --port 8050

# Serve models
trajectory-serve --host 0.0.0.0 --port 8000
```

## Project Structure

```
trajectory_prediction/
├── src/trajectory_prediction/    # Core package
│   ├── data/                    # ETL pipeline and data processing
│   ├── models/                  # ML model implementations
│   ├── features/               # Feature engineering
│   ├── evaluation/             # Metrics and validation
│   ├── visualization/          # Plotting and dashboard components
│   └── config/                 # Configuration management
├── tests/                      # Test suite
├── notebooks/                  # Jupyter analysis notebooks
├── configs/                    # Hydra configuration files
├── docker/                     # Container definitions
└── docs/                       # Documentation
```

## Models Implemented

1. **Baseline Models**: Constant Velocity/Acceleration
2. **Polynomial Regression**: Physics-informed features
3. **K-Nearest Neighbors**: Trajectory similarity matching
4. **Gaussian Process**: Uncertainty-aware predictions
5. **Tree Ensemble**: XGBoost with trajectory features

## Dataset Support

- **NGSIM**: US highway trajectory data
- Extensible factory pattern for additional datasets

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install

# Run tests
pytest tests/ -v

# Type checking
mypy src/

# Code formatting
black src/ tests/
isort src/ tests/
```

## Deployment

```bash
# Build containers
podman build -f docker/Containerfile -t trajectory-prediction .

# Launch services
podman-compose up -d
```

## Documentation

- [Architecture Overview](docs/architecture/README.md)
- [Model Comparison](docs/models/comparison.md)
- [API Reference](docs/api/README.md)
- [Deployment Guide](docs/deployment/README.md)

## License

MIT License - see [LICENSE](LICENSE) file for details.