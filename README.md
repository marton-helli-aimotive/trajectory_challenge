# Trajectory Prediction Challenge

Advanced vehicle trajectory prediction pipeline using modern Python practices and machine learning engineering patterns.

## Overview

This project demonstrates a comprehensive approach to vehicle trajectory prediction, featuring:

- Scalable ETL pipeline with async data processing
- Multiple ML models (baseline, polynomial, KNN, Gaussian Process, ensemble)
- Comprehensive evaluation framework with safety-critical metrics
- Interactive dashboard for model comparison and visualization
- Production-ready MLOps pipeline with monitoring

## Project Structure

```
trajectory_challenge/
├── src/
│   └── trajectory_prediction/      # Main package
│       ├── data/                   # ETL and data processing
│       ├── models/                 # ML models and training
│       ├── evaluation/             # Metrics and validation
│       ├── visualization/          # Dashboard and plotting
│       └── utils/                  # Common utilities
├── tests/                          # Test suite
├── configs/                        # Hydra configuration files
├── notebooks/                      # Jupyter notebooks
├── docs/                          # Documentation
├── scripts/                       # Utility scripts
├── pyproject.toml                 # Project configuration
├── Containerfile                  # Container definition
└── docker-compose.yml            # Multi-service setup
```

## Quick Start

### Environment Setup

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone and setup**:
   ```bash
   git clone <repository-url>
   cd trajectory_challenge
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip install -e ".[dev]"
   ```

3. **Setup pre-commit hooks**:
   ```bash
   pre-commit install
   ```

### Development

- **Run tests**: `pytest`
- **Type checking**: `mypy src/`
- **Linting**: `ruff check src/`
- **Format code**: `black src/ tests/`
- **Run dashboard**: `streamlit run scripts/dashboard/trajectory_dashboard_app.py`

## Features

### Data Processing
- Async ETL pipeline for concurrent data ingestion
- Columnar storage with Apache Parquet
- NGSIM dataset support with extensible schema
- Data quality validation and cleaning

### Machine Learning Models
- Baseline models (Constant Velocity, Constant Acceleration)
- Polynomial regression with engineered features
- K-Nearest Neighbors with dynamic time warping
- Gaussian Process regression with uncertainty quantification
- Tree-based ensembles (Random Forest, XGBoost)

### Evaluation Framework
- Safety-critical metrics (TTC, minimum distance)
- Standard trajectory metrics (ADE, FDE, RMSE)
- Statistical significance testing
- Uncertainty quantification

### Visualization
- Interactive Dash/Streamlit dashboard
- Real-time model comparison
- 2D/3D trajectory visualization
- Error analysis and heatmaps

### MLOps
- MLflow experiment tracking
- Hydra configuration management
- Model monitoring with Evidently AI
- Containerized deployment

## Requirements

- Python 3.11+
- See `pyproject.toml` for complete dependency list

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run pre-commit hooks
5. Submit a pull request

## Documentation

Detailed documentation is available in the `docs/` directory and will be published at [readthedocs.io](https://trajectory-prediction.readthedocs.io).
