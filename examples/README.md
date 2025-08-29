# Trajectory Prediction Examples

This directory contains practical examples demonstrating how to use the Trajectory Prediction System in various scenarios. Each example is self-contained and includes detailed explanations.

## Quick Start Examples

### [basic_prediction.py](basic_prediction.py)
**Simple trajectory prediction with baseline models**
- Load trajectory data
- Create and use prediction models
- Visualize results
- Basic performance evaluation

```bash
python examples/basic_prediction.py
```

### [api_client_example.py](api_client_example.py)
**Using the REST API for predictions**
- Start API server
- Make prediction requests
- Handle responses and errors
- Batch processing

```bash
# Terminal 1: Start API server
python -m trajectory_prediction.api.server

# Terminal 2: Run example
python examples/api_client_example.py
```

## Model Training and Evaluation

### [model_training.py](model_training.py)
**Train models with custom data**
- Load training datasets
- Configure model parameters
- Train advanced models (KNN, Polynomial, GP)
- Save and load trained models

```bash
python examples/model_training.py --data data/ngsim_highway.parquet --models knn polynomial
```

### [model_comparison.py](model_comparison.py)
**Compare multiple prediction models**
- Systematic model evaluation
- Statistical significance testing
- Performance benchmarking
- Generate comparison reports

```bash
python examples/model_comparison.py --test-data data/test_trajectories.parquet
```

### [ensemble_prediction.py](ensemble_prediction.py)
**Ensemble methods for improved accuracy**
- Weighted ensemble combinations
- Dynamic model selection
- Uncertainty aggregation
- Ensemble evaluation

```bash
python examples/ensemble_prediction.py
```

## Real-World Applications

### [autonomous_vehicle_simulation.py](autonomous_vehicle_simulation.py)
**Autonomous vehicle path planning**
- Integration with path planning systems
- Real-time trajectory prediction
- Collision avoidance scenarios
- Safety-critical decision making

```bash
python examples/autonomous_vehicle_simulation.py --scenario highway_merge
```

### [traffic_monitoring.py](traffic_monitoring.py)
**Traffic flow analysis and prediction**
- Multi-vehicle trajectory tracking
- Traffic pattern analysis
- Congestion prediction
- Incident detection

```bash
python examples/traffic_monitoring.py --video data/traffic_video.mp4
```

### [fleet_management.py](fleet_management.py)
**Fleet vehicle optimization**
- Route optimization
- Delivery time prediction
- Fuel efficiency analysis
- Dynamic routing

```bash
python examples/fleet_management.py --fleet-data data/delivery_routes.json
```

## Streaming and Real-Time

### [streaming_prediction.py](streaming_prediction.py)
**Real-time streaming trajectory prediction**
- Kafka/RabbitMQ integration
- Continuous data processing
- Low-latency predictions
- Real-time visualization

```bash
python examples/streaming_prediction.py --stream-source kafka --topic vehicle_telemetry
```

### [websocket_server.py](websocket_server.py)
**WebSocket server for live updates**
- Real-time browser integration
- Live trajectory visualization
- Interactive prediction dashboard
- Multi-client support

```bash
python examples/websocket_server.py
# Open http://localhost:8080 in browser
```

## Data Processing and Analysis

### [data_preprocessing.py](data_preprocessing.py)
**Data cleaning and preparation**
- Handle missing data
- Trajectory smoothing
- Outlier detection and removal
- Data quality assessment

```bash
python examples/data_preprocessing.py --input raw_gps_data.csv --output clean_trajectories.parquet
```

### [synthetic_data_generation.py](synthetic_data_generation.py)
**Generate synthetic training data**
- Various trajectory patterns
- Configurable scenarios
- Noise injection
- Dataset augmentation

```bash
python examples/synthetic_data_generation.py --scenario urban --count 5000 --output synthetic_urban.parquet
```

### [trajectory_analysis.py](trajectory_analysis.py)
**Advanced trajectory analytics**
- Pattern discovery
- Clustering analysis
- Anomaly detection
- Statistical analysis

```bash
python examples/trajectory_analysis.py --data data/city_trajectories.parquet --analysis clustering
```

## Visualization and Dashboards

### [interactive_dashboard.py](interactive_dashboard.py)
**Launch interactive Streamlit dashboard**
- Model comparison interface
- Real-time prediction visualization
- Parameter tuning
- Performance monitoring

```bash
python examples/interactive_dashboard.py
# Equivalent to: streamlit run src/trajectory_prediction/visualization/dashboard.py
```

### [custom_visualization.py](custom_visualization.py)
**Create custom visualizations**
- Publication-quality plots
- Animation creation
- 3D trajectory visualization
- Export to various formats

```bash
python examples/custom_visualization.py --trajectory data/sample.parquet --output visualization.mp4
```

## Integration Examples

### [ros_integration.py](ros_integration.py)
**Robot Operating System (ROS) integration**
- ROS node implementation
- Message conversion
- Real-time vehicle control
- Sensor data processing

```bash
# Requires ROS installation
rosrun trajectory_prediction ros_integration.py
```

### [carla_simulation.py](carla_simulation.py)
**CARLA simulator integration**
- Autonomous driving simulation
- Sensor data processing
- Multi-agent scenarios
- Performance evaluation

```bash
# Requires CARLA simulator
python examples/carla_simulation.py --host localhost --port 2000
```

### [sumo_integration.py](sumo_integration.py)
**SUMO traffic simulator integration**
- Traffic simulation analysis
- Large-scale scenarios
- Validation studies
- Performance benchmarking

```bash
# Requires SUMO installation
python examples/sumo_integration.py --config sumo_scenario.cfg
```

## Performance and Optimization

### [performance_benchmarking.py](performance_benchmarking.py)
**Comprehensive performance testing**
- Model inference speed
- Memory usage analysis
- Scalability testing
- Resource optimization

```bash
python examples/performance_benchmarking.py --models all --iterations 1000
```

### [gpu_acceleration.py](gpu_acceleration.py)
**GPU acceleration examples (future)**
- CUDA/OpenCL optimization
- Batch processing on GPU
- Memory management
- Performance comparison

```bash
python examples/gpu_acceleration.py --device cuda
```

### [distributed_prediction.py](distributed_prediction.py)
**Distributed computing examples**
- Multi-node processing
- Load balancing
- Fault tolerance
- Performance scaling

```bash
python examples/distributed_prediction.py --nodes 4 --tasks 10000
```

## Custom Model Development

### [custom_model_example.py](custom_model_example.py)
**Implement custom prediction models**
- Model interface implementation
- Training pipeline integration
- Evaluation framework
- Model registration

```bash
python examples/custom_model_example.py
```

### [physics_informed_model.py](physics_informed_model.py)
**Physics-informed neural networks**
- Incorporate physical constraints
- Energy conservation
- Momentum preservation
- Advanced modeling techniques

```bash
python examples/physics_informed_model.py --constraints momentum energy
```

### [uncertainty_quantification.py](uncertainty_quantification.py)
**Advanced uncertainty methods**
- Bayesian neural networks
- Monte Carlo dropout
- Ensemble uncertainty
- Calibration techniques

```bash
python examples/uncertainty_quantification.py --method bayesian
```

## Production Deployment

### [production_api.py](production_api.py)
**Production-ready API deployment**
- Load balancing
- Health monitoring
- Logging and metrics
- Error handling

```bash
python examples/production_api.py --workers 4 --max-requests 1000
```

### [kubernetes_deployment.py](kubernetes_deployment.py)
**Kubernetes deployment utilities**
- Container orchestration
- Auto-scaling
- Service mesh integration
- Monitoring setup

```bash
python examples/kubernetes_deployment.py --namespace trajectory-prod
```

### [monitoring_setup.py](monitoring_setup.py)
**Monitoring and observability**
- Prometheus metrics
- Grafana dashboards
- Alert configuration
- Performance tracking

```bash
python examples/monitoring_setup.py --enable-prometheus --grafana-port 3000
```

## Testing and Validation

### [end_to_end_testing.py](end_to_end_testing.py)
**Complete system testing**
- Integration testing
- Performance validation
- Regression testing
- Automated test suites

```bash
python examples/end_to_end_testing.py --test-suite complete
```

### [model_validation.py](model_validation.py)
**Model validation and verification**
- Cross-validation
- Out-of-sample testing
- Robustness analysis
- Fairness evaluation

```bash
python examples/model_validation.py --model polynomial --cross-validate --folds 5
```

## Usage Instructions

### Prerequisites
```bash
# Install the trajectory prediction system
pip install trajectory-prediction

# Install additional dependencies for examples
pip install -r examples/requirements.txt
```

### Running Examples

1. **Basic Examples**: Start with `basic_prediction.py` to understand fundamentals
2. **API Examples**: Use `api_client_example.py` after starting the API server
3. **Advanced Examples**: Require specific data files or external systems
4. **Custom Examples**: Modify parameters and configurations as needed

### Data Requirements

Most examples work with synthetic data generated automatically. For real-world examples:

- **NGSIM Data**: Download from [FHWA NGSIM](https://ops.fhwa.dot.gov/trafficanalysistools/ngsim.htm)
- **Argoverse**: Available from [Argoverse Dataset](https://www.argoverse.org/)
- **Custom Data**: Use CSV/Parquet format with required columns

### Configuration

Examples use configuration files from `config/` directory:
- `config/examples.yaml`: Example-specific settings
- `config/models.yaml`: Model configurations
- `config/data.yaml`: Data processing settings

### Getting Help

If you encounter issues:
1. Check the [troubleshooting guide](../docs/troubleshooting/common_issues.md)
2. Review the [FAQ](../docs/troubleshooting/faq.md)
3. Open an issue on GitHub
4. Join community discussions

### Contributing Examples

We welcome new examples! Please:
1. Follow the existing code structure
2. Include comprehensive documentation
3. Add appropriate error handling
4. Provide sample data or generation code
5. Update this README with your example

Happy predicting! 🚗✨