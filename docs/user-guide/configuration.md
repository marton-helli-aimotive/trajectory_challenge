# Configuration Reference

This guide provides comprehensive information about configuring the Trajectory Prediction System for different environments and use cases.

## Configuration Overview

The system supports multiple configuration methods with the following precedence (highest to lowest):
1. **Command-line arguments**
2. **Environment variables**
3. **Configuration files**
4. **Default values**

## Configuration Files

### File Locations

The system looks for configuration files in the following order:

```bash
# 1. Explicit config file (highest priority)
--config /path/to/config.yaml

# 2. Current directory
./config.yaml
./trajectory_prediction.yaml

# 3. User config directory
~/.config/trajectory_prediction/config.yaml

# 4. System config directory
/etc/trajectory_prediction/config.yaml
```

### Configuration File Format

Configuration files use YAML format with hierarchical sections:

```yaml
# config.yaml - Main configuration file
api:
  host: "localhost"
  port: 8000
  debug: false
  workers: 4

models:
  cache_size: 100
  default_horizon: 5.0
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

monitoring:
  log_level: "INFO"
  metrics_enabled: true
```

## Configuration Sections

### 1. API Configuration

Controls the REST API server behavior:

```yaml
api:
  # Server binding
  host: "0.0.0.0"                    # Bind address (0.0.0.0 for all interfaces)
  port: 8000                         # Server port
  
  # Server behavior
  debug: false                       # Enable debug mode
  hot_reload: false                  # Enable hot reload in development
  workers: 4                         # Number of worker processes
  timeout: 30                        # Request timeout in seconds
  
  # CORS settings
  cors_enabled: true                 # Enable CORS
  cors_origins: ["*"]               # Allowed origins (* for all)
  cors_methods: ["GET", "POST"]     # Allowed methods
  cors_headers: ["*"]               # Allowed headers
  
  # Rate limiting
  rate_limit_enabled: false         # Enable rate limiting
  rate_limit_requests: 100          # Requests per minute
  rate_limit_window: 60             # Time window in seconds
  
  # Authentication
  auth_enabled: false               # Enable API key authentication
  api_keys: []                      # List of valid API keys
  
  # SSL/TLS
  ssl_enabled: false                # Enable HTTPS
  ssl_cert_path: ""                 # Path to SSL certificate
  ssl_key_path: ""                  # Path to SSL private key
```

**Environment Variables:**
```bash
TRAJECTORY_API_HOST=localhost
TRAJECTORY_API_PORT=8000
TRAJECTORY_API_DEBUG=false
TRAJECTORY_API_WORKERS=4
TRAJECTORY_API_TIMEOUT=30
TRAJECTORY_API_SSL_ENABLED=false
```

### 2. Model Configuration

Configures prediction models and their behavior:

```yaml
models:
  # Model management
  cache_size: 100                   # Number of models to keep in memory
  model_timeout: 30                 # Model loading timeout (seconds)
  
  # Default prediction parameters
  default_horizon: 5.0              # Default prediction horizon (seconds)
  default_time_step: 0.1            # Default time step (seconds)
  max_horizon: 30.0                 # Maximum allowed horizon
  min_time_step: 0.01               # Minimum allowed time step
  
  # Available models
  available_models:
    - constant_velocity
    - constant_acceleration
    - polynomial
    - knn
    - gaussian_process
  
  # Model-specific configurations
  constant_velocity:
    noise_std: 0.1                  # Uncertainty standard deviation
    max_speed: 50.0                 # Maximum allowed speed (m/s)
  
  constant_acceleration:
    noise_std: 0.15                 # Base uncertainty
    max_acceleration: 8.0           # Maximum acceleration (m/s²)
    physics_validation: true        # Enforce physics constraints
  
  polynomial:
    max_degree: 4                   # Maximum polynomial degree
    regularization_strength: 1e-6   # Bayesian regularization strength
    physics_constraints: true       # Apply physics constraints
  
  knn:
    k: 5                           # Number of neighbors
    similarity_metric: "dtw"        # Distance metric (dtw, euclidean)
    distance_threshold: 10.0        # Maximum distance for valid neighbors
    feature_weights:
      position: 1.0
      velocity: 0.7
      acceleration: 0.5
  
  gaussian_process:
    kernel_type: "rbf_periodic"     # Kernel type
    length_scale: 1.0               # RBF kernel length scale
    noise_level: 0.1                # Noise level
    optimize_hyperparameters: true  # Auto-tune hyperparameters
```

**Environment Variables:**
```bash
TRAJECTORY_MODEL_CACHE_SIZE=100
TRAJECTORY_DEFAULT_HORIZON=5.0
TRAJECTORY_DEFAULT_TIME_STEP=0.1
TRAJECTORY_MAX_HORIZON=30.0
```

### 3. Data Configuration

Controls data processing, storage, and management:

```yaml
data:
  # Paths
  data_path: "./data"               # Main data directory
  model_path: "./models"            # Trained models directory
  cache_path: "./cache"             # Cache directory
  log_path: "./logs"                # Log files directory
  
  # Data processing
  batch_size: 32                    # Processing batch size
  num_workers: 4                    # Parallel processing workers
  max_trajectory_length: 200        # Maximum trajectory points
  min_trajectory_length: 2          # Minimum trajectory points
  
  # Data validation
  validate_schema: true             # Validate data schemas
  remove_outliers: true             # Remove statistical outliers
  smooth_trajectories: false        # Apply smoothing filter
  
  # Storage format
  default_format: "parquet"         # Default storage format
  compression: "snappy"             # Compression algorithm
  
  # Quality thresholds
  min_speed: 0.1                    # Minimum vehicle speed (m/s)
  max_speed: 60.0                   # Maximum vehicle speed (m/s)
  max_acceleration: 15.0            # Maximum acceleration (m/s²)
  max_position_jump: 50.0           # Maximum position jump (m)
  
  # Coordinate system
  coordinate_system: "cartesian"     # Coordinate system (cartesian, geographic)
  origin_lat: 0.0                   # Origin latitude (for geographic)
  origin_lon: 0.0                   # Origin longitude (for geographic)
```

**Environment Variables:**
```bash
TRAJECTORY_DATA_PATH=./data
TRAJECTORY_MODEL_PATH=./models
TRAJECTORY_CACHE_PATH=./cache
TRAJECTORY_BATCH_SIZE=32
TRAJECTORY_NUM_WORKERS=4
```

### 4. Monitoring Configuration

Configures logging, metrics, and observability:

```yaml
monitoring:
  # Logging
  log_level: "INFO"                 # Log level (DEBUG, INFO, WARNING, ERROR)
  log_format: "structured"          # Log format (structured, simple)
  log_file: "trajectory.log"        # Log file name (in log_path)
  log_rotation: true                # Enable log rotation
  log_max_size: "100MB"             # Max log file size
  log_backup_count: 5               # Number of backup files
  
  # Metrics
  metrics_enabled: true             # Enable metrics collection
  metrics_port: 9090                # Prometheus metrics port
  metrics_path: "/metrics"          # Metrics endpoint path
  
  # Performance monitoring
  performance_logging: true         # Log performance metrics
  resource_monitoring: true         # Monitor CPU, memory usage
  health_check_interval: 30         # Health check interval (seconds)
  
  # Alerting
  alerting_enabled: false           # Enable alerting
  alert_thresholds:
    error_rate: 0.05                # Error rate threshold (5%)
    response_time: 1000             # Response time threshold (ms)
    memory_usage: 0.8               # Memory usage threshold (80%)
  
  # Tracing
  tracing_enabled: false            # Enable distributed tracing
  trace_sample_rate: 0.1            # Trace sampling rate
```

**Environment Variables:**
```bash
TRAJECTORY_LOG_LEVEL=INFO
TRAJECTORY_LOG_FILE=trajectory.log
TRAJECTORY_METRICS_ENABLED=true
TRAJECTORY_METRICS_PORT=9090
TRAJECTORY_PERFORMANCE_LOGGING=true
```

### 5. Caching Configuration

Controls caching behavior for improved performance:

```yaml
caching:
  # Cache backends
  backend: "memory"                 # Cache backend (memory, redis, hybrid)
  
  # Memory cache settings
  memory_cache_size: 1000           # Max items in memory cache
  memory_cache_ttl: 300             # Time to live (seconds)
  
  # Redis cache settings
  redis_url: "redis://localhost:6379"  # Redis connection URL
  redis_db: 0                       # Redis database number
  redis_prefix: "trajectory:"       # Key prefix
  redis_ttl: 3600                   # Redis TTL (seconds)
  redis_pool_size: 10               # Connection pool size
  
  # Cache policies
  cache_predictions: true           # Cache prediction results
  cache_models: true                # Cache loaded models
  cache_features: true              # Cache extracted features
  
  # Cache sizes
  prediction_cache_size: 10000      # Max cached predictions
  model_cache_size: 100             # Max cached models
  feature_cache_size: 5000          # Max cached feature sets
```

**Environment Variables:**
```bash
TRAJECTORY_CACHE_BACKEND=memory
TRAJECTORY_MEMORY_CACHE_SIZE=1000
TRAJECTORY_REDIS_URL=redis://localhost:6379
TRAJECTORY_CACHE_PREDICTIONS=true
```

### 6. Database Configuration

For systems using persistent storage:

```yaml
database:
  # Database connection
  url: "sqlite:///./trajectory.db"  # Database URL
  echo: false                       # Echo SQL queries
  pool_size: 5                      # Connection pool size
  max_overflow: 10                  # Max overflow connections
  
  # Tables
  trajectory_table: "trajectories"  # Main trajectory table
  model_table: "models"             # Model metadata table
  metrics_table: "metrics"          # Metrics table
  
  # Migration
  auto_migrate: true                # Auto-run migrations
  migration_path: "./migrations"    # Migration scripts path
```

**Environment Variables:**
```bash
TRAJECTORY_DB_URL=sqlite:///./trajectory.db
TRAJECTORY_DB_ECHO=false
TRAJECTORY_AUTO_MIGRATE=true
```

### 7. Security Configuration

Security-related settings:

```yaml
security:
  # API Security
  api_key_required: false           # Require API key
  api_keys: []                      # Valid API keys
  
  # Rate limiting
  rate_limiting: false              # Enable rate limiting
  rate_limit_per_minute: 100        # Requests per minute
  rate_limit_burst: 10              # Burst allowance
  
  # CORS
  cors_enabled: true                # Enable CORS
  allowed_origins: ["*"]            # Allowed origins
  
  # Headers
  security_headers: true            # Add security headers
  content_type_nosniff: true        # X-Content-Type-Options
  frame_options: "DENY"             # X-Frame-Options
  xss_protection: "1; mode=block"   # X-XSS-Protection
  
  # Data privacy
  anonymize_data: false             # Anonymize trajectory data
  encrypt_at_rest: false            # Encrypt stored data
  retention_days: 90                # Data retention period
```

### 8. Development Configuration

Settings for development environments:

```yaml
development:
  # Debug settings
  debug: true                       # Enable debug mode
  auto_reload: true                 # Auto-reload on code changes
  profiling_enabled: true           # Enable performance profiling
  
  # Testing
  test_mode: false                  # Enable test mode
  use_synthetic_data: true          # Use synthetic data when real data unavailable
  mock_external_services: false     # Mock external service calls
  
  # Validation
  strict_validation: true           # Strict input validation
  validate_all_responses: false     # Validate all API responses
```

## Environment-Specific Configurations

### Development Environment

```yaml
# config/development.yaml
api:
  host: "localhost"
  port: 8000
  debug: true
  hot_reload: true
  workers: 1

models:
  cache_size: 10
  default_horizon: 3.0

data:
  batch_size: 16
  validate_schema: true

monitoring:
  log_level: "DEBUG"
  performance_logging: true
  
development:
  debug: true
  auto_reload: true
  use_synthetic_data: true
```

### Production Environment

```yaml
# config/production.yaml
api:
  host: "0.0.0.0"
  port: 8000
  debug: false
  workers: 4
  cors_enabled: true
  rate_limit_enabled: true
  rate_limit_requests: 1000

models:
  cache_size: 100
  default_horizon: 5.0

data:
  batch_size: 32
  num_workers: 4

monitoring:
  log_level: "INFO"
  metrics_enabled: true
  performance_logging: true

caching:
  backend: "redis"
  redis_url: "redis://redis:6379"
  cache_predictions: true

security:
  api_key_required: true
  rate_limiting: true
  security_headers: true
```

### Testing Environment

```yaml
# config/testing.yaml
api:
  host: "localhost"
  port: 0  # Use random available port
  debug: false
  workers: 1

models:
  cache_size: 5
  default_horizon: 2.0

data:
  batch_size: 8
  validate_schema: true

monitoring:
  log_level: "WARNING"
  metrics_enabled: false
  
development:
  test_mode: true
  use_synthetic_data: true
  strict_validation: true
```

## Configuration Management

### Loading Configuration

```python
from trajectory_prediction.config import load_config

# Load from file
config = load_config("config.yaml")

# Load from environment
config = load_config()

# Load with overrides
config = load_config("config.yaml", overrides={"api.port": 8001})
```

### Environment Variable Override

Set environment variables to override configuration:

```bash
# Override API port
export TRAJECTORY_API_PORT=8001

# Override log level
export TRAJECTORY_LOG_LEVEL=DEBUG

# Override Redis URL
export TRAJECTORY_REDIS_URL=redis://prod-redis:6379
```

### Command-Line Override

Use command-line arguments for runtime configuration:

```bash
# Start API server with overrides
python -m trajectory_prediction.api.server \
  --host 0.0.0.0 \
  --port 8080 \
  --workers 2 \
  --log-level DEBUG
```

### Configuration Validation

The system validates configuration at startup:

```python
from trajectory_prediction.config import validate_config

# Validate configuration
errors = validate_config(config)
if errors:
    print("Configuration errors:")
    for error in errors:
        print(f"  - {error}")
```

### Dynamic Configuration

Some settings can be updated at runtime:

```python
from trajectory_prediction.config import update_config

# Update log level
update_config({"monitoring.log_level": "DEBUG"})

# Update cache size
update_config({"models.cache_size": 200})
```

## Configuration Examples

### High-Performance Setup

For high-throughput scenarios:

```yaml
api:
  workers: 8
  timeout: 10

models:
  cache_size: 200
  
data:
  batch_size: 64
  num_workers: 8

caching:
  backend: "redis"
  prediction_cache_size: 50000
  model_cache_size: 50
```

### Memory-Constrained Setup

For resource-limited environments:

```yaml
api:
  workers: 2

models:
  cache_size: 5
  available_models:
    - constant_velocity  # Only fast models

data:
  batch_size: 8
  max_trajectory_length: 50

caching:
  backend: "memory"
  memory_cache_size: 100
```

### Security-Focused Setup

For secure production environments:

```yaml
api:
  ssl_enabled: true
  ssl_cert_path: "/etc/ssl/certs/trajectory.crt"
  ssl_key_path: "/etc/ssl/private/trajectory.key"

security:
  api_key_required: true
  rate_limiting: true
  security_headers: true
  cors_enabled: false
  anonymize_data: true

monitoring:
  log_level: "WARNING"  # Reduce log verbosity
  tracing_enabled: true
```

## Troubleshooting Configuration

### Common Issues

#### Port Already in Use
```yaml
api:
  port: 8001  # Use different port
```

#### Memory Issues
```yaml
models:
  cache_size: 10      # Reduce cache size
data:
  batch_size: 8       # Smaller batches
```

#### Permission Errors
```yaml
data:
  data_path: "/tmp/trajectory_data"    # Use accessible path
  log_path: "/tmp/trajectory_logs"
```

### Configuration Validation

Check configuration validity:

```bash
# Validate configuration file
python -m trajectory_prediction.config validate config.yaml

# Show effective configuration
python -m trajectory_prediction.config show

# Test configuration
python -m trajectory_prediction.config test --config config.yaml
```

This configuration reference provides comprehensive control over the Trajectory Prediction System's behavior across different environments and use cases.