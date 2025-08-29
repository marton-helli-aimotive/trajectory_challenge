# REST API Reference

The Trajectory Prediction API provides RESTful endpoints for real-time trajectory prediction, model management, and system monitoring.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, the API supports optional API key authentication:

```bash
# Optional: Include API key in headers
curl -H "X-API-Key: your-api-key" http://localhost:8000/predict
```

## API Endpoints

### Prediction Endpoints

#### Single Trajectory Prediction

**POST** `/predict`

Predict the future trajectory for a single vehicle.

**Request Body:**
```json
{
  "trajectory": {
    "trajectory_id": "traj_001",
    "vehicle_id": "vehicle_001",
    "points": [
      {
        "timestamp": 0.0,
        "x": 0.0,
        "y": 0.0,
        "vx": 10.0,
        "vy": 0.0
      },
      {
        "timestamp": 0.1,
        "x": 1.0,
        "y": 0.0,
        "vx": 10.0,
        "vy": 0.0
      }
    ]
  },
  "config": {
    "prediction_horizon": 5.0,
    "time_step": 0.1,
    "models": ["constant_velocity", "constant_acceleration"],
    "include_uncertainty": true
  }
}
```

**Response:**
```json
{
  "request_id": "req_12345",
  "model_name": "constant_velocity",
  "predicted_trajectory": {
    "trajectory_id": "traj_001_pred",
    "vehicle_id": "vehicle_001",
    "positions": [
      {"x": 2.0, "y": 0.0},
      {"x": 3.0, "y": 0.0},
      {"x": 4.0, "y": 0.0}
    ],
    "velocities": [
      {"vx": 10.0, "vy": 0.0},
      {"vx": 10.0, "vy": 0.0},
      {"vx": 10.0, "vy": 0.0}
    ],
    "timestamps": [0.2, 0.3, 0.4]
  },
  "confidence": 0.85,
  "uncertainty": {
    "position_0": 0.1,
    "position_1": 0.15,
    "position_2": 0.2
  },
  "inference_time": 0.025,
  "metadata": {
    "model_version": "1.0.0",
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

**Example Usage:**

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "trajectory": {
      "trajectory_id": "example_001",
      "vehicle_id": "car_001",
      "points": [
        {"timestamp": 0.0, "x": 0.0, "y": 0.0, "vx": 15.0, "vy": 0.0},
        {"timestamp": 0.1, "x": 1.5, "y": 0.0, "vx": 15.0, "vy": 0.0},
        {"timestamp": 0.2, "x": 3.0, "y": 0.0, "vx": 15.0, "vy": 0.0}
      ]
    },
    "config": {
      "prediction_horizon": 3.0,
      "time_step": 0.1,
      "models": ["constant_velocity"]
    }
  }'
```

#### Batch Trajectory Prediction

**POST** `/predict/batch`

Predict trajectories for multiple vehicles simultaneously.

**Request Body:**
```json
{
  "trajectories": [
    {
      "trajectory_id": "traj_001",
      "vehicle_id": "vehicle_001",
      "points": [...]
    },
    {
      "trajectory_id": "traj_002", 
      "vehicle_id": "vehicle_002",
      "points": [...]
    }
  ],
  "config": {
    "prediction_horizon": 5.0,
    "time_step": 0.1,
    "models": ["constant_velocity", "polynomial"]
  }
}
```

**Response:**
```json
{
  "predictions": [
    {
      "request_id": "batch_req_001",
      "model_name": "constant_velocity",
      "predicted_trajectory": {...},
      "confidence": 0.85,
      "inference_time": 0.023
    },
    {
      "request_id": "batch_req_002", 
      "model_name": "constant_velocity",
      "predicted_trajectory": {...},
      "confidence": 0.78,
      "inference_time": 0.019
    }
  ],
  "batch_metadata": {
    "total_predictions": 2,
    "total_inference_time": 0.042,
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

#### Ensemble Prediction

**POST** `/predict/ensemble`

Get ensemble predictions from multiple models with aggregation.

**Request Body:**
```json
{
  "trajectory": {...},
  "config": {
    "prediction_horizon": 5.0,
    "models": ["constant_velocity", "constant_acceleration", "polynomial"],
    "ensemble_method": "weighted_average",
    "model_weights": {
      "constant_velocity": 0.4,
      "constant_acceleration": 0.3,
      "polynomial": 0.3
    }
  }
}
```

**Response:**
```json
{
  "ensemble_prediction": {
    "predicted_trajectory": {...},
    "ensemble_confidence": 0.88,
    "model_contributions": {
      "constant_velocity": 0.85,
      "constant_acceleration": 0.82,
      "polynomial": 0.90
    }
  },
  "individual_predictions": [...],
  "ensemble_metadata": {
    "method": "weighted_average",
    "total_models": 3,
    "inference_time": 0.045
  }
}
```

### Model Management Endpoints

#### List Available Models

**GET** `/models`

Get list of available prediction models.

**Response:**
```json
{
  "models": [
    {
      "name": "constant_velocity",
      "type": "baseline",
      "version": "1.0.0",
      "status": "active",
      "description": "Constant velocity motion model",
      "created_at": "2024-01-01T00:00:00Z",
      "metrics": {
        "rmse": 0.45,
        "mae": 0.32,
        "inference_time_ms": 12
      }
    },
    {
      "name": "constant_acceleration",
      "type": "baseline", 
      "version": "1.0.0",
      "status": "active",
      "description": "Constant acceleration motion model",
      "created_at": "2024-01-01T00:00:00Z",
      "metrics": {
        "rmse": 0.38,
        "mae": 0.28,
        "inference_time_ms": 15
      }
    }
  ],
  "total_models": 2,
  "active_models": 2
}
```

#### Model Details

**GET** `/models/{model_name}`

Get detailed information about a specific model.

**Response:**
```json
{
  "name": "constant_velocity",
  "type": "baseline",
  "version": "1.0.0",
  "status": "active",
  "description": "Physics-based constant velocity trajectory prediction",
  "parameters": {
    "prediction_horizon": "configurable",
    "time_step": "configurable",
    "uncertainty_estimation": true
  },
  "performance_metrics": {
    "accuracy": {
      "rmse": 0.45,
      "mae": 0.32,
      "ade": 0.38,
      "fde": 0.52
    },
    "performance": {
      "avg_inference_time_ms": 12,
      "throughput_pred_per_sec": 2150,
      "memory_usage_mb": 45
    }
  },
  "training_info": {
    "training_samples": 50000,
    "validation_accuracy": 0.87,
    "training_time": "N/A - Physics-based model",
    "last_updated": "2024-01-01T00:00:00Z"
  },
  "usage_examples": [
    {
      "scenario": "Highway driving",
      "expected_accuracy": "High",
      "notes": "Best for straight-line motion"
    }
  ]
}
```

### System Monitoring Endpoints

#### Health Check

**GET** `/health`

Get system health status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0",
  "uptime_seconds": 86400,
  "system_info": {
    "models_loaded": 5,
    "total_requests": 12500,
    "cache_hit_rate": 0.78,
    "avg_response_time_ms": 45
  },
  "resource_usage": {
    "cpu_percent": 25.5,
    "memory_percent": 35.2,
    "disk_usage_percent": 12.8
  },
  "dependencies": {
    "database": "healthy",
    "cache": "healthy",
    "model_store": "healthy"
  }
}
```

#### Metrics

**GET** `/metrics`

Get detailed system metrics (Prometheus format).

**Response:**
```prometheus
# HELP prediction_requests_total Total number of prediction requests
# TYPE prediction_requests_total counter
prediction_requests_total{model="constant_velocity"} 8500
prediction_requests_total{model="constant_acceleration"} 4000

# HELP prediction_response_time_seconds Response time for predictions
# TYPE prediction_response_time_seconds histogram
prediction_response_time_seconds_bucket{model="constant_velocity",le="0.01"} 6500
prediction_response_time_seconds_bucket{model="constant_velocity",le="0.05"} 8200
prediction_response_time_seconds_bucket{model="constant_velocity",le="0.1"} 8500

# HELP model_inference_time_seconds Time spent on model inference
# TYPE model_inference_time_seconds histogram
model_inference_time_seconds_bucket{model="constant_velocity",le="0.005"} 7500
model_inference_time_seconds_bucket{model="constant_velocity",le="0.01"} 8200
```

#### Performance Stats

**GET** `/stats`

Get performance statistics and analytics.

**Response:**
```json
{
  "time_period": "last_24h",
  "request_stats": {
    "total_requests": 12500,
    "successful_requests": 12345,
    "failed_requests": 155,
    "success_rate": 0.9876
  },
  "performance_stats": {
    "avg_response_time_ms": 45.2,
    "p95_response_time_ms": 89.5,
    "p99_response_time_ms": 145.2,
    "max_response_time_ms": 234.1
  },
  "model_usage": {
    "constant_velocity": {
      "requests": 7500,
      "avg_confidence": 0.84,
      "avg_inference_time_ms": 12.3
    },
    "constant_acceleration": {
      "requests": 3200,
      "avg_confidence": 0.81,
      "avg_inference_time_ms": 15.7
    },
    "polynomial": {
      "requests": 1800,
      "avg_confidence": 0.88,
      "avg_inference_time_ms": 28.4
    }
  },
  "cache_stats": {
    "cache_hits": 9750,
    "cache_misses": 2750,
    "cache_hit_rate": 0.78,
    "cache_size_mb": 125.4
  }
}
```

## Error Responses

The API uses standard HTTP status codes and returns detailed error information:

### 400 Bad Request
```json
{
  "error": "validation_error",
  "message": "Invalid trajectory data",
  "details": {
    "field": "trajectory.points",
    "issue": "Minimum 2 points required",
    "received": 1
  },
  "request_id": "req_error_001"
}
```

### 404 Not Found
```json
{
  "error": "model_not_found", 
  "message": "Requested model 'advanced_lstm' not available",
  "available_models": ["constant_velocity", "constant_acceleration", "polynomial"],
  "request_id": "req_error_002"
}
```

### 429 Too Many Requests
```json
{
  "error": "rate_limit_exceeded",
  "message": "Rate limit exceeded: 100 requests per minute",
  "retry_after": 30,
  "current_usage": {
    "requests_this_minute": 100,
    "requests_remaining": 0
  },
  "request_id": "req_error_003"
}
```

### 500 Internal Server Error
```json
{
  "error": "internal_error",
  "message": "Model prediction failed",
  "request_id": "req_error_004",
  "support_info": {
    "error_code": "MODEL_PRED_001",
    "timestamp": "2024-01-15T10:30:00Z",
    "contact": "support@trajectory-prediction.ai"
  }
}
```

## Rate Limits

Default rate limits per API key:

- **Free Tier**: 100 requests/minute, 1000 requests/day
- **Premium Tier**: 1000 requests/minute, 50000 requests/day
- **Enterprise**: Custom limits

Rate limit headers are included in all responses:
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 85
X-RateLimit-Reset: 1642248600
```

## Request/Response Examples

### Python Client Example

```python
import requests

# Single prediction
response = requests.post('http://localhost:8000/predict', json={
    'trajectory': {
        'trajectory_id': 'python_example',
        'vehicle_id': 'car_001',
        'points': [
            {'timestamp': 0.0, 'x': 0.0, 'y': 0.0, 'vx': 15.0, 'vy': 0.0},
            {'timestamp': 0.1, 'x': 1.5, 'y': 0.0, 'vx': 15.0, 'vy': 0.0}
        ]
    },
    'config': {
        'prediction_horizon': 2.0,
        'models': ['constant_velocity']
    }
})

prediction = response.json()
print(f"Confidence: {prediction['confidence']}")
print(f"Future positions: {prediction['predicted_trajectory']['positions']}")
```

### JavaScript/Node.js Example

```javascript
const axios = require('axios');

const predictTrajectory = async () => {
  try {
    const response = await axios.post('http://localhost:8000/predict', {
      trajectory: {
        trajectory_id: 'js_example',
        vehicle_id: 'car_002',
        points: [
          { timestamp: 0.0, x: 0.0, y: 0.0, vx: 12.0, vy: 2.0 },
          { timestamp: 0.1, x: 1.2, y: 0.2, vx: 12.0, vy: 2.0 }
        ]
      },
      config: {
        prediction_horizon: 3.0,
        models: ['constant_velocity', 'constant_acceleration']
      }
    });
    
    console.log('Prediction:', response.data);
  } catch (error) {
    console.error('Error:', error.response.data);
  }
};

predictTrajectory();
```

### cURL Examples

```bash
# Health check
curl http://localhost:8000/health

# List models
curl http://localhost:8000/models

# Get model details
curl http://localhost:8000/models/constant_velocity

# Batch prediction
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "trajectories": [
      {
        "trajectory_id": "batch_001",
        "vehicle_id": "car_001", 
        "points": [
          {"timestamp": 0.0, "x": 0.0, "y": 0.0, "vx": 10.0, "vy": 0.0},
          {"timestamp": 0.1, "x": 1.0, "y": 0.0, "vx": 10.0, "vy": 0.0}
        ]
      }
    ],
    "config": {
      "prediction_horizon": 2.0,
      "models": ["constant_velocity"]
    }
  }'
```

## Webhooks (Future)

Support for webhooks to receive real-time prediction updates:

```json
{
  "webhook_url": "https://your-app.com/webhook/predictions",
  "events": ["prediction.completed", "prediction.failed"],
  "secret": "your-webhook-secret"
}
```

## OpenAPI/Swagger Documentation

Interactive API documentation is available at:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc` 
- **OpenAPI JSON**: `http://localhost:8000/openapi.json`