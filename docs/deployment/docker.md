# Docker Deployment Guide

This guide covers containerized deployment of the Trajectory Prediction System using Docker and Docker Compose for development, testing, and production environments.

## Docker Overview

The system provides multiple Docker configurations:
- **Development**: Hot reload, debugging tools, mounted volumes
- **Production**: Optimized images, security hardening, multi-stage builds
- **Testing**: Isolated testing environment with test data
- **Multi-service**: Full stack with Redis, database, monitoring

## Prerequisites

### System Requirements
- **Docker**: 20.10 or higher
- **Docker Compose**: 2.0 or higher (V2 recommended)
- **Memory**: 4GB+ available to Docker
- **Storage**: 5GB free space for images and volumes

### Installation
```bash
# Install Docker (Ubuntu/Debian)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group
sudo usermod -aG docker $USER

# Install Docker Compose (if not included)
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Verify installation
docker --version
docker-compose --version
```

## Docker Images

### Base Images

#### Production Image
```dockerfile
# docker/Dockerfile
FROM python:3.9-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ src/
COPY pyproject.toml .
COPY README.md .

# Install application
RUN pip install -e .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "-m", "trajectory_prediction.api.server"]
```

#### Development Image
```dockerfile
# docker/Dockerfile.dev
FROM python:3.9-slim

# Install system dependencies including dev tools
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    git \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install development dependencies
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt

# Install pre-commit
RUN pip install pre-commit

# Create development user
RUN useradd --create-home --shell /bin/bash --uid 1000 dev
USER dev

# Expose ports for API and dashboard
EXPOSE 8000 8501

# Default to bash for development
CMD ["/bin/bash"]
```

#### Multi-stage Production Build
```dockerfile
# docker/Dockerfile.prod
# Build stage
FROM python:3.9-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Copy source and build application
COPY src/ src/
COPY pyproject.toml README.md ./
RUN pip install --user -e .

# Production stage
FROM python:3.9-slim as production

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Create application directory
WORKDIR /app

# Copy application code
COPY --from=builder /build/src ./src
COPY --from=builder /build/pyproject.toml ./

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
USER app

# Set PATH to include local packages
ENV PATH=/root/.local/bin:$PATH

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "trajectory_prediction.api.server"]
```

### Specialized Images

#### API Server Image
```dockerfile
# docker/Dockerfile.api
FROM trajectory-prediction:base

# Install gunicorn for production serving
RUN pip install gunicorn[gevent]

# Copy API-specific configuration
COPY config/production.yaml config/
COPY scripts/start-api.sh ./

RUN chmod +x start-api.sh

EXPOSE 8000

CMD ["./start-api.sh"]
```

#### Dashboard Image
```dockerfile
# docker/Dockerfile.dashboard
FROM trajectory-prediction:base

# Install Streamlit
RUN pip install streamlit

# Copy dashboard files
COPY src/trajectory_prediction/visualization/ ./visualization/

EXPOSE 8501

CMD ["streamlit", "run", "visualization/dashboard.py", "--server.address", "0.0.0.0"]
```

## Docker Compose Configurations

### Development Environment
```yaml
# docker-compose.dev.yml
version: '3.8'

services:
  trajectory-api:
    build:
      context: .
      dockerfile: docker/Dockerfile.dev
    container_name: trajectory-dev
    ports:
      - "8000:8000"
      - "8501:8501"
    volumes:
      - .:/workspace
      - dev-cache:/workspace/.cache
    environment:
      - TRAJECTORY_API_DEBUG=true
      - TRAJECTORY_HOT_RELOAD=true
      - TRAJECTORY_LOG_LEVEL=DEBUG
    working_dir: /workspace
    command: >
      bash -c "
        pip install -e . &&
        python -m trajectory_prediction.api.server
      "
    networks:
      - trajectory-dev

  redis:
    image: redis:7-alpine
    container_name: trajectory-redis-dev
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    command: redis-server --appendonly yes
    networks:
      - trajectory-dev

  mlflow:
    image: python:3.9-slim
    container_name: trajectory-mlflow-dev
    ports:
      - "5000:5000"
    volumes:
      - mlflow-data:/mlflow
    command: >
      bash -c "
        pip install mlflow &&
        mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri /mlflow
      "
    networks:
      - trajectory-dev

volumes:
  dev-cache:
  redis-data:
  mlflow-data:

networks:
  trajectory-dev:
    driver: bridge
```

### Production Environment
```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  trajectory-api:
    build:
      context: .
      dockerfile: docker/Dockerfile.prod
    container_name: trajectory-api
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data:ro
      - ./models:/app/models:ro
      - api-logs:/app/logs
    environment:
      - TRAJECTORY_ENV=production
      - TRAJECTORY_REDIS_URL=redis://redis:6379
      - TRAJECTORY_LOG_LEVEL=INFO
    depends_on:
      - redis
    restart: unless-stopped
    networks:
      - trajectory-prod
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
        reservations:
          cpus: '1.0'
          memory: 1G

  trajectory-dashboard:
    build:
      context: .
      dockerfile: docker/Dockerfile.dashboard
    container_name: trajectory-dashboard
    ports:
      - "8501:8501"
    environment:
      - TRAJECTORY_API_URL=http://trajectory-api:8000
    depends_on:
      - trajectory-api
    restart: unless-stopped
    networks:
      - trajectory-prod

  redis:
    image: redis:7-alpine
    container_name: trajectory-redis
    volumes:
      - redis-data:/data
    command: redis-server --appendonly yes --maxmemory 256mb --maxmemory-policy allkeys-lru
    restart: unless-stopped
    networks:
      - trajectory-prod
    deploy:
      resources:
        limits:
          memory: 512M

  nginx:
    image: nginx:alpine
    container_name: trajectory-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./docker/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./docker/ssl:/etc/nginx/ssl:ro
    depends_on:
      - trajectory-api
      - trajectory-dashboard
    restart: unless-stopped
    networks:
      - trajectory-prod

  prometheus:
    image: prom/prometheus:latest
    container_name: trajectory-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./docker/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
    restart: unless-stopped
    networks:
      - trajectory-prod

  grafana:
    image: grafana/grafana:latest
    container_name: trajectory-grafana
    ports:
      - "3000:3000"
    volumes:
      - grafana-data:/var/lib/grafana
      - ./docker/grafana/dashboards:/etc/grafana/provisioning/dashboards:ro
      - ./docker/grafana/datasources:/etc/grafana/provisioning/datasources:ro
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin123
    restart: unless-stopped
    networks:
      - trajectory-prod

volumes:
  redis-data:
  api-logs:
  prometheus-data:
  grafana-data:

networks:
  trajectory-prod:
    driver: bridge
```

### Testing Environment
```yaml
# docker-compose.test.yml
version: '3.8'

services:
  trajectory-test:
    build:
      context: .
      dockerfile: docker/Dockerfile.dev
    container_name: trajectory-test
    volumes:
      - .:/workspace
    environment:
      - TRAJECTORY_ENV=test
      - TRAJECTORY_LOG_LEVEL=WARNING
    working_dir: /workspace
    command: >
      bash -c "
        pip install -e . &&
        python -m pytest tests/ --cov=trajectory_prediction --cov-report=xml
      "
    networks:
      - trajectory-test

  test-redis:
    image: redis:7-alpine
    container_name: trajectory-test-redis
    command: redis-server --save \"\"
    networks:
      - trajectory-test

networks:
  trajectory-test:
    driver: bridge
```

## Build and Deployment Commands

### Development Deployment

#### Quick Start
```bash
# Start development environment
docker-compose -f docker-compose.dev.yml up -d

# View logs
docker-compose -f docker-compose.dev.yml logs -f trajectory-api

# Access services
# API: http://localhost:8000
# Dashboard: http://localhost:8501 (after starting manually)
# Redis: localhost:6379
# MLflow: http://localhost:5000

# Stop services
docker-compose -f docker-compose.dev.yml down
```

#### Development Workflow
```bash
# Build development image
docker build -f docker/Dockerfile.dev -t trajectory-prediction:dev .

# Start with shell access
docker run -it \
  -v $(pwd):/workspace \
  -p 8000:8000 \
  --name trajectory-dev \
  trajectory-prediction:dev

# Run tests in container
docker exec trajectory-dev python -m pytest tests/

# Start API server
docker exec -d trajectory-dev python -m trajectory_prediction.api.server

# Start dashboard
docker exec -d trajectory-dev streamlit run src/trajectory_prediction/visualization/dashboard.py --server.address 0.0.0.0
```

### Production Deployment

#### Single Command Deployment
```bash
# Deploy production stack
docker-compose -f docker-compose.prod.yml up -d

# Check service status
docker-compose -f docker-compose.prod.yml ps

# View logs
docker-compose -f docker-compose.prod.yml logs -f

# Scale API service
docker-compose -f docker-compose.prod.yml up -d --scale trajectory-api=3
```

#### Rolling Updates
```bash
# Build new image with version tag
docker build -f docker/Dockerfile.prod -t trajectory-prediction:v1.2.0 .
docker tag trajectory-prediction:v1.2.0 trajectory-prediction:latest

# Update services one by one
docker-compose -f docker-compose.prod.yml up -d --no-deps trajectory-api

# Health check
curl -f http://localhost:8000/health

# Update remaining services
docker-compose -f docker-compose.prod.yml up -d
```

#### Blue-Green Deployment
```bash
# Start new version (green)
docker-compose -f docker-compose.prod.yml -f docker-compose.green.yml up -d

# Test green deployment
curl -f http://localhost:8001/health

# Switch traffic to green (update load balancer)
docker-compose -f docker-compose.prod.yml -f docker-compose.switch.yml up -d

# Stop blue deployment
docker-compose -f docker-compose.prod.yml down
```

### Testing Deployment

#### Run Test Suite
```bash
# Run all tests
docker-compose -f docker-compose.test.yml up --abort-on-container-exit

# Run specific test category
docker-compose -f docker-compose.test.yml run --rm trajectory-test \
  python -m pytest tests/unit/ -v

# Run with coverage
docker-compose -f docker-compose.test.yml run --rm trajectory-test \
  python -m pytest tests/ --cov=trajectory_prediction --cov-report=html

# Performance benchmarks
docker-compose -f docker-compose.test.yml run --rm trajectory-test \
  python -m pytest tests/performance/ --benchmark-only
```

## Configuration Management

### Environment Variables
```bash
# Create environment file for production
cat > .env.prod << EOF
TRAJECTORY_ENV=production
TRAJECTORY_API_HOST=0.0.0.0
TRAJECTORY_API_PORT=8000
TRAJECTORY_API_WORKERS=4
TRAJECTORY_REDIS_URL=redis://redis:6379
TRAJECTORY_LOG_LEVEL=INFO
TRAJECTORY_METRICS_ENABLED=true
EOF

# Use with docker-compose
docker-compose -f docker-compose.prod.yml --env-file .env.prod up -d
```

### Secrets Management
```bash
# Create Docker secrets
echo "your-api-key" | docker secret create trajectory_api_key -
echo "your-db-password" | docker secret create trajectory_db_password -

# Use in docker-compose (Swarm mode)
version: '3.8'
services:
  trajectory-api:
    # ... other config
    secrets:
      - trajectory_api_key
      - trajectory_db_password
    environment:
      - TRAJECTORY_API_KEY_FILE=/run/secrets/trajectory_api_key

secrets:
  trajectory_api_key:
    external: true
  trajectory_db_password:
    external: true
```

### Volume Mounts
```bash
# Persistent data volumes
docker volume create trajectory_models
docker volume create trajectory_data
docker volume create trajectory_logs

# Mount in production
docker run -d \
  -v trajectory_models:/app/models \
  -v trajectory_data:/app/data \
  -v trajectory_logs:/app/logs \
  trajectory-prediction:latest
```

## Monitoring and Logging

### Container Health Checks
```bash
# Check container health
docker ps --format "table {{.Names}}\t{{.Status}}"

# View health check logs
docker inspect trajectory-api --format='{{json .State.Health}}' | jq

# Manual health check
docker exec trajectory-api curl -f http://localhost:8000/health
```

### Log Management
```bash
# View real-time logs
docker-compose logs -f trajectory-api

# Export logs
docker logs trajectory-api > api-logs-$(date +%Y%m%d).log

# Configure log rotation
cat > /etc/docker/daemon.json << EOF
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  }
}
EOF

# Restart Docker daemon
sudo systemctl restart docker
```

### Resource Monitoring
```bash
# Monitor resource usage
docker stats trajectory-api trajectory-dashboard

# Resource limits in compose
services:
  trajectory-api:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 512M
```

## Optimization and Performance

### Image Optimization
```bash
# Multi-stage build for smaller images
FROM python:3.9-slim as builder
# ... build dependencies and install packages

FROM python:3.9-slim
# ... copy only necessary files

# Use .dockerignore to exclude unnecessary files
cat > .dockerignore << EOF
.git
.gitignore
*.pyc
__pycache__
.pytest_cache
htmlcov
.coverage
docs/
tests/
*.md
EOF
```

### Caching Strategies
```bash
# Layer caching - order matters
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .

# BuildKit for advanced caching
export DOCKER_BUILDKIT=1
docker build --cache-from trajectory-prediction:cache .
```

### Network Optimization
```bash
# Custom network with specific subnet
docker network create --subnet=172.18.0.0/16 trajectory-net

# Use host networking for performance (Linux only)
docker run --network host trajectory-prediction:latest

# Optimize compose networks
networks:
  trajectory-prod:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

## Security Considerations

### Container Security
```dockerfile
# Use non-root user
RUN useradd --create-home --shell /bin/bash --uid 1000 app
USER app

# Read-only root filesystem
docker run --read-only --tmpfs /tmp trajectory-prediction:latest

# Drop capabilities
docker run --cap-drop ALL --cap-add NET_BIND_SERVICE trajectory-prediction:latest
```

### Image Security
```bash
# Scan for vulnerabilities
docker scan trajectory-prediction:latest

# Use minimal base images
FROM python:3.9-alpine  # Instead of python:3.9

# Multi-stage builds to reduce attack surface
FROM builder as production
COPY --from=builder /app /app
# Don't copy build tools and source code
```

### Network Security
```bash
# Isolate networks
docker network create --internal trajectory-internal

# Use secrets for sensitive data
echo "secret-key" | docker secret create api_key -
```

## Troubleshooting

### Common Issues

#### Container Won't Start
```bash
# Check logs
docker logs trajectory-api

# Debug with shell
docker run -it --entrypoint /bin/bash trajectory-prediction:latest

# Check resource limits
docker system df
docker system events
```

#### Network Issues
```bash
# Test connectivity
docker exec trajectory-api ping redis

# Check port bindings
docker port trajectory-api

# Inspect networks
docker network inspect trajectory-prod
```

#### Performance Issues
```bash
# Monitor resources
docker stats --no-stream

# Check disk usage
docker system df

# Cleanup unused resources
docker system prune -f
```

#### Volume Issues
```bash
# Check volume mounts
docker inspect trajectory-api --format='{{range .Mounts}}{{.Source}}:{{.Destination}}{{end}}'

# Fix permissions
docker exec --user root trajectory-api chown -R app:app /app/data
```

### Debugging Tools
```bash
# Enter running container
docker exec -it trajectory-api /bin/bash

# Copy files from container
docker cp trajectory-api:/app/logs/error.log ./

# Run commands in container
docker exec trajectory-api python -c "import trajectory_prediction; print('OK')"
```

This Docker deployment guide provides comprehensive instructions for containerizing and deploying the Trajectory Prediction System across different environments with proper monitoring, security, and optimization considerations.