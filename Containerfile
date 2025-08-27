# Multi-stage Dockerfile for Vehicle Trajectory Prediction System
# Stage 1: Base image with Python and system dependencies
FROM python:3.9-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libgomp1 \
    libgdal-dev \
    libproj-dev \
    proj-data \
    proj-bin \
    libgeos-dev \
    libspatialindex-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash app

# Stage 2: Development image
FROM base as development

# Install development dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY pyproject.toml ./
RUN pip install --upgrade pip && \
    pip install -e .[dev]

# Copy source code
COPY src/ ./src/
COPY configs/ ./configs/
COPY tests/ ./tests/

# Create necessary directories
RUN mkdir -p data/{raw,processed,features,cache} logs notebooks

# Set ownership
RUN chown -R app:app /app

# Switch to non-root user
USER app

# Expose ports
EXPOSE 8000 8888

# Default command for development
CMD ["python", "-m", "pytest", "tests/", "-v"]

# Stage 3: Production image
FROM base as production

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY pyproject.toml ./
RUN pip install --upgrade pip && \
    pip install -e .

# Copy source code
COPY src/ ./src/
COPY configs/ ./configs/

# Create necessary directories
RUN mkdir -p data/{raw,processed,features,cache} logs

# Set ownership
RUN chown -R app:app /app

# Switch to non-root user
USER app

# Expose ports
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command for production
CMD ["python", "-m", "vehicle_trajectory_prediction.cli", "serve"]

# Stage 4: GPU-enabled image (optional)
FROM base as gpu

# Install CUDA dependencies (if needed)
# This stage can be used when GPU support is required
# Uncomment and modify as needed for your GPU setup

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies with GPU support
COPY pyproject.toml ./
RUN pip install --upgrade pip && \
    pip install -e .[gpu]

# Copy source code
COPY src/ ./src/
COPY configs/ ./configs/

# Create necessary directories
RUN mkdir -p data/{raw,processed,features,cache} logs

# Set ownership
RUN chown -R app:app /app

# Switch to non-root user
USER app

# Expose ports
EXPOSE 8000

# Default command for GPU-enabled production
CMD ["python", "-m", "vehicle_trajectory_prediction.cli", "serve"]