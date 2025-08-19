# Multi-stage build for trajectory prediction
FROM python:3.11-slim as builder

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml ./

# Create virtual environment and install dependencies
RUN uv venv .venv
RUN . .venv/bin/activate && uv pip install -e ".[dev]"

# Production stage
FROM python:3.11-slim as production

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash trajectory
USER trajectory
WORKDIR /home/trajectory

# Copy virtual environment from builder
COPY --from=builder --chown=trajectory:trajectory /app/.venv /home/trajectory/.venv

# Copy source code
COPY --chown=trajectory:trajectory src/ ./src/
COPY --chown=trajectory:trajectory configs/ ./configs/
COPY --chown=trajectory:trajectory pyproject.toml ./

# Set environment variables
ENV PATH="/home/trajectory/.venv/bin:$PATH"
ENV PYTHONPATH="/home/trajectory/src:$PYTHONPATH"

# Create necessary directories
RUN mkdir -p data logs outputs models mlruns mlartifacts

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import trajectory_prediction; print('OK')" || exit 1

# Default command
CMD ["python", "-m", "trajectory_prediction.cli"]
