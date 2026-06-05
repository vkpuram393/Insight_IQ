# -------------------------
# Stage 1: Build
# -------------------------
FROM cvsh.jfrog.io/dockerhub/library/python:3.12-slim AS builder

WORKDIR /app

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1

# Copy requirements first (better layer caching)
COPY requirements.txt .

# Upgrade pip and install dependencies into a local folder
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt \
    --target=/app/deps

# Install spaCy large English model (needed by Presidio)
RUN pip install --no-cache-dir \
    https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.7.1/en_core_web_lg-3.7.1-py3-none-any.whl \
    --target=/app/deps

# Copy application code
COPY . .

# -------------------------
# Stage 2: Runtime (Distroless)
# -------------------------
FROM cvsh.jfrog.io/cvsdigital-docker/devsecops/distroless/python-3.12-distroless:latest

WORKDIR /app

# Copy everything from builder
COPY --from=builder /app /app

# Set Python path
ENV PYTHONPATH="/app:/app/deps"

# App port
EXPOSE 8000

# Run application
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
