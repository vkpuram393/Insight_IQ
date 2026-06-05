# -------------------------
# Stage 1: Build
# -------------------------
FROM cvsh.jfrog.io/cvsdigital-docker/dockerhub/python:3.12-slim AS builder

WORKDIR /app

COPY requirements.txt .

# Install dependencies into local folder
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt \
    --target=/app/deps

# Install spacy model (build stage only)
RUN pip install --no-cache-dir https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.7.1/en_core_web_lg-3.7.1-py3-none-any.whl --target=/app/deps

COPY . .

# -------------------------
# Stage 2: Runtime (DISTROLESS)
# -------------------------
FROM cvsh.jfrog.io/cvsdigital-docker/devsecops/distroless/python-3.12-distroless:latest

WORKDIR /app

COPY --from=builder /app /app

ENV PYTHONPATH="/app:/app/deps"

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]



# FROM langchain/langgraph-api:3.11

# # # Ensure CA certificates available (Debian-based assumption; remove if already present)
# RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
#  && rm -rf /var/lib/apt/lists/*

# # # CVS Health Root CA certificate for Zscaler proxy
# # COPY certs/CVSHealthRoot.cer /usr/local/share/ca-certificates/CVSHealthRoot.crt
# # RUN update-ca-certificates

# # # Set environment variables for Python SSL certificate handling
# # ENV SSL_CERT_FILE=/usr/local/share/ca-certificates/CVSHealthRoot.crt
# # ENV REQUESTS_CA_BUNDLE=/usr/local/share/ca-certificates/CVSHealthRoot.crt

# # Note: DATABASE_URI is not used by this application - removed
# # The application uses persistence_store_type, mongodb_connection_string, etc. from config.py
# ENV PYTHONPATH=/api
# WORKDIR /api

# # Build args to remove warning
# ARG dep=none

# # Copy requirement specs early
# COPY requirements.txt .

# # Native TLS to leverage system certs (optional; use only if needed)
# ENV UV_NATIVE_TLS=1
# ENV PYTHONDONTWRITEBYTECODE=1
# ENV PIP_NO_CACHE_DIR=1

# # Install uv if base image does not already have it; otherwise skip
# RUN pip install --upgrade uv pip

# # Correct install command with -r
# RUN uv pip install --system --no-cache-dir -r requirements.txt

# # Install spacy model required by Presidio AnalyzerEngine
# # This prevents runtime downloads that cause pod restarts
# # Presidio requires en-core-web-lg for PII detection
# # Installing directly via pip with the wheel file (compatible with spacy 3.7.2)
# # Using pip (not uv pip) because uv may not handle direct wheel URLs correctly
# RUN pip install --no-cache-dir https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.7.1/en_core_web_lg-3.7.1-py3-none-any.whl

# # Copy application code
# COPY . .

# # Expose port (update if using different)
# EXPOSE 8000

# # Override base image entrypoint to use our application
# # The base image may have its own LangGraph API server, so we explicitly set our entrypoint
# ENTRYPOINT []

# # Start our application (main.py is at root, not in app/ subdirectory)
# # CRITICAL: Do NOT use --reload flag in production to prevent server restarts
# # Database file writes (checkpoints.db, telemetry.db) trigger reloads if --reload is enabled
# # Default behavior (no --reload flag) = reload disabled = stable production behavior
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
