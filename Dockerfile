FROM langchain/langgraph-api:3.11

# Ensure CA certificates available (Debian-based assumption; remove if already present)
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# CVS Health Root CA certificate for Zscaler proxy
COPY certs/CVSHealthRoot.cer /usr/local/share/ca-certificates/CVSHealthRoot.crt
RUN update-ca-certificates

# Set environment variables for Python SSL certificate handling
ENV SSL_CERT_FILE=/usr/local/share/ca-certificates/CVSHealthRoot.crt
ENV REQUESTS_CA_BUNDLE=/usr/local/share/ca-certificates/CVSHealthRoot.crt

# Set default DATABASE_URI for LangGraph API base image (required by base image config)
# This can be overridden via environment variables in Kubernetes deployment
ENV DATABASE_URI=sqlite:///./checkpoints.db

WORKDIR /api

# Build args to remove warning
ARG dep=none

# Copy requirement specs early
COPY requirements.txt .

# Native TLS to leverage system certs (optional; use only if needed)
ENV UV_NATIVE_TLS=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1

# Install uv if base image does not already have it; otherwise skip
RUN pip install --upgrade uv pip

# Correct install command with -r
RUN uv pip install --system --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port (update if using different)
EXPOSE 8000

# Start (adjust entrypoint/command to your app)
# Note: main.py is at the root, not in an app directory
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]