FROM langchain/langgraph-api:3.11

# Ensure CA certificates available (Debian-based assumption; remove if already present)
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Optional: corporate root CA (uncomment and provide file)
# COPY corp-root-ca.pem /usr/local/share/ca-certificates/corp-root-ca.crt
# RUN update-ca-certificates

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
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]