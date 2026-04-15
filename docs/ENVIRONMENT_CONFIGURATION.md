# Environment-Specific Configuration Guide

## Overview

This guide explains how to manage configuration across different environments (development, QA, UAT, production).

## Configuration Loading Priority

Pydantic Settings loads configuration in this order (highest to lowest priority):

1. **Environment Variables** (highest priority - used in production)
2. **`.env` file** (for local development convenience)
3. **Default values in `config.py`** (fallback)

## Best Practices

### ✅ Production Deployments

**DO NOT use `.env` files in production.** Instead, set environment variables directly:

#### Kubernetes
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: myclaims-config
data:
  PERSISTENCE_STORE_TYPE: "mongodb"
  MONGODB_DATABASE_NAME: "myClaims-PT"
  PROJECT_ID: "pbm-prod-myclaims"
---
apiVersion: v1
kind: Secret
metadata:
  name: myclaims-secrets
stringData:
  MONGODB_CONNECTION_STRING: "mongodb+srv://..."
```

#### Docker/Docker Compose
```yaml
# docker-compose.yml
services:
  app:
    environment:
      - PERSISTENCE_STORE_TYPE=mongodb
      - MONGODB_DATABASE_NAME=myClaims-PT
    env_file:
      - .env.production  # Only for non-sensitive defaults
```

#### Google Cloud Run / GKE
```bash
gcloud run deploy myclaims-api \
  --set-env-vars="PERSISTENCE_STORE_TYPE=mongodb,MONGODB_DATABASE_NAME=myClaims-PT" \
  --set-secrets="MONGODB_CONNECTION_STRING=mongodb-conn:latest"
```

#### GitHub Actions / CI/CD
```yaml
# .github/workflows/deploy.yml
env:
  PERSISTENCE_STORE_TYPE: mongodb
  MONGODB_DATABASE_NAME: myClaims-PT
secrets:
  MONGODB_CONNECTION_STRING: ${{ secrets.MONGODB_CONNECTION_STRING_PROD }}
```

### ✅ Local Development

For local development, you can use `.env` files for convenience:

#### Option 1: Single `.env` file (simplest)
```bash
# .env (for your current environment)
PERSISTENCE_STORE_TYPE=mongodb
MONGODB_DATABASE_NAME=myclaims-QA
MONGODB_CONNECTION_STRING=mongodb+srv://...
```

#### Option 2: Environment-specific `.env` files (recommended)
```bash
# .env.dev (development)
PERSISTENCE_STORE_TYPE=sqlite
TELEMETRY_DB_PATH=data/telemetry.db

# .env.qa (QA testing)
PERSISTENCE_STORE_TYPE=mongodb
MONGODB_DATABASE_NAME=myclaims-QA
MONGODB_CONNECTION_STRING=mongodb+srv://myclaims_qa:...@...

# .env.uat (UAT)
PERSISTENCE_STORE_TYPE=mongodb
MONGODB_DATABASE_NAME=myClaims-UAT
MONGODB_CONNECTION_STRING=mongodb+srv://myClaims_uat:...@...

# .env.prod (production - for local testing only, never commit!)
PERSISTENCE_STORE_TYPE=mongodb
MONGODB_DATABASE_NAME=myClaims-PT
MONGODB_CONNECTION_STRING=mongodb+srv://myClaims_pt:...@...
```

**Switch between environments:**
```bash
# Switch to QA
cp .env.qa .env

# Switch to UAT
cp .env.uat .env

# Switch back to dev
cp .env.dev .env
```

## Environment-Specific Settings

### Development
```bash
# .env.dev
PERSISTENCE_STORE_TYPE=sqlite
ENVIRONMENT=development
DEBUG=true
ENABLE_TELEMETRY=true
```

### QA
```bash
# .env.qa
PERSISTENCE_STORE_TYPE=mongodb
MONGODB_DATABASE_NAME=myclaims-QA
MONGODB_CONNECTION_STRING=mongodb+srv://myclaims_qa:...@...
ENVIRONMENT=qa
DEBUG=false
ENABLE_TELEMETRY=true
```

### UAT
```bash
# .env.uat
PERSISTENCE_STORE_TYPE=mongodb
MONGODB_DATABASE_NAME=myClaims-UAT
MONGODB_CONNECTION_STRING=mongodb+srv://myClaims_uat:...@...
ENVIRONMENT=uat
DEBUG=false
ENABLE_TELEMETRY=true
```

### Production
```bash
# .env.prod (local testing only - NEVER commit!)
PERSISTENCE_STORE_TYPE=mongodb
MONGODB_DATABASE_NAME=myClaims-PT
MONGODB_CONNECTION_STRING=mongodb+srv://myClaims_pt:...@...
ENVIRONMENT=production
DEBUG=false
ENABLE_TELEMETRY=true
```

## How Production Knows Which Config to Use

**Production deployments do NOT use `.env` files.** Instead:

1. **Environment variables are set by the deployment platform:**
   - Kubernetes: ConfigMaps and Secrets
   - Docker: Environment variables in docker-compose or `-e` flags
   - Cloud platforms: Environment variables in deployment configuration
   - CI/CD: Environment variables in pipeline configuration

2. **The application reads from environment variables directly:**
   ```python
   # config.py automatically reads from environment variables
   # No .env file needed in production
   settings = Settings()  # Reads from os.environ
   ```

3. **Example production deployment:**
   ```bash
   # Kubernetes deployment
   kubectl set env deployment/myclaims-api \
     PERSISTENCE_STORE_TYPE=mongodb \
     MONGODB_DATABASE_NAME=myClaims-PT \
     MONGODB_CONNECTION_STRING="mongodb+srv://..."
   ```

## Security Best Practices

### ✅ DO:
- Store secrets in secure secret management systems (Kubernetes Secrets, AWS Secrets Manager, Azure Key Vault)
- Use environment variables for production
- Keep `.env` files in `.gitignore` (already done)
- Use `.env.example` as a template (without secrets)

### ❌ DON'T:
- Commit `.env` files with secrets to git
- Use `.env` files in production containers
- Hardcode secrets in code or config files
- Share `.env` files via email or chat

## Example: Complete Setup

### 1. Create environment-specific `.env` files (local only)
```bash
# .env.dev
PERSISTENCE_STORE_TYPE=sqlite

# .env.qa  
PERSISTENCE_STORE_TYPE=mongodb
MONGODB_DATABASE_NAME=myclaims-QA
MONGODB_CONNECTION_STRING=mongodb+srv://myclaims_qa:...@...

# .env.uat
PERSISTENCE_STORE_TYPE=mongodb
MONGODB_DATABASE_NAME=myClaims-UAT
MONGODB_CONNECTION_STRING=mongodb+srv://myClaims_uat:...@...
```

### 2. Add to `.gitignore` (already done)
```gitignore
.env
.env.*
!.env.example
```

### 3. Production deployment (Kubernetes example)
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myclaims-api
spec:
  template:
    spec:
      containers:
      - name: app
        env:
        - name: PERSISTENCE_STORE_TYPE
          value: "mongodb"
        - name: MONGODB_DATABASE_NAME
          value: "myClaims-PT"
        - name: MONGODB_CONNECTION_STRING
          valueFrom:
            secretKeyRef:
              name: mongodb-secrets
              key: connection-string
```

## Troubleshooting

### Issue: Application using wrong environment
**Check:**
1. What environment variables are set: `env | grep PERSISTENCE`
2. If `.env` file exists and what it contains
3. Application logs for configuration values

### Issue: Production using `.env` file
**Solution:** Ensure `.env` file is NOT in the Docker image:
```dockerfile
# Dockerfile should NOT copy .env
# COPY . .  # This is OK - .env is in .gitignore
```

### Issue: Can't switch environments locally
**Solution:** Make sure you're copying the right file:
```bash
# Check current .env
cat .env | grep PERSISTENCE_STORE_TYPE

# Switch to QA
cp .env.qa .env

# Verify
cat .env | grep PERSISTENCE_STORE_TYPE
```

## Summary

- **Local Development:** Use `.env` files (`.env.dev`, `.env.qa`, etc.)
- **Production:** Use environment variables set by deployment platform
- **Priority:** Environment variables > `.env` file > defaults
- **Security:** Never commit `.env` files with secrets
- **Flexibility:** Switch local environments by copying `.env.{env}` to `.env`

