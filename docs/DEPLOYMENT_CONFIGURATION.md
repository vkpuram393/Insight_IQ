# Deployment Configuration Guide

## Where Configuration is Loaded From in Deployments

When you deploy the application to **dev** or **qa**, here's where each configuration source is picked up:

### 1. `config.py` (Python Settings)

**Location:** Part of the application code (always deployed)

**Loading Priority:**
1. **Environment Variables** (highest priority - set by deployment platform)
2. `.env` file (if exists in container - **NOT recommended for production**)
3. Default values in `config.py` class (fallback)

**How it works:**
```python
# config.py is Python code - always part of the deployment
settings = Settings()  # Automatically reads from environment variables
```

**In Dev/QA Deployment:**
- Set environment variables in your deployment platform (Kubernetes, Docker, Cloud Run, etc.)
- Example:
  ```bash
  PERSISTENCE_STORE_TYPE=mongodb
  MONGODB_DATABASE_NAME=myclaims-QA
  MONGODB_CONNECTION_STRING=mongodb+srv://...
  ```

### 2. `domain_config.json` (Domain Configuration)

**Location:** Part of the application code (always deployed)

**Loading:** Hardcoded file path in the code:
```python
# In nodes/context.py, nodes/confidence.py, etc.
config_path = Path(__file__).parent.parent / "config" / "domain_config.json"
```

**Path in deployment:**
- Local: `{project_root}/config/domain_config.json`
- Docker: `/api/config/domain_config.json` (inside container)
- Kubernetes: `{app_directory}/config/domain_config.json` (in pod)

**Important:** This file is **always read from the deployed code directory**. It's part of your application code, not environment-specific.

**If you need environment-specific domain configs:**
- Option 1: Use environment variables to override specific values
- Option 2: Mount different config files via ConfigMaps (Kubernetes)
- Option 3: Build different Docker images per environment

### 3. `.env` File

**Location:** Should NOT be in production deployments

**Loading:** Only if file exists in working directory

**Best Practice:**
- ❌ **DO NOT** include `.env` files in Docker images
- ❌ **DO NOT** commit `.env` files to git
- ✅ **DO** use environment variables set by deployment platform

## Deployment Scenarios

### Scenario 1: Kubernetes Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myclaims-api
spec:
  template:
    spec:
      containers:
      - name: app
        image: myclaims-api:latest
        env:
        # config.py reads from these environment variables
        - name: PERSISTENCE_STORE_TYPE
          value: "mongodb"
        - name: MONGODB_DATABASE_NAME
          value: "myclaims-QA"
        - name: MONGODB_CONNECTION_STRING
          valueFrom:
            secretKeyRef:
              name: mongodb-secrets
              key: connection-string
        # domain_config.json is read from:
        # /api/config/domain_config.json (inside container)
```

**What gets loaded:**
- ✅ `config.py`: From environment variables (PERSISTENCE_STORE_TYPE, etc.)
- ✅ `domain_config.json`: From `/api/config/domain_config.json` (in container)
- ❌ `.env`: Not used (shouldn't exist in container)

### Scenario 2: Docker Deployment

```dockerfile
# Dockerfile
COPY . .  # Copies all code including config.py and domain_config.json
# .env is NOT copied (it's in .gitignore)
```

```bash
# Run container with environment variables
docker run -e PERSISTENCE_STORE_TYPE=mongodb \
           -e MONGODB_DATABASE_NAME=myclaims-QA \
           -e MONGODB_CONNECTION_STRING="mongodb+srv://..." \
           myclaims-api:latest
```

**What gets loaded:**
- ✅ `config.py`: From environment variables passed to `docker run`
- ✅ `domain_config.json`: From `/api/config/domain_config.json` (copied during build)
- ❌ `.env`: Not used (not in image)

### Scenario 3: Google Cloud Run / GKE

```bash
gcloud run deploy myclaims-api \
  --image gcr.io/project/myclaims-api:latest \
  --set-env-vars="PERSISTENCE_STORE_TYPE=mongodb,MONGODB_DATABASE_NAME=myclaims-QA" \
  --set-secrets="MONGODB_CONNECTION_STRING=mongodb-conn:latest"
```

**What gets loaded:**
- ✅ `config.py`: From environment variables set in Cloud Run
- ✅ `domain_config.json`: From code directory in container
- ❌ `.env`: Not used

## Configuration File Locations in Deployment

### In Docker Container:
```
/api/
├── config/
│   ├── config.py          # Python code (always there)
│   └── domain_config.json # JSON file (always there, read at runtime)
├── nodes/
│   └── context.py         # Reads domain_config.json from ../config/
├── .env                   # ❌ Should NOT exist in production
└── main.py
```

### How Each is Loaded:

1. **`config.py`**:
   - ✅ Always part of code
   - ✅ Reads from `os.environ` (environment variables)
   - ✅ Falls back to `.env` if exists (but shouldn't in prod)
   - ✅ Falls back to defaults in class

2. **`domain_config.json`**:
   - ✅ Always part of code (copied during build)
   - ✅ Read from hardcoded path: `{app_dir}/config/domain_config.json`
   - ⚠️ Same file for all environments (unless you mount different one)

3. **`.env`**:
   - ❌ Should NOT be in production
   - ❌ Should NOT be in Docker image
   - ✅ Only for local development

## Making `domain_config.json` Environment-Specific

Currently, `domain_config.json` is hardcoded. If you need environment-specific configs:

### Option 1: Environment Variable Override (Recommended)
```python
# In nodes/context.py
def _load_config() -> Dict[str, Any]:
    import os
    config_path = os.getenv(
        "DOMAIN_CONFIG_PATH",
        Path(__file__).parent.parent / "config" / "domain_config.json"
    )
    # ... load from config_path
```

Then in deployment:
```yaml
env:
  - name: DOMAIN_CONFIG_PATH
    value: "/config/domain_config.qa.json"  # Mounted via ConfigMap
```

### Option 2: ConfigMap (Kubernetes)
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: domain-config-qa
data:
  domain_config.json: |
    {
      "confidence_threshold": 0.6,
      ...
    }
---
# Mount in deployment
volumeMounts:
- name: domain-config
  mountPath: /api/config/domain_config.json
  subPath: domain_config.json
volumes:
- name: domain-config
  configMap:
    name: domain-config-qa
```

## Summary

When deployed to **dev** or **qa**:

| Configuration Source | Location | How It's Loaded |
|---------------------|----------|-----------------|
| `config.py` | Part of code | Reads from **environment variables** (set by deployment platform) |
| `domain_config.json` | Part of code | Read from **hardcoded path** in container (`/api/config/domain_config.json`) |
| `.env` | ❌ Not used | Should **NOT** exist in production deployments |

**Key Points:**
- ✅ `config.py` values come from **environment variables** in deployment
- ✅ `domain_config.json` comes from **deployed code** (same for all environments unless you mount different one)
- ❌ `.env` is **NOT used** in production (use environment variables instead)

