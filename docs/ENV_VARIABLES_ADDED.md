# Environment Variables Added to .env

## Summary

Added the recommended environment-specific configuration variables to `.env` file. These values now override the defaults in `config.py`.

## Variables Added to .env

The following environment variables were added to your `.env` file:

```bash
# GCP/Cloud Configuration
PROJECT_ID=pbm-nonprod-myclaims
LOCATION=us-central1

# Environment Identification
ENVIRONMENT=development
DEBUG=true

# External API Configuration
SWAGGER_URL=https://claiminquiry-exp-qa.myclaims.pss-np.caremark.com
```

## Already Present in .env

These were already in your `.env` file:
- `PERSISTENCE_STORE_TYPE=mongodb`
- `MONGODB_CONNECTION_STRING=mongodb+srv://...`
- `MONGODB_DATABASE_NAME=myclaims-DEV`

## Changes Made

### 1. Updated `config.py` Comments

Added `⚠️ Overridden by ENV_VAR` comments to indicate which values are overridden:
- `project_id` - Overridden by `PROJECT_ID` env var
- `location` - Overridden by `LOCATION` env var
- `swagger_url` - Overridden by `SWAGGER_URL` env var
- `environment` - Overridden by `ENVIRONMENT` env var
- `debug` - Overridden by `DEBUG` env var
- All API keys/secrets - Overridden by respective env vars

### 2. Created Kubernetes Example

Created `k8s/configmap-example.yaml` showing how to set these in Kubernetes deployments.

## Verification

All values are now being read from `.env` file:

```bash
✅ PROJECT_ID: pbm-nonprod-myclaims (from .env)
✅ SWAGGER_URL: https://claiminquiry-exp-qa.myclaims.pss-np.caremark.com (from .env)
✅ ENVIRONMENT: development (from .env)
✅ DEBUG: True (from .env)
✅ PERSISTENCE_STORE_TYPE: mongodb (from .env)
✅ MONGODB_DATABASE_NAME: myclaims-DEV (from .env)
```

## Next Steps for Deployment

### For QA Environment:

1. **Create Kubernetes ConfigMap:**
   ```yaml
   apiVersion: v1
   kind: ConfigMap
   metadata:
     name: myclaims-config-qa
   data:
     PROJECT_ID: "pbm-nonprod-myclaims"
     LOCATION: "us-central1"
     ENVIRONMENT: "qa"
     DEBUG: "false"
     SWAGGER_URL: "https://claiminquiry-exp-qa.myclaims.pss-np.caremark.com"
     PERSISTENCE_STORE_TYPE: "mongodb"
     MONGODB_DATABASE_NAME: "myclaims-QA"
   ```

2. **Create Kubernetes Secret:**
   ```yaml
   apiVersion: v1
   kind: Secret
   metadata:
     name: myclaims-secrets-qa
   type: Opaque
   stringData:
     MONGODB_CONNECTION_STRING: "mongodb+srv://myclaims_qa:...@..."
     OPENAI_API_KEY: "your-qa-key"
     AZURE_OPENAI_KEY: "your-qa-key"
   ```

3. **Reference in Deployment:**
   ```yaml
   envFrom:
     - configMapRef:
         name: myclaims-config-qa
     - secretRef:
         name: myclaims-secrets-qa
   ```

## Code Impact

**Zero code changes** - All existing code continues to work:
- `settings.project_id` → Reads from `PROJECT_ID` env var
- `settings.swagger_url` → Reads from `SWAGGER_URL` env var
- `settings.environment` → Reads from `ENVIRONMENT` env var
- `settings.debug` → Reads from `DEBUG` env var

The code access pattern remains exactly the same!

