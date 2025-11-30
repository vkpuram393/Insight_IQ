# Configuration Migration Guide

## Overview

This guide identifies which configuration values should be moved from `config.py` and `domain_config.json` to environment variables (`.env` for local, deployment YAML for upper environments).

## Analysis: What Should Be Environment-Specific?

### ✅ **MUST be Environment-Specific** (Move to .env/Deployment)

These values differ per environment and should NOT be hardcoded:

#### 1. Database/Connection Settings
- `PERSISTENCE_STORE_TYPE` - Different per environment (sqlite for dev, mongodb for qa/prod)
- `MONGODB_CONNECTION_STRING` - Different credentials per environment
- `MONGODB_DATABASE_NAME` - Different databases (myclaims-DEV, myclaims-QA, myClaims-UAT, myClaims-PT)
- `TELEMETRY_DB_PATH` - Only for SQLite (dev)
- `CHECKPOINT_DB_PATH` - Only for SQLite (dev)

#### 2. GCP/Cloud Settings
- `PROJECT_ID` - Different per environment (pbm-nonprod-myclaims, pbm-prod-myclaims, etc.)
- `LOCATION` - May differ (us-central1, us-east1, etc.)

#### 3. API Keys & Secrets (CRITICAL - Never commit!)
- `OPENAI_API_KEY` - Different per environment
- `AZURE_OPENAI_KEY` - Different per environment
- `AZURE_OPENAI_ENDPOINT` - Different per environment
- `AZURE_TENANT_ID`, `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET` - Different per environment
- `MONGODB_CONNECTION_STRING` - Contains passwords (SECRET!)
- `LANGSMITH_API_KEY` - Optional, but different per environment
- `LANGFUSE_SECRET_KEY`, `LANGFUSE_PUBLIC_KEY` - Optional, different per environment

#### 4. External Service URLs
- `SWAGGER_URL` - Different per environment (QA vs UAT vs Prod)
  - Current: `https://claiminquiry-exp-qa.myclaims.pss-np.caremark.com`
  - Should be: Different for each environment

#### 5. Environment Identification
- `ENVIRONMENT` - Should explicitly be "development", "qa", "uat", "production"
- `DEBUG` - Should be `false` in upper environments, `true` in dev

#### 6. Optional: Feature Flags (if they differ per environment)
- `ENABLE_TELEMETRY` - May want `false` in dev, `true` in prod
- `ENABLE_CHECKPOINTING` - May differ per environment

### ⚠️ **SHOULD be Environment-Specific** (Consider moving)

These might differ per environment but are currently hardcoded:

#### From `config.py`:
- `LLM_MODEL` - Might use different models per environment (dev: gemini-2.5-flash, prod: gemini-2.0-flash-exp)
- `LLM_TEMPERATURE` - Might want different values (dev: 0.1, prod: 0.7)
- `USE_GOOGLE_EMBEDDINGS` - Might differ per environment

#### From `domain_config.json`:
- `confidence_threshold` - Might want different thresholds per environment
- `conversation_history_window` - Might differ per environment

### ❌ **SHOULD Stay in Code** (Don't move)

These are application logic/business rules, not environment-specific:

#### From `config.py`:
- `CONFIDENCE_THRESHOLD` - Business logic (though overridden by domain_config.json)
- `CONVERSATION_HISTORY_LIMIT` - Business logic
- `USE_CVS_INTENT_CLASSIFIER` - Feature flag (same across environments)
- `USE_EMBEDDING_CLASSIFIER` - Feature flag (same across environments)
- `ENABLE_SAFETY_PRECHECK` - Security feature (same across environments)
- `ENABLE_SAFETY_POSTCHECK` - Security feature (same across environments)
- `ENABLE_SEMANTIC_CACHE` - Feature flag (same across environments)
- `REMOVE_PUNCTUATION_IN_NORMALIZATION` - Business logic
- `ENABLE_STREAMING` - Feature flag (same across environments)
- `STREAMING_CHUNK_SIZE` - Application behavior
- `STREAMING_DELAY_MS` - Application behavior
- `STREAM_NODE_UPDATES` - Application behavior
- `STREAM_USER_FACING_NODES` - Application behavior (list)

#### From `domain_config.json`:
- `clarification_messages` - Business logic (user-facing messages)
- `required_slots_by_intent` - Business logic (domain rules)

## Recommended Migration

### Step 1: Move Critical Environment-Specific Values

#### For `.env` (Local Development):
```bash
# .env.dev
# Database
PERSISTENCE_STORE_TYPE=sqlite
TELEMETRY_DB_PATH=data/telemetry.db
CHECKPOINT_DB_PATH=checkpoints.db

# GCP
PROJECT_ID=pbm-nonprod-myclaims
LOCATION=us-central1

# Environment
ENVIRONMENT=development
DEBUG=true

# External APIs
SWAGGER_URL=https://claiminquiry-exp-qa.myclaims.pss-np.caremark.com

# Secrets (never commit!)
OPENAI_API_KEY=your-dev-key
AZURE_OPENAI_KEY=your-dev-key
AZURE_OPENAI_ENDPOINT=your-dev-endpoint
MONGODB_CONNECTION_STRING=mongodb+srv://...  # If testing MongoDB locally
```

```bash
# .env.qa
# Database
PERSISTENCE_STORE_TYPE=mongodb
MONGODB_DATABASE_NAME=myclaims-QA
MONGODB_CONNECTION_STRING=mongodb+srv://myclaims_qa:...@...

# GCP
PROJECT_ID=pbm-nonprod-myclaims
LOCATION=us-central1

# Environment
ENVIRONMENT=qa
DEBUG=false

# External APIs
SWAGGER_URL=https://claiminquiry-exp-qa.myclaims.pss-np.caremark.com

# Secrets
OPENAI_API_KEY=your-qa-key
AZURE_OPENAI_KEY=your-qa-key
AZURE_OPENAI_ENDPOINT=your-qa-endpoint
```

#### For Deployment YAML (Kubernetes Example):
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: myclaims-config-qa
  namespace: myclaims-qa
data:
  # Database
  PERSISTENCE_STORE_TYPE: "mongodb"
  MONGODB_DATABASE_NAME: "myclaims-QA"
  
  # GCP
  PROJECT_ID: "pbm-nonprod-myclaims"
  LOCATION: "us-central1"
  
  # Environment
  ENVIRONMENT: "qa"
  DEBUG: "false"
  
  # External APIs
  SWAGGER_URL: "https://claiminquiry-exp-qa.myclaims.pss-np.caremark.com"
  
  # Optional: Feature flags
  ENABLE_TELEMETRY: "true"
  ENABLE_CHECKPOINTING: "true"
---
apiVersion: v1
kind: Secret
metadata:
  name: myclaims-secrets-qa
  namespace: myclaims-qa
type: Opaque
stringData:
  # Secrets (never in ConfigMap!)
  MONGODB_CONNECTION_STRING: "mongodb+srv://myclaims_qa:...@..."
  OPENAI_API_KEY: "your-qa-key"
  AZURE_OPENAI_KEY: "your-qa-key"
  AZURE_OPENAI_ENDPOINT: "your-qa-endpoint"
  AZURE_TENANT_ID: "your-tenant-id"
  AZURE_CLIENT_ID: "your-client-id"
  AZURE_CLIENT_SECRET: "your-client-secret"
```

### Step 2: Update `config.py` Defaults

Keep defaults as fallbacks, but make it clear they're overridden:

```python
# config.py
project_id: str = "pbm-nonprod-myclaims"  # ⚠️ Overridden by PROJECT_ID env var
location: str = "us-central1"  # ⚠️ Overridden by LOCATION env var
swagger_url: str = "https://claiminquiry-exp-qa.myclaims.pss-np.caremark.com"  # ⚠️ Overridden by SWAGGER_URL env var
environment: str = "development"  # ⚠️ Overridden by ENVIRONMENT env var
debug: bool = True  # ⚠️ Overridden by DEBUG env var
```

### Step 3: Optional - Make `domain_config.json` Environment-Aware

If you need different thresholds per environment:

```python
# In nodes/context.py
def _load_config() -> Dict[str, Any]:
    import os
    from pathlib import Path
    
    # Try environment-specific config first
    env = os.getenv("ENVIRONMENT", "development")
    env_config_path = Path(__file__).parent.parent / "config" / f"domain_config.{env}.json"
    
    if env_config_path.exists():
        config_path = env_config_path
    else:
        # Fallback to default
        config_path = Path(__file__).parent.parent / "config" / "domain_config.json"
    
    # ... load from config_path
```

Then create:
- `config/domain_config.development.json`
- `config/domain_config.qa.json`
- `config/domain_config.uat.json`
- `config/domain_config.production.json`

## Summary: What to Move

### ✅ **MUST Move to .env/Deployment:**

| Config Value | Current Location | Move To |
|-------------|------------------|---------|
| `PERSISTENCE_STORE_TYPE` | config.py default | .env / Deployment |
| `MONGODB_CONNECTION_STRING` | config.py default | .env / Secrets |
| `MONGODB_DATABASE_NAME` | config.py default | .env / Deployment |
| `PROJECT_ID` | config.py hardcoded | .env / Deployment |
| `SWAGGER_URL` | config.py hardcoded | .env / Deployment |
| `ENVIRONMENT` | config.py default | .env / Deployment |
| `DEBUG` | config.py default | .env / Deployment |
| All API keys/secrets | config.py defaults | .env / Secrets |

### ⚠️ **Consider Moving (Optional):**

| Config Value | Current Location | Consider Moving If |
|-------------|------------------|-------------------|
| `LLM_MODEL` | config.py | Different models per environment |
| `LLM_TEMPERATURE` | config.py | Different values per environment |
| `USE_GOOGLE_EMBEDDINGS` | config.py | Different providers per environment |
| `confidence_threshold` | domain_config.json | Different thresholds per environment |
| `conversation_history_window` | domain_config.json | Different windows per environment |

### ❌ **Keep in Code:**

- Business logic settings (confidence thresholds, conversation limits)
- Feature flags that are the same across environments
- Application behavior settings (streaming config, node lists)
- Domain rules (required slots, clarification messages)

## Priority Order

1. **CRITICAL (Do First):**
   - Move all secrets to Secrets/Secrets Manager
   - Move database connection strings
   - Move environment identification (ENVIRONMENT, DEBUG)

2. **HIGH (Do Second):**
   - Move PROJECT_ID, LOCATION
   - Move SWAGGER_URL
   - Move MONGODB_DATABASE_NAME

3. **MEDIUM (Do Third):**
   - Consider moving LLM_MODEL, LLM_TEMPERATURE if they differ
   - Consider making domain_config.json environment-aware

4. **LOW (Optional):**
   - Keep business logic in code unless it truly differs per environment

