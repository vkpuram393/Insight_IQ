# How Configuration Loading Works

## Answer: **Pydantic Settings Library** (Not Your Code!)

The configuration loading order is handled by **Pydantic Settings** (the `BaseSettings` class), not your application code.

## Where It Happens

### In `config/config.py`:

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    project_id: str = "pbm-nonprod-myclaims"  # Default (fallback)
    # ... other fields

    class Config:
        env_file = ".env"  # Tells Pydantic to read .env file

# This line triggers Pydantic's automatic loading:
settings = Settings()  # ← Pydantic does the magic here!
```

## Loading Order (Automatic by Pydantic)

When `Settings()` is instantiated, **Pydantic automatically** checks in this order:

### 1. **Environment Variables** (HIGHEST PRIORITY)
   - Checks `os.environ` (system environment variables)
   - Example: `PROJECT_ID=pbm-nonprod-myclaims`
   - **How set:**
     - **Local:** `export PROJECT_ID=value` or in shell
     - **Production:** Kubernetes ConfigMap/Secrets, Docker `-e` flags, Cloud Run env vars

### 2. **`.env` File** (SECOND PRIORITY)
   - Only if `env_file = ".env"` is set in `Config` class
   - Only if `.env` file exists in working directory
   - **Local development only** - should NOT exist in production containers

### 3. **Class Defaults** (LOWEST PRIORITY - FALLBACK)
   - Hardcoded values in `Settings` class
   - Used only if env var and `.env` are not set

## How It Works in Different Environments

### Local Development:

```python
# config.py
settings = Settings()  # Pydantic checks:
# 1. os.environ['PROJECT_ID'] → Not set
# 2. .env file → Finds PROJECT_ID=pbm-nonprod-myclaims ✅
# 3. Class default → Not needed (already found in .env)
```

**Result:** Uses value from `.env` file

### Production/QA/UAT (Kubernetes):

```python
# config.py (same code!)
settings = Settings()  # Pydantic checks:
# 1. os.environ['PROJECT_ID'] → Set by Kubernetes ConfigMap ✅
# 2. .env file → File doesn't exist in container (or ignored)
# 3. Class default → Not needed (already found in env var)
```

**Result:** Uses value from environment variable (set by Kubernetes)

## Key Points

### ✅ **It's Automatic**
- Your code doesn't need to check anything
- Pydantic handles it all when `Settings()` is called
- Happens **once at startup** (when module is imported)

### ✅ **Same Code Everywhere**
- `config.py` is identical in local and production
- Only the **source of values** changes:
  - **Local:** `.env` file
  - **Production:** Environment variables (Kubernetes, Docker, etc.)

### ✅ **No Code Changes Needed**
- All your code just does: `settings.project_id`
- Pydantic ensures it gets the right value automatically

## Example Flow

### Local Development:

```bash
# .env file exists
PROJECT_ID=pbm-nonprod-myclaims
```

```python
# config.py (line 142)
settings = Settings()  # Pydantic reads from .env
```

```python
# Any other file
from config.config import settings
print(settings.project_id)  # → "pbm-nonprod-myclaims" (from .env)
```

### Production (Kubernetes):

```yaml
# deployment.yaml
env:
  - name: PROJECT_ID
    value: "pbm-prod-myclaims"
```

```python
# config.py (same code!)
settings = Settings()  # Pydantic reads from env var
```

```python
# Any other file (same code!)
from config.config import settings
print(settings.project_id)  # → "pbm-prod-myclaims" (from env var)
```

## Verification

You can see this in action:

```python
import os
from config.config import settings

# Check what Pydantic actually loaded:
print(f"PROJECT_ID: {settings.project_id}")

# Check if it came from env var:
if 'PROJECT_ID' in os.environ:
    print("✅ Loaded from environment variable")
elif os.path.exists('.env'):
    print("✅ Loaded from .env file")
else:
    print("✅ Using class default")
```

## Summary

| Question | Answer |
|----------|--------|
| **Who does the loading?** | **Pydantic Settings library** (not your code) |
| **When does it happen?** | Once at startup when `settings = Settings()` is called |
| **Where in code?** | `config/config.py` line 142 |
| **Local environment?** | Reads from `.env` file (if exists) |
| **Production environment?** | Reads from environment variables (Kubernetes, Docker, etc.) |
| **Does your code need to change?** | ❌ **NO** - Same code everywhere! |

The magic is in **Pydantic Settings** - it's a library feature that automatically handles the priority order for you!

