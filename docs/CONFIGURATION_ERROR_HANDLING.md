# Configuration Error Handling

## Overview

The application now has **startup validation** that catches configuration errors **before** the application starts. This prevents runtime failures from wrong settings (e.g., QA environment picking up local `.env` values).

## What Gets Validated

### 1. **Critical Settings** (Must Pass - App Won't Start if Failed)
- ✅ `PERSISTENCE_STORE_TYPE` is valid (sqlite, mongodb, firestore, bigquery)
- ✅ `MONGODB_CONNECTION_STRING` is set (if using MongoDB)
- ✅ `MONGODB_DATABASE_NAME` is set (if using MongoDB)
- ✅ `PROJECT_ID` is set
- ✅ `SWAGGER_URL` is set

### 2. **Environment-Specific Validation** (Errors + Warnings)
- ✅ Environment value is valid (development, dev, qa, uat, production, prod)
- ✅ `PROJECT_ID` matches environment (e.g., QA shouldn't use dev project)
- ✅ `MONGODB_DATABASE_NAME` matches environment (e.g., QA shouldn't use dev database)
- ✅ `SWAGGER_URL` matches environment (e.g., QA shouldn't point to dev URL)
- ⚠️  `DEBUG` flag warning in upper environments
- ⚠️  `.env` file warning in upper environments (should use env vars only)

## How It Works

### At Startup (`main.py`):

```python
# Configuration Validation
from config.validation import validate_all

if not validate_all():
    print("❌ Configuration validation FAILED. Application cannot start.")
    sys.exit(1)  # App won't start if validation fails
```

### What Happens:

1. **Application starts loading**
2. **Configuration validation runs** (before app fully starts)
3. **If errors found:**
   - ❌ Logs detailed error messages
   - ❌ Application **exits immediately** (won't start)
   - ✅ Clear error messages show what's wrong
4. **If warnings only:**
   - ⚠️  Logs warnings
   - ✅ Application continues (warnings don't block startup)
5. **If all valid:**
   - ✅ Logs success message
   - ✅ Application starts normally

## Example Error Messages

### Scenario: QA Environment with Dev Settings

```
🚨 CONFIGURATION VALIDATION FAILED
================================================================================
❌ CONFIGURATION MISMATCH: Environment is 'qa' but PROJECT_ID contains 'dev': 'pbm-nonprod-myclaims'. 
This suggests local .env values are being used in upper environment!
❌ CONFIGURATION MISMATCH: Environment is 'qa' but MONGODB_DATABASE_NAME is 'myclaims-DEV' (looks like dev). 
This suggests wrong environment variables are set!
================================================================================
Application will fail to start. Please fix configuration errors above.
================================================================================
```

### Scenario: Missing Critical Setting

```
🚨 CRITICAL SETTINGS VALIDATION FAILED
================================================================================
❌ MONGODB_CONNECTION_STRING is not set or is using default localhost. 
Current value: 'mongodb://localhost:27017'
================================================================================
```

## Where Validation Runs

### File: `config/validation.py`
- `validate_critical_settings()` - Checks required settings
- `validate_environment_config()` - Checks environment-specific values
- `validate_all()` - Runs all validations

### File: `main.py` (Line ~33)
- Calls `validate_all()` at startup
- Exits if validation fails

## Exception Handling

### ✅ **Startup Validation** (NEW)
- **When:** Before application starts
- **What:** Validates all critical and environment-specific settings
- **Action:** Application won't start if critical errors found
- **Benefit:** Catches misconfigurations immediately

### ✅ **Runtime Error Handling** (Existing)
- **When:** During request processing
- **What:** Catches errors when settings are used
- **Action:** Logs to exceptions table, returns error to user
- **Benefit:** Graceful error handling during operation

### ✅ **Pydantic Validation** (Built-in)
- **When:** When `Settings()` is instantiated
- **What:** Validates data types (string, int, bool, etc.)
- **Action:** Raises `ValidationError` if type mismatch
- **Benefit:** Type safety

## Testing Configuration Validation

### Test with Wrong Settings:

```python
import os
os.environ['ENVIRONMENT'] = 'qa'
os.environ['MONGODB_DATABASE_NAME'] = 'myclaims-DEV'  # Wrong for QA!

from config.validation import validate_all
validate_all()  # Will show errors
```

### Test Locally:

```bash
# Set wrong environment
export ENVIRONMENT=qa
export MONGODB_DATABASE_NAME=myclaims-DEV

# Start app - validation will catch it
python main.py
```

## What Gets Logged

### Errors (Block Startup):
- Configuration mismatches (e.g., QA using dev database)
- Missing required settings
- Invalid values

### Warnings (Don't Block Startup):
- DEBUG enabled in upper environments
- `.env` file exists in production
- Minor mismatches (e.g., URL doesn't match environment exactly)

## Best Practices

### ✅ **Do:**
- Set environment variables in deployment (Kubernetes, Docker, etc.)
- Use `.env` file only for local development
- Review validation errors before deploying
- Fix all errors before starting application

### ❌ **Don't:**
- Include `.env` files in production containers
- Ignore validation warnings (they indicate potential issues)
- Override validation (it's there to protect you!)

## Summary

| Aspect | Details |
|--------|---------|
| **When validated** | At startup (before app starts) |
| **What's checked** | Critical settings + environment-specific values |
| **On failure** | Application exits with clear error messages |
| **On success** | Application starts normally |
| **Location** | `config/validation.py` + `main.py` |
| **Benefit** | Catches misconfigurations immediately, prevents runtime failures |

**You now have comprehensive exception handling to catch configuration issues clearly!** ✅

