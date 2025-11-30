# Hot Reload Guide

## Current Status: **Hot Reload is DISABLED by Default** ❌

By default, the application runs **without hot reload** (`reload=False` in `main.py`). This means you need to **manually restart** the server for code changes to take effect.

## How to Enable Hot Reload

### Option 1: Use Start Script with Environment Variable (Recommended)

**Linux/Mac:**
```bash
RELOAD=true ./start_server.sh
```

**Windows (PowerShell):**
```powershell
$env:RELOAD="true"; .\start_server_windows.ps1
```

### Option 2: Run Uvicorn Directly with --reload

```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

### Option 3: Modify main.py (Not Recommended)

Change line 125 in `main.py`:
```python
# Change from:
uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False, log_level="debug")

# To:
uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True, log_level="debug")
```

**⚠️ Note:** The comment says "Run WITHOUT reload for reliable breakpoints" - hot reload can interfere with debugging.

## What Hot Reload Does

When enabled, Uvicorn watches for file changes and **automatically restarts** the server when:
- Python files (`.py`) are modified
- Files in watched directories change

**You'll see:**
```
INFO:     Detected file change in 'config/validation.py'. Reloading...
INFO:     Application startup complete.
```

## What Requires Restart (Even with Hot Reload)

### ✅ **Hot Reload Handles These:**
- Changes to Python code (`.py` files)
- Route definitions
- Node logic
- Configuration changes in Python files

### ❌ **These ALWAYS Require Manual Restart:**

1. **Environment Variables (`.env` file)**
   - Changes to `.env` file are **NOT** detected
   - Must restart to pick up new environment variables
   - **Why:** Pydantic Settings loads `.env` at startup only

2. **Configuration in `config.py` (default values)**
   - Changes to default values in `Settings` class
   - **Why:** Settings are instantiated once at startup

3. **Dependencies (`requirements.txt`)**
   - Installing new packages
   - **Why:** Python imports are cached

4. **Database Schema Changes**
   - SQLite database structure
   - MongoDB collection structure
   - **Why:** Connections are established at startup

5. **External Configuration Files**
   - `domain_config.json` changes
   - **Why:** Files are loaded/cached at startup

6. **Startup Code Changes**
   - Changes to `main.py` startup hooks
   - Changes to `@app.on_event("startup")` handlers
   - **Why:** These run once at server start

## Current Configuration

### Default (No Hot Reload):
```python
# main.py line 125
uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False, log_level="debug")
```

### With Hot Reload (via script):
```bash
RELOAD=true ./start_server.sh
# Uses: uvicorn main:app --host "$HOST" --port "$PORT" --reload
```

## When to Use Hot Reload

### ✅ **Use Hot Reload When:**
- Actively developing Python code
- Making frequent code changes
- Testing route changes
- Not debugging (no breakpoints)

### ❌ **Don't Use Hot Reload When:**
- Debugging with breakpoints (can cause issues)
- Testing configuration changes (won't pick up `.env` changes)
- Making startup/initialization changes
- Need stable environment for testing

## Quick Reference

| Change Type | Hot Reload Works? | Manual Restart Needed? |
|-------------|-------------------|------------------------|
| Python code (routes, nodes, etc.) | ✅ Yes | ❌ No |
| `.env` file | ❌ No | ✅ Yes |
| `config.py` defaults | ❌ No | ✅ Yes |
| `domain_config.json` | ❌ No | ✅ Yes |
| New dependencies | ❌ No | ✅ Yes |
| Database schema | ❌ No | ✅ Yes |
| Startup hooks | ❌ No | ✅ Yes |

## Troubleshooting

### Hot Reload Not Working?
1. **Check if enabled:** Look for `--reload` flag in process
2. **Check file permissions:** Uvicorn needs read access
3. **Check file paths:** Must be in watched directories
4. **Restart manually:** Sometimes hot reload gets stuck

### "Reload loop" or "Infinite reloads"?
- Disable hot reload: `RELOAD=false ./start_server.sh`
- Check for syntax errors causing reload failures
- Check file watcher limits (too many files)

### Changes Not Picking Up?
- **Environment variables:** Always requires restart
- **Configuration files:** Always requires restart
- **Python code:** Should work with hot reload (if enabled)

## Best Practice

**For Development:**
```bash
# Enable hot reload for active development
RELOAD=true ./start_server.sh
```

**For Testing/Stable Environment:**
```bash
# Disable hot reload for stable testing
./start_server.sh  # or RELOAD=false ./start_server.sh
```

**For Configuration Changes:**
```bash
# Always restart manually after .env or config.py changes
# Hot reload won't help here
```

## Summary

- **Default:** Hot reload is **OFF** (`reload=False`)
- **Enable:** Set `RELOAD=true` environment variable
- **Python code:** Hot reload works (if enabled)
- **Configuration:** Always requires manual restart
- **Best for:** Active development, not debugging

