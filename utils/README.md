# Utils - Component Testing Endpoints

This folder contains individual test endpoints for each component of the application.

## Purpose

Allow developers to test their specific functionality independently without running the entire application pipeline. Perfect for:
- Parallel development
- Unit testing
- Quick iterations
- Debugging specific components
- Integration testing

## Files

- `test_endpoints.py` - All test endpoint definitions
- `__init__.py` - Package initialization

## Usage

These endpoints are automatically loaded when you run `main.py`.

Access them at: `http://localhost:8000/utils/*`

## Documentation

See the root-level **`TEMP_ENDPOINTS.md`** for:
- Complete list of all endpoints
- cURL examples for each endpoint
- Testing workflows
- Developer-specific guides

## Quick Test

```bash
# Health check
curl http://localhost:8000/utils/health

# Test intent classifier
curl -X POST http://localhost:8000/utils/test-intent \
  -H 'Content-Type: application/json' \
  -d '{"text":"why was my claim rejected"}'
```

## API Documentation

When the app is running, visit:
```
http://localhost:8000/docs
```

All utils endpoints are under the "Testing Utils" tag.

