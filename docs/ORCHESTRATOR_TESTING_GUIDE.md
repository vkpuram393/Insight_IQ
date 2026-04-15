# Orchestrator Testing Guide

## Prerequisites

### Install pytest (Required)

Pytest is not in `requirements.txt`. Install it:

```bash
pip install pytest pytest-asyncio
```

Or add to `requirements.txt`:
```
pytest==8.3.4
pytest-asyncio==0.24.0
```

Then:
```bash
pip install -r requirements.txt
```

---

## Running Tests

### 1. Run All Orchestrator Tests (24 tests)

```bash
# PowerShell
cd c:\Users\C942964\Downloads\KHK-feature-Orchestration-PBM-AI\pss-myclaims-ai-agent
python -m pytest tests/test_orchestrator.py -v

# Or shorter:
pytest tests/test_orchestrator.py -v
```

**Expected output:**
```
tests/test_orchestrator.py::TestOrchestratorNormalization::test_basic_normalization PASSED
tests/test_orchestrator.py::TestOrchestratorNormalization::test_lowercase_conversion PASSED
... (24 tests total)
======================== 24 passed in X.XXs ========================
```

---

### 2. Run Orchestrator Error Model Tests (3 tests)

```bash
pytest tests/test_error_models.py::TestHelperFunctions::test_create_orchestrator_empty_input_error -v
pytest tests/test_error_models.py::TestHelperFunctions::test_create_orchestrator_invalid_type_error -v
pytest tests/test_error_models.py::TestHelperFunctions::test_create_orchestrator_normalization_error -v
```

**Or run all error model tests:**
```bash
pytest tests/test_error_models.py -v
```

---

### 3. Run Orchestrator Node Model Tests (7 tests)

```bash
pytest tests/test_node_models.py::TestOrchestratorResult -v
```

**Or run all node model tests:**
```bash
pytest tests/test_node_models.py -v
```

---

### 4. Run ALL Tests (All 34 orchestrator tests + existing tests)

```bash
pytest tests/ -v
```

---

### 5. Run Tests with Coverage

```bash
pip install pytest-cov

pytest tests/test_orchestrator.py --cov=nodes.orchestrator --cov-report=term-missing -v
```

---

## Test Organization

### Total Tests Created: 34

#### 1. Integration Tests (`test_orchestrator.py`) - 24 tests
```
TestOrchestratorNormalization (6 tests)
├── test_basic_normalization
├── test_lowercase_conversion
├── test_whitespace_handling
├── test_punctuation_removal
├── test_original_text_preserved
└── test_unicode_normalization

TestOrchestratorErrorHandling (3 tests)
├── test_empty_input_error
├── test_invalid_type_handling
└── test_graceful_fallback_on_exception

TestOrchestratorStateIntegration (3 tests)
├── test_output_compatible_with_downstream_nodes
├── test_metadata_structure
└── test_existing_metadata_preserved

TestOrchestratorNormalizationSteps (2 tests)
├── test_unicode_normalization
└── test_normalization_steps_tracking

TestOrchestratorTelemetry (1 test)
└── test_error_logged_on_empty_input

TestOrchestratorRealWorldScenarios (3 tests)
├── test_typical_claim_query
├── test_messy_input_with_extra_spaces
└── test_special_characters_handling
```

#### 2. Error Model Tests (`test_error_models.py`) - 3 tests
```
TestHelperFunctions (added to existing class)
├── test_create_orchestrator_empty_input_error
├── test_create_orchestrator_invalid_type_error
└── test_create_orchestrator_normalization_error
```

#### 3. Node Model Tests (`test_node_models.py`) - 7 tests
```
TestOrchestratorResult (new class)
├── test_create_basic_orchestrator_result
├── test_orchestrator_result_with_normalization_steps
├── test_orchestrator_result_with_error
├── test_orchestrator_result_serialization
├── test_create_orchestrator_result_helper
└── test_orchestrator_result_with_all_fields
```

---

## Quick Test Commands

### Run only orchestrator-related tests:
```bash
# All 34 orchestrator tests
pytest tests/test_orchestrator.py tests/test_error_models.py::TestHelperFunctions::test_create_orchestrator_empty_input_error tests/test_error_models.py::TestHelperFunctions::test_create_orchestrator_invalid_type_error tests/test_error_models.py::TestHelperFunctions::test_create_orchestrator_normalization_error tests/test_node_models.py::TestOrchestratorResult -v

# Simpler: Run specific files
pytest tests/test_orchestrator.py -v
pytest tests/test_error_models.py -k orchestrator -v
pytest tests/test_node_models.py -k orchestrator -v
```

### Run with different verbosity:
```bash
# Minimal output
pytest tests/test_orchestrator.py

# Verbose (test names)
pytest tests/test_orchestrator.py -v

# Very verbose (full output)
pytest tests/test_orchestrator.py -vv

# Show print statements
pytest tests/test_orchestrator.py -v -s
```

### Run specific test:
```bash
pytest tests/test_orchestrator.py::TestOrchestratorNormalization::test_basic_normalization -v
```

### Run tests matching pattern:
```bash
# All error-related tests
pytest tests/test_orchestrator.py -k error -v

# All normalization tests
pytest tests/test_orchestrator.py -k normalization -v
```

---

## Troubleshooting

### Error: "No module named pytest"
**Solution:**
```bash
pip install pytest pytest-asyncio
```

### Error: "No module named nodes.orchestrator"
**Solution:** Run from project root:
```bash
cd c:\Users\C942964\Downloads\KHK-feature-Orchestration-PBM-AI\pss-myclaims-ai-agent
pytest tests/test_orchestrator.py -v
```

### Error: Import errors in tests
**Solution:** Ensure all dependencies installed:
```bash
pip install -r requirements.txt
pip install pytest pytest-asyncio
```

### Tests fail due to missing config
**Solution:** Ensure `.env` file exists with required settings:
```
REMOVE_PUNCTUATION_IN_NORMALIZATION=true
ENABLE_TELEMETRY=true
```

---

## Continuous Integration

Add to CI/CD pipeline:

```yaml
# Example GitHub Actions
- name: Run Orchestrator Tests
  run: |
    pip install pytest pytest-asyncio
    pytest tests/test_orchestrator.py -v --tb=short
    pytest tests/test_error_models.py -k orchestrator -v
    pytest tests/test_node_models.py -k orchestrator -v
```

---

## Test Coverage Report

Generate HTML coverage report:

```bash
pip install pytest-cov
pytest tests/test_orchestrator.py --cov=nodes.orchestrator --cov-report=html
```

Open `htmlcov/index.html` in browser.

---

## Summary

**Quick Start:**
1. Install: `pip install pytest pytest-asyncio`
2. Run: `pytest tests/test_orchestrator.py -v`
3. Expected: 24 tests passing

**Full Test Suite:**
```bash
pytest tests/ -v  # All tests including orchestrator
```

**Orchestrator-Only:**
```bash
pytest tests/test_orchestrator.py -v                    # 24 tests
pytest tests/test_error_models.py -k orchestrator -v    # 3 tests
pytest tests/test_node_models.py -k orchestrator -v     # 7 tests
```

**Total: 34 orchestrator tests**

---

## Files Modified/Created Summary

### Created Files:
1. ✅ `tests/test_orchestrator.py` - 24 integration tests
2. ✅ `TESTING_GUIDE.md` - This file

### Modified Files:
1. ✅ `tests/test_error_models.py` - Added 3 tests
2. ✅ `tests/test_node_models.py` - Added 7 tests
3. ✅ `core/node_models.py` - Added OrchestratorResult
4. ✅ `core/error_models.py` - Added 3 error helpers
5. ✅ `nodes/orchestrator.py` - Integrated structured models

---

**Status**: Ready to test! 🧪

