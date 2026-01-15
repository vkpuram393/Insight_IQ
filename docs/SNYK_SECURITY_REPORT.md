# Snyk Security Report

**Scan Date:** 2026-01-15  
**Branch:** `Kunwar-rollback`  
**Scanned Directory:** `pss-myclaims-ai-agent`

---

## Summary

| Scan Type | Issues Found | Status |
|-----------|--------------|--------|
| **Code Scan (SAST)** | 1 | ⚠️ Pre-existing (not from new code) |
| **Dependency Scan (SCA)** | 0 | ✅ No vulnerabilities |

---

## Code Scan Results (SAST)

### Issue 1: Origin Validation Error (CORS)

| Field | Value |
|-------|-------|
| **Severity** | Medium |
| **File** | `api/routes.py` |
| **Line** | 286 |
| **CWEs** | CWE-942, CWE-346 |
| **Title** | Too Permissive CORS |

**Description:**  
CORS policy `"*"` might be too permissive. This allows malicious code on other origins to make requests to this server.

**Note:** This is a **PRE-EXISTING issue** in `api/routes.py`, NOT related to the memory diagnostics changes.

**Location:** `api/routes.py:285-286`
```python
allow_origins=["*"]
```

**Recommendation:**  
Consider restricting CORS origins to specific trusted domains in production, or accept the risk for internal/development APIs.

---

## Dependency Scan Results (SCA)

### psutil==5.9.8

| Field | Value |
|-------|-------|
| **Package** | psutil |
| **Version** | 5.9.8 |
| **Vulnerabilities** | ✅ **None found** |
| **License** | BSD-3-Clause |

**Conclusion:** The `psutil` dependency is secure with no known vulnerabilities.

---

## New Code Analysis

The memory diagnostics endpoints added to `main.py` were scanned. **No security issues were found** in the new code.

### Endpoints Added:
- `GET /debug/memory` - Read-only, no security concerns
- `GET /debug/memory/detailed` - Read-only, no security concerns  
- `POST /debug/memory/reset-baseline` - Internal state only, no security concerns
- `POST /debug/memory/gc` - Triggers GC, no security concerns

### Security Considerations for Debug Endpoints:

| Consideration | Status |
|---------------|--------|
| No sensitive data exposed | ✅ Only memory metrics |
| No authentication bypass | ✅ N/A (diagnostics only) |
| No injection vulnerabilities | ✅ No user input processed |
| No file system access | ✅ Only reads `/proc/self/status` (read-only) |

---

## Conclusion

✅ **The memory diagnostics implementation passes security review.**

- No new vulnerabilities introduced
- `psutil` dependency has no known CVEs
- The only issue found (CORS) is pre-existing and unrelated to this change

---

## Scan Commands Used

```bash
# Code scan (SAST)
snyk code test --severity-threshold=medium

# Dependency scan (SCA)  
snyk test --severity-threshold=medium --skip-unresolved
```

