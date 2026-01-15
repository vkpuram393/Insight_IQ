# Memory Diagnostics Guide

This guide explains how to use the memory diagnostics endpoints to analyze POD memory usage and diagnose potential memory issues.

## Overview

The memory diagnostics endpoints provide **non-interfering, observability-only** access to memory metrics. They are designed to:

- ✅ **Never interfere** with normal agent operations
- ✅ **Work in production** POD environments
- ✅ **Use existing dependencies** (psutil is already in requirements.txt)
- ✅ **Provide actionable insights** for memory analysis

## Endpoints Reference

| Endpoint | Method | Interference | Purpose |
|----------|--------|--------------|---------|
| `/debug/memory` | GET | **NONE** | Quick memory snapshot |
| `/debug/memory/detailed` | GET | **MINIMAL** | Detailed heap analysis with tracemalloc |
| `/debug/memory/reset-baseline` | POST | **NONE** | Reset tracemalloc baseline |
| `/debug/memory/gc` | POST | **LOW** | Explicitly trigger garbage collection |

---

## Endpoint Details

### 1. GET /debug/memory

**Quick memory snapshot** - 100% read-only, zero interference.

**What it does:**
- Reads process RSS/VMS using psutil (what Grafana shows)
- Reads GC statistics without triggering collection
- Reads Linux `/proc/self/status` for detailed kernel metrics (in POD)

**When to use:**
- As a quick health check
- To see current memory state without any overhead
- Safe to call repeatedly in production

**Example Response:**
```json
{
  "timestamp": "2026-01-15T14:30:00.123456",
  "endpoint": "quick_snapshot",
  "interference_level": "none",
  "process_memory": {
    "rss_mb": 2210.45,
    "rss_bytes": 2317893632,
    "vms_mb": 3500.12,
    "percent": 28.5
  },
  "gc_stats": {
    "enabled": true,
    "thresholds": [700, 10, 10],
    "counts": {
      "gen0_pending": 245,
      "gen1_pending": 3,
      "gen2_pending": 0
    },
    "generation_stats": [
      {"generation": 0, "collections": 1523, "collected": 45678, "uncollectable": 0},
      {"generation": 1, "collections": 152, "collected": 12345, "uncollectable": 0},
      {"generation": 2, "collections": 15, "collected": 5678, "uncollectable": 0}
    ]
  },
  "linux_proc_status": {
    "VmSize": "3584000 kB",
    "VmRSS": "2263500 kB",
    "VmPeak": "2358000 kB",
    "VmHWM": "2263500 kB",
    "VmData": "2100000 kB",
    "VmStk": "136 kB"
  }
}
```

---

### 2. GET /debug/memory/detailed

**Detailed heap analysis** - Uses tracemalloc to track Python memory allocations.

**What it does:**
- First call: Starts tracemalloc and captures a baseline snapshot
- Subsequent calls: Shows memory growth since baseline and top allocators

**Overhead:**
- ~3-5% CPU overhead after activation
- Minimal memory for storing allocation traces

**When to use:**
- To identify which code files are consuming memory
- To track memory growth over time
- To diagnose suspected memory leaks

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `top_allocators` | int | 15 | Number of top memory consumers to show (1-50) |

**Example Response (first call):**
```json
{
  "timestamp": "2026-01-15T14:30:00.123456",
  "endpoint": "detailed_analysis",
  "interference_level": "minimal",
  "process_memory": {
    "rss_mb": 2210.45,
    "vms_mb": 3500.12,
    "percent": 28.5
  },
  "tracemalloc": {
    "status": "just_activated",
    "message": "Tracemalloc started and baseline captured. Call this endpoint again to see memory changes.",
    "overhead": "~3-5% CPU, minimal memory for storing traces"
  }
}
```

**Example Response (subsequent calls):**
```json
{
  "timestamp": "2026-01-15T14:35:00.123456",
  "endpoint": "detailed_analysis",
  "process_memory": {
    "rss_mb": 2250.12,
    "vms_mb": 3550.45,
    "percent": 29.1
  },
  "tracemalloc": {
    "status": "active",
    "current_heap_mb": 450.23,
    "peak_heap_mb": 520.10,
    "top_allocators": [
      {"location": "classifiers/embedded_classifier.py:95", "size_mb": 120.5, "count": 50},
      {"location": "services/pii_protection.py:278", "size_mb": 85.2, "count": 5000},
      {"location": "numpy/core/numeric.py:150", "size_mb": 45.0, "count": 1200}
    ]
  },
  "memory_growth_since_baseline": [
    {"location": "classifiers/embedded_classifier.py:95", "growth_mb": 98.5, "new_allocations": 49},
    {"location": "services/pii_protection.py:278", "growth_mb": 75.0, "new_allocations": 4500}
  ],
  "interpretation": {
    "rss_vs_heap_gap_mb": 1800.0,
    "diagnosis": "Large gap suggests memory held by native libraries or allocator fragmentation",
    "hint": "If RSS stays high but heap is low, memory is held by Python allocator (normal behavior)."
  }
}
```

---

### 3. POST /debug/memory/reset-baseline

**Reset tracemalloc baseline** - For fresh memory growth comparison.

**When to use:**
- After making code changes
- To start a new analysis session
- To reset the comparison point

**Example Response:**
```json
{
  "status": "success",
  "message": "Baseline reset. Future /debug/memory/detailed calls will compare against this point.",
  "timestamp": "2026-01-15T14:40:00.123456"
}
```

---

### 4. POST /debug/memory/gc

**Force garbage collection** - Explicitly triggers GC for diagnostic purposes.

**⚠️ Note:** This is an **active operation**. Only call when you explicitly want to test GC behavior.

**What it does:**
- Runs `gc.collect()` on all three generations
- Measures memory before and after
- Reports what was collected

**When to use:**
- To test if forcing GC reduces memory
- To check for circular references
- To understand GC behavior

**Example Response:**
```json
{
  "timestamp": "2026-01-15T14:45:00.123456",
  "action": "garbage_collection_forced",
  "objects_collected": {
    "generation_0": 1523,
    "generation_1": 245,
    "generation_2": 89,
    "total": 1857
  },
  "memory_impact": {
    "before_rss_mb": 2210.45,
    "after_rss_mb": 2208.12,
    "freed_mb": 2.33,
    "freed_bytes": 2443264
  },
  "interpretation": {
    "objects_collected_meaning": "High count indicates circular references were cleaned up",
    "rss_unchanged_meaning": "Normal - Python allocator keeps memory pages for reuse. This is NOT a memory leak."
  }
}
```

---

## Testing Workflow

### Step 1: Establish Baseline (After POD Starts)

```bash
# Get quick snapshot immediately after POD starts
curl -s "http://<pod-ip>:8001/debug/memory" | jq .

# Expected: RSS ~500-800 MB (embeddings + models loading)
```

### Step 2: Enable Detailed Tracking

```bash
# First call starts tracemalloc
curl -s "http://<pod-ip>:8001/debug/memory/detailed" | jq .

# Response will show: "status": "just_activated"
```

### Step 3: Simulate Load

```bash
# Run some agent queries (10-20 requests)
for i in {1..10}; do
  curl -X POST "http://<pod-ip>:8001/pss/pbmassist/v1/chat" \
    -H "Content-Type: application/json" \
    -d '{"text": "What is the status of claim 123456789?", "session_id": "test-'$i'"}'
  sleep 1
done
```

### Step 4: Analyze Memory Growth

```bash
# Check what grew since baseline
curl -s "http://<pod-ip>:8001/debug/memory/detailed?top_allocators=20" | jq .

# Look at:
# - "memory_growth_since_baseline" → What code allocated memory that wasn't freed
# - "top_allocators" → Which files consume the most memory
# - "rss_vs_heap_gap_mb" → If huge, it's allocator fragmentation
```

### Step 5: Test GC Behavior (Optional)

```bash
# Force GC to see if it helps
curl -X POST "http://<pod-ip>:8001/debug/memory/gc" | jq .

# Interpretation:
# - "freed_mb" ~0 but "objects_collected" high → Circular refs cleaned but memory not returned to OS (normal)
# - "freed_mb" significant → Memory was actually returned to OS
```

### Step 6: Check After Idle Period

```bash
# Wait 5-10 minutes with no traffic, then check
curl -s "http://<pod-ip>:8001/debug/memory" | jq .

# If RSS stays constant → Allocator fragmentation (normal)
# If RSS keeps growing with no traffic → Potential memory leak
```

---

## Interpreting Results

### What Each Metric Means

| Metric | Meaning |
|--------|---------|
| `rss_mb` | **Resident Set Size** - Physical memory used by the process. This is what Grafana shows. |
| `vms_mb` | **Virtual Memory Size** - Total virtual address space including memory-mapped files. |
| `current_heap_mb` | **Python Heap** - Memory tracked by Python's allocator (tracemalloc). |
| `peak_heap_mb` | **Peak Python Heap** - Highest Python heap usage since tracemalloc started. |
| `rss_vs_heap_gap_mb` | **Gap** - Difference between RSS and Python heap. Large gaps indicate native library memory or allocator overhead. |
| `gen0/gen1/gen2_pending` | **GC Pending** - Objects waiting to be examined by garbage collector. |
| `collections` | **GC Runs** - How many times each GC generation has run. |
| `collected` | **Objects Collected** - Total objects freed by GC. |
| `uncollectable` | **Uncollectable** - Objects with circular refs and `__del__` methods (potential leak). |

### Common Scenarios

| Scenario | RSS | Python Heap | Diagnosis |
|----------|-----|-------------|-----------|
| High RSS, low heap | 2.2 GB | 400 MB | Allocator fragmentation (normal) |
| Both high and stable | 2.2 GB | 1.8 GB | Large baseline (embeddings, models) |
| Both growing per request | ↑ | ↑ | **Real memory leak** - check `memory_growth_since_baseline` |
| RSS doesn't drop after GC | Unchanged | Lower | Allocator holds pages for reuse (normal) |

### What "RSS Doesn't Drop" Means

Python's memory allocator (pymalloc) and the C library's malloc **do not return memory to the OS** after freeing objects. Instead:

1. Memory is kept in internal "free lists" for reuse
2. RSS stays at the **high-water mark**
3. Only when entire memory arenas are empty can memory potentially be returned (rare with fragmentation)

**This is normal behavior, not a memory leak.**

---

## FAQ

### Q: Is it safe to call these endpoints in production?

**Yes.** The main `/debug/memory` endpoint is 100% read-only. The `/debug/memory/detailed` endpoint adds ~3-5% CPU overhead after first call but doesn't affect agent functionality.

### Q: Will forcing GC break anything?

**No.** GC only collects **unreachable** objects (objects with zero references). It never deletes objects your code is still using. It may cause a brief pause (~10-100ms) but is safe.

### Q: Why does RSS stay high even after forcing GC?

This is normal Python behavior. The memory allocator keeps freed memory pages for potential reuse rather than returning them to the OS. This is an optimization, not a leak.

### Q: How do I know if there's a real memory leak?

A real leak shows:
1. Python heap (tracemalloc) growing continuously with each request
2. `memory_growth_since_baseline` pointing to specific code locations
3. Memory growth even during idle periods

### Q: What should I do if I find a leak?

Check the `top_allocators` and `memory_growth_since_baseline` to identify the code location. Common causes:
- Singleton dictionaries growing unbounded (e.g., token storage, session cache)
- Objects being re-instantiated per request instead of reused
- Circular references preventing cleanup

