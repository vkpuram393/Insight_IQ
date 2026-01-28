# Memory Optimization Testing Guide

This guide helps you test memory leak fixes locally before deploying to GKE cluster.

## Prerequisites

1. Python 3.11+ installed
2. All dependencies installed (`pip install -r requirements.txt`)
3. Server running locally (`python main.py` or `uvicorn main:app`)
4. Memory monitoring tools (optional but recommended)

---

## Step 1: Baseline Memory Measurement

### Start the server and measure initial memory:

```bash
# Terminal 1: Start server
python main.py

# Terminal 2: Check initial memory
curl http://localhost:8000/pss/pbmassist/v1/cleanup/memory/stats | python -m json.tool
```

**Expected:** Memory should be around 18-25% of available RAM (similar to GKE baseline).

**Note the baseline values:**
- `process_memory.rss_mb` - Resident Set Size (actual memory used)
- `process_memory.percent` - Percentage of system memory
- `memory_store.cache_keys` - Number of cache entries
- `memory_store.active_sessions` - Number of active sessions

---

## Step 2: Simulate Test Scenarios

### Option A: Use Batch Test Endpoint (Recommended)

Create a test script to simulate multiple test runs:

```bash
# Create test_script.sh
cat > test_memory.sh << 'EOF'
#!/bin/bash

BASE_URL="http://localhost:8000/pss/pbmassist/v1"
TEST_PROMPTS=(
    "What is the status of claim 233211748898001?"
    "Show me pricing for claim 233211748898001 sequence 001"
    "What are the rejection reasons for claim 233211748898001?"
    "Tell me about claim 233211748898001"
    "Show claim details for 233211748898001"
)

echo "🧪 Running ${#TEST_PROMPTS[@]} test prompts..."

for i in {1..50}; do
    echo "Test run $i/50"
    for prompt in "${TEST_PROMPTS[@]}"; do
        curl -X POST "$BASE_URL/test/batch" \
            -H "Content-Type: application/json" \
            -d "{\"text\": \"$prompt\", \"session_id\": \"test-session-$i\"}" \
            -s > /dev/null
    done
    echo "  ✅ Completed batch $i"
    
    # Check memory every 10 batches
    if [ $((i % 10)) -eq 0 ]; then
        echo "  📊 Memory stats after batch $i:"
        curl -s "$BASE_URL/cleanup/memory/stats" | python -m json.tool | grep -E "(rss_mb|percent|active_sessions|cache_keys)"
    fi
done

echo "✅ All tests completed"
EOF

chmod +x test_memory.sh
./test_memory.sh
```

### Option B: Use Python Script (More Control)

```python
# test_memory_leak.py
import requests
import time
import json
from typing import Dict, Any

BASE_URL = "http://localhost:8000/pss/pbmassist/v1"

def get_memory_stats() -> Dict[str, Any]:
    """Get current memory statistics"""
    response = requests.get(f"{BASE_URL}/cleanup/memory/stats")
    return response.json()

def run_batch_test(prompt: str, session_id: str) -> Dict[str, Any]:
    """Run a single batch test"""
    response = requests.post(
        f"{BASE_URL}/test/batch",
        json={"text": prompt, "session_id": session_id}
    )
    return response.json()

def main():
    test_prompts = [
        "What is the status of claim 233211748898001?",
        "Show me pricing for claim 233211748898001 sequence 001",
        "What are the rejection reasons for claim 233211748898001?",
        "Tell me about claim 233211748898001",
        "Show claim details for 233211748898001"
    ]
    
    print("📊 Initial Memory Stats:")
    initial_stats = get_memory_stats()
    print(f"  RSS: {initial_stats['process_memory']['rss_mb']} MB")
    print(f"  Percent: {initial_stats['process_memory']['percent']}%")
    print(f"  Sessions: {initial_stats['memory_store'].get('active_sessions', 0)}")
    print(f"  Cache Keys: {initial_stats['memory_store'].get('cache_keys', 0)}")
    print()
    
    # Run multiple test batches
    for batch in range(1, 51):
        print(f"🧪 Running batch {batch}/50...")
        
        for i, prompt in enumerate(test_prompts):
            session_id = f"test-session-{batch}-{i}"
            result = run_batch_test(prompt, session_id)
            if result.get('exception'):
                print(f"  ⚠️ Error in batch {batch}, prompt {i}: {result['exception']}")
        
        # Check memory every 10 batches
        if batch % 10 == 0:
            stats = get_memory_stats()
            rss_mb = stats['process_memory']['rss_mb']
            percent = stats['process_memory']['percent']
            sessions = stats['memory_store'].get('active_sessions', 0)
            cache_keys = stats['memory_store'].get('cache_keys', 0)
            
            print(f"  📊 After batch {batch}:")
            print(f"    RSS: {rss_mb} MB (Δ: {rss_mb - initial_stats['process_memory']['rss_mb']:.1f} MB)")
            print(f"    Percent: {percent}% (Δ: {percent - initial_stats['process_memory']['percent']:.1f}%)")
            print(f"    Sessions: {sessions}")
            print(f"    Cache Keys: {cache_keys}")
            print()
    
    # Final stats
    print("📊 Final Memory Stats:")
    final_stats = get_memory_stats()
    print(f"  RSS: {final_stats['process_memory']['rss_mb']} MB")
    print(f"  Percent: {final_stats['process_memory']['percent']}%")
    print(f"  Sessions: {final_stats['memory_store'].get('active_sessions', 0)}")
    print(f"  Cache Keys: {final_stats['memory_store'].get('cache_keys', 0)}")
    print()
    
    # Memory growth analysis
    rss_growth = final_stats['process_memory']['rss_mb'] - initial_stats['process_memory']['rss_mb']
    percent_growth = final_stats['process_memory']['percent'] - initial_stats['process_memory']['percent']
    
    print("📈 Memory Growth Analysis:")
    print(f"  RSS Growth: {rss_growth:.1f} MB")
    print(f"  Percent Growth: {percent_growth:.1f}%")
    
    if rss_growth > 500:  # More than 500MB growth
        print("  ⚠️ WARNING: Significant memory growth detected!")
    elif rss_growth < 200:
        print("  ✅ Memory growth is within acceptable limits")
    
    return final_stats

if __name__ == "__main__":
    main()
```

Run it:
```bash
python test_memory_leak.py
```

---

## Step 3: Test Cleanup Endpoints

### Test Normal Cleanup:

```bash
# After running tests, trigger cleanup
curl -X POST http://localhost:8000/pss/pbmassist/v1/cleanup/memory

# Check memory after cleanup
curl http://localhost:8000/pss/pbmassist/v1/cleanup/memory/stats | python -m json.tool
```

**Expected:** Memory should decrease after cleanup.

### Test Aggressive Cleanup:

```bash
# Trigger aggressive cleanup
curl -X POST "http://localhost:8000/pss/pbmassist/v1/cleanup/memory?aggressive=true"

# Check results
curl http://localhost:8000/pss/pbmassist/v1/cleanup/memory/stats | python -m json.tool
```

**Expected:** More aggressive cleanup, lower memory usage.

---

## Step 4: Monitor Memory Over Time

### Option A: Continuous Monitoring Script

```python
# monitor_memory.py
import requests
import time
import json
from datetime import datetime

BASE_URL = "http://localhost:8000/pss/pbmassist/v1"

def monitor_memory(duration_minutes=30, interval_seconds=60):
    """Monitor memory usage over time"""
    print(f"📊 Monitoring memory for {duration_minutes} minutes (checking every {interval_seconds}s)")
    print()
    
    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)
    
    measurements = []
    
    while time.time() < end_time:
        try:
            stats = requests.get(f"{BASE_URL}/cleanup/memory/stats").json()
            rss_mb = stats['process_memory']['rss_mb']
            percent = stats['process_memory']['percent']
            sessions = stats['memory_store'].get('active_sessions', 0)
            cache_keys = stats['memory_store'].get('cache_keys', 0)
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            measurements.append({
                'time': timestamp,
                'rss_mb': rss_mb,
                'percent': percent,
                'sessions': sessions,
                'cache_keys': cache_keys
            })
            
            print(f"[{timestamp}] RSS: {rss_mb:.1f} MB | {percent:.1f}% | Sessions: {sessions} | Cache: {cache_keys}")
            
            time.sleep(interval_seconds)
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(interval_seconds)
    
    # Summary
    if measurements:
        print()
        print("📈 Summary:")
        initial = measurements[0]
        final = measurements[-1]
        print(f"  Initial: {initial['rss_mb']:.1f} MB ({initial['percent']:.1f}%)")
        print(f"  Final: {final['rss_mb']:.1f} MB ({final['percent']:.1f}%)")
        print(f"  Growth: {final['rss_mb'] - initial['rss_mb']:.1f} MB ({final['percent'] - initial['percent']:.1f}%)")
        
        # Check for memory leaks
        if final['rss_mb'] - initial['rss_mb'] > 500:
            print("  ⚠️ WARNING: Potential memory leak detected!")
        else:
            print("  ✅ Memory usage is stable")

if __name__ == "__main__":
    monitor_memory(duration_minutes=30, interval_seconds=60)
```

Run it:
```bash
python monitor_memory.py
```

### Option B: Use System Tools

**Linux/Mac:**
```bash
# Monitor process memory
watch -n 5 'ps aux | grep "python.*main.py" | grep -v grep'

# Or use htop/top
htop -p $(pgrep -f "python.*main.py")
```

**Windows (PowerShell):**
```powershell
# Monitor memory
while ($true) {
    $proc = Get-Process | Where-Object {$_.ProcessName -eq "python"}
    $proc | Select-Object ProcessName, @{Name="Memory(MB)";Expression={[math]::Round($_.WS/1MB,2)}}, CPU
    Start-Sleep -Seconds 5
}
```

---

## Step 5: Test Periodic Cleanup

The periodic cleanup task runs automatically. To verify it's working:

1. **Check logs** for cleanup messages:
   ```bash
   # Look for cleanup messages in server logs
   tail -f server.log | grep "🧹"
   ```

2. **Monitor memory** during idle period:
   - Run tests to increase memory
   - Stop tests and wait 1 hour
   - Check if memory decreases (periodic cleanup should run)

3. **Force cleanup** to verify it works:
   ```bash
   curl -X POST http://localhost:8000/pss/pbmassist/v1/cleanup/memory
   ```

---

## Step 6: Test Memory Limits

### Test Session Limits:

```python
# test_session_limits.py
import requests

BASE_URL = "http://localhost:8000/pss/pbmassist/v1"

# Create 1500 sessions (more than the 1000 limit)
for i in range(1500):
    session_id = f"limit-test-{i}"
    requests.post(
        f"{BASE_URL}/test/batch",
        json={"text": "test", "session_id": session_id}
    )
    if i % 100 == 0:
        print(f"Created {i} sessions...")

# Check stats
stats = requests.get(f"{BASE_URL}/cleanup/memory/stats").json()
print(f"Active sessions: {stats['memory_store'].get('active_sessions', 0)}")
print(f"Expected: ~1000 (limit should be enforced)")
```

**Expected:** Active sessions should not exceed 1000 (oldest sessions cleaned up).

### Test Cache Limits:

Similar test but for cache keys (limit is 5000).

---

## Step 7: Verify Cleanup After Tests

The `/test/batch` endpoint automatically cleans up after each request. Verify:

```bash
# Run a test
curl -X POST http://localhost:8000/pss/pbmassist/v1/test/batch \
  -H "Content-Type: application/json" \
  -d '{"text": "test query"}'

# Check server logs for cleanup message
# Should see: "🧹 Post-test cleanup: {...}"
```

---

## Step 8: Compare Before/After

### Before Fixes (Baseline):
- Memory grows continuously with each test
- No automatic cleanup
- Memory doesn't decrease after tests
- Can reach 133%+ and crash

### After Fixes (Expected):
- Memory stabilizes after initial growth
- Automatic cleanup after tests
- Memory decreases during idle periods
- Stays within limits (sessions < 1000, cache < 5000)
- Periodic cleanup runs every hour (or more frequently if memory is high)

---

## Success Criteria

✅ **Memory Growth:**
- After 50 test batches: Memory growth < 200MB
- After 100 test batches: Memory growth < 300MB
- Memory should stabilize, not continuously grow

✅ **Cleanup Works:**
- Manual cleanup reduces memory
- Automatic cleanup runs after batch tests
- Periodic cleanup runs every hour

✅ **Limits Enforced:**
- Sessions don't exceed 1000
- Cache keys don't exceed 5000
- Old sessions cleaned up automatically

✅ **No Memory Leaks:**
- Memory returns to baseline after cleanup
- No continuous growth during idle periods
- Memory stays within acceptable range (< 80% of available)

---

## Troubleshooting

### Memory Still Growing:
1. Check if cleanup is running: `curl http://localhost:8000/pss/pbmassist/v1/cleanup/memory/stats`
2. Manually trigger cleanup: `curl -X POST http://localhost:8000/pss/pbmassist/v1/cleanup/memory?aggressive=true`
3. Check server logs for errors
4. Verify periodic cleanup task is running (check logs)

### Cleanup Not Working:
1. Check if memory store type is correct (should be "inmemory" for local testing)
2. Verify cleanup endpoints are accessible
3. Check server logs for cleanup errors
4. Try aggressive cleanup mode

### High Memory Usage:
1. Check number of active sessions (should be < 1000)
2. Check cache keys (should be < 5000)
3. Trigger manual cleanup
4. Check if there are other processes using memory

---

## Next Steps

Once local testing passes:
1. ✅ Memory stabilizes after tests
2. ✅ Cleanup works correctly
3. ✅ Limits are enforced
4. ✅ No memory leaks detected

Then deploy to GKE cluster and monitor:
- Use Grafana/Prometheus to monitor pod memory
- Check pod logs for cleanup messages
- Monitor memory usage over 24-48 hours
- Verify pods don't crash

---

## Quick Test Commands

```bash
# 1. Check initial memory
curl http://localhost:8000/pss/pbmassist/v1/cleanup/memory/stats

# 2. Run 10 test requests
for i in {1..10}; do
  curl -X POST http://localhost:8000/pss/pbmassist/v1/test/batch \
    -H "Content-Type: application/json" \
    -d "{\"text\": \"test query $i\"}" -s > /dev/null
done

# 3. Check memory after tests
curl http://localhost:8000/pss/pbmassist/v1/cleanup/memory/stats

# 4. Trigger cleanup
curl -X POST http://localhost:8000/pss/pbmassist/v1/cleanup/memory

# 5. Check memory after cleanup
curl http://localhost:8000/pss/pbmassist/v1/cleanup/memory/stats
```

---

## Additional Tools

### Memory Profiler (Optional):
```bash
pip install memory-profiler

# Profile a test run
python -m memory_profiler test_memory_leak.py
```

### Visual Memory Graph:
Use the monitoring script and plot results with matplotlib or export to CSV for analysis.

