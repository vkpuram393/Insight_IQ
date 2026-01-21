#!/usr/bin/env python3
"""
Quick Memory Optimization Test Script

This script tests memory leak fixes by:
1. Measuring baseline memory
2. Running multiple test requests
3. Monitoring memory growth
4. Testing cleanup functionality
5. Verifying memory returns to baseline

Usage:
    python scripts/test_memory_optimization.py
"""

import requests
import time
import json
import sys
from typing import Dict, Any, List
from datetime import datetime

BASE_URL = "http://localhost:8001/pss/pbmassist/v1"

def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def get_memory_stats() -> Dict[str, Any]:
    """Get current memory statistics"""
    try:
        response = requests.get(f"{BASE_URL}/cleanup/memory/stats", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ Error getting memory stats: {e}")
        print("   Make sure the server is running on http://localhost:8001")
        sys.exit(1)

def print_memory_stats(stats: Dict[str, Any], label: str = ""):
    """Print formatted memory statistics"""
    pm = stats.get('process_memory', {})
    ms = stats.get('memory_store', {})
    gc = stats.get('gc_stats', {})
    
    print(f"\n📊 Memory Stats {label}:")
    print(f"  RSS: {pm.get('rss_mb', 'N/A')} MB")
    print(f"  Percent: {pm.get('percent', 'N/A')}%")
    print(f"  Active Sessions: {ms.get('active_sessions', 'N/A')}")
    print(f"  Cache Keys: {ms.get('cache_keys', 'N/A')}")
    if gc.get('counts'):
        print(f"  GC Pending: Gen0={gc['counts'].get('gen0', 'N/A')}, "
              f"Gen1={gc['counts'].get('gen1', 'N/A')}, "
              f"Gen2={gc['counts'].get('gen2', 'N/A')}")

def run_batch_test(prompt: str, session_id: str) -> Dict[str, Any]:
    """Run a single batch test"""
    try:
        response = requests.post(
            f"{BASE_URL}/test/batch",
            json={"text": prompt, "session_id": session_id},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"  ⚠️ Error in batch test: {e}")
        return {"exception": str(e)}

def test_cleanup(aggressive: bool = False) -> Dict[str, Any]:
    """Test cleanup functionality"""
    try:
        url = f"{BASE_URL}/cleanup/memory"
        if aggressive:
            url += "?aggressive=true"
        response = requests.post(url, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"  ⚠️ Error in cleanup: {e}")
        return {}

def main():
    print_section("Memory Optimization Test Suite")
    print("\nThis test will:")
    print("  1. Measure baseline memory")
    print("  2. Run 50 test batches (250 total requests)")
    print("  3. Monitor memory growth")
    print("  4. Test cleanup functionality")
    print("  5. Verify memory returns to baseline")
    print("\nMake sure the server is running on http://localhost:8000")
    
    input("\nPress Enter to start...")
    
    # Test prompts
    test_prompts = [
        "What is the status of claim 233211748898001?",
        "Show me pricing for claim 233211748898001 sequence 001",
        "What are the rejection reasons for claim 233211748898001?",
        "Tell me about claim 233211748898001",
        "Show claim details for 233211748898001"
    ]
    
    # Step 1: Baseline
    print_section("Step 1: Baseline Memory Measurement")
    print("⚠️  NOTE: If using embedding classifier, initial memory includes ~800MB for embeddings.")
    print("   This is expected and happens once at startup. Memory should stabilize after this.")
    print()
    initial_stats = get_memory_stats()
    print_memory_stats(initial_stats, "(Initial)")
    
    # Check if embeddings are loaded
    if initial_stats.get('embedding_classifier'):
        emb_info = initial_stats['embedding_classifier']
        if emb_info.get('estimated_memory_mb'):
            print(f"  📊 Embedding Classifier: ~{emb_info['estimated_memory_mb']} MB (expected)")
    
    initial_rss = initial_stats['process_memory']['rss_mb']
    initial_percent = initial_stats['process_memory']['percent']
    
    # Step 2: Run tests
    print_section("Step 2: Running Test Batches")
    print(f"Running {len(test_prompts)} prompts × 50 batches = 250 total requests...")
    print("(This may take a few minutes)\n")
    
    memory_measurements: List[Dict[str, Any]] = []
    
    for batch in range(1, 51):
        batch_start = time.time()
        
        # Run all prompts for this batch
        for i, prompt in enumerate(test_prompts):
            session_id = f"test-session-{batch}-{i}"
            result = run_batch_test(prompt, session_id)
            if result.get('exception'):
                print(f"  ⚠️ Error in batch {batch}, prompt {i+1}: {result['exception']}")
        
        batch_time = time.time() - batch_start
        
        # Measure memory every 10 batches
        if batch % 10 == 0:
            stats = get_memory_stats()
            rss_mb = stats['process_memory']['rss_mb']
            percent = stats['process_memory']['percent']
            sessions = stats['memory_store'].get('active_sessions', 0)
            cache_keys = stats['memory_store'].get('cache_keys', 0)
            
            memory_measurements.append({
                'batch': batch,
                'rss_mb': rss_mb,
                'percent': percent,
                'sessions': sessions,
                'cache_keys': cache_keys
            })
            
            rss_growth = rss_mb - initial_rss
            percent_growth = percent - initial_percent
            
            print(f"  ✅ Batch {batch}/50 completed ({batch_time:.1f}s)")
            print(f"     RSS: {rss_mb:.1f} MB (Δ: {rss_growth:+.1f} MB)")
            print(f"     Percent: {percent:.1f}% (Δ: {percent_growth:+.1f}%)")
            print(f"     Sessions: {sessions}, Cache: {cache_keys}")
            print()
        else:
            print(f"  ✅ Batch {batch}/50 completed ({batch_time:.1f}s)", end='\r')
    
    # Step 3: Check final memory
    print_section("Step 3: Final Memory After Tests")
    final_stats = get_memory_stats()
    print_memory_stats(final_stats, "(After Tests)")
    
    rss_growth = final_stats['process_memory']['rss_mb'] - initial_rss
    percent_growth = final_stats['process_memory']['percent'] - initial_percent
    
    print(f"\n📈 Memory Growth:")
    print(f"  RSS: {rss_growth:+.1f} MB")
    print(f"  Percent: {percent_growth:+.1f}%")
    
    # Step 4: Test cleanup
    print_section("Step 4: Testing Cleanup")
    print("Triggering normal cleanup...")
    cleanup_result = test_cleanup(aggressive=False)
    if cleanup_result.get('results'):
        print(f"  ✅ Cleanup completed:")
        print(f"     Sessions cleaned: {cleanup_result['results'].get('sessions_cleaned', 0)}")
        print(f"     Checkpoints cleaned: {cleanup_result['results'].get('checkpoints_cleaned', 0)}")
        print(f"     Cache keys cleaned: {cleanup_result['results'].get('cache_keys_cleaned', 0)}")
        print(f"     Objects collected: {cleanup_result['results'].get('objects_collected', 0)}")
    
    # Wait a moment for cleanup to complete
    time.sleep(2)
    
    # Check memory after cleanup
    print("\nChecking memory after cleanup...")
    after_cleanup_stats = get_memory_stats()
    print_memory_stats(after_cleanup_stats, "(After Cleanup)")
    
    cleanup_rss_reduction = final_stats['process_memory']['rss_mb'] - after_cleanup_stats['process_memory']['rss_mb']
    cleanup_percent_reduction = final_stats['process_memory']['percent'] - after_cleanup_stats['process_memory']['percent']
    
    print(f"\n📉 Memory Reduction from Cleanup:")
    print(f"  RSS: {cleanup_rss_reduction:.1f} MB")
    print(f"  Percent: {cleanup_percent_reduction:.1f}%")
    
    # Step 5: Test aggressive cleanup
    print_section("Step 5: Testing Aggressive Cleanup")
    print("Triggering aggressive cleanup...")
    aggressive_result = test_cleanup(aggressive=True)
    if aggressive_result.get('results'):
        print(f"  ✅ Aggressive cleanup completed:")
        print(f"     Sessions cleaned: {aggressive_result['results'].get('sessions_cleaned', 0)}")
        print(f"     Checkpoints cleaned: {aggressive_result['results'].get('checkpoints_cleaned', 0)}")
        print(f"     Objects collected: {aggressive_result['results'].get('objects_collected', 0)}")
    
    time.sleep(2)
    
    # Final check
    print("\nChecking final memory...")
    final_after_cleanup_stats = get_memory_stats()
    print_memory_stats(final_after_cleanup_stats, "(Final)")
    
    # Summary
    print_section("Test Summary")
    
    total_rss_growth = final_after_cleanup_stats['process_memory']['rss_mb'] - initial_rss
    total_percent_growth = final_after_cleanup_stats['process_memory']['percent'] - initial_percent
    
    print(f"\n📊 Overall Results:")
    print(f"  Initial RSS: {initial_rss:.1f} MB ({initial_percent:.1f}%)")
    print(f"  Final RSS: {final_after_cleanup_stats['process_memory']['rss_mb']:.1f} MB "
          f"({final_after_cleanup_stats['process_memory']['percent']:.1f}%)")
    print(f"  Total Growth: {total_rss_growth:+.1f} MB ({total_percent_growth:+.1f}%)")
    print()
    
    # Success criteria
    print("✅ Success Criteria:")
    
    # Adjust threshold if embeddings are loaded (they add ~800MB initially)
    embedding_memory = final_after_cleanup_stats.get('embedding_classifier', {}).get('estimated_memory_mb', 0)
    if embedding_memory > 0:
        # If embeddings are loaded, the initial jump is expected
        # We care more about growth AFTER the initial load
        # Check if memory is stabilizing (growth in last batches < 50MB)
        if memory_measurements and len(memory_measurements) >= 3:
            recent_growth = memory_measurements[-1]['rss_mb'] - memory_measurements[-3]['rss_mb']
            if recent_growth < 50:
                print(f"  ✅ Memory growth after initial load < 50MB: {recent_growth:.1f} MB (stabilizing)")
            else:
                print(f"  ⚠️ Memory still growing: {recent_growth:.1f} MB in last 20 batches")
        print(f"  ℹ️  Initial embedding load: ~{embedding_memory:.1f} MB (expected, happens once)")
    else:
        # No embeddings loaded - use standard threshold
        if total_rss_growth < 300:
            print(f"  ✅ Memory growth < 300MB: {total_rss_growth:.1f} MB")
        else:
            print(f"  ❌ Memory growth >= 300MB: {total_rss_growth:.1f} MB")
    
    if final_after_cleanup_stats['memory_store'].get('active_sessions', 0) < 1000:
        print(f"  ✅ Sessions within limit: {final_after_cleanup_stats['memory_store'].get('active_sessions', 0)} < 1000")
    else:
        print(f"  ❌ Sessions exceed limit: {final_after_cleanup_stats['memory_store'].get('active_sessions', 0)} >= 1000")
    
    if final_after_cleanup_stats['memory_store'].get('cache_keys', 0) < 5000:
        print(f"  ✅ Cache keys within limit: {final_after_cleanup_stats['memory_store'].get('cache_keys', 0)} < 5000")
    else:
        print(f"  ❌ Cache keys exceed limit: {final_after_cleanup_stats['memory_store'].get('cache_keys', 0)} >= 5000")
    
    if cleanup_rss_reduction > 0:
        print(f"  ✅ Cleanup reduces memory: {cleanup_rss_reduction:.1f} MB")
    else:
        print(f"  ⚠️ Cleanup didn't reduce memory significantly")
    
    print()
    print("="*80)
    print("  Test Complete!")
    print("="*80)
    print()
    
    # Memory growth analysis
    if memory_measurements:
        print("📈 Memory Growth Over Time:")
        for m in memory_measurements:
            growth = m['rss_mb'] - initial_rss
            print(f"  Batch {m['batch']:2d}: {m['rss_mb']:6.1f} MB (Δ: {growth:+.1f} MB)")
        
        # Check if memory is stabilizing
        if len(memory_measurements) >= 3:
            recent_growth = memory_measurements[-1]['rss_mb'] - memory_measurements[-3]['rss_mb']
            if recent_growth < 50:
                print("\n  ✅ Memory is stabilizing (growth < 50MB in last 20 batches)")
            else:
                print(f"\n  ⚠️ Memory still growing (growth {recent_growth:.1f}MB in last 20 batches)")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

