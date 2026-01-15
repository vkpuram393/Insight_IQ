from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import sys
import os
import asyncio
from datetime import datetime
from services.llm_connection import generate
# BOOT diagnostics -----------------------------------------------------------
print("[BOOT] __name__ =", __name__)        # Breakpoint candidate
print("[BOOT] __file__ =", __file__)
print("[BOOT] executable =", sys.executable)
print("[BOOT] python version =", sys.version)

# Optional: verify expected env vars
print("[BOOT] ENVIRONMENT =", os.environ.get("ENVIRONMENT"))
print("[BOOT] USE_REDIS_MEMORY =", os.environ.get("USE_REDIS_MEMORY"))

def FORCE_BREAKPOINT(): val = 42 # set breakpoint here FORCE_BREAKPOINT()

# Import application components ---------------------------------------------
try:
    from api.routes import router as api_router
    from config.config import settings
    from langgraph_agent import init_graph, close_graph
    print("[BOOT] Imports succeeded")
except Exception as e:
    import traceback
    print("[BOOT] IMPORT FAILURE:", e)
    traceback.print_exc()
    # Re-raise to fail fast (better than silent)
    raise

# Configuration Validation ---------------------------------------------------
try:
    from config.validation import validate_all
    print("[BOOT] Running configuration validation...")
    if not validate_all():
        print("[BOOT] ❌ Configuration validation FAILED. Application cannot start.")
        print("[BOOT] Please review the error messages above and fix configuration issues.")
        sys.exit(1)
    print("[BOOT] ✅ Configuration validation passed")
except Exception as e:
    import traceback
    print("[BOOT] ⚠️  Configuration validation error (non-fatal):", e)
    traceback.print_exc()
    # Continue startup - validation errors are logged but don't block startup
    # (This allows the app to start even if validation has issues)

app = FastAPI(
    title="PBM LangGraph Framework",
    description="2 Agents + 9 Nodes",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup / Shutdown hooks --------------------------------------------------
@app.on_event("startup")
async def startup_event():
    print("[STARTUP] init_graph begin")      # Breakpoint candidate
    try:
        await init_graph()
        print("[STARTUP] init_graph done")
        
        # Pre-initialize embedding classifier to ensure MongoDB has embeddings
        # (Embedding generation takes ~3 minutes, must happen before queries arrive)
        if settings.use_embedding_classifier:
            print("[STARTUP] Pre-initializing embedding classifier...")
            import concurrent.futures
            from classifiers.embedded_classifier import CVSIntentEmbedded
            
            def init_classifier():
                return CVSIntentEmbedded()  # This triggers MongoDB setup
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(init_classifier)
                future.result()  # No timeout - let it complete (fails naturally if broken)
            
            print("[STARTUP] Embedding classifier ready (MongoDB populated)")
        
    except Exception as e:
        import traceback
        print("[STARTUP] init_graph ERROR:", e)
        traceback.print_exc()
        raise

@app.on_event("shutdown")
async def shutdown_event():
    print("[SHUTDOWN] close_graph begin")
    try:
        await close_graph()
        print("[SHUTDOWN] close_graph done")
    except Exception as e:
        import traceback
        print("[SHUTDOWN] close_graph ERROR:", e)
        traceback.print_exc()

# Routes --------------------------------------------------------------------
app.include_router(api_router, prefix="/pss/pbmassist/v1")

# Test/Utils endpoints for individual component testing
try:
    from utils.test_endpoints import router as utils_router
    app.include_router(utils_router)
    print("[BOOT] Utils test endpoints loaded")
except Exception as e:
    print(f"[BOOT] Could not load utils endpoints: {e}")

@app.get("/")
async def root():
    # Breakpoint candidate
    return {
        "message": "PBM LangGraph Framework",
        "version": "2.0.0",
        "agents": 2,
        "nodes": 9,
        "framework": "LangGraph"
    }

@app.get("/health")
async def health():
    # Breakpoint candidate
    x=1
    return {"status": "healthy"}

@app.get("/llm_test")
async def llm_test():
    # Breakpoint candidate
    response = generate("Hello, how are you?")
    return {"status": "llm_test", "response": response}


@app.get("/redis/sessions")
async def list_redis_sessions():
    """List all session IDs that have conversations stored in Redis"""
    try:
        from memory import MemoryStoreFactory
        from config.config import settings
        
        memory_store = MemoryStoreFactory.get_instance(settings.memory_store_type)
        
        # Check if it's Redis
        if type(memory_store).__name__ != "RedisStore":
            return {
                "error": f"Memory store is not Redis. Current type: {type(memory_store).__name__}",
                "suggestion": "Set MEMORY_STORE_TYPE=redis in .env file"
            }
        
        # List all sessions
        session_ids = await memory_store.list_all_sessions()
        
        return {
            "total_sessions": len(session_ids),
            "sessions": session_ids
        }
        
    except Exception as e:
        return {
            "status": "❌ FAILED",
            "error": str(e)
        }

@app.get("/redis/session/{session_id}/info")
async def get_session_history(session_id: str):
    """
    Get complete session information from Redis.
    
    Returns:
    - Complete conversation history (all messages)
    - Session facts
    - Redis keys for the session
    - Connection diagnostics
    
    Automatically handles session_id format variations (with/without braces).
    """
    try:
        from memory import MemoryStoreFactory
        from config.config import settings
        
        memory_store = MemoryStoreFactory.get_instance(settings.memory_store_type)
        
        # Try both session_id formats (with and without braces)
        session_id_variants = [
            session_id,  # Original format from URL
            session_id.strip("{}"),  # Without braces
            f"{{{session_id}}}" if not session_id.startswith("{") else session_id  # With braces
        ]
        
        history = []
        facts = []
        found_session_id = None
        
        # Try each variant until we find data
        for variant in session_id_variants:
            variant_history = await memory_store.get_session_history(variant)
            variant_facts = await memory_store.get_session_facts(variant)
            
            if variant_history or variant_facts:
                history = variant_history
                facts = variant_facts
                found_session_id = variant
                break
        
        # If Redis, add diagnostic information and keys
        diagnostics = {}
        redis_keys = {}
        
        if type(memory_store).__name__ == "RedisStore":
            connection_status = await memory_store.get_connection_status()
            diagnostics["redis_connection"] = connection_status
            
            # Get all keys for the found session_id (or try all variants)
            target_session_id = found_session_id or session_id
            keys = await memory_store.get_all_session_keys(target_session_id)
            
            if keys:
                redis_keys = {
                    "key_count": len(keys),
                    "keys": keys,
                    "search_pattern": f"session:{target_session_id}:*"
                }
            else:
                # Try all variants to find keys
                all_keys_found = {}
                for variant in session_id_variants:
                    variant_keys = await memory_store.get_all_session_keys(variant)
                    if variant_keys:
                        all_keys_found[variant] = variant_keys
                
                if all_keys_found:
                    # Use the first variant that has keys
                    first_variant_with_keys = list(all_keys_found.keys())[0]
                    redis_keys = {
                        "key_count": len(all_keys_found[first_variant_with_keys]),
                        "keys": all_keys_found[first_variant_with_keys],
                        "search_pattern": f"session:{first_variant_with_keys}:*",
                        "alternative_formats": all_keys_found
                    }
                else:
                    # Try to list all sessions to see what's available
                    try:
                        all_sessions = await memory_store.list_all_sessions()
                        diagnostics["available_sessions"] = all_sessions[:20]  # Limit to first 20
                        diagnostics["total_sessions"] = len(all_sessions)
                    except Exception:
                        pass
        
        result = {
            "session_id": session_id,
            "found_session_id": found_session_id or session_id,
            "message_count": len(history),
            "fact_count": len(facts),
            "history": history,
            "facts": facts,
            "retrieved_at": datetime.now().isoformat()
        }
        
        # Add keys if available
        if redis_keys:
            result["redis_keys"] = redis_keys
        
        # Add diagnostics if available
        if diagnostics:
            result["diagnostics"] = diagnostics
        
        return result
        
    except Exception as e:
        import traceback
        return {
            "status": "❌ FAILED",
            "error": str(e),
            "traceback": traceback.format_exc()
        }

@app.delete("/redis/sessions/cleanup")
async def delete_old_sessions(days: int = Query(..., ge=1, le=365, description="Number of days - sessions older than this will be deleted (required, 1-365 days)")):
    """
    Delete all sessions older than specified number of days.
    
    Args:
        days: Number of days (REQUIRED). Sessions older than this will be deleted.
              Must be between 1 and 365 days.
    
    Example:
        DELETE /redis/sessions/cleanup?days=2
        DELETE /redis/sessions/cleanup?days=7
    
    Returns:
        Statistics about deleted sessions
    """
    try:
        from memory import MemoryStoreFactory
        from config.config import settings
        
        memory_store = MemoryStoreFactory.get_instance(settings.memory_store_type)
        
        # Check if it's Redis
        if type(memory_store).__name__ != "RedisStore":
            return {
                "status": "❌ FAILED",
                "error": f"Memory store is not Redis. Current type: {type(memory_store).__name__}",
                "message": "Session cleanup is only available for Redis store"
            }
        
        # Delete old sessions
        result = await memory_store.delete_sessions_older_than_days(days)
        
        return {
            "status": "✅ SUCCESS",
            "message": f"Deleted sessions older than {days} days",
            "deleted_sessions": result.get("deleted_sessions", 0),
            "deleted_keys": result.get("deleted_keys", 0),
            "skipped_sessions": result.get("skipped_sessions", 0),
            "cutoff_date": result.get("cutoff_date"),
            "days": result.get("days", days)
        }
        
    except Exception as e:
        import traceback
        return {
            "status": "❌ FAILED",
            "error": str(e),
            "traceback": traceback.format_exc()
        }


# ==============================================================================
# MEMORY DIAGNOSTICS ENDPOINTS
# Non-interfering, observability-only endpoints for POD memory analysis.
# These endpoints do NOT affect agent functionality - safe to call in production.
# ==============================================================================

import gc
import tracemalloc
from typing import Dict, Any, Optional as OptionalType

# Module-level state for tracemalloc (isolated, only used by debug endpoints)
_debug_tracemalloc_active: bool = False
_debug_baseline_snapshot: OptionalType[Any] = None


@app.get("/debug/memory")
async def debug_memory_snapshot():
    """
    Quick memory snapshot for POD diagnostics.
    
    100% READ-ONLY - does NOT trigger garbage collection or interfere with
    normal agent operations. Safe to call in production at any time.
    
    Returns:
        - Process RSS/VMS (what Grafana typically shows)
        - GC statistics (read-only, no collection triggered)
        - Linux /proc/self/status metrics (when running in POD)
        - Tracemalloc status
    """
    import psutil
    
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    
    result = {
        "timestamp": datetime.now().isoformat(),
        "endpoint": "quick_snapshot",
        "interference_level": "none",
        
        # Process memory (what Grafana shows)
        "process_memory": {
            "rss_mb": round(mem_info.rss / 1024 / 1024, 2),
            "rss_bytes": mem_info.rss,
            "vms_mb": round(mem_info.vms / 1024 / 1024, 2),
            "percent": round(process.memory_percent(), 2),
        },
        
        # GC stats (read-only, no collection triggered)
        "gc_stats": {
            "enabled": gc.isenabled(),
            "thresholds": gc.get_threshold(),
            "counts": {
                "gen0_pending": gc.get_count()[0],
                "gen1_pending": gc.get_count()[1],
                "gen2_pending": gc.get_count()[2],
            },
            "generation_stats": [
                {
                    "generation": i,
                    "collections": s.get("collections", 0),
                    "collected": s.get("collected", 0),
                    "uncollectable": s.get("uncollectable", 0)
                }
                for i, s in enumerate(gc.get_stats())
            ]
        },
        
        # Tracemalloc status
        "tracemalloc": {
            "is_active": tracemalloc.is_tracing(),
            "hint": "Call GET /debug/memory/detailed to enable heap tracking"
        }
    }
    
    # Linux POD metrics from /proc/self/status (graceful fallback for Windows)
    try:
        with open('/proc/self/status', 'r') as f:
            proc_metrics = {}
            for line in f:
                for key in ['VmSize', 'VmRSS', 'VmPeak', 'VmHWM', 'VmData', 'VmStk']:
                    if line.startswith(key + ':'):
                        proc_metrics[key] = line.split(':')[1].strip()
            result["linux_proc_status"] = proc_metrics
    except FileNotFoundError:
        result["linux_proc_status"] = {"note": "Not available (non-Linux environment)"}
    except Exception as e:
        result["linux_proc_status"] = {"error": str(e)}
    
    return result


@app.get("/debug/memory/detailed")
async def debug_memory_detailed(
    top_allocators: int = Query(15, ge=1, le=50, description="Number of top memory allocators to show")
):
    """
    Detailed memory analysis with Python heap tracking via tracemalloc.
    
    FIRST CALL: Starts tracemalloc and captures baseline (~3-5% CPU overhead).
    SUBSEQUENT CALLS: Shows memory growth since baseline and top allocators.
    
    Use this to identify which code files are consuming memory.
    Does NOT trigger garbage collection.
    
    Args:
        top_allocators: Number of top memory-consuming code locations to show
    
    Returns:
        - Everything from /debug/memory (process memory, GC stats)
        - Python heap size tracked by tracemalloc
        - Top memory allocators by file/line
        - Memory growth since baseline
        - Interpretation hints
    """
    global _debug_tracemalloc_active, _debug_baseline_snapshot
    import psutil
    
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    
    result = {
        "timestamp": datetime.now().isoformat(),
        "endpoint": "detailed_analysis",
        "interference_level": "minimal",
        
        "process_memory": {
            "rss_mb": round(mem_info.rss / 1024 / 1024, 2),
            "vms_mb": round(mem_info.vms / 1024 / 1024, 2),
            "percent": round(process.memory_percent(), 2),
        }
    }
    
    # Initialize or read tracemalloc
    if not _debug_tracemalloc_active:
        tracemalloc.start(25)  # 25 frames of traceback for detailed analysis
        _debug_tracemalloc_active = True
        _debug_baseline_snapshot = tracemalloc.take_snapshot()
        
        result["tracemalloc"] = {
            "status": "just_activated",
            "message": "Tracemalloc started and baseline captured. Call this endpoint again to see memory changes.",
            "overhead": "~3-5% CPU, minimal memory for storing traces"
        }
    else:
        current_snapshot = tracemalloc.take_snapshot()
        current_size, peak_size = tracemalloc.get_traced_memory()
        
        # Top allocators by current size
        top_stats = current_snapshot.statistics('lineno')[:top_allocators]
        
        result["tracemalloc"] = {
            "status": "active",
            "current_heap_mb": round(current_size / 1024 / 1024, 2),
            "peak_heap_mb": round(peak_size / 1024 / 1024, 2),
            "top_allocators": [
                {
                    "location": str(stat.traceback),
                    "size_mb": round(stat.size / 1024 / 1024, 4),
                    "count": stat.count
                }
                for stat in top_stats
            ]
        }
        
        # Growth since baseline
        if _debug_baseline_snapshot:
            diff_stats = current_snapshot.compare_to(_debug_baseline_snapshot, 'lineno')
            growth = [s for s in diff_stats[:top_allocators] if s.size_diff > 0]
            
            result["memory_growth_since_baseline"] = [
                {
                    "location": str(stat.traceback),
                    "growth_mb": round(stat.size_diff / 1024 / 1024, 4),
                    "new_allocations": stat.count_diff
                }
                for stat in growth
            ]
        
        # Interpretation
        rss_mb = result["process_memory"]["rss_mb"]
        heap_mb = result["tracemalloc"]["current_heap_mb"]
        gap_mb = round(rss_mb - heap_mb, 2)
        
        result["interpretation"] = {
            "rss_vs_heap_gap_mb": gap_mb,
            "diagnosis": (
                "Gap is normal (C extensions, numpy, allocator overhead)"
                if gap_mb < 500
                else "Large gap suggests memory held by native libraries or allocator fragmentation"
            ),
            "hint": (
                "If RSS stays high but heap is low, memory is held by Python allocator (normal behavior). "
                "Consider periodic POD restart if memory pressure is a concern."
            )
        }
    
    return result


@app.post("/debug/memory/reset-baseline")
async def debug_memory_reset_baseline():
    """
    Reset the tracemalloc baseline for fresh memory growth comparison.
    
    Use after making code changes or to start a new analysis session.
    Does NOT interfere with normal operations.
    """
    global _debug_baseline_snapshot
    
    if not _debug_tracemalloc_active:
        return {
            "status": "error",
            "message": "Tracemalloc not active. Call GET /debug/memory/detailed first to start tracking."
        }
    
    _debug_baseline_snapshot = tracemalloc.take_snapshot()
    return {
        "status": "success",
        "message": "Baseline reset. Future /debug/memory/detailed calls will compare against this point.",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/debug/memory/gc")
async def debug_memory_force_gc():
    """
    Explicitly trigger garbage collection for diagnostic purposes.
    
    ⚠️ ACTIVE OPERATION - Only call when you explicitly want to test GC behavior.
    This is safe but will briefly pause Python execution (typically <100ms).
    
    Use to answer: "If I force GC, does RSS drop? Are there circular references?"
    
    Returns:
        - Objects collected per generation
        - Memory before and after GC
        - Whether RSS actually decreased (rare due to allocator behavior)
        - Interpretation of results
    """
    import psutil
    
    process = psutil.Process(os.getpid())
    
    # Measure before GC
    before = process.memory_info()
    before_rss = before.rss
    
    # Run GC on all generations
    collected_gen0 = gc.collect(generation=0)
    collected_gen1 = gc.collect(generation=1)
    collected_gen2 = gc.collect(generation=2)
    total_collected = collected_gen0 + collected_gen1 + collected_gen2
    
    # Measure after GC
    after = process.memory_info()
    after_rss = after.rss
    freed_bytes = before_rss - after_rss
    
    return {
        "timestamp": datetime.now().isoformat(),
        "action": "garbage_collection_forced",
        
        "objects_collected": {
            "generation_0": collected_gen0,
            "generation_1": collected_gen1,
            "generation_2": collected_gen2,
            "total": total_collected
        },
        
        "memory_impact": {
            "before_rss_mb": round(before_rss / 1024 / 1024, 2),
            "after_rss_mb": round(after_rss / 1024 / 1024, 2),
            "freed_mb": round(freed_bytes / 1024 / 1024, 2),
            "freed_bytes": freed_bytes
        },
        
        "interpretation": {
            "objects_collected_meaning": (
                "High count indicates circular references were cleaned up"
                if total_collected > 100
                else "Low count means most memory is already managed or held by live references"
            ),
            "rss_unchanged_meaning": (
                "Normal - Python allocator keeps memory pages for reuse. This is NOT a memory leak."
                if freed_bytes < 1024 * 1024  # < 1MB
                else f"Memory returned to OS: {round(freed_bytes / 1024 / 1024, 2)} MB"
            )
        }
    }


# Entry point ---------------------------------------------------------------
if __name__ == "__main__":
    print("🚀 LangGraph Multi-Agent Framework")
    print(f"🤖 Agents: 2 (Intent, Response)")
    print(f"🔧 Nodes: 9 functions")
    print(f"💾 Checkpointing: SQLite")
    print(f"🎯 Mode: {'Mock' if settings.use_mock_llm else 'Real'} LLM")
    print("[BOOT] about to call uvicorn.run")

    # Hot reload configuration
    # ⚠️ IMPORTANT: reload=True causes server restarts when database files are written
    # (checkpoints.db, telemetry.db, or their WAL/SHM files)
    # Default to False for stability - enable via RELOAD=true environment variable if needed
    enable_reload = os.environ.get("RELOAD", "false").lower() == "true"
    
    if enable_reload:
        print("[BOOT] ⚠️  Hot reload is ENABLED")
        print("[BOOT]    Database file writes (checkpoints.db, telemetry.db) may trigger restarts")
        print("[BOOT]    To disable: set RELOAD=false or remove RELOAD environment variable")
    else:
        print("[BOOT] ✅ Hot reload is DISABLED (default for stability)")
        print("[BOOT]    To enable: set RELOAD=true environment variable")
        print("[BOOT]    Note: Manual server restart required for code changes")
    
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8001,
        reload=enable_reload,
        log_level="debug"
    )

    # This line executes only when server stops
    print("[BOOT] uvicorn.run returned (server stopped)")