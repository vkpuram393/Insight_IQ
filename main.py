from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import sys
import os
import asyncio
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

@app.get("/redis/session/{session_id}/history")
async def get_session_history(session_id: str):
    """Get conversation history for a specific session"""
    try:
        from memory import MemoryStoreFactory
        from config.config import settings
        
        memory_store = MemoryStoreFactory.get_instance(settings.memory_store_type)
        
        # Get conversation history
        history = await memory_store.get_session_history(session_id)
        
        return {
            "session_id": session_id,
            "message_count": len(history),
            "history": history
        }
        
    except Exception as e:
        return {
            "status": "❌ FAILED",
            "error": str(e)
        }

@app.get("/redis/session/{session_id}/facts")
async def get_session_facts(session_id: str):
    """Get extracted facts for a specific session"""
    try:
        from memory import MemoryStoreFactory
        from config.config import settings
        
        memory_store = MemoryStoreFactory.get_instance(settings.memory_store_type)
        
        # Get session facts
        facts = await memory_store.get_session_facts(session_id)
        
        return {
            "session_id": session_id,
            "fact_count": len(facts),
            "facts": facts
        }
        
    except Exception as e:
        return {
            "status": "❌ FAILED",
            "error": str(e)
        }

@app.get("/redis/session/{session_id}/keys")
async def get_session_keys(session_id: str):
    """Get all Redis keys for a specific session"""
    try:
        from memory import MemoryStoreFactory
        from config.config import settings
        
        memory_store = MemoryStoreFactory.get_instance(settings.memory_store_type)
        
        # Check if it's Redis
        if type(memory_store).__name__ != "RedisStore":
            return {
                "error": f"Memory store is not Redis. Current type: {type(memory_store).__name__}"
            }
        
        # Get all keys for this session
        keys = await memory_store.get_all_session_keys(session_id)
        
        return {
            "session_id": session_id,
            "key_count": len(keys),
            "keys": keys
        }
        
    except Exception as e:
        return {
            "status": "❌ FAILED",
            "error": str(e)
        }

@app.get("/redis/session/{session_id}/all")
async def get_session_all_data(session_id: str):
    """Get all data (history + facts + keys) for a specific session"""
    try:
        from memory import MemoryStoreFactory
        from config.config import settings
        
        memory_store = MemoryStoreFactory.get_instance(settings.memory_store_type)
        
        result = {
            "session_id": session_id,
            "store_type": type(memory_store).__name__,
            "history": {
                "message_count": 0,
                "messages": []
            },
            "facts": {
                "fact_count": 0,
                "facts": []
            }
        }
        
        # If Redis, add connection diagnostics
        if type(memory_store).__name__ == "RedisStore":
            # Get connection status
            connection_status = await memory_store.get_connection_status()
            result["redis_connection"] = connection_status
            
            # Try to get keys (this will show what keys exist)
            keys = await memory_store.get_all_session_keys(session_id)
            result["redis_keys"] = {
                "key_count": len(keys),
                "keys": keys,
                "search_pattern": f"session:{session_id}:*"
            }
            
            # If not connected, return early with diagnostic info
            if not connection_status.get("connected", False):
                result["warning"] = "Redis is not connected. Data retrieval may be incomplete."
                return result
            
            # Also try searching for keys with different session_id formats
            # (in case session_id is stored with/without braces)
            session_id_variants = [
                session_id,  # Original
                session_id.strip("{}"),  # Without braces
                f"{{{session_id}}}" if not session_id.startswith("{") else session_id  # With braces
            ]
            
            all_keys_found = []
            for variant in session_id_variants:
                variant_keys = await memory_store.get_all_session_keys(variant)
                if variant_keys:
                    all_keys_found.extend(variant_keys)
                    if variant != session_id:
                        result["redis_keys"]["alternative_search"] = {
                            "session_id_variant": variant,
                            "keys_found": variant_keys
                        }
            
            # Get all session data
            history = await memory_store.get_session_history(session_id)
            facts = await memory_store.get_session_facts(session_id)
            
            result["history"] = {
                "message_count": len(history),
                "messages": history
            }
            result["facts"] = {
                "fact_count": len(facts),
                "facts": facts
            }
        else:
            # For non-Redis stores, just get the data
            history = await memory_store.get_session_history(session_id)
            facts = await memory_store.get_session_facts(session_id)
            
            result["history"] = {
                "message_count": len(history),
                "messages": history
            }
            result["facts"] = {
                "fact_count": len(facts),
                "facts": facts
            }
        
        return result
        
    except Exception as e:
        import traceback
        return {
            "status": "❌ FAILED",
            "error": str(e),
            "traceback": traceback.format_exc()
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