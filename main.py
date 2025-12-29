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
        port=8000,
        reload=enable_reload,
        log_level="debug"
    )

    # This line executes only when server stops
    print("[BOOT] uvicorn.run returned (server stopped)")