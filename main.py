from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import sys
import os
import asyncio

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
    from core.config import settings
    from langgraph_agent import init_graph, close_graph
    print("[BOOT] Imports succeeded")
except Exception as e:
    import traceback
    print("[BOOT] IMPORT FAILURE:", e)
    traceback.print_exc()
    # Re-raise to fail fast (better than silent)
    raise

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
    # Run endpoint tests in development mode if enabled
    if settings.environment == "development" and os.environ.get("RUN_TESTS_ON_STARTUP", "false").lower() == "true":
        print("[STARTUP] Running endpoint tests...")
        try:
            import subprocess
            result = subprocess.run(
                ["python", "test_all_endpoints.py"],
                capture_output=True,
                text=True,
                timeout=60
            )
            if result.returncode == 0:
                print("[STARTUP] ✅ All endpoint tests passed")
            else:
                print(f"[STARTUP] ⚠️  Endpoint tests had issues (return code: {result.returncode})")
                print(f"[STARTUP] Test output: {result.stdout[-500:] if len(result.stdout) > 500 else result.stdout}")
        except Exception as e:
            print(f"[STARTUP] ⚠️  Could not run tests: {e}")
    
    print("[STARTUP] init_graph begin")      # Breakpoint candidate
    try:
        await init_graph()
        print("[STARTUP] init_graph done")
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
app.include_router(api_router, prefix="/api/v1")

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

# Entry point ---------------------------------------------------------------
if __name__ == "__main__":
    print("🚀 LangGraph Multi-Agent Framework")
    print(f"🤖 Agents: 2 (Intent, Response)")
    print(f"🔧 Nodes: 9 functions")
    print(f"💾 Checkpointing: SQLite")
    print(f"🎯 Mode: {'Mock' if settings.use_mock_llm else 'Real'} LLM")
    print("[BOOT] about to call uvicorn.run")

    # Run WITHOUT reload for reliable breakpoints
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False, log_level="debug")

    # This line executes only when server stops
    print("[BOOT] uvicorn.run returned (server stopped)")