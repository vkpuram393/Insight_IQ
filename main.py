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
    print("[STARTUP] init_graph begin")      # Breakpoint candidate
    try:
        await init_graph()
        print("[STARTUP] init_graph done")
    except Exception as e:
        import traceback
        print("[STARTUP] init_graph ERROR:", e)
        traceback.print_exc()
        raise
    
    # Run endpoint tests to verify all endpoints are working (development only)
    # Run in background task since server only accepts connections after startup completes
    if settings.environment == "development":
        async def run_endpoint_tests():
            await asyncio.sleep(1.0)  # Give server a moment to be ready
            print("[STARTUP] Running endpoint health checks...")
            try:
                import subprocess
                import socket
                # Wait for server to be ready to accept requests (check if port 8000 is listening)
                max_attempts = 15
                for attempt in range(max_attempts):
                    try:
                        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        sock.settimeout(0.3)
                        result = sock.connect_ex(('127.0.0.1', 8000))
                        sock.close()
                        if result == 0:
                            break
                    except:
                        pass
                    await asyncio.sleep(0.2)
                else:
                    print("[STARTUP] ⚠️  Server not ready, skipping endpoint tests")
                    return
                
                result = subprocess.run(
                    ["python", "test_all_endpoints.py"],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                if result.returncode == 0:
                    # Extract test summary from output
                    output_lines = result.stdout.split('\n')
                    summary_line = [line for line in output_lines if "Passed:" in line]
                    if summary_line:
                        # Clean up the summary line (remove extra ✅ if present)
                        summary = summary_line[0].strip().replace("✅ ", "")
                        print(f"[STARTUP] ✅ {summary}")
                    else:
                        print("[STARTUP] ✅ All endpoint tests passed (16/16)")
                else:
                    print(f"[STARTUP] ⚠️  Endpoint tests had issues (return code: {result.returncode})")
                    # Show last few lines of output for debugging
                    output_lines = result.stdout.split('\n')
                    error_lines = [line for line in output_lines if "❌" in line or "Failed" in line]
                    if error_lines:
                        print(f"[STARTUP] Test errors: {error_lines[-3:]}")
            except subprocess.TimeoutExpired:
                print("[STARTUP] ⚠️  Endpoint tests timed out")
            except Exception as e:
                print(f"[STARTUP] ⚠️  Could not run endpoint tests: {e}")
        
        # Schedule the test task to run after startup
        asyncio.create_task(run_endpoint_tests())

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