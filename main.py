"""
LangGraph Multi-Agent Framework

📍 BREAKPOINT: Line 56 - Start debugging
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from api.routes import router as api_router
from core.config import settings
from langgraph_agent import init_graph, close_graph

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
    await init_graph()

@app.on_event("shutdown")
async def shutdown_event():
    await close_graph()

app.include_router(api_router, prefix="/api/v1")

@app.get("/")
async def root():
    return {
        "message": "PBM LangGraph Framework",
        "version": "2.0.0",
        "agents": 2,
        "nodes": 9,
        "framework": "LangGraph"
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    print("🚀 LangGraph Multi-Agent Framework")
    print(f"🤖 Agents: 2 (Intent, Response)")
    print(f"🔧 Nodes: 9 functions")
    print(f"💾 Checkpointing: SQLite")
    print(f"🎯 Mode: {'Mock' if settings.use_mock_llm else 'Real'} LLM")

    # Run without reload to keep async saver stable while debugging
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
