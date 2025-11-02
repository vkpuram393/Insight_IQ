# PBM LangGraph Multi-Agent Framework 🤖

## 🎯 Architecture

**2 LLM Agents:**
1. Intent Classification Agent
2. Response Generation Agent

**9 Function Nodes:**
- safety_precheck
- check_cache  
- build_context
- confidence_check (router)
- clarification
- call_claims_tool
- safety_postcheck
- update_memory
- cache_response

## 🚀 Quick Start

```bash
# Install
pip install -r requirements.txt

# Run
python main.py

# Test
POST http://localhost:8000/api/v1/chat
{"text": "What's my claim status?"}
```

## 🐛 Debugging

**PyCharm:**
- Breakpoint at main.py line 56
- Breakpoint at api/routes.py line 33
- Breakpoint at langgraph_agent.py line 150

**LangGraph Studio:**
See docs/LANGGRAPH_STUDIO.md

## 📊 Graph Visualization

```python
from langgraph_agent import create_graph
graph = create_graph()
graph.get_graph().draw_mermaid()  # View graph structure
```

## 🎓 Learning

Every file has detailed comments for beginners!
Start with: state/schema.py → langgraph_agent.py
