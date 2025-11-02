# LangGraph Studio Guide 🎨

## What is LangGraph Studio?

LangGraph Studio is a visual IDE for debugging LangGraph applications.
It shows you:
- The graph structure (visual flowchart)
- State at each node
- Execution path taken
- Step-by-step debugging

## Installation

```bash
# Install LangGraph CLI
pip install langgraph-cli

# Or use Docker
docker pull langchain/langgraph-studio
```

## Setup for This Project

1. **Create langgraph.json:**

```json
{
  "dependencies": ["requirements.txt"],
  "graphs": {
    "agent": "langgraph_agent.py:create_graph"
  },
  "env": ".env"
}
```

2. **Start Studio:**

```bash
langgraph up
```

3. **Open browser:**
http://localhost:8000

## Using Studio

### 1. View Graph Structure
- Click "Graph" tab
- See visual flowchart of your nodes
- Blue boxes = nodes
- Arrows = edges
- Diamond = conditional router

### 2. Test Your Graph
- Click "Playground"
- Enter: `{"text": "What's my claim status?", "session_id": "test"}`
- Click "Run"
- Watch execution in real-time!

### 3. Debug Step-by-Step
- Click "Step" mode
- Execute one node at a time
- Inspect state after each node
- See exactly what changed

### 4. View State
- Click any node
- See state before and after
- Inspect all fields
- Track how data flows

### 5. Time Travel
- LangGraph saves every state
- Rewind to any point
- See what happened
- Debug issues easily

## Common Workflows

### Testing Intent Classification
1. Start Studio
2. Send: "Why was my claim rejected?"
3. Watch it hit intent_agent
4. See confidence score
5. Check routing decision

### Debugging Low Confidence
1. Send: "Hello"
2. Watch confidence_check_router
3. See it route to clarification
4. Check generated question

### Viewing Tool Calls
1. Send: "Claim status?"
2. Watch call_claims_tool node
3. See API results in state
4. Check how response_agent uses it

## Keyboard Shortcuts

- `Space` - Run/Pause
- `S` - Step forward
- `R` - Reset
- `D` - Download state

## Tips

1. **Always check state** - Most bugs are state issues
2. **Use step mode** - Don't run full graph immediately
3. **Watch routers** - Conditional edges are tricky
4. **Check checkpoints** - See conversation history

## Troubleshooting

**Studio won't start?**
- Check langgraph.json is correct
- Ensure requirements.txt installed
- Try: `langgraph up --verbose`

**Graph not showing?**
- Check langgraph_agent.py path
- Ensure create_graph() works
- Test: `python -c "from langgraph_agent import create_graph; create_graph()"`

**Can't see state?**
- Click on node circles
- Enable "Show State" toggle
- Check state schema matches

You're ready to visualize and debug! 🎉
