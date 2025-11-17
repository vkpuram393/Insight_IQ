# Orchestrator Node Architecture - AgentState vs Pydantic Models

## 🎯 Quick Answer

**The orchestrator node MUST:**
1. ✅ Accept `AgentState` as input (LangGraph requirement)
2. ✅ Return `Dict[str, Any]` that updates `AgentState` fields
3. ✅ Only use fields that exist in `AgentState`

**The orchestrator node CAN:**
1. ✅ Use Pydantic models **internally** for validation/processing
2. ✅ Define its own Pydantic models for internal logic
3. ✅ Convert Pydantic models to `Dict[str, Any]` before returning

**The orchestrator node CANNOT:**
1. ❌ Define its own separate object structure (must work within AgentState)
2. ❌ Return a Pydantic model directly (must return Dict)
3. ❌ Use fields that don't exist in AgentState

---

## 📋 The Requirement

### LangGraph Requirement

**All nodes in LangGraph MUST:**
- Accept `AgentState` (TypedDict) as input parameter
- Return `Dict[str, Any]` (partial update to AgentState)
- Only update fields that exist in `AgentState`

```python
# ✅ CORRECT - This is what LangGraph expects
async def orchestrator_node(state: AgentState) -> Dict[str, Any]:
    # Must accept AgentState
    # Must return Dict[str, Any]
    return {
        "text": "...",      # Must be a field in AgentState
        "uuid": "...",      # Must be a field in AgentState
        "domain": "..."     # Must be a field in AgentState
    }
```

```python
# ❌ WRONG - This won't work
async def orchestrator_node(input: MyCustomPydanticModel) -> MyCustomPydanticModel:
    # LangGraph won't call this - it expects AgentState!
    pass
```

---

## ✅ Solution: Use Pydantic Models Internally

You CAN use Pydantic models for validation/processing, but you must:
1. Accept `AgentState` as input
2. Convert to Pydantic model internally
3. Process/validate with Pydantic
4. Convert back to `Dict[str, Any]` for return

### Example: Orchestrator Node with Pydantic Model

```python
# nodes/orchestrator.py

from typing import Dict, Any
from pydantic import BaseModel, Field, field_validator
from state.schema import AgentState
from core.logger import get_logger
import uuid
from datetime import datetime

logger = get_logger(__name__)

# ========================================================================
# PYDANTIC MODEL FOR INTERNAL VALIDATION
# ========================================================================

class OrchestratorInput(BaseModel):
    """Pydantic model for orchestrator input validation"""
    text: str = Field(..., min_length=1, description="User input text")
    session_id: str = Field(..., min_length=1, description="Session identifier")
    user_info: Dict[str, Any] = Field(default_factory=dict, description="User metadata")
    uuid: str | None = Field(None, description="Request UUID")
    domain: str | None = Field(None, description="Domain context")
    
    @field_validator('text')
    @classmethod
    def validate_text(cls, v: str) -> str:
        """Normalize text"""
        return v.strip()
    
    @field_validator('domain')
    @classmethod
    def validate_domain(cls, v: str | None) -> str:
        """Set default domain if not provided"""
        return v or "claims"


class OrchestratorOutput(BaseModel):
    """Pydantic model for orchestrator output validation"""
    text: str = Field(..., description="Normalized text")
    uuid: str = Field(..., description="Request UUID")
    domain: str = Field(..., description="Domain context")
    normalized_data: Dict[str, Any] = Field(..., description="Pre-processed data")
    
    def to_agent_state_update(self) -> Dict[str, Any]:
        """Convert to AgentState update format"""
        return {
            "text": self.text,
            "uuid": self.uuid,
            "domain": self.domain,
            "metadata": {
                "normalized_data": self.normalized_data,
                "orchestrator_processed": True
            }
        }


# ========================================================================
# ORCHESTRATOR NODE (MUST ACCEPT AGENTSTATE)
# ========================================================================

async def orchestrator_node(state: AgentState) -> Dict[str, Any]:
    """
    Orchestrator Node - First node in the graph
    
    MUST accept AgentState (LangGraph requirement)
    CAN use Pydantic models internally for validation
    MUST return Dict[str, Any] that updates AgentState
    """
    logger.info("🎯 Orchestrator Node: Processing input")
    
    try:
        # ====================================================================
        # STEP 1: Extract from AgentState (required fields)
        # ====================================================================
        text = state.get("text", "")
        session_id = state.get("session_id", "unknown")
        user_info = state.get("user_info", {})
        existing_uuid = state.get("uuid")
        existing_domain = state.get("domain")
        
        # ====================================================================
        # STEP 2: Convert to Pydantic model for validation
        # ====================================================================
        # Create Pydantic model from AgentState fields
        orchestrator_input = OrchestratorInput(
            text=text,
            session_id=session_id,
            user_info=user_info,
            uuid=existing_uuid,
            domain=existing_domain
        )
        # Pydantic validates and normalizes here!
        
        # ====================================================================
        # STEP 3: Process with Pydantic model
        # ====================================================================
        # Generate UUID if not provided
        request_uuid = orchestrator_input.uuid or str(uuid.uuid4())
        
        # Determine domain (Pydantic already set default)
        domain = orchestrator_input.domain
        
        # Normalize text (Pydantic already did this)
        normalized_text = orchestrator_input.text
        
        # Create normalized data
        normalized_data = {
            "cleaned_text": normalized_text.lower(),
            "original_text": text,
            "language": "en",
            "timestamp": datetime.now().isoformat(),
            "normalized_at": datetime.now().isoformat()
        }
        
        # ====================================================================
        # STEP 4: Create output Pydantic model (for validation)
        # ====================================================================
        orchestrator_output = OrchestratorOutput(
            text=normalized_text,
            uuid=request_uuid,
            domain=domain,
            normalized_data=normalized_data
        )
        # Pydantic validates output structure here!
        
        # ====================================================================
        # STEP 5: Convert back to Dict for AgentState update
        # ====================================================================
        # MUST return Dict[str, Any] - LangGraph requirement!
        return orchestrator_output.to_agent_state_update()
        
    except Exception as e:
        logger.error(f"Orchestrator node error: {e}")
        # Return error state (still must be Dict[str, Any])
        return {
            "error": str(e),
            "metadata": {
                **state.get("metadata", {}),
                "orchestrator_error": True
            }
        }
```

---

## 🔍 Key Points

### 1. **Must Accept AgentState**

```python
# ✅ CORRECT
async def orchestrator_node(state: AgentState) -> Dict[str, Any]:
    pass

# ❌ WRONG
async def orchestrator_node(input: OrchestratorInput) -> OrchestratorOutput:
    pass
```

### 2. **Can Use Pydantic Internally**

```python
# ✅ CORRECT - Use Pydantic for validation
async def orchestrator_node(state: AgentState) -> Dict[str, Any]:
    # Convert AgentState to Pydantic model
    input_model = OrchestratorInput(**state)
    
    # Validate and process with Pydantic
    output_model = process_with_pydantic(input_model)
    
    # Convert back to Dict
    return output_model.model_dump()  # or .to_agent_state_update()
```

### 3. **Must Return Dict with AgentState Fields**

```python
# ✅ CORRECT - Only fields that exist in AgentState
return {
    "text": "...",           # ✅ Field in AgentState
    "uuid": "...",           # ✅ Field in AgentState
    "domain": "...",         # ✅ Field in AgentState
    "metadata": {...}        # ✅ Field in AgentState
}

# ❌ WRONG - Field doesn't exist in AgentState
return {
    "custom_field": "..."    # ❌ Not in AgentState!
}
```

---

## 📊 Complete Example

### Step 1: Define Pydantic Models (Internal Use)

```python
# core/orchestrator_models.py

from pydantic import BaseModel, Field

class OrchestratorInput(BaseModel):
    """Internal validation model"""
    text: str
    session_id: str
    user_info: Dict[str, Any]
    uuid: str | None = None
    domain: str | None = None

class OrchestratorOutput(BaseModel):
    """Internal validation model"""
    text: str
    uuid: str
    domain: str
    normalized_data: Dict[str, Any]
    
    def to_agent_state_update(self) -> Dict[str, Any]:
        """Convert to AgentState format"""
        return {
            "text": self.text,
            "uuid": self.uuid,
            "domain": self.domain,
            "metadata": {"normalized_data": self.normalized_data}
        }
```

### Step 2: Orchestrator Node (Must Use AgentState)

```python
# nodes/orchestrator.py

from state.schema import AgentState
from core.orchestrator_models import OrchestratorInput, OrchestratorOutput

async def orchestrator_node(state: AgentState) -> Dict[str, Any]:
    """
    Orchestrator node - MUST accept AgentState
    """
    # Extract from AgentState (only fields that exist)
    text = state.get("text", "")
    session_id = state.get("session_id", "")
    user_info = state.get("user_info", {})
    
    # Use Pydantic for validation (internal)
    input_model = OrchestratorInput(
        text=text,
        session_id=session_id,
        user_info=user_info,
        uuid=state.get("uuid"),
        domain=state.get("domain")
    )
    
    # Process with Pydantic model
    output_model = process_orchestrator_logic(input_model)
    
    # Convert back to Dict (AgentState format)
    return output_model.to_agent_state_update()
```

---

## 🎯 Summary

| Aspect | Requirement |
|--------|------------|
| **Input** | ✅ MUST accept `AgentState` (TypedDict) |
| **Internal Processing** | ✅ CAN use Pydantic models for validation |
| **Output** | ✅ MUST return `Dict[str, Any]` |
| **Output Fields** | ✅ MUST only use fields that exist in `AgentState` |
| **Custom Objects** | ❌ CANNOT define separate object structure |

### Pattern:

```
AgentState (input)
    ↓
Extract fields
    ↓
Pydantic Model (internal validation)
    ↓
Process/Validate
    ↓
Pydantic Model (output validation)
    ↓
Convert to Dict[str, Any]
    ↓
Return (updates AgentState)
```

---

## ✅ Best Practice

1. **Accept AgentState** - Required by LangGraph
2. **Use Pydantic internally** - For validation and type safety
3. **Return Dict** - Convert Pydantic model to Dict before returning
4. **Only AgentState fields** - Don't add new fields that don't exist in AgentState

This gives you:
- ✅ Type safety (Pydantic validation)
- ✅ LangGraph compatibility (AgentState input/output)
- ✅ Clean code (structured models internally)
- ✅ Flexibility (can change internal models without affecting graph)

