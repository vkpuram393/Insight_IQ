# llm_connection.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import os
from typing import Any, Dict, List, Optional
from google import genai
from google.genai import types
from config.config import settings
from core.logger import get_logger

logger = get_logger(__name__)

# ------------------------------------------------------------------
# Config (env vars or defaults)
# ------------------------------------------------------------------
PROJECT_ID = settings.project_id
LOCATION = settings.location
MODEL_ID = settings.llm_model

# Create client for Gemini on Vertex AI
client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

# ------------------------------------------------------------------
# Models
# ------------------------------------------------------------------
class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="User text prompt")
    system_instruction: Optional[str] = Field(default=None)
    temperature: float = settings.llm_temperature
    top_p: float = settings.top_p
    max_output_tokens: int = settings.max_output_tokens
    include_thoughts: bool = False  # Enable thinking mode (gemini-2.5-flash supports this)
    safety_thresholds: Optional[Dict[str, str]] = None  # ← v2-friendly
    model: str = Field(default=MODEL_ID, description="e.g., 'gemini-2.5-flash'")


class GenerateResponse(BaseModel):
    """Enhanced response with metadata for truncation detection and debugging"""
    text: str
    finish_reason: str = "UNKNOWN"    # STOP, MAX_TOKENS, SAFETY, RECITATION, OTHER
    is_truncated: bool = False         # True if MAX_TOKENS detected
    thoughts: Optional[str] = None     # Chain of thought (Issue 2 - thinking mode)
    # Token usage metrics (Issue 3)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


# -----------------------------
# Core generator (reusable)
# -----------------------------
def _generate_core(req: GenerateRequest) -> GenerateResponse:
    # Build user content
    parts: List[types.Part] = [types.Part.from_text(text=req.prompt)]
    contents = [types.Content(role="user", parts=parts)]

    # Optional system instruction
    system_instruction = (
        [types.Part.from_text(text=req.system_instruction)]
        if req.system_instruction else None
    )

    # Optional thinking config (Issue 2: enables chain-of-thought visibility)
    # Note: google-genai SDK v1.0.0 ThinkingConfig only accepts include_thoughts parameter
    thinking_config = (
        types.ThinkingConfig(include_thoughts=True)
        if req.include_thoughts else None
    )

    # Optional safety settings
    safety_settings = None
    if req.safety_thresholds:
        safety_settings = [
            types.SafetySetting(category=cat, threshold=th)
            for cat, th in req.safety_thresholds.items()
        ]

    gen_config = types.GenerateContentConfig(
        temperature=req.temperature,
        top_p=req.top_p,
        max_output_tokens=req.max_output_tokens,
        safety_settings=safety_settings,
        system_instruction=system_instruction,
        thinking_config=thinking_config,
    )

    # Call the model
    resp = client.models.generate_content(
        model=req.model,
        contents=contents,
        config=gen_config,
    )

    # =========================================================================
    # CRITICAL FIX (Issue 1): Extract finish_reason and detect truncation
    # =========================================================================
    finish_reason = "UNKNOWN"
    is_truncated = False
    thoughts_text = None
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    
    if hasattr(resp, 'candidates') and resp.candidates:
        candidate = resp.candidates[0]
        
        # Extract finish_reason from candidate (Issue 2: Clean format)
        if hasattr(candidate, 'finish_reason'):
            fr = candidate.finish_reason
            # Extract clean enum name: "FinishReason.STOP" -> "STOP"
            finish_reason = fr.name if hasattr(fr, 'name') else str(fr).split('.')[-1]
            
            # Detect truncation (MAX_TOKENS means response was cut off)
            if finish_reason == 'MAX_TOKENS':
                is_truncated = True
                logger.warning(
                    f"⚠️ RESPONSE TRUNCATED! finish_reason={finish_reason}, "
                    f"response_length={len(resp.text or '')} chars"
                )
            elif finish_reason == 'SAFETY':
                logger.warning(f"🚫 Response blocked by safety: {finish_reason}")
            elif finish_reason == 'RECITATION':
                logger.warning(f"📚 Response blocked by recitation filter: {finish_reason}")
        
        # =====================================================================
        # Issue 2: Extract thoughts if thinking mode is enabled
        # =====================================================================
        if req.include_thoughts and hasattr(candidate, 'content') and candidate.content:
            thought_parts = []
            parts = candidate.content.parts or []
            
            for part in parts:
                # Check for thought content (robust - multiple attribute names)
                is_thought = getattr(part, 'thought', False) or getattr(part, 'thinking', False)
                if is_thought and hasattr(part, 'text') and part.text:
                    thought_parts.append(part.text)
            
            if thought_parts:
                thoughts_text = "\n".join(thought_parts)
                logger.info(f"🧠 Captured {len(thoughts_text)} chars of thinking from {len(thought_parts)} parts")
            else:
                logger.debug(f"🧠 Thinking mode enabled but no thought parts in response (model: {req.model})")
    
    # =========================================================================
    # Issue 3: Extract token usage from usage_metadata
    # =========================================================================
    if hasattr(resp, 'usage_metadata') and resp.usage_metadata:
        um = resp.usage_metadata
        prompt_tokens = getattr(um, 'prompt_token_count', 0) or 0
        completion_tokens = getattr(um, 'candidates_token_count', 0) or 0
        total_tokens = getattr(um, 'total_token_count', 0) or 0
        logger.debug(f"📊 Tokens: prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}")

    # Return enhanced response with metadata
    return GenerateResponse(
        text=resp.text or "",
        finish_reason=finish_reason,
        is_truncated=is_truncated,
        thoughts=thoughts_text,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens
    )


# -----------------------------
# Public convenience wrapper
# -----------------------------
def generate(prompt: str, **overrides) -> str:
    """
    Call the LLM with a simple string prompt and return the text.

    You can override any GenerateRequest field:
        generate("Hi", temperature=0.2, model="gemini-2.5-pro")
    """
    req = GenerateRequest(prompt=prompt, **overrides)
    return _generate_core(req).text
