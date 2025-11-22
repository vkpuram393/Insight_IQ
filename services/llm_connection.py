# llm_connection.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import os
from typing import Any, Dict, List, Optional
from google import genai
from google.genai import types
from config.config import settings

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
    thinking_budget: int = 0
    include_thoughts: bool = False
    safety_thresholds: Optional[Dict[str, str]] = None  # ← v2-friendly
    model: str = Field(default=MODEL_ID, description="e.g., 'gemini-2.5-flash'")


class GenerateResponse(BaseModel):
    text: str
    # model: str
    # usage: Optional[Dict[str, Any]] = None
    # raw_candidates: Optional[List[Dict[str, Any]]] = None
    # raw_response: Optional[Dict[str, Any]] = None  # optional, if you want the whole thing


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

    # Optional thinking config
    thinking_config = (
        types.ThinkingConfig(
            thinking_budget=req.thinking_budget,
            include_thoughts=req.include_thoughts,
        )
        if (req.thinking_budget or req.include_thoughts) else None
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

    # Return only JSON-friendly content
    return GenerateResponse(
        text=resp.text or "",
        # usage=(resp.usage_metadata.model_dump(mode="json")
        #        if getattr(resp, "usage_metadata", None) else None),
        # raw_candidates=([c.model_dump(mode="json") for c in resp.candidates]
        #                 if getattr(resp, "candidates", None) else None),
        # raw_response=resp.model_dump(mode="json"),
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
