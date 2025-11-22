"""
External Service Integrations

This module contains integrations with external services:
- LLM services (Gemini)
- PII protection services
- Azure embeddings services
"""

# Re-export commonly used items for convenience
from services.llm_connection import client as gemini_client, GenerateRequest
from services.pii_protection import get_pii_service, PIIProtectionService, SafetyCheck
from services.azure_embeddings import get_azure_embeddings, get_embedding

__all__ = [
    "gemini_client",
    "GenerateRequest",
    "get_pii_service",
    "PIIProtectionService",
    "SafetyCheck",
    "get_azure_embeddings",
    "get_embedding",
]

