"""
Overrides_api — Prior Authorization (PA) lookup domain module.

Mirrors the structure of Claims_search_api/ but targets the Overrides API
(/pss/myclaims/override/exp/v1/priorauth/search).

Public exports (lazy):
    call_overrides_tool_node — async LangGraph node
    is_overrides_query        — predicate used by _route_after_build_context
    build_override_prompt     — LLM prompt builder for the response_agent
    answer_overrides_query    — high-level convenience wrapper

NOTE: imports are intentionally NOT eagerly resolved here. Each consumer should
import directly from the submodule (e.g.
`from Overrides_api.overrides_node import call_overrides_tool_node`) so that
subsystem-level imports (Redis, services.llm_connection, etc.) only happen
when the consumer actually needs them. This mirrors the lazy-import discipline
used by Claims_search_api/.
"""

# Light-weight re-exports only — these submodules have NO subsystem imports.
from .intent_router import is_overrides_query, OVERRIDE_INTENTS

__all__ = [
    "is_overrides_query",
    "OVERRIDE_INTENTS",
]
