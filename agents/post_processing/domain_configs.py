"""
Domain-config registry for the rendering engine.

The rendering agent (`myclaims_rendering_agent.py`) is fully
domain-agnostic. Every domain-specific knob (field aliases, status code
maps, null-as-zero rules, blocked fields) lives in a per-domain module:

    claims_rendering_config.py            -> "claims" (default)
    claim_history_rendering_config.py     -> "claim_history_search"
    member_rendering_config.py            -> FUTURE
    overrides_rendering_config.py         -> FUTURE

`get_config(domain)` returns the right module. Unknown domains fall back
to the "claims" config so legacy / new intents that haven't been
explicitly registered keep working with the safe default.

`resolve_domain(...)` infers the domain when the caller hasn't passed
one explicitly — preserving Code-1 behaviour for the pre-existing
claims-domain code paths.
"""

from typing import Any, Optional

from agents.post_processing import claims_rendering_config as _claims_cfg
from agents.post_processing import claim_history_rendering_config as _chs_cfg

_REGISTRY = {
    "claims":               _claims_cfg,
    "claim_history_search": _chs_cfg,
}


def get_config(domain: Optional[str]):
    """Return the rendering config module for *domain*.

    Falls back to the claims config if *domain* is None / empty / unknown
    so that existing claims-domain call sites stay byte-equivalent.
    """
    if not domain:
        return _claims_cfg
    return _REGISTRY.get(domain, _claims_cfg)


def resolve_domain(
    intent: str,
    tool_results: Any,
    state_domain: Optional[str] = None,
) -> str:
    """Infer the rendering domain from the request context.

    Priority:
      1. Explicit *state_domain* if present (multidomain classifier output).
      2. tool_results.data.is_claim_history_search flag (CHS pipeline).
      3. tool_name == "claims_search" / "claims_search_v2"        (CHS).
      4. Default: "claims".
    """
    if state_domain:
        return str(state_domain).strip() or "claims"

    if isinstance(tool_results, dict):
        data = tool_results.get("data") if isinstance(tool_results.get("data"), dict) else {}
        if isinstance(data, dict) and data.get("is_claim_history_search"):
            return "claim_history_search"
        tool_name = str(tool_results.get("tool_name") or "")
        if tool_name in ("claims_search", "claims_search_v2"):
            return "claim_history_search"

    return "claims"
