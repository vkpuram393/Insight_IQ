"""
Overrides_api.api_utils — two-step Prior Authorization (PA) lookup.

Mirrors Claims_search_api/api_utils.py.

Step 1: claim_id  -> claims_list_api -> member CAGM (carrier/account/group/memberId)
Step 2: CAGM      -> Overrides API   -> PA records

Step 1 reuses Claims_search_api.api_utils.fetch_claim_list (same CVS claim_list_api
endpoint already used by the claims-search pipeline). Step 2 is the new POST
to /pss/myclaims/override/exp/v1/priorauth/search.

When the Overrides API errors (5xx, timeout) and settings.enable_api_fallback is
True, a static fallback dataset (config/overriders.json) is returned so the
downstream LLM still has data to render.
"""

import json as _json
import logging
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import httpx

from config.config import settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration (env-overridable via settings)
# ─────────────────────────────────────────────────────────────────────────────

def _cfg(attr: str, default: Any) -> Any:
    """Read a settings field with a hard-coded default fallback."""
    return getattr(settings, attr, default)


def _overrides_url() -> str:
    base = _cfg("overrides_api_base_url", "https://internal-sit1-apix.cvshealth.com")
    path = _cfg("overrides_api_path", "/pss/myclaims/override/exp/v1/priorauth/search")
    return base.rstrip("/") + path


def _id_source() -> str:
    return _cfg("overrides_id_source", "6003")


def _order_by() -> int:
    return _cfg("overrides_order_by", 4)


def _follow_me() -> bool:
    return bool(_cfg("overrides_enable_follow_me_logic", True))


def _http_timeout() -> float:
    return float(_cfg("overrides_api_timeout_seconds", 30.0))


_FALLBACK_FILE = Path(__file__).parent.parent / "config" / "overriders.json"


def log_startup_config() -> None:
    """Mirrors Claims_search_api.api_utils.log_startup_config — call from main.py startup."""
    logger.info("[OverridesAPI] Startup: BASE_URL = %s", _overrides_url())
    logger.info("[OverridesAPI] Startup: idSource=%s orderBy=%s followMe=%s",
                _id_source(), _order_by(), _follow_me())
    api_key = _cfg("overrides_api_x_api_key", "") or ""
    if api_key:
        logger.info("[OverridesAPI] Startup: OVERRIDES_API_X_API_KEY loaded (ends ...%s)",
                    api_key[-4:])
    else:
        logger.warning("[OverridesAPI] Startup: OVERRIDES_API_X_API_KEY not set — "
                       "will fall back to request x-api-key header.")


# ─────────────────────────────────────────────────────────────────────────────
# Header builder (mirrors Claims_search_api/api_utils._build_headers)
# ─────────────────────────────────────────────────────────────────────────────

def _build_headers(bearer_token: str, x_api_key: str, x_clientrefid: str) -> Dict[str, str]:
    auth = (bearer_token or "").strip()
    if auth and not auth.lower().startswith("bearer "):
        auth = "Bearer " + auth
    if auth:
        logger.info("[OverridesAPI] Auth token present (prefix: %s...)", auth[:20])
    else:
        logger.warning("[OverridesAPI] Auth token EMPTY — request will 401 in production.")
    correlation_id = f"CVS-{uuid.uuid4()}"
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": auth,
        "x-correlation-id": correlation_id,
        "x-request-id": str(uuid.uuid4()),
        "x-clientrefid": (x_clientrefid or "").strip(),
    }
    if x_api_key:
        headers["x-api-key"] = x_api_key.strip()
    return headers


# ─────────────────────────────────────────────────────────────────────────────
# CAGM extraction — handles BOTH old (claimList[0].primary.beneficiary) and
# new (claims[0].member) response shapes.  The CHS extractor at
# Claims_search_api.api_utils._extract_member_id_from_list already does this;
# we extend it to also pull carrier/account/group + personCode.
# ─────────────────────────────────────────────────────────────────────────────

def _extract_cagm(list_response: Dict[str, Any]) -> Optional[Dict[str, Optional[str]]]:
    """
    Parse the claim-list API response and extract carrierId/accountId/groupId/memberId.

    Tries multiple nesting paths because CVS API response shape varies
    across environments (claimList[0].primary.beneficiary vs claims[0].member).

    Returns None when no path yields any of carrierId/accountId/groupId/memberId.
    """
    resp = list_response or {}

    # --- Path 1 (new): claims[0].member ---
    claims = resp.get("claims") or []
    if claims:
        first = claims[0] or {}
        member_block = first.get("member") or {}
        result = _build_cagm(member_block, fallback=first)
        if result:
            logger.info("[OverridesAPI] CAGM resolved via path 1 (claims.member)")
            return result

    # --- Path 2 (old): claimList[0].primary.{beneficiary, member, medD} ---
    claim_list = resp.get("claimList") or []
    if claim_list:
        first = claim_list[0] or {}
        primary = first.get("primary") or {}
        for block_key in ("beneficiary", "member", "medD"):
            member_block = primary.get(block_key) or {}
            if member_block:
                result = _build_cagm(member_block, fallback=first)
                if result:
                    logger.info("[OverridesAPI] CAGM resolved via path 2 (claimList.primary.%s)", block_key)
                    return result

    logger.warning("[OverridesAPI] CAGM not found in claim-list response (all paths missed).")
    return None


def _build_cagm(member_block: Dict[str, Any], *, fallback: Dict[str, Any]) -> Optional[Dict[str, Optional[str]]]:
    carrier_id = (
        member_block.get("carrierId") or member_block.get("carrier") or fallback.get("carrierId")
    )
    account_id = (
        member_block.get("accountId") or member_block.get("account") or fallback.get("accountId")
    )
    group_id = (
        member_block.get("groupId") or member_block.get("group") or fallback.get("groupId")
    )
    member_id = (
        member_block.get("memberId") or member_block.get("member")
        or member_block.get("cardholderId") or fallback.get("memberId")
    )
    person_code = member_block.get("personCode") or member_block.get("person")

    if not (carrier_id or account_id or group_id or member_id):
        return None
    return {
        "carrierId": carrier_id,
        "accountId": account_id,
        "groupId": group_id,
        "memberId": member_id,
        "personCode": person_code,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — resolve member CAGM from claim_id (reuses fetch_claim_list)
# ─────────────────────────────────────────────────────────────────────────────

async def get_member_cagm_from_claim(
    claim_number: str,
    bearer_token: str,
    x_api_key: str,
    x_clientrefid: str,
) -> Optional[Dict[str, Optional[str]]]:
    """
    Step 1 — call Claims_search_api.api_utils.fetch_claim_list to resolve member CAGM.

    Lazy-imports fetch_claim_list to avoid a hard dependency at module load time.
    """
    # Lazy import keeps load time light and avoids circular imports.
    from Claims_search_api.api_utils import fetch_claim_list

    cid = (claim_number or "").strip()
    if not cid:
        logger.error("[OverridesAPI] Step 1: claim_number is empty.")
        return None

    logger.info("[OverridesAPI] Step 1 — resolving member CAGM for claim_number=%r", cid)
    try:
        list_response = await fetch_claim_list(
            claim_id=cid,
            bearer_token=bearer_token,
            x_api_key=x_api_key,
            x_clientrefid=x_clientrefid,
        )
        cagm = _extract_cagm(list_response)
        if cagm:
            logger.info("[OverridesAPI] Step 1 resolved CAGM: carrierId=%r memberId=%r",
                        cagm.get("carrierId"), cagm.get("memberId"))
        else:
            logger.warning("[OverridesAPI] Step 1: CAGM not found in claim-list response.")
        return cagm
    except Exception as exc:
        logger.error("[OverridesAPI] Step 1 failed: %s\n%s", exc, traceback.format_exc())
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — POST CAGM to Overrides API
# ─────────────────────────────────────────────────────────────────────────────

def _build_overrides_payload(cagm: Dict[str, Optional[str]]) -> Dict[str, Any]:
    return {
        "idSource":               _id_source(),
        "carrier":                cagm.get("carrierId") or "",
        "account":                cagm.get("accountId") or "",
        "group":                  cagm.get("groupId") or "",
        "memberId":               cagm.get("memberId") or "",
        "orderBy":                _order_by(),
        "enableFollowMeLogic":    _follow_me(),
    }


def _load_fallback() -> Dict[str, Any]:
    """Load mock PA records from config/overriders.json."""
    try:
        with open(_FALLBACK_FILE, encoding="utf-8") as f:
            data = _json.load(f)
            logger.info("[OverridesAPI] Loaded fallback from %s", _FALLBACK_FILE)
            return data
    except Exception as exc:
        logger.warning("[OverridesAPI] Could not load fallback file: %s", exc)
        return {"priorAuthorizations": [], "_fallback_note": "fallback_file_unreadable"}


async def get_overrides_by_member(
    cagm: Dict[str, Optional[str]],
    bearer_token: str,
    x_api_key: str,
    x_clientrefid: str,
) -> Dict[str, Any]:
    """
    Step 2 — POST to the Overrides API with member CAGM.

    Falls back to config/overriders.json when:
      - The API returns 5xx or times out, AND
      - settings.enable_api_fallback is True.
    """
    url = _overrides_url()
    effective_key = _cfg("overrides_api_x_api_key", "") or x_api_key
    headers = _build_headers(bearer_token, effective_key, x_clientrefid)
    payload = _build_overrides_payload(cagm)

    logger.info("[OverridesAPI] Step 2 — POST %s", url)
    logger.info("[OverridesAPI] Payload: %s", _json.dumps(payload))

    try:
        async with httpx.AsyncClient(timeout=_http_timeout()) as client:
            resp = await client.post(url, json=payload, headers=headers)
            logger.info("[OverridesAPI] Response: %s", resp.status_code)
            if resp.status_code >= 400:
                logger.error("[OverridesAPI] Error body: %s", resp.text[:1000])
            resp.raise_for_status()
            return resp.json()

    except (httpx.TimeoutException, httpx.HTTPStatusError) as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        is_server_error = status is None or status >= 500
        if is_server_error and getattr(settings, "enable_api_fallback", True):
            logger.warning("[OverridesAPI] Server error / timeout — using fallback. Error: %s", exc)
            return _load_fallback()
        raise

    except Exception as exc:
        logger.error("[OverridesAPI] Unexpected error: %s\n%s", exc, traceback.format_exc())
        if getattr(settings, "enable_api_fallback", True):
            logger.warning("[OverridesAPI] Using fallback after unexpected error.")
            return _load_fallback()
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: full two-step orchestration
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_overrides_for_claim(
    claim_number: str,
    bearer_token: str,
    x_api_key: str,
    x_clientrefid: str,
) -> Dict[str, Any]:
    """
    End-to-end Step 1 + Step 2 in a single call.

    Returns:
        {
            "data": <raw Overrides API response>,
            "cagm": {...},
            "claim_id": "...",
            "step": "complete" | "step1_failed" | "step2_failed",
            "error": str (when step != "complete")
        }
    """
    cagm = await get_member_cagm_from_claim(claim_number, bearer_token, x_api_key, x_clientrefid)
    if not cagm:
        return {
            "data": {"priorAuthorizations": []},
            "cagm": None,
            "claim_id": claim_number,
            "step": "step1_failed",
            "error": "CAGM unresolved from claim_number",
        }

    try:
        data = await get_overrides_by_member(cagm, bearer_token, x_api_key, x_clientrefid)
    except Exception as exc:
        return {
            "data": {"priorAuthorizations": []},
            "cagm": cagm,
            "claim_id": claim_number,
            "step": "step2_failed",
            "error": f"{type(exc).__name__}: {exc}",
        }

    return {
        "data": data,
        "cagm": cagm,
        "claim_id": claim_number,
        "step": "complete",
        "error": "",
    }
