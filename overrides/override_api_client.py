"""
Override API client — Prior Authorization lookup.

Two-step flow:
  Step 1: Fetch claim list via claims_list_api (existing) → extract member CAGM
           (carrier / account / group / memberId / personCode)
  Step 2: POST to Overrides API with member CAGM → PA records

The Overrides API is a separate service from the claims API and uses a
different base URL and payload shape.
"""

import json
import uuid
import traceback
from typing import Any, Dict, Optional
from pathlib import Path

import httpx

from config.config import settings
from core.logger import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants (all overridable via env vars / Settings)
# ─────────────────────────────────────────────────────────────────────────────

def _cfg(attr: str, default: Any) -> Any:
    return getattr(settings, attr, default)


_OVERRIDES_BASE_URL = lambda: _cfg("overrides_api_base_url", "https://internal-sit1-apix.cvshealth.com")
_OVERRIDES_PATH = lambda: _cfg("overrides_api_path", "/pss/myclaims/override/exp/v1/priorauth/search")
_ID_SOURCE = lambda: _cfg("overrides_id_source", "6003")
_ORDER_BY = lambda: _cfg("overrides_order_by", 4)
_FOLLOW_ME = lambda: _cfg("overrides_enable_follow_me_logic", True)

_HTTP_TIMEOUT = 30.0
_FALLBACK_FILE = Path(__file__).parent.parent / "config" / "overriders.json"


# ─────────────────────────────────────────────────────────────────────────────
# Header builder (mirrors Claims_search_api/api_utils._build_headers)
# ─────────────────────────────────────────────────────────────────────────────

def _build_headers(bearer_token: str, x_api_key: str, x_clientrefid: str) -> Dict[str, str]:
    auth = (bearer_token or "").strip()
    if auth and not auth.lower().startswith("bearer "):
        auth = "Bearer " + auth
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
# Step 1 helper: extract member CAGM from claim-list response
# ─────────────────────────────────────────────────────────────────────────────

def _extract_cagm(list_response: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """
    Parse the claim-list API response to find carrierId/accountId/groupId/memberId.

    Checks several known nesting paths because the CVS API response shape
    varies across environments.
    """
    claims = (list_response or {}).get("claims") or []
    if not claims:
        return None

    first = claims[0] or {}

    # Try multiple nested paths
    member_block = (
        first.get("member")
        or (first.get("primary") or {}).get("member")
        or (first.get("primary") or {}).get("beneficiary")
        or (first.get("primary") or {}).get("medD")
        or {}
    )

    carrier_id = (
        member_block.get("carrierId")
        or member_block.get("carrier")
        or first.get("carrierId")
    )
    account_id = (
        member_block.get("accountId")
        or member_block.get("account")
        or first.get("accountId")
    )
    group_id = (
        member_block.get("groupId")
        or member_block.get("group")
        or first.get("groupId")
    )
    member_id = (
        member_block.get("memberId")
        or member_block.get("member")
        or member_block.get("cardholderId")
        or first.get("memberId")
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
# Step 1: Resolve member CAGM from a claim number
# ─────────────────────────────────────────────────────────────────────────────

async def get_member_cagm_from_claim(
    claim_number: str,
    bearer_token: str,
    x_api_key: str,
    x_clientrefid: str,
) -> Optional[Dict[str, str]]:
    """
    Step 1 — call the existing claims-list API to resolve member CAGM.

    Reuses the same Step-1 endpoint (claim_list_api) already used by
    Claims_search_api.api_utils.fetch_claim_list — no new API contract.
    """
    from Claims_search_api.api_utils import fetch_claim_list

    logger.info(f"[OverrideAPI] Step 1 — resolving member CAGM for claim_number={claim_number!r}")
    try:
        list_response = await fetch_claim_list(
            claim_id=claim_number,
            bearer_token=bearer_token,
            x_api_key=x_api_key,
            x_clientrefid=x_clientrefid,
        )
        cagm = _extract_cagm(list_response)
        if cagm:
            logger.info(f"[OverrideAPI] Step 1 resolved CAGM: carrierId={cagm.get('carrierId')!r}")
        else:
            logger.warning("[OverrideAPI] Step 1: CAGM not found in claim list response")
        return cagm
    except Exception as exc:
        logger.error(f"[OverrideAPI] Step 1 failed: {exc}\n{traceback.format_exc()}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Call the Overrides API with member CAGM
# ─────────────────────────────────────────────────────────────────────────────

def _build_overrides_payload(cagm: Dict[str, str]) -> Dict[str, Any]:
    """Build the POST body for the Overrides API from a CAGM dict."""
    return {
        "idSource": _ID_SOURCE(),
        "carrier": cagm.get("carrierId", ""),
        "account": cagm.get("accountId", ""),
        "group": cagm.get("groupId", ""),
        "memberId": cagm.get("memberId", ""),
        "orderBy": _ORDER_BY(),
        "enableFollowMeLogic": _FOLLOW_ME(),
    }


def _load_fallback() -> Dict[str, Any]:
    """Load mock PA records from config/overriders.json."""
    try:
        with open(_FALLBACK_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning(f"[OverrideAPI] Could not load fallback file: {exc}")
        return {"priorAuthorizations": []}


async def get_overrides_by_member(
    cagm: Dict[str, str],
    bearer_token: str,
    x_api_key: str,
    x_clientrefid: str,
) -> Dict[str, Any]:
    """
    Step 2 — POST to the Overrides API with member CAGM.

    Falls back to config/overriders.json when:
      - The API returns a 5xx error or times out, AND
      - settings.enable_api_fallback is True

    Returns the parsed JSON response (or fallback data).
    """
    url = _OVERRIDES_BASE_URL().rstrip("/") + _OVERRIDES_PATH()
    headers = _build_headers(bearer_token, x_api_key, x_clientrefid)
    payload = _build_overrides_payload(cagm)

    logger.info(f"[OverrideAPI] Step 2 — POST {url}")
    logger.info(f"[OverrideAPI] Payload: {json.dumps(payload)}")

    try:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            resp = await client.post(url, json=payload, headers=headers)
            logger.info(f"[OverrideAPI] Response: {resp.status_code}")
            if resp.status_code >= 400:
                logger.error(f"[OverrideAPI] Error body: {resp.text[:1000]}")
            resp.raise_for_status()
            return resp.json()

    except (httpx.TimeoutException, httpx.HTTPStatusError) as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        is_server_error = status is None or (status is not None and status >= 500)

        if is_server_error and getattr(settings, "enable_api_fallback", True):
            logger.warning(
                f"[OverrideAPI] Server error / timeout — using fallback data "
                f"(enable_api_fallback=True). Error: {exc}"
            )
            return _load_fallback()
        raise

    except Exception as exc:
        logger.error(f"[OverrideAPI] Unexpected error: {exc}\n{traceback.format_exc()}")
        if getattr(settings, "enable_api_fallback", True):
            logger.warning("[OverrideAPI] Using fallback data after unexpected error")
            return _load_fallback()
        raise
