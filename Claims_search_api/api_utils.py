"""Claims_search_api.api_utils - 2-step claims search via internal API gateway."""

import json as _json
import logging
import uuid
from typing import Any, Dict, List, Optional

import httpx

from config.config import settings

logger = logging.getLogger(__name__)

# Single endpoint: search by member ID or claim ID directly
_SEARCH_URL = settings.claims_history_search_url
_LIST_URL = settings.claim_list_api
MAX_RAW_CLAIMS = 500
_HTTP_TIMEOUT = 60.0


def log_startup_config() -> None:
    """
    Log Claims Search API config at pod startup.
    Call this from main.py startup_event so it appears in pod startup logs
    rather than at the time of the first request.
    """
    _configured_key = settings.claims_history_search_x_api_key
    if _configured_key:
        logger.info(
            "[ClaimsSearch] Startup: CLAIMS_HISTORY_SEARCH_X_API_KEY loaded (ends ...%s)",
            _configured_key[-4:],
        )
    else:
        logger.warning(
            "[ClaimsSearch] Startup: CLAIMS_HISTORY_SEARCH_X_API_KEY is NOT set"
            " — will fall back to request x-api-key"
        )
    logger.info("[ClaimsSearch] Startup: LIST_URL  = %s", _LIST_URL)
    logger.info("[ClaimsSearch] Startup: SEARCH_URL = %s", _SEARCH_URL)


def _build_headers(bearer_token, x_api_key, x_clientrefid):
    auth = (bearer_token or "").strip()
    if auth and not auth.lower().startswith("bearer "):
        auth = "Bearer " + auth
    # Log token presence/prefix at INFO so DEV/QA pod logs capture it
    if auth:
        logger.info("[ClaimsSearch] Auth token present (prefix: %s...)", auth[:20])
    else:
        logger.warning("[ClaimsSearch] Auth token is EMPTY — request will 401. "
                       "In DEV/QA: APIGEE strips Authorization header — "
                       "UI must send auth_token inside the request body JSON.")
    correlation_id = f"CVS-{uuid.uuid4()}"
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": auth,
        "x-correlation-id": correlation_id,
        "x-clientrefid": (x_clientrefid).strip(),
    }
    if x_api_key:
        headers["x-api-key"] = x_api_key.strip()
    return headers


def _sort_and_cap_claims(claims):
    if not claims:
        return []
    def _fd(c):
        return (c.get("claimInformation") or {}).get("fillDate") or ""
    s = sorted(claims, key=_fd, reverse=True)
    if len(s) > MAX_RAW_CLAIMS:
        logger.info("[ClaimsSearch] Capping %d -> %d", len(s), MAX_RAW_CLAIMS)
        s = s[:MAX_RAW_CLAIMS]
    return s


async def _post(url, body, headers, label):
    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
        logger.info("[ClaimsSearch] -> POST %s (%s)", url, label)
        logger.info("[ClaimsSearch] Body: %s", _json.dumps(body))
        r = await client.post(url, json=body, headers=headers)
        logger.info("[ClaimsSearch] Response: %s", r.status_code)
        if r.status_code >= 400:
            logger.error("[ClaimsSearch] %s failed: %s | %s", label, r.status_code, r.text[:1000])
        r.raise_for_status()
        return r.json()

def _extract_member_id_from_list(list_response):
    """
    Extract member_id or cardholderId from claim_list_api response.

    Expected path:
    claimList[0].primary.beneficiary.memberId
    """

    try:
        claim_list = (list_response or {}).get("claimList") or []
        if not claim_list:
            return None

        primary = (claim_list[0] or {}).get("primary") or {}
        beneficiary = primary.get("beneficiary") or {}

        # Prefer memberId, fallback to cardholderId
        return beneficiary.get("memberId") or beneficiary.get("cardholderId")

    except Exception as e:
        logger.error("[ClaimsSearch] Failed to extract memberId: %s", str(e))
        return None



async def fetch_claim_list(claim_id, bearer_token, x_api_key, x_clientrefid):
    """Step 1: call claim_list_api with claim_id to resolve member_id."""
    effective_key = settings.claims_history_search_x_api_key or x_api_key
    _key_source = "config (CLAIMS_HISTORY_SEARCH_X_API_KEY)" if settings.claims_history_search_x_api_key else "request header"
    logger.info("[ClaimsSearch] Step 1 x-api-key source: %s | key set: %s", _key_source, bool(effective_key))
    headers = _build_headers(bearer_token, effective_key, x_clientrefid)
    #body = {"claimsRequest": {"claimId": str(claim_id).strip()}}
    body = {"claimsRequest":{"claimId":str(claim_id).strip()},"additionalRequestInfo":{"claimType":"","environment":""}}
    return await _post(_LIST_URL, body, headers, "claim list claim_id=" + str(claim_id))


async def fetch_claims_by_member(member_id, bearer_token, x_api_key, x_clientrefid):
    """Step 2: call claims_history_search_url with member_id to get full claim history."""
    effective_key = settings.claims_history_search_x_api_key or x_api_key
    _key_source = "config (CLAIMS_HISTORY_SEARCH_X_API_KEY)" if settings.claims_history_search_x_api_key else "request header"
    logger.info("[ClaimsSearch] Step 2 x-api-key source: %s | key set: %s", _key_source, bool(effective_key))
    headers = _build_headers(bearer_token, effective_key, x_clientrefid)
    body = {
        "claimListsearch": {"key": {"member": str(member_id).strip()}},
        "additionalRequestInfo": {"claimType": "", "environment": ""},
    }
    return await _post(_SEARCH_URL, body, headers, "search member=" + str(member_id))


async def extract_list_api_response_structure(claim_id, bearer_token, x_api_key, x_clientrefid, *, member_id=None):
    """
    2-step flow:
      Step 1 — call settings.claim_list_api with claim_id → resolve member_id.
      Step 2 — call settings.claims_history_search_url with member_id → full history.
    If member_id is supplied directly, Step 1 is skipped.
    """
    mid = (member_id or "").strip()
    if not mid:
        cid = (claim_id or "").strip()
        if not cid:
            raise ValueError("either claim_id or member_id is required")
        logger.info("[ClaimsSearch] Step 1 — resolving member_id for claim_id=%r", cid)
        list_response = await fetch_claim_list(cid, bearer_token, x_api_key, x_clientrefid)
        #print("[ClaimsSearch] Step 1 response: %s", _json.dumps(list_response, indent=2))
        mid = _extract_member_id_from_list(list_response)
        if not mid:
            raise ValueError(f"member_id not found in claim list response for claim_id={cid!r}")
        logger.info("[ClaimsSearch] Step 1 resolved member_id=%r", mid)

    logger.info("[ClaimsSearch] Step 2 — fetching history for member_id=%r", mid)
    result = await fetch_claims_by_member(mid, bearer_token, x_api_key, x_clientrefid)
    raw = result.get("claims") or []
    logger.info("[ClaimsSearch] -> %d claim(s) for member_id %s", len(raw), mid)
    capped = _sort_and_cap_claims(raw)
    return {**result, "claims": capped, "totalCount": len(capped)}


def extract_member_cagm_from_response(response):
    out = {"member": None, "memberId": None, "carrierId": None, "accountId": None,
           "groupId": None, "firstName": None, "lastName": None, "dateOfBirth": None}
    try:
        claims = (response or {}).get("claims") or []
        if not claims:
            return out
        member = (claims[0] or {}).get("member") or {}
        mid = member.get("memberId") or member.get("member") or member.get("cardholderId")
        out["member"] = mid
        out["memberId"] = mid
        out["carrierId"] = member.get("carrierId")
        out["accountId"] = member.get("accountId")
        out["groupId"] = member.get("groupId")
        out["firstName"] = member.get("firstName")
        out["lastName"] = member.get("lastName")
        out["dateOfBirth"] = member.get("dateOfBirth")
        return out
    except Exception as e:
        out["error"] = str(e)
        return out
    
# if __name__ == "__main__":
#     import asyncio
#     import os

#     # Example usage
#     CLAIM_ID = os.getenv("CLAIM_ID", "261587613904003")
#     BEARER_TOKEN = os.getenv("BEARER_TOKEN", "")
#     X_API_KEY = os.getenv("X_API_KEY", "")
#     X_CLIENTREFID = os.getenv("X_CLIENTREFID", "")

#     async def main():
#         result = await extract_list_api_response_structure(CLAIM_ID, BEARER_TOKEN, X_API_KEY, X_CLIENTREFID)
#         print(_json.dumps(result, indent=2))

#     asyncio.run(main())
