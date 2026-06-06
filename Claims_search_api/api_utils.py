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

# Log at startup so pod logs immediately show whether the key was injected
_configured_key = settings.claims_history_search_x_api_key
if _configured_key:
    logger.info("[ClaimsSearch] Startup: CLAIMS_HISTORY_SEARCH_X_API_KEY loaded (ends ...%s)", _configured_key[-4:])
else:
    logger.warning("[ClaimsSearch] Startup: CLAIMS_HISTORY_SEARCH_X_API_KEY is NOT set — will fall back to request x-api-key")


def _build_headers(bearer_token, x_api_key, x_clientrefid):
    auth = (bearer_token or "").strip()
    if auth and not auth.lower().startswith("bearer "):
        auth = "Bearer " + auth
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
    """Extract member_id from the claim_list_api response (claims[0].member.memberId)."""
    claims = (list_response or {}).get("claims") or []
    if not claims:
        return None
    member = (claims[0] or {}).get("member") or {}
    return member.get("memberId") or member.get("cardholderId")


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
#     CLAIM_ID = os.getenv("CLAIM_ID", "260302639954275")
#     BEARER_TOKEN = os.getenv("BEARER_TOKEN", "Bearer eyJhbGciOiJSUzI1NiIsImtpZCI6InRBSjlZYmlRbXFwYm9vQU5razdsR2RJd04zdER0NnJNUkxYSG0xWUE4SlUiLCJ0eXAiOiJKV1QifQ.eyJhdWQiOiJkZWYzNWU5Ni0xZWIwLTQ4NzctOTdlMS01MTE5OTJhMjQ4YmMiLCJpc3MiOiJodHRwczovL25nYW15cGJtbm9ucHJvZC5iMmNsb2dpbi5jb20vMmEyNzYyZTctZDU2Zi00N2FiLWE3MmEtZGYwMTAyYzQ1NzlhL3YyLjAvIiwiZXhwIjoxNzc4MzI3OTk0LCJuYmYiOjE3NzgzMjQzOTQsInRpZCI6ImZhYmI2MWI4LTNhZmUtNGU3NS1iOTM0LWE0N2Y3ODJiOGNkNyIsImdpdmVuX25hbWUiOiJ2YW1zaGkiLCJmYW1pbHlfbmFtZSI6ImtyaXNobmEiLCJlbWFpbCI6IlZhbXNoaS5LcmlzaG5hMkBjdnNoZWFsdGguY29tIiwibmFtZSI6ImtyaXNobmEsIHZhbXNoaSIsImlkcCI6Imh0dHBzOi8vbG9naW4ubWljcm9zb2Z0b25saW5lLmNvbS9mYWJiNjFiOC0zYWZlLTRlNzUtYjkzNC1hNDdmNzgyYjhjZDcvdjIuMCIsInN1YiI6IjJiOGJmNmQ3LTk5MzYtNGVjMS1hYzcwLTliMDlkN2MyZGI2NiIsIm5vbmNlIjoiY1hScFpHOVBUWEJPWldzNVFqSmpWVnB0TWtoRlMzaEJiVVYzTjJ4RFJUUk5VbTEtVDBGek9UVjJVSFZrIiwic2NwIjoiVXNlci5SZWFkIiwiYXpwIjoiZGVmMzVlOTYtMWViMC00ODc3LTk3ZTEtNTExOTkyYTI0OGJjIiwidmVyIjoiMS4wIiwiaWF0IjoxNzc4MzI0Mzk0fQ.p17WhSGWrfrsPUIORQJRCzIA9KXQEiEN4qmqvx4TjLEjdUtEM0_Q7epAgMvU58Z3tjSXmxKItpK3p_WgGibnoAjEwv-Rw2v_mmOgC0G1Optjvr2nhbm8kuR6I23Px3c1qydcmko6aHAqqXiE2NEplbVOf_wZEqo7Xv0TbtOyN4QBz_VrOGcHeLile80wVCMyhKPGmBiRAQPWCagavEGy1IfnfUASvGrn-BLowcj7sMC9-cP9I9413xsbTH6Vu9W7NStNNWA8M4j-feEGMof3hBCp9l9jSdcpnXC3rlZI9eyZkZfYEcKUdTODlsozX9Nc8IM4_ZeTSIe-ec0AWkdWog")
#     X_API_KEY = os.getenv("X_API_KEY", "")
#     X_CLIENTREFID = os.getenv("X_CLIENTREFID", "b732784c-c490-46f7-a5e2-d595ca57bdd5")

#     async def main():
#         result = await extract_list_api_response_structure(CLAIM_ID, BEARER_TOKEN, X_API_KEY, X_CLIENTREFID)
#         print(_json.dumps(result, indent=2))

#     asyncio.run(main())