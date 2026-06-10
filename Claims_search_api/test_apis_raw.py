"""
Raw API test — tests Step 1 (byclaimnumber) and Step 2 (claims/search) independently.

Usage (from the Insight_IQ directory):
    python -m Claims_search_api.test_apis_raw

Edit the CONFIG block below to change claim_id / member_id / credentials.
"""
import asyncio
import json
import uuid

import httpx

# ─────────────────────────── CONFIG ──────────────────────────────────────────
BEARER       = "Bearer eyJhbGciOiJSUzI1NiIsImtpZCI6InRBSjlZYmlRbXFwYm9vQU5razdsR2RJd04zdER0NnJNUkxYSG0xWUE4SlUiLCJ0eXAiOiJKV1QifQ.eyJhdWQiOiJkZWYzNWU5Ni0xZWIwLTQ4NzctOTdlMS01MTE5OTJhMjQ4YmMiLCJpc3MiOiJodHRwczovL25nYW15cGJtbm9ucHJvZC5iMmNsb2dpbi5jb20vMmEyNzYyZTctZDU2Zi00N2FiLWE3MmEtZGYwMTAyYzQ1NzlhL3YyLjAvIiwiZXhwIjoxNzc4MzI2NjAxLCJuYmYiOjE3NzgzMjMwMDEsInRpZCI6ImZhYmI2MWI4LTNhZmUtNGU3NS1iOTM0LWE0N2Y3ODJiOGNkNyIsImdpdmVuX25hbWUiOiJ2YW1zaGkiLCJmYW1pbHlfbmFtZSI6ImtyaXNobmEiLCJlbWFpbCI6IlZhbXNoaS5LcmlzaG5hMkBjdnNoZWFsdGguY29tIiwibmFtZSI6ImtyaXNobmEsIHZhbXNoaSIsImlkcCI6Imh0dHBzOi8vbG9naW4ubWljcm9zb2Z0b25saW5lLmNvbS9mYWJiNjFiOC0zYWZlLTRlNzUtYjkzNC1hNDdmNzgyYjhjZDcvdjIuMCIsInN1YiI6IjJiOGJmNmQ3LTk5MzYtNGVjMS1hYzcwLTliMDlkN2MyZGI2NiIsIm5vbmNlIjoiY1hScFpHOVBUWEJPWldzNVFqSmpWVnB0TWtoRlMzaEJiVVYzTjJ4RFJUUk5VbTEtVDBGek9UVjJVSFZrIiwic2NwIjoiVXNlci5SZWFkIiwiYXpwIjoiZGVmMzVlOTYtMWViMC00ODc3LTk3ZTEtNTExOTkyYTI0OGJjIiwidmVyIjoiMS4wIiwiaWF0IjoxNzc4MzIzMDAxfQ.lrt9audwAg4bfF50ccFhGxkWt1C_QXidXVgiYokUAuu8gxSvLyhW2HicuEqDSVWqy7G_wLVABzvCtC5n0w5UoiAywsF1l9Ok5zaYoAbwLyKPBGFR5GA6TmHU-10ZuS8S0JpSRqJ_RJLv4mWxP2ASW1N_bllacLi6aqxrKIV8nkaYEbBOzXqPotY70if7gLsgHYCUoz_tGv1KlALtX_my9ubMBtjz-uioApdkwUN6iBoF6F9qXq2thOZpsZMNA_n2KuBB0dpDIIq_8SsDvroS1BaKqKgIj4jk67XiULEj3NtXccmVQzoE-Q5kGjcphMTIO9jqrxMud3VmaRsA0KEDfw"
X_API_KEY    = "fbbae75e-cd91-47a5-bb65-b68f525a66e3"
X_CLIENTREFID = "a6e14406-73fc-4d31-bfe8-89c3b0e287bb"

CLAIM_ID  = "260358603304176"   # 15-digit claim number for Step 1
MEMBER_ID = "4807045053"        # known-good member ID to test Step 2 directly
CORRELATION_ID = "7d5deec1-2719-4255-b537-3482010cf15017803"  # optional fixed correlation ID for testing
STEP1_URL = "https://internal-sit1-apix.cvshealth.com/pss/myclaims/claims/v1/claim/byclaimnumber"
STEP2_URL = "https://internal-sit1-apix.cvshealth.com/pss/myclaims/claims/exp/v1/claims/search"
# ─────────────────────────────────────────────────────────────────────────────


def _headers():
    cid = f"CVS-{uuid.uuid4()}"
    return {
        "accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": BEARER,
        #"x-correlation-id": cid,
        "x-clientrefid": X_CLIENTREFID,
        "x-api-key": X_API_KEY,
        "x-correlation-id": CORRELATION_ID
    }


def _print_response(label, r):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Status : {r.status_code}")
    print(f"  URL    : {r.url}")
    try:
        body = r.json()
        print(f"  Body   :\n{json.dumps(body, indent=2)[:3000]}")
    except Exception:
        print(f"  Body   : {r.text[:3000]}")
    print(f"{'='*60}")


async def test_step1_byclaimnumber():
    """Test Step 1: resolve claim_id -> member_id via byclaimnumber."""
    print(f"\n>>> STEP 1 — byclaimnumber")
    print(f"    URL      : {STEP1_URL}")
    body = {"claimsRequest": {"claimId": CLAIM_ID}}
    print(f"    Body     : {json.dumps(body)}")

    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(STEP1_URL, json=body, headers=_headers())
    _print_response("STEP 1 byclaimnumber", r)

    if r.status_code == 200:
        resp = r.json()
        claim_list = (resp.get("claimList") or [])
        if claim_list:
            beneficiary = ((claim_list[0] or {}).get("primary") or {}).get("beneficiary") or {}
            mid = beneficiary.get("memberId") or beneficiary.get("cardholderId")
            print(f"\n  >>> Extracted member_id = {mid!r}")
            return mid
    return None


async def test_step2_search_by_member(member_id):
    """Test Step 2: get full claim history by member_id."""
    print(f"\n>>> STEP 2 — claims/search (member={member_id})")
    print(f"    URL      : {STEP2_URL}")
    body = {
        "claimListsearch": {"key": {"member": str(member_id)}},
        "additionalRequestInfo": {"claimType": "", "environment": ""},
    }
    print(f"    Body     : {json.dumps(body)}")

    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(STEP2_URL, json=body, headers=_headers())
    _print_response("STEP 2 claims/search", r)

    if r.status_code == 200:
        claims = (r.json().get("claims") or [])
        print(f"\n  >>> Total claims returned = {len(claims)}")
    return r.status_code


async def main():
    print("=" * 60)
    print("  RAW API TEST — internal-sit1-apix.cvshealth.com")
    print("=" * 60)

    # ── Test Step 1 ──────────────────────────────────────────────
    resolved_member_id = await test_step1_byclaimnumber()

    # ── Test Step 2 with resolved member_id (if Step 1 worked) ──
    if resolved_member_id:
        await test_step2_search_by_member(resolved_member_id)
    else:
        print(f"\n  Step 1 failed or returned no member_id.")
        print(f"  Falling back to hardcoded MEMBER_ID={MEMBER_ID!r} for Step 2 test.")

    # ── Always test Step 2 with the hardcoded known-good member_id ──
    print(f"\n>>> STEP 2 (standalone) — using hardcoded MEMBER_ID={MEMBER_ID!r}")
    await test_step2_search_by_member(MEMBER_ID)


if __name__ == "__main__":
    asyncio.run(main())
