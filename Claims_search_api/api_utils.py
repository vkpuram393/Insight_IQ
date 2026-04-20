import httpx
from typing import Any, Dict

async def extract_list_api_response_structure(claim_id: str, bearer_token: str, x_api_key: str, x_clientrefid: str) -> Dict[str, Any]:
    """
    Fetches the LIST API response for a given claim ID.
    """
    url = "https://internal-sit1-apix.cvshealth.com/pss/myclaims/claims/exp/v1/claims/search"
    headers = {
        "Content-Type": "application/json",
        "Authorization": bearer_token,
        "x-api-key": x_api_key,
        "x-clientrefid": x_clientrefid
    }
    body = {"claimsRequest": {"claimId": claim_id}, "additionalRequestInfo": {"claimType": "", "environment": ""}}
    async with httpx.AsyncClient(timeout=60.0) as client:
        print(f"\n🔍 Fetching ALL claims for claim number {claim_id}...")
        response = await client.post(url, json=body, headers=headers)
        response.raise_for_status()
        return response.json()

def extract_member_cagm_from_response(response: Dict[str, Any]) -> Dict[str, str]:
    """
    Extracts member, carrierId, accountId, and groupId from the API response.
    Handles common response structures.
    """
    try:
        claim = response.get('claims', [{}])[0]
        member_id = claim.get('member', {}).get('memberId') or claim.get('member', {}).get('member')
        carrier_id = claim.get('member', {}).get('carrierId')
        account_id = claim.get('member', {}).get('accountId')
        group_id = claim.get('member', {}).get('groupId')
        return {
            "member": member_id,
            "carrierId": carrier_id,
            "accountId": account_id,
            "groupId": group_id
        }
    except Exception as e:
        return {"error": str(e)}
