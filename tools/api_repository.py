from typing import List,Dict,Any
from core.node_models import API_REPOSITORY
from config.config import settings
from functools import lru_cache

# Get BASE_URL from .env via settings
BASE_URL = settings.swagger_url

# ============================================================================
# API REPOSITORY
# ============================================================================
@lru_cache(maxsize=1)
def get_api_repository()->List[API_REPOSITORY]:
    """
    Returns a list of API_REPOSITORY objects representing all external APIs the agent can dynamically route to.
    
    API Selection Logic (Entity-Based):
    - If claimNumber + claimSequence → /byclaimnumberandseq (detailed view, enriched 2-step flow)
    - If claimId only → /byclaimnumber (list/search view)
    """
    registry:List[API_REPOSITORY] = [
        API_REPOSITORY(
            name="get_claim_details",
            endpoint=settings.api_endpoint_claim_details,
            method="POST",
            required_entities=["claimNumber", "claimSequence"],  # Both required for details endpoint
            intent_keywords=["details", "claim details"],
            description="Fetch detailed claim info by claim number and sequence. Used when both claimNumber + claimSequence are provided.",
            body_template=lambda e: {
                "claimDetailsRequest": {
                    "claimNumber": e["claimNumber"],
                    "claimSequence": e.get("claimSequence", "1"),
                    "expMockFlag": "N"
                }
            }
        ),
        API_REPOSITORY(
            name="get_claim_list",
            endpoint=settings.api_endpoint_claim_list,
            method="POST",
            required_entities=["claimId"],  # Used when only claimId is provided (no sequence)
            intent_keywords=["claim_search", "find_claim", "lookup_claim", "status", "check", "track", "rejection", "rejected", "deny", "denied"],
            description="Search claim by claimId. Used when only claimId is provided (no sequence).",
            body_template=lambda e: {
                "claimsRequest": {
                    "claimId": e["claimId"]
                }
            }
        )
    ]
    for api in registry:
        api.full_url = f"{BASE_URL}{api.endpoint}"
    return registry