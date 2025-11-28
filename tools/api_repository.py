from typing import List,Dict,Any
from core.node_models import API_REPOSITORY
from dotenv import load_dotenv
import os
from config.config import settings
from functools import lru_cache

BASE_URL = settings.swagger_url

# ============================================================================
# API REPOSITORY
# ============================================================================
@lru_cache(maxsize=1)
def get_api_repository()->List[API_REPOSITORY]:
    """
    Returns a list of API_REPOSITORY objects representing all external APIs the agent can dynamically route to
    """
    registry:List[API_REPOSITORY] = [
        API_REPOSITORY(
            name="get_claim_details",
            endpoint="/myclaims/claims/v1/details",
            method="POST",
            required_entities=["claimNumber", "claimSequence"],
            intent_keywords=["details","claim details"],
            description="Fetch claim details by claim number and sequence.",
            body_template=lambda e: {
                "claimDetailsRequest": {
                    "claimNumber": e["claimNumber"],
                    "claimSequence": e["claimSequence"],
                    "expMockFlag": e.get("expMockFlag", "N")
                }
            }
        ),
        API_REPOSITORY(
            name="get_claim_list",
            endpoint="/myclaims/claims/v1/list",
            method="POST",
            required_entities=["claimId"],
            intent_keywords= ["claim_search", "find_claim", "lookup_claim", "status", "check", "track", "rejection", "rejected", "deny", "denied"],
            description="Search claim by claimId (fetches summary, status, and rejection details).",
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