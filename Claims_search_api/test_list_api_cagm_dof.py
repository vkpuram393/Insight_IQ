"""
Test: Extract CAGM + DOF from Claims List API (ALL claims per claim number)

CAGM = Carrier, Account, Group, Member ID
DOF  = Date of Fill

This test calls the /utils/test-claims-list endpoint which returns ALL
claims/sequences for a given claim number, then extracts CAGM + DOF
from each one.

Usage:
    python tests/test_list_api_cagm_dof.py --claim 252862109057000

    # Multiple claim numbers:
    python tests/test_list_api_cagm_dof.py --claims 252862109057000,253152732536005

    # With custom server URL:
    python tests/test_list_api_cagm_dof.py --claim 252862109057000 --url http://127.0.0.1:8001
"""

import asyncio
import argparse
import json
import sys
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from api_utils import extract_list_api_response_structure, extract_member_cagm_from_response

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# HTTP Client - Calls the running server's list endpoint
# ============================================================================

async def call_claims_list_api(
    claim_number: str,
    auth_token: str,
    base_url: str = "http://127.0.0.1:8001"
) -> Dict[str, Any]:
    """
    Call the /utils/test-claims-list endpoint to get ALL claims for a claim number.
    
    Args:
        claim_number: 15-digit claim number
        auth_token: Bearer token for authentication
        base_url: Server base URL
        
    Returns:
        The full API response dict with all claims
    """
    import httpx
    
    url = f"{base_url}/utils/test-claims-list"
    headers = {
        "Content-Type": "application/json",
        "Authorization": auth_token
    }
    body = {
        "claim_number": claim_number
    }
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        print(f"\n🔍 Fetching ALL claims for claim number {claim_number}...")
        response = await client.post(url, json=body, headers=headers)
        response.raise_for_status()
        return response.json()


# ============================================================================
# Main Test Runner
# ============================================================================

async def run_test(claim_numbers: List[str], auth_token: str, base_url: str):
    """Run CAGM + DOF extraction for all sequences under each claim number."""
    
    all_results = {}  # claim_number -> list of CAGM+DOF records
    
    for claim_number in claim_numbers:
        try:
            # Call the list API to get ALL claims for this claim number
            response = await call_claims_list_api(
                claim_number=claim_number,
                auth_token=auth_token,
                base_url=base_url
            )
            
            total_claims = response.get("total_claims", 0)
            claims = response.get("claims", [])
            raw_keys = response.get("raw_response_keys", "unknown")
            found_key = response.get("found_in_key", "unknown")
            
            print(f"  📦 Raw response keys: {raw_keys}")
            print(f"  📦 Claims found in key: '{found_key}'")
            
            if total_claims == 0 or not claims:
                print(f"  ⚠️  No claims found for claim number {claim_number}")
                # Dump raw response to help debug
                debug_file = os.path.join(os.path.dirname(__file__), f"debug_raw_response_{claim_number}.json")
                with open(debug_file, "w") as f:
                    json.dump(response, f, indent=2, default=str)
                print(f"  📁 Raw response dumped to: {debug_file}")
                all_results[claim_number] = [{
                    "claim_number": claim_number,
                    "error": f"No claims found (response keys: {raw_keys})"
                }]
                continue
            
            print(f"  📋 Found {total_claims} claim(s) for {claim_number}")
            
            # Debug: show first claim keys
            if claims and isinstance(claims[0], dict):
                print(f"  📦 First claim keys: {list(claims[0].keys())}")
            
            # Extract CAGM + DOF from EACH claim in the list
            records = []
            for idx, claim in enumerate(claims):
                record = extract_member_cagm_from_response(claim)
                records.append(record)
                seq = record.get("sequence_number", "?")
                status = record.get("claim_status", "?")
                carrier = record.get("carrier_id", "?")
                print(f"    ✅ Seq {seq} - Status: {status} - Carrier: {carrier} - {record.get('drug_name', 'N/A')}")
            
            all_results[claim_number] = records
            
            # Save raw response for reference
            debug_file = os.path.join(os.path.dirname(__file__), f"debug_raw_response_{claim_number}.json")
            with open(debug_file, "w") as f:
                json.dump(response, f, indent=2, default=str)
            print(f"  📁 Raw response saved to: {debug_file}")
            
        except Exception as e:
            print(f"  ❌ Failed for claim {claim_number}: {e}")
            import traceback
            traceback.print_exc()
            all_results[claim_number] = [{
                "claim_number": claim_number,
                "error": str(e)
            }]
    
    # Print results for each claim number
    for claim_number, records in all_results.items():
        print(format_cagm_dof_table(claim_number, records))
    
    # Save as JSON
    output_file = os.path.join(os.path.dirname(__file__), "cagm_dof_results.json")
    with open(output_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "claim_numbers_queried": len(claim_numbers),
            "results": {
                cn: {
                    "total_sequences": len(recs),
                    "records": recs
                }
                for cn, recs in all_results.items()
            }
        }, f, indent=2)
    print(f"\n📁 Results saved to: {output_file}")
    
    return all_results


def parse_args():
    parser = argparse.ArgumentParser(description="Extract CAGM + DOF from Claims List API (ALL sequences)")
    parser.add_argument("--claim", type=str, help="Single claim number (returns all sequences)")
    parser.add_argument("--claims", type=str, help="Multiple claim numbers comma-separated: 'claim1,claim2'")
    parser.add_argument("--token", type=str, help="Bearer token (or set AUTH_TOKEN env var)")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8001", help="Server base URL")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Get auth token
    auth_token = args.token or os.environ.get("AUTH_TOKEN", "")
    if not auth_token:
        print("❌ No auth token provided. Use --token or set AUTH_TOKEN env var.")
        print("   Example: python tests/test_list_api_cagm_dof.py --claim 252862109057000 --token 'Bearer eyJ...'")
        sys.exit(1)
    
    # Ensure token has Bearer prefix
    if not auth_token.startswith("Bearer "):
        auth_token = f"Bearer {auth_token}"
    
    # Parse claim numbers
    claim_numbers = []
    if args.claims:
        claim_numbers = [cn.strip() for cn in args.claims.split(",") if cn.strip()]
    elif args.claim:
        claim_numbers.append(args.claim)
    else:
        # Default test claim
        claim_numbers.append("252862109057000")
    
    print(f"🏥 Claims API - CAGM + DOF Extractor (ALL Sequences)")
    print(f"   Server: {args.url}")
    print(f"   Claim numbers to query: {len(claim_numbers)}")
    for cn in claim_numbers:
        print(f"     - {cn}")
    
    # Run
    results = asyncio.run(run_test(claim_numbers, auth_token, args.url))
    
    # Summary
    total_claims = sum(len(recs) for recs in results.values())
    total_errors = sum(1 for recs in results.values() for r in recs if r.get("error"))
    
    if total_errors:
        print(f"\n⚠️  {total_errors} error(s) across {total_claims} total claim(s)")
        sys.exit(1)
    else:
        print(f"\n✅ Successfully extracted CAGM+DOF for {total_claims} claim(s) across {len(claim_numbers)} claim number(s)")
        sys.exit(0)
