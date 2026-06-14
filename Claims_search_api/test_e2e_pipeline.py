"""
End-to-end test for the claims search pipeline.

Exercises the full flow:
  user_query → filter → format → response_agent (mock LLM)

Uses the embedded SAMPLE_RESPONSE so no actual API call is needed.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force mock LLM mode so test runs without Gemini credentials
os.environ["USE_MOCK_LLM"] = "true"

from Claims_search_api.search import generalized_claims_query
from Claims_search_api.llm_formatter import format_claims_for_llm
from Claims_search_api.api_utils import extract_member_cagm_from_response
from Claims_search_api.test_formatter import SAMPLE_RESPONSE


class MockClaimsResponseAgent:
    """Minimal mock that avoids importing services.llm_connection."""

    def generate_response(self, user_query, claims_context, member_info=None,
                          total_claims=0, filtered_claims=0):
        claim_count = claims_context.count("--- Claim ")
        if claim_count == 0:
            return "No claims were found matching your query."
        return (
            f"[MOCK] Found {claim_count} claim(s) matching your query: "
            f'"{user_query}". In production, the LLM would provide a '
            f"detailed conversational answer based on the claims data."
        )


def run_pipeline(user_query: str, api_response: dict) -> dict:
    """Simulate the full pipeline without HTTP or real API calls."""
    claims = api_response.get("claims", [])
    total_count = len(claims)

    # Step 1: Filter
    filtered = generalized_claims_query(claims, user_query)
    filtered_count = len(filtered)

    # Step 2: Format
    filtered_response = {**api_response, "claims": filtered, "totalCount": filtered_count}
    llm_context = format_claims_for_llm(filtered_response, user_query=None, is_member_history=True)

    # Step 3: Generate response (mock LLM)
    member_info = extract_member_cagm_from_response(api_response)
    agent = MockClaimsResponseAgent()
    response = agent.generate_response(
        user_query=user_query,
        claims_context=llm_context,
        member_info=member_info,
        total_claims=total_count,
        filtered_claims=filtered_count,
    )

    return {
        "query": user_query,
        "filtered_count": filtered_count,
        "total_count": total_count,
        "response": response,
        "context_size": len(llm_context),
    }


def main():
    test_queries = [
        "show me all claims with reject code 79",
        "When was LEVOTHYROXINE taken last for this member?",
        "how much did the member pay for LEVOTHYROXINE?",
        "show me all rejected claims",
        "show claims manufactured by MACLEODS",
        "show all generic drug claims",
        "which claims used prior authorization?",
        "show claims with 90 day supply",
        "show claims under plan LICS2",
        "show retail pharmacy claims",
    ]

    print("=" * 70)
    print("END-TO-END CLAIMS SEARCH PIPELINE TEST (Mock LLM)")
    print("=" * 70)
    print(f"Sample data: {len(SAMPLE_RESPONSE['claims'])} claims\n")

    all_pass = True
    for query in test_queries:
        result = run_pipeline(query, SAMPLE_RESPONSE)
        status = "PASS" if result["filtered_count"] > 0 else "WARN (0 matches)"
        if result["filtered_count"] == 0:
            all_pass = False

        print(f"{'PASS' if result['filtered_count'] > 0 else 'WARN':4s} | "
              f"Filtered: {result['filtered_count']}/{result['total_count']} | "
              f"Context: {result['context_size']:5d} chars | "
              f"Query: {query}")
        print(f"      Response: {result['response']}")
        print()

    print("=" * 70)
    if all_pass:
        print("ALL QUERIES RETURNED RESULTS - Pipeline OK")
    else:
        print("SOME QUERIES RETURNED 0 RESULTS (check sample data coverage)")
    print("=" * 70)


if __name__ == "__main__":
    main()
