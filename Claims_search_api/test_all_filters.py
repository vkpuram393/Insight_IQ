"""Quick validation of all new filter scenarios."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Claims_search_api.search import generalized_claims_query
from Claims_search_api.test_formatter import SAMPLE_RESPONSE

queries = [
    ("claim number 260302639954275", "ClaimNum"),
    ("NDC 33342-0395-44", "NDC"),
    ("manufactured by MACLEODS", "Manufacturer"),
    ("show all generic drug claims", "Generic"),
    ("show brand name claims", "Brand"),
    ("show all refills for this member", "Refills"),
    ("show claims with 90 day supply", "DaysSupply"),
    ("which claims used prior authorization?", "PriorAuth"),
    ("show claims with diagnosis code E1129", "Diagnosis"),
    ("show claims with settlement code 358", "Settlement"),
    ("show retail pharmacy claims", "PharmType"),
    ("show claims under plan LICS2", "Plan"),
    ("show me claims filled at CVS PHARMACY 00610", "Pharmacy"),
    ("show claims by prescriber NOEUV", "Prescriber"),
    ("how much did the member pay for LEVOTHYROXINE?", "Pricing"),
    ("show me all rejected claims", "Status"),
    ("show me all claims with reject code 79", "RejectCode"),
    ("When was LEVOTHYROXINE taken last for this member?", "DrugLast"),
    ("give me all the claims for this member in january", "Month"),
]

all_claims = SAMPLE_RESPONSE.get("claims", [])
print(f"Total claims in sample: {len(all_claims)}\n")
print(f"{'Label':14s} | {'Count':5s} | {'Drugs matched'}")
print("-" * 70)

for query, label in queries:
    result = generalized_claims_query(all_claims, query)
    drugs = [c.get("drug", {}).get("productName", "?")[:25] for c in result]
    print(f"{label:14s} | {len(result):5d} | {drugs}")

print("\nAll filters executed successfully!")
