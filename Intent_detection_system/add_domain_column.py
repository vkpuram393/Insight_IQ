"""
Adds a 'domain' column to Testdata.csv by mapping each intent to its
domain key using DOMAIN_REGISTRY from VamsiSir.py.

Rule: claim_status (appears in both cap_api and claim_history_search)
      is always mapped to 'cap_api'.
"""

import pandas as pd
from VamsiSir import embeddingVars

# ── Build intent → domain mapping from DOMAIN_REGISTRY ──────────────────────
DOMAIN_REGISTRY = embeddingVars.DOMAIN_REGISTRY

intent_to_domain: dict[str, str] = {}

for domain_key, domain_info in DOMAIN_REGISTRY.items():
    for intent in domain_info["intents"]:
        # Only set if not already mapped (first occurrence wins).
        # DOMAIN_REGISTRY is ordered: cap_api comes first, so claim_status
        # naturally maps to cap_api — which also matches the explicit rule.
        if intent not in intent_to_domain:
            intent_to_domain[intent] = domain_key

print("Intent → Domain mapping:")
for intent, domain in intent_to_domain.items():
    print(f"  {intent:30s} → {domain}")

# ── Load CSV, add domain column, save ────────────────────────────────────────
df = pd.read_csv("Testdata.csv")

df["domain"] = df["Intent"].map(intent_to_domain)

# Report any unmapped intents
unmapped = df[df["domain"].isna()]["Intent"].unique()
if len(unmapped):
    print(f"\n⚠️  Unmapped intents: {unmapped}")
else:
    print("\n✅ All intents mapped successfully.")

df.to_csv("Testdata.csv", index=False)
print(f"\n✅ 'domain' column added and saved to Testdata.csv")
print(df[["Intent", "domain"]].drop_duplicates().to_string(index=False))
