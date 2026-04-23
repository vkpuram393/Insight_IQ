"""
Comprehensive test data label audit.
Checks EVERY row against keyword rules derived from the intent definitions.
Outputs a corrected CSV and a detailed change log.
"""
import pandas as pd, re, os

df = pd.read_csv("Testdata_corrected.csv")
print(f"Loaded {len(df)} test records")

changes = []

def fix(idx, new_intent, new_domain, reason):
    old_intent = df.at[idx, "Intent"]
    old_domain = df.at[idx, "domain"]
    if old_intent != new_intent or old_domain != new_domain:
        changes.append({
            "row": idx + 2,
            "prompt": df.at[idx, "Prompt"][:80],
            "old_intent": old_intent,
            "new_intent": new_intent,
            "old_domain": old_domain,
            "new_domain": new_domain,
            "reason": reason,
        })
        df.at[idx, "Intent"] = new_intent
        df.at[idx, "domain"] = new_domain

DOMAIN_MAP = {
    "claim_status":"cap_api","multi_claim_summary":"cap_api","pharmacy_info":"cap_api",
    "prescriber_info":"cap_api","pricing_info":"cap_api","reimbursement_info":"cap_api",
    "rejection_reasons":"cap_api","settlement_info":"cap_api","rx_details":"cap_api",
    "reversal_info":"cap_api","cob_info":"cap_api","generic_availability":"cap_api",
    "daw_info":"cap_api","government_claim_type":"cap_api","mail_order_info":"cap_api",
    "medicare_part_d":"cap_api","network_info":"cap_api","prior_auth_info":"cap_api",
    "approval_info":"benefits_api","audit_info":"benefits_api","beneficiary_info":"benefits_api",
    "compound_info":"claim_history_search","date_range_claims":"claim_history_search",
    "drug_info":"claim_history_search","drug_interaction_info":"claim_history_search",
    "fill_date_info":"claim_history_search",
    "greeting":"general","help":"general","out_of_scope":"general",
}

for idx, row in df.iterrows():
    p = row["Prompt"]
    pl = p.lower().strip()
    intent = row["Intent"]
    domain = row["domain"]

    # ── 1. Prescriber queries mislabeled as rx_details or drug_info ───────
    prescriber_words = ["prescriber", "physician", "doctor's name", "doctor's details",
                        "who prescribed", "who ordered the medication",
                        "prescribing physician", "ordering provider",
                        "prescriber npi", "npi and name"]
    if intent in ("rx_details", "drug_info") and any(w in pl for w in prescriber_words):
        fix(idx, "prescriber_info", "cap_api", f"Contains prescriber keyword, not {intent}")

    # "What medication was prescribed" → prescriber_info (asking about WHO prescribed)
    if intent == "drug_info" and "what medication was prescribed" in pl:
        fix(idx, "prescriber_info", "cap_api", "'what medication was prescribed' asks about prescriber")

    # ── 2. Settlement queries mislabeled as claim_status ─────────────────
    settlement_words = ["settlement detail", "settlement information", "settlement report",
                        "settlement summary", "settlement feedback", "settlement status",
                        "settlement codes", "response information for claim"]
    if intent == "claim_status" and any(w in pl for w in settlement_words):
        fix(idx, "settlement_info", "cap_api", "Contains settlement keyword, not claim_status")

    # ── 3. Greeting vs out_of_scope ──────────────────────────────────────
    clear_greetings = ["hello", "hi there", "good morning", "good afternoon",
                       "good evening", "greetings", "hello there", "good day",
                       "howdy", "hey there", "welcome", "hiya"]
    if intent == "out_of_scope" and pl.strip().rstrip("!.,") in clear_greetings:
        fix(idx, "greeting", "general", f"'{pl.strip()}' is a greeting")

    if intent == "out_of_scope" and pl.startswith("hello"):
        fix(idx, "greeting", "general", "Starts with 'hello' = greeting")

    # "Hi, good to see you" — conversational greeting
    if intent == "out_of_scope" and ("good to see you" in pl or "how are you" in pl):
        fix(idx, "greeting", "general", "Conversational greeting")

    # Short ambiguous ones: "hey", "hi", "morning" stay as labeled unless clearly wrong
    if intent == "greeting" and pl.strip() in ["hey", "hi", "morning", "afternoon", "evening"]:
        # These are borderline but more greeting than OOS
        pass

    # ── 4. R&R queries mislabeled as claim_status ────────────────────────
    if intent == "claim_status" and re.search(r'\br&r\b', pl, re.IGNORECASE):
        fix(idx, "reversal_info", "cap_api", "R&R = reversal, not claim_status")

    # ── 5. "Store information" = pharmacy_info, not claim_status ─────────
    if intent == "claim_status" and "store information" in pl:
        fix(idx, "pharmacy_info", "cap_api", "'store information' = pharmacy, not claim_status")

    # ── 6. "Fill details" = rx_details, not claim_status ─────────────────
    if intent == "claim_status" and "fill details" in pl:
        fix(idx, "rx_details", "cap_api", "'fill details' = rx_details, not claim_status")

    # ── 7. Domain consistency: ensure domain matches intent ──────────────
    expected_domain = DOMAIN_MAP.get(intent)
    if expected_domain and domain != expected_domain:
        fix(idx, intent, expected_domain, f"Domain should be {expected_domain} for {intent}")

    # ── 8. "How do I fix my car?" labeled rejection_reasons → out_of_scope
    if intent == "rejection_reasons" and ("fix my car" in pl or "change a tire" in pl):
        fix(idx, "out_of_scope", "general", "Unrelated to pharmacy claims")

    # ── 9. "Tell me about history" labeled multi_claim_summary → out_of_scope
    if intent == "multi_claim_summary" and pl.strip() == "tell me about history.":
        fix(idx, "out_of_scope", "general", "Generic 'history' question, not claims")

    # ── 10. "Physician report" labeled drug_interaction_info → prescriber_info
    if intent == "drug_interaction_info" and "physician report" in pl:
        fix(idx, "prescriber_info", "cap_api", "'physician report' = prescriber, not DUR")

    # ── 11. "Prescriber NPI and name" labeled drug_info → prescriber_info
    if intent == "drug_info" and "prescriber npi" in pl:
        fix(idx, "prescriber_info", "cap_api", "'prescriber NPI' = prescriber, not drug")

    # ── 12. "Prescribing physician for claim" labeled drug_info → prescriber_info
    if intent == "drug_info" and "prescribing physician" in pl:
        fix(idx, "prescriber_info", "cap_api", "'prescribing physician' = prescriber")

    # ── 13. "Who ordered the medication" labeled pharmacy_info → prescriber_info
    if intent == "pharmacy_info" and "who ordered" in pl:
        fix(idx, "prescriber_info", "cap_api", "'who ordered' = prescriber, not pharmacy")

    # ── 14. "Ordering provider" labeled pharmacy_info → prescriber_info
    if intent == "pharmacy_info" and "ordering provider" in pl:
        fix(idx, "prescriber_info", "cap_api", "'ordering provider' = prescriber")

    # ── 15. "What was the pharmacy paid" labeled pharmacy_info → reimbursement_info
    if intent == "pharmacy_info" and "was the pharmacy paid" in pl:
        fix(idx, "reimbursement_info", "cap_api", "'pharmacy paid' = reimbursement")

    # ── 16. "DUR processing details" labeled claim_status → drug_interaction_info
    if intent == "claim_status" and ("dur processing" in pl or "dur edit" in pl):
        fix(idx, "drug_interaction_info", "claim_history_search", "DUR = drug interaction")

    # ── 17. "When did the pharmacy fill" labeled pharmacy_info → fill_date_info
    if intent == "pharmacy_info" and "when did the pharmacy fill" in pl:
        fix(idx, "fill_date_info", "claim_history_search", "'when did pharmacy fill' = fill date")

    # ── 18. "GPI number" labeled claim_status → drug_info
    if intent == "claim_status" and "gpi number" in pl:
        fix(idx, "drug_info", "claim_history_search", "GPI = drug info, not claim status")

    # ── 19. "Claim submission help" labeled claim_status → help
    if intent == "claim_status" and "submission help" in pl:
        fix(idx, "help", "general", "'submission help' = help intent")

    # ── 20. "Best practices for claim submission" labeled out_of_scope → help
    if intent == "out_of_scope" and "best practices" in pl and "claim" in pl:
        fix(idx, "help", "general", "'best practices for claim submission' = help")

    # ── 21. "Manual adjustment details" labeled audit_info → reversal_info
    if intent == "audit_info" and "manual adjustment" in pl:
        fix(idx, "reversal_info", "cap_api", "'manual adjustment' = reversal, not audit")

    # ── 22. "Modification details" labeled audit_info → check context
    # "Modification details for claim 242831720377166" - this could be audit or reversal
    # Keep as audit_info since it's about change history

    # ── 23. "Approved ingredient cost" labeled compound_info → pricing_info
    if intent == "compound_info" and "approved ingredient cost" in pl:
        fix(idx, "pricing_info", "cap_api", "'approved ingredient cost' = pricing")

    # ── 24. "Fill number information" labeled approval_info → rx_details
    if intent == "approval_info" and "fill number" in pl and "tf" not in pl and "transition" not in pl:
        fix(idx, "rx_details", "cap_api", "'fill number information' = rx_details")

    # ── 25. "MIC cost breakdown" labeled cob_info → compound_info
    if intent == "cob_info" and "mic" in pl.lower() and "cost breakdown" in pl:
        fix(idx, "compound_info", "claim_history_search", "MIC = compound, not COB")

    # ── 26. "Resubmissions for claim" labeled rejection_reasons → reversal_info
    if intent == "rejection_reasons" and "resubmission" in pl:
        fix(idx, "reversal_info", "cap_api", "'resubmissions' = reversal, not rejection")

    # ── 27. "How do I submit a claim correctly" labeled rejection_reasons → help
    if intent == "rejection_reasons" and "how do i submit" in pl:
        fix(idx, "help", "general", "'how do i submit' = help, not rejection")

    # ── 28. "Feedback codes" labeled settlement_info is correct (keep)
    # ── 29. "Response codes for claim" labeled rejection_reasons → settlement_info
    if intent == "rejection_reasons" and "response codes" in pl and "rejection" not in pl and "reject" not in pl:
        fix(idx, "settlement_info", "cap_api", "'response codes' = settlement")


# ── Print all changes ────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  LABEL CORRECTIONS: {len(changes)} changes")
print(f"{'='*70}")

# Group by type
from collections import Counter
type_counts = Counter((c["old_intent"], c["new_intent"]) for c in changes)
print(f"\n  Summary:")
for (old, new), count in type_counts.most_common():
    print(f"    {old} → {new}: {count} queries")

print(f"\n  Details:")
for c in changes:
    print(f"    Row {c['row']}: {c['old_intent']} → {c['new_intent']}")
    print(f"      \"{c['prompt']}\"")
    print(f"      Reason: {c['reason']}")

# ── Save ─────────────────────────────────────────────────────────────────────
output_path = "Testdata_corrected.csv"
df.to_csv(output_path, index=False)
print(f"\n  Saved {len(df)} records → {output_path}")
print(f"  Total corrections: {len(changes)}")

# Verify domain consistency
inconsistent = 0
for idx, row in df.iterrows():
    expected = DOMAIN_MAP.get(row["Intent"])
    if expected and row["domain"] != expected:
        inconsistent += 1
        print(f"  WARNING: Row {idx+2} intent={row['Intent']} domain={row['domain']} expected={expected}")
print(f"\n  Domain consistency check: {inconsistent} inconsistencies remaining")
