"""
CAP API Domain — LLM Fallback Prompt for Single-Claim Operations

Uses the existing production prompt (claim_prompt_template) that has been
battle-tested in the live system for the cap_api domain.

This domain covers all intents that operate on a SINGLE specific claim:
  claim_status, prescriber_info, pharmacy_info, pricing_info,
  reimbursement_info, rejection_reasons, settlement_info, rx_details,
  reversal_info, cob_info, generic_availability, daw_info,
  government_claim_type, mail_order_info, medicare_part_d,
  network_info, prior_auth_info, multi_claim_summary

KEY DISTINCTION: cap_api = details about ONE claim (by claim number).
                 claim_history_search = SEARCH/FILTER across MANY claims.
"""

from prompt_templates.prompt_template import claim_prompt_template

CAP_API_PROMPT = claim_prompt_template
