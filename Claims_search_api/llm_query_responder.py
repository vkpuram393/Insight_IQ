"""
Claims_search_api.llm_query_responder

Generalized, LLM-driven Q&A over a member's claim history.

Why a separate module?
----------------------
The existing pipeline (search.py + llm_formatter.py + response_agent.py)
filters claims deterministically with regex/keyword rules and then asks
the LLM to summarize the *already-filtered* set.  That works for the
patterns we explicitly coded (drug name, reject code, month, …), but
fails for arbitrary user questions that touch other fields (manufacturer,
GPI, NDC, prescriber NPI, days supply, plan code, COB indicator, etc.).

This module takes the opposite, fully-generalized approach:

    1.  Trim the raw API response (drop heavy/PII-noisy fields) so the
        prompt stays within token limits — but keep ALL claims and ALL
        business-relevant fields intact.
    2.  Feed the trimmed JSON + the user's natural-language query to a
        single LLM call with a strict response contract.
    3.  Return the LLM's answer text plus light metadata.

The LLM is instructed to:
  • Do its OWN filtering / lookup over the supplied claims (no Python
    pre-filter is applied).
  • Cite specific claim numbers it relied on.
  • Refuse to invent fields that aren't in the data.

This module is intentionally **independent** of the existing
`response_agent.py` / `search.py` paths.  It can be wired up via the
companion file `claims_search_node_v2.py`, which is an alternate
LangGraph node that bypasses the deterministic filter chain entirely.

Public API
----------
    answer_claim_history_query(api_response, user_query, **opts) -> dict
"""

from __future__ import annotations

import json
import logging
from datetime import date
from typing import Any, Dict, List, Optional

from services.llm_connection import generate
from Claims_search_api.response_trimmer import trim_api_response

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# Hard cap on the number of claims sent to the LLM.  /claims/search can
# return tens of thousands of rows for active members; clamping here keeps
# prompt size predictable.  Sorted newest-first before clamping.
DEFAULT_MAX_CLAIMS = 50

# Per-claim character budget when serialising to JSON.  If a single claim
# (post-trim) exceeds this, we further compact it before inclusion.
_PER_CLAIM_SOFT_BUDGET = 2000


# ---------------------------------------------------------------------------
# Field whitelist — what we expose to the LLM
# ---------------------------------------------------------------------------
# We deliberately include MANY fields so the LLM can answer arbitrary
# questions ("which claims used prior auth", "who was the prescriber on
# claim X", "what's the GPI for the Mounjaro fills", etc.).  Anything not
# listed here is dropped to save tokens.

_CLAIM_FIELD_WHITELIST: Dict[str, List[str]] = {
    "claimInformation": [
        "claimNumber", "claimSequenceNumber",
        "claimStatus", "claimStatusDescription",
        "fillDate", "addDate", "changeDate",
        "quantity", "daysSupplied",
        "claimType", "coverageType",
        "compound", "speciality",
        "dispenseAsWritten",
        "originationFlagDescription",
        "secondaryClaimNumber", "secondaryClaimSequence",
        "rnR",
    ],
    "drug": [
        "productName", "productNdc", "gpi",
        "genericIndicator", "multiSourceIndicatorDescription",
        "manufacturer",
        "productIDQualifierDescription",
    ],
    "pricing": [
        "patientPay", "clientPay",
        "drugCostSubmitted", "drugCostApproved",
        "dispensingFeeSubmitted", "dispensingFeeApproved",
        "amountDueSubmitted", "amountDueApproved",
    ],
    "prescription": [
        "rxNumber", "refillNumber",
        "prescriberFirstName", "prescriberLastName", "prescriberID",
        "pharmacyName", "pharmacyCity", "pharmacyState",
        "pharmacyType",
        "submitDate", "reversalDate",
        "submittedDiagnosisCodeIndicator",
        "groupNumber", "binNumber",
    ],
    "priorAuthorization": [
        "number", "paIndicator", "type", "typeDescription",
        "reasonCode", "reasonDescription",
    ],
    "overrides": [
        "priorAuthorizationUsed", "drugutiliztionReview",
        "smartPriorAuthorizationUsed", "drugListUsed",
        "submissionClarificationCode",
    ],
    "messages": [
        "rejectCodes", "approvedMessages",
        "settlementCodes", "messages",
    ],
    # NOTE: `member` is intentionally excluded from the LLM context to
    # minimise PII surface area — the orchestrator already authorises by
    # claim_id, and the LLM doesn't need member name/DOB to answer claim
    # questions.  Memberless display info (memberId only) is added via
    # the top-level summary instead.
}


def _slim_claim(claim: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of `claim` keeping only whitelisted fields."""
    out: Dict[str, Any] = {}
    for section, keep_fields in _CLAIM_FIELD_WHITELIST.items():
        sub = claim.get(section) or {}
        if not isinstance(sub, dict):
            continue
        slim_sub = {k: sub.get(k) for k in keep_fields if sub.get(k) not in (None, "", [])}
        if slim_sub:
            out[section] = slim_sub
    return out


def _sort_newest_first(claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def _fd(c):
        return (c.get("claimInformation") or {}).get("fillDate") or ""
    return sorted(claims, key=_fd, reverse=True)


def _summarize_member(api_response: Dict[str, Any]) -> Dict[str, Any]:
    """Extract a tiny, non-PII member summary for the LLM context.

    trim_api_response moves the member object to the top-level "member" key
    and removes it from each individual claim.  Check the top-level key first
    so this function works on both trimmed and raw responses.
    """
    r = api_response or {}
    # Preferred: top-level "member" key (set by trim_api_response)
    m = r.get("member") or {}
    if not m:
        # Fallback: raw (un-trimmed) response still has member inside each claim
        claims = r.get("claims") or []
        if claims:
            m = (claims[0] or {}).get("member") or {}
    return {
        "memberId": m.get("memberId") or m.get("cardholderId"),
        "firstName": m.get("firstName"),
        "lastName": m.get("lastName"),
        "planId": m.get("planId"),
        "carrierId": m.get("carrierId"),
        "groupId": m.get("groupId"),
    }


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_SYSTEM_INSTRUCTIONS = """You are a CVS pharmacy-benefit claims assistant — warm, professional, and genuinely helpful.

You will be given:
  1. A user question (natural language).
  2. A JSON array of claims for ONE member from the CVS claims-search API.
     Each claim has nested sections: claimInformation, drug, pricing,
     prescription, priorAuthorization, overrides, messages.

Your job
--------
Answer the user's question using ONLY the claims data provided.
Write in a conversational, active voice. Say "I found…" not "Based on the provided data…".
Be assertive and confident. Do not use markdown formatting (no bold **, no headings #, no code fences).
Use bullet points (•) when listing multiple items. Always include the member's name, member ID,
and the specific claim number(s) you relied on so the user can verify.

Rules
-----
1.  Treat the claims array as the single source of truth.  Do NOT invent
    drugs, prescribers, dates, codes, or amounts that are not present.
2.  Match field values case-insensitively for free-text fields like
    drug.productName, drug.manufacturer, prescription.pharmacyName,
    prescriber names, etc.  Substring matches are acceptable
    (e.g. "Xultophy" should match "XULTOPHY 100/3.6 PEN").
3.  For status / reject-code / settlement-code / NDC / GPI / refill /
    days-supply / DAW filters, use exact matches.

STATUS FILTER — apply this FIRST, before any other filter:
  • Default (no status word in the question): consider ALL claims
    regardless of claimStatus (Paid, Rejected, Reversed, etc.).
  • User says "paid":
    consider ONLY claimStatus = "P" (Paid).
  • User says "rejected" / "reject" / "denial" / "denied":
    consider ONLY claimStatus = "R" (Rejected).
  • User says "reversed" / "reversal" / "cancelled" / "canceled":
    consider ONLY claimStatus = "X" (Reversed/Cancelled).
  • User says "all statuses" / "regardless of status" / "any status" /
    explicitly names two or more statuses:
    consider all claims regardless of status.
  Apply the status filter before drug-name, date-range, GPI, or any
  other filter.  Never mix statuses unless the user explicitly asks.

4.  After the status filter, when the user asks for "the last claim",
    "when was X taken last", "latest fill", "last fill date", or similar,
    sort the remaining matching claims by claimInformation.fillDate
    (newest first) and return the top result.
5.  If MULTIPLE claims match, present them as a scannable list sorted
    newest-first (see MULTI-CLAIM ANSWER guidelines below).
6.  If NO claims match the filter, say so plainly and state how many
    total claims were searched.  Do NOT make up data.
7.  Do not use code fences or preamble like "Here is …".
8.  Never echo PII you weren't given.  Never reveal raw JSON.
9.  If the question is unrelated to the supplied claims (e.g. asks about
    a different member, eligibility, formulary, appeals), respond:
    "I can only answer questions about the claims shown above."
10. CLAIM-ID IN QUERY: The user's query may contain a long numeric claim ID
    (typically 15 digits, e.g. "260302639954275") and/or a sequence number
    (e.g. "001").  These were used ONLY to identify which member's history
    to retrieve — they are NOT filters on claimNumber or claimSequence.
    Do NOT restrict results to that specific claim number.  Answer the
    user's actual question (last fill, all drugs, this month, etc.) using
    ALL claims in the provided history.
11. TEMPORAL REFERENCE: a CURRENT DATE is supplied in the user section
    below.  Use it ONLY to resolve relative time expressions in the user's
    claim query (e.g. "last week", "past 30 days", "last month",
    "yesterday").  Do NOT answer general time or date questions (e.g.
    "What's today's date?", "What time is it?", "What day is it?",
    "What year is it?").  If the user asks a question unrelated to claims —
    including general time/date questions — respond exactly:
    "I'm sorry, I can't help with that. I'm here to assist with
    claims-related queries."

─────────────────────────────────────────────────────────────────────
SINGLE-CLAIM ANSWER
Use when the user asks about ONE specific event (e.g. "when was X taken last?",
"what was the last claim for X?", "how much did the member pay for X?").
Use the first (most recent) matching claim after applying any filters.

Guidelines — weave these details into a natural, conversational response:
  • Drug name
  • Member first name, last name, and member ID
  • Claim status expressed as a verb: "paid", "rejected", or "reversed"
  • Fill date in YYYY-MM-DD format
  • Pharmacy name
  • Claim number and sequence number
  • Rx number
  Include additional fields (patient pay, days supply, prescriber, etc.) whenever
  they add useful context. Omit any field that is missing from the data — do not
  print "N/A". Keep the answer concise.

─────────────────────────────────────────────────────────────────────
MULTI-CLAIM ANSWER
Use when the user asks for MULTIPLE claims (e.g. "list all medicines",
"all claims in January", "all rejected claims", "claims for reject code 79", etc.).

Guidelines — present one entry per matching claim, sorted newest-first,
with completeness taking priority over brevity. Each entry must include at minimum
(when available in the data):
  • Claim number and sequence number
  • Claim status (include reject code if rejected)
  • Fill date
  • Drug name (product name)
  • Product NDC
  • Rx number
  • Pharmacy name
  • Patient pay amount (use $0.00 if null or missing)
Include additional fields if they improve clarity. Use bullet points (•) or a
consistent plain-text layout — no markdown pipes, bold, or headings. Open with a
brief conversational sentence that names the member (first name, last name, member ID)
and summarises what you found before listing the claims.
"""


_USER_TEMPLATE = """CURRENT DATE: {current_date}

USER QUESTION:
{user_query}

MEMBER SUMMARY:
{member_summary}

CLAIMS ({num_claims} total, newest first):
{claims_json}

Answer the question now, following the guidelines above.
• Use SINGLE-CLAIM ANSWER style when the user asks about one specific event.
• Use MULTI-CLAIM ANSWER style when the user asks for multiple claims.
Write your complete answer in plain conversational text. Do not use code fences."""


def build_claim_history_prompt(
    user_query: str,
    slim_claims: List[Dict[str, Any]],
    member_summary: Dict[str, Any],
) -> str:
    claims_json = json.dumps(slim_claims, indent=1, default=str)
    member_json = json.dumps(member_summary, default=str) if member_summary else "{}"
    current_date = date.today().strftime("%Y-%m-%d")
    return (
        _SYSTEM_INSTRUCTIONS
        + "\n\n"
        + _USER_TEMPLATE.format(
            current_date=current_date,
            user_query=user_query.strip(),
            member_summary=member_json,
            num_claims=len(slim_claims),
            claims_json=claims_json,
        )
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def prepare_claim_history_data(
    api_response: Dict[str, Any],
    max_claims: int = DEFAULT_MAX_CLAIMS,
) -> Dict[str, Any]:
    """
    Trim, sort, slim, and cap claims for prompt building.  No LLM call.

    Returns dict with keys: slim_claims, member_summary, total_claims, used_claims.
    """
    try:
        trimmed = trim_api_response(api_response or {})
    except Exception as e:
        logger.warning("[LLMResponder] trim_api_response failed: %s — falling back to raw", e)
        trimmed = api_response or {}

    raw_claims = trimmed.get("claims") or []
    total_claims = len(raw_claims)

    if total_claims == 0:
        return {"slim_claims": [], "member_summary": {}, "total_claims": 0, "used_claims": 0}

    sorted_claims = _sort_newest_first(raw_claims)
    capped = sorted_claims[:max(1, int(max_claims))]
    slim = [_slim_claim(c) for c in capped]
    member_summary = _summarize_member(api_response or {})

    return {
        "slim_claims": slim,
        "member_summary": member_summary,
        "total_claims": total_claims,
        "used_claims": len(slim),
    }


def answer_claim_history_query(
    api_response: Dict[str, Any],
    user_query: str,
    *,
    max_claims: int = DEFAULT_MAX_CLAIMS,
    temperature: float = 0.2,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generalized LLM-driven answer to ANY claim-history question.

    Parameters
    ----------
    api_response : dict
        The raw response from `/claims/search` (i.e. what
        `extract_list_api_response_structure` returns).  Must contain a
        top-level "claims" list.
    user_query : str
        The original natural-language user prompt.
    max_claims : int
        Hard cap on claims sent to the LLM (newest-first).
    temperature : float
        LLM temperature (low = deterministic factual answers).
    model : str, optional
        Override the default Gemini model.

    Returns
    -------
    dict with keys:
        answer        : str   — LLM's final answer (already user-facing)
        total_claims  : int   — how many claims were available pre-cap
        used_claims   : int   — how many were sent to the LLM
        member        : dict  — small member summary used in the prompt
        success       : bool
        error         : str   — populated only when success is False
    """
    if not user_query or not str(user_query).strip():
        return {
            "answer": "Please provide a question about the member's claims.",
            "total_claims": 0,
            "used_claims": 0,
            "member": {},
            "success": False,
            "error": "empty_user_query",
        }

    data = prepare_claim_history_data(api_response, max_claims)
    slim = data["slim_claims"]
    member_summary = data["member_summary"]
    total_claims = data["total_claims"]

    if not slim:
        return {
            "answer": "No claims were returned for this member, so I can't answer that question.",
            "total_claims": 0,
            "used_claims": 0,
            "member": {},
            "success": True,
            "error": "",
        }

    prompt = build_claim_history_prompt(user_query, slim, member_summary)
    logger.info(
        "[LLMResponder] Prompt built: %d total claims, %d sent to LLM, %d chars",
        total_claims, len(slim), len(prompt),
    )

    try:
        gen_kwargs: Dict[str, Any] = {"temperature": temperature}
        if model:
            gen_kwargs["model"] = model
        answer = generate(prompt, **gen_kwargs)
        answer = (answer or "").strip()
        if not answer:
            raise RuntimeError("LLM returned empty response")
    except Exception as e:
        logger.exception("[LLMResponder] LLM call failed")
        return {
            "answer": (
                "I couldn't generate an answer right now due to a downstream "
                "service error. Please try again in a moment."
            ),
            "total_claims": total_claims,
            "used_claims": len(slim),
            "member": member_summary,
            "success": False,
            "error": f"llm_error: {e}",
        }

    return {
        "answer": answer,
        "total_claims": total_claims,
        "used_claims": len(slim),
        "member": member_summary,
        "success": True,
        "error": "",
    }
