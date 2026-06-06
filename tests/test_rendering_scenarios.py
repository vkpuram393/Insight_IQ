"""
Two-tier rendering scenario tests — TEXT / TABLE.

Covers every path through the render_mode decision system:
  • Tier 1  MUST_RENDER_INTENTS  → always html_table (Python overrides text_only)
  • Tier 2  LLM decides          → text (render_mode=text_only) | table (data shape)
  • NO_RENDER_INTENTS            → always text (Gate 1)
  • None guard                   → render_mode=None + no render_dsl → text

Run with:  pytest tests/test_rendering_scenarios.py -v
"""
import pytest
from agents.post_processing.myclaims_rendering_agent import MyclaimsRenderingAgent
from agents.post_processing.rendering_themes import (
    MUST_RENDER_INTENTS,
    NO_RENDER_INTENTS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def agent():
    return MyclaimsRenderingAgent()


# ---------------------------------------------------------------------------
# Shared tool-result builders
# ---------------------------------------------------------------------------

def _tool_multi_row():
    """Two-claim payload → multi-row table path."""
    return {
        "tool_name": "get_claims",
        "status": "success",
        "data": {
            "claims": [
                {
                    "claimNumber": "RX10001",
                    "claimStatusDescription": "Paid",
                    "fillDate": "20250301",
                    "productName": "LISINOPRIL 10MG",
                    "patientPay": "5.00",
                    "clientPay": "40.00",
                },
                {
                    "claimNumber": "RX10002",
                    "claimStatusDescription": "Denied",
                    "fillDate": "20250310",
                    "productName": "METFORMIN 500MG",
                    "patientPay": "0.00",
                    "clientPay": "0.00",
                },
            ]
        },
    }


def _tool_single_row():
    """One-claim payload — card vs table decided by column count."""
    return {
        "tool_name": "get_claim",
        "status": "success",
        "data": {
            "claims": [
                {
                    "claimNumber": "RX10001",
                    "claimStatusDescription": "Paid",
                    "fillDate": "20250301",
                    "productName": "LISINOPRIL 10MG",
                    "patientPay": "5.00",
                    "clientPay": "40.00",
                }
            ]
        },
    }


def _tool_prescriber():
    return {
        "tool_name": "get_prescriber",
        "status": "success",
        "data": {
            "prescribers": [
                {
                    "prescriberName": "DR. JOHN DOE",
                    "npi": "1234567890",
                    "dea": "BD1234567",
                }
            ]
        },
    }


def _tool_pharmacy():
    return {
        "tool_name": "get_pharmacy",
        "status": "success",
        "data": {
            "pharmacies": [
                {
                    "pharmacyName": "CVS PHARMACY #1234",
                    "address": "123 MAIN ST",
                    "city": "ATLANTA",
                    "state": "GA",
                    "phone": "4045551234",
                }
            ]
        },
    }


def _tool_pricing():
    return {
        "tool_name": "get_pricing",
        "status": "success",
        "data": {
            "pricing": [
                {
                    "ingredientCost": "42.00",
                    "dispensingFee": "1.50",
                    "patientPay": "5.00",
                    "planPaid": "38.50",
                    "basisOfReimbursement": "AWP",
                }
            ]
        },
    }


def _tool_copay():
    return {
        "tool_name": "get_copay",
        "status": "success",
        "data": {
            "copayTiers": [
                {"tier": "1 - Preferred Generic", "copay": "10.00"},
                {"tier": "2 - Generic", "copay": "20.00"},
                {"tier": "3 - Preferred Brand", "copay": "45.00"},
            ]
        },
    }


def _tool_rejection():
    return {
        "tool_name": "get_rejections",
        "status": "success",
        "data": {
            "rejections": [
                {"rejectCode": "70", "rejectDescription": "Product/Service Not Covered"},
                {"rejectCode": "75", "rejectDescription": "Prior Authorization Required"},
            ]
        },
    }


# ---------------------------------------------------------------------------
# Shared DSL builders
# ---------------------------------------------------------------------------

def _dsl_6col():
    """6-column DSL — single row + >5 cols → table."""
    return {
        "layout": "table",
        "sections": [{
            "columns": [
                {"header": "Claim #",     "field": "claimNumber",             "format": "text"},
                {"header": "Status",      "field": "claimStatusDescription",  "format": "status_badge"},
                {"header": "Fill Date",   "field": "fillDate",                "format": "date"},
                {"header": "Drug",        "field": "productName",             "format": "title"},
                {"header": "Patient Pay", "field": "patientPay",              "format": "currency"},
                {"header": "Plan Paid",   "field": "clientPay",               "format": "currency"},
            ],
        }],
    }


def _dsl_3col():
    """3-column DSL — single row + ≤5 cols → card."""
    return {
        "layout": "table",
        "sections": [{
            "columns": [
                {"header": "Claim #", "field": "claimNumber",            "format": "text"},
                {"header": "Status",  "field": "claimStatusDescription", "format": "status_badge"},
                {"header": "Drug",    "field": "productName",            "format": "title"},
            ],
        }],
    }


def _dsl_prescriber():
    return {
        "layout": "table",
        "sections": [{
            "columns": [
                {"header": "Prescriber Name", "field": "prescriberName", "format": "title"},
                {"header": "NPI",             "field": "npi",            "format": "text"},
                {"header": "DEA",             "field": "dea",            "format": "text"},
            ],
        }],
    }


def _dsl_pharmacy():
    return {
        "layout": "table",
        "sections": [{
            "columns": [
                {"header": "Pharmacy", "field": "pharmacyName", "format": "title"},
                {"header": "Address",  "field": "address",      "format": "text"},
                {"header": "City",     "field": "city",         "format": "text"},
                {"header": "State",    "field": "state",        "format": "text"},
                {"header": "Phone",    "field": "phone",        "format": "text"},
            ],
        }],
    }


def _dsl_pricing():
    return {
        "layout": "table",
        "sections": [{
            "columns": [
                {"header": "Ingredient Cost", "field": "ingredientCost",       "format": "currency"},
                {"header": "Dispensing Fee",   "field": "dispensingFee",        "format": "currency"},
                {"header": "Patient Pay",      "field": "patientPay",           "format": "currency"},
                {"header": "Plan Paid",        "field": "planPaid",             "format": "currency"},
                {"header": "Basis",            "field": "basisOfReimbursement", "format": "text"},
            ],
        }],
    }


def _dsl_copay():
    return {
        "layout": "table",
        "sections": [{
            "columns": [
                {"header": "Tier",  "field": "tier",  "format": "text"},
                {"header": "Copay", "field": "copay", "format": "currency"},
            ],
        }],
    }


def _dsl_rejection():
    return {
        "layout": "table",
        "sections": [{
            "columns": [
                {"header": "Reject Code",  "field": "rejectCode",        "format": "text"},
                {"header": "Description",  "field": "rejectDescription",  "format": "text"},
            ],
        }],
    }


# =============================================================================
# TEXT SCENARIOS
# Expected render_format == "text" (no HTML output)
# =============================================================================

class TestTextScenarios:
    """
    Prompts whose complete answer is a single value, date, yes/no, or phrase.
    The LLM outputs render_mode="text_only" → rendering agent returns plain text.
    Applies to Tier 3 intents (not in MUST_RENDER or ALWAYS_CARD).

    Prompts represented:
      "What is the status of claim RX10001?"      → claim_status,  text_only
      "How much did the patient pay?"              → claim_details, text_only
      "When was this prescription filled?"         → rx_details,    text_only
      "What is the DAW code?"                      → claim_details, text_only
      "Was prior auth required?"                   → claim_status,  text_only
      "Is this a compound claim?"                  → claim_status,  text_only
      "What is the member ID?"                     → claim_status,  text_only
      Greeting / Help / Out-of-scope               → NO_RENDER,    always text
      Tool call failed                             → always text
      suppress_table backward-compat               → Tier 3 intent, text
    """

    # ── Single-value answers ─────────────────────────────────────────────────

    def test_claim_status_text_only(self, agent):
        """'What is the status of claim RX10001?' → text_only → no HTML."""
        result = agent.execute(
            "claim_status",
            _tool_single_row(),
            {},
            render_dsl=_dsl_3col(),
            render_mode="text_only",
        )
        assert result.render_format == "text"
        assert result.html_content == ""

    def test_patient_pay_text_only(self, agent):
        """'How much did the patient pay for this claim?' → text_only."""
        result = agent.execute(
            "claim_details",
            _tool_single_row(),
            {},
            render_dsl=_dsl_3col(),
            render_mode="text_only",
        )
        assert result.render_format == "text"
        assert result.html_content == ""

    def test_fill_date_text_only(self, agent):
        """'When was this prescription filled?' → text_only via claim_details (Tier 3)."""
        result = agent.execute(
            "claim_details",
            _tool_single_row(),
            {},
            render_dsl=_dsl_3col(),
            render_mode="text_only",
        )
        assert result.render_format == "text"

    def test_daw_code_text_only(self, agent):
        """'What is the DAW code?' → text_only."""
        result = agent.execute(
            "claim_details",
            _tool_single_row(),
            {},
            render_dsl=_dsl_3col(),
            render_mode="text_only",
        )
        assert result.render_format == "text"

    # ── Yes/No answers ───────────────────────────────────────────────────────

    def test_prior_auth_yes_no_text_only(self, agent):
        """'Was prior auth required?' → text_only."""
        result = agent.execute(
            "claim_status",
            _tool_single_row(),
            {},
            render_dsl=_dsl_3col(),
            render_mode="text_only",
        )
        assert result.render_format == "text"

    def test_compound_claim_yes_no_text_only(self, agent):
        """'Is this a compound claim?' → text_only."""
        result = agent.execute(
            "claim_status",
            _tool_single_row(),
            {},
            render_dsl=_dsl_3col(),
            render_mode="text_only",
        )
        assert result.render_format == "text"

    def test_member_id_text_only(self, agent):
        """'What is the member ID?' → text_only."""
        result = agent.execute(
            "claim_status",
            _tool_single_row(),
            {},
            render_dsl=_dsl_3col(),
            render_mode="text_only",
        )
        assert result.render_format == "text"

    # ── No-render intents (Gate 1 fires before render_mode is checked) ───────

    def test_greeting_always_text(self, agent):
        """greeting → text regardless of render_mode — Gate 1 fires first."""
        result = agent.execute(
            "greeting",
            {"status": "success", "data": {}},
            {},
            render_mode="table",
        )
        assert result.render_format == "text"
        assert result.success is False

    def test_help_always_text(self, agent):
        result = agent.execute("help", {"status": "success", "data": {}}, {})
        assert result.render_format == "text"

    def test_out_of_scope_always_text(self, agent):
        result = agent.execute("out_of_scope", {"status": "success", "data": {}}, {})
        assert result.render_format == "text"

    def test_unknown_intent_in_no_render_list(self, agent):
        result = agent.execute("unknown", {"status": "success", "data": {}}, {})
        assert result.render_format == "text"

    def test_empty_query_intent(self, agent):
        result = agent.execute("empty_query", {"status": "success", "data": {}}, {})
        assert result.render_format == "text"

    # ── Tool call failure → always text (Gate 2) ─────────────────────────────

    def test_tool_failure_returns_text(self, agent):
        """Failed API call → text even when render_mode=table."""
        result = agent.execute(
            "claim_list",
            {"status": "error", "data": {}, "message": "API timeout"},
            {},
            render_mode="table",
        )
        assert result.render_format == "text"
        assert result.success is False

    def test_tool_failure_must_render_intent_still_text(self, agent):
        """MUST_RENDER intent with failed tool → Gate 2 fires → text."""
        result = agent.execute(
            "pricing_info",
            {"status": "error", "data": {}},
            {},
            render_dsl=_dsl_pricing(),
        )
        assert result.render_format == "text"

    # ── suppress_table backward-compat (Tier 3 only) ─────────────────────────

    def test_suppress_table_flag_returns_text_for_tier3(self, agent):
        """Old suppress_table:true in DSL still suppresses for Tier 3 intents."""
        dsl = {
            "suppress_table": True,
            "sections": [{"columns": [
                {"header": "Status", "field": "claimStatusDescription", "format": "text"},
                {"header": "Drug",   "field": "productName",            "format": "text"},
            ]}],
        }
        result = agent.execute(
            "claim_status",    # Tier 3 — not MUST_RENDER or ALWAYS_CARD
            _tool_single_row(),
            {},
            render_dsl=dsl,
        )
        assert result.render_format == "text"

    # ── No render_dsl + no render_mode → no-data message ─────────────────────

    def test_no_dsl_no_render_mode_defaults_to_text(self, agent):
        """render_mode=None and no render_dsl → skip rendering, return text."""
        result = agent.execute("claim_status", _tool_single_row(), {})
        assert result.render_format == "text"
        assert result.html_content == ""


# =============================================================================
# TABLE SCENARIOS
# Expected render_format == "html_table"
# =============================================================================

class TestTableScenarios:
    """
    Tier 1 MUST_RENDER_INTENTS always produce html_table regardless of render_mode.
    Tier 2 intents produce html_table for multi-row data or single row with ≥6 cols.

    Prompts represented:
      "Show pricing breakdown"              → pricing_info,      MUST_RENDER
      "What are the copay tiers?"           → copay_info,        MUST_RENDER
      "Show all my claims"                  → claim_list,        MUST_RENDER
      "Why was this claim denied?"          → rejection_reasons, MUST_RENDER
      "COB details"                         → cob_info,          MUST_RENDER
      "Deductible status"                   → deductible_info,   MUST_RENDER
      "Highest cost claims"                 → expensive_claims,  MUST_RENDER
      "Show all claims (multi-row)"         → claim_status,      Tier 3 → multi-row → table
      "Full claim details (many cols)"      → claim_status,      Tier 3 → 6 cols → table
    """

    # ── Tier 1: MUST_RENDER_INTENTS ──────────────────────────────────────────

    def test_pricing_info_always_table(self, agent):
        """'Show pricing breakdown' → pricing_info → always table even if LLM says text_only."""
        result = agent.execute(
            "pricing_info",
            _tool_pricing(),
            {},
            render_dsl=_dsl_pricing(),
            render_mode="text_only",
        )
        assert result.render_format == "html_table"
        assert result.success is True

    def test_copay_info_always_table(self, agent):
        """'What are the copay tiers?' → copay_info → always table."""
        result = agent.execute(
            "copay_info",
            _tool_copay(),
            {},
            render_dsl=_dsl_copay(),
            render_mode="text_only",
        )
        assert result.render_format == "html_table"

    def test_claim_list_always_table(self, agent):
        """'Show all my claims' → claim_list → always table."""
        result = agent.execute(
            "claim_list",
            _tool_multi_row(),
            {},
            render_dsl=_dsl_6col(),
            render_mode="text_only",
        )
        assert result.render_format == "html_table"

    def test_rejection_reasons_always_table(self, agent):
        """'Why was this claim denied?' → rejection_reasons → always table."""
        result = agent.execute(
            "rejection_reasons",
            _tool_rejection(),
            {},
            render_dsl=_dsl_rejection(),
            render_mode="text_only",
        )
        assert result.render_format == "html_table"

    def test_cob_info_always_table(self, agent):
        """'COB details' → cob_info → always table."""
        tool = {
            "tool_name": "get_cob",
            "status": "success",
            "data": {"cobInfo": [{"cobType": "Primary", "otherCarrier": "AETNA", "otherCarrierPaid": "30.00", "planPaid": "10.00"}]},
        }
        dsl = {"sections": [{"columns": [
            {"header": "Type",          "field": "cobType",          "format": "text"},
            {"header": "Other Carrier", "field": "otherCarrier",     "format": "text"},
            {"header": "Other Paid",    "field": "otherCarrierPaid", "format": "currency"},
            {"header": "Plan Paid",     "field": "planPaid",         "format": "currency"},
        ]}]}
        result = agent.execute("cob_info", tool, {}, render_dsl=dsl, render_mode="text_only")
        assert result.render_format == "html_table"

    def test_deductible_info_always_table(self, agent):
        """'Deductible status' → deductible_info → always table."""
        tool = {
            "tool_name": "get_deductible",
            "status": "success",
            "data": {"deductibles": [{"deductibleType": "Individual", "metAmount": "250.00", "remainingAmount": "750.00", "maxAmount": "1000.00"}]},
        }
        dsl = {"sections": [{"columns": [
            {"header": "Type",      "field": "deductibleType",   "format": "text"},
            {"header": "Met",       "field": "metAmount",        "format": "currency"},
            {"header": "Remaining", "field": "remainingAmount",  "format": "currency"},
            {"header": "Max",       "field": "maxAmount",        "format": "currency"},
        ]}]}
        result = agent.execute("deductible_info", tool, {}, render_dsl=dsl, render_mode="text_only")
        assert result.render_format == "html_table"

    def test_claim_summary_respects_text_only(self, agent):
        """claim_summary moved to LLM-decides — text_only is now respected."""
        tool = {
            "tool_name": "get_summary",
            "status": "success",
            "data": {"summary": [{"totalClaims": "24", "totalPatientPay": "120.00", "totalPlanPaid": "580.00"}]},
        }
        dsl = {"sections": [{"columns": [
            {"header": "Total Claims", "field": "totalClaims",      "format": "text"},
            {"header": "Patient Paid", "field": "totalPatientPay",  "format": "currency"},
            {"header": "Plan Paid",    "field": "totalPlanPaid",    "format": "currency"},
        ]}]}
        result = agent.execute("claim_summary", tool, {}, render_dsl=dsl, render_mode="text_only")
        assert result.render_format == "text"

    def test_claim_summary_renders_table_when_llm_says_table(self, agent):
        """claim_summary with render_mode=table + multi-row data → html_table."""
        tool = {
            "tool_name": "get_summary",
            "status": "success",
            "data": {"summary": [
                {"totalClaims": "24", "totalPatientPay": "120.00", "totalPlanPaid": "580.00"},
                {"totalClaims": "12", "totalPatientPay": "60.00",  "totalPlanPaid": "290.00"},
            ]},
        }
        dsl = {"sections": [{"columns": [
            {"header": "Total Claims", "field": "totalClaims",      "format": "text"},
            {"header": "Patient Paid", "field": "totalPatientPay",  "format": "currency"},
            {"header": "Plan Paid",    "field": "totalPlanPaid",    "format": "currency"},
        ]}]}
        result = agent.execute("claim_summary", tool, {}, render_dsl=dsl, render_mode="table")
        assert result.render_format == "html_table"

    def test_reversal_info_respects_text_only(self, agent):
        """reversal_info moved to LLM-decides — text_only is now respected."""
        tool = {
            "tool_name": "get_reversal",
            "status": "success",
            "data": {"reversals": [{"originalClaimNumber": "RX10001", "reversalDate": "20250401", "reversalReason": "Patient returned medication"}]},
        }
        dsl = {"sections": [{"columns": [
            {"header": "Original Claim", "field": "originalClaimNumber", "format": "text"},
            {"header": "Reversal Date",  "field": "reversalDate",        "format": "date"},
            {"header": "Reason",         "field": "reversalReason",      "format": "text"},
        ]}]}
        result = agent.execute("reversal_info", tool, {}, render_dsl=dsl, render_mode="text_only")
        assert result.render_format == "text"

    def test_date_range_search_respects_text_only(self, agent):
        """date_range_search moved to LLM-decides — text_only is now respected."""
        result = agent.execute(
            "date_range_search",
            _tool_multi_row(),
            {},
            render_dsl=_dsl_6col(),
            render_mode="text_only",
        )
        assert result.render_format == "text"

    def test_expensive_claims_respects_text_only(self, agent):
        """expensive_claims moved to LLM-decides — text_only is now respected."""
        result = agent.execute(
            "expensive_claims",
            _tool_multi_row(),
            {},
            render_dsl=_dsl_6col(),
            render_mode="text_only",
        )
        assert result.render_format == "text"

    # ── Tier 3: Data-shape driven table ──────────────────────────────────────

    def test_multi_row_renders_table(self, agent):
        """Two rows → table for any Tier 3 intent (data shape beats LLM)."""
        result = agent.execute(
            "claim_status",
            _tool_multi_row(),
            {},
            render_dsl=_dsl_6col(),
        )
        assert result.render_format == "html_table"
        assert result.success is True

    def test_single_row_6col_is_text(self, agent):
        """1 row + 6 visible columns → text (single row always text regardless of col count)."""
        result = agent.execute(
            "claim_status",
            _tool_single_row(),
            {},
            render_dsl=_dsl_6col(),
        )
        assert result.render_format == "text"

    def test_single_row_any_cols_is_text(self, agent):
        """Single row with many columns → text (multiple rows required for table)."""
        dsl = {"sections": [{"columns": [
            {"header": "C1", "field": "claimNumber",            "format": "text"},
            {"header": "C2", "field": "claimStatusDescription", "format": "text"},
            {"header": "C3", "field": "fillDate",               "format": "date"},
            {"header": "C4", "field": "productName",            "format": "title"},
            {"header": "C5", "field": "patientPay",             "format": "currency"},
            {"header": "C6", "field": "clientPay",              "format": "currency"},
        ]}]}
        result = agent.execute("claim_status", _tool_single_row(), {}, render_dsl=dsl)
        assert result.render_format == "text"

    # ── Table HTML structure ──────────────────────────────────────────────────

    def test_table_contains_table_tag(self, agent):
        """html_table output must contain a <table> element."""
        result = agent.execute(
            "pricing_info",
            _tool_pricing(),
            {},
            render_dsl=_dsl_pricing(),
        )
        assert "<table" in result.html_content
        assert result.success is True

    def test_table_contains_actual_data(self, agent):
        """Table must render field values from both rows."""
        result = agent.execute(
            "claim_list",
            _tool_multi_row(),
            {},
            render_dsl=_dsl_6col(),
        )
        assert "RX10001" in result.html_content
        assert "RX10002" in result.html_content

    def test_table_answer_header_uses_intent_title(self, agent):
        """answer_header must match TABLE_TITLES[intent]."""
        result = agent.execute(
            "pricing_info",
            _tool_pricing(),
            {},
            render_dsl=_dsl_pricing(),
        )
        assert "Pricing Details" in result.answer_header

    def test_table_css_content_empty(self, agent):
        """css_content is empty — styling handled by frontend."""
        result = agent.execute(
            "claim_list",
            _tool_multi_row(),
            {},
            render_dsl=_dsl_6col(),
        )
        assert result.css_content == ""
        assert "<style>" not in result.html_content

    def test_table_has_no_button(self, agent):
        """Buttons removed — frontend handles expand/collapse."""
        result = agent.execute(
            "claim_list",
            _tool_multi_row(),
            {},
            render_dsl=_dsl_6col(),
        )
        assert "<button" not in result.html_content
        assert "requestFullscreen" not in result.html_content
        assert "mc-poc-" in result.html_content


# =============================================================================
# SAFETY NET OVERRIDE SCENARIOS
# Python intent-tier safety nets override the LLM's render_mode decision
# =============================================================================

class TestSafetyNetOverrides:
    """
    These tests verify the three-tier safety net cannot be defeated.
    The LLM outputs the wrong render_mode — Python overrides it.
    """

    def test_all_must_render_intents_override_text_only(self, agent):
        """Every MUST_RENDER intent → html_table even when render_mode=text_only."""
        for intent in MUST_RENDER_INTENTS:
            result = agent.execute(
                intent,
                _tool_pricing(),
                {},
                render_dsl=_dsl_pricing(),
                render_mode="text_only",
            )
            assert result.render_format == "html_table", (
                f"MUST_RENDER intent '{intent}' returned {result.render_format!r} "
                f"— expected 'html_table' when render_mode=text_only"
            )

    def test_must_render_with_no_render_mode_still_renders(self, agent):
        """MUST_RENDER intent with render_mode=None still renders table."""
        result = agent.execute(
            "pricing_info",
            _tool_pricing(),
            {},
            render_dsl=_dsl_pricing(),
            render_mode=None,
        )
        assert result.render_format == "html_table"

    def test_llm_decides_with_no_render_mode_uses_data_shape(self, agent):
        """render_mode=None + render_dsl present → data shape decides (3 cols, 1 row → text)."""
        result = agent.execute(
            "prescriber_info",
            _tool_prescriber(),
            {},
            render_dsl=_dsl_prescriber(),
            render_mode=None,
        )
        assert result.render_format == "text"

    def test_no_render_intent_overrides_table_render_mode(self, agent):
        """NO_RENDER intent with render_mode=table → text (Gate 1 fires first)."""
        for intent in NO_RENDER_INTENTS:
            result = agent.execute(
                intent,
                _tool_multi_row(),
                {},
                render_dsl=_dsl_6col(),
                render_mode="table",
            )
            assert result.render_format == "text", (
                f"NO_RENDER intent '{intent}' returned {result.render_format!r} "
                f"— expected 'text' even when render_mode=table"
            )

    @pytest.mark.parametrize("intent", ["claim_status", "claim_details", "benefits_info"])
    def test_tier3_intents_respect_text_only(self, agent, intent):
        """Tier 3 intents (not in any special set) must respect render_mode=text_only."""
        result = agent.execute(
            intent,
            _tool_single_row(),
            {},
            render_dsl=_dsl_3col(),
            render_mode="text_only",
        )
        assert result.render_format == "text", (
            f"Tier 3 intent '{intent}' returned {result.render_format!r} "
            f"— expected 'text' when render_mode=text_only"
        )

    def test_suppress_table_overrides_must_render_escape_hatch(self, agent):
        """suppress_table escape hatch fires BEFORE MUST_RENDER gate — allows text for genuinely absent data."""
        dsl = {
            "suppress_table": True,
            "sections": [{"columns": [
                {"header": "Patient Pay", "field": "patientPay", "format": "currency"},
                {"header": "Plan Paid",   "field": "planPaid",   "format": "currency"},
            ]}],
        }
        result = agent.execute(
            "pricing_info",    # MUST_RENDER but suppress_table escape hatch fires first
            _tool_pricing(),
            {},
            render_dsl=dsl,
        )
        assert result.render_format == "text"

    def test_suppress_table_honored_for_llm_decides_intents(self, agent):
        """suppress_table is honored for non-MUST_RENDER intents."""
        dsl = {
            "suppress_table": True,
            "sections": [{"columns": [
                {"header": "Prescriber Name", "field": "prescriberName", "format": "title"},
                {"header": "NPI",             "field": "npi",            "format": "text"},
                {"header": "DEA",             "field": "dea",            "format": "text"},
            ]}],
        }
        result = agent.execute(
            "prescriber_info",    # LLM-decides — suppress_table works
            _tool_prescriber(),
            {},
            render_dsl=dsl,
        )
        assert result.render_format == "text"


# =============================================================================
# BOUNDARY / COLUMN-COUNT EDGE CASES
# =============================================================================

class TestColumnCountBoundary:
    """
    The text-vs-table boundary for single-row results sits at visible-column count = 6.
      <6 cols → text  (LLM prose is sufficient)
      ≥6 cols → table (too complex for prose)
    These tests pin both sides of that boundary for LLM-decides intents.
    """

    def test_5_visible_cols_single_row_is_text(self, agent):
        """Exactly 5 visible columns, 1 row → text (<6 threshold)."""
        dsl = {"sections": [{"columns": [
            {"header": "A", "field": "claimNumber",            "format": "text"},
            {"header": "B", "field": "claimStatusDescription", "format": "text"},
            {"header": "C", "field": "fillDate",               "format": "date"},
            {"header": "D", "field": "productName",            "format": "title"},
            {"header": "E", "field": "patientPay",             "format": "currency"},
        ]}]}
        result = MyclaimsRenderingAgent().execute(
            "claim_status", _tool_single_row(), {}, render_dsl=dsl
        )
        assert result.render_format == "text"

    def test_6_visible_cols_single_row_is_text(self, agent):
        """Single row with 6 columns → text (single row always text regardless of column count)."""
        result = MyclaimsRenderingAgent().execute(
            "claim_status", _tool_single_row(), {}, render_dsl=_dsl_6col()
        )
        assert result.render_format == "text"

    def test_unknown_render_mode_value_treated_as_none(self, agent):
        """render_mode='garbage' (not in VALID_RENDER_MODES) falls through to data shape."""
        # Single row + 3 cols → text (< 6 cols)
        result = agent.execute(
            "claim_status",
            _tool_single_row(),
            {},
            render_dsl=_dsl_3col(),
            render_mode="garbage",
        )
        assert result.render_format == "text"

    def test_empty_data_no_dsl_returns_text(self, agent):
        """Empty claims list + no render_dsl → text (no columns to extract)."""
        result = agent.execute(
            "claim_list",
            {"tool_name": "get_claims", "status": "success", "data": {"claims": []}},
            {},
        )
        assert result.render_format == "text"

    def test_render_time_always_set(self, agent):
        """render_time_ms must be ≥ 0 for every execution path."""
        cases = [
            ("claim_status",   _tool_single_row(), _dsl_3col(),    "text_only"),
            ("pricing_info",   _tool_pricing(),    _dsl_pricing(), "text_only"),
            ("prescriber_info",_tool_prescriber(), _dsl_prescriber(), None),
            ("greeting",       {"status": "success", "data": {}}, None, None),
        ]
        for intent, tool, dsl, rm in cases:
            result = agent.execute(intent, tool, {}, render_dsl=dsl, render_mode=rm)
            assert result.render_time_ms >= 0, f"render_time_ms not set for intent={intent}"


# =============================================================================
# process_rendering() INTEGRATION — render_mode flows end-to-end
# =============================================================================

class TestProcessRenderingWithRenderMode:
    """Integration: render_mode in the state dict flows through api/routes.py."""

    def test_text_only_state_returns_text_format(self):
        """render_mode=text_only in state → process_rendering returns render_format=text."""
        from api.routes import process_rendering
        state = {
            "intent": "claim_status",
            "tool_results": _tool_single_row(),
            "entities": {},
            "needs_clarification": False,
            "render_dsl": _dsl_3col(),
            "render_mode": "text_only",
        }
        result = process_rendering(state)
        assert result["render_format"] == "text"

    def test_render_mode_key_in_result_dict(self):
        """process_rendering must include render_mode in its return dict."""
        from api.routes import process_rendering
        state = {
            "intent": "claim_list",
            "tool_results": _tool_multi_row(),
            "entities": {},
            "needs_clarification": False,
            "render_dsl": _dsl_6col(),
            "render_mode": "table",
        }
        result = process_rendering(state)
        assert "render_mode" in result
        assert result["render_mode"] == "table"

    def test_must_render_intent_overrides_text_only_in_state(self):
        """MUST_RENDER intent with render_mode=text_only in state → html_table."""
        from api.routes import process_rendering
        state = {
            "intent": "pricing_info",
            "tool_results": _tool_pricing(),
            "entities": {},
            "needs_clarification": False,
            "render_dsl": _dsl_pricing(),
            "render_mode": "text_only",
        }
        result = process_rendering(state)
        assert result["render_format"] == "html_table"

    def test_always_card_intent_honors_text_only_in_state(self):
        """ALWAYS_CARD intent with render_mode=text_only in state → text (LLM-decides tier)."""
        from api.routes import process_rendering
        state = {
            "intent": "prescriber_info",
            "tool_results": _tool_prescriber(),
            "entities": {},
            "needs_clarification": False,
            "render_dsl": _dsl_prescriber(),
            "render_mode": "text_only",
        }
        result = process_rendering(state)
        assert result["render_format"] == "text"

    def test_underscore_key_tool_results_works(self):
        """Streaming events use _tool_results — process_rendering must handle both."""
        from api.routes import process_rendering
        state = {
            "intent": "claim_list",
            "_tool_results": _tool_multi_row(),
            "_entities": {},
            "needs_clarification": False,
            "render_dsl": _dsl_6col(),
            "render_mode": "table",
        }
        result = process_rendering(state)
        assert result["render_format"] == "html_table"

    def test_needs_clarification_always_text(self):
        """needs_clarification=True bypasses rendering regardless of intent."""
        from api.routes import process_rendering
        state = {
            "intent": "pricing_info",
            "tool_results": _tool_pricing(),
            "needs_clarification": True,
            "render_dsl": _dsl_pricing(),
            "render_mode": "table",
        }
        result = process_rendering(state)
        assert result["render_format"] == "text"

    def test_empty_state_never_raises(self):
        """process_rendering must not raise on an empty/incomplete state dict."""
        from api.routes import process_rendering
        result = process_rendering({})
        assert result["render_format"] == "text"
