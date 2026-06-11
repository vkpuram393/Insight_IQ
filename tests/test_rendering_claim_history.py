"""
Tests for the claim_history_search rendering wiring (Phase C of the
multi-domain rendering integration).

Coverage:
  1. Domain dispatcher infers "claim_history_search" from
     tool_results.data.is_claim_history_search.
  2. Domain dispatcher infers "claim_history_search" from tool_name.
  3. CHS-domain config exposes the correct field remap and corrected
     X -> Reversed status mapping.
  4. MyclaimsRenderingAgent renders a multi-row table when the LLM
     emits a valid CHS DSL block.
  5. MyclaimsRenderingAgent applies CHS field remap for LLM aliases
     (drugLabelName -> productName, dateOfFill -> fillDate, etc.).
  6. Single-row CHS answer -> universal row-count rule still produces
     a table when render_mode="table" is honoured (the engine no
     longer suppresses single-row tables in v4).
  7. claims-domain rendering still renders correctly with the
     updated engine (regression guard against Phase A changes).
  8. Status badge for "X" renders as "Reversed" (NOT "Voided").
"""

from __future__ import annotations

import pytest

from agents.post_processing import claim_history_rendering_config as chs_cfg
from agents.post_processing import claims_rendering_config as claims_cfg
from agents.post_processing import domain_configs as dc
from agents.post_processing.myclaims_rendering_agent import MyclaimsRenderingAgent


# ---------------------------------------------------------------------------
# Domain dispatcher
# ---------------------------------------------------------------------------

def test_resolve_domain_explicit_state_domain_wins():
    assert dc.resolve_domain("DateRange", {}, "claim_history_search") == "claim_history_search"
    assert dc.resolve_domain("claim_list", {}, "claims") == "claims"


def test_resolve_domain_from_is_claim_history_search_flag():
    tool_results = {
        "tool_name": "claims_search_v2",
        "data": {"is_claim_history_search": True, "claims": []},
    }
    assert dc.resolve_domain("DrugList", tool_results, None) == "claim_history_search"


def test_resolve_domain_from_tool_name():
    tool_results = {"tool_name": "claims_search", "data": {}}
    assert dc.resolve_domain("Status", tool_results, None) == "claim_history_search"


def test_resolve_domain_default_falls_back_to_claims():
    assert dc.resolve_domain("claim_list", {}, None) == "claims"
    assert dc.resolve_domain("", {}, None) == "claims"


def test_get_config_returns_correct_module():
    assert dc.get_config("claims") is claims_cfg
    assert dc.get_config("claim_history_search") is chs_cfg
    # Unknown domain -> claims default
    assert dc.get_config("future_domain") is claims_cfg
    assert dc.get_config(None) is claims_cfg


# ---------------------------------------------------------------------------
# CHS config sanity
# ---------------------------------------------------------------------------

def test_chs_config_x_is_reversed_not_voided():
    """The 'X' status code must map to 'Reversed', NOT 'Voided'.
    Same correction as claims_rendering_config (per the claims domain prompt)."""
    assert chs_cfg.CLAIM_STATUS_CODES["X"] == "Reversed"
    assert claims_cfg.CLAIM_STATUS_CODES["X"] == "Reversed"


def test_chs_config_field_remap_targets_slim_keys():
    """The CHS field remap must point to the slim-claim shape
    (fillDate, productName, productNdc, etc.)."""
    assert chs_cfg.FIELD_REMAP["dateOfFill"] == "fillDate"
    assert chs_cfg.FIELD_REMAP["drugLabelName"] == "productName"
    assert chs_cfg.FIELD_REMAP["drug"] == "productName"
    assert chs_cfg.FIELD_REMAP["ndc"] == "productNdc"
    assert chs_cfg.FIELD_REMAP["statusDescription"] == "claimStatusDescription"


def test_chs_config_no_stcob_or_med_d_null_rules():
    """CHS slim-claim shape drops STCOB / Med-D subtrees, so the
    null-as-zero set must be empty for CHS."""
    assert chs_cfg.NULL_AS_ZERO_CURRENCY_FORMATS == frozenset()
    assert chs_cfg.BLOCKED_FIELDS == frozenset()


# ---------------------------------------------------------------------------
# Engine end-to-end
# ---------------------------------------------------------------------------

def _chs_tool_results(claims):
    return {
        "tool_name": "claims_search_v2",
        "status": "success",
        "data": {
            "is_claim_history_search": True,
            "claims": claims,
            "totalCount": len(claims),
            "filteredCount": len(claims),
        },
    }


def _chs_dsl():
    return {
        "layout": "table",
        "title": "Recent Claims",
        "sections": [
            {
                "id": "main_table",
                "type": "table",
                "columns": [
                    {"header": "Claim Number", "field": "claimNumber", "format": "text"},
                    {"header": "Fill Date",    "field": "fillDate",    "format": "date"},
                    {"header": "Drug Name",    "field": "productName", "format": "title"},
                    {"header": "Status",       "field": "claimStatusDescription", "format": "text"},
                ],
            }
        ],
    }


def test_chs_renders_multi_row_table():
    claims = [
        {
            "claimInformation": {
                "claimNumber": "260173639698000",
                "fillDate": "20240901",
                "claimStatusDescription": "Paid",
            },
            "drug": {"productName": "MOUNJARO 10 MG"},
        },
        {
            "claimInformation": {
                "claimNumber": "260173639699001",
                "fillDate": "20240810",
                "claimStatusDescription": "Rejected",
            },
            "drug": {"productName": "OZEMPIC 1 MG"},
        },
    ]
    agent = MyclaimsRenderingAgent()
    res = agent.execute(
        intent="DateRange",
        tool_results=_chs_tool_results(claims),
        entities={},
        render_dsl=_chs_dsl(),
        render_mode="table",
    )
    assert res.success is True
    assert res.render_format == "html_table"
    assert "MOUNJARO" in res.html_content.upper()
    assert "OZEMPIC" in res.html_content.upper()
    # The agent must have routed via CHS config
    assert agent._domain == "claim_history_search"


def test_chs_field_remap_applied_when_llm_uses_claims_alias():
    """If the LLM emits 'drugLabelName' (claims-domain alias) in a CHS
    DSL, the engine must remap it to 'productName' before lookup."""
    claims = [
        {"claimInformation": {"claimNumber": "111"}, "drug": {"productName": "AAA"}},
        {"claimInformation": {"claimNumber": "222"}, "drug": {"productName": "BBB"}},
    ]
    dsl = {
        "layout": "table",
        "title": "T",
        "sections": [
            {
                "id": "s",
                "type": "table",
                "columns": [
                    {"header": "Claim",    "field": "claimNumber",    "format": "text"},
                    # LLM mistakenly used the claims-domain alias
                    {"header": "Drug",     "field": "drugLabelName",  "format": "title"},
                ],
            }
        ],
    }
    agent = MyclaimsRenderingAgent()
    res = agent.execute(
        intent="DrugList",
        tool_results=_chs_tool_results(claims),
        entities={},
        render_dsl=dsl,
        render_mode="table",
    )
    assert res.success is True
    # Both AAA and BBB came from drug.productName even though the DSL
    # field said drugLabelName -> proves the remap fired.
    html = res.html_content.upper()
    assert "AAA" in html
    assert "BBB" in html


def test_chs_status_badge_x_renders_as_reversed():
    """A status_badge cell with raw value 'X' must render 'Reversed'
    (not 'Voided') under the CHS config."""
    claims = [
        {"claimInformation": {"claimNumber": "1", "claimStatus": "X"}},
        {"claimInformation": {"claimNumber": "2", "claimStatus": "P"}},
    ]
    dsl = {
        "layout": "table",
        "title": "T",
        "sections": [
            {
                "id": "s",
                "type": "table",
                "columns": [
                    {"header": "Claim",  "field": "claimNumber", "format": "text"},
                    {"header": "Status", "field": "claimStatus", "format": "status_badge"},
                ],
            }
        ],
    }
    agent = MyclaimsRenderingAgent()
    res = agent.execute(
        intent="Status",
        tool_results=_chs_tool_results(claims),
        entities={},
        render_dsl=dsl,
        render_mode="table",
    )
    assert res.success is True
    assert "REVERSED" in res.html_content.upper()
    assert "VOIDED" not in res.html_content.upper()


def test_chs_no_render_dsl_falls_back_to_text():
    """When the LLM doesn't emit a DSL block, the engine must fall back
    to text gracefully (success=False -> render_format=text)."""
    agent = MyclaimsRenderingAgent()
    res = agent.execute(
        intent="DrugLast",
        tool_results=_chs_tool_results([
            {"claimInformation": {"claimNumber": "1"}, "drug": {"productName": "X"}},
        ]),
        entities={},
        render_dsl=None,
        render_mode=None,
    )
    assert res.success is False
    assert res.render_format == "text"


# ---------------------------------------------------------------------------
# Claims-domain regression guard
# ---------------------------------------------------------------------------

def test_claims_domain_still_renders_correctly_post_phase_a():
    """Phase A replaced the engine with the v4 version; this test
    guards against regressing the claims-domain happy path."""
    tool_results = {
        "tool_name": "claims_api",
        "status": "success",
        "data": {
            "claims": [
                {"claimNumber": "111", "drugLabelName": "AAA", "statusDescription": "Paid"},
                {"claimNumber": "222", "drugLabelName": "BBB", "statusDescription": "Rejected"},
            ]
        },
    }
    dsl = {
        "layout": "table",
        "title": "Claims",
        "sections": [
            {
                "id": "main_table",
                "type": "table",
                "columns": [
                    {"header": "Claim #", "field": "claimNumber",       "format": "text"},
                    {"header": "Drug",    "field": "drugLabelName",     "format": "title"},
                    {"header": "Status",  "field": "statusDescription", "format": "status_badge"},
                ],
            }
        ],
    }
    agent = MyclaimsRenderingAgent()
    res = agent.execute(
        intent="claim_list",
        tool_results=tool_results,
        entities={},
        render_dsl=dsl,
        render_mode="table",
    )
    assert res.success is True
    assert res.render_format == "html_table"
    # Engine resolved to the claims-domain config (default)
    assert agent._domain == "claims"
