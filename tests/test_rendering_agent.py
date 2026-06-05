"""Tests for post-processing rendering agent."""
import json
import pytest
from unittest.mock import patch
from agents.post_processing.myclaims_rendering_agent import MyclaimsRenderingAgent
from agents.post_processing.rendering_themes import NO_RENDER_INTENTS


@pytest.fixture
def agent():
    return MyclaimsRenderingAgent()


def _shape_a_render_dsl():
    """render_dsl with flat field names matching _shape_a_tool_results() data."""
    return {
        "layout": "table",
        "title": "Claims Data",
        "sections": [{
            "id": "main_table",
            "type": "table",
            "title": "Claims Data",
            "columns": [
                {"header": "Claim #", "field": "claimNumber", "format": "text"},
                {"header": "Status", "field": "claimStatusDescription", "format": "status_badge"},
                {"header": "Fill Date", "field": "fillDate", "format": "date"},
                {"header": "Drug", "field": "productName", "format": "title"},
                {"header": "Patient Pay", "field": "patientPay", "format": "currency"},
                {"header": "Plan Paid", "field": "clientPay", "format": "currency"},
                {"header": "Qty", "field": "quantity", "format": "text"},
                {"header": "Days Supply", "field": "daysSupplied", "format": "text"},
                {"header": "Reject Codes", "field": "rejectCodes", "format": "reject_codes"},
            ],
        }],
    }


# ---------------------------------------------------------------------------
# Shape A — data.claims[]
# ---------------------------------------------------------------------------

def _shape_a_tool_results():
    return {
        "tool_name": "get_claims",
        "status": "success",
        "data": {
            "claims": [
                {
                    "claimInformation": {
                        "claimNumber": "RX12345",
                        "claimStatusDescription": "Paid",
                        "fillDate": "20250115",
                        "quantity": "30",
                        "daysSupplied": "30",
                    },
                    "drug": {"productName": "LISINOPRIL 10MG"},
                    "pricing": {"patientPay": "5.00", "clientPay": "40.00"},
                },
                {
                    "claimInformation": {
                        "claimNumber": "RX12346",
                        "claimStatusDescription": "Denied",
                        "fillDate": "20250120",
                    },
                    "drug": {"productName": "METFORMIN 500MG"},
                    "pricing": {"patientPay": "0.00", "clientPay": "0"},
                    "messages": {"rejectCodes": ["70", "75"]},
                },
            ]
        },
    }


class TestShapeA:
    def test_basic_rendering(self, agent):
        result = agent.execute("claim_list", _shape_a_tool_results(), {},
                               render_dsl=_shape_a_render_dsl())
        assert result.render_format == "html_table"
        assert result.success is True
        assert "Claims Data" in result.answer_header
        assert "2 records" in result.answer_header

    def test_html_contains_claim_data(self, agent):
        result = agent.execute("claim_list", _shape_a_tool_results(), {},
                               render_dsl=_shape_a_render_dsl())
        assert "RX12345" in result.html_content
        assert "RX12346" in result.html_content
        assert "Lisinopril 10Mg" in result.html_content

    def test_zero_dollar_renders_correctly(self, agent):
        result = agent.execute("claim_list", _shape_a_tool_results(), {},
                               render_dsl=_shape_a_render_dsl())
        assert "$0.00" in result.html_content

    def test_reject_codes_rendered(self, agent):
        result = agent.execute("claim_list", _shape_a_tool_results(), {},
                               render_dsl=_shape_a_render_dsl())
        assert "70" in result.html_content

    def test_no_button_in_html(self, agent):
        """Buttons removed — frontend handles expand/collapse."""
        result = agent.execute("claim_list", _shape_a_tool_results(), {},
                               render_dsl=_shape_a_render_dsl())
        assert "<button" not in result.html_content
        assert "onclick" not in result.html_content
        assert "requestFullscreen" not in result.html_content

    def test_table_directly_in_container(self, agent):
        """Table is directly in container — no hidden wrapper div."""
        result = agent.execute("claim_list", _shape_a_tool_results(), {},
                               render_dsl=_shape_a_render_dsl())
        assert "table-wrapper" not in result.html_content
        assert "<table" in result.html_content

    def test_css_content_empty(self, agent):
        """css_content is empty — styling handled by frontend."""
        result = agent.execute("claim_list", _shape_a_tool_results(), {},
                               render_dsl=_shape_a_render_dsl())
        assert result.css_content == ""

    def test_html_classes_scoped(self, agent):
        """HTML still uses mc-poc- scoped class names for frontend to target."""
        result = agent.execute("claim_list", _shape_a_tool_results(), {},
                               render_dsl=_shape_a_render_dsl())
        assert "mc-poc-" in result.html_content
        assert "<style>" not in result.html_content

    def test_status_badges_use_semantic_class(self, agent):
        """Status badges use mc-poc-status-{label} class instead of inline styles."""
        result = agent.execute("claim_list", _shape_a_tool_results(), {},
                               render_dsl=_shape_a_render_dsl())
        assert 'style=' not in result.html_content
        assert 'mc-poc-status-' in result.html_content

    def test_render_time_tracked(self, agent):
        result = agent.execute("claim_list", _shape_a_tool_results(), {},
                               render_dsl=_shape_a_render_dsl())
        assert result.render_time_ms > 0


# ---------------------------------------------------------------------------
# No render_dsl → no table (shape detection removed)
# ---------------------------------------------------------------------------

class TestNoRenderDsl:
    def test_no_render_dsl_returns_no_data(self):
        """Without render_dsl, no table is rendered."""
        result = MyclaimsRenderingAgent().execute(
            "claim_list", _shape_a_tool_results(), {},
        )
        assert result.success is True
        assert "No claims data found" in result.html_content

    def test_render_dsl_with_matching_fields_renders_arbitrary_data(self):
        """render_dsl dynamically finds fields in any data structure."""
        tool_results = {
            "tool_name": "benefits_api",
            "status": "success",
            "data": {
                "copay_schedule": [
                    {"tier": "1 - Preferred Generic", "copay": "10.00"},
                ]
            },
        }
        dsl = {
            "sections": [{
                "columns": [
                    {"header": "Tier", "field": "tier", "format": "text"},
                    {"header": "Copay", "field": "copay", "format": "currency"},
                ],
            }]
        }
        result = MyclaimsRenderingAgent().execute("copay_info", tool_results, {},
                                                   render_dsl=dsl)
        assert result.success is True
        assert result.render_format == "html_table"
        assert "1 - Preferred Generic" in result.html_content
        assert "$10.00" in result.html_content


# ---------------------------------------------------------------------------
# Guard conditions
# ---------------------------------------------------------------------------

class TestGuards:
    def test_blocklist_intent_skipped(self, agent):
        for intent in NO_RENDER_INTENTS:
            result = agent.execute(intent, _shape_a_tool_results(), {})
            assert result.render_format == "text"
            assert result.success is False
            assert result.css_content == ""

    def test_failed_status_skipped(self, agent):
        tool_results = {"status": "error", "data": {}, "message": "API timeout"}
        result = agent.execute("claim_list", tool_results, {})
        assert result.render_format == "text"
        assert result.success is False

    def test_enum_status_renders_correctly(self, agent):
        """ToolExecutionStatus(str, Enum) members from Pydantic v2 .dict() must not break rendering."""
        from enum import Enum

        class FakeStatus(str, Enum):
            SUCCESS = "success"

        tool_results = {
            "tool_name": "get_claims",
            "status": FakeStatus.SUCCESS,
            "data": {"claims": [
                {
                    "claimInformation": {"claimNumber": "RX99999",
                                         "claimStatusDescription": "Paid",
                                         "fillDate": "20250401"},
                    "drug": {"productName": "TESTDRUG 10MG"},
                    "pricing": {"patientPay": "5.00"},
                },
            ]},
        }
        result = agent.execute("claim_list", tool_results, {},
                               render_dsl=_shape_a_render_dsl())
        assert result.render_format == "html_table"
        assert result.success is True
        assert "RX99999" in result.html_content

    def test_empty_data_renders_fallback(self, agent):
        tool_results = {"status": "success", "data": {"claims": []}}
        result = agent.execute("claim_list", tool_results, {})
        assert result.render_format == "html_table"
        assert "No claims data found" in result.html_content
        assert result.css_content
        assert "<style>" not in result.css_content


# ---------------------------------------------------------------------------
# Intent-aware titles
# ---------------------------------------------------------------------------

class TestIntentTitles:
    @pytest.mark.parametrize("intent,expected_title", [
        ("claim_list", "Claims Data"),
        ("claim_status", "Claim Status"),
        ("claim_pending", "Pending Claims"),
        ("copay_info", "Copay Schedule"),
        ("cob_info", "Coordination of Benefits"),
        ("benefits_info", "Benefits Information"),
    ])
    def test_intent_title(self, agent, intent, expected_title):
        result = agent.execute(intent, _shape_a_tool_results(), {},
                               render_dsl=_shape_a_render_dsl())
        assert expected_title in result.answer_header

    def test_unknown_intent_renders_with_dsl(self, agent):
        """Unknown intent renders data when render_dsl is provided."""
        result = agent.execute("some_new_intent", _shape_a_tool_results(), {},
                               render_dsl=_shape_a_render_dsl())
        assert result.success is True
        assert result.render_format == "html_table"
        assert "RX12345" in result.html_content


# ---------------------------------------------------------------------------
# process_rendering() integration
# ---------------------------------------------------------------------------

class TestProcessRendering:
    def test_process_rendering_with_tool_results(self):
        from api.routes import process_rendering
        state = {
            "intent": "claim_list",
            "tool_results": _shape_a_tool_results(),
            "entities": {},
            "needs_clarification": False,
            "render_dsl": _shape_a_render_dsl(),
        }
        result = process_rendering(state)
        assert result["render_format"] == "html_table"
        assert result["html_content"] is not None
        assert "<table" in result["html_content"]
        assert result["css_content"] is not None
        assert "<style>" not in result["css_content"]
        assert ".mc-poc-" in result["css_content"]

    def test_process_rendering_with_underscore_keys(self):
        from api.routes import process_rendering
        event_data = {
            "intent": "claim_list",
            "_tool_results": _shape_a_tool_results(),
            "_entities": {},
            "needs_clarification": False,
            "render_dsl": _shape_a_render_dsl(),
        }
        result = process_rendering(event_data)
        assert result["render_format"] == "html_table"

    def test_process_rendering_greeting_skipped(self):
        from api.routes import process_rendering
        state = {
            "intent": "greeting",
            "tool_results": None,
            "needs_clarification": False,
        }
        result = process_rendering(state)
        assert result["render_format"] == "text"

    def test_process_rendering_clarification_skipped(self):
        from api.routes import process_rendering
        state = {
            "intent": "claim_list",
            "tool_results": _shape_a_tool_results(),
            "needs_clarification": True,
        }
        result = process_rendering(state)
        assert result["render_format"] == "text"

    def test_process_rendering_enum_status(self):
        """Pydantic v2 .dict() returns enum members — process_rendering handles them."""
        from enum import Enum
        from api.routes import process_rendering

        class FakeStatus(str, Enum):
            SUCCESS = "success"

        tool_results = dict(_shape_a_tool_results())
        tool_results["status"] = FakeStatus.SUCCESS
        state = {
            "intent": "claim_list",
            "tool_results": tool_results,
            "entities": {},
            "needs_clarification": False,
            "render_dsl": _shape_a_render_dsl(),
        }
        result = process_rendering(state)
        assert result["render_format"] == "html_table"
        assert result["html_content"] is not None

    def test_process_rendering_never_raises(self):
        from api.routes import process_rendering
        result = process_rendering({})
        assert result["render_format"] == "text"


# ---------------------------------------------------------------------------
# LLM structure extraction
# ---------------------------------------------------------------------------

class TestStructureExtraction:
    """Unit tests for the Phase 1-2 extraction infrastructure.

    These tests never call Vertex AI — generate() is either mocked or the
    test exercises a cache-hit path where it is never reached.
    """

    # ── SchemaExtractor ──────────────────────────────────────────────────

    def test_schema_strips_pii(self):
        from agents.post_processing.schema_extractor import extract_schema
        schema = extract_schema({"claimNumber": "260173639698000", "patientPay": 50.0})
        assert schema["claimNumber"] == "string"
        assert schema["patientPay"] == "number"
        assert "260173639698000" not in json.dumps(schema)

    def test_schema_handles_nested(self):
        from agents.post_processing.schema_extractor import extract_schema
        raw = {"data": {"claims": [{"claimInformation": {"claimNumber": "RX1"}}]}}
        schema = extract_schema(raw)
        assert schema["data"]["claims"][0]["claimInformation"]["claimNumber"] == "string"

    def test_schema_handles_empty_list(self):
        from agents.post_processing.schema_extractor import extract_schema
        schema = extract_schema({"items": []})
        assert schema["items"] == []

    def test_schema_handles_booleans(self):
        from agents.post_processing.schema_extractor import extract_schema
        schema = extract_schema({"active": True, "count": 3, "name": "X"})
        assert schema["active"] == "boolean"
        assert schema["count"] == "number"
        assert schema["name"] == "string"

    # ── PathExtractor ─────────────────────────────────────────────────────

    def test_get_by_path_nested(self):
        from agents.post_processing.path_extractor import get_by_path
        obj = {"claimInformation": {"claimNumber": "RX12345"}}
        assert get_by_path(obj, "claimInformation.claimNumber") == "RX12345"

    def test_get_by_path_missing_key_returns_empty(self):
        from agents.post_processing.path_extractor import get_by_path
        assert get_by_path({"a": {"b": 1}}, "a.c.d") == ""

    def test_get_by_path_never_raises(self):
        from agents.post_processing.path_extractor import get_by_path
        assert get_by_path(None, "any.path") == ""
        assert get_by_path({}, "") == ""
        assert get_by_path("scalar", "some.key") == ""

    def test_extract_rows_correct_values(self):
        from agents.post_processing.path_extractor import extract_rows
        from agents.post_processing.column_mapping import ColumnDef, ColumnMapping
        mapping = ColumnMapping(
            data_path="data.claims",
            columns=[
                ColumnDef("Claim #", "claimInformation.claimNumber", "text"),
                ColumnDef("Status",  "claimInformation.claimStatusDescription", "status_badge"),
            ],
            tool_name="get_claims",
            intent="claim_list",
            created_at="",
        )
        rows = extract_rows(_shape_a_tool_results(), mapping)
        assert len(rows) == 2
        assert rows[0]["Claim #"] == "RX12345"
        assert rows[1]["Claim #"] == "RX12346"

    def test_extract_rows_empty_list(self):
        from agents.post_processing.path_extractor import extract_rows
        from agents.post_processing.column_mapping import ColumnDef, ColumnMapping
        mapping = ColumnMapping(
            data_path="data.claims",
            columns=[ColumnDef("Claim #", "claimInformation.claimNumber", "text"),
                     ColumnDef("Status", "claimInformation.claimStatusDescription", "status_badge")],
            tool_name="t", intent="i", created_at="",
        )
        rows = extract_rows({"status": "success", "data": {"claims": []}}, mapping)
        assert rows == []

    # ── ExtractionCache ──────────────────────────────────────────────────

    def test_cache_roundtrip(self, tmp_path):
        from agents.post_processing.extraction_cache import ExtractionCache
        from agents.post_processing.column_mapping import ColumnDef, ColumnMapping
        cache = ExtractionCache(cache_path=str(tmp_path / "test_cache.json"))
        mapping = ColumnMapping(
            data_path="data.claims",
            columns=[
                ColumnDef("Claim #", "claimInformation.claimNumber", "text"),
                ColumnDef("Status",  "claimInformation.claimStatusDescription", "status_badge"),
            ],
            tool_name="claims_api",
            intent="claim_list",
            created_at="2025-01-01T00:00:00+00:00",
        )
        cache.set(mapping)
        retrieved = cache.get("claims_api", "claim_list")
        assert retrieved is not None
        assert retrieved.data_path == "data.claims"
        assert retrieved.columns[0].header == "Claim #"
        assert retrieved.columns[1].format == "status_badge"

    def test_cache_miss_returns_none(self, tmp_path):
        from agents.post_processing.extraction_cache import ExtractionCache
        cache = ExtractionCache(cache_path=str(tmp_path / "empty_cache.json"))
        assert cache.get("nonexistent_tool", "nonexistent_intent") is None

    def test_cache_load_failure_returns_empty(self, tmp_path):
        from agents.post_processing.extraction_cache import ExtractionCache
        bad_path = str(tmp_path / "bad.json")
        with open(bad_path, "w") as f:
            f.write("not valid json {{{{")
        cache = ExtractionCache(cache_path=bad_path)
        # Should not raise — returns None on load failure
        assert cache.get("any", "any") is None

    # ── validate_columns ─────────────────────────────────────────────────

    def test_invalid_paths_filtered(self):
        from agents.post_processing.column_mapping import ColumnDef
        from agents.post_processing.structure_extractor import validate_columns
        sample = {"claimInformation": {"claimNumber": "RX1", "claimStatusDescription": "Paid"}}
        cols = validate_columns([
            ColumnDef("Bad",     "nonexistent.field",                       "text"),
            ColumnDef("Claim #", "claimInformation.claimNumber",            "text"),
            ColumnDef("Status",  "claimInformation.claimStatusDescription", "status_badge"),
        ], sample)
        assert len(cols) == 2
        assert cols[0].header == "Claim #"
        assert cols[1].header == "Status"

    def test_fewer_than_2_valid_paths_raises(self):
        from agents.post_processing.column_mapping import ColumnDef
        from agents.post_processing.structure_extractor import validate_columns
        with pytest.raises(ValueError):
            validate_columns(
                [ColumnDef("Bad", "bad.path", "text")],
                {"claimInformation": {"claimNumber": "X"}},
            )

    # ── StructureExtractor JSON parsing ──────────────────────────────────

    def test_json_parse_direct(self):
        from agents.post_processing.structure_extractor import StructureExtractor
        extractor = StructureExtractor()
        parsed = extractor._parse_json(
            '{"data_path": "data.claims", "columns": ['
            '{"header": "Claim #", "path": "claimInformation.claimNumber", "format": "text"},'
            '{"header": "Status", "path": "claimInformation.claimStatusDescription", "format": "status_badge"}'
            ']}'
        )
        assert parsed["data_path"] == "data.claims"
        assert len(parsed["columns"]) == 2

    def test_json_parse_strips_markdown_fences(self):
        from agents.post_processing.structure_extractor import StructureExtractor
        extractor = StructureExtractor()
        raw = (
            "```json\n"
            '{"data_path": "data.claims", "columns": ['
            '{"header": "A", "path": "x.y", "format": "text"},'
            '{"header": "B", "path": "x.z", "format": "date"}'
            "]}\n```"
        )
        parsed = extractor._parse_json(raw)
        assert parsed["data_path"] == "data.claims"

    def test_json_parse_embedded_in_text(self):
        from agents.post_processing.structure_extractor import StructureExtractor
        extractor = StructureExtractor()
        raw = (
            "Here is the mapping:\n"
            '{"data_path": "data.claims", "columns": ['
            '{"header": "Claim #", "path": "a.b", "format": "text"},'
            '{"header": "Status", "path": "c.d", "format": "status_badge"}'
            "]}"
        )
        parsed = extractor._parse_json(raw)
        assert len(parsed["columns"]) == 2

    def test_json_parse_raises_on_no_columns_key(self):
        from agents.post_processing.structure_extractor import StructureExtractor
        extractor = StructureExtractor()
        with pytest.raises(ValueError, match="No parseable JSON"):
            extractor._parse_json('{"data_path": "x"}')  # missing "columns"

    # ── Integration: render_dsl with flat field names → dynamic extraction ──

    def test_render_dsl_used_when_provided(self, agent):
        """render_dsl with flat field names drives dynamic table extraction."""
        render_dsl = {
            "sections": [{
                "columns": [
                    {"header": "Claim #", "field": "claimNumber", "format": "text"},
                    {"header": "Status", "field": "claimStatusDescription", "format": "status_badge"},
                    {"header": "Drug", "field": "productName", "format": "title"},
                ],
            }]
        }
        result = agent.execute("claim_list", _shape_a_tool_results(), {},
                               render_dsl=render_dsl)
        assert result.render_format == "html_table"
        assert result.success is True
        assert "RX12345" in result.html_content
        assert "RX12346" in result.html_content

    # ── Integration: no render_dsl → no table ────────────────────────────

    def test_no_table_without_render_dsl(self, agent):
        """Without render_dsl, no table is rendered (no shape detection fallback)."""
        result = agent.execute("claim_list", _shape_a_tool_results(), {})
        assert result.success is True
        assert "No claims data found" in result.html_content

    # ── Integration: render_dsl with formatting ──────────────────────────

    def test_render_dsl_renders_table_with_formatting(self, agent):
        """render_dsl columns apply correct format types in _cell()."""
        render_dsl = {
            "sections": [{
                "columns": [
                    {"header": "Claim #", "field": "claimNumber", "format": "text"},
                    {"header": "Status", "field": "claimStatusDescription", "format": "status_badge"},
                    {"header": "Fill Date", "field": "fillDate", "format": "date"},
                    {"header": "Drug", "field": "productName", "format": "title"},
                    {"header": "Patient Pay", "field": "patientPay", "format": "currency"},
                ],
            }]
        }
        result = agent.execute("claim_list", _shape_a_tool_results(), {},
                               render_dsl=render_dsl)
        assert result.render_format == "html_table"
        assert result.success is True
        assert "RX12345" in result.html_content
        assert "RX12346" in result.html_content
        assert "01/15/2025" in result.html_content
        assert "Paid" in result.html_content or "PAID" in result.html_content
        assert "$5.00" in result.html_content
        assert "Lisinopril 10Mg" in result.html_content


# ---------------------------------------------------------------------------
# render_dsl edge cases
# ---------------------------------------------------------------------------

class TestRenderDsl:
    """Tests for _extract_from_dsl edge cases."""

    def test_render_dsl_bad_fields_returns_no_data(self):
        """Wrong field names in render_dsl → all-empty columns → no data."""
        bad_dsl = {
            "sections": [{
                "columns": [
                    {"header": "Bad1", "field": "totally_fake_field", "format": "text"},
                    {"header": "Bad2", "field": "another_fake", "format": "text"},
                ],
            }]
        }
        result = MyclaimsRenderingAgent().execute(
            "claim_list", _shape_a_tool_results(), {},
            render_dsl=bad_dsl,
        )
        assert result.success is True
        assert "No claims data found" in result.html_content

    def test_render_dsl_items_fallback_when_columns_empty(self):
        """When section has items but no columns, items are used as columns."""
        items_dsl = {
            "sections": [{
                "columns": [],
                "items": [
                    {"label": "Claim #", "field": "claimNumber", "format": "text"},
                    {"label": "Drug", "field": "productName", "format": "title"},
                ],
            }]
        }
        result = MyclaimsRenderingAgent().execute(
            "claim_list", _shape_a_tool_results(), {},
            render_dsl=items_dsl,
        )
        assert result.success is True
        assert result.render_format == "html_table"
        assert "RX12345" in result.html_content
        assert "Lisinopril 10Mg" in result.html_content

    def test_render_dsl_columns_correct_headers(self):
        """DSL column headers appear verbatim in the rendered HTML."""
        dsl = {
            "sections": [{
                "columns": [
                    {"header": "My Claim Number", "field": "claimNumber", "format": "text"},
                    {"header": "Medication Name", "field": "productName", "format": "title"},
                ],
            }]
        }
        result = MyclaimsRenderingAgent().execute(
            "claim_list", _shape_a_tool_results(), {},
            render_dsl=dsl,
        )
        assert result.success is True
        assert "My Claim Number" in result.html_content
        assert "Medication Name" in result.html_content

    def test_render_dsl_finds_deeply_nested_fields(self):
        """Fields nested inside sub-objects are found dynamically."""
        tool_results = {
            "tool_name": "test",
            "status": "success",
            "data": {
                "wrapper": {
                    "deep": {
                        "records": [
                            {"id": "A1", "info": {"name": "TestItem", "amount": "99.50"}}
                        ]
                    }
                }
            },
        }
        dsl = {
            "sections": [{
                "columns": [
                    {"header": "ID", "field": "id", "format": "text"},
                    {"header": "Name", "field": "name", "format": "text"},
                    {"header": "Amount", "field": "amount", "format": "currency"},
                ],
            }]
        }
        result = MyclaimsRenderingAgent().execute(
            "claim_list", tool_results, {}, render_dsl=dsl,
        )
        assert result.success is True
        assert "A1" in result.html_content
        assert "TestItem" in result.html_content
        assert "$99.50" in result.html_content

    def test_render_dsl_dot_notation_stripped_to_last_segment(self):
        """Dot-notation fields are stripped to last segment before lookup."""
        dsl = {
            "sections": [{
                "columns": [
                    {"header": "Claim Number", "field": "claimDetails.primary.claimNumber", "format": "text"},
                    {"header": "Drug", "field": "list_data.primary.drug.productName", "format": "title"},
                    {"header": "Patient Pay", "field": "submitted.pricing.patientPay", "format": "currency"},
                ],
            }]
        }
        result = MyclaimsRenderingAgent().execute(
            "claim_list", _shape_a_tool_results(), {},
            render_dsl=dsl,
        )
        assert result.success is True
        assert result.render_format == "html_table"
        assert "RX12345" in result.html_content
        assert "Lisinopril" in result.html_content
        assert "$5.00" in result.html_content


# ------------------------------------------------------------------ #
# Header-aware disambiguation                                          #
# ------------------------------------------------------------------ #

def _ambiguous_tool_results():
    """Mirrors the InsightIQ merged data with prescriber/beneficiary ambiguity."""
    return {
        "tool_name": "claim_details",
        "status": "success",
        "data": {
            "claimDetails": {
                "primary": {
                    "claimNumber": "260173639698000",
                    "lastNameFirstName": "WOODHOUSE,ROSEMARY",
                },
            },
            "list_data": {
                "primary": {
                    "number": "260173639698000",
                    "statusDescription": "Paid",
                    "prescriber": {
                        "id": "1639632797",
                        "firstName": "LIAM",
                        "lastName": "GENDIG",
                    },
                    "beneficiary": {
                        "memberId": "783CPSBB01",
                        "firstName": "ROSEMARY",
                        "lastName": "WOODHOUSE",
                    },
                    "pharmacy": {
                        "id": "0574017",
                        "name": "CVS PHARMACY 09176",
                    },
                },
            },
        },
    }


class TestHeaderDisambiguation:
    """Verify _find_field_value() respects context_hint from column headers."""

    def test_prescriber_lastname_with_hint_returns_prescriber(self):
        dsl = {"sections": [{"columns": [
            {"header": "Claim Number", "field": "claimNumber", "format": "text"},
            {"header": "Prescriber Last Name", "field": "lastName", "format": "text"},
            {"header": "Prescriber First Name", "field": "firstName", "format": "text"},
            {"header": "Prescriber NPI", "field": "id", "format": "text"},
        ]}]}
        result = MyclaimsRenderingAgent().execute(
            "prescriber_info", _ambiguous_tool_results(), {}, render_dsl=dsl,
        )
        assert result.success is True
        assert "GENDIG" in result.html_content
        assert "LIAM" in result.html_content
        assert "WOODHOUSE" not in result.html_content
        assert "1639632797" in result.html_content

    def test_member_lastname_with_hint_returns_member(self):
        dsl = {"sections": [{"columns": [
            {"header": "Status", "field": "statusDescription", "format": "text"},
            {"header": "Member Last Name", "field": "lastName", "format": "text"},
            {"header": "Member First Name", "field": "firstName", "format": "text"},
        ]}]}
        result = MyclaimsRenderingAgent().execute(
            "claim_list", _ambiguous_tool_results(), {}, render_dsl=dsl,
        )
        assert result.success is True
        assert "WOODHOUSE" in result.html_content
        assert "ROSEMARY" in result.html_content

    def test_pharmacy_id_with_hint_returns_pharmacy_id(self):
        dsl = {"sections": [{"columns": [
            {"header": "Pharmacy ID", "field": "id", "format": "text"},
            {"header": "Pharmacy Name", "field": "name", "format": "text"},
        ]}]}
        result = MyclaimsRenderingAgent().execute(
            "pharmacy_info", _ambiguous_tool_results(), {}, render_dsl=dsl,
        )
        assert result.success is True
        assert "0574017" in result.html_content
        assert "CVS PHARMACY 09176" in result.html_content

    def test_no_hint_still_returns_first_match(self):
        """Without a recognizable keyword in the header, first-match behavior preserved."""
        dsl = {"sections": [{"columns": [
            {"header": "Last Name", "field": "lastName", "format": "text"},
        ]}]}
        result = MyclaimsRenderingAgent().execute(
            "claim_list", _ambiguous_tool_results(), {}, render_dsl=dsl,
        )
        assert result.success is True
        assert ("GENDIG" in result.html_content) or ("WOODHOUSE" in result.html_content)

    def test_hint_fallback_when_subobject_lacks_field(self):
        """If the hinted sub-object doesn't have the field, fall through to normal search."""
        dsl = {"sections": [{"columns": [
            {"header": "Prescriber Claim Number", "field": "claimNumber", "format": "text"},
        ]}]}
        result = MyclaimsRenderingAgent().execute(
            "claim_list", _ambiguous_tool_results(), {}, render_dsl=dsl,
        )
        assert result.success is True
        assert "260173639698000" in result.html_content

    def test_context_hint_extraction(self):
        agent = MyclaimsRenderingAgent()
        assert agent._context_hint_from_header("Prescriber Last Name") == "prescriber"
        assert agent._context_hint_from_header("Member ID") == "beneficiary"
        assert agent._context_hint_from_header("Drug Name") == "drug"
        assert agent._context_hint_from_header("Pharmacy NPI") == "pharmacy"
        assert agent._context_hint_from_header("Claim Number") == ""
        assert agent._context_hint_from_header("Status") == ""


# ------------------------------------------------------------------ #
# Array traversal                                                       #
# ------------------------------------------------------------------ #

class TestArrayTraversal:
    """Verify _find_field_value() traverses list items (reject codes etc.)."""

    def test_reject_code_found_in_array(self):
        tool_results = {
            "tool_name": "claim_details", "status": "success",
            "data": {
                "claimDetails": {
                    "primary": {
                        "claimNumber": "260173639698000",
                        "drugLabelName": "ABILIFY MYCI TAB 10MG M",
                        "additionalDetails": {
                            "settlementCodes": {
                                "settlementCodesDetail": [
                                    {"responseRejectCode": "76",
                                     "settlementMessage": "PLAN LIMITATION EXCEEDED"},
                                    {"responseRejectCode": "87",
                                     "settlementMessage": "SECONDARY PAYER DENIED"},
                                ],
                            },
                        },
                    },
                },
            },
        }
        dsl = {"sections": [{"columns": [
            {"header": "Claim Number", "field": "claimNumber", "format": "text"},
            {"header": "Reject Code", "field": "responseRejectCode", "format": "reject_codes"},
            {"header": "Reject Reason", "field": "settlementMessage", "format": "text"},
        ]}]}
        result = MyclaimsRenderingAgent().execute(
            "rejection_reasons", tool_results, {}, render_dsl=dsl,
        )
        assert result.success is True
        assert "76" in result.html_content
        assert "87" in result.html_content
        assert "PLAN LIMITATION EXCEEDED" in result.html_content

    def test_multiple_reject_codes_joined(self):
        record = {
            "codes": [
                {"responseRejectCode": "76", "msg": "A"},
                {"responseRejectCode": "87", "msg": "B"},
            ],
        }
        result = MyclaimsRenderingAgent._find_field_value(record, "responseRejectCode")
        assert result is not None
        assert "76" in str(result)
        assert "87" in str(result)

    def test_single_item_array_no_pipe(self):
        record = {
            "items": [{"code": "42", "msg": "Single error"}],
        }
        result = MyclaimsRenderingAgent._find_field_value(record, "code")
        assert result == "42"
        assert "|" not in result

    def test_existing_dict_traversal_unaffected(self):
        """Dict-only traversal still works (no regression)."""
        record = {
            "claimInformation": {
                "claimNumber": "RX99",
                "fillDate": "20250101",
            },
        }
        assert MyclaimsRenderingAgent._find_field_value(record, "claimNumber") == "RX99"
        assert MyclaimsRenderingAgent._find_field_value(record, "fillDate") == "20250101"
        assert MyclaimsRenderingAgent._find_field_value(record, "missing") is None


# ------------------------------------------------------------------ #
# Deep hint search                                                      #
# ------------------------------------------------------------------ #

class TestDeepHintSearch:
    """Verify _find_hinted_container finds sub-objects at any depth."""

    def test_pharmacy_hint_finds_nested_pharmacy(self):
        """Pharmacy sub-object nested inside list_data.primary should be found."""
        record = {
            "data": {
                "claimDetails": {"primary": {"drugLabelName": "ABILIFY"}},
                "list_data": {
                    "primary": {
                        "pharmacy": {
                            "id": "0574017",
                            "name": "CVS PHARMACY 09176",
                            "city": "TULARE",
                        },
                        "prescriber": {
                            "id": "1639632797",
                            "name": "LIAM GENDIG",
                        },
                    },
                },
            },
        }
        result = MyclaimsRenderingAgent._find_field_value(
            record, "name", context_hint="pharmacy",
        )
        assert result == "CVS PHARMACY 09176"

    def test_pharmacy_id_hint_finds_pharmacy_not_prescriber(self):
        """'id' with pharmacy hint should return pharmacy ID, not prescriber ID."""
        record = {
            "data": {
                "list_data": {
                    "primary": {
                        "pharmacy": {"id": "0574017"},
                        "prescriber": {"id": "1639632797"},
                    },
                },
            },
        }
        result = MyclaimsRenderingAgent._find_field_value(
            record, "id", context_hint="pharmacy",
        )
        assert result == "0574017"

    def test_prescriber_hint_finds_nested_prescriber(self):
        record = {
            "data": {
                "list_data": {
                    "primary": {
                        "pharmacy": {"name": "CVS PHARMACY 09176"},
                        "prescriber": {"firstName": "LIAM", "lastName": "GENDIG"},
                        "beneficiary": {"firstName": "ROSEMARY", "lastName": "WOODHOUSE"},
                    },
                },
            },
        }
        result = MyclaimsRenderingAgent._find_field_value(
            record, "lastName", context_hint="prescriber",
        )
        assert result == "GENDIG"

    def test_hint_fallback_when_hinted_object_lacks_field(self):
        """If hinted sub-object doesn't have the field, Phase 2 finds it."""
        record = {
            "data": {
                "list_data": {
                    "primary": {
                        "pharmacy": {"id": "0574017"},
                    },
                },
            },
            "claimNumber": "260173639698000",
        }
        result = MyclaimsRenderingAgent._find_field_value(
            record, "claimNumber", context_hint="pharmacy",
        )
        assert result == "260173639698000"

    def test_pharmacy_full_table_renders_correctly(self):
        """End-to-end: pharmacy DSL with bare 'name'/'id' renders pharmacy data."""
        tool_results = {
            "tool_name": "claim_details", "status": "success",
            "data": {
                "claimDetails": {
                    "primary": {
                        "claimNumber": "260173639698000",
                        "sequenceNumber": "001",
                        "drugLabelName": "ABILIFY MYCI TAB 10MG M",
                    },
                },
                "list_data": {
                    "primary": {
                        "number": "260173639698000",
                        "pharmacy": {
                            "id": "0574017",
                            "name": "CVS PHARMACY 09176",
                            "city": "TULARE",
                            "state": "CA",
                            "zip": "93274",
                        },
                        "prescriber": {
                            "id": "1639632797",
                            "firstName": "LIAM",
                            "lastName": "GENDIG",
                        },
                    },
                },
            },
        }
        dsl = {"sections": [{"columns": [
            {"header": "Claim Number", "field": "claimNumber", "format": "text"},
            {"header": "Pharmacy Name", "field": "name", "format": "text"},
            {"header": "Pharmacy ID", "field": "id", "format": "text"},
            {"header": "Pharmacy City", "field": "city", "format": "text"},
        ]}]}
        result = MyclaimsRenderingAgent().execute(
            "pharmacy_info", tool_results, {}, render_dsl=dsl,
        )
        assert result.success is True
        assert "CVS PHARMACY 09176" in result.html_content
        assert "0574017" in result.html_content
        assert "TULARE" in result.html_content
        assert "ABILIFY" not in result.html_content
        assert "1639632797" not in result.html_content


# ---------------------------------------------------------------------------
# Change 4 — suppress_table
# ---------------------------------------------------------------------------

class TestSuppressTable:
    """When render_dsl contains suppress_table: true, the engine returns text-only."""

    def test_suppress_table_returns_text_format(self, agent):
        # Use a Tier-3 intent (not in MUST_RENDER or ALWAYS_CARD) so suppress_table
        # backward-compat fires. MUST_RENDER intents (e.g. copay_info) override suppress_table.
        dsl = {
            "layout": "table",
            "suppress_table": True,
            "title": "Claim Status",
            "sections": [{"id": "s1", "type": "table", "columns": [
                {"header": "Claim #", "field": "claimNumber", "format": "text"},
            ]}],
        }
        result = agent.execute("claim_status", _shape_a_tool_results(), {}, render_dsl=dsl)
        assert result.render_format != "html_table"
        assert result.html_content == ""

    def test_suppress_table_false_renders_normally(self, agent):
        dsl = _shape_a_render_dsl()
        dsl["suppress_table"] = False
        result = agent.execute("claim_list", _shape_a_tool_results(), {}, render_dsl=dsl)
        assert result.render_format == "html_table"
        assert result.success is True
        assert "RX12345" in result.html_content

    def test_suppress_table_missing_renders_normally(self, agent):
        dsl = _shape_a_render_dsl()
        result = agent.execute("claim_list", _shape_a_tool_results(), {}, render_dsl=dsl)
        assert result.render_format == "html_table"
        assert result.success is True


# ---------------------------------------------------------------------------
# Change 2 — multi-section support
# ---------------------------------------------------------------------------

def _multi_section_tool_results():
    return {
        "tool_name": "claim_details",
        "status": "success",
        "data": {
            "claimDetails": {
                "primary": {
                    "claimNumber": "RX99999",
                    "claimStatusDescription": "Paid",
                    "drugLabelName": "METFORMIN 500MG",
                    "fillDate": "20250301",
                    "clientPatientPayAmount": "25.00",
                    "clientTotalAmount": "150.00",
                },
            },
            "list_data": {
                "primary": {
                    "pharmacy": {
                        "name": "CVS PHARMACY 09176",
                        "id": "0574017",
                        "city": "TULARE",
                        "state": "CA",
                    },
                    "prescriber": {
                        "firstName": "JOHN",
                        "lastName": "DOE",
                        "id": "1234567890",
                    },
                },
            },
        },
    }


class TestMultiSection:
    """Tests that multiple DSL sections render stacked tables."""

    def test_two_sections_both_render(self, agent):
        dsl = {
            "layout": "table",
            "title": "Claim Overview",
            "sections": [
                {
                    "id": "claim_info",
                    "type": "table",
                    "title": "Claim Information",
                    "columns": [
                        {"header": "Claim Number", "field": "claimNumber", "format": "text"},
                        {"header": "Status", "field": "claimStatusDescription", "format": "status_badge"},
                        {"header": "Drug Name", "field": "drugLabelName", "format": "title"},
                    ],
                },
                {
                    "id": "pharmacy_info",
                    "type": "table",
                    "title": "Pharmacy Details",
                    "columns": [
                        {"header": "Pharmacy Name", "field": "name", "format": "text"},
                        {"header": "Pharmacy ID", "field": "id", "format": "text"},
                        {"header": "Pharmacy City", "field": "city", "format": "text"},
                    ],
                },
            ],
        }
        result = agent.execute(
            "claim_details", _multi_section_tool_results(), {}, render_dsl=dsl,
        )
        assert result.success is True
        assert result.render_format == "html_table"
        assert "RX99999" in result.html_content
        assert "CVS PHARMACY 09176" in result.html_content

    def test_single_section_uses_intent_title(self, agent):
        """Single-section DSL should use intent-based title, not section title."""
        dsl = {
            "layout": "table",
            "title": "Claims Data",
            "sections": [{
                "id": "s1",
                "type": "table",
                "title": "Some Section Title",
                "columns": [
                    {"header": "Claim #", "field": "claimNumber", "format": "text"},
                    {"header": "Status", "field": "claimStatusDescription", "format": "status_badge"},
                ],
            }],
        }
        result = agent.execute(
            "claim_status", _multi_section_tool_results(), {}, render_dsl=dsl,
        )
        assert result.success is True
        assert "Claim Status" in result.answer_header

    def test_section_with_no_matching_data_is_skipped(self, agent):
        dsl = {
            "layout": "table",
            "title": "Mixed",
            "sections": [
                {
                    "id": "good",
                    "type": "table",
                    "title": "Good Section",
                    "columns": [
                        {"header": "Claim Number", "field": "claimNumber", "format": "text"},
                    ],
                },
                {
                    "id": "bad",
                    "type": "table",
                    "title": "Bad Section",
                    "columns": [
                        {"header": "Nonexistent", "field": "totallyFakeField", "format": "text"},
                    ],
                },
            ],
        }
        result = agent.execute(
            "claim_details", _multi_section_tool_results(), {}, render_dsl=dsl,
        )
        assert result.success is True
        assert "RX99999" in result.html_content


# ---------------------------------------------------------------------------
# Change 3 — pivot layout
# ---------------------------------------------------------------------------

def _pivot_tool_results():
    return {
        "tool_name": "claim_details",
        "status": "success",
        "data": {
            "claimDetails": {
                "primary": {
                    "claimNumber": "260173639698000",
                    "drugLabelName": "ABILIFY MYCI TAB 10MG M",
                    "clientIngredientCost": "1631.52",
                    "clientDispensingFee": "0.50",
                    "clientPatientPayAmount": "50.00",
                    "clientTotalAmount": "1582.02",
                    "clientIngredientCost2": "0.00",
                    "clientDispensingFee2": "0.00",
                    "clientPatientPayAmount2": "0.00",
                    "clientTotalAmount2": "0.00",
                    "responseIngredCostPaid3": "1631.52",
                    "responseDispensingFeeP3": "0.50",
                    "responsePatientPayAmount3": "50.00",
                    "responseTotalAmountPaid3": "1582.02",
                },
            },
        },
    }


def _pivot_dsl():
    return {
        "layout": "pivot",
        "title": "Pricing Breakdown",
        "sections": [{
            "id": "pricing",
            "type": "table",
            "identifier_columns": [
                {"header": "Claim Number", "field": "claimNumber", "format": "text"},
                {"header": "Drug Name", "field": "drugLabelName", "format": "title"},
            ],
            "groups": [
                {
                    "label": "Ingredient Cost",
                    "fields": {
                        "Primary": {"field": "clientIngredientCost", "format": "currency"},
                        "Secondary": {"field": "clientIngredientCost2", "format": "currency"},
                        "Final": {"field": "responseIngredCostPaid3", "format": "currency"},
                    },
                },
                {
                    "label": "Dispensing Fee",
                    "fields": {
                        "Primary": {"field": "clientDispensingFee", "format": "currency"},
                        "Secondary": {"field": "clientDispensingFee2", "format": "currency"},
                        "Final": {"field": "responseDispensingFeeP3", "format": "currency"},
                    },
                },
                {
                    "label": "Patient Pay",
                    "fields": {
                        "Primary": {"field": "clientPatientPayAmount", "format": "currency"},
                        "Secondary": {"field": "clientPatientPayAmount2", "format": "currency"},
                        "Final": {"field": "responsePatientPayAmount3", "format": "currency"},
                    },
                },
                {
                    "label": "Total Paid",
                    "fields": {
                        "Primary": {"field": "clientTotalAmount", "format": "currency"},
                        "Secondary": {"field": "clientTotalAmount2", "format": "currency"},
                        "Final": {"field": "responseTotalAmountPaid3", "format": "currency"},
                    },
                },
            ],
        }],
    }


class TestPivotLayout:
    """Tests for the pivot (categories-as-rows) layout."""

    def test_pivot_renders_successfully(self, agent):
        result = agent.execute(
            "pricing_info", _pivot_tool_results(), {}, render_dsl=_pivot_dsl(),
        )
        assert result.success is True
        assert result.render_format == "html_table"

    def test_pivot_has_component_column(self, agent):
        result = agent.execute(
            "pricing_info", _pivot_tool_results(), {}, render_dsl=_pivot_dsl(),
        )
        assert "Component" in result.html_content

    def test_pivot_has_group_labels_as_rows(self, agent):
        result = agent.execute(
            "pricing_info", _pivot_tool_results(), {}, render_dsl=_pivot_dsl(),
        )
        assert "Ingredient Cost" in result.html_content
        assert "Dispensing Fee" in result.html_content
        assert "Patient Pay" in result.html_content
        assert "Total Paid" in result.html_content

    def test_pivot_has_value_columns(self, agent):
        result = agent.execute(
            "pricing_info", _pivot_tool_results(), {}, render_dsl=_pivot_dsl(),
        )
        assert "Primary" in result.html_content
        assert "Secondary" in result.html_content
        assert "Final" in result.html_content

    def test_pivot_extracts_real_values(self, agent):
        result = agent.execute(
            "pricing_info", _pivot_tool_results(), {}, render_dsl=_pivot_dsl(),
        )
        assert "$1,631.52" in result.html_content
        assert "$50.00" in result.html_content
        assert "$0.50" in result.html_content

    def test_pivot_includes_identifier_columns(self, agent):
        result = agent.execute(
            "pricing_info", _pivot_tool_results(), {}, render_dsl=_pivot_dsl(),
        )
        assert "260173639698000" in result.html_content

    def test_pivot_row_count(self, agent):
        """4 groups = 4 rows in the answer header."""
        result = agent.execute(
            "pricing_info", _pivot_tool_results(), {}, render_dsl=_pivot_dsl(),
        )
        assert "4 records" in result.answer_header

    def test_pivot_empty_groups_returns_no_data(self, agent):
        dsl = {
            "layout": "pivot",
            "title": "Empty",
            "sections": [{"id": "s1", "type": "table", "groups": []}],
        }
        result = agent.execute(
            "pricing_info", _pivot_tool_results(), {}, render_dsl=dsl,
        )
        assert "No data" in result.answer_header

    def test_pivot_no_sections_returns_no_data(self, agent):
        dsl = {"layout": "pivot", "title": "No Sections", "sections": []}
        result = agent.execute(
            "pricing_info", _pivot_tool_results(), {}, render_dsl=dsl,
        )
        assert "No data" in result.answer_header


# ---------------------------------------------------------------------------
# render_dsl.py — pivot dataclass tests
# ---------------------------------------------------------------------------

class TestRenderDSLPivot:
    """Tests for RenderPivotGroup / RenderPivotField in render_dsl.py."""

    def test_pivot_group_from_dict(self):
        from agents.post_processing.render_dsl import RenderPivotGroup
        d = {
            "label": "Ingredient Cost",
            "fields": {
                "Primary": {"field": "clientIngredientCost", "format": "currency"},
                "Secondary": {"field": "clientIngredientCost2", "format": "currency"},
            },
        }
        group = RenderPivotGroup.from_dict(d)
        assert group.label == "Ingredient Cost"
        assert len(group.fields) == 2
        assert group.fields["Primary"].field == "clientIngredientCost"
        assert group.fields["Primary"].format == "currency"

    def test_pivot_group_to_dict_roundtrip(self):
        from agents.post_processing.render_dsl import RenderPivotGroup
        d = {
            "label": "Dispensing Fee",
            "fields": {
                "Primary": {"field": "clientDispensingFee", "format": "currency"},
            },
        }
        group = RenderPivotGroup.from_dict(d)
        out = group.to_dict()
        assert out["label"] == "Dispensing Fee"
        assert out["fields"]["Primary"]["field"] == "clientDispensingFee"

    def test_pivot_field_invalid_format_defaults(self):
        from agents.post_processing.render_dsl import RenderPivotField
        pf = RenderPivotField.from_dict({"field": "foo", "format": "INVALID"})
        assert pf.format == "text"

    def test_section_with_groups_roundtrip(self):
        from agents.post_processing.render_dsl import RenderSection
        d = {
            "id": "pricing",
            "type": "table",
            "groups": [
                {"label": "Cost", "fields": {"A": {"field": "f1", "format": "currency"}}},
            ],
            "identifier_columns": [
                {"header": "Claim #", "field": "claimNumber", "format": "text"},
            ],
        }
        section = RenderSection.from_dict(d)
        assert len(section.groups) == 1
        assert len(section.identifier_columns) == 1
        out = section.to_dict()
        assert "groups" in out
        assert "identifier_columns" in out


# ---------------------------------------------------------------------------
# Pivot fallback — layout:"table" but groups, no columns
# ---------------------------------------------------------------------------

class TestPivotFallback:
    """When LLM emits layout:'table' but section has groups and no columns,
    the engine should fall through to _extract_pivot() instead of failing."""

    def test_table_layout_with_groups_falls_back_to_pivot(self, agent):
        dsl = {
            "layout": "table",
            "title": "Pricing Breakdown",
            "sections": [{
                "id": "pricing",
                "type": "table",
                "identifier_columns": [
                    {"header": "Claim Number", "field": "claimNumber", "format": "text"},
                ],
                "groups": [
                    {
                        "label": "Ingredient Cost",
                        "fields": {
                            "Primary": {"field": "clientIngredientCost", "format": "currency"},
                            "Final": {"field": "responseIngredCostPaid3", "format": "currency"},
                        },
                    },
                    {
                        "label": "Patient Pay",
                        "fields": {
                            "Primary": {"field": "clientPatientPayAmount", "format": "currency"},
                            "Final": {"field": "responsePatientPayAmount3", "format": "currency"},
                        },
                    },
                ],
            }],
        }
        result = agent.execute(
            "pricing_info", _pivot_tool_results(), {}, render_dsl=dsl,
        )
        assert result.success is True
        assert result.render_format == "html_table"
        assert "Ingredient Cost" in result.html_content
        assert "Patient Pay" in result.html_content
        assert "260173639698000" in result.html_content


# ---------------------------------------------------------------------------
# Blocked fields — description43Name stripped from DSL
# ---------------------------------------------------------------------------

class TestBlockedFields:
    """Columns with blocked field names are stripped before rendering."""

    def test_description43Name_stripped(self, agent):
        tool_results = {
            "tool_name": "claim_details", "status": "success",
            "data": {
                "claimDetails": {
                    "primary": {
                        "claimNumber": "RX99999",
                        "drugLabelName": "ABILIFY MYCI TAB 10MG M",
                        "submittedQuantityDispensed": "30.0",
                        "submittedDaysSupply": "30",
                    },
                },
            },
        }
        dsl = {
            "layout": "table",
            "title": "Drug Details",
            "sections": [{
                "id": "drug",
                "type": "table",
                "columns": [
                    {"header": "Drug Name", "field": "drugLabelName", "format": "title"},
                    {"header": "Quantity", "field": "submittedQuantityDispensed", "format": "text"},
                    {"header": "Strength", "field": "description43Name", "format": "text"},
                    {"header": "Dosage Form", "field": "description43Name", "format": "text"},
                ],
            }],
        }
        result = agent.execute("drug_info", tool_results, {}, render_dsl=dsl)
        assert result.success is True
        assert "Abilify Myci Tab 10Mg M" in result.html_content
        assert "Strength" not in result.html_content
        assert "Dosage Form" not in result.html_content

    def test_all_blocked_returns_no_data(self, agent):
        tool_results = {"tool_name": "x", "status": "success", "data": {}}
        dsl = {
            "layout": "table",
            "title": "Bad",
            "sections": [{
                "id": "s1",
                "type": "table",
                "columns": [
                    {"header": "Strength", "field": "description43Name", "format": "text"},
                ],
            }],
        }
        result = agent.execute("drug_info", tool_results, {}, render_dsl=dsl)
        assert "No data" in result.answer_header
