"""
Tests for Claims API Orchestrator

Run with: pytest tests/test_claims_api.py -v
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, Any
import asyncio

from tools.claims_api import (
    get_claim_details,
    get_claim_list,
    combine_claim_details_and_list,
    call_claims_tool_node,
    normalize_entities,
    match_api,
    call_external_api,
    ENTITY_MAP
)
from core.node_models import ToolResult, ToolExecutionStatus, API_REPOSITORY
from core.errors.models import AgentError, ErrorCode
from core.errors.exceptions import ExternalAPIError, ToolTimeoutError
from core.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_claim_details():
    """Sample claim details response"""
    return {
        "status": "success",
        "header": {
            "xcorrelationid": "test-123",
            "xconsumerAppName": "TEST-APP"
        },
        "claimDetails": {
            "primary": {
                "medD": {
                    "claimStatus": "R",
                    "accountId": "A-123"
                }
            },
            "linkedClaim": {
                "stcob": {
                    "claimNumber": "123456789",
                    "claimSequence": "1"
                }
            }
        }
    }


@pytest.fixture
def sample_claim_list():
    """Sample claim list response"""
    return {
        "claims": [
            {
                "claimInformation": {
                    "claimNumber": "123456789",
                    "claimSequenceNumber": "1",
                    "claimStatus": "P"
                },
                "member": {
                    "memberId": "M123",
                    "firstName": "John",
                    "lastName": "Doe"
                },
                "drug": {
                    "productName": "Aspirin",
                    "gpi": "12345"
                }
            }
        ]
    }


@pytest.fixture
def mock_api_definition():
    """Mock API definition"""
    return API_REPOSITORY(
        name="get_claim_details",
        endpoint="/myclaims/claims/v1/details",
        method="POST",
        required_entities=["claimNumber", "claimSequence"],
        intent_keywords=["details", "claim details"],
        description="Fetch claim details",
        body_template=lambda e: {
            "claimDetailsRequest": {
                "claimNumber": e["claimNumber"],
                "claimSequence": e["claimSequence"]
            }
        }
    )


@pytest.fixture
def sample_entities():
    """Sample normalized entities"""
    return {
        "claimNumber": "123456789",
        "claimSequence": "1"
    }


# ============================================================================
# TEST: get_claim_details()
# ============================================================================

class TestGetClaimDetails:
    """Tests for get_claim_details function"""
    
    @patch('tools.claims_api.get_api_repository')
    @patch('tools.claims_api.call_external_api')
    def test_success_scenario(self, mock_call_api, mock_get_repo, mock_api_definition, sample_claim_details):
        """Test successful claim details fetch"""
        logger.info("Testing get_claim_details success scenario")
        
        # Setup mocks
        mock_get_repo.return_value = [mock_api_definition]
        mock_call_api.return_value = sample_claim_details
        
        # Execute
        result = get_claim_details("123456789", "1")
        
        # Assertions
        assert result == sample_claim_details
        assert result["status"] == "success"
        assert "claimDetails" in result
        mock_call_api.assert_called_once()
        
    @patch('tools.claims_api.get_api_repository')
    @patch('tools.claims_api.call_external_api')
    @patch('tools.claims_api.get_fallback_details')
    def test_fallback_on_api_failure(self, mock_fallback, mock_call_api, mock_get_repo, 
                                     mock_api_definition, sample_claim_details):
        """Test fallback when API fails"""
        logger.info("Testing get_claim_details fallback on API failure")
        
        # Setup mocks
        mock_get_repo.return_value = [mock_api_definition]
        mock_call_api.side_effect = ExternalAPIError("API failed", details={}, retriable=False)
        mock_fallback.return_value = sample_claim_details
        
        # Execute
        result = get_claim_details("123456789", "1")
        
        # Assertions
        assert result == sample_claim_details
        mock_fallback.assert_called_once_with("123456789", "1")
        
    @patch('tools.claims_api.get_api_repository')
    @patch('tools.claims_api.get_fallback_details')
    def test_fallback_on_missing_api_definition(self, mock_fallback, mock_get_repo, sample_claim_details):
        """Test fallback when API definition not found"""
        logger.info("Testing get_claim_details fallback on missing API definition")
        
        # Setup mocks - return empty list (no API found)
        mock_get_repo.return_value = []
        mock_fallback.return_value = sample_claim_details
        
        # Execute
        result = get_claim_details("123456789", "1")
        
        # Assertions
        assert result == sample_claim_details
        mock_fallback.assert_called_once()


# ============================================================================
# TEST: get_claim_list()
# ============================================================================

class TestGetClaimList:
    """Tests for get_claim_list function"""
    
    @patch('tools.claims_api.get_api_repository')
    @patch('tools.claims_api.call_external_api')
    def test_success_scenario(self, mock_call_api, mock_get_repo, sample_claim_list):
        """Test successful claim list fetch"""
        logger.info("Testing get_claim_list success scenario")
        
        # Setup mocks
        mock_api = Mock()
        mock_api.name = "get_claim_list"
        mock_api.body_template = lambda e: {"claimsRequest": {"claimId": e["claimId"]}}
        mock_get_repo.return_value = [mock_api]
        mock_call_api.return_value = sample_claim_list
        
        # Execute
        result = get_claim_list("123456789", "1")
        
        # Assertions
        assert result == sample_claim_list
        assert "claims" in result
        assert len(result["claims"]) > 0
        mock_call_api.assert_called_once()
        
    @patch('tools.claims_api.get_api_repository')
    @patch('tools.claims_api.call_external_api')
    @patch('tools.claims_api.get_fallback_list')
    def test_fallback_on_timeout(self, mock_fallback, mock_call_api, mock_get_repo, sample_claim_list):
        """Test fallback when API times out"""
        logger.info("Testing get_claim_list fallback on timeout")
        
        # Setup mocks
        mock_api = Mock()
        mock_api.name = "get_claim_list"
        mock_api.body_template = lambda e: {"claimsRequest": {"claimId": e["claimId"]}}
        mock_get_repo.return_value = [mock_api]
        mock_call_api.side_effect = ToolTimeoutError("Timeout", details={}, retriable=True)
        mock_fallback.return_value = sample_claim_list
        
        # Execute
        result = get_claim_list("123456789", "1")
        
        # Assertions
        assert result == sample_claim_list
        mock_fallback.assert_called_once_with("123456789", "1")
        
    def test_default_claim_sequence(self):
        """Test that claimSequence defaults to '1'"""
        logger.info("Testing get_claim_list default claimSequence")
        
        with patch('tools.claims_api.get_api_repository') as mock_repo, \
             patch('tools.claims_api.call_external_api') as mock_call:
            
            mock_api = Mock()
            mock_api.name = "get_claim_list"
            mock_api.body_template = lambda e: {"claimsRequest": {"claimId": e["claimId"]}}
            mock_repo.return_value = [mock_api]
            mock_call.return_value = {"claims": []}
            
            # Call without claimSequence
            result = get_claim_list("123456789")
            
            # Should still work with default
            assert result == {"claims": []}


# ============================================================================
# TEST: combine_claim_details_and_list()
# ============================================================================

class TestCombineClaimDetailsAndList:
    """Tests for combine_claim_details_and_list function"""
    
    @patch('tools.claims_api.get_claim_details')
    @patch('tools.claims_api.get_claim_list')
    def test_successful_match_and_merge(self, mock_get_list, mock_get_details, 
                                        sample_claim_details, sample_claim_list):
        """Test successful matching and merging"""
        logger.info("Testing combine_claim_details_and_list with successful match")
        
        # Setup mocks
        mock_get_details.return_value = sample_claim_details
        mock_get_list.return_value = sample_claim_list
        
        # Execute
        result = combine_claim_details_and_list("123456789", "1")
        
        # Assertions
        assert "list_data" in result
        assert result["list_data"]["claimInformation"]["claimNumber"] == "123456789"
        assert result["status"] == "success"
        assert result["claimDetails"] == sample_claim_details["claimDetails"]
        
    @patch('tools.claims_api.get_claim_details')
    @patch('tools.claims_api.get_claim_list')
    def test_no_match_in_list(self, mock_get_list, mock_get_details, sample_claim_details):
        """Test when no matching claim found in list"""
        logger.info("Testing combine_claim_details_and_list with no match")
        
        # Setup mocks - list has different claim number
        mock_get_details.return_value = sample_claim_details
        mock_get_list.return_value = {
            "claims": [
                {
                    "claimInformation": {
                        "claimNumber": "999999999",  # Different claim number
                        "claimSequenceNumber": "1"
                    }
                }
            ]
        }
        
        # Execute
        result = combine_claim_details_and_list("123456789", "1")
        
        # Assertions
        assert "list_data" not in result  # No list_data added when no match
        assert result["status"] == "success"
        
    @patch('tools.claims_api.get_claim_details')
    @patch('tools.claims_api.get_claim_list')
    def test_empty_claim_list(self, mock_get_list, mock_get_details, sample_claim_details):
        """Test when claim list is empty"""
        logger.info("Testing combine_claim_details_and_list with empty list")
        
        # Setup mocks
        mock_get_details.return_value = sample_claim_details
        mock_get_list.return_value = {"claims": []}
        
        # Execute
        result = combine_claim_details_and_list("123456789", "1")
        
        # Assertions
        assert "list_data" not in result
        assert result["status"] == "success"
        
    @patch('tools.claims_api.get_claim_details')
    @patch('tools.claims_api.get_claim_list')
    def test_handle_different_response_structures(self, mock_get_list, mock_get_details, sample_claim_details):
        """Test handling various response structures"""
        logger.info("Testing combine_claim_details_and_list with different structures")
        
        # Setup mocks - list as direct array
        mock_get_details.return_value = sample_claim_details
        mock_get_list.return_value = [
            {
                "claimInformation": {
                    "claimNumber": "123456789",
                    "claimSequenceNumber": "1"
                }
            }
        ]
        
        # Execute
        result = combine_claim_details_and_list("123456789", "1")
        
        # Assertions
        assert "list_data" in result


# ============================================================================
# TEST: normalize_entities()
# ============================================================================

class TestNormalizeEntities:
    """Tests for normalize_entities function"""
    
    def test_normalize_dict_entities(self):
        """Test normalizing entities from dict"""
        logger.info("Testing normalize_entities with dict input")
        
        entities_dict = {
            "claim_number": "123456789",
            "claim_sequence": "1",
            "member_id": "M123"
        }
        
        result = normalize_entities(entities_dict)
        
        # Assertions
        assert result["claimNumber"] == "123456789"
        assert result["claimSequence"] == "1"
        assert result["memberId"] == "M123"
        assert "claim_number" not in result  # Old keys removed
        
    def test_normalize_with_raw_entities(self):
        """Test normalizing entities with raw_entities"""
        logger.info("Testing normalize_entities with raw_entities")
        
        entities_dict = {
            "claim_number": "123456789",
            "raw_entities": {
                "claim_sequence": "1"
            }
        }
        
        result = normalize_entities(entities_dict)
        
        # Assertions
        assert result["claimNumber"] == "123456789"
        assert result["claimSequence"] == "1"
        
    def test_normalize_unknown_keys(self):
        """Test that unknown keys pass through"""
        logger.info("Testing normalize_entities with unknown keys")
        
        entities_dict = {
            "claim_number": "123456789",
            "customField": "customValue"
        }
        
        result = normalize_entities(entities_dict)
        
        # Assertions
        assert result["claimNumber"] == "123456789"
        assert result["customField"] == "customValue"  # Unknown key passes through
        
    def test_entity_map_completeness(self):
        """Test that ENTITY_MAP has all expected mappings"""
        logger.info("Testing ENTITY_MAP completeness")
        
        expected_keys = [
            "claim_number", "member_id", "prescription_number",
            "medication_name", "date_from", "date_to",
            "claim_sequence", "claim_id"
        ]
        
        for key in expected_keys:
            assert key in ENTITY_MAP, f"Missing {key} in ENTITY_MAP"


# ============================================================================
# TEST: match_api()
# ============================================================================

class TestMatchApi:
    """Tests for match_api function"""
    
    def test_match_with_all_entities(self):
        """Test API matching with all required entities present"""
        logger.info("Testing match_api with all required entities")
        
        with patch('tools.claims_api.get_api_repository') as mock_repo:
            # Setup mock APIs
            api1 = Mock()
            api1.required_entities = ["claimNumber", "claimSequence"]
            api1.intent_keywords = ["details", "claim details"]
            
            api2 = Mock()
            api2.required_entities = ["claimId"]
            api2.intent_keywords = ["list", "search"]
            
            mock_repo.return_value = [api1, api2]
            
            # Test matching
            entities = {"claimNumber": "123", "claimSequence": "1"}
            intent = "get claim details"
            
            result = match_api(intent, entities)
            
            # Should match api1
            assert result == api1
            
    def test_match_missing_required_entity(self):
        """Test API matching when required entity is missing"""
        logger.info("Testing match_api with missing required entity")
        
        with patch('tools.claims_api.get_api_repository') as mock_repo:
            api1 = Mock()
            api1.required_entities = ["claimNumber", "claimSequence"]
            api1.intent_keywords = ["details"]
            
            mock_repo.return_value = [api1]
            
            # Missing claimSequence
            entities = {"claimNumber": "123"}
            intent = "get claim details"
            
            result = match_api(intent, entities)
            
            # Should not match
            assert result is None
            
    def test_match_by_keyword_score(self):
        """Test API matching based on keyword scoring"""
        logger.info("Testing match_api keyword scoring")
        
        with patch('tools.claims_api.get_api_repository') as mock_repo:
            # API with more keyword matches should win
            api1 = Mock()
            api1.required_entities = []
            api1.intent_keywords = ["claim"]
            
            api2 = Mock()
            api2.required_entities = []
            api2.intent_keywords = ["claim", "status"]
            
            mock_repo.return_value = [api1, api2]
            
            entities = {}
            intent = "claim status"
            
            result = match_api(intent, entities)
            
            # Should match api2 (has both keywords)
            assert result == api2


# ============================================================================
# TEST: call_claims_tool_node() - Main Orchestrator
# ============================================================================

class TestCallClaimsToolNode:
    """Tests for call_claims_tool_node async function"""
    
    @pytest.mark.asyncio
    async def test_claim_details_intent_success(self, sample_claim_details, sample_claim_list):
        """Test call_claims_tool_node with claim_details intent"""
        logger.info("Testing call_claims_tool_node with claim_details intent")
        
        with patch('tools.claims_api.combine_claim_details_and_list') as mock_combine, \
             patch('tools.claims_api.extract_logging_context') as mock_ctx:
            
            mock_ctx.return_value = {}
            mock_combine.return_value = {
                **sample_claim_details,
                "list_data": sample_claim_list["claims"][0]
            }
            
            state = {
                "intent": "claim_details",
                "entities": {
                    "claimNumber": "123456789",
                    "claimSequence": "1"
                }
            }
            
            result = await call_claims_tool_node(state)
            
            # Assertions
            assert "tool_results" in result
            tool_results = result["tool_results"]
            assert tool_results["tool_name"] == "claim_details_enriched"
            assert tool_results["status"] == "success"
            assert "list_data" in tool_results["data"]
            
    @pytest.mark.asyncio
    async def test_missing_intent(self):
        """Test call_claims_tool_node with missing intent"""
        logger.info("Testing call_claims_tool_node with missing intent")
        
        with patch('tools.claims_api.extract_logging_context') as mock_ctx:
            mock_ctx.return_value = {}
            
            state = {
                "entities": {"claimNumber": "123"}
            }
            
            result = await call_claims_tool_node(state)
            
            # Assertions
            assert "tool_results" in result
            tool_results = result["tool_results"]
            assert tool_results["status"] == "failure"
            assert "intent" in tool_results["error_message"].lower()
            
    @pytest.mark.asyncio
    async def test_missing_entities(self):
        """Test call_claims_tool_node with missing entities"""
        logger.info("Testing call_claims_tool_node with missing entities")
        
        with patch('tools.claims_api.extract_logging_context') as mock_ctx:
            mock_ctx.return_value = {}
            
            state = {
                "intent": "claim_details"
            }
            
            result = await call_claims_tool_node(state)
            
            # Assertions
            assert "tool_results" in result
            tool_results = result["tool_results"]
            assert tool_results["status"] == "failure"
            assert "entities" in tool_results["error_message"].lower()
            
    @pytest.mark.asyncio
    async def test_claim_details_missing_required_entities(self):
        """Test claim_details intent with missing required entities"""
        logger.info("Testing claim_details with missing required entities")
        
        with patch('tools.claims_api.extract_logging_context') as mock_ctx, \
             patch('tools.claims_api.normalize_entities') as mock_normalize:
            
            mock_ctx.return_value = {}
            mock_normalize.return_value = {"claimNumber": "123"}  # Missing claimSequence
            
            state = {
                "intent": "claim_details",
                "entities": {"claimNumber": "123"}
            }
            
            result = await call_claims_tool_node(state)
            
            # Assertions
            assert "tool_results" in result
            tool_results = result["tool_results"]
            assert tool_results["status"] == "failure"
            
    @pytest.mark.asyncio
    async def test_standard_flow_with_matching(self, sample_claim_details):
        """Test standard flow for non-claim_details intents"""
        logger.info("Testing standard flow with API matching")
        
        with patch('tools.claims_api.extract_logging_context') as mock_ctx, \
             patch('tools.claims_api.normalize_entities') as mock_normalize, \
             patch('tools.claims_api.match_api') as mock_match, \
             patch('tools.claims_api.call_external_api') as mock_call:
            
            mock_ctx.return_value = {}
            mock_normalize.return_value = {"claimNumber": "123", "claimSequence": "1"}
            
            # Mock matched API
            mock_api = Mock()
            mock_api.name = "get_claim_details"
            mock_api.body_template = lambda e: {"request": e}
            mock_api.full_url = "http://test.com/api"
            mock_match.return_value = mock_api
            
            mock_call.return_value = sample_claim_details
            
            state = {
                "intent": "other_intent",
                "entities": {"claimNumber": "123", "claimSequence": "1"}
            }
            
            result = await call_claims_tool_node(state)
            
            # Assertions
            assert "tool_results" in result
            tool_results = result["tool_results"]
            assert tool_results["status"] == "success"
            assert tool_results["tool_name"] == "get_claim_details"
            
    @pytest.mark.asyncio
    async def test_no_matching_api_found(self):
        """Test when no matching API is found"""
        logger.info("Testing when no matching API found")
        
        with patch('tools.claims_api.extract_logging_context') as mock_ctx, \
             patch('tools.claims_api.normalize_entities') as mock_normalize, \
             patch('tools.claims_api.match_api') as mock_match:
            
            mock_ctx.return_value = {}
            mock_normalize.return_value = {"someEntity": "value"}
            mock_match.return_value = None  # No match
            
            state = {
                "intent": "unknown_intent",
                "entities": {"someEntity": "value"}
            }
            
            result = await call_claims_tool_node(state)
            
            # Assertions
            assert "tool_results" in result
            tool_results = result["tool_results"]
            assert tool_results["status"] == "failure"
            assert "no matching api" in tool_results["error_message"].lower()


# ============================================================================
# TEST: call_external_api() - HTTP Client
# ============================================================================

class TestCallExternalApi:
    """Tests for call_external_api function with retry logic"""
    
    @patch('tools.claims_api.requests.request')
    def test_successful_api_call(self, mock_request, mock_api_definition):
        """Test successful external API call"""
        logger.info("Testing call_external_api success")
        
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {"status": "success", "data": {}}
        mock_response.raise_for_status = Mock()
        mock_request.return_value = mock_response
        
        # Execute
        result = call_external_api(mock_api_definition, {"test": "data"})
        
        # Assertions
        assert result == {"status": "success", "data": {}}
        mock_request.assert_called_once()
        
    @patch('tools.claims_api.requests.request')
    def test_timeout_raises_tool_timeout_error(self, mock_request, mock_api_definition):
        """Test that timeout raises ToolTimeoutError"""
        logger.info("Testing call_external_api timeout")
        
        # Mock timeout
        import requests
        mock_request.side_effect = requests.exceptions.Timeout("Request timed out")
        
        # Execute and assert exception
        with pytest.raises(ToolTimeoutError):
            call_external_api(mock_api_definition, {"test": "data"})
            
    @patch('tools.claims_api.requests.request')
    def test_http_error_raises_external_api_error(self, mock_request, mock_api_definition):
        """Test that HTTP errors raise ExternalAPIError"""
        logger.info("Testing call_external_api HTTP error")
        
        # Mock HTTP error
        import requests
        mock_response = Mock()
        mock_response.status_code = 500
        error = requests.RequestException("Server error")
        error.response = mock_response
        mock_request.side_effect = error
        
        # Execute and assert exception
        with pytest.raises(ExternalAPIError):
            call_external_api(mock_api_definition, {"test": "data"})
            
    @patch('tools.claims_api.requests.request')
    def test_non_json_response_raises_error(self, mock_request, mock_api_definition):
        """Test that non-JSON response raises ExternalAPIError"""
        logger.info("Testing call_external_api non-JSON response")
        
        # Mock non-JSON response
        mock_response = Mock()
        mock_response.json.side_effect = ValueError("Not JSON")
        mock_response.status_code = 200
        mock_response.text = "<html>Not JSON</html>"
        mock_response.raise_for_status = Mock()
        mock_request.return_value = mock_response
        
        # Execute and assert exception
        with pytest.raises(ExternalAPIError) as exc_info:
            call_external_api(mock_api_definition, {"test": "data"})
        
        assert "Non-JSON response" in str(exc_info.value)


# ============================================================================
# TEST: Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for the complete flow"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_claim_details_flow(self):
        """Test complete end-to-end flow for claim_details intent"""
        logger.info("Testing end-to-end claim_details flow")
        
        with patch('tools.claims_api.get_api_repository') as mock_repo, \
             patch('tools.claims_api.call_external_api') as mock_call, \
             patch('tools.claims_api.extract_logging_context') as mock_ctx:
            
            # Setup mocks
            mock_ctx.return_value = {}
            
            # Mock API definitions
            details_api = Mock()
            details_api.name = "get_claim_details"
            details_api.body_template = lambda e: {"claimDetailsRequest": e}
            
            list_api = Mock()
            list_api.name = "get_claim_list"
            list_api.body_template = lambda e: {"claimsRequest": e}
            
            mock_repo.return_value = [details_api, list_api]
            
            # Mock API responses
            details_response = {
                "status": "success",
                "claimDetails": {"primary": {"claimStatus": "R"}}
            }
            
            list_response = {
                "claims": [{
                    "claimInformation": {
                        "claimNumber": "123456789",
                        "claimSequenceNumber": "1"
                    },
                    "member": {"firstName": "John"}
                }]
            }
            
            mock_call.side_effect = [details_response, list_response]
            
            # Execute
            state = {
                "intent": "claim_details",
                "entities": {
                    "claimNumber": "123456789",
                    "claimSequence": "1"
                }
            }
            
            result = await call_claims_tool_node(state)
            
            # Assertions
            assert result["tool_results"]["status"] == "success"
            assert "list_data" in result["tool_results"]["data"]
            assert result["tool_results"]["data"]["list_data"]["member"]["firstName"] == "John"


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

