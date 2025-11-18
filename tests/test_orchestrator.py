"""
Tests for Orchestrator Node

Integration tests for the orchestrator node normalization and error handling.

Run with: pytest tests/test_orchestrator.py -v
"""

import pytest
from nodes.orchestrator import orchestrator_node
from state.schema import create_initial_state
from core.logger import get_logger

logger = get_logger(__name__)


class TestOrchestratorNormalization:
    """Tests for orchestrator normalization pipeline"""
    
    @pytest.mark.asyncio
    async def test_basic_normalization(self):
        """Test basic text normalization"""
        logger.info("Testing basic orchestrator normalization")
        state = create_initial_state(
            text="  What's MY claim STATUS?  ",
            session_id="test-session-001",
            user_info={}
        )
        
        result = await orchestrator_node(state)
        
        # Verify text is normalized
        assert isinstance(result["text"], str)
        # Should be lowercase, trimmed, punctuation removed
        assert result["text"] == "whats my claim status"
        
        # Verify metadata exists
        assert "orchestrator_metadata" in result["metadata"]
        metadata = result["metadata"]["orchestrator_metadata"]
        
        assert metadata["normalization_applied"] is True
        assert metadata["original_length"] == 27  # Fixed: actual length
        assert metadata["chars_removed"] > 0
    
    @pytest.mark.asyncio
    async def test_lowercase_conversion(self):
        """Test lowercase conversion"""
        state = create_initial_state(
            text="HELLO WORLD",
            session_id="test-session-002",
            user_info={}
        )
        
        result = await orchestrator_node(state)
        
        assert result["text"] == "hello world"
    
    @pytest.mark.asyncio
    async def test_whitespace_handling(self):
        """Test whitespace trimming and collapsing"""
        state = create_initial_state(
            text="  hello    world  ",
            session_id="test-session-003",
            user_info={}
        )
        
        result = await orchestrator_node(state)
        
        # Multiple spaces collapsed to single space
        assert result["text"] == "hello world"
    
    @pytest.mark.asyncio
    async def test_punctuation_removal(self):
        """Test punctuation removal"""
        state = create_initial_state(
            text="What's the status?!",
            session_id="test-session-004",
            user_info={}
        )
        
        result = await orchestrator_node(state)
        
        # Punctuation should be removed
        assert "?" not in result["text"]
        assert "!" not in result["text"]
        assert "'" not in result["text"]
    
    @pytest.mark.asyncio
    async def test_original_text_preserved(self):
        """Test that original text is preserved in metadata"""
        original = "Hello World!"
        state = create_initial_state(
            text=original,
            session_id="test-session-005",
            user_info={}
        )
        
        result = await orchestrator_node(state)
        
        # Original text should be preserved
        assert result["metadata"]["original_text"] == original
        assert result["metadata"]["orchestrator_metadata"]["original_text"] == original


class TestOrchestratorErrorHandling:
    """Tests for orchestrator error handling"""
    
    @pytest.mark.asyncio
    async def test_empty_input_error(self):
        """Test empty input error handling"""
        logger.info("Testing empty input error handling")
        state = create_initial_state(
            text="",
            session_id="test-session-101",
            user_info={"user_id": "user-123"}
        )
        
        result = await orchestrator_node(state)
        
        # Should return empty text
        assert result["text"] == ""
        
        # Should have error code
        assert "error" in result
        assert result["error"] == "E1001"
        
        # Should have error in metadata
        metadata = result["metadata"]["orchestrator_metadata"]
        assert metadata["normalization_applied"] is False
        assert metadata["error"] is not None
        assert metadata["error"]["error_code"] == "E1001"
        assert metadata["error"]["category"] == "validation"
    
    @pytest.mark.asyncio
    async def test_invalid_type_handling(self):
        """Test invalid type conversion"""
        state = create_initial_state(
            text=123,  # Invalid: should be string
            session_id="test-session-102",
            user_info={}
        )
        
        result = await orchestrator_node(state)
        
        # Should convert to string and continue
        assert isinstance(result["text"], str)
        assert result["text"] == "123"
    
    @pytest.mark.asyncio
    async def test_graceful_fallback_on_exception(self):
        """Test graceful fallback on normalization exception"""
        # This would require mocking internal functions to trigger exception
        # For now, we test that the node handles any input gracefully
        state = create_initial_state(
            text="Test input",
            session_id="test-session-103",
            user_info={}
        )
        
        result = await orchestrator_node(state)
        
        # Should always return valid state
        assert "text" in result
        assert isinstance(result["text"], str)
        assert "metadata" in result


class TestOrchestratorStateIntegration:
    """Tests for state integration and compatibility"""
    
    @pytest.mark.asyncio
    async def test_output_compatible_with_downstream_nodes(self):
        """Test that output is compatible with downstream nodes"""
        state = create_initial_state(
            text="Hello world",
            session_id="test-session-201",
            user_info={}
        )
        
        result = await orchestrator_node(state)
        
        # Required fields for downstream nodes
        assert "text" in result
        assert isinstance(result["text"], str)
        
        # Safety node expects to call .lower() on text
        assert result["text"].lower() == result["text"]  # Already lowercase
    
    @pytest.mark.asyncio
    async def test_metadata_structure(self):
        """Test metadata structure"""
        state = create_initial_state(
            text="Test",
            session_id="test-session-202",
            user_info={}
        )
        
        result = await orchestrator_node(state)
        
        # Verify metadata structure
        assert "orchestrator_metadata" in result["metadata"]
        orchestrator_meta = result["metadata"]["orchestrator_metadata"]
        
        # Required fields
        assert "normalized_text" in orchestrator_meta
        assert "original_text" in orchestrator_meta
        assert "normalization_applied" in orchestrator_meta
        assert "original_length" in orchestrator_meta
        assert "normalized_length" in orchestrator_meta
        assert "timestamp" in orchestrator_meta
    
    @pytest.mark.asyncio
    async def test_existing_metadata_preserved(self):
        """Test that existing metadata is preserved"""
        state = create_initial_state(
            text="Test",
            session_id="test-session-203",
            user_info={}
        )
        
        # Add existing metadata
        state["metadata"] = {"existing_key": "existing_value"}
        
        result = await orchestrator_node(state)
        
        # Existing metadata should be preserved
        assert "existing_key" in result["metadata"]
        assert result["metadata"]["existing_key"] == "existing_value"
        
        # New metadata should be added
        assert "orchestrator_metadata" in result["metadata"]


class TestOrchestratorNormalizationSteps:
    """Tests for individual normalization steps"""
    
    @pytest.mark.asyncio
    async def test_unicode_normalization(self):
        """Test Unicode NFD normalization"""
        # café with composed é
        state = create_initial_state(
            text="café",
            session_id="test-session-301",
            user_info={}
        )
        
        result = await orchestrator_node(state)
        
        # Should be normalized
        assert "orchestrator_metadata" in result["metadata"]
        metadata = result["metadata"]["orchestrator_metadata"]
        assert "unicode_nfd" in metadata.get("normalization_steps", [])
    
    @pytest.mark.asyncio
    async def test_normalization_steps_tracking(self):
        """Test that normalization steps are tracked"""
        state = create_initial_state(
            text="Hello World!",
            session_id="test-session-302",
            user_info={}
        )
        
        result = await orchestrator_node(state)
        
        metadata = result["metadata"]["orchestrator_metadata"]
        steps = metadata.get("normalization_steps", [])
        
        # Expected steps
        expected_steps = [
            "strip_whitespace",
            "lowercase",
            "collapse_spaces",
            "unicode_nfd",
            "remove_zero_width"
        ]
        
        for step in expected_steps:
            assert step in steps


class TestOrchestratorTelemetry:
    """Tests for telemetry integration"""
    
    @pytest.mark.asyncio
    async def test_error_logged_on_empty_input(self):
        """Test that errors are logged to telemetry"""
        state = create_initial_state(
            text="",
            session_id="test-session-401",
            user_info={"user_id": "user-456"}
        )
        
        result = await orchestrator_node(state)
        
        # Should have error information
        assert result["error"] == "E1001"
        metadata = result["metadata"]["orchestrator_metadata"]
        assert metadata["error"] is not None
        
        # Error should have telemetry-relevant fields
        error = metadata["error"]
        assert "error_code" in error
        assert "category" in error
        assert "severity" in error
        assert "timestamp" in error


class TestOrchestratorUUID:
    """Tests for UUID generation"""
    
    @pytest.mark.asyncio
    async def test_uuid_generated_when_not_provided(self):
        """Test that orchestrator generates UUID when not provided"""
        state = create_initial_state(
            text="Test input",
            session_id="test-session-uuid-001",
            user_info={}
        )
        
        result = await orchestrator_node(state)
        
        # Should have UUID
        assert "uuid" in result
        assert result["uuid"] is not None
        assert isinstance(result["uuid"], str)
        
        # Should be valid UUID format
        import uuid as uuid_lib
        try:
            uuid_lib.UUID(result["uuid"])
            assert True
        except ValueError:
            assert False, "Generated UUID is not valid format"
    
    @pytest.mark.asyncio
    async def test_uuid_preserved_when_provided(self):
        """Test that existing UUID is preserved"""
        existing_uuid = "550e8400-e29b-41d4-a716-446655440000"
        state = create_initial_state(
            text="Test input",
            session_id="test-session-uuid-002",
            user_info={}
        )
        state["uuid"] = existing_uuid
        
        result = await orchestrator_node(state)
        
        # Should preserve existing UUID
        assert result["uuid"] == existing_uuid
    
    @pytest.mark.asyncio
    async def test_uuid_generated_on_empty_input_error(self):
        """Test that UUID is generated even on empty input error"""
        state = create_initial_state(
            text="",
            session_id="test-session-uuid-003",
            user_info={}
        )
        
        result = await orchestrator_node(state)
        
        # Should have UUID even on error
        assert "uuid" in result
        assert result["uuid"] is not None
        assert isinstance(result["uuid"], str)
    
    @pytest.mark.asyncio
    async def test_uuid_flows_to_downstream_nodes(self):
        """Test that UUID flows correctly through state"""
        state = create_initial_state(
            text="Test input for UUID flow",
            session_id="test-session-uuid-004",
            user_info={"user_id": "user-123"}
        )
        
        result = await orchestrator_node(state)
        
        # UUID should be in result
        assert "uuid" in result
        generated_uuid = result["uuid"]
        
        # Simulate state merge (like LangGraph does)
        state.update(result)
        
        # UUID should now be accessible in merged state
        assert state.get("uuid") == generated_uuid


class TestOrchestratorRealWorldScenarios:
    """Tests for real-world scenarios"""
    
    @pytest.mark.asyncio
    async def test_typical_claim_query(self):
        """Test typical claim status query"""
        state = create_initial_state(
            text="What's the status of my claim?",
            session_id="test-session-501",
            user_info={"user_id": "user-789"}
        )
        
        result = await orchestrator_node(state)
        
        # Should normalize properly
        assert "claim" in result["text"]
        assert "status" in result["text"]
        
        # Should have successful normalization
        metadata = result["metadata"]["orchestrator_metadata"]
        assert metadata["normalization_applied"] is True
        assert metadata["error"] is None
    
    @pytest.mark.asyncio
    async def test_messy_input_with_extra_spaces(self):
        """Test messy input with extra whitespace"""
        state = create_initial_state(
            text="   What   is    my   claim    status?   ",
            session_id="test-session-502",
            user_info={}
        )
        
        result = await orchestrator_node(state)
        
        # Should collapse all spaces
        assert "   " not in result["text"]
        assert "  " not in result["text"]
        
        # Words should be separated by single spaces
        assert result["text"] == "what is my claim status"
    
    @pytest.mark.asyncio
    async def test_special_characters_handling(self):
        """Test handling of special characters"""
        state = create_initial_state(
            text="Claim #12345 - Status???",
            session_id="test-session-503",
            user_info={}
        )
        
        result = await orchestrator_node(state)
        
        # Special characters should be removed
        assert "#" not in result["text"]
        assert "-" not in result["text"]
        assert "?" not in result["text"]


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

