"""
Tests for Node Result Models

Run with: pytest tests/test_node_models.py -v
"""

import pytest
from datetime import datetime
from core.node_models import (
    # Intent models
    IntentResult,
    IntentType,
    IntentComplexity,
    EntityExtractionResult,
    # Safety models
    SafetyResult,
    SafetyCheckType,
    SafetyViolationType,
    GuardrailResult,
    # Tool models
    ToolResult,
    ToolExecutionStatus,
    # Context models
    ContextResult,
    ConversationMessage,
    SessionFact,
    # Clarification models
    ClarificationResult,
    ClarificationType,
    # Response models
    ResponsePayload,
    ResponseType,
    ResponseSource,
    # Cache models
    CacheResult,
    CacheStatus,
    # Confidence models
    ConfidenceCheckResult,
    ConfidenceCheckDecision,
    # Helper functions
    create_intent_result,
    create_safety_result,
    create_tool_result,
    create_response_payload,
)
from core.logger import get_logger

logger = get_logger(__name__)


class TestEntityExtractionResult:
    """Tests for EntityExtractionResult model"""
    
    def test_create_basic_entity_result(self):
        """Test creating basic entity result"""
        result = EntityExtractionResult(
            claim_number="CLM-12345",
            member_id="MEM-67890"
        )
        
        assert result.claim_number == "CLM-12345"
        assert result.member_id == "MEM-67890"
        assert result.prescription_number is None
        assert isinstance(result.raw_entities, dict)
    
    def test_entity_result_with_all_fields(self):
        """Test entity result with all fields populated"""
        result = EntityExtractionResult(
            claim_number="CLM-12345",
            member_id="MEM-67890",
            prescription_number="RX-11111",
            medication_name="Lipitor",
            date_from="2025-01-01",
            date_to="2025-12-31",
            raw_entities={"confidence": 0.95}
        )
        
        assert result.claim_number == "CLM-12345"
        assert result.medication_name == "Lipitor"
        assert result.raw_entities["confidence"] == 0.95
    
    def test_entity_result_serialization(self):
        """Test entity result can be serialized"""
        result = EntityExtractionResult(claim_number="CLM-12345")
        result_dict = result.model_dump()
        
        assert isinstance(result_dict, dict)
        assert result_dict["claim_number"] == "CLM-12345"


class TestIntentResult:
    """Tests for IntentResult model"""
    
    def test_create_basic_intent_result(self):
        """Test creating basic intent result"""
        logger.info("Testing basic intent result creation")
        try:
            result = IntentResult(
                intent="claim_status",
                confidence=0.85
            )
            
            assert result.intent == "claim_status"
            assert result.confidence == 0.85
            assert result.needs_clarification is False
            assert isinstance(result.all_scores, dict)
            logger.info("✅ Basic intent result test passed")
        except Exception as e:
            logger.error(f"Basic intent result test failed: {e}")
            raise
    
    def test_intent_result_with_entities(self):
        """Test intent result with extracted entities"""
        entities = EntityExtractionResult(claim_number="CLM-12345")
        
        result = IntentResult(
            intent="claim_status",
            confidence=0.92,
            entities=entities
        )
        
        assert result.entities.claim_number == "CLM-12345"
    
    def test_confidence_validation(self):
        """Test confidence must be between 0 and 1"""
        logger.info("Testing confidence validation")
        try:
            with pytest.raises(ValueError):
                IntentResult(intent="test", confidence=1.5)
            logger.debug("Correctly rejected confidence > 1.0")
            
            with pytest.raises(ValueError):
                IntentResult(intent="test", confidence=-0.1)
            logger.debug("Correctly rejected confidence < 0.0")
            logger.info("✅ Confidence validation test passed")
        except Exception as e:
            logger.error(f"Confidence validation test failed: {e}")
            raise
    
    def test_intent_result_with_all_fields(self):
        """Test intent result with all fields"""
        result = IntentResult(
            intent="claim_status",
            confidence=0.85,
            needs_clarification=False,
            all_scores={"claim_status": 0.85, "claim_details": 0.42},
            top_candidates=[("claim_status", 0.85)],
            is_simple=True,
            is_complex=False,
            reasoning="Detected status keyword",
            classification_method="keyword_matching",
            processing_time_ms=15.3
        )
        
        assert result.intent == "claim_status"
        assert result.is_simple is True
        assert result.classification_method == "keyword_matching"
        assert result.processing_time_ms == 15.3
    
    def test_create_intent_result_helper(self):
        """Test helper function for creating intent result"""
        result = create_intent_result(
            intent="greeting",
            confidence=1.0,
            is_simple=True
        )
        
        assert isinstance(result, IntentResult)
        assert result.intent == "greeting"
        assert result.confidence == 1.0


class TestSafetyResult:
    """Tests for SafetyResult/GuardrailResult model"""
    
    def test_create_passed_safety_result(self):
        """Test creating safety result that passed"""
        result = SafetyResult(
            check_type=SafetyCheckType.PRECHECK,
            passed=True
        )
        
        assert result.check_type == SafetyCheckType.PRECHECK
        assert result.passed is True
        assert result.violation_type is None
    
    def test_create_failed_safety_result(self):
        """Test creating safety result that failed"""
        result = SafetyResult(
            check_type=SafetyCheckType.PRECHECK,
            passed=False,
            violation_type=SafetyViolationType.SELF_HARM,
            block_reason="Detected self-harm content",
            detected_keywords=["suicide"],
            confidence_score=0.95,
            suggested_action="block",
            user_message="I cannot process that request."
        )
        
        assert result.passed is False
        assert result.violation_type == SafetyViolationType.SELF_HARM
        assert "suicide" in result.detected_keywords
        assert result.confidence_score == 0.95
    
    def test_guardrail_result_alias(self):
        """Test GuardrailResult is alias for SafetyResult"""
        result = GuardrailResult(
            check_type=SafetyCheckType.POSTCHECK,
            passed=True
        )
        
        assert isinstance(result, SafetyResult)
        assert result.check_type == SafetyCheckType.POSTCHECK
    
    def test_create_safety_result_helper(self):
        """Test helper function"""
        result = create_safety_result(
            check_type=SafetyCheckType.PRECHECK,
            passed=True
        )
        
        assert isinstance(result, SafetyResult)
        assert result.passed is True


class TestToolResult:
    """Tests for ToolResult model"""
    
    def test_create_successful_tool_result(self):
        """Test creating successful tool result"""
        result = ToolResult(
            tool_name="claims_api",
            status=ToolExecutionStatus.SUCCESS,
            data={
                "claim_id": "CLM-12345",
                "status": "processing"
            }
        )
        
        assert result.tool_name == "claims_api"
        assert result.status == ToolExecutionStatus.SUCCESS
        assert result.data["claim_id"] == "CLM-12345"
        assert result.error_message is None
    
    def test_create_failed_tool_result(self):
        """Test creating failed tool result"""
        result = ToolResult(
            tool_name="claims_api",
            status=ToolExecutionStatus.FAILURE,
            data={},
            error_message="Connection timeout",
            error_code="TIMEOUT",
            is_retryable=True
        )
        
        assert result.status == ToolExecutionStatus.FAILURE
        assert result.error_message == "Connection timeout"
        assert result.is_retryable is True
    
    def test_tool_result_with_metadata(self):
        """Test tool result with full metadata"""
        result = ToolResult(
            tool_name="claims_api",
            status=ToolExecutionStatus.SUCCESS,
            data={"claim_id": "CLM-12345"},
            execution_time_ms=245.8,
            api_endpoint="/api/v1/claims/CLM-12345",
            http_status_code=200,
            retry_count=1,
            from_cache=False
        )
        
        assert result.execution_time_ms == 245.8
        assert result.http_status_code == 200
        assert result.retry_count == 1
    
    def test_create_tool_result_helper(self):
        """Test helper function"""
        result = create_tool_result(
            tool_name="test_api",
            status=ToolExecutionStatus.SUCCESS,
            data={"test": "data"}
        )
        
        assert isinstance(result, ToolResult)
        assert result.tool_name == "test_api"


class TestContextResult:
    """Tests for ContextResult model"""
    
    def test_create_empty_context_result(self):
        """Test creating empty context result"""
        result = ContextResult()
        
        assert len(result.conversation_history) == 0
        assert len(result.relevant_facts) == 0
        assert result.history_length == 0
        assert result.facts_count == 0
    
    def test_create_context_with_history(self):
        """Test context result with conversation history"""
        messages = [
            ConversationMessage(
                role="user",
                content="What's my claim status?",
                timestamp="2025-11-10T10:29:00Z"
            ),
            ConversationMessage(
                role="assistant",
                content="Your claim is processing.",
                timestamp="2025-11-10T10:29:05Z"
            )
        ]
        
        result = ContextResult(
            conversation_history=messages,
            history_length=len(messages)
        )
        
        assert len(result.conversation_history) == 2
        assert result.history_length == 2
        assert result.conversation_history[0].role == "user"
    
    def test_create_context_with_facts(self):
        """Test context result with session facts"""
        facts = [
            SessionFact(
                fact_type="claim_mention",
                data={"claim_number": "CLM-12345"},
                extracted_at="2025-11-10T10:29:00Z"
            )
        ]
        
        result = ContextResult(
            relevant_facts=facts,
            facts_count=len(facts)
        )
        
        assert len(result.relevant_facts) == 1
        assert result.facts_count == 1
        assert result.relevant_facts[0].fact_type == "claim_mention"


class TestClarificationResult:
    """Tests for ClarificationResult model"""
    
    def test_create_clarification_result(self):
        """Test creating clarification result"""
        result = ClarificationResult(
            clarifying_question="Could you provide your claim number?",
            clarification_type=ClarificationType.MISSING_ENTITY
        )
        
        assert result.needs_clarification is True
        assert "claim number" in result.clarifying_question
        assert result.clarification_type == ClarificationType.MISSING_ENTITY
    
    def test_clarification_with_missing_entities(self):
        """Test clarification for missing entities"""
        result = ClarificationResult(
            clarifying_question="Could you provide your claim number?",
            clarification_type=ClarificationType.MISSING_ENTITY,
            original_intent="claim_status",
            missing_entities=["claim_number"],
            expected_entity_types=["claim_number"]
        )
        
        assert "claim_number" in result.missing_entities
        assert "claim_number" in result.expected_entity_types
        assert result.original_intent == "claim_status"
    
    def test_clarification_attempt_tracking(self):
        """Test clarification attempt tracking"""
        result = ClarificationResult(
            clarifying_question="Please clarify",
            clarification_type=ClarificationType.AMBIGUOUS_INTENT,
            current_attempt=2,
            max_clarification_attempts=3
        )
        
        assert result.current_attempt == 2
        assert result.max_clarification_attempts == 3


class TestResponsePayload:
    """Tests for ResponsePayload model"""
    
    def test_create_basic_response_payload(self):
        """Test creating basic response payload"""
        result = ResponsePayload(
            response="Your claim is being processed.",
            response_type=ResponseType.DIRECT_ANSWER,
            response_source=ResponseSource.LLM_GENERATED
        )
        
        assert result.response == "Your claim is being processed."
        assert result.response_type == ResponseType.DIRECT_ANSWER
        assert result.response_source == ResponseSource.LLM_GENERATED
    
    def test_response_with_token_usage(self):
        """Test response with token usage tracking"""
        result = ResponsePayload(
            response="Test response",
            response_type=ResponseType.DIRECT_ANSWER,
            response_source=ResponseSource.LLM_GENERATED,
            input_tokens=150,
            output_tokens=45,
            total_tokens=195,
            estimated_cost_usd=0.0001
        )
        
        assert result.input_tokens == 150
        assert result.output_tokens == 45
        assert result.total_tokens == 195
        assert result.estimated_cost_usd == 0.0001
    
    def test_response_with_quality_metrics(self):
        """Test response with quality metrics"""
        result = ResponsePayload(
            response="Test response",
            response_type=ResponseType.DIRECT_ANSWER,
            response_source=ResponseSource.LLM_GENERATED,
            confidence_score=0.92,
            completeness_score=0.95
        )
        
        assert result.confidence_score == 0.92
        assert result.completeness_score == 0.95
    
    def test_response_with_tools_used(self):
        """Test response with tools tracking"""
        result = ResponsePayload(
            response="Test response",
            response_type=ResponseType.DIRECT_ANSWER,
            response_source=ResponseSource.LLM_GENERATED,
            tools_used=["claims_api", "member_api"]
        )
        
        assert len(result.tools_used) == 2
        assert "claims_api" in result.tools_used
    
    def test_create_response_payload_helper(self):
        """Test helper function"""
        result = create_response_payload(
            response="Test",
            response_type=ResponseType.GREETING,
            response_source=ResponseSource.TEMPLATE
        )
        
        assert isinstance(result, ResponsePayload)
        assert result.response == "Test"


class TestCacheResult:
    """Tests for CacheResult model"""
    
    def test_create_cache_hit_result(self):
        """Test creating cache hit result"""
        result = CacheResult(
            cache_hit=True,
            status=CacheStatus.HIT,
            cached_response="Cached response text",
            cached_intent="claim_status",
            cached_confidence=0.89
        )
        
        assert result.cache_hit is True
        assert result.status == CacheStatus.HIT
        assert result.cached_response == "Cached response text"
    
    def test_create_cache_miss_result(self):
        """Test creating cache miss result"""
        result = CacheResult(
            cache_hit=False,
            status=CacheStatus.MISS
        )
        
        assert result.cache_hit is False
        assert result.status == CacheStatus.MISS
        assert result.cached_response is None
    
    def test_cache_result_with_metadata(self):
        """Test cache result with full metadata"""
        result = CacheResult(
            cache_hit=True,
            status=CacheStatus.HIT,
            cached_response="Test",
            cache_key="cache:abc123",
            cache_age_seconds=245.3,
            ttl_seconds=3600,
            retrieval_time_ms=3.2,
            cache_backend="redis"
        )
        
        assert result.cache_key == "cache:abc123"
        assert result.cache_age_seconds == 245.3
        assert result.ttl_seconds == 3600
        assert result.cache_backend == "redis"


class TestConfidenceCheckResult:
    """Tests for ConfidenceCheckResult model"""
    
    def test_create_proceed_decision(self):
        """Test creating proceed decision"""
        result = ConfidenceCheckResult(
            decision=ConfidenceCheckDecision.PROCEED,
            confidence=0.85,
            threshold=0.70,
            passed_threshold=True,
            decision_reason="Confidence meets threshold",
            next_node="tool_call"
        )
        
        assert result.decision == ConfidenceCheckDecision.PROCEED
        assert result.passed_threshold is True
        assert result.next_node == "tool_call"
    
    def test_create_clarify_decision(self):
        """Test creating clarify decision"""
        result = ConfidenceCheckResult(
            decision=ConfidenceCheckDecision.CLARIFY,
            confidence=0.62,
            threshold=0.70,
            passed_threshold=False,
            required_entities=["claim_number"],
            missing_entities=["claim_number"],
            entities_complete=False,
            decision_reason="Missing claim_number",
            next_node="clarification"
        )
        
        assert result.decision == ConfidenceCheckDecision.CLARIFY
        assert result.passed_threshold is False
        assert "claim_number" in result.missing_entities
        assert result.entities_complete is False


class TestIntegrationScenarios:
    """Integration tests for realistic scenarios"""
    
    def test_complete_intent_flow(self):
        """Test complete intent classification flow"""
        logger.info("Testing complete intent classification flow")
        try:
            # Create entities
            entities = EntityExtractionResult(claim_number="CLM-12345")
            logger.debug(f"Created entities: {entities.claim_number}")
            
            # Create intent result
            intent_result = IntentResult(
                intent="claim_status",
                confidence=0.89,
                entities=entities,
                is_simple=True,
                classification_method="keyword_matching"
            )
            logger.debug(f"Created intent result: {intent_result.intent}")
            
            assert intent_result.intent == "claim_status"
            assert intent_result.entities.claim_number == "CLM-12345"
            
            # Serialize
            result_dict = intent_result.model_dump()
            logger.debug(f"Serialized result: {list(result_dict.keys())}")
            assert result_dict["intent"] == "claim_status"
            assert result_dict["entities"]["claim_number"] == "CLM-12345"
            logger.info("✅ Complete intent flow test passed")
        except Exception as e:
            logger.error(f"Complete intent flow test failed: {e}")
            raise
    
    def test_complete_tool_call_flow(self):
        """Test complete tool execution flow"""
        # Successful tool call
        tool_result = ToolResult(
            tool_name="claims_api",
            status=ToolExecutionStatus.SUCCESS,
            data={"claim_id": "CLM-12345", "status": "approved"},
            execution_time_ms=234.5,
            http_status_code=200
        )
        
        assert tool_result.status == ToolExecutionStatus.SUCCESS
        assert tool_result.data["status"] == "approved"
        
        # Use in response generation
        response = ResponsePayload(
            response="Your claim has been approved!",
            response_type=ResponseType.DIRECT_ANSWER,
            response_source=ResponseSource.LLM_GENERATED,
            tools_used=[tool_result.tool_name]
        )
        
        assert "claims_api" in response.tools_used
    
    def test_clarification_flow(self):
        """Test complete clarification flow"""
        # Low confidence intent
        intent_result = IntentResult(
            intent="claim_status",
            confidence=0.55,
            needs_clarification=True
        )
        
        # Create clarification
        clarification = ClarificationResult(
            clarifying_question="Could you provide more details?",
            clarification_type=ClarificationType.LOW_CONFIDENCE,
            original_intent=intent_result.intent
        )
        
        assert clarification.needs_clarification is True
        assert clarification.original_intent == "claim_status"


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

