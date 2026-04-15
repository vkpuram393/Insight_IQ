"""
Tests for Error Models

Run with: pytest tests/test_error_models.py -v
"""

import pytest
from datetime import datetime
from core.errors.models import (
    AgentError,
    ErrorResponse,
    ErrorCode,
    ErrorCategory,
    ErrorSeverity,
    ErrorDetail,
    create_validation_error,
    create_low_confidence_error,
    create_safety_error,
    create_api_error,
    create_llm_error,
    create_session_error,
    create_internal_error,
    # Orchestrator-specific helpers
    create_orchestrator_empty_input_error,
    create_orchestrator_invalid_type_error,
    create_orchestrator_normalization_error,
)
from core.logger import get_logger

logger = get_logger(__name__)


class TestAgentErrorModel:
    """Tests for the AgentError Pydantic model"""
    
    def test_create_basic_error(self):
        """Test creating a basic error"""
        logger.info("Testing basic error creation")
        try:
            error = AgentError(
                error_code=ErrorCode.INVALID_INPUT,
                category=ErrorCategory.VALIDATION,
                severity=ErrorSeverity.LOW,
                message="Test error message",
                user_message="User-friendly error message"
            )
            
            assert error.error_code == ErrorCode.INVALID_INPUT
            assert error.category == ErrorCategory.VALIDATION
            assert error.severity == ErrorSeverity.LOW
            assert error.message == "Test error message"
            assert error.user_message == "User-friendly error message"
            assert error.is_retryable is False
            assert error.timestamp is not None
            logger.info("✅ Basic error creation test passed")
        except Exception as e:
            logger.error(f"Basic error creation test failed: {e}")
            raise
    
    def test_error_with_context(self):
        """Test error with session and request context"""
        error = AgentError(
            error_code=ErrorCode.LOW_CONFIDENCE,
            category=ErrorCategory.CONFIDENCE,
            severity=ErrorSeverity.MEDIUM,
            message="Confidence too low",
            user_message="Please clarify your request",
            session_id="test-session-123",
            request_id="req-456",
            user_id="user-789"
        )
        
        assert error.session_id == "test-session-123"
        assert error.request_id == "req-456"
        assert error.user_id == "user-789"
    
    def test_error_with_metadata(self):
        """Test error with additional metadata"""
        error = AgentError(
            error_code=ErrorCode.API_TIMEOUT,
            category=ErrorCategory.API_CALL,
            severity=ErrorSeverity.HIGH,
            message="API timeout",
            user_message="Request timed out",
            metadata={
                "api_endpoint": "/claims/status",
                "timeout_seconds": 30,
                "attempt": 3
            }
        )
        
        assert error.metadata["api_endpoint"] == "/claims/status"
        assert error.metadata["timeout_seconds"] == 30
        assert error.metadata["attempt"] == 3
    
    def test_error_with_details(self):
        """Test error with detailed field information"""
        error = AgentError(
            error_code=ErrorCode.INVALID_INPUT,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.LOW,
            message="Validation failed",
            user_message="Invalid input",
            details=[
                ErrorDetail(
                    field="email",
                    value="invalid-email",
                    constraint="Valid email format required",
                    suggestion="Use format: user@example.com"
                )
            ]
        )
        
        assert len(error.details) == 1
        assert error.details[0].field == "email"
        assert error.details[0].value == "invalid-email"
        assert error.details[0].suggestion == "Use format: user@example.com"
    
    def test_retryable_error(self):
        """Test error with retry information"""
        error = AgentError(
            error_code=ErrorCode.API_RATE_LIMIT,
            category=ErrorCategory.RATE_LIMIT,
            severity=ErrorSeverity.MEDIUM,
            message="Rate limit exceeded",
            user_message="Too many requests, please wait",
            is_retryable=True,
            retry_after_seconds=60
        )
        
        assert error.is_retryable is True
        assert error.retry_after_seconds == 60
    
    def test_error_serialization(self):
        """Test error can be serialized to/from JSON"""
        logger.info("Testing error serialization")
        try:
            original_error = AgentError(
                error_code=ErrorCode.INTERNAL_ERROR,
                category=ErrorCategory.SYSTEM,
                severity=ErrorSeverity.CRITICAL,
                message="System failure",
                user_message="An error occurred",
                session_id="test-123",
                metadata={"component": "intent_agent"}
            )
            
            # Serialize to JSON
            error_json = original_error.model_dump_json()
            logger.debug(f"Serialized error: {error_json[:100]}...")
            assert isinstance(error_json, str)
            
            # Deserialize from JSON
            restored_error = AgentError.model_validate_json(error_json)
            
            assert restored_error.error_code == original_error.error_code
            assert restored_error.category == original_error.category
            assert restored_error.severity == original_error.severity
            assert restored_error.message == original_error.message
            assert restored_error.session_id == original_error.session_id
            assert restored_error.metadata == original_error.metadata
            logger.info("✅ Error serialization test passed")
        except Exception as e:
            logger.error(f"Error serialization test failed: {e}")
            raise
    
    def test_error_dict_conversion(self):
        """Test error can be converted to dict"""
        error = AgentError(
            error_code=ErrorCode.INVALID_INPUT,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.LOW,
            message="Test",
            user_message="Test user message"
        )
        
        error_dict = error.model_dump()
        
        assert isinstance(error_dict, dict)
        assert error_dict["error_code"] == "E1001"
        assert error_dict["category"] == "validation"
        assert error_dict["severity"] == "low"
        assert "timestamp" in error_dict


class TestErrorResponse:
    """Tests for the ErrorResponse model"""
    
    def test_create_error_response(self):
        """Test creating an error response"""
        error = create_validation_error(
            message="Invalid input",
            session_id="test-123"
        )
        
        response = ErrorResponse(
            error=error,
            session_id="test-123"
        )
        
        assert response.success is False
        assert response.error == error
        assert response.session_id == "test-123"
        assert response.timestamp is not None
    
    def test_error_response_serialization(self):
        """Test error response can be serialized"""
        error = create_validation_error(
            message="Test error",
            session_id="test-456"
        )
        
        response = ErrorResponse(error=error, session_id="test-456")
        response_dict = response.model_dump()
        
        assert response_dict["success"] is False
        assert "error" in response_dict
        assert response_dict["session_id"] == "test-456"
        assert "timestamp" in response_dict


class TestHelperFunctions:
    """Tests for error creation helper functions"""
    
    def test_create_validation_error(self):
        """Test validation error helper"""
        error = create_validation_error(
            message="Email is invalid",
            field="email",
            value="bad-email",
            session_id="test-123"
        )
        
        assert error.error_code == ErrorCode.INVALID_INPUT
        assert error.category == ErrorCategory.VALIDATION
        assert error.severity == ErrorSeverity.LOW
        assert error.session_id == "test-123"
        assert error.is_retryable is False
        assert len(error.details) == 1
        assert error.details[0].field == "email"
    
    def test_create_low_confidence_error(self):
        """Test low confidence error helper"""
        error = create_low_confidence_error(
            confidence=0.45,
            threshold=0.70,
            intent="claim_status",
            session_id="test-123"
        )
        
        assert error.error_code == ErrorCode.LOW_CONFIDENCE
        assert error.category == ErrorCategory.CONFIDENCE
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.related_intent == "claim_status"
        assert error.metadata["confidence"] == 0.45
        assert error.metadata["threshold"] == 0.70
        assert error.is_retryable is False
    
    def test_create_safety_error_precheck(self):
        """Test safety error helper for precheck"""
        error = create_safety_error(
            reason="Harmful content detected",
            is_precheck=True,
            session_id="test-123"
        )
        
        assert error.error_code == ErrorCode.SAFETY_PRECHECK_FAILED
        assert error.category == ErrorCategory.SAFETY
        assert error.severity == ErrorSeverity.HIGH
        assert error.node_name == "safety_precheck"
        assert error.metadata["reason"] == "Harmful content detected"
        assert error.is_retryable is False
    
    def test_create_safety_error_postcheck(self):
        """Test safety error helper for postcheck"""
        error = create_safety_error(
            reason="Generated unsafe content",
            is_precheck=False,
            session_id="test-123"
        )
        
        assert error.error_code == ErrorCode.SAFETY_POSTCHECK_FAILED
        assert error.node_name == "safety_postcheck"
    
    def test_create_api_error(self):
        """Test API error helper"""
        error = create_api_error(
            api_name="Claims API",
            error_message="Connection timeout",
            status_code=504,
            session_id="test-123",
            is_retryable=True
        )
        
        assert error.error_code == ErrorCode.API_CONNECTION_ERROR
        assert error.category == ErrorCategory.API_CALL
        assert error.severity == ErrorSeverity.HIGH
        assert error.metadata["api"] == "Claims API"
        assert error.metadata["status_code"] == 504
        assert error.is_retryable is True
        assert error.retry_after_seconds == 30
    
    def test_create_llm_error(self):
        """Test LLM error helper"""
        error = create_llm_error(
            error_message="Rate limit exceeded",
            session_id="test-123",
            is_retryable=True
        )
        
        assert error.error_code == ErrorCode.LLM_CONNECTION_ERROR
        assert error.category == ErrorCategory.LLM
        assert error.severity == ErrorSeverity.CRITICAL
        assert error.is_retryable is True
        assert error.retry_after_seconds == 60
    
    def test_create_session_error(self):
        """Test session error helper"""
        error = create_session_error(
            session_id="test-123",
            error_message="Session expired"
        )
        
        assert error.error_code == ErrorCode.SESSION_NOT_FOUND
        assert error.category == ErrorCategory.SESSION
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.session_id == "test-123"
        assert error.is_retryable is False
    
    def test_create_internal_error(self):
        """Test internal error helper"""
        error = create_internal_error(
            error_message="Unexpected error",
            stacktrace="Traceback: ...",
            session_id="test-123",
            node_name="intent_agent"
        )
        
        assert error.error_code == ErrorCode.INTERNAL_ERROR
        assert error.category == ErrorCategory.SYSTEM
        assert error.severity == ErrorSeverity.CRITICAL
        assert error.stacktrace == "Traceback: ..."
        assert error.node_name == "intent_agent"
        assert error.is_retryable is False

    
    def test_create_orchestrator_empty_input_error(self):
        """Test orchestrator empty input error helper"""
        error = create_orchestrator_empty_input_error(
            session_id="test-session-123",
            user_id="user-456"
        )
        
        assert error.error_code == ErrorCode.INVALID_INPUT
        assert error.category == ErrorCategory.VALIDATION
        assert error.severity == ErrorSeverity.LOW
        assert error.message == "Orchestrator received empty input text"
        assert error.user_message == "Please provide a message to continue."
        assert error.session_id == "test-session-123"
        assert error.user_id == "user-456"
        assert error.node_name == "orchestrator"
        assert error.is_retryable is False
        assert error.metadata["error_type"] == "empty_input"
    
    def test_create_orchestrator_invalid_type_error(self):
        """Test orchestrator invalid type error helper"""
        error = create_orchestrator_invalid_type_error(
            input_type=int,
            session_id="test-session-123",
            user_id="user-456"
        )
        
        assert error.error_code == ErrorCode.INVALID_FORMAT
        assert error.category == ErrorCategory.VALIDATION
        assert error.severity == ErrorSeverity.LOW
        assert "invalid input type: int" in error.message.lower()
        assert "invalid input format" in error.user_message.lower()
        assert error.session_id == "test-session-123"
        assert error.user_id == "user-456"
        assert error.node_name == "orchestrator"
        assert error.is_retryable is False
        assert error.metadata["error_type"] == "invalid_type"
        assert error.metadata["received_type"] == "int"
    
    def test_create_orchestrator_normalization_error(self):
        """Test orchestrator normalization error helper"""
        test_exception = ValueError("Unicode normalization failed")
        error = create_orchestrator_normalization_error(
            exception=test_exception,
            session_id="test-session-123",
            user_id="user-456",
            stacktrace="Traceback: ..."
        )
        
        assert error.error_code == ErrorCode.INTERNAL_ERROR
        assert error.category == ErrorCategory.SYSTEM
        assert error.severity == ErrorSeverity.MEDIUM
        assert "normalization failed" in error.message.lower()
        assert "processing your input" in error.user_message.lower()
        assert error.session_id == "test-session-123"
        assert error.user_id == "user-456"
        assert error.node_name == "orchestrator"
        assert error.is_retryable is True
        assert error.stacktrace == "Traceback: ..."
        assert error.metadata["error_type"] == "normalization_failure"
        assert error.metadata["exception_type"] == "ValueError"
        assert error.metadata["exception_message"] == "Unicode normalization failed"


class TestErrorCodes:
    """Tests for error code organization"""
    
    def test_error_code_values(self):
        """Test error codes follow numbering scheme"""
        # Validation (1xxx)
        assert ErrorCode.INVALID_INPUT == "E1001"
        assert ErrorCode.MISSING_REQUIRED_FIELD == "E1002"
        
        # Intent (2xxx)
        assert ErrorCode.INTENT_DETECTION_FAILED == "E2001"
        assert ErrorCode.LOW_CONFIDENCE == "E2002"
        
        # Safety (3xxx)
        assert ErrorCode.SAFETY_PRECHECK_FAILED == "E3001"
        assert ErrorCode.SAFETY_POSTCHECK_FAILED == "E3002"
        
        # API (4xxx)
        assert ErrorCode.API_CONNECTION_ERROR == "E4001"
        assert ErrorCode.API_TIMEOUT == "E4002"
        
        # LLM (5xxx)
        assert ErrorCode.LLM_CONNECTION_ERROR == "E5001"
        
        # Memory (6xxx)
        assert ErrorCode.MEMORY_STORE_ERROR == "E6001"
        
        # System (9xxx)
        assert ErrorCode.INTERNAL_ERROR == "E9001"
        assert ErrorCode.UNKNOWN_ERROR == "E9999"
    
    def test_all_error_codes_unique(self):
        """Test all error codes are unique"""
        codes = [code.value for code in ErrorCode]
        assert len(codes) == len(set(codes))


class TestErrorCategories:
    """Tests for error categories"""
    
    def test_all_categories_exist(self):
        """Test all expected categories exist"""
        expected_categories = [
            "validation", "authentication", "intent_classification",
            "confidence", "clarification", "safety", "api_call",
            "tool_execution", "memory", "cache", "persistence",
            "llm", "context", "session", "timeout", "rate_limit",
            "system", "unknown"
        ]
        
        for category in expected_categories:
            assert category in [c.value for c in ErrorCategory]


class TestErrorSeverity:
    """Tests for error severity levels"""
    
    def test_all_severity_levels_exist(self):
        """Test all severity levels exist"""
        assert ErrorSeverity.LOW == "low"
        assert ErrorSeverity.MEDIUM == "medium"
        assert ErrorSeverity.HIGH == "high"
        assert ErrorSeverity.CRITICAL == "critical"


class TestIntegrationScenarios:
    """Integration tests for realistic error scenarios"""
    
    def test_validation_error_scenario(self):
        """Test complete validation error scenario"""
        # User submits invalid input
        error = create_validation_error(
            message="Input text exceeds maximum length",
            field="text",
            value="a" * 3000,
            session_id="session-123"
        )
        
        # Create API response
        response = ErrorResponse(error=error, session_id="session-123")
        response_dict = response.model_dump()
        
        # Verify response structure
        assert response_dict["success"] is False
        assert response_dict["error"]["user_message"] == "The information provided is not valid. Please check your input and try again."
    
    def test_low_confidence_scenario(self):
        """Test low confidence handling scenario"""
        # Agent detects low confidence
        error = create_low_confidence_error(
            confidence=0.45,
            threshold=0.70,
            intent="unknown",
            session_id="session-456"
        )
        
        # Should trigger clarification flow
        assert error.category == ErrorCategory.CONFIDENCE
        assert "Could you rephrase" in error.user_message
        assert error.metadata["confidence"] == 0.45
    
    def test_api_failure_with_retry_scenario(self):
        """Test API failure with retry logic scenario"""
        # External API fails
        error = create_api_error(
            api_name="Claims API",
            error_message="Service temporarily unavailable",
            status_code=503,
            session_id="session-789",
            is_retryable=True
        )
        
        # Should be marked as retryable
        assert error.is_retryable is True
        assert error.retry_after_seconds == 30
        assert error.severity == ErrorSeverity.HIGH
        
        # User gets friendly message
        assert "try again" in error.user_message.lower()
    
    def test_critical_llm_failure_scenario(self):
        """Test critical LLM failure scenario"""
        # LLM service fails
        error = create_llm_error(
            error_message="Model unavailable",
            session_id="session-999",
            is_retryable=True
        )
        
        # Should be critical severity
        assert error.severity == ErrorSeverity.CRITICAL
        assert error.category == ErrorCategory.LLM
        
        # Response should be created
        response = ErrorResponse(error=error, session_id="session-999")
        assert response.success is False


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

