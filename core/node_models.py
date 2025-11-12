"""
Node Result Models for Conversational Agent System

Comprehensive Pydantic models for all node outputs in the LangGraph agent system.
Each model represents the structured output from a specific node type.
"""

from typing import Optional, Dict, Any, List, Tuple
from pydantic import BaseModel, Field, field_validator
from datetime import datetime, timezone
from enum import Enum


# ============================================================================
# INTENT CLASSIFICATION MODELS
# ============================================================================

class IntentType(str, Enum):
    """Supported intent types"""
    # Claim intents
    CLAIM_STATUS = "claim_status"
    CLAIM_DETAILS = "claim_details"
    CLAIM_PENDING = "claim_pending"
    CLAIM_LIST = "claim_list"
    CLAIM_SUMMARY = "claim_summary"
    REJECTION_REASONS = "rejection_reasons"
    EXPENSIVE_CLAIMS = "expensive_claims"
    DATE_RANGE_SEARCH = "date_range_search"
    
    # Member intents
    MEMBER_INFO = "member_info"
    BENEFITS_INFO = "benefits_info"
    COPAY_INFO = "copay_info"
    DEDUCTIBLE_INFO = "deductible_info"
    
    # General intents
    GREETING = "greeting"
    HELP = "help"
    APPEAL_INFO = "appeal_info"
    
    # Meta intents
    EMPTY_QUERY = "empty_query"
    OUT_OF_SCOPE = "out_of_scope"
    UNKNOWN = "unknown"


class IntentComplexity(str, Enum):
    """Intent complexity level"""
    SIMPLE = "simple"   # Can use templates/patterns
    COMPLEX = "complex"  # Requires LLM processing


class EntityExtractionResult(BaseModel):
    """Extracted entities from user input"""
    claim_number: Optional[str] = Field(None, description="Extracted claim number")
    member_id: Optional[str] = Field(None, description="Extracted member ID")
    prescription_number: Optional[str] = Field(None, description="Prescription number")
    medication_name: Optional[str] = Field(None, description="Medication name")
    date_from: Optional[str] = Field(None, description="Start date for range queries")
    date_to: Optional[str] = Field(None, description="End date for range queries")
    raw_entities: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Additional extracted entities"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "claim_number": "CLM-12345",
                "member_id": "MEM-67890",
                "medication_name": "Lipitor",
                "raw_entities": {"confidence_score": 0.92}
            }
        }


class IntentResult(BaseModel):
    """
    Result from intent classification node
    
    Contains the detected user intent, confidence score,
    extracted entities, and metadata about the classification.
    """
    intent: str = Field(..., description="Detected intent name")
    confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Confidence score (0.0 to 1.0)"
    )
    needs_clarification: bool = Field(
        default=False, 
        description="Whether clarification is needed"
    )
    
    # Classification metadata
    all_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Scores for all candidate intents"
    )
    top_candidates: List[Tuple[str, float]] = Field(
        default_factory=list,
        description="Top N intent candidates with scores"
    )
    is_simple: bool = Field(
        default=False, 
        description="Whether intent can use simple templates"
    )
    is_complex: bool = Field(
        default=False, 
        description="Whether intent requires LLM processing"
    )
    
    # Extracted information
    entities: Optional[EntityExtractionResult] = Field(
        None, 
        description="Extracted entities from input"
    )
    
    # Reasoning and metadata
    reasoning: Optional[str] = Field(
        None, 
        description="Explanation of classification decision"
    )
    classification_method: Optional[str] = Field(
        None, 
        description="Method used (keyword, pattern, llm)"
    )
    processing_time_ms: Optional[float] = Field(
        None, 
        description="Time taken for classification"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Classification timestamp"
    )
    
    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Ensure confidence is between 0 and 1"""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "intent": "claim_status",
                "confidence": 0.85,
                "needs_clarification": False,
                "all_scores": {
                    "claim_status": 0.85,
                    "claim_details": 0.42,
                    "claim_list": 0.15
                },
                "top_candidates": [
                    ("claim_status", 0.85),
                    ("claim_details", 0.42)
                ],
                "is_simple": True,
                "is_complex": False,
                "entities": {
                    "claim_number": "CLM-12345"
                },
                "reasoning": "Detected 'status' and 'claim' keywords with high weight",
                "classification_method": "keyword_matching",
                "processing_time_ms": 15.3,
                "timestamp": "2025-11-10T10:30:45.123456Z"
            }
        }


# ============================================================================
# SAFETY/GUARDRAIL MODELS
# ============================================================================

class SafetyCheckType(str, Enum):
    """Type of safety check"""
    PRECHECK = "precheck"   # Input validation
    POSTCHECK = "postcheck"  # Output validation


class SafetyViolationType(str, Enum):
    """Types of safety violations"""
    SELF_HARM = "self_harm"
    VIOLENCE = "violence"
    HATE_SPEECH = "hate_speech"
    PII_EXPOSURE = "pii_exposure"
    MALICIOUS_INPUT = "malicious_input"
    EXCESSIVE_LENGTH = "excessive_length"
    INAPPROPRIATE_CONTENT = "inappropriate_content"
    NONE = "none"


class SafetyResult(BaseModel):
    """
    Result from safety/guardrail checks
    
    Also known as GuardrailResult. Validates both input (precheck)
    and output (postcheck) for safety violations.
    """
    check_type: SafetyCheckType = Field(..., description="Type of safety check")
    passed: bool = Field(..., description="Whether safety check passed")
    
    # Violation details
    violation_type: Optional[SafetyViolationType] = Field(
        None, 
        description="Type of violation if failed"
    )
    block_reason: Optional[str] = Field(
        None, 
        description="Detailed reason for blocking"
    )
    
    # Detection metadata
    detected_keywords: List[str] = Field(
        default_factory=list,
        description="Harmful keywords detected"
    )
    confidence_score: Optional[float] = Field(
        None, 
        ge=0.0, 
        le=1.0,
        description="Confidence in safety assessment"
    )
    
    # Suggested actions
    suggested_action: Optional[str] = Field(
        None, 
        description="Recommended action (block, warn, allow)"
    )
    user_message: Optional[str] = Field(
        None, 
        description="Safe message to show user if blocked"
    )
    
    # Processing metadata
    processing_time_ms: Optional[float] = Field(
        None, 
        description="Time taken for safety check"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Check timestamp"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "check_type": "precheck",
                "passed": False,
                "violation_type": "self_harm",
                "block_reason": "Detected self-harm related content",
                "detected_keywords": ["suicide"],
                "confidence_score": 0.95,
                "suggested_action": "block",
                "user_message": "I cannot process that request.",
                "processing_time_ms": 8.2,
                "timestamp": "2025-11-10T10:30:45.123456Z"
            }
        }


# Create alias for backward compatibility
GuardrailResult = SafetyResult


# ============================================================================
# TOOL/API CALL MODELS
# ============================================================================

class ToolExecutionStatus(str, Enum):
    """Tool execution status"""
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    PARTIAL = "partial"


class ToolResult(BaseModel):
    """
    Result from external API/tool calls
    
    Contains data retrieved from external systems like Claims API,
    Prescription systems, or Member services.
    """
    tool_name: str = Field(..., description="Name of the tool/API called")
    status: ToolExecutionStatus = Field(..., description="Execution status")
    
    # Result data
    data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Data returned by the tool"
    )
    
    # Error information
    error_message: Optional[str] = Field(
        None, 
        description="Error message if execution failed"
    )
    error_code: Optional[str] = Field(
        None, 
        description="Error code for programmatic handling"
    )
    
    # Execution metadata
    execution_time_ms: Optional[float] = Field(
        None, 
        description="Time taken to execute tool"
    )
    api_endpoint: Optional[str] = Field(
        None, 
        description="API endpoint called"
    )
    http_status_code: Optional[int] = Field(
        None, 
        description="HTTP status code if applicable"
    )
    
    # Retry information
    retry_count: int = Field(default=0, description="Number of retries attempted")
    is_retryable: bool = Field(default=False, description="Whether call can be retried")
    
    # Cache information
    from_cache: bool = Field(default=False, description="Whether result from cache")
    cache_key: Optional[str] = Field(None, description="Cache key used")
    
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Execution timestamp"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "tool_name": "claims_api",
                "status": "success",
                "data": {
                    "claim_id": "CLM-12345",
                    "status": "processing",
                    "submitted_date": "2025-01-10",
                    "expected_completion": "5-7 business days"
                },
                "execution_time_ms": 245.8,
                "api_endpoint": "/api/v1/claims/CLM-12345",
                "http_status_code": 200,
                "retry_count": 0,
                "is_retryable": True,
                "from_cache": False,
                "timestamp": "2025-11-10T10:30:45.123456Z"
            }
        }


# ============================================================================
# CONTEXT BUILDING MODELS
# ============================================================================

class ConversationMessage(BaseModel):
    """Single message in conversation history"""
    role: str = Field(..., description="Role: user or assistant")
    content: str = Field(..., description="Message content")
    timestamp: Optional[str] = Field(None, description="Message timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class SessionFact(BaseModel):
    """Important fact extracted from conversation"""
    fact_type: str = Field(..., description="Type of fact (e.g., claim_mention)")
    data: Dict[str, Any] = Field(..., description="Fact data")
    extracted_at: Optional[str] = Field(None, description="When fact was extracted")
    relevance_score: Optional[float] = Field(None, description="Relevance to current query")


class ContextResult(BaseModel):
    """
    Result from context building node
    
    Contains conversation history and relevant facts needed
    to provide context-aware responses.
    """
    conversation_history: List[ConversationMessage] = Field(
        default_factory=list,
        description="Recent conversation messages"
    )
    relevant_facts: List[SessionFact] = Field(
        default_factory=list,
        description="Important facts from session"
    )
    
    # Context statistics
    history_length: int = Field(default=0, description="Number of messages in history")
    facts_count: int = Field(default=0, description="Number of relevant facts")
    context_window_size: int = Field(default=10, description="Max messages to keep")
    
    # Memory metadata
    memory_source: Optional[str] = Field(
        None, 
        description="Source of memory (inmemory, redis, memorystore)"
    )
    retrieval_time_ms: Optional[float] = Field(
        None, 
        description="Time to retrieve context"
    )
    
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Context retrieval timestamp"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "conversation_history": [
                    {
                        "role": "user",
                        "content": "What's my claim status?",
                        "timestamp": "2025-11-10T10:29:00Z"
                    },
                    {
                        "role": "assistant",
                        "content": "Your claim #12345 is processing.",
                        "timestamp": "2025-11-10T10:29:05Z"
                    }
                ],
                "relevant_facts": [
                    {
                        "fact_type": "claim_mention",
                        "data": {"claim_number": "CLM-12345"},
                        "extracted_at": "2025-11-10T10:29:00Z"
                    }
                ],
                "history_length": 2,
                "facts_count": 1,
                "context_window_size": 10,
                "memory_source": "redis",
                "retrieval_time_ms": 12.4,
                "timestamp": "2025-11-10T10:30:45.123456Z"
            }
        }


# ============================================================================
# CLARIFICATION MODELS
# ============================================================================

class ClarificationType(str, Enum):
    """Type of clarification needed"""
    LOW_CONFIDENCE = "low_confidence"
    MISSING_ENTITY = "missing_entity"
    AMBIGUOUS_INTENT = "ambiguous_intent"
    INCOMPLETE_INFO = "incomplete_info"
    OUT_OF_SCOPE = "out_of_scope"


class ClarificationResult(BaseModel):
    """
    Result from clarification node
    
    Generated when the system needs more information from the user
    to properly handle their request.
    """
    needs_clarification: bool = Field(
        default=True, 
        description="Whether clarification is needed"
    )
    clarifying_question: str = Field(..., description="Question to ask user")
    clarification_type: ClarificationType = Field(
        ..., 
        description="Type of clarification needed"
    )
    
    # Context for clarification
    original_intent: Optional[str] = Field(
        None, 
        description="Original detected intent"
    )
    missing_entities: List[str] = Field(
        default_factory=list,
        description="Entities that are missing"
    )
    suggested_intents: List[str] = Field(
        default_factory=list,
        description="Possible intents user might mean"
    )
    
    # Question generation metadata
    question_template: Optional[str] = Field(
        None, 
        description="Template used to generate question"
    )
    confidence_score: Optional[float] = Field(
        None, 
        description="Confidence in this clarification approach"
    )
    
    # Follow-up handling
    expected_entity_types: List[str] = Field(
        default_factory=list,
        description="Entity types expected in user's answer"
    )
    max_clarification_attempts: int = Field(
        default=3, 
        description="Max clarification rounds allowed"
    )
    current_attempt: int = Field(
        default=1, 
        description="Current clarification attempt number"
    )
    
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Clarification generation timestamp"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "needs_clarification": True,
                "clarifying_question": "Could you provide your claim number?",
                "clarification_type": "missing_entity",
                "original_intent": "claim_status",
                "missing_entities": ["claim_number"],
                "suggested_intents": [],
                "question_template": "Could you provide your {entity_name}?",
                "confidence_score": 0.95,
                "expected_entity_types": ["claim_number"],
                "max_clarification_attempts": 3,
                "current_attempt": 1,
                "timestamp": "2025-11-10T10:30:45.123456Z"
            }
        }


# ============================================================================
# RESPONSE GENERATION MODELS
# ============================================================================

class ResponseType(str, Enum):
    """Type of response generated"""
    DIRECT_ANSWER = "direct_answer"
    CLARIFICATION = "clarification"
    ERROR = "error"
    GREETING = "greeting"
    HELP = "help"
    FALLBACK = "fallback"


class ResponseSource(str, Enum):
    """Source of response content"""
    LLM_GENERATED = "llm_generated"
    TEMPLATE = "template"
    CACHED = "cached"
    FALLBACK = "fallback"


class ResponsePayload(BaseModel):
    """
    Result from response generation node
    
    Contains the final response to be sent to the user,
    along with metadata about how it was generated.
    """
    response: str = Field(..., description="The generated response text")
    response_type: ResponseType = Field(..., description="Type of response")
    response_source: ResponseSource = Field(..., description="How response was generated")
    
    # Generation metadata
    llm_model: Optional[str] = Field(None, description="LLM model used (if applicable)")
    prompt_template: Optional[str] = Field(None, description="Prompt template used")
    temperature: Optional[float] = Field(None, description="LLM temperature setting")
    
    # Token usage (for cost tracking)
    input_tokens: Optional[int] = Field(None, description="Input tokens used")
    output_tokens: Optional[int] = Field(None, description="Output tokens generated")
    total_tokens: Optional[int] = Field(None, description="Total tokens consumed")
    estimated_cost_usd: Optional[float] = Field(None, description="Estimated cost in USD")
    
    # Quality metrics
    confidence_score: Optional[float] = Field(
        None, 
        description="Confidence in response quality"
    )
    completeness_score: Optional[float] = Field(
        None, 
        description="How complete the answer is"
    )
    
    # Context used
    context_used: bool = Field(
        default=False, 
        description="Whether conversation history was used"
    )
    tools_used: List[str] = Field(
        default_factory=list,
        description="Tools/APIs called for this response"
    )
    
    # Timing
    generation_time_ms: Optional[float] = Field(
        None, 
        description="Time taken to generate response"
    )
    total_processing_time_ms: Optional[float] = Field(
        None, 
        description="Total time from input to response"
    )
    
    # Safety and validation
    safety_checked: bool = Field(default=False, description="Whether safety postcheck ran")
    safety_passed: bool = Field(default=True, description="Whether passed safety check")
    
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Response generation timestamp"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "response": "Your claim #12345 is currently being processed. It was submitted on January 10, 2025 and is expected to be completed within 5-7 business days.",
                "response_type": "direct_answer",
                "response_source": "llm_generated",
                "llm_model": "gemini-2.5-flash",
                "temperature": 0.7,
                "input_tokens": 234,
                "output_tokens": 89,
                "total_tokens": 323,
                "estimated_cost_usd": 0.000161,
                "confidence_score": 0.92,
                "completeness_score": 0.95,
                "context_used": True,
                "tools_used": ["claims_api"],
                "generation_time_ms": 456.2,
                "total_processing_time_ms": 892.7,
                "safety_checked": True,
                "safety_passed": True,
                "timestamp": "2025-11-10T10:30:45.123456Z"
            }
        }


# ============================================================================
# CACHE MODELS
# ============================================================================

class CacheStatus(str, Enum):
    """Cache operation status"""
    HIT = "hit"
    MISS = "miss"
    ERROR = "error"


class CacheResult(BaseModel):
    """
    Result from cache operations
    
    Tracks cache hits/misses and cached response retrieval.
    """
    cache_hit: bool = Field(..., description="Whether cache had result")
    status: CacheStatus = Field(..., description="Cache operation status")
    
    # Cached data (if hit)
    cached_response: Optional[str] = Field(None, description="Cached response text")
    cached_intent: Optional[str] = Field(None, description="Cached intent")
    cached_confidence: Optional[float] = Field(None, description="Cached confidence")
    
    # Cache metadata
    cache_key: Optional[str] = Field(None, description="Cache key used")
    cache_age_seconds: Optional[float] = Field(
        None, 
        description="Age of cached entry in seconds"
    )
    ttl_seconds: Optional[int] = Field(None, description="Time-to-live in seconds")
    
    # Performance
    retrieval_time_ms: Optional[float] = Field(
        None, 
        description="Time to retrieve from cache"
    )
    cache_backend: Optional[str] = Field(
        None, 
        description="Cache backend (redis, memorystore, inmemory)"
    )
    
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Cache check timestamp"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "cache_hit": True,
                "status": "hit",
                "cached_response": "Your claim is being processed.",
                "cached_intent": "claim_status",
                "cached_confidence": 0.89,
                "cache_key": "cache:a3f2b1c9d8e7f6",
                "cache_age_seconds": 245.3,
                "ttl_seconds": 3600,
                "retrieval_time_ms": 3.2,
                "cache_backend": "redis",
                "timestamp": "2025-11-10T10:30:45.123456Z"
            }
        }


# ============================================================================
# CONFIDENCE CHECK MODELS
# ============================================================================

class ConfidenceCheckDecision(str, Enum):
    """Decision from confidence check"""
    PROCEED = "proceed"
    CLARIFY = "clarify"
    FALLBACK = "fallback"


class ConfidenceCheckResult(BaseModel):
    """
    Result from confidence checking/routing node
    
    Determines whether to proceed with processing or ask for clarification.
    """
    decision: ConfidenceCheckDecision = Field(..., description="Routing decision")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    threshold: float = Field(..., description="Confidence threshold used")
    passed_threshold: bool = Field(..., description="Whether confidence meets threshold")
    
    # Entity completeness
    required_entities: List[str] = Field(
        default_factory=list,
        description="Entities required for this intent"
    )
    missing_entities: List[str] = Field(
        default_factory=list,
        description="Required entities that are missing"
    )
    entities_complete: bool = Field(
        default=True, 
        description="Whether all required entities present"
    )
    
    # Reasoning
    decision_reason: str = Field(..., description="Why this decision was made")
    next_node: str = Field(..., description="Next node to route to")
    
    # Metadata
    intent_checked: Optional[str] = Field(None, description="Intent being checked")
    
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Check timestamp"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "decision": "clarify",
                "confidence": 0.62,
                "threshold": 0.70,
                "passed_threshold": False,
                "required_entities": ["claim_number"],
                "missing_entities": ["claim_number"],
                "entities_complete": False,
                "decision_reason": "Confidence below threshold and missing claim_number",
                "next_node": "clarification",
                "intent_checked": "claim_rejection_reason",
                "timestamp": "2025-11-10T10:30:45.123456Z"
            }
        }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_intent_result(
    intent: str,
    confidence: float,
    **kwargs
) -> IntentResult:
    """Helper to create IntentResult with defaults"""
    return IntentResult(
        intent=intent,
        confidence=confidence,
        **kwargs
    )


def create_safety_result(
    check_type: SafetyCheckType,
    passed: bool,
    **kwargs
) -> SafetyResult:
    """Helper to create SafetyResult"""
    return SafetyResult(
        check_type=check_type,
        passed=passed,
        **kwargs
    )


def create_tool_result(
    tool_name: str,
    status: ToolExecutionStatus,
    data: Dict[str, Any],
    **kwargs
) -> ToolResult:
    """Helper to create ToolResult"""
    return ToolResult(
        tool_name=tool_name,
        status=status,
        data=data,
        **kwargs
    )


def create_response_payload(
    response: str,
    response_type: ResponseType,
    response_source: ResponseSource,
    **kwargs
) -> ResponsePayload:
    """Helper to create ResponsePayload"""
    return ResponsePayload(
        response=response,
        response_type=response_type,
        response_source=response_source,
        **kwargs
    )

