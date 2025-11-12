"""
Generic Serialization Helpers Usage Examples

Demonstrates the new TypeVar-based generic serialization helpers
that work with ANY Pydantic model.
"""

from core.error_models import (
    AgentError,
    ErrorCode,
    ErrorCategory,
    ErrorSeverity,
    ErrorResponse,
    create_validation_error,
)

from core.node_models import (
    IntentResult,
    ToolResult,
    ToolExecutionStatus,
    ResponsePayload,
    ResponseType,
    ResponseSource,
    ConversationMessage,
    SessionFact,
    ContextResult,
)

# Import generic serialization helpers from utils
from utils.serialization import (
    to_dict,
    from_dict,
    to_json,
    from_json,
    copy_model,
    to_dict_list,
    from_dict_list,
)


# ============================================================================
# Example 1: Generic to_dict() - Works with ANY model
# ============================================================================

def example_generic_to_dict():
    """Example: Using generic to_dict() with different model types"""
    # Works with AgentError
    error = create_validation_error(
        message="Invalid input",
        field="email",
        session_id="session-123"
    )
    error_dict = to_dict(error)
    print(f"✅ Error as dict: {error_dict['error_code']}")
    
    # Works with IntentResult
    intent = IntentResult(
        intent="claim_status",
        confidence=0.85
    )
    intent_dict = to_dict(intent)
    print(f"✅ Intent as dict: {intent_dict['intent']}")
    
    # Works with ToolResult
    tool = ToolResult(
        tool_name="claims_api",
        status=ToolExecutionStatus.SUCCESS,
        data={"claim_id": "CLM-12345"}
    )
    tool_dict = to_dict(tool)
    print(f"✅ Tool as dict: {tool_dict['tool_name']}")
    
    # Works with ResponsePayload
    response = ResponsePayload(
        response="Your claim is approved!",
        response_type=ResponseType.DIRECT_ANSWER,
        response_source=ResponseSource.LLM_GENERATED
    )
    response_dict = to_dict(response)
    print(f"✅ Response as dict: {response_dict['response_type']}")


# ============================================================================
# Example 2: Generic from_dict() - Create any model from dict
# ============================================================================

def example_generic_from_dict():
    """Example: Using generic from_dict() with type parameter"""
    # Create models from dictionaries
    
    # Error from dict
    error_dict = {
        "error_code": "E1001",
        "category": "validation",
        "severity": "low",
        "message": "Test error",
        "user_message": "Please fix this"
    }
    error = from_dict(AgentError, error_dict)
    print(f"✅ Created AgentError: {error.error_code}")
    
    # Intent from dict
    intent_dict = {
        "intent": "claim_status",
        "confidence": 0.92
    }
    intent = from_dict(IntentResult, intent_dict)
    print(f"✅ Created IntentResult: {intent.intent}")
    
    # Tool from dict
    tool_dict = {
        "tool_name": "claims_api",
        "status": "success",
        "data": {"result": "approved"}
    }
    tool = from_dict(ToolResult, tool_dict)
    print(f"✅ Created ToolResult: {tool.tool_name}")


# ============================================================================
# Example 3: Generic copy_model() - Copy any model with updates
# ============================================================================

def example_generic_copy_model():
    """Example: Using generic copy_model() for any Pydantic model"""
    # Copy error with updates
    original_error = AgentError(
        error_code=ErrorCode.API_TIMEOUT,
        category=ErrorCategory.API_CALL,
        severity=ErrorSeverity.HIGH,
        message="API timeout",
        user_message="Request timed out",
        session_id="session-123"
    )
    
    new_error = copy_model(
        original_error,
        session_id="session-456",
        retry_after_seconds=30
    )
    print(f"✅ Original session: {original_error.session_id}")
    print(f"✅ New session: {new_error.session_id}")
    print(f"✅ New retry_after: {new_error.retry_after_seconds}")
    
    # Copy intent with confidence update
    original_intent = IntentResult(
        intent="claim_status",
        confidence=0.75
    )
    
    updated_intent = copy_model(original_intent, confidence=0.92)
    print(f"✅ Original confidence: {original_intent.confidence}")
    print(f"✅ Updated confidence: {updated_intent.confidence}")
    
    # Copy tool result with retry info
    original_tool = ToolResult(
        tool_name="claims_api",
        status=ToolExecutionStatus.SUCCESS,
        data={"claim_id": "CLM-12345"}
    )
    
    retry_tool = copy_model(original_tool, retry_count=1)
    print(f"✅ Original retry_count: {original_tool.retry_count}")
    print(f"✅ New retry_count: {retry_tool.retry_count}")


# ============================================================================
# Example 4: Generic to_dict_list() and from_dict_list()
# ============================================================================

def example_generic_list_operations():
    """Example: Generic list operations work with any model type"""
    # Work with list of errors
    errors = [
        create_validation_error(
            message="Field required",
            field="email",
            session_id=f"session-{i}"
        )
        for i in range(3)
    ]
    
    error_dicts = to_dict_list(errors)
    print(f"✅ Converted {len(error_dicts)} errors to dicts")
    
    restored_errors = from_dict_list(AgentError, error_dicts)
    print(f"✅ Restored {len(restored_errors)} errors from dicts")
    
    # Work with list of intents
    intents = [
        IntentResult(intent="claim_status", confidence=0.85),
        IntentResult(intent="refill_info", confidence=0.92),
        IntentResult(intent="member_info", confidence=0.78),
    ]
    
    intent_dicts = to_dict_list(intents)
    print(f"✅ Converted {len(intent_dicts)} intents to dicts")
    
    restored_intents = from_dict_list(IntentResult, intent_dicts)
    print(f"✅ Restored {len(restored_intents)} intents from dicts")
    
    # Work with conversation messages
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
    
    message_dicts = to_dict_list(messages)
    print(f"✅ Converted {len(message_dicts)} messages to dicts")
    
    restored_messages = from_dict_list(ConversationMessage, message_dicts)
    print(f"✅ Restored {len(restored_messages)} messages from dicts")


# ============================================================================
# Example 5: JSON Operations
# ============================================================================

def example_json_operations():
    """Example: Generic JSON serialization"""
    # Error to JSON
    error = create_validation_error(
        message="Invalid claim number",
        field="claim_number",
        session_id="session-123"
    )
    
    error_json = to_json(error)
    print(f"✅ Error as JSON (length): {len(error_json)}")
    
    restored_error = from_json(AgentError, error_json)
    print(f"✅ Restored error: {restored_error.error_code}")
    
    # Intent to JSON
    intent = IntentResult(
        intent="claim_status",
        confidence=0.89,
        classification_method="keyword_matching"
    )
    
    intent_json = to_json(intent)
    print(f"✅ Intent as JSON (length): {len(intent_json)}")
    
    restored_intent = from_json(IntentResult, intent_json)
    print(f"✅ Restored intent: {restored_intent.intent}")


# ============================================================================
# Example 6: Mixed Model Types in One Workflow
# ============================================================================

def example_mixed_workflow():
    """Example: Working with multiple model types using same generic helpers"""
    # Step 1: Create and serialize intent
    intent = IntentResult(
        intent="claim_status",
        confidence=0.92
    )
    
    state_metadata = {
        "intent": to_dict(intent),  # Generic helper
        "timestamp": "2025-11-10T10:30:00Z"
    }
    
    # Step 2: Create and add tool result
    tool = ToolResult(
        tool_name="claims_api",
        status=ToolExecutionStatus.SUCCESS,
        data={"claim_id": "CLM-12345", "status": "approved"}
    )
    
    state_metadata["tool"] = to_dict(tool)  # Same generic helper
    
    # Step 3: Create and add response
    response = ResponsePayload(
        response="Your claim has been approved!",
        response_type=ResponseType.DIRECT_ANSWER,
        response_source=ResponseSource.LLM_GENERATED,
        tools_used=["claims_api"]
    )
    
    state_metadata["response"] = to_dict(response)  # Same generic helper
    
    # Step 4: Later, restore all from metadata
    restored_intent = from_dict(IntentResult, state_metadata["intent"])
    restored_tool = from_dict(ToolResult, state_metadata["tool"])
    restored_response = from_dict(ResponsePayload, state_metadata["response"])
    
    print(f"✅ Restored intent: {restored_intent.intent}")
    print(f"✅ Restored tool: {restored_tool.tool_name}")
    print(f"✅ Restored response type: {restored_response.response_type}")


# ============================================================================
# Example 7: Type Safety Demonstration
# ============================================================================

def example_type_safety():
    """Example: Generic helpers provide type safety"""
    # IDE will infer correct return type
    
    # to_dict() returns Dict[str, Any]
    intent = IntentResult(intent="test", confidence=0.8)
    intent_dict = to_dict(intent)  # Type: Dict[str, Any]
    
    # from_dict() returns the model type specified
    restored: IntentResult = from_dict(IntentResult, intent_dict)  # Type: IntentResult
    
    # copy_model() returns same type as input
    new_intent: IntentResult = copy_model(intent, confidence=0.9)  # Type: IntentResult
    
    # to_dict_list() works with any list
    intents = [intent, new_intent]
    dicts = to_dict_list(intents)  # Type: List[Dict[str, Any]]
    
    # from_dict_list() returns typed list
    restored_list: list[IntentResult] = from_dict_list(IntentResult, dicts)  # Type: List[IntentResult]
    
    print(f"✅ Type safety: All operations are type-safe!")


# ============================================================================
# Example 8: Real-World Scenario - Caching
# ============================================================================

def example_caching_scenario():
    """Example: Using generic helpers for caching"""
    # Simulated cache operations
    
    # Cache intent result
    intent = IntentResult(
        intent="claim_status",
        confidence=0.89,
        classification_method="keyword_matching"
    )
    
    cache_key = "intent:session-123"
    cache_value = to_dict(intent)
    # await cache.set(cache_key, cache_value, ttl=3600)
    print(f"✅ Cached intent with key: {cache_key}")
    
    # Later, retrieve from cache
    # cached_value = await cache.get(cache_key)
    cached_value = cache_value  # Simulated
    restored_intent = from_dict(IntentResult, cached_value)
    print(f"✅ Retrieved intent from cache: {restored_intent.intent}")
    
    # Cache conversation history
    messages = [
        ConversationMessage(role="user", content="Hello"),
        ConversationMessage(role="assistant", content="Hi there!"),
    ]
    
    history_key = "history:session-123"
    history_value = to_dict_list(messages)
    # await cache.set(history_key, history_value, ttl=3600)
    print(f"✅ Cached {len(history_value)} messages")
    
    # Retrieve history
    # cached_history = await cache.get(history_key)
    cached_history = history_value  # Simulated
    restored_messages = from_dict_list(ConversationMessage, cached_history)
    print(f"✅ Retrieved {len(restored_messages)} messages from cache")


# ============================================================================
# Example 9: Benefits Over Type-Specific Helpers
# ============================================================================

def example_benefits():
    """Example: Why generic helpers are better"""
    print("\n" + "=" * 70)
    print("BENEFITS OF GENERIC HELPERS")
    print("=" * 70)
    
    print("\n✅ 1. ONE set of functions for ALL model types")
    print("   - to_dict() works with Error, Intent, Tool, Response, etc.")
    print("   - No need to remember intent_to_dict, tool_to_dict, etc.")
    
    print("\n✅ 2. Consistent API across all models")
    print("   - Same pattern: to_dict(model)")
    print("   - Same pattern: from_dict(ModelClass, data)")
    
    print("\n✅ 3. Type safety with TypeVar")
    print("   - Editor autocomplete works perfectly")
    print("   - Return types are correctly inferred")
    
    print("\n✅ 4. Less code to maintain")
    print("   - 7 generic functions vs 40+ specific functions")
    print("   - Legacy functions still work (backward compatible)")
    
    print("\n✅ 5. Easy to extend")
    print("   - New models automatically work with generic helpers")
    print("   - No need to add new helper functions")


# ============================================================================
# Run All Examples
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("GENERIC SERIALIZATION HELPERS - EXAMPLES")
    print("=" * 70)
    
    print("\n1. Generic to_dict() with different model types:")
    print("-" * 70)
    example_generic_to_dict()
    
    print("\n2. Generic from_dict() with type parameter:")
    print("-" * 70)
    example_generic_from_dict()
    
    print("\n3. Generic copy_model() for any model:")
    print("-" * 70)
    example_generic_copy_model()
    
    print("\n4. Generic list operations:")
    print("-" * 70)
    example_generic_list_operations()
    
    print("\n5. JSON operations:")
    print("-" * 70)
    example_json_operations()
    
    print("\n6. Mixed workflow with multiple models:")
    print("-" * 70)
    example_mixed_workflow()
    
    print("\n7. Type safety demonstration:")
    print("-" * 70)
    example_type_safety()
    
    print("\n8. Real-world caching scenario:")
    print("-" * 70)
    example_caching_scenario()
    
    print("\n9. Benefits explanation:")
    print("-" * 70)
    example_benefits()
    
    print("\n" + "=" * 70)
    print("ALL EXAMPLES COMPLETED!")
    print("=" * 70)
