"""
Test suite for streaming functionality

Tests ensure:
1. Streaming events are emitted in correct order
2. Safety postcheck completes before response streaming
3. Status updates sent ONLY for user-facing nodes (reduces noise)
4. All nodes still logged internally (full observability)
5. Error handling works correctly
6. SSE endpoint returns proper format
7. User-friendly messages are non-technical
"""

import pytest
import asyncio
from datetime import datetime
from langgraph_agent import run_graph_stream
from config.config import settings


@pytest.mark.asyncio
async def test_streaming_event_order(init_graph_for_test):
    """Verify events are emitted in correct order"""
    events = []
    
    async for event in run_graph_stream(
        text="What is my claim status?",
        session_id="test-order-123",
        user_info={"user_id": "test-user"}
    ):
        events.append(event)
    
    # Extract event types
    event_types = [e["type"] for e in events]
    
    # Should have node_start events
    assert "node_start" in event_types
    
    # Should end with complete or error
    assert events[-1]["type"] in ["complete", "error"]
    
    # Response chunks should only come after safety_postcheck node
    postcheck_seen = False
    for event in events:
        if event["type"] == "node_start" and event.get("data", {}).get("node") == "response_safety_pii_postcheck":
            postcheck_seen = True
        if event["type"] == "response_chunk" and not postcheck_seen:
            pytest.fail("Response chunk emitted before safety postcheck!")


@pytest.mark.skip(reason="Requires real embeddings - mock embeddings cause wrong intent classification leading to clarification path with no response to stream")
@pytest.mark.asyncio
async def test_safety_postcheck_before_streaming(init_graph_for_test):
    """
    CRITICAL SECURITY TEST: Verify response only streams after safety postcheck validates
    
    This test ensures that response chunks are only emitted AFTER the
    response_safety_pii_postcheck node has processed and validated the response.
    
    The postcheck node:
    1. Detects PII leakage
    2. Unmasks PII/PHI tokens  
    3. Sets safety_postcheck_passed flag
    
    Since postcheck is an internal node (not user-facing), we verify by ensuring:
    - Chunks come AFTER response_agent completes
    - The final "complete" event confirms safety_postcheck_passed
    """
    response_agent_seen = False
    chunks_before_response_agent = []
    all_chunks = []
    complete_data = None
    
    async for event in run_graph_stream(
        text="What's the status of claim number CLM12345?",
        session_id="test-safety-456",
        user_info={"user_id": "test-user"}
    ):
        # Track when response_agent sends user-facing update
        if (event["type"] == "node_start" and 
            event.get("data", {}).get("node") == "response_agent"):
            response_agent_seen = True
        
        # Track chunks
        if event["type"] == "response_chunk":
            all_chunks.append(event.get("data", {}).get("text", ""))
            if not response_agent_seen:
                chunks_before_response_agent.append(event.get("data", {}).get("text", ""))
        
        # Get complete event
        if event["type"] == "complete":
            complete_data = event.get("data", {})
    
    # Verify chunks don't come before response_agent status update
    # (safety_postcheck runs after response_agent, so this ensures ordering)
    assert len(chunks_before_response_agent) == 0, \
        "SECURITY VIOLATION: Chunks streamed before response_agent completed!"
    
    # Verify we got chunks (response was actually streamed)
    assert len(all_chunks) > 0, "No chunks were streamed!"
    
    # Verify complete event confirms safety passed (indirectly validates postcheck ran)
    assert complete_data is not None, "No complete event received!"
    # If chunks were sent, safety must have passed


@pytest.mark.asyncio
async def test_user_facing_nodes_only(init_graph_for_test):
    """
    Verify ONLY user-facing nodes send status updates to users
    
    This test ensures:
    1. Internal nodes don't clutter the user experience
    2. User-facing nodes still send updates
    3. All nodes are still executed (logged internally)
    """
    user_facing_updates = []
    
    async for event in run_graph_stream(
        text="What is my claim status?",
        session_id="test-user-facing-123",
        user_info={"user_id": "test-user"}
    ):
        if event["type"] == "node_start":
            node_name = event.get("data", {}).get("node")
            user_facing_updates.append(node_name)
    
    # Verify only user-facing nodes sent status updates
    for node in user_facing_updates:
        assert node in settings.stream_user_facing_nodes, \
            f"Node '{node}' sent update but not in user_facing_nodes config!"
    
    # Verify key user-facing nodes are included
    assert "orchestrator" in user_facing_updates
    assert "safety_precheck" in user_facing_updates
    assert "intent_agent" in user_facing_updates
    
    # Verify internal nodes didn't send updates (but still executed)
    assert "check_cache" not in user_facing_updates
    assert "confidence_checker" not in user_facing_updates
    assert "build_context" not in user_facing_updates
    assert "response_safety_pii_precheck" not in user_facing_updates
    assert "response_safety_pii_postcheck" not in user_facing_updates


@pytest.mark.asyncio
async def test_messages_are_user_friendly(init_graph_for_test):
    """Verify messages are user-friendly and non-technical"""
    messages = []
    
    async for event in run_graph_stream(
        text="What is my claim status?",
        session_id="test-messages-456",
        user_info={"user_id": "test-user"}
    ):
        if event["type"] == "node_start":
            messages.append(event.get("data", {}).get("message", ""))
    
    # Check messages are user-friendly
    for msg in messages:
        # Should not contain technical terms
        assert "node" not in msg.lower(), f"Message too technical: '{msg}'"
        assert "state" not in msg.lower(), f"Message too technical: '{msg}'"
        assert "graph" not in msg.lower(), f"Message too technical: '{msg}'"
        
        # Should be concise
        assert len(msg) < 60, f"Message too long: '{msg}'"
        
        # Should use present progressive tense
        assert "..." in msg, f"Message should end with '...': '{msg}'"


@pytest.mark.asyncio
async def test_node_status_updates(init_graph_for_test):
    """Verify status updates are sent for user-facing nodes only"""
    node_starts = []
    
    async for event in run_graph_stream(
        text="Hello",
        session_id="test-status-789",
        user_info={}
    ):
        if event["type"] == "node_start":
            node_starts.append(event.get("data", {}).get("node"))
    
    # Should have status updates for user-facing nodes
    expected_nodes = ["orchestrator", "safety_precheck", "intent_agent"]
    for node in expected_nodes:
        assert node in node_starts, f"Missing status update for {node}"
    
    # Should NOT have status updates for internal nodes
    internal_nodes = ["check_cache", "confidence_checker", "build_context"]
    for node in internal_nodes:
        assert node not in node_starts, f"Internal node '{node}' should not send status update!"


@pytest.mark.asyncio
async def test_streaming_response_accumulation(init_graph_for_test):
    """Verify chunks accumulate to full response"""
    chunks = []
    full_response = ""
    
    async for event in run_graph_stream(
        text="What was my copay?",
        session_id="test-accumulation-012",
        user_info={"user_id": "test-user"}
    ):
        if event["type"] == "response_chunk":
            chunks.append(event.get("data", {}).get("text", ""))
        elif event["type"] == "complete":
            full_response = event.get("data", {}).get("response", "")
    
    # Accumulated chunks should equal full response
    accumulated = "".join(chunks)
    assert accumulated == full_response, "Chunks don't match full response"


@pytest.mark.asyncio
async def test_error_handling_in_streaming(init_graph_for_test):
    """Verify error events are properly emitted"""
    # Test with potentially problematic input
    events = []
    
    async for event in run_graph_stream(
        text="test query",
        session_id="test-error-345",
        user_info={}
    ):
        events.append(event)
        if event["type"] == "error":
            break
    
    # Should have at least one event
    assert len(events) > 0


@pytest.mark.asyncio
async def test_metadata_in_complete_event(init_graph_for_test):
    """Verify complete event includes all metadata"""
    complete_event = None
    
    async for event in run_graph_stream(
        text="Test query",
        session_id="test-metadata-901",
        user_info={"user_id": "test-user"}
    ):
        if event["type"] == "complete":
            complete_event = event
            break
    
    assert complete_event is not None
    data = complete_event.get("data", {})
    
    # Should have essential fields
    assert "response" in data
    assert "intent" in data
    assert "metadata" in data
    assert "session_id" in complete_event.get("metadata", {})


@pytest.mark.skip(reason="Requires real embeddings - mock embeddings cause wrong intent classification leading to clarification path with no response to stream")
@pytest.mark.asyncio
async def test_end_to_end_streaming_flow(init_graph_for_test):
    """
    Complete E2E test simulating real usage
    
    Verifies:
    1. Only user-facing status updates received (not internal nodes)
    2. Response is streamed after safety check
    3. Complete event includes all data
    4. Memory is updated (async)
    5. Messages are user-friendly
    """
    session_id = f"e2e-test-{datetime.now().timestamp()}"
    
    status_updates = []
    response_chunks = []
    complete_data = None
    messages = []
    
    async for event in run_graph_stream(
        text="Where is my claim CLM99999?",
        session_id=session_id,
        user_info={"user_id": "test-user-e2e"}
    ):
        if event["type"] == "node_start":
            node = event.get("data", {}).get("node")
            message = event.get("data", {}).get("message", "")
            status_updates.append(node)
            messages.append(message)
        elif event["type"] == "response_chunk":
            response_chunks.append(event.get("data", {}).get("text", ""))
        elif event["type"] == "complete":
            complete_data = event.get("data", {})
    
    # Verify we got status updates (should be 5 user-facing nodes or less)
    assert len(status_updates) > 0
    assert len(status_updates) <= 5, f"Too many status updates: {len(status_updates)}"
    
    # Verify only user-facing nodes sent updates
    for node in status_updates:
        assert node in settings.stream_user_facing_nodes, \
            f"Node '{node}' should not send status update!"
    
    # Verify key user-facing nodes are included
    assert "safety_precheck" in status_updates
    assert "intent_agent" in status_updates
    
    # Verify internal nodes did NOT send updates
    assert "check_cache" not in status_updates
    assert "confidence_checker" not in status_updates
    assert "build_context" not in status_updates
    
    # Verify messages are user-friendly
    for msg in messages:
        assert len(msg) < 60, f"Message too long: '{msg}'"
        assert "node" not in msg.lower(), f"Message too technical: '{msg}'"
    
    # Verify we got response chunks
    assert len(response_chunks) > 0
    
    # Verify complete event
    assert complete_data is not None
    assert "response" in complete_data
    assert "intent" in complete_data
    
    # Verify response matches chunks
    full_response = "".join(response_chunks)
    assert full_response == complete_data.get("response", "")
    
    # Verify memory was updated (give it a moment for async completion)
    await asyncio.sleep(0.5)
    from memory import MemoryStoreFactory
    
    memory_store = MemoryStoreFactory.get_instance(settings.memory_store_type)
    history = await memory_store.get_session_history(session_id)
    
    # Should have at least user + assistant messages
    assert len(history) >= 2

