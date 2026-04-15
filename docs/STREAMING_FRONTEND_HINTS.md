# Streaming API - Frontend Integration Guide

**Angular Implementation Reference**

> 📚 **Looking for backend/architecture docs?** See `STREAMING_DOCUMENTATION.md` for technical details, testing, and backend implementation. This guide focuses on frontend integration patterns.

---

## 🎯 Understanding SSE Rendering Patterns

### **How Server-Sent Events Work**

The backend streams events using the SSE protocol (RFC 6455). Each event is independent and sequential:

```
event: node_start
data: {"node": "orchestrator", "message": "Processing your request..."}

event: node_start
data: {"node": "safety_precheck", "message": "Checking safety and privacy..."}

event: response_chunk
data: {"text": "Your claim status...", "chunk_index": 0}
```

### **Two Key Rendering Strategies**

To create a clean user experience, use two different rendering patterns for different event types:

| Event Type | Rendering Pattern | Implementation |
|------------|------------------|----------------|
| `node_start` | **REPLACE** | Show only current processing step |
| `response_chunk` | **APPEND** | Build complete response progressively |

### **Visual Example**

**Good UX (Recommended):**
```
┌────────────────────────────────────────────────┐
│  [Status: Understanding your question...]      │  ← Status replaces each time
├────────────────────────────────────────────────┤
│  Based on your claim 847293156420183,          │  ← Response accumulates
│  sequence 1 shows a copay of $25 for...        │
└────────────────────────────────────────────────┘
```

**Poor UX (Avoid):**
```
Processing your request...
Checking safety and privacy...
Understanding your question...
Retrieving your claims information...
Preparing your response...
Based on your claim...
```
☝️ Stacking all status messages creates visual noise

### **Angular Implementation Pattern**

```typescript
// Component State
statusMessage: string = '';      // Single value - gets replaced
currentResponse: string = '';    // Accumulated string - gets appended

// Event Handler
handleStreamEvent(event: StreamEvent) {
  if (event.type === 'node_start') {
    // REPLACE pattern: assignment operator overwrites
    this.statusMessage = event.data.message;
  }
  
  if (event.type === 'response_chunk') {
    // APPEND pattern: concatenation operator accumulates
    this.currentResponse += event.data.text;
    
    // Hide status once response starts
    if (event.data.chunk_index === 0) {
      this.statusMessage = '';
    }
  }
}
```

```html
<!-- Template: Two separate UI elements -->
<div *ngIf="statusMessage" class="status-indicator">
  {{ statusMessage }}
</div>

<div *ngIf="currentResponse" class="response-display">
  {{ currentResponse }}
</div>
```

**Why This Works:** Using two separate UI elements with different update strategies (replace vs append) provides the optimal user experience seen in modern AI chat applications.

---

## Quick Summary

**Streaming API Features:**
- Receive **5 key status updates** during processing (cleaner UX than showing all internal steps)
- User-friendly, non-technical status messages
- Progressive response chunks for "typing" effect
- All backend processing still executes normally (internal steps are logged but not streamed)

**The 5 Status Updates You'll Receive:**
1. "Processing your request..." (orchestrator)
2. "Checking safety and privacy..." (safety_precheck)
3. "Understanding your question..." (intent_agent)
4. "Retrieving your claims information..." (call_claims_tool)
5. "Preparing your response..." (response_agent)

**Internal Processing Steps (Not Shown to Frontend):**
- Cache checks, confidence analysis, context building, PII masking/unmasking, memory updates, response caching

**Event Flow:**
Status updates → Response chunks → Completion event

---

## Overview

The `/api/v1/chat/stream` endpoint provides **real-time streaming** of the AI agent's responses using **Server-Sent Events (SSE)**. Instead of waiting for the entire response to be generated (which can take 10-15 seconds), the streaming endpoint provides:

1. **Progressive status updates** for **5 key milestones** (not all internal processing steps)
2. **Incremental response chunks** that display text as it's being generated
3. **Better user experience** with 50-70% reduction in perceived latency

**What's New in V2.0:**
- ✅ Only **5 significant status updates** shown (down from 10+)
- ✅ More user-friendly messages matching industry standards
- ✅ Clearer narrative: "Processing → Understanding → Fetching → Generating"
- ✅ All internal processing still happens, just not shown to users

Think of it like watching a document being typed in real-time rather than waiting for the entire document to be completed before seeing anything.

## Why We Chose Server-Sent Events (SSE)?

SSE is the standard technology for server-to-client streaming and provides several advantages over alternatives:

| Feature | SSE (Our Choice) | WebSockets | Long Polling |
|---------|------------------|------------|--------------|
| Complexity | Simple HTTP POST | Complex protocol | High overhead |
| Browser Support | Native EventSource API | Needs library | Manual implementation |
| Reconnection | Automatic | Manual | Manual |
| Direction | Server → Client | Bidirectional | Both |
| Firewall Issues | None (uses HTTP) | Sometimes blocked | None |
| Use Case Fit | ✅ Perfect for streaming responses | ❌ Overkill for one-way | ❌ Inefficient |

**For frontend developers:** You can use the standard Fetch API to consume SSE streams - no special libraries needed!

---

## Endpoint Details

### Base URL

**POST** `/api/v1/chat/stream`

Full example: `http://your-backend-domain.com/api/v1/chat/stream`

---

## Input Structure (Request Body)

### Complete Example

```json
{
  "text": "What are the details for claim 847293156420183 sequence 1?",
  "session_id": "sess-user-12345-20241124",
  "user_info": {
    "user": "khk",
    "user_id": "12345",
    "member_id": "M987654"
  }
}
```

### Field Explanations

| Field | Type | Required | Description | Example |
|-------|------|----------|-------------|---------|
| `text` | string | **Yes** | The user's question or query. This is what the AI will process and respond to. | `"What are my claim details?"` |
| `session_id` | string | **Recommended** | Unique identifier for the conversation session. Maintains context across multiple messages in the same conversation. If omitted, a new session is created. | `"sess-user-12345-20241124"` |
| `user_info` | object | **Recommended** | Contains user metadata. While optional, it's highly recommended for personalization and logging. | `{"user": "khk", "user_id": "12345"}` |
| `user_info.user` | string | Optional | Username or user identifier for display | `"khk"` |
| `user_info.user_id` | string | Optional | Internal user ID for tracking and analytics | `"12345"` |
| `user_info.member_id` | string | Optional | Insurance member ID if applicable | `"M987654"` |

### Important Notes for Frontend Implementation

1. **Session Management**: Generate a unique `session_id` when the user starts a new conversation. Reuse the same `session_id` for follow-up questions in the same conversation to maintain context.

2. **User Context**: Always include `user_info` when available. The backend uses this for:
   - Personalizing responses
   - Logging and audit trails
   - Analytics and usage tracking
   - PII/PHI protection validation

3. **Text Input**: The `text` field accepts natural language queries. No special formatting required - users can ask questions as they normally would.

---

## Output Structure (Response Stream)

### Response Format

**Content-Type:** `text/event-stream`

**Key Headers:**
- `Cache-Control: no-cache` - Prevents caching of the stream
- `Connection: keep-alive` - Keeps connection open for streaming
- `X-Accel-Buffering: no` - Disables proxy buffering
- `Access-Control-Allow-Origin: *` - Allows cross-origin requests (configure for production)

### SSE Event Format

Each event in the stream follows this structure:

```
event: <event_type>
data: <json_payload>

```

**Note:** There's a blank line after each event - this is part of the SSE specification.

---

## Understanding the Complete Event Flow

### ⚡ What You'll See (V2.0 - Simplified)

When you send a request, you'll receive **5 key status updates** that tell a clear story:

```
event: node_start
data: {"node": "orchestrator", "message": "Processing your request..."}

event: node_start
data: {"node": "safety_precheck", "message": "Checking safety and privacy..."}

event: node_start
data: {"node": "intent_agent", "message": "Understanding your question..."}

event: node_start
data: {"node": "call_claims_tool", "message": "Retrieving your claims information..."}

event: node_start
data: {"node": "response_agent", "message": "Preparing your response..."}

event: response_chunk
data: {"text": "Based on your claim 847293156420183, ", "chunk_index": 0, "total_length": 250}

event: response_chunk
data: {"text": "the sequence 1 details are as follows...", "chunk_index": 1, "total_length": 250}

event: complete
data: {"response": "Based on your claim 847293156420183...", "intent": "claim_details", "confidence": 0.95, "needs_clarification": false, "metadata": {...}}
```

### 🤔 What Happened to the Other Nodes?

**Short answer:** They're still running, just not shown to you!

The backend executes many internal processing steps (cache checks, confidence analysis, context building, safety validation, etc.) but only shows you the **5 significant milestones** that matter from a user experience perspective.

**Why only 5 status updates?**
- ✅ **Reduces noise** - Users don't need to see every internal step
- ✅ **Clearer story** - "Received → Understanding → Fetching → Generating"
- ✅ **Industry standard** - Matches ChatGPT, Claude, Copilot patterns
- ✅ **Better UX** - Less overwhelming for users

**All internal nodes still:**
- ✅ Execute normally (no changes to processing)
- ✅ Get logged for debugging
- ✅ Tracked in telemetry
- ✅ Monitored for errors

You just don't see status updates for internal optimization and security steps.

### What This Means for Your UI

Think of the event flow in three phases:

**Phase 1: Processing Updates (5 node_start events)**
- Show these as status indicators to the user
- Example: Display "Checking safety and privacy..." while safety check runs
- Clear narrative: Processing → Privacy → Understanding → Fetching → Generating

**Phase 2: Response Streaming (response_chunk events)** 
- Display these chunks progressively to show the response being "typed out"
- Append each chunk to build the complete response
- This is where the "streaming" magic happens

**Phase 3: Completion (complete event)**
- Final event with full response and metadata
- Use metadata for analytics, logging, confidence scores
- Hide status indicators and show final response

---

## Event Types - Detailed Explanation

The streaming endpoint emits 5 different event types. Understanding each type is crucial for proper frontend integration.

### 1. `node_start` - Processing Stage Begins

**When it's emitted:** Every time the backend starts processing a new stage (node) in the AI pipeline.

**What it means for your UI:** This is your cue to update the status indicator to show users what's happening. The backend is actively working on their request.

**Real Example:**

```
event: node_start
data: {"node": "safety_precheck", "message": "Checking safety and privacy..."}
```

**Data Structure:**

```typescript
{
  node: string;           // Technical node name (for debugging/logging)
  message: string;        // User-friendly message (display this to users!)
}
```

**Frontend Integration Tips:**

```typescript
if (event.type === 'node_start') {
  // Update your status indicator
  this.statusMessage = event.data.message;
  
  // Example: Show a spinner or progress indicator
  this.showLoadingSpinner(event.data.message);
  
  // Optional: Track for analytics
  this.analytics.track('processing_stage', { node: event.data.node });
}
```

**The 5 Key Status Updates You'll See:**

| Node Name | User-Friendly Message | What's Happening |
|-----------|----------------------|------------------|
| `orchestrator` | "Processing your request..." | Request received and being initialized |
| `safety_precheck` | "Checking safety and privacy..." | Input being scanned for security and PII violations (builds trust) |
| `intent_agent` | "Understanding your question..." | AI is comprehending what type of question this is |
| `call_claims_tool` | "Retrieving your claims information..." | Fetching your data from external APIs (most visible to users) |
| `response_agent` | "Preparing your response..." | AI is generating the final answer |

**Internal Nodes (Not Shown but Still Running):**
- `check_cache` - Internal optimization for faster responses
- `confidence_checker` - Internal routing logic
- `build_context` - Internal context building from history
- `response_safety_pii_precheck` - Internal security (PII masking)
- `response_safety_pii_postcheck` - Internal security (PII unmasking)
- `update_memory` - Internal conversation storage
- `cache_response` - Internal caching

---

### 2. `node_complete` - Processing Stage Finished

**When it's emitted:** Immediately after each processing stage completes successfully.

**What it means for your UI:** The current processing stage is done. If you're building a progress bar or step indicator, this is when you mark that step as complete.

**Real Example:**

```
event: node_complete
data: {"node": "safety_precheck"}
```

**Data Structure:**

```typescript
{
  node: string;           // Name of the node that just completed
}
```

**Frontend Integration Tips:**

```typescript
if (event.type === 'node_complete') {
  // Optional: Mark stage as complete in UI
  this.markStageComplete(event.data.node);
  
  // Optional: For progress tracking
  this.completedStages.push(event.data.node);
  this.updateProgressBar();
}
```

**Note:** You don't typically need to show `node_complete` events to users. They're more useful for debugging and building progress indicators.

---

### 3. `response_chunk` - Actual Response Text (Streaming!)

**When it's emitted:** After the AI generates the response AND after it passes final safety validation. The complete response is split into ~50 character chunks and streamed to you.

**What it means for your UI:** THIS IS THE MAIN CONTENT! Display these chunks progressively to create the "typing" effect.

**Real Example:**

```
event: response_chunk
data: {"text": "Based on your claim 847293156420183, ", "chunk_index": 0, "total_length": 250}

event: response_chunk
data: {"text": "the details for sequence 1 show a cop", "chunk_index": 1, "total_length": 250}

event: response_chunk
data: {"text": "ay of $25 for your prescription...", "chunk_index": 2, "total_length": 250}
```

**Data Structure:**

```typescript
{
  text: string;           // A piece of the response (append this to your display)
  chunk_index: number;    // Sequential index starting from 0
  total_length: number;   // Total character count of the complete response
}
```

**Frontend Integration Tips:**

```typescript
// Initialize at the start of the request
this.currentResponse = '';

// On each response_chunk event
if (event.type === 'response_chunk') {
  // Append chunk to build the full response
  this.currentResponse += event.data.text;
  
  // Optional: Calculate progress percentage
  const progress = (this.currentResponse.length / event.data.total_length) * 100;
  this.updateProgressBar(progress);
  
  // Optional: Auto-scroll to show new text
  this.scrollToBottom();
  
  // Clear status message once first chunk arrives
  if (event.data.chunk_index === 0) {
    this.statusMessage = '';
  }
}
```

**Critical Security Note:** 

⚠️ Response chunks are ONLY emitted AFTER the final safety/PII postcheck completes. This ensures:
- No PII/PHI leakage
- All masked tokens (like `[PII_PERSON_1]`) are properly unmasked
- Response meets HIPAA compliance requirements

You will NEVER receive chunks with masked tokens like `[PII_PERSON_1]` - if you do, that's a bug!

---

### 4. `complete` - Everything is Done

**When it's emitted:** After all response chunks have been sent and the entire request is complete.

**What it means for your UI:** This is the final event. Use it to:
1. Confirm the response is fully delivered
2. Extract metadata for analytics
3. Handle any post-processing (logging, storing conversation, etc.)

**Real Example:**

```
event: complete
data: {
  "response": "Based on your claim 847293156420183, the details for sequence 1 show a copay of $25 for your prescription...",
  "intent": "claim_details",
  "confidence": 0.95,
  "needs_clarification": false,
  "clarifying_question": null,
  "metadata": {
    "duration_ms": 3240,
    "user_id": "user-123",
    "streaming": true,
    "session_id": "sess-user-12345"
  }
}
```

**Data Structure:**

```typescript
{
  response: string;                  // Complete response text (same as all chunks combined)
  intent: string | null;             // What type of question was this? (e.g., "claim_details", "rx_details")
  confidence: number | null;         // How confident was the AI? (0.0 to 1.0, higher is better)
  needs_clarification: boolean;      // Does the AI need more info from the user?
  clarifying_question: string | null; // If needs_clarification is true, this is the follow-up question
  metadata: {
    duration_ms: number;             // Total processing time in milliseconds
    user_id: string;                 // User identifier from your request
    streaming: boolean;              // Always true for this endpoint
    session_id: string;              // Session ID from your request
    [key: string]: any;              // Additional metadata fields
  }
}
```

**Frontend Integration Tips:**

```typescript
if (event.type === 'complete') {
  // Stop showing loading indicators
  this.isStreaming = false;
  this.statusMessage = '';
  
  // Store the complete message in conversation history
  this.conversationHistory.push({
    role: 'assistant',
    content: event.data.response,
    intent: event.data.intent,
    confidence: event.data.confidence,
    timestamp: new Date()
  });
  
  // Analytics tracking
  this.analytics.track('response_complete', {
    intent: event.data.intent,
    confidence: event.data.confidence,
    duration: event.data.metadata.duration_ms,
    session_id: event.data.metadata.session_id
  });
  
  // If clarification is needed, show the clarifying question
  if (event.data.needs_clarification && event.data.clarifying_question) {
    this.showClarificationPrompt(event.data.clarifying_question);
  }
  
  // Optional: Show confidence indicator to user if confidence is low
  if (event.data.confidence && event.data.confidence < 0.7) {
    this.showLowConfidenceWarning();
  }
}
```

**Understanding the Metadata:**

- **duration_ms**: Total time from request to completion. Useful for performance monitoring.
- **intent**: Helps you understand what the user was asking about. Common values:
  - `claim_status` - User asking about claim status
  - `claim_details` - User asking for detailed claim information
  - `rx_details` - User asking about prescriptions
  - `coverage_inquiry` - User asking about coverage
  - `general_greeting` - Simple greetings or small talk
  - `out_of_scope` - Question outside the agent's knowledge
- **confidence**: Values typically range from 0.65 to 0.99. Lower confidence (<0.7) might mean the agent wasn't sure.

---

### 5. `error` - Something Went Wrong

**When it's emitted:** When an error occurs at any point during processing.

**What it means for your UI:** Show an error message to the user and stop showing loading indicators.

**Common Error Examples:**

**Safety Violation:**
```
event: error
data: {
  "message": "I apologize, but I cannot display that information due to privacy protection. Please try rephrasing your question.",
  "reason": "safety_violation"
}
```

**General Error:**
```
event: error
data: {
  "message": "An error occurred while processing your request. Please try again.",
  "reason": "internal_error"
}
```

**Data Structure:**

```typescript
{
  message: string;        // User-friendly error message (display this to users)
  reason?: string;        // Error category for debugging/logging
}
```

**Frontend Integration Tips:**

```typescript
if (event.type === 'error') {
  // Stop all loading indicators
  this.isStreaming = false;
  this.statusMessage = '';
  this.currentResponse = '';
  
  // Show error to user
  this.showErrorMessage(event.data.message);
  
  // Log for debugging
  console.error('Stream error:', event.data);
  this.analytics.trackError({
    message: event.data.message,
    reason: event.data.reason,
    session_id: this.sessionId
  });
  
  // Specific handling based on error reason
  if (event.data.reason === 'safety_violation') {
    // User tried to do something that violates safety policies
    this.showSafetyViolationUI();
  } else {
    // Generic error - offer retry
    this.showRetryOption();
  }
}
```

**Common Error Reasons:**

| Reason | Meaning | User Action |
|--------|---------|-------------|
| `safety_violation` | Content violated safety/privacy policies | Rephrase question or remove sensitive info |
| `internal_error` | Backend encountered an error | Retry the request |
| `timeout` | Request took too long | Try a simpler question or retry |
| `invalid_request` | Request format was invalid | Check request structure |

---

---

## Frontend Integration Guide

### Integration Steps Overview

1. **Create a service** to handle SSE streaming connection
2. **Parse SSE events** from the stream (event type + JSON data)
3. **Update UI components** based on event types
4. **Handle errors** and connection issues gracefully
5. **Manage session state** across multiple messages

---

## Angular Integration

### Step 1: Create the Streaming Service

Create `services/chat-stream.service.ts`:

```typescript
// chat-stream.service.ts
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';

// Interface for typed events
export interface StreamEvent {
  type: 'node_start' | 'node_complete' | 'response_chunk' | 'complete' | 'error';
  data: any;
}

@Injectable({
  providedIn: 'root'
})
export class ChatStreamService {
  private baseUrl = 'http://your-backend-domain.com'; // Update this!
  
  /**
   * Stream chat responses using Server-Sent Events
   * 
   * @param message - User's question
   * @param sessionId - Conversation session ID
   * @param userInfo - Additional user context
   * @returns Observable of StreamEvent objects
   */
  streamChat(
    message: string, 
    sessionId: string,
    userInfo?: { user?: string; user_id?: string; member_id?: string }
  ): Observable<StreamEvent> {
    return new Observable(observer => {
      // Build request payload
      const payload = {
        text: message,
        session_id: sessionId,
        user_info: userInfo || { user_id: this.getCurrentUserId() }
      };
      
      // Start fetch request
      fetch(`${this.baseUrl}/api/v1/chat/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload)
      })
      .then(response => {
        // Check for HTTP errors
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        // Get reader for response body stream
        const reader = response.body!.getReader();
        const decoder = new TextDecoder();
        
        let buffer = '';
        
        // Process stream asynchronously
        const processStream = async () => {
          try {
            while (true) {
              const { done, value } = await reader.read();
              
              if (done) {
                observer.complete();
                break;
              }
              
              // Decode bytes to text and add to buffer
              buffer += decoder.decode(value, { stream: true });
              
              // Split buffer by newlines
              const lines = buffer.split('\n');
              
              // Keep incomplete line in buffer
              buffer = lines.pop() || '';
              
              // Track current event type
              let currentEvent = 'message';
              
              // Parse each line
              for (const line of lines) {
                if (line.startsWith('event:')) {
                  // Extract event type
                  currentEvent = line.substring(7).trim();
                } else if (line.startsWith('data:')) {
                  try {
                    // Parse JSON data and emit event
                    const data = JSON.parse(line.substring(6));
                    observer.next({ type: currentEvent as any, data });
                  } catch (e) {
                    console.error('Error parsing SSE data:', e, line);
                  }
                }
              }
            }
          } catch (error) {
            observer.error(error);
          }
        };
        
        processStream();
      })
      .catch(error => {
        observer.error(error);
      });
      
      // Cleanup function (called when subscription is cancelled)
      return () => {
        // You can add abort controller logic here if needed
      };
    });
  }
  
  /**
   * Get current user ID from your auth service
   */
  private getCurrentUserId(): string {
    // TODO: Implement your user ID retrieval logic
    // Example: return this.authService.getUserId();
    return 'current-user-id';
  }
}
```

**Key Implementation Details:**

1. **Buffer Management**: SSE events can arrive in incomplete chunks. We maintain a `buffer` to handle partial lines.

2. **Event Parsing**: SSE format is:
   ```
   event: <type>
   data: <json>
   
   ```
   We parse these into typed `StreamEvent` objects.

3. **Error Handling**: Network errors and JSON parsing errors are caught and emitted through the Observable's error channel.

4. **Stream Cleanup**: The return function in Observable handles cleanup when subscription ends.

---

### Step 2: Implement the Chat Component

Create `components/chat.component.ts`:

```typescript
// chat.component.ts
import { Component, OnInit, OnDestroy } from '@angular/core';
import { ChatStreamService, StreamEvent } from '../services/chat-stream.service';
import { Subscription } from 'rxjs';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  intent?: string;
  confidence?: number;
}

@Component({
  selector: 'app-chat',
  templateUrl: './chat.component.html',
  styleUrls: ['./chat.component.css']
})
export class ChatComponent implements OnInit, OnDestroy {
  // UI State
  currentResponse = '';          // Accumulates response chunks
  statusMessage = '';            // Current processing status
  isStreaming = false;           // Whether actively receiving stream
  userInput = '';                // Input field value
  
  // Conversation State
  messages: Message[] = [];      // Full conversation history
  sessionId = '';                // Current session ID
  
  // Subscriptions
  private streamSubscription?: Subscription;
  
  constructor(private chatService: ChatStreamService) {}
  
  ngOnInit() {
    // Generate session ID when component initializes
    this.sessionId = this.generateSessionId();
    console.log('Chat session started:', this.sessionId);
  }
  
  ngOnDestroy() {
    // Clean up subscription when component is destroyed
    if (this.streamSubscription) {
      this.streamSubscription.unsubscribe();
    }
  }
  
  /**
   * Send user message and handle streaming response
   */
  sendMessage(message: string) {
    if (!message || !message.trim()) {
      return; // Don't send empty messages
    }
    
    // Store user message in conversation history
    this.messages.push({
      role: 'user',
      content: message,
      timestamp: new Date()
    });
    
    // Reset state for new response
    this.currentResponse = '';
    this.statusMessage = '';
    this.isStreaming = true;
    this.userInput = ''; // Clear input field
    
    // Get user info (replace with your actual user data)
    const userInfo = {
      user: this.getUserName(),
      user_id: this.getUserId()
    };
    
    // Start streaming
    this.streamSubscription = this.chatService.streamChat(
      message, 
      this.sessionId,
      userInfo
    ).subscribe({
      next: (event: StreamEvent) => this.handleStreamEvent(event),
      error: (err) => this.handleStreamError(err),
      complete: () => this.handleStreamComplete()
    });
  }
  
  /**
   * Handle individual stream events
   * 
   * KEY PATTERN: Status messages REPLACE, response chunks APPEND
   */
  private handleStreamEvent(event: StreamEvent) {
    switch (event.type) {
      case 'node_start':
        // ✅ REPLACE pattern: Assignment operator overwrites previous status
        // This ensures users only see the CURRENT processing step
        this.statusMessage = event.data.message;
        console.log(`[${event.data.node}] Started:`, event.data.message);
        break;
      
      case 'node_complete':
        // Optional: Log completion for debugging
        console.log(`[${event.data.node}] Completed`);
        break;
      
      case 'response_chunk':
        // ✅ APPEND pattern: += operator accumulates text
        // This builds the complete response progressively
        this.currentResponse += event.data.text;
        
        // Clear status message once first chunk arrives
        // (user no longer needs to see "Preparing response...")
        if (event.data.chunk_index === 0) {
          this.statusMessage = '';
        }
        
        // Auto-scroll to show new text
        setTimeout(() => this.scrollToBottom(), 0);
        break;
      
      case 'complete':
        // Response is complete!
        this.isStreaming = false;
        this.statusMessage = '';
        
        console.log('Response complete:', {
          intent: event.data.intent,
          confidence: event.data.confidence,
          duration: event.data.metadata?.duration_ms
        });
        
        // Store assistant message in conversation history
        this.messages.push({
          role: 'assistant',
          content: event.data.response,
          intent: event.data.intent,
          confidence: event.data.confidence,
          timestamp: new Date()
        });
        
        // Handle clarification if needed
        if (event.data.needs_clarification && event.data.clarifying_question) {
          this.showClarification(event.data.clarifying_question);
        }
        
        // Track analytics
        this.trackResponseAnalytics(event.data);
        break;
      
      case 'error':
        // Handle error from backend
        this.isStreaming = false;
        this.statusMessage = '';
        this.currentResponse = '';
        this.showError(event.data.message);
        console.error('Backend error:', event.data);
        break;
    }
  }
  
  /**
   * Handle stream errors (network, parsing, etc.)
   */
  private handleStreamError(err: any) {
    console.error('Stream error:', err);
    this.isStreaming = false;
    this.statusMessage = '';
    
    // Show user-friendly error message
    if (err.name === 'AbortError') {
      this.showError('Request was cancelled');
    } else if (err.message?.includes('NetworkError') || err.message?.includes('Failed to fetch')) {
      this.showError('Network error. Please check your connection and try again.');
    } else {
      this.showError('Failed to get response. Please try again.');
    }
  }
  
  /**
   * Handle stream completion
   */
  private handleStreamComplete() {
    console.log('Stream connection closed');
    this.isStreaming = false;
  }
  
  /**
   * Generate unique session ID
   */
  private generateSessionId(): string {
    const timestamp = Date.now();
    const random = Math.random().toString(36).substring(2, 11);
    return `sess-${timestamp}-${random}`;
  }
  
  /**
   * Auto-scroll chat to bottom
   */
  private scrollToBottom() {
    const chatContainer = document.querySelector('.chat-messages');
    if (chatContainer) {
      chatContainer.scrollTop = chatContainer.scrollHeight;
    }
  }
  
  /**
   * Show error message to user
   */
  private showError(message: string) {
    // TODO: Implement your error UI (toast, snackbar, etc.)
    alert(message); // Replace with proper UI component
  }
  
  /**
   * Show clarification prompt
   */
  private showClarification(question: string) {
    // TODO: Implement clarification UI
    console.log('Clarification needed:', question);
  }
  
  /**
   * Track response analytics
   */
  private trackResponseAnalytics(data: any) {
    // TODO: Send to your analytics service
    console.log('Analytics:', {
      intent: data.intent,
      confidence: data.confidence,
      duration_ms: data.metadata?.duration_ms,
      session_id: this.sessionId
    });
  }
  
  /**
   * Get current user's name (replace with your auth logic)
   */
  private getUserName(): string {
    // TODO: Get from your auth service
    return 'CurrentUser';
  }
  
  /**
   * Get current user's ID (replace with your auth logic)
   */
  private getUserId(): string {
    // TODO: Get from your auth service
    return 'user-id-123';
  }
  
  /**
   * Cancel ongoing stream (if user navigates away)
   */
  cancelStream() {
    if (this.streamSubscription) {
      this.streamSubscription.unsubscribe();
      this.isStreaming = false;
      this.statusMessage = '';
    }
  }
}
```

**Key Implementation Details:**

1. **State Management**: 
   - `currentResponse` builds up from chunks
   - `statusMessage` shows current processing stage
   - `isStreaming` controls UI loading states
   - `messages` maintains full conversation history

2. **Event Handling**: Each event type updates different parts of the UI state

3. **Error Handling**: Both stream errors and backend errors are handled gracefully

4. **Session Management**: Session ID is generated once per conversation and reused for context

5. **Auto-scroll**: Ensures new text is always visible as it streams in

---

### Step 3: Create the Template

Create `components/chat.component.html`:

```html
<!-- chat.component.html -->
<div class="chat-container">
  
  <!-- Conversation History -->
  <div class="chat-messages">
    <!-- Loop through all messages -->
    <div *ngFor="let message of messages" 
         class="message"
         [class.user-message]="message.role === 'user'"
         [class.assistant-message]="message.role === 'assistant'">
      
      <div class="message-header">
        <span class="message-role">{{ message.role === 'user' ? 'You' : 'MyClaims Assistant' }}</span>
        <span class="message-time">{{ message.timestamp | date:'short' }}</span>
      </div>
      
      <div class="message-content">
        {{ message.content }}
      </div>
      
      <!-- Show confidence indicator for assistant messages -->
      <div *ngIf="message.role === 'assistant' && message.confidence" 
           class="message-meta">
        <span class="confidence" 
              [class.low-confidence]="message.confidence < 0.7">
          Confidence: {{ (message.confidence * 100).toFixed(0) }}%
        </span>
      </div>
    </div>
    
    <!-- Current Streaming Response -->
    <div *ngIf="currentResponse" class="message assistant-message streaming">
      <div class="message-header">
        <span class="message-role">MyClaims Assistant</span>
        <span class="streaming-indicator">●</span>
      </div>
      
      <div class="message-content">
        {{ currentResponse }}
        <!-- Animated cursor while streaming -->
        <span class="typing-cursor" *ngIf="isStreaming">|</span>
      </div>
    </div>
  </div>
  
  <!-- Status Indicator (shows processing stage) -->
  <div *ngIf="statusMessage" class="status-bar">
    <div class="status-content">
      <span class="spinner"></span>
      <span class="status-text">{{ statusMessage }}</span>
    </div>
  </div>
  
  <!-- Input Area -->
  <div class="input-area">
    <input 
      type="text" 
      [(ngModel)]="userInput"
      (keyup.enter)="sendMessage(userInput)"
      [disabled]="isStreaming"
      placeholder="Ask about your claims..."
      class="chat-input"
    />
    <button 
      (click)="sendMessage(userInput)"
      [disabled]="isStreaming || !userInput?.trim()"
      class="send-button"
    >
      <span *ngIf="!isStreaming">Send</span>
      <span *ngIf="isStreaming">Sending...</span>
    </button>
  </div>
  
</div>
```

**Template Explanation:**

1. **Chat Messages Section**: Displays full conversation history with both user and assistant messages

2. **Current Streaming Response**: Shows the response being built in real-time with animated cursor

3. **Status Bar**: Appears at bottom to show current processing stage (e.g., "Understanding your question...")

4. **Input Area**: Text input with send button, disabled during streaming to prevent duplicate requests

5. **Confidence Indicator**: Shows AI confidence level for each response (optional, can be hidden)

---

### Step 4: Add Styling

Create `components/chat.component.css`:

```css
/* chat.component.css */

/* Main container */
.chat-container {
  display: flex;
  flex-direction: column;
  height: 100vh;
  max-width: 900px;
  margin: 0 auto;
  background-color: #f5f5f5;
}

/* Messages area (scrollable) */
.chat-messages {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

/* Individual message bubble */
.message {
  max-width: 75%;
  padding: 12px 16px;
  border-radius: 12px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

/* User message (right-aligned, blue) */
.user-message {
  align-self: flex-end;
  background-color: #007bff;
  color: white;
}

/* Assistant message (left-aligned, white) */
.assistant-message {
  align-self: flex-start;
  background-color: white;
  color: #333;
}

/* Streaming message has pulse animation */
.assistant-message.streaming {
  animation: pulse 2s ease-in-out infinite;
}

@keyframes pulse {
  0%, 100% { box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); }
  50% { box-shadow: 0 2px 8px rgba(0, 123, 255, 0.3); }
}

/* Message header */
.message-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
  font-size: 12px;
  opacity: 0.7;
}

.message-role {
  font-weight: 600;
}

.message-time {
  font-size: 11px;
}

.streaming-indicator {
  color: #28a745;
  animation: blink 1.5s ease-in-out infinite;
}

/* Message content */
.message-content {
  line-height: 1.6;
  white-space: pre-wrap;
  word-wrap: break-word;
  font-size: 14px;
}

/* Message metadata (confidence, etc.) */
.message-meta {
  margin-top: 8px;
  padding-top: 8px;
  border-top: 1px solid rgba(0, 0, 0, 0.1);
  font-size: 11px;
  opacity: 0.6;
}

.confidence {
  padding: 2px 6px;
  border-radius: 4px;
  background-color: #28a745;
  color: white;
}

.confidence.low-confidence {
  background-color: #ffc107;
  color: #333;
}

/* Typing cursor animation */
.typing-cursor {
  display: inline-block;
  width: 2px;
  height: 1em;
  background-color: currentColor;
  margin-left: 2px;
  animation: blink 1s step-end infinite;
}

@keyframes blink {
  50% { opacity: 0; }
}

/* Status bar at bottom */
.status-bar {
  padding: 12px 20px;
  background-color: #fff;
  border-top: 1px solid #ddd;
  box-shadow: 0 -2px 8px rgba(0, 0, 0, 0.05);
}

.status-content {
  display: flex;
  align-items: center;
  gap: 12px;
  color: #666;
  font-size: 14px;
}

/* Loading spinner */
.spinner {
  width: 18px;
  height: 18px;
  border: 2px solid #e0e0e0;
  border-top-color: #007bff;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

/* Input area */
.input-area {
  display: flex;
  gap: 12px;
  padding: 16px 20px;
  background-color: white;
  border-top: 1px solid #ddd;
}

.chat-input {
  flex: 1;
  padding: 12px 16px;
  border: 1px solid #ddd;
  border-radius: 8px;
  font-size: 14px;
  outline: none;
  transition: border-color 0.2s;
}

.chat-input:focus {
  border-color: #007bff;
  box-shadow: 0 0 0 3px rgba(0, 123, 255, 0.1);
}

.chat-input:disabled {
  background-color: #f5f5f5;
  cursor: not-allowed;
}

.send-button {
  padding: 12px 24px;
  background-color: #007bff;
  color: white;
  border: none;
  border-radius: 8px;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  transition: background-color 0.2s;
}

.send-button:hover:not(:disabled) {
  background-color: #0056b3;
}

.send-button:disabled {
  background-color: #ccc;
  cursor: not-allowed;
}

/* Responsive design */
@media (max-width: 768px) {
  .message {
    max-width: 85%;
  }
  
  .input-area {
    padding: 12px;
  }
  
  .chat-messages {
    padding: 12px;
  }
}
```

**Styling Explanation:**

1. **Layout**: Uses flexbox for a responsive chat interface with scrollable message area

2. **Message Bubbles**: Different styles for user vs assistant messages (right/left aligned, different colors)

3. **Animations**:
   - Blinking cursor while streaming
   - Spinning loader in status bar
   - Pulse effect on streaming message
   - Blinking indicator dot

4. **Accessibility**: Clear focus states, good color contrast, disabled states

5. **Responsive**: Adapts to mobile screens

---

## 🎨 Rendering Strategy: Replace vs Append (Detailed)

This section explains exactly how to achieve the UX where status updates replace each other instead of stacking.

### **The Problem**

If you naively append everything, you get noise:

```
❌ BAD UX (Everything appends):
Processing your request...
Checking safety and privacy...
Understanding your question...
Retrieving your claims information...
Preparing your response...
Based on your claim 847293156420183...
```

Users see a wall of text including internal processing steps they don't care about.

### **The Solution**

Use **two separate UI elements** with different update strategies:

```html
<!-- Element 1: Status indicator (REPLACE strategy) -->
<div id="status-indicator">
  {{ statusMessage }}  ← Gets REPLACED on each node_start
</div>

<!-- Element 2: Response display (APPEND strategy) -->
<div id="response-display">
  {{ currentResponse }}  ← Gets APPENDED on each response_chunk
</div>
```

### **Implementation in Angular**

**TypeScript (Component):**
```typescript
export class ChatComponent {
  // Status: SINGLE VALUE (replaces)
  statusMessage: string = '';
  
  // Response: ACCUMULATED STRING (appends)
  currentResponse: string = '';
  
  handleStreamEvent(event: StreamEvent) {
    if (event.type === 'node_start') {
      // ✅ Assignment = REPLACE
      this.statusMessage = event.data.message;
      
      // What happens:
      // User sees: "Processing your request..."
      // Next event: "Checking safety and privacy..." (previous message gone)
      // Next event: "Understanding your question..." (previous message gone)
      // Result: Only CURRENT status visible
    }
    
    if (event.type === 'response_chunk') {
      // ✅ Concatenation = APPEND
      this.currentResponse += event.data.text;
      
      // What happens:
      // First chunk: "Based on your claim..."
      // Second chunk: "Based on your claim 847293156420183..."
      // Third chunk: "Based on your claim 847293156420183, sequence 1..."
      // Result: Response ACCUMULATES
      
      // Hide status once response starts
      this.statusMessage = '';
    }
  }
}
```

**HTML (Template):**
```html
<div class="chat-interface">
  
  <!-- Status Bar: Shows current processing step -->
  <div *ngIf="statusMessage" class="status-bar">
    <span class="spinner"></span>
    <span class="status-text">{{ statusMessage }}</span>
  </div>
  
  <!-- Response Display: Shows accumulated response -->
  <div *ngIf="currentResponse" class="response-display">
    {{ currentResponse }}
    <span *ngIf="isStreaming" class="cursor">|</span>
  </div>
  
</div>
```

**CSS (Styling):**
```css
/* Status bar: Fixed position at bottom */
.status-bar {
  position: fixed;
  bottom: 80px;
  left: 0;
  right: 0;
  padding: 12px 20px;
  background: white;
  border-top: 1px solid #ddd;
  box-shadow: 0 -2px 8px rgba(0,0,0,0.05);
  display: flex;
  align-items: center;
  gap: 12px;
}

/* Response display: Scrollable content area */
.response-display {
  padding: 20px;
  background: white;
  border-radius: 12px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  white-space: pre-wrap;
  word-wrap: break-word;
}
```

### **Visual Flow Example**

**Step 1:** Request sent
```
┌─────────────────────────────────┐
│ [Processing your request...]    │ ← status-bar visible
└─────────────────────────────────┘
```

**Step 2:** Safety check starts
```
┌────────────────────────────────────┐
│ [Checking safety and privacy...]   │ ← status REPLACED (not appended)
└────────────────────────────────────┘
```

**Step 3:** Intent understanding starts
```
┌──────────────────────────────────┐
│ [Understanding your question...] │ ← status REPLACED again
└──────────────────────────────────┘
```

**Step 4:** First response chunk arrives
```
┌─────────────────────────────────┐
│ Based on your claim 2531527...  │ ← status hidden, response shown
└─────────────────────────────────┘
```

**Step 5:** More chunks arrive
```
┌──────────────────────────────────────────┐
│ Based on your claim 847293156420183,     │ ← text APPENDED
│ sequence 1 shows a copay of $25 for...   │
└──────────────────────────────────────────┘
```

### **Common Mistakes to Avoid**

❌ **Mistake 1: Appending status messages**
```typescript
// DON'T DO THIS
this.statusMessage += event.data.message + '\n';
// Results in: "Processing...\nChecking...\nUnderstanding..."
```

✅ **Correct: Replace status messages**
```typescript
// DO THIS
this.statusMessage = event.data.message;
// Results in: "Understanding..." (only current status)
```

❌ **Mistake 2: Replacing response chunks**
```typescript
// DON'T DO THIS
this.currentResponse = event.data.text;
// Results in: Only last chunk visible
```

✅ **Correct: Append response chunks**
```typescript
// DO THIS
this.currentResponse += event.data.text;
// Results in: Full response accumulated
```

❌ **Mistake 3: Using same element for both**
```html
<!-- DON'T DO THIS -->
<div>{{ messages }}</div>
<!-- Where messages = all status + response mixed -->
```

✅ **Correct: Separate elements**
```html
<!-- DO THIS -->
<div class="status">{{ statusMessage }}</div>
<div class="response">{{ currentResponse }}</div>
```

### **How ChatGPT/Claude Do It**

For reference, here's how industry leaders handle this:

**ChatGPT:**
- Status: Single element with spinner, shows "Thinking..." → "Searching web..." → (disappears)
- Response: Separate element, accumulates tokens with typing cursor

**Claude:**
- Status: Single element, shows "Claude is thinking..." → (disappears)
- Response: Separate element, accumulates with animated gradient

**Perplexity:**
- Status: Single element, shows "Searching sources..." → "Analyzing results..." → (disappears)
- Response: Separate element, accumulates with source citations

**Our Implementation:**
- Status: Single element, shows 5 key milestones (orchestrator → safety → intent → fetch → generate)
- Response: Separate element, accumulates ~50 char chunks

---

---

## Alternative: React Integration

For teams using React, the core concepts are identical. The **replace vs append** pattern is implemented using React state setters instead of component properties.

### React Hook Implementation

```typescript
// hooks/useChatStream.ts
import { useState, useCallback } from 'react';

interface StreamEvent {
  type: 'node_start' | 'node_complete' | 'response_chunk' | 'complete' | 'error';
  data: any;
}

export function useChatStream() {
  // ✅ Status: REPLACE pattern (setState overwrites)
  const [status, setStatus] = useState('');
  
  // ✅ Response: APPEND pattern (setState with prev => prev + new)
  const [response, setResponse] = useState('');
  
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const streamChat = useCallback(async (message: string, sessionId: string) => {
    setResponse('');
    setStatus('');
    setIsStreaming(true);
    setError(null);
    
    try {
      const res = await fetch('/api/v1/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: message,
          session_id: sessionId,
          user_info: { user_id: 'your-user-id' }
        })
      });
      
      if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
      
      const reader = res.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        
        let currentEvent = 'message';
        
        for (const line of lines) {
          if (line.startsWith('event:')) {
            currentEvent = line.substring(7).trim();
          } else if (line.startsWith('data:')) {
            const data = JSON.parse(line.substring(6));
            
            // Handle events with replace/append patterns
            switch (currentEvent) {
              case 'node_start':
                // ✅ REPLACE: Direct state update overwrites previous status
                setStatus(data.message);
                break;
                
              case 'response_chunk':
                // ✅ APPEND: Use functional update to accumulate text
                setResponse(prev => prev + data.text);
                
                // Clear status once response starts
                if (data.chunk_index === 0) {
                  setStatus('');
                }
                break;
                
              case 'complete':
                setIsStreaming(false);
                setStatus('');
                break;
                
              case 'error':
                setError(data.message);
                setIsStreaming(false);
                setStatus('');
                break;
            }
          }
        }
      }
      
      setIsStreaming(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
      setIsStreaming(false);
    }
  }, []);
  
  return { response, status, isStreaming, error, streamChat };
}
```

### React Component Usage

```typescript
function ChatComponent() {
  const { response, status, isStreaming, error, streamChat } = useChatStream();
  const [input, setInput] = useState('');
  const sessionId = useRef(`sess-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`).current;
  
  const handleSend = () => {
    if (input.trim()) {
      streamChat(input, sessionId);
      setInput('');
    }
  };
  
  return (
    <div className="chat-container">
      
      {/* Status Bar: Single element that REPLACES */}
      {status && (
        <div className="status-bar">
          <span className="spinner"></span>
          <span className="status-text">{status}</span>
        </div>
      )}
      
      {/* Response Display: Element that APPENDS */}
      {response && (
        <div className="response-display">
          {response}
          {isStreaming && <span className="cursor">|</span>}
        </div>
      )}
      
      {/* Error Display */}
      {error && (
        <div className="error-display">{error}</div>
      )}
      
      {/* Input Area */}
      <div className="input-area">
        <input 
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && handleSend()}
          disabled={isStreaming}
          placeholder="Ask about your claims..."
        />
        <button onClick={handleSend} disabled={isStreaming || !input.trim()}>
          {isStreaming ? 'Sending...' : 'Send'}
        </button>
      </div>
      
    </div>
  );
}
```

**Key Pattern in React:**
```typescript
// ✅ Status REPLACE: Direct setState
setStatus(data.message);  // Overwrites previous value

// ✅ Response APPEND: Functional setState
setResponse(prev => prev + data.text);  // Accumulates previous + new
```

**Key Differences from Angular:**
- Uses hooks (`useState`, `useCallback`) instead of class properties
- State updates via `setState` instead of direct assignment
- **REPLACE** = `setState(newValue)`
- **APPEND** = `setState(prev => prev + newValue)`
- Same SSE parsing logic, same event handling logic
- Refer to Angular example above for detailed explanations

---

---

## Performance Characteristics & What to Expect

### Typical Request Timeline

Understanding the timing helps you set proper loading states and user expectations:

| Stage | Time | What User Sees | Backend Event |
|-------|------|---------------|---------------|
| **Request sent** | 0s | Input submitted | - |
| Processing | 0-0.5s | "Processing your request..." | `node_start: orchestrator` |
| Safety check | 0.5-1s | "Checking safety and privacy..." | `node_start: safety_precheck` |
| *Internal: Cache check* | *1-1.5s* | *(No update - internal)* | *(Logged only)* |
| Understanding | 1.5-3s | "Understanding your question..." | `node_start: intent_agent` |
| *Internal: Confidence* | *3-3.5s* | *(No update - internal)* | *(Logged only)* |
| *Internal: Context* | *3.5-4s* | *(No update - internal)* | *(Logged only)* |
| Data retrieval | 4-8s | "Retrieving your claims information..." | `node_start: call_claims_tool` |
| *Internal: Safety prep* | *8-8.5s* | *(No update - internal)* | *(Logged only)* |
| Response generation | 8.5-12s | "Preparing your response..." | `node_start: response_agent` |
| *Internal: Safety validate* | *12-12.5s* | *(No update - internal)* | *(Logged only)* |
| **🎉 First chunk** | **~12.5s** | **"Based on your claim..."** | `response_chunk` (index 0) |
| Streaming chunks | 12.5-14s | Text appears progressively | Multiple `response_chunk` events |
| **Complete** | **~14s** | Full response visible | `complete` event |

**Note:** Internal steps still execute but don't send status updates to reduce noise.

### Why Streaming Still Improves UX

**Without Streaming:**
- User waits 14 seconds staring at a spinner
- Anxiety builds: "Is it working?"
- No feedback on progress
- **Perceived wait time: 14 seconds**

**With Streaming:**
- Status updates every 1-2 seconds
- User knows exactly what's happening
- Response appears progressively after 12.5s
- User can start reading before completion
- **Perceived wait time: 50-70% less** (~5-7 seconds)

### Real-World Performance Benefits

1. **Engagement**: Users stay engaged with constant visual feedback
2. **Trust**: Transparent processing builds confidence
3. **Efficiency**: Users can start reading immediately when chunks arrive
4. **Error Detection**: Failed stages are immediately visible
5. **Perceived Speed**: Feels much faster than waiting for everything

### Network & Resource Considerations

**Bandwidth Usage:**
- **Non-streaming endpoint**: ~2KB response in one payload
- **Streaming endpoint**: ~2.2KB response (10% overhead from SSE formatting)
- **Overhead breakdown**: Event headers (~50 bytes per event), JSON formatting
- **Verdict**: Negligible bandwidth increase for significant UX improvement

**Connection Management:**
- Single HTTP connection kept alive for ~14 seconds
- Auto-closes after `complete` event
- Browser automatically reconnects if connection drops
- No special cleanup needed in frontend

**Backend Resource Usage:**
- Same processing time as non-streaming
- Minimal additional memory (buffering chunks)
- No performance degradation

### Expected Chunk Behavior

- **Chunk Size**: ~50 characters per chunk (backend configurable)
- **Chunk Frequency**: ~50-100ms between chunks
- **Total Chunks**: Varies by response length (typical: 5-10 chunks for 250-500 char response)
- **Chunk Ordering**: Always sequential (index 0, 1, 2, ...)

**Example Response Chunking:**

```
Response: "Based on your claim 847293156420183, sequence 1 shows a copay of $25 for your prescription Lisinopril filled on 2024-01-15."

Chunks:
0: "Based on your claim 847293156420183, sequence "
1: "1 shows a copay of $25 for your prescription"
2: " Lisinopril filled on 2024-01-15."
```

---

## Error Handling Guide

Proper error handling ensures a smooth user experience even when things go wrong.

### Error Types You'll Encounter

#### 1. Backend Errors (via `error` event)

These come from the backend as SSE `error` events:

```
event: error
data: {"message": "User-friendly error message", "reason": "error_category"}
```

**Common Backend Error Reasons:**

| Reason | When It Happens | How to Handle |
|--------|----------------|---------------|
| `safety_violation` | Input or output violates safety policies | Show message: "Please rephrase your question to remove sensitive information" |
| `internal_error` | Backend encountered unexpected error | Show retry button: "Something went wrong. Please try again." |
| `timeout` | Request took too long to process | Show retry with: "Request timed out. Please try a simpler question." |
| `invalid_request` | Request format was wrong | Log error and show: "Please try again" (this shouldn't happen in production) |

**Example Handler:**

```typescript
if (event.type === 'error') {
  this.isStreaming = false;
  this.statusMessage = '';
  this.currentResponse = '';
  
  // Always show the user-friendly message from backend
  this.showErrorToUser(event.data.message);
  
  // Log for debugging
  console.error('Backend error:', {
    message: event.data.message,
    reason: event.data.reason,
    session_id: this.sessionId
  });
  
  // Track in analytics
  this.analytics.trackError('backend_error', event.data.reason);
  
  // Specific handling
  switch (event.data.reason) {
    case 'safety_violation':
      // Don't offer retry for safety violations
      this.showSafetyGuidelines();
      break;
    
    case 'internal_error':
    case 'timeout':
      // Offer retry for these errors
      this.showRetryButton();
      break;
  }
}
```

#### 2. Network Errors (via Observable error channel)

These occur when the connection itself fails:

```typescript
error: (err) => {
  this.isStreaming = false;
  this.statusMessage = '';
  
  console.error('Network/Stream error:', err);
  
  // Categorize the error
  if (err.name === 'AbortError') {
    // User navigated away or cancelled
    console.log('Request was cancelled by user');
    
  } else if (err.message?.includes('Failed to fetch') || 
             err.message?.includes('NetworkError')) {
    // Network connectivity issue
    this.showError('Unable to connect. Please check your internet connection and try again.');
    this.showRetryButton();
    
  } else if (err.message?.includes('timeout')) {
    // Request timeout
    this.showError('Request timed out. Please try again.');
    this.showRetryButton();
    
  } else {
    // Unknown error
    this.showError('An unexpected error occurred. Please try again.');
    this.showRetryButton();
    
    // Log to your error tracking service
    this.logErrorToService(err);
  }
}
```

### Implementing Retry Logic

For transient errors (network issues, timeouts), offer automatic retry:

```typescript
export class ChatComponent {
  private retryAttempts = 0;
  private maxRetries = 3;
  private retryDelay = 2000; // 2 seconds
  
  /**
   * Send message with automatic retry on failure
   */
  sendMessageWithRetry(message: string) {
    this.retryAttempts = 0;
    this.attemptSendMessage(message);
  }
  
  private attemptSendMessage(message: string) {
    this.chatService.streamChat(message, this.sessionId, this.getUserInfo())
      .subscribe({
        next: (event) => this.handleStreamEvent(event),
        
        error: (err) => {
          // Check if error is retryable
          const isRetryable = this.isRetryableError(err);
          
          if (isRetryable && this.retryAttempts < this.maxRetries) {
            this.retryAttempts++;
            
            // Show retry status to user
            this.statusMessage = `Connection failed. Retrying (${this.retryAttempts}/${this.maxRetries})...`;
            
            // Retry after delay
            setTimeout(() => {
              console.log(`Retry attempt ${this.retryAttempts}`);
              this.attemptSendMessage(message);
            }, this.retryDelay);
            
          } else {
            // Max retries reached or non-retryable error
            this.handleFinalError(err);
          }
        },
        
        complete: () => {
          // Reset retry counter on success
          this.retryAttempts = 0;
        }
      });
  }
  
  /**
   * Determine if an error should trigger retry
   */
  private isRetryableError(err: any): boolean {
    // Retry network errors and timeouts
    return err.message?.includes('NetworkError') ||
           err.message?.includes('Failed to fetch') ||
           err.message?.includes('timeout');
  }
  
  /**
   * Handle error after all retries exhausted
   */
  private handleFinalError(err: any) {
    this.isStreaming = false;
    this.statusMessage = '';
    
    if (this.retryAttempts >= this.maxRetries) {
      this.showError(
        `Unable to connect after ${this.maxRetries} attempts. Please check your connection and try again.`
      );
    } else {
      this.showError('An error occurred. Please try again.');
    }
    
    // Log to error tracking
    this.analytics.trackError('max_retries_exceeded', err.message);
  }
}
```

### Implementing Manual Retry Button

Allow users to manually retry failed requests:

```typescript
export class ChatComponent {
  lastFailedMessage: string | null = null;
  showRetryButton = false;
  
  sendMessage(message: string) {
    // Store message for potential retry
    this.lastFailedMessage = message;
    this.showRetryButton = false;
    
    // ... send message logic ...
  }
  
  handleStreamError(err: any) {
    this.isStreaming = false;
    this.showRetryButton = true; // Show retry button
    
    // Handle error...
  }
  
  retryLastMessage() {
    if (this.lastFailedMessage) {
      this.sendMessage(this.lastFailedMessage);
    }
  }
}
```

**Template:**

```html
<!-- Add retry button in your template -->
<div *ngIf="showRetryButton" class="retry-container">
  <button (click)="retryLastMessage()" class="retry-button">
    🔄 Retry
  </button>
</div>
```

### Timeout Configuration

Set appropriate timeouts for the fetch request:

```typescript
// In chat-stream.service.ts
streamChat(message: string, sessionId: string, userInfo?: any): Observable<StreamEvent> {
  return new Observable(observer => {
    // Create AbortController for timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 60000); // 60 second timeout
    
    fetch(`${this.baseUrl}/api/v1/chat/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: message, session_id: sessionId, user_info: userInfo }),
      signal: controller.signal // Attach abort signal
    })
    .then(response => {
      clearTimeout(timeoutId); // Clear timeout on success
      // ... process stream ...
    })
    .catch(error => {
      clearTimeout(timeoutId);
      observer.error(error);
    });
    
    // Cleanup function
    return () => {
      clearTimeout(timeoutId);
      controller.abort(); // Cancel request if subscription is cancelled
    };
  });
}
```

### Safety Violation Handling

Safety violations require special handling since they're not retryable:

```typescript
if (event.type === 'error' && event.data.reason === 'safety_violation') {
  // Don't show retry button
  this.showRetryButton = false;
  
  // Show specific guidance
  this.showModal({
    title: 'Privacy Protection',
    message: event.data.message,
    guidance: [
      'Please avoid including:',
      '• Full names',
      '• Social Security numbers',
      '• Dates of birth',
      '• Full addresses',
      'You can reference claims by number instead.'
    ]
  });
}
```

---

---

## Fallback to Non-Streaming Endpoint (Optional)

If your backend also provides a non-streaming endpoint, you can use it as a fallback:

**Non-Streaming Endpoint:** `POST /api/v1/chat`

**When to use:**
- Streaming fails repeatedly
- User is on very slow/unreliable connection
- Mobile app wants to minimize battery usage
- Legacy browser doesn't support streaming well

**Angular Example:**

```typescript
// In your service, add non-streaming method
nonStreamingChat(message: string, sessionId: string): Observable<any> {
  return this.http.post(`${this.baseUrl}/api/v1/chat`, {
    text: message,
    session_id: sessionId,
    user_info: this.getUserInfo()
  });
}

// In your component, use as fallback
sendMessageWithFallback(message: string) {
  // Try streaming first
  this.chatService.streamChat(message, this.sessionId, this.getUserInfo())
    .pipe(
      catchError(err => {
        console.warn('Streaming failed, using fallback', err);
        
        // Fall back to non-streaming
        return this.chatService.nonStreamingChat(message, this.sessionId).pipe(
          map(response => ({
            type: 'complete' as const,
            data: response
          }))
        );
      })
    )
    .subscribe({
      next: (event) => this.handleStreamEvent(event),
      error: (err) => this.handleStreamError(err)
    });
}
```

**Note:** Check with your backend team if the non-streaming `/chat` endpoint is available. The streaming endpoint is recommended for the best user experience.

---

---

## Quick Integration Checklist

Use this checklist to ensure your integration is complete:

### Backend Setup
- [ ] Confirm backend URL and endpoint path `/api/v1/chat/stream`
- [ ] Verify CORS is configured for your frontend domain
- [ ] Test endpoint with cURL or Postman (see Testing section below)
- [ ] Confirm streaming is enabled in backend config

### Frontend Implementation
- [ ] Create streaming service with SSE parsing logic
- [ ] Implement component with state management
- [ ] Add template with message display, status indicator, and input
- [ ] Add CSS styling for chat UI
- [ ] Implement error handling and retry logic
- [ ] Add session ID generation and management
- [ ] Connect to your authentication service for user_info
- [ ] Test with various queries

### User Experience
- [ ] Status messages display during processing
- [ ] Response chunks appear progressively
- [ ] Loading indicators work correctly
- [ ] Errors show user-friendly messages
- [ ] Input is disabled during streaming
- [ ] Auto-scroll works as new text appears
- [ ] Confidence indicators display (optional)

### Production Readiness
- [ ] Replace mock user IDs with real authentication
- [ ] Update backend URL for production
- [ ] Configure proper CORS origins (not `*`)
- [ ] Add analytics tracking
- [ ] Add error logging/monitoring
- [ ] Test on mobile devices
- [ ] Test with slow network connections
- [ ] Add loading timeouts
- [ ] Document for your team

---

## Testing Your Integration

### 1. Quick Browser Console Test

Test the endpoint directly in browser DevTools console:

```javascript
// Test streaming in browser console
(async () => {
  console.log('🚀 Starting streaming test...');
  
  const response = await fetch('http://your-backend-url.com/api/v1/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      text: 'What are the details for claim 847293156420183 sequence 1?',
      session_id: 'test-console-' + Date.now(),
      user_info: { user: 'test', user_id: 'test-123' }
    })
  });

  console.log('📡 Response status:', response.status);
  
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let eventType = 'message';

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      console.log('✅ Stream complete');
      break;
    }
    
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';
    
    for (const line of lines) {
      if (line.startsWith('event:')) {
        eventType = line.substring(7).trim();
      } else if (line.startsWith('data:')) {
        const data = JSON.parse(line.substring(6));
        console.log(`[${eventType}]`, data);
      }
    }
  }
})();
```

### 2. cURL Testing

Test from command line:

**Windows PowerShell** (Use `curl.exe` to bypass PowerShell alias):
```powershell
curl.exe -N -X POST http://localhost:8000/api/v1/chat/stream -H "Content-Type: application/json" -H "Accept: text/event-stream" -d "{\"text\": \"What is my claim status?\", \"session_id\": \"curl-test-123\"}"
```

**Important for Windows:** 
- Use `curl.exe` (not `curl`) to avoid PowerShell's `Invoke-WebRequest` alias
- Use double quotes with escaped inner quotes `\"` for JSON strings
- Single-line format for easier copy-paste

**Windows Git Bash / WSL** (Standard curl syntax):
```bash
curl -N -X POST http://localhost:8000/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"text": "What is my claim status?", "session_id": "curl-test-123"}'
```

**Linux / Mac** (Standard curl syntax):
```bash
curl -N -X POST http://localhost:8000/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"text": "What is my claim status?", "session_id": "curl-test-123"}'
```

**Alternative: Using PowerShell's Invoke-WebRequest:**
```powershell
$body = @{
    text = "What are the details for claim 847293156420183 sequence 1?"
    session_id = "test-curl-123"
    user_info = @{
        user = "test"
        user_id = "test-123"
    }
} | ConvertTo-Json

Invoke-WebRequest -Uri "http://localhost:8000/api/v1/chat/stream" `
    -Method POST `
    -Headers @{"Content-Type"="application/json"; "Accept"="text/event-stream"} `
    -Body $body
```

**Expected Output:**
```
event: node_start
data: {"node": "orchestrator", "message": "Processing your request..."}

event: node_start
data: {"node": "safety_precheck", "message": "Checking safety and privacy..."}

event: response_chunk
data: {"text": "Your claim status...", "chunk_index": 0}

event: complete
data: {"response": "...", "intent": "claim_status"}
```

### 3. Test Queries to Try

| Query | Purpose | Expected Intent |
|-------|---------|-----------------|
| "Hello" | Simple greeting | `general_greeting` |
| "What is my claim status?" | General claim query | `claim_status` |
| "What are the details for claim 847293156420183 sequence 1?" | Specific claim details | `claim_details` |
| "What was my copay for Lisinopril?" | Prescription query | `rx_details` |
| "Tell me about my coverage" | Coverage inquiry | `coverage_inquiry` |
| "My SSN is 123-45-6789" | Safety violation test | Should trigger `error` event with `safety_violation` |

### 4. Expected Event Flow

For a typical query, you should see **ONLY user-facing events** (not internal nodes):

**What You'll Receive (V2.0 - Simplified):**
```
✅ event: node_start (orchestrator) - "Processing your request..."
✅ event: node_complete (orchestrator)
✅ event: node_start (safety_precheck) - "Checking safety and privacy..."
✅ event: node_complete (safety_precheck)
✅ event: node_start (intent_agent) - "Understanding your question..."
✅ event: node_complete (intent_agent)
✅ event: node_start (call_claims_tool) - "Retrieving your claims information..."
✅ event: node_complete (call_claims_tool)
✅ event: node_start (response_agent) - "Preparing your response..."
✅ event: node_complete (response_agent)
✅ event: response_chunk (multiple chunks)
✅ event: complete
```

**What Runs Internally (Not Shown to You):**
```
🔒 check_cache (internal optimization)
🔒 confidence_checker (internal routing)
🔒 build_context (internal context building)
🔒 response_safety_pii_precheck (internal security - masks PII)
🔒 response_safety_pii_postcheck (internal security - unmasks PII)
🔒 update_memory (internal conversation storage)
🔒 cache_response (internal caching)
```

**Why This Matters:**
- You receive **~10 events total** (5 node_start + 5 node_complete + chunks + complete)
- Instead of **~24 events** (all internal nodes included)
- **All backend functionality still works** - internal nodes execute normally
- **Better UX** - less noise for users

### 5. What to Verify

| Aspect | How to Check | Expected Result |
|--------|-------------|-----------------|
| **Connection** | Check HTTP status | `200 OK` |
| **Content Type** | Check response header | `text/event-stream` |
| **Event Format** | Inspect event structure | Each event has `event:` and `data:` lines |
| **Status Messages** | Look at `node_start` events | User-friendly messages in `data.message` |
| **Response Chunks** | Look at `response_chunk` events | Progressive text chunks with index |
| **Complete Event** | Final `complete` event | Full response with metadata |
| **No PII Tokens** | Inspect all text | No `[PII_PERSON_1]` or similar tokens |
| **Timing** | Measure from send to first chunk | ~8-14 seconds typically |

---

---

## Security & Privacy Considerations

### HIPAA Compliance - How We Protect Sensitive Data

The streaming implementation is designed to be HIPAA compliant. Here's how:

#### 1. **Input Validation (Safety Precheck)**

**What happens:** Before processing your request, the backend scans for:
- Social Security numbers
- Credit card numbers  
- Dates of birth
- Full addresses
- Other PII patterns

**Why it matters:** Prevents users from accidentally submitting sensitive data that shouldn't be processed

**Frontend impact:** If detected, you'll receive an `error` event with `reason: "safety_violation"`

#### 2. **Response Validation (Safety Postcheck)**

**What happens:** After the AI generates a response, but BEFORE any chunks are streamed, the backend:
- Scans for PII/PHI leakage
- Validates all masked tokens are properly unmasked
- Ensures no sensitive data patterns are present

**Why it matters:** This is the critical safety gate. Response chunks ONLY stream after this check passes.

**Frontend impact:** You'll see the `response_safety_pii_postcheck` node running before any `response_chunk` events arrive

#### 3. **Token Masking & Unmasking**

During processing, the backend may temporarily mask sensitive data as tokens like `[PII_PERSON_1]`, `[PII_DATE_1]`, etc. These tokens are ALWAYS unmasked before streaming.

**What you should NEVER see:** Masked tokens in your chunks. If you see something like `[PII_PERSON_1]` in a response chunk, that's a critical bug.

**What you should see:** Properly formatted, natural language responses

#### 4. **No Intermediate Data Exposure**

**What it means:** Status updates (`node_start` events) never contain user data or PII. They're generic messages about processing stages.

**Example of safe status message:** ✅ "Fetching your claims data from API..."  
**Never includes:** ❌ "Fetching claim 123456789 for John Doe..."

### CORS Configuration for Production

The endpoint includes CORS headers to allow cross-origin requests. 

**Current Default (Development):**
```
Access-Control-Allow-Origin: *
```

**⚠️ For Production:** Update the backend to whitelist only your frontend domain:

```python
# Backend configuration (for your backend team)
"Access-Control-Allow-Origin": "https://your-angular-app.company.com"
```

**Frontend consideration:** If you're running your Angular app on `https://claims.company.com`, ensure the backend CORS is configured for that exact domain.

### Secure Session Management

**Session IDs should:**
- Be unique per conversation
- Not contain user PII
- Be randomly generated
- Be reasonably long (20+ characters)

**Good session ID format:**
```typescript
`sess-${Date.now()}-${Math.random().toString(36).substring(2, 11)}`
// Example: sess-1700000000000-a8f3k9x2m
```

**Bad session ID format (DO NOT USE):**
```typescript
`sess-${userId}-${userName}` // ❌ Contains PII
`sess-1` // ❌ Too predictable
```

### Authentication & Authorization

**Important:** The streaming endpoint should be protected by your authentication layer.

**Recommended approach:**

1. **Add authentication headers** to your fetch request:

```typescript
fetch(`${this.baseUrl}/api/v1/chat/stream`, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${this.authService.getAccessToken()}` // Add auth token
  },
  body: JSON.stringify(payload)
})
```

2. **Backend validates token** before processing request

3. **User context** is derived from authenticated user, not solely from request payload

**Security best practice:** Don't rely on `user_info` in the request body for authorization. Use it only for logging/analytics. Actual user identity should come from the authenticated session.

---

---

## Troubleshooting Common Issues

### Issue 1: Not receiving any events

**Symptoms:** Fetch request succeeds (200 OK) but no events appear

**Possible causes:**
- Backend streaming is disabled
- Response buffering by proxy/gateway
- Not reading the stream correctly

**Solutions:**
1. Check backend logs to confirm streaming is enabled
2. Verify `Content-Type: text/event-stream` header in response
3. Check if proxy/gateway is buffering (look for `X-Accel-Buffering: no` header)
4. Verify your SSE parsing logic (see service implementation above)

### Issue 2: Seeing `[PII_PERSON_1]` tokens in response

**Symptoms:** Response chunks contain masked tokens like `[PII_PERSON_1]` or `[PII_DATE_1]`

**Possible causes:**
- Bug in backend safety postcheck
- Unmasking logic not running

**Solutions:**
1. This is a CRITICAL bug - report to backend team immediately
2. Include session_id for debugging
3. Do NOT display these tokens to users

### Issue 3: Status messages not updating

**Symptoms:** `node_start` events not appearing or not updating UI

**Possible causes:**
- Not handling `node_start` event type
- Status message state not bound to template
- Events being emitted but UI not updating

**Solutions:**
1. Verify `node_start` case in your event handler
2. Check `statusMessage` variable is bound in template: `{{ statusMessage }}`
3. Ensure change detection is working (Angular should handle automatically)

### Issue 4: Response chunks not displaying progressively

**Symptoms:** All chunks appear at once instead of progressively

**Possible causes:**
- Not appending chunks incrementally
- Not triggering change detection
- Buffering issue

**Solutions:**
1. Verify you're using `+=` to append: `this.currentResponse += event.data.text`
2. Check if you're resetting `currentResponse` between chunks (don't do this!)
3. Ensure template is bound to `currentResponse`: `{{ currentResponse }}`

### Issue 5: CORS errors

**Symptoms:** `CORS policy: No 'Access-Control-Allow-Origin' header` error in console

**Possible causes:**
- Backend CORS not configured
- Wrong backend URL
- Preflight OPTIONS request failing

**Solutions:**
1. Verify backend URL is correct
2. Contact backend team to configure CORS for your frontend domain
3. Check if OPTIONS preflight request is succeeding in Network tab

### Issue 6: Connection drops or timeouts

**Symptoms:** Stream stops midway, timeout errors

**Possible causes:**
- Network interruption
- Backend timeout
- No timeout configured in fetch

**Solutions:**
1. Implement retry logic (see Error Handling section)
2. Add timeout to fetch with AbortController
3. Test with longer timeout value (60 seconds recommended)

### Issue 7: Memory leaks

**Symptoms:** Browser slows down after multiple requests

**Possible causes:**
- Not unsubscribing from observables
- Not cleaning up event listeners

**Solutions:**
1. Always unsubscribe in `ngOnDestroy`:
```typescript
ngOnDestroy() {
  if (this.streamSubscription) {
    this.streamSubscription.unsubscribe();
  }
}
```
2. Store subscription: `this.streamSubscription = this.chatService.streamChat(...)`

---

## Integration Summary

### What You Need to Know

**Backend provides:**
- SSE streaming endpoint at `/api/v1/chat/stream`
- Real-time status updates as request processes
- Progressive response chunks
- Complete metadata on finish
- HIPAA-compliant safety validation

**Frontend needs to:**
- Parse SSE event stream (event type + JSON data)
- Display status updates to reduce perceived latency
- Append response chunks progressively for "typing" effect
- Handle errors gracefully with retry logic
- Manage session IDs across conversation
- Implement proper authentication

**User experience:**
- See real-time status of their request
- Start reading response immediately when it arrives
- Feel like the system is 50-70% faster
- Get clear error messages if something fails

### Key Integration Points

| Component | Your Responsibility | Backend Provides |
|-----------|-------------------|------------------|
| **Request** | Send properly formatted JSON with text, session_id, user_info | Accept and validate request |
| **Authentication** | Include auth headers in request | Validate user identity |
| **Event Parsing** | Parse SSE format (event: + data:) | Send properly formatted SSE events |
| **UI Updates** | Update status, chunks, errors in real-time | Send events as processing happens |
| **Error Handling** | Show user-friendly errors, implement retry | Send descriptive error events |
| **Session Management** | Generate and track session IDs | Use session for conversation context |

### Next Steps

1. **Start with testing:** Use browser console or cURL to verify endpoint works
2. **Implement service:** Create SSE parsing service (copy from Angular example above)
3. **Build component:** Implement chat component with state management
4. **Add UI/UX:** Create template and styling for messages and status
5. **Test thoroughly:** Try different queries, test error cases, test on mobile
6. **Production prep:** Add auth, update URLs, configure CORS, add monitoring

---

## Support & Resources

### Need Help?

**For backend issues:**
- Check backend server logs with your `session_id`
- Verify endpoint is accessible: `curl http://backend-url/api/v1/chat/stream`
- Confirm streaming is enabled in backend config

**For frontend issues:**
- Check browser console for JavaScript errors
- Inspect Network tab to see SSE events
- Verify event parsing logic matches examples above
- Test with browser console snippet (see Testing section)

**For integration questions:**
- Review this document section by section
- Compare your implementation with Angular example above
- Test individual pieces (service, then component, then template)

### Quick Reference

**Endpoint:** `POST /api/v1/chat/stream`

**Request:**
```json
{
  "text": "user question",
  "session_id": "unique-session-id",
  "user_info": {"user": "username", "user_id": "id"}
}
```

**Event Types:**
- `node_start` → Update status
- `node_complete` → Optional: mark stage complete
- `response_chunk` → Append to response
- `complete` → Store final data, clear loading
- `error` → Show error message

**Response Format:** SSE (Server-Sent Events)
```
event: <type>
data: <json>

```

---

## 📋 Quick Reference: Key Patterns

### Core Rendering Concept

SSE events arrive sequentially from the backend. The frontend determines how to display them - either replacing previous content or appending to it.

### Implementation (Copy-Paste Ready)

**Angular:**
```typescript
// Status: REPLACE (assignment operator)
if (event.type === 'node_start') {
  this.statusMessage = event.data.message;  // ← Overwrites previous
}

// Response: APPEND (concatenation operator)
if (event.type === 'response_chunk') {
  this.currentResponse += event.data.text;  // ← Accumulates
}
```

```html
<!-- Two separate UI elements -->
<div *ngIf="statusMessage" class="status">{{ statusMessage }}</div>
<div *ngIf="currentResponse" class="response">{{ currentResponse }}</div>
```

**React:**
```typescript
// Status: REPLACE (direct setState)
if (event.type === 'node_start') {
  setStatus(event.data.message);  // ← Overwrites previous
}

// Response: APPEND (functional setState)
if (event.type === 'response_chunk') {
  setResponse(prev => prev + event.data.text);  // ← Accumulates
}
```

```jsx
{/* Two separate UI elements */}
{status && <div className="status">{status}</div>}
{response && <div className="response">{response}</div>}
```

### Why This Pattern

1. **SSE Protocol:** Events are sequential and immutable per RFC 6455
2. **Industry Standard:** Modern AI chat applications (ChatGPT, Claude, Perplexity) use this pattern
3. **Separation of Concerns:** Backend streams data, frontend controls presentation
4. **Optimal UX:** Cleaner interface than showing all processing steps

### UI Component Strategy

| Component | Purpose | Update Strategy |
|-----------|---------|-----------------|
| Status Element | Shows current processing step | **REPLACE** on each update |
| Response Element | Displays AI's answer | **APPEND** each chunk |

### Event Summary

**User-Facing Status Updates (5 total):**
1. orchestrator → "Processing your request..."
2. safety_precheck → "Checking safety and privacy..."
3. intent_agent → "Understanding your question..."
4. call_claims_tool → "Retrieving your claims information..."
5. response_agent → "Preparing your response..."

**Additional Events:**
- Multiple `response_chunk` events (progressive response text)
- One `complete` event (final metadata and full response)

**Internal Processing (Not Exposed via Events):**
- check_cache, confidence_checker, build_context, response_safety_pii_precheck, response_safety_pii_postcheck, update_memory, cache_response

### Integration Checklist

- [ ] Implement streaming service with SSE parsing
- [ ] Create component with two state variables (`statusMessage`, `currentResponse`)
- [ ] Build template with separate status and response elements
- [ ] Test endpoint with curl or Postman
- [ ] Verify status updates use REPLACE pattern
- [ ] Verify response chunks use APPEND pattern
- [ ] Add error handling and retry logic
- [ ] Apply application styling

---

## Appendix: API Version & Changes

### Current Version: v2.0

**What's New in V2.0:**
- ✅ **Selective status updates**: Only 5 key milestones shown (down from 12+)
- ✅ **User-friendly messages**: Non-technical, consumer-grade wording
- ✅ **Same API contract**: No breaking changes to request/response format
- ✅ **Same backend functionality**: All internal nodes still execute and log

**V1.0 Features (Still Included):**
- ✅ SSE-based real-time streaming
- ✅ Progressive response chunks
- ✅ HIPAA-compliant safety validation
- ✅ Comprehensive error handling
- ✅ Session-based conversation context
- ✅ Intent classification with confidence scores

**Migration from V1.0:**
- ✅ **No code changes required** in existing frontend implementations
- ✅ You'll simply receive fewer `node_start`/`node_complete` events
- ✅ All response chunks and complete events work exactly the same
- ✅ Optional: Update UI to take advantage of cleaner event stream

### Monitoring for Updates

Watch for:
- New event types (will be documented here)
- Changes to event data structure
- New error reasons
- Performance improvements
- Additional user-facing nodes

Any breaking changes will be communicated with migration guide.

---

## 🤝 Support & Contact

**For Questions About:**
- **Rendering Strategy**: Refer to "Rendering Strategy: Replace vs Append" section above
- **Angular Implementation**: See Angular Integration section with complete service + component
- **React Implementation**: See React Integration section with hooks
- **Testing**: Use curl commands in Testing section
- **Backend Issues**: Contact backend team with `session_id` from your request
- **This Documentation**: Feedback welcome to improve clarity

**Remember:** The backend is production-ready. Focus your integration effort on the frontend rendering logic (replace vs append) shown in this guide.

---

*Last Updated: Based on latest V2.0 streaming implementation with selective user-facing node updates*

