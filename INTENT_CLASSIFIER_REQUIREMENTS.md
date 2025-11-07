## State
- [ ] Define Pydantic models for intent classifier outputs (IntentResult, EntityResult, SlotValidation, SuggestedLink)

## Intent Classification Core
- [ ] Build keyword-weighted CVS intent classifier with 28+ intents in `backend/services/cvs_intent_classifier.py`
- [ ] Define keyword weights for all intents: claim_status, drug_info, rejection_reasons, pharmacy_info, prescriber_info, pricing_info, coverage_info, prior_auth_info, appeal_info, member_info, prescription_history, compound_info, mail_order_info, network_info, daw_info, dur_info, date_range_claims, multi_claim_summary, audit_info, general_question, greeting, help, out_of_scope, etc.
- [ ] Implement whole-word matching using regex word boundaries to avoid substring false positives (e.g., "claim" should not match "exclaim")
- [ ] Normalize confidence scores by number of MATCHED keywords, not total words in query: `score = sum(keyword_weights) / len(matched_keywords)`
- [ ] Strip punctuation from queries before matching: `query_cleaned = re.sub(r'[^\w\s]', ' ', query_lower)`
- [ ] Return top intent with confidence score ≥ 0.50 for direct routing; if < 0.50, route to out_of_scope

### Keyword Weight Tuning Best Practices
- [ ] **Remove overly generic keywords** that appear in many intents: 'is' (0.2), 'my' (0.3), 'claim' (in rejection_reasons), 'why' (0.3)
  - Problem: Generic keywords with low weights DILUTE the average score when matched alongside strong keywords
  - Example: `rejection_reasons` matched 'rejected' (1.0) + 'why' (0.3) = 0.65 avg, loses to `claim_status` with 'claim' (0.8) = 0.80 avg
- [ ] **Use domain-specific keywords** with high weights: 'rejected' (1.0), 'denied' (0.9), 'status' (0.9), 'drug' (0.9)
- [ ] **Boost location/action keywords** when they're primary indicators: 'where' (0.7), 'track' (0.6), 'check' (0.6)
- [ ] **Avoid keyword overlap** between similar intents: Don't put 'claim' in both `claim_status` AND `rejection_reasons`
  - Instead: Use specific differentiators like 'rejected', 'denied' for rejection_reasons; 'status', 'where', 'track' for claim_status
- [ ] **Test for false positives** after adding new keywords: Run test queries to ensure new keywords don't hijack unrelated intents
- [ ] **Monitor confidence scores**: If scores cluster around threshold (0.45-0.55), adjust keyword weights to create clearer separation

## Entity Extraction
- [ ] Implement regex-based entity extractor in `backend/services/entity_extractor.py`
- [ ] Define regex patterns for: claim_id (`CLM\d+`, `\d{8,12}`), member_id (`MBR\d+`, `\d{9,11}`), prescription_id (`RX\d+`), phone (`\d{3}-\d{3}-\d{4}`), amounts (`\$\d+\.?\d*`)
- [ ] Define date extraction patterns: month names (January, February, ...), relative dates (last month, yesterday), quarters (Q1, Q2), date formats (MM/DD/YYYY, YYYY-MM-DD)
- [ ] Support extraction of multiple claim IDs in single query for batch operations
- [ ] Validate extracted entities: check claim ID format, check member ID checksum (if applicable), validate date ranges
- [ ] Return EntityResult with extraction method='regex' and validation status

## LLM Entity Extraction (Integration with LLM)
- [ ] Implement LLM entity extraction node in `backend/graph/nodes.py::llm_entity_extraction_node`
- [ ] When regex extraction fails to find required slots, use LLM to extract from conversation history
- [ ] Build prompt template: "Based on the conversation history, extract the following entities: {missing_slots}. Conversation: {last_5_messages}"
- [ ] Get appropriate LLM service
- [ ] Parse LLM response and update state with extracted entities
- [ ] Return EntityResult with extraction method='llm' and update slot validation
- [ ] If LLM extraction still fails, route to "ask user" node

## Slot Validation (Integration with Template System)
- [ ] Define required slots per intent in `backend/services/cvs_template_registry.py`
- [ ] Map intents to required fields: claim_status needs [claim_number], rx_details needs [prescription_id], member_info needs [member_id]
- [ ] Define optional slots that improve accuracy but aren't required: member_id for claim_status (improves specificity)
- [ ] Implement slot validation node in `backend/graph/nodes.py::slot_validation_node`
- [ ] Check if all required slots are filled from extracted entities
- [ ] If missing slots → route to LLM entity extraction node
- [ ] If still missing after LLM → route to "ask user" node
- [ ] Return SlotValidation model with missing_slots list

## Routing Integration (LangGraph Edges)
- [ ] Build conditional edge in `backend/graph/edges.py::route_after_intent_classification`
- [ ] Route based on intent type: if intent='out_of_scope' → polite_rejection, if intent='general_question' → rag_knowledge_base, if intent in claim_intents → slot_validation, if intent='greeting' → return greeting template
- [ ] Build conditional edge `route_slot_validation`: if missing_slots → llm_entity_extraction, else → data_cache_lookup
- [ ] Build conditional edge `route_llm_extraction`: if still missing → ask_user, else → data_cache_lookup
- [ ] Implement confidence threshold routing: if confidence < 0.6 → llm_double_check node

## Suggested Links Generation (Integration with Memory)
- [ ] Implement suggested links generation within intent classifier in `backend/services/cvs_intent_classifier.py::generate_suggested_links`
- [ ] For each intent, define 2-3 common follow-up intents: claim_status → [rejection_reasons, drug_info, appeal_info]
- [ ] Query FAQ repository (or other TBD) for top 2 relevant FAQs based on current intent 
- [ ] Query long-term follow-up memory (MongoDB) for top 3 popular follow-ups for current intent
- [ ] Aggregate and deduplicate links: remove duplicate intents, prioritize by relevance + popularity
- [ ] Return list of SuggestedLink models (limit to 3-5 links total)
- [ ] Format links for frontend display as buttons

## Integration with Data Cache
- [ ] After slot validation succeeds, check data cache for existing API response in `backend/graph/nodes.py::data_cache_lookup_node` 
- [ ] Cache key format: `{user_session_id}:{claim_id}:{endpoint}` with x-minute TTL
- [ ] If cache hit → skip API call, use cached data for template rendering
- [ ] If cache miss → call API, save response to cache, then render template

## Integration with Query Focus Detection
- [ ] After data cache lookup (hit or miss), detect query focus in `backend/graph/nodes.py::query_focus_detection_node`
- [ ] Analyze query keywords to determine focus: "drug" keywords → focus='drug', "cost/price" keywords → focus='pricing', "status" keywords → focus='status'
- [ ] Use query focus + intent to select most appropriate template: intent=claim_status + focus=drug → use drug_details template
- [ ] This allows correction of intent misclassification: if intent is wrong but focus is right, correct template is still selected

## PII Handling in Intent Classifier
- [ ] Implement PII masking in entity extraction results before logging
- [ ] Mask claim IDs as CLM*** in logs: `masked_claim_id = f"CLM{'*' * (len(claim_id) - 3)}"`
- [ ] Mask member IDs as MBR*** in logs: `masked_member_id = f"MBR{'*' * (len(member_id) - 3)}"`
- [ ] Never log full member names, SSNs, or phone numbers extracted from queries
- [ ] Before storing queries in long-term memory, remove all PII: replace claim IDs with "my claim", member IDs with "my account", names with "I"

## Integration with Long-Term Memory (MongoDB)
- [ ] After intent classification, normalize query and store in follow-up memory if it's a follow-up question
- [ ] Query normalization for storage: remove PII (claim IDs, member IDs, names), convert to lowercase, use canonical phrasing
- [ ] Check if current question is a follow-up to previous question in session history
- [ ] If yes, search long-term memory for similar initial question (cosine similarity > 0.75 on embeddings)
- [ ] If found, search within that initial question's follow-ups for similar follow-up (similarity > 0.75)
- [ ] If match found → increment score; if not found → insert new follow-up with score=1
- [ ] Use follow-up scores to generate suggested links for next turn

## Confidence Scoring & Clarification
- [ ] Implement confidence thresholds in `backend/services/cvs_intent_classifier.py::classify`
- [ ] Low confidence (< 0.6): Route to LLM double-check node
- [ ] Medium confidence (0.6 - 0.75): Proceed but add hedging language in response ("It looks like...", "This might be...") (could be removed for simplicity and use only 2 high and low confidence)
- [ ] High confidence (≥ 0.75): Proceed with direct response
- [ ] Track confidence calibration: log (predicted_confidence, actual_correctness) for evaluation
- [ ] If ambiguous intent detected (top 2 intents have similar scores, diff < 0.1), ask clarifying question: "Did you mean to check claim status or rejection reasons?"

## Conversation Continuity & Multi-Turn Slot Filling
- [ ] Implement session state persistence in `backend/graph/state.py::ChatbotState` with `waiting_for_slot` and `pending_intent` fields
- [ ] When asking user for missing slot, save continuation state: `waiting_for_slot = 'claim_id'`, `pending_intent = 'claim_status'`
- [ ] In next turn, check if continuation state exists in `backend/graph/nodes.py::enhanced_intent_classifier_node`
- [ ] If `waiting_for_slot` is set → skip intent classification, restore `pending_intent`, extract entities from current query
- [ ] **CRITICAL**: Detect intent change to prevent infinite loops: if user's new query has strong different intent (confidence ≥ 0.70), clear continuation and honor new intent
- [ ] Clear continuation flags (`waiting_for_slot`, `pending_intent`) when: (1) API call succeeds, (2) user changes intent, or (3) RAG path is taken
- [ ] Persist continuation state in `session.metadata` in `backend/api/main.py` to survive page refreshes
- [ ] Pass `waiting_for_slot` and `pending_intent` between `main.py` → `chatbot_graph.py` → `nodes.py` for proper restoration

