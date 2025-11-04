# Intent Classifier 



### Classification & Extraction
1. **CVS intents** with keyword-weighted scoring 
2. **Entity extraction** (claim IDs, member IDs, dates, amounts) using regex
3. **Confidence scores** normalized by matched keywords

### Slot Management
5. **Required slots per intent** (e.g., claim_status needs claim_id)
6. **LLM context extraction** when slots missing from query
7. **Ask user** if slots still missing after LLM extraction

### Response Generation
8. **CVS templates** (claim status, rejection, drug info, pharmacy, prescriber, ....)
9. **Query focus detection** (drug, pricing, status, rejection)
10. **Template fallback to LLM** if required fields missing

### Memory & Links
11. **Short-term memory** (session conversation history, 5-10 messages)
12. **Short-term cache** (API data per session)
13. **Long-term FAQ** (MongoDB with embeddings, vector search)
14. **Long-term follow-ups** (question pairs with scoring)
15. **Suggested links** (3-5 per response, from FAQ + follow-ups)

### Data Structures
16. **Intent → Entity → Slot** relationship (see example below)
17. **MongoDB schema** for question pairs (initial + follow-ups with scores)
18. **Follow-up scoring** 
19. **Embedding model** (embedding model))
20. **Output dictionary** with intent, entities, response, links, api_data

## 📊 Example: Intent → Entity → Slot

```python
# Query
"What is the status of claim CLM123?"

# Intent Classification
{"intent": "claim_status", "confidence": 0.92}

# Entity Extraction
{"entities": {"claim_id": ["CLM123"]}}

# Slot Validation
{"slots": {"claim_id": "CLM123"}, "missing": []}
# ✅ All required slots filled → Proceed to API
```

## 🔄 Processing Flows

**Flow 1:** Complete info → Intent → Entity → API → Template → Response  
**Flow 2:** Missing info → Intent → LLM extract from history(if available) → API → Response  
**Flow 3:** Still missing → Intent → Ask user → Response  
**Flow 4:** Template fails → LLM fallback → Response
**Flow 5:** Mixed/unclear info → LLM → API → Response
**Flow 6:** Mixed/unclear info → LLM → RAG → Response






