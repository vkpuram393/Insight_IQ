#!/bin/bash

# Test CVS Intent Classifier via HTTP Endpoints
# Usage: ./test_cvs_classifier_endpoints.sh

BASE_URL="http://localhost:8000"
PASSED=0
FAILED=0

echo "=========================================="
echo "🧪 CVS Intent Classifier Endpoint Tests"
echo "=========================================="
echo ""
echo "Prerequisites:"
echo "1. Server running: python main.py"
echo "2. Config: use_cvs_intent_classifier = True"
echo ""
echo "Starting tests..."
echo ""

test_intent() {
    local name="$1"
    local query="$2"
    local expected_intent="$3"
    local expected_confidence="$4"
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Test: $name"
    echo "Query: \"$query\""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    response=$(curl -s -X POST "$BASE_URL/utils/test-intent-agent" \
        -H 'Content-Type: application/json' \
        -d "{\"text\":\"$query\"}")
    
    if echo "$response" | jq empty 2>/dev/null; then
        intent=$(echo "$response" | jq -r '.intent')
        confidence=$(echo "$response" | jq -r '.confidence')
        entities=$(echo "$response" | jq -r '.entities')
        is_complex=$(echo "$response" | jq -r '.is_complex')
        needs_clarification=$(echo "$response" | jq -r '.needs_clarification')
        
        echo "Result:"
        echo "  Intent: $intent"
        echo "  Confidence: $confidence"
        echo "  Entities: $entities"
        echo "  Is Complex: $is_complex"
        echo "  Needs Clarification: $needs_clarification"
        
        if [ "$intent" = "$expected_intent" ]; then
            echo "✅ PASS: Intent matches expected ($expected_intent)"
            ((PASSED++))
        else
            echo "❌ FAIL: Expected intent '$expected_intent', got '$intent'"
            ((FAILED++))
        fi
    else
        echo "❌ FAIL: Invalid JSON response"
        echo "$response"
        ((FAILED++))
    fi
    echo ""
}

# ========== TEST CASES ==========

# Route 1: Simple API Query
test_intent "Route 1: Simple API Query" \
    "Where is claim CLM12345?" \
    "claim_status" \
    "0.70"

# Route 2: Complex Query
test_intent "Route 2: Complex Query (Aggregation)" \
    "Summarize my claims for October" \
    "claim_status" \
    "0.50"

# Route 3: Out of Scope
test_intent "Route 3: Out of Scope" \
    "Tell me a joke" \
    "out_of_scope" \
    "0.00"

# Route 5: Low Confidence, No Entity
test_intent "Route 5: General Question" \
    "What medication did I get?" \
    "drug_info" \
    "0.55"

# Route 6: Missing Slots
test_intent "Route 6: Missing Required Slots" \
    "Show me my claim" \
    "claim_status" \
    "0.70"

# Route 7: Low Confidence BUT Has Entity
test_intent "Route 7: Low Conf + Entity" \
    "Check status CLM12345" \
    "claim_status" \
    "0.50"

# Route 8: Greeting
test_intent "Route 8: Greeting" \
    "Hello" \
    "greeting" \
    "1.00"

# Route 10: FAQ Search
test_intent "Route 10: FAQ Search" \
    "What is prior authorization?" \
    "prior_auth_info" \
    "0.90"

# Route 11: Multi-Intent (Missing Slots)
test_intent "Route 11: Multi-Intent Query" \
    "Why was my claim rejected and when will I get my medication?" \
    "rejection_reasons" \
    "0.90"

# Route 12: Empty Query
test_intent "Route 12: Empty Query" \
    "" \
    "out_of_scope" \
    "0.00"

# Additional CVS-Specific Tests
test_intent "Drug Info Query" \
    "What drug is on my claim?" \
    "drug_info" \
    "0.70"

test_intent "Pharmacy Info Query" \
    "Where was my prescription filled?" \
    "pharmacy_info" \
    "0.80"

test_intent "Pricing Info Query" \
    "How much do I owe?" \
    "pricing_info" \
    "0.80"

test_intent "Rejection with Entity" \
    "Why was claim CLM12345 rejected?" \
    "rejection_reasons" \
    "0.95"

test_intent "Multi-Claim Query" \
    "Show me claims CLM111 and CLM222" \
    "claim_status" \
    "0.75"

# ========== SUMMARY ==========

echo "=========================================="
echo "📊 TEST SUMMARY"
echo "=========================================="
echo "✅ Passed: $PASSED"
echo "❌ Failed: $FAILED"
echo "=========================================="

if [ $FAILED -eq 0 ]; then
    echo "🎉 ALL TESTS PASSED!"
    exit 0
else
    echo "⚠️  Some tests failed. Review output above."
    exit 1
fi

