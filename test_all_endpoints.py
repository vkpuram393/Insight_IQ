#!/usr/bin/env python3
"""
Test all endpoints from TEMP_ENDPOINTS.md to ensure nothing is broken
"""
import requests
import json
import time
import sys

BASE_URL = "http://localhost:8000"

def test_endpoint(name, method, url, payload=None, expected_status=200):
    """Test a single endpoint"""
    try:
        if method == "GET":
            response = requests.get(url, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=payload, timeout=10, headers={"Content-Type": "application/json"})
        elif method == "DELETE":
            response = requests.delete(url, timeout=10)
        else:
            print(f"❌ {name}: Unknown method {method}")
            return False
        
        if response.status_code == expected_status:
            print(f"✅ {name}: Status {response.status_code}")
            try:
                data = response.json()
                if isinstance(data, dict) and "error" in data:
                    print(f"   ⚠️  Response contains error: {data.get('error')}")
                    return False
            except:
                pass
            return True
        else:
            print(f"❌ {name}: Expected {expected_status}, got {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"❌ {name}: Connection refused - is server running?")
        return False
    except Exception as e:
        print(f"❌ {name}: Error - {str(e)}")
        return False

def main():
    print("🧪 Testing all endpoints from TEMP_ENDPOINTS.md\n")
    
    # Wait for server to be ready
    print("⏳ Waiting for server to be ready...")
    for i in range(10):
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=2)
            if response.status_code == 200:
                print("✅ Server is ready!\n")
                break
        except:
            time.sleep(1)
    else:
        print("❌ Server is not responding. Please start the server first.")
        sys.exit(1)
    
    results = []
    
    # 1. Health Check
    print("1. Testing Health Check")
    results.append(("Health Check", test_endpoint("Health Check", "GET", f"{BASE_URL}/health")))
    print()
    
    # 2. Utils Health Check
    print("2. Testing Utils Health Check")
    results.append(("Utils Health", test_endpoint("Utils Health", "GET", f"{BASE_URL}/utils/health")))
    print()
    
    # 3. Intent Classification
    print("3. Testing Intent Classification")
    results.append(("Test Intent", test_endpoint(
        "Test Intent",
        "POST",
        f"{BASE_URL}/utils/test-intent",
        {"text": "why was my claim rejected"}
    )))
    print()
    
    # 4. Intent Agent
    print("4. Testing Intent Agent")
    results.append(("Test Intent Agent", test_endpoint(
        "Test Intent Agent",
        "POST",
        f"{BASE_URL}/utils/test-intent-agent",
        {"text": "Claim 12345 was rejected, why?", "user_info": {"user_id": "test_user"}}
    )))
    print()
    
    # 5. Cache Operations
    print("5. Testing Cache Operations")
    results.append(("Test Cache Set", test_endpoint(
        "Test Cache Set",
        "POST",
        f"{BASE_URL}/utils/test-cache",
        {"key": "test_key_123", "value": {"data": "test value", "number": 42}, "ttl_seconds": 3600}
    )))
    results.append(("Test Cache Get", test_endpoint(
        "Test Cache Get",
        "POST",
        f"{BASE_URL}/utils/test-cache",
        {"key": "test_key_123"}
    )))
    print()
    
    # 6. Persistence/Telemetry
    print("6. Testing Persistence/Telemetry")
    results.append(("Test Persistence", test_endpoint(
        "Test Persistence",
        "POST",
        f"{BASE_URL}/utils/test-persistence",
        {"event_type": "CACHE_HIT", "session_id": "test_session_456", "data": {"key": "some_cache_key", "hit_count": 5}}
    )))
    print()
    
    # 7. Session Memory
    print("7. Testing Session Memory")
    session_id = f"test_session_{int(time.time())}"
    results.append(("Test Session History", test_endpoint(
        "Test Session History",
        "POST",
        f"{BASE_URL}/utils/test-session-history",
        {"session_id": session_id, "role": "user", "content": "Hello, my claim number is 12345"}
    )))
    print()
    
    # 8. Context Building
    print("8. Testing Context Building")
    results.append(("Test Context Building", test_endpoint(
        "Test Context Building",
        "POST",
        f"{BASE_URL}/utils/test-context-building",
        {
            "text": "What's my claim status?",
            "intent": "claim_status",
            "confidence": 0.92,
            "entities": {"claim_number": "12345678"},
            "slots": {"claim_number": "12345678"},
            "required_slots": ["claim_number"],
            "missing_slots": [],
            "session_id": session_id,
            "uuid": "req-uuid-context-test",
            "domain": "claims",
            "user_info": {"user_id": "member_222"}
        }
    )))
    print()
    
    # 9. Safety Precheck
    print("9. Testing Safety Precheck")
    results.append(("Test Safety Precheck", test_endpoint(
        "Test Safety Precheck",
        "POST",
        f"{BASE_URL}/utils/test-safety-precheck",
        {"text": "What is my claim status?", "session_id": "safe_test"}
    )))
    print()
    
    # 10. Safety Postcheck
    print("10. Testing Safety Postcheck")
    results.append(("Test Safety Postcheck", test_endpoint(
        "Test Safety Postcheck",
        "POST",
        f"{BASE_URL}/utils/test-safety-postcheck",
        {"text": "Generated response to check", "session_id": "safety_test"}
    )))
    print()
    
    # 11. Clarification
    print("11. Testing Clarification")
    results.append(("Test Clarification", test_endpoint(
        "Test Clarification",
        "POST",
        f"{BASE_URL}/utils/test-clarification",
        {"text": "why was my claim rejected"}
    )))
    print()
    
    # 12. NEW: Confidence Checker (Low Confidence)
    print("12. Testing Confidence Checker (Low Confidence)")
    results.append(("Test Confidence Checker (Low)", test_endpoint(
        "Test Confidence Checker (Low)",
        "POST",
        f"{BASE_URL}/utils/test-confidence-checker",
        {
            "text": "why was my claim rejected",
            "intent": "claim_rejection_reason",
            "confidence": 0.45,
            "entities": {},
            "session_id": "test-session-123",
            "uuid": "req-uuid-456",
            "domain": "claims",
            "user_info": {"user_id": "member_222"}
        }
    )))
    print()
    
    # 13. NEW: Confidence Checker (High Confidence)
    print("13. Testing Confidence Checker (High Confidence)")
    results.append(("Test Confidence Checker (High)", test_endpoint(
        "Test Confidence Checker (High)",
        "POST",
        f"{BASE_URL}/utils/test-confidence-checker",
        {
            "text": "what is the status of claim 12345678",
            "intent": "claim_status",
            "confidence": 0.92,
            "entities": {"claim_number": "12345678"},
            "session_id": "test-session-123",
            "uuid": "req-uuid-789",
            "domain": "claims",
            "user_info": {"user_id": "member_222"}
        }
    )))
    print()
    
    # 14. Claims API
    print("14. Testing Claims API")
    results.append(("Test Claims API", test_endpoint(
        "Test Claims API",
        "POST",
        f"{BASE_URL}/utils/test-claims-api",
        {"text": "Get status for claim 12345", "intent": "claim_status", "entities": {"claim_number": "12345"}}
    )))
    print()
    
    # 15. Response Agent
    print("15. Testing Response Agent")
    results.append(("Test Response Agent", test_endpoint(
        "Test Response Agent",
        "POST",
        f"{BASE_URL}/utils/test-response-agent",
        {"text": "Why was my claim rejected?"}
    )))
    print()
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    print()
    
    if passed < total:
        print("Failed tests:")
        for name, result in results:
            if not result:
                print(f"  ❌ {name}")
        sys.exit(1)
    else:
        print("🎉 All tests passed!")
        sys.exit(0)

if __name__ == "__main__":
    main()

