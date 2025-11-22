// ============================================================================
// POSTMAN TEST SCRIPT - Visual Indicators
// ============================================================================
// Copy this entire script into the "Tests" tab of your Postman requests
// Works with: /api/v1/chat, /utils/test-intent, /utils/test-intent-agent, etc.

// ============================================================================
// CONFIGURATION (Customize these thresholds)
// ============================================================================
const CONFIDENCE_THRESHOLD = parseFloat(pm.environment.get("confidence_threshold") || "0.70");
const RESPONSE_TIME_THRESHOLD = parseFloat(pm.environment.get("response_time_threshold") || "5000");

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

function getStatusIcon(status, value = null) {
    if (status === 'error') return '🔴';
    if (status === 'warning') return '🟡';
    if (status === 'success') return '✅';
    if (status === 'info') return 'ℹ️';
    
    // For confidence scores
    if (value !== null && value < CONFIDENCE_THRESHOLD) return '🔴';
    if (value !== null && value >= CONFIDENCE_THRESHOLD) return '✅';
    
    return '⚪';
}

function getColorClass(value, threshold, reverse = false) {
    if (reverse) {
        return value > threshold ? 'error' : 'success';
    }
    return value < threshold ? 'error' : 'success';
}

// ============================================================================
// PARSE RESPONSE
// ============================================================================
let jsonData;
try {
    jsonData = pm.response.json();
} catch (e) {
    pm.test("Response is valid JSON 🔴", function () {
        pm.expect.fail("Response is not valid JSON: " + pm.response.text());
    });
    return;
}

// ============================================================================
// 1. STATUS CODE CHECK
// ============================================================================
pm.test(`Status Code ${pm.response.code === 200 ? '✅' : '🔴'} ${pm.response.code}`, function () {
    pm.response.to.have.status(200);
});

// ============================================================================
// 2. CONFIDENCE SCORE INDICATORS
// ============================================================================
if (jsonData.confidence !== undefined && jsonData.confidence !== null) {
    let confidence = parseFloat(jsonData.confidence);
    let isLow = confidence < CONFIDENCE_THRESHOLD;
    let icon = getStatusIcon('confidence', confidence);
    let status = isLow ? 'LOW' : 'GOOD';
    let color = isLow ? 'red' : 'green';
    
    // Test assertion (shows green/red in Test Results)
    pm.test(`Confidence Score ${icon} ${confidence} (${status})`, function () {
        pm.expect(confidence).to.be.a('number');
        pm.expect(confidence).to.be.at.least(0);
        pm.expect(confidence).to.be.at.most(1);
        
        if (isLow) {
            // This will fail, showing red X in test results
            pm.expect(confidence).to.be.above(
                CONFIDENCE_THRESHOLD, 
                `⚠️ LOW CONFIDENCE: ${confidence} is below threshold (${CONFIDENCE_THRESHOLD})`
            );
        }
    });
    
    // Console output with visual indicator
    if (isLow) {
        console.log(`🔴 CONFIDENCE: ${confidence} (LOW - Below Threshold ${CONFIDENCE_THRESHOLD})`);
    } else {
        console.log(`✅ CONFIDENCE: ${confidence} (GOOD)`);
    }
    
    // Set environment variable for visualization
    pm.environment.set("last_confidence", confidence);
    pm.environment.set("last_confidence_status", isLow ? "low" : "good");
    pm.environment.set("last_confidence_icon", icon);
}

// ============================================================================
// 3. ERROR INDICATORS
// ============================================================================
if (jsonData.error || jsonData.detail || jsonData.error_occurred) {
    let errorMessage = jsonData.error || jsonData.detail || jsonData.error_message || 'Unknown error';
    let icon = getStatusIcon('error');
    
    // Test assertion (will fail, showing red X)
    pm.test(`Error Detected ${icon}`, function () {
        pm.expect(jsonData.error || jsonData.detail).to.not.exist;
    });
    
    // Console output
    console.log(`🔴 ERROR: ${errorMessage}`);
    pm.environment.set("has_error", "true");
    pm.environment.set("error_message", errorMessage);
} else {
    // Success indicator
    let icon = getStatusIcon('success');
    pm.test(`No Errors ${icon}`, function () {
        pm.expect(jsonData.error).to.not.exist;
        pm.expect(jsonData.detail).to.not.exist;
    });
    console.log(`✅ No errors detected`);
    pm.environment.set("has_error", "false");
}

// ============================================================================
// 4. INTENT CLASSIFICATION INDICATORS
// ============================================================================
if (jsonData.intent) {
    let isUnknown = jsonData.intent === 'unknown' || jsonData.intent === null;
    let icon = isUnknown ? '🟡' : '✅';
    
    pm.test(`Intent ${icon} ${jsonData.intent}`, function () {
        pm.expect(jsonData.intent).to.exist;
        if (isUnknown) {
            console.log(`🟡 WARNING: Intent is 'unknown'`);
        } else {
            console.log(`✅ Intent: ${jsonData.intent}`);
        }
    });
}

// ============================================================================
// 5. RESPONSE TIME INDICATORS
// ============================================================================
let responseTime = pm.response.responseTime;
let isSlow = responseTime > RESPONSE_TIME_THRESHOLD;
let icon = isSlow ? '🟡' : '✅';

pm.test(`Response Time ${icon} ${responseTime}ms`, function () {
    pm.expect(responseTime).to.be.below(RESPONSE_TIME_THRESHOLD);
});

if (isSlow) {
    console.log(`🟡 SLOW RESPONSE: ${responseTime}ms (threshold: ${RESPONSE_TIME_THRESHOLD}ms)`);
} else {
    console.log(`✅ Response Time: ${responseTime}ms`);
}

// Also check response_time_ms if present in response
if (jsonData.response_time_ms) {
    let apiResponseTime = jsonData.response_time_ms;
    if (apiResponseTime > RESPONSE_TIME_THRESHOLD) {
        console.log(`🟡 API Response Time: ${apiResponseTime}ms (slow)`);
    }
}

// ============================================================================
// 6. NEEDS CLARIFICATION INDICATOR
// ============================================================================
if (jsonData.needs_clarification !== undefined) {
    let icon = jsonData.needs_clarification ? '🟡' : '✅';
    pm.test(`Clarification Status ${icon}`, function () {
        pm.expect(jsonData.needs_clarification).to.be.a('boolean');
    });
    
    if (jsonData.needs_clarification) {
        console.log(`🟡 Needs Clarification: ${jsonData.clarifying_question || 'Yes'}`);
    } else {
        console.log(`✅ No clarification needed`);
    }
}

// ============================================================================
// 7. RESPONSE VALIDATION
// ============================================================================
if (jsonData.response) {
    pm.test(`Response Text ${getStatusIcon('success')}`, function () {
        pm.expect(jsonData.response).to.be.a('string');
        pm.expect(jsonData.response.length).to.be.above(0);
    });
    console.log(`✅ Response text present (${jsonData.response.length} chars)`);
}

// ============================================================================
// 8. SUMMARY OUTPUT
// ============================================================================
console.log('\n' + '='.repeat(60));
console.log('📊 TEST SUMMARY');
console.log('='.repeat(60));
console.log(`Status: ${pm.response.code === 200 ? '✅ SUCCESS' : '🔴 FAILED'} (${pm.response.code})`);
console.log(`Response Time: ${responseTime}ms ${isSlow ? '🟡 (SLOW)' : '✅'}`);

if (jsonData.confidence !== undefined) {
    let conf = parseFloat(jsonData.confidence);
    let confIcon = conf < CONFIDENCE_THRESHOLD ? '🔴' : '✅';
    console.log(`Confidence: ${confIcon} ${conf} ${conf < CONFIDENCE_THRESHOLD ? '(LOW)' : '(GOOD)'}`);
}

if (jsonData.intent) {
    console.log(`Intent: ${jsonData.intent === 'unknown' ? '🟡' : '✅'} ${jsonData.intent}`);
}

if (jsonData.error || jsonData.detail) {
    console.log(`Error: 🔴 ${jsonData.error || jsonData.detail}`);
} else {
    console.log(`Errors: ✅ None`);
}

console.log('='.repeat(60) + '\n');

// ============================================================================
// 9. SET ENVIRONMENT VARIABLES FOR NEXT REQUEST
// ============================================================================
pm.environment.set("last_request_status", pm.response.code === 200 ? "success" : "error");
pm.environment.set("last_response_time", responseTime);

