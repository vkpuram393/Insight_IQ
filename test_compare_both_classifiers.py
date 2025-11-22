"""
Compare Keyword vs Embedding Classifiers
Tests all 118 use cases with both classifiers and shows detailed comparison
"""

import sys
import os
from typing import Dict, List

# Suppress warnings
os.environ['PYTHONWARNINGS'] = 'ignore'

from agents.keyword_classifier import get_cvs_intent_classifier
from agents.embedded_classifier import CVSIntentEmbedded
from test_use_cases import TEST_CASES

print("=" * 120)
print("🔬 CLASSIFIER COMPARISON: Keywords vs Embeddings")
print("=" * 120)

# Filter for CVS queries only (UC1-UC45)
cvs_test_cases = [
    test for test in TEST_CASES 
    if 'UC' in test.get('description', '') and ':' in test.get('description', '')
]

# Extract UC number and filter UC1-UC45
filtered_tests = []
for test in cvs_test_cases:
    desc = test.get('description', '')
    if 'UC' in desc:
        try:
            uc_num = int(desc.split('UC')[1].split(':')[0])
            if 1 <= uc_num <= 45:
                filtered_tests.append(test)
        except:
            pass

print(f"\n📊 Testing {len(filtered_tests)} CVS production queries (UC1-UC45) with BOTH classifiers...")
print("🎯 Method 1: Keyword-based (fast, rule-based)")
print("🎯 Method 2: Embedding-based (semantic similarity)")

# Use filtered test cases
TEST_CASES = filtered_tests

# Initialize both classifiers
print("\n⏳ Initializing classifiers...")
print("   Loading keyword classifier...")
keyword_classifier = get_cvs_intent_classifier()
print("   ✅ Keyword classifier ready")

print("   Loading embedding classifier (this takes ~30-40 seconds)...")
embedding_classifier = CVSIntentEmbedded()
print("   ✅ Embedding classifier ready")

print("\n✅ Both classifiers ready\n")

# Run tests
keyword_results = []
embedding_results = []

for i, test in enumerate(TEST_CASES, 1):
    query = test['query']
    expected = test['expected_intent']
    
    # Show progress every 20 queries
    if i % 20 == 0:
        print(f"Progress: {i}/{len(TEST_CASES)} tests completed...")
    
    # Test with keyword classifier
    kw_result = keyword_classifier.classify(query)
    kw_intent = kw_result['intent']
    kw_pass = (kw_intent == expected)
    keyword_results.append({
        'query': query,
        'expected': expected,
        'actual': kw_intent,
        'confidence': kw_result['confidence'],
        'passed': kw_pass
    })
    
    # Test with embedding classifier
    emb_result = embedding_classifier.classify(query)
    emb_intent = emb_result['intent']
    emb_pass = (emb_intent == expected)
    embedding_results.append({
        'query': query,
        'expected': expected,
        'actual': emb_intent,
        'confidence': emb_result['confidence'],
        'passed': emb_pass
    })

# Calculate statistics
kw_passed = sum(1 for r in keyword_results if r['passed'])
emb_passed = sum(1 for r in embedding_results if r['passed'])
total = len(TEST_CASES)

kw_accuracy = (kw_passed / total) * 100
emb_accuracy = (emb_passed / total) * 100

# Find differences
both_correct = sum(1 for i in range(total) if keyword_results[i]['passed'] and embedding_results[i]['passed'])
both_wrong = sum(1 for i in range(total) if not keyword_results[i]['passed'] and not embedding_results[i]['passed'])
emb_fixed = sum(1 for i in range(total) if not keyword_results[i]['passed'] and embedding_results[i]['passed'])
emb_broke = sum(1 for i in range(total) if keyword_results[i]['passed'] and not embedding_results[i]['passed'])

# Print results
print("\n" + "=" * 120)
print("📊 RESULTS SUMMARY")
print("=" * 120)

print(f"\n🔤 KEYWORD CLASSIFIER:")
print(f"   ✅ Passed: {kw_passed}/{total}")
print(f"   ❌ Failed: {total - kw_passed}/{total}")
print(f"   📈 Accuracy: {kw_accuracy:.1f}%")

print(f"\n🧠 EMBEDDING CLASSIFIER:")
print(f"   ✅ Passed: {emb_passed}/{total}")
print(f"   ❌ Failed: {total - emb_passed}/{total}")
print(f"   📈 Accuracy: {emb_accuracy:.1f}%")

diff = emb_accuracy - kw_accuracy
if diff > 0:
    indicator = "📈"
elif diff < 0:
    indicator = "📉"
else:
    indicator = "➖"

print(f"\n📊 COMPARISON:")
print(f"   Improvement: {diff:+.1f}% {indicator}")
print(f"   Both Correct: {both_correct}")
print(f"   Both Wrong: {both_wrong}")
print(f"   Embedding Fixed: {emb_fixed} 🎯")
print(f"   Embedding Broke: {emb_broke} ⚠️")

# Show improvements (embedding fixed what keyword missed)
if emb_fixed > 0:
    print("\n" + "=" * 120)
    print(f"✅ IMPROVEMENTS: Embedding classifier fixed {emb_fixed} queries that keyword missed")
    print("=" * 120)
    
    improvements = []
    for i in range(total):
        if not keyword_results[i]['passed'] and embedding_results[i]['passed']:
            improvements.append((i, keyword_results[i], embedding_results[i]))
    
    for idx, (i, kw, emb) in enumerate(improvements[:10], 1):  # Show first 10
        test = TEST_CASES[i]
        print(f"\n   Test {i+1}: \"{test['query']}\"")
        print(f"      Expected: {test['expected_intent']}")
        print(f"      Keyword got: {kw['actual']} ❌")
        print(f"      Embedding got: {emb['actual']} ✅ (confidence: {emb['confidence']:.2f})")
    
    if len(improvements) > 10:
        print(f"\n   ... and {len(improvements) - 10} more improvements")

# Show regressions (embedding broke what keyword got right)
if emb_broke > 0:
    print("\n" + "=" * 120)
    print(f"⚠️  REGRESSIONS: Embedding classifier broke {emb_broke} queries that keyword got right")
    print("=" * 120)
    
    regressions = []
    for i in range(total):
        if keyword_results[i]['passed'] and not embedding_results[i]['passed']:
            regressions.append((i, keyword_results[i], embedding_results[i]))
    
    for idx, (i, kw, emb) in enumerate(regressions[:10], 1):  # Show first 10
        test = TEST_CASES[i]
        print(f"\n   Test {i+1}: \"{test['query']}\"")
        print(f"      Expected: {test['expected_intent']}")
        print(f"      Keyword got: {kw['actual']} ✅")
        print(f"      Embedding got: {emb['actual']} ❌")
    
    if len(regressions) > 10:
        print(f"\n   ... and {len(regressions) - 10} more regressions")

# Show queries both got wrong
if both_wrong > 0:
    print("\n" + "=" * 120)
    print(f"❌ BOTH WRONG: {both_wrong} queries that BOTH classifiers missed")
    print("=" * 120)
    
    both_failed = []
    for i in range(total):
        if not keyword_results[i]['passed'] and not embedding_results[i]['passed']:
            both_failed.append((i, keyword_results[i], embedding_results[i]))
    
    for idx, (i, kw, emb) in enumerate(both_failed[:5], 1):  # Show first 5
        test = TEST_CASES[i]
        print(f"\n   Test {i+1}: \"{test['query']}\"")
        print(f"      Expected: {test['expected_intent']}")
        print(f"      Keyword got: {kw['actual']} ❌")
        print(f"      Embedding got: {emb['actual']} ❌")
    
    if len(both_failed) > 5:
        print(f"\n   ... and {len(both_failed) - 5} more")

# Recommendation
print("\n" + "=" * 120)
print("🎯 RECOMMENDATION")
print("=" * 120)

if diff > 3:
    print(f"\n📈 EMBEDDING IS BETTER: +{diff:.1f}% accuracy improvement")
    print("   - Use embedding classifier for production")
    print("   - Trade-off: Slower (~0.2s per query) and costs ~$0.0001 per query")
elif diff < -3:
    print(f"\n📉 KEYWORD IS BETTER: -{diff:.1f}% accuracy drop with embedding")
    print("   - Stick with keyword classifier")
    print("   - Faster and zero cost per query")
else:
    print(f"\n➖ NO SIGNIFICANT DIFFERENCE: {diff:+.1f}% difference")
    print("   - Use keyword classifier for speed and cost efficiency")
    print("   - Use embedding classifier when semantic understanding is critical")

print("=" * 120)

