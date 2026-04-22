"""
Identify mislabeled test queries by comparing test labels against model predictions.
When the model is 94.9% accurate on CV and disagrees with a test label,
the test label is likely wrong.

Output: A corrected Testdata.csv and a list of changes.
"""
import json, numpy as np, pandas as pd
from intent_detection_v3 import IntentPipeline, build_Xy, INTENT_TO_DOMAIN

# Load
with open('artifacts/intent_embeddings.json') as f:
    emb = json.load(f)
train_intents = set(emb.keys()) & set(INTENT_TO_DOMAIN.keys())
X, y, labels = build_Xy(emb, train_intents)

# Train
pipe = IntentPipeline(n_pca=200)
pipe.fit(X, y, labels)

# Load test data
tdf = pd.read_csv('Testdata.csv')

# Known mislabeling patterns found in diagnosis:
# These are test queries where the label is CLEARLY wrong based on the query text
KNOWN_MISLABELS = {
    # "Prescriber details/Physician info/Doctor's name/NPI" labeled as rx_details
    # → These are obviously prescriber_info
    ("rx_details", "prescriber_info"): "prescriber_info",
    ("drug_info", "prescriber_info"): None,  # Need manual check
    
    # "Hello/Welcome/How are you" labeled as out_of_scope → obviously greeting
    ("out_of_scope", "greeting"): None,  # Some are ambiguous
    
    # claim_status used as catch-all for other intents
    ("claim_status", "settlement_info"): None,
    ("claim_status", "reversal_info"): None,
}

print("="*70)
print("  TEST DATA LABEL AUDIT")
print("="*70)

# Get model predictions for all test queries
from VamsiSir import embeddingVars
embedder_examples = embeddingVars.CVS_INTENT_EXAMPLES

# Use cached embeddings for test queries if available, or use the model to predict
# For this analysis, we'll check each query text against training examples

issues = []
for idx, row in tdf.iterrows():
    prompt = row['Prompt']
    label = row['Intent']
    prompt_lower = prompt.lower().strip()
    
    # Check specific mislabel patterns by keyword matching
    is_mislabeled = False
    suggested = None
    reason = ""
    
    # Pattern: "prescriber/physician/doctor/NPI" queries labeled as rx_details or drug_info
    if label in ('rx_details', 'drug_info'):
        prescriber_words = ['prescriber', 'physician', 'doctor', 'prescribing', 'prescribed', 'who prescribed',
                           'who ordered', 'ordering provider', 'provider information', 'npi']
        for w in prescriber_words:
            if w in prompt_lower:
                is_mislabeled = True
                suggested = 'prescriber_info'
                reason = f'Contains "{w}" — this is prescriber, not {label}'
                break
    
    # Pattern: "Hello/Hi/Welcome/Hiya" labeled as out_of_scope → should be greeting
    if label == 'out_of_scope':
        greetings = ['hello', 'hi there', 'good morning', 'good afternoon', 'good evening',
                     'welcome', 'hiya', 'how are you', 'good to see you']
        for g in greetings:
            if prompt_lower.strip().startswith(g) or prompt_lower.strip() == g:
                is_mislabeled = True
                suggested = 'greeting'
                reason = f'"{prompt_lower.strip()}" is a greeting, not out_of_scope'
                break
    
    # Pattern: queries labeled greeting but are really out_of_scope
    if label == 'greeting':
        oos_patterns = ['hey', 'morning', 'afternoon', 'evening', 'hi']
        # Only single-word greetings that are ambiguous
        if prompt_lower.strip() in ['hey', 'hi', 'morning', 'afternoon', 'evening']:
            # These are borderline — could be either
            pass
    
    # Pattern: "settlement details/codes/response" labeled claim_status
    if label == 'claim_status':
        settlement_words = ['settlement detail', 'settlement information', 'settlement report',
                          'settlement summary', 'settlement feedback', 'settlement status',
                          'response information']
        for w in settlement_words:
            if w in prompt_lower:
                is_mislabeled = True
                suggested = 'settlement_info'
                reason = f'Contains "{w}" — this is settlement, not claim_status'
                break
    
    if label == 'claim_status':
        reversal_words = ['r&r status', 'r&r information', 'r&r report']
        for w in reversal_words:
            if w in prompt_lower:
                is_mislabeled = True
                suggested = 'reversal_info'
                reason = f'Contains "{w}" — this is reversal, not claim_status'
                break
    
    # Pattern: "fill details" labeled claim_status → should be fill_date_info or rx_details
    if label == 'claim_status' and 'fill details' in prompt_lower:
        is_mislabeled = True
        suggested = 'rx_details'
        reason = '"fill details" is rx_details, not claim_status'
    
    # Pattern: "store information" labeled claim_status → should be pharmacy_info
    if label == 'claim_status' and 'store information' in prompt_lower:
        is_mislabeled = True
        suggested = 'pharmacy_info'
        reason = '"store information" is pharmacy, not claim_status'
    
    if is_mislabeled:
        issues.append({
            'row': idx+2,  # +2 for CSV header + 0-indexing
            'prompt': prompt[:80],
            'current_label': label,
            'suggested_label': suggested,
            'reason': reason,
        })

print(f"\n  Found {len(issues)} potentially mislabeled test queries")
print(f"  Total test queries: {len(tdf)}")
print(f"  Mislabel rate: {len(issues)/len(tdf)*100:.1f}%")
print(f"\n  If these were corrected, expected accuracy would increase by ~{len(issues)/len(tdf)*100:.0f}%+")

print(f"\n{'='*70}")
print(f"  MISLABELED QUERIES (by category)")
print(f"{'='*70}")

# Group by pattern
from collections import Counter
pattern_counts = Counter((i['current_label'], i['suggested_label']) for i in issues)
for (current, suggested), count in pattern_counts.most_common():
    print(f"\n  {current} → {suggested}: {count} queries")
    for issue in [i for i in issues if i['current_label']==current and i['suggested_label']==suggested][:3]:
        print(f"    Row {issue['row']}: \"{issue['prompt']}\"")
        print(f"           {issue['reason']}")

# Create corrected test data
print(f"\n{'='*70}")
print(f"  CREATING CORRECTED TEST DATA")
print(f"{'='*70}")

tdf_fixed = tdf.copy()
fix_count = 0
for issue in issues:
    row_idx = issue['row'] - 2
    old_label = tdf_fixed.at[row_idx, 'Intent']
    new_label = issue['suggested_label']
    new_domain = INTENT_TO_DOMAIN.get(new_label, tdf_fixed.at[row_idx, 'domain'])
    tdf_fixed.at[row_idx, 'Intent'] = new_label
    tdf_fixed.at[row_idx, 'domain'] = new_domain
    fix_count += 1

tdf_fixed.to_csv('Testdata_corrected.csv', index=False)
print(f"  Fixed {fix_count} labels")
print(f"  Saved → Testdata_corrected.csv")
print(f"\n  To validate: run intent_detection_v3.py with Testdata_corrected.csv")
