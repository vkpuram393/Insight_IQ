"""Diagnose train/test mismatch by looking at actual test examples."""
import pandas as pd, json, numpy as np
from VamsiSir import embeddingVars
from intent_detection_v3 import INTENT_TO_DOMAIN, build_Xy, IntentPipeline

# Load test data
tdf = pd.read_csv('Testdata.csv')

# Show examples for the TOP confusion pairs
CONFUSION_PAIRS = [
    ("rx_details", "prescriber_info"),
    ("claim_status", "settlement_info"),
    ("drug_info", "prescriber_info"),
    ("approval_info", "prior_auth_info"),
    ("audit_info", "fill_date_info"),
    ("compound_info", "pricing_info"),
    ("out_of_scope", "greeting"),
    ("claim_status", "reversal_info"),
    ("audit_info", "claim_status"),
]

print("="*70)
print("  TRAIN/TEST MISMATCH ANALYSIS")
print("  Showing test queries that the model gets wrong")
print("="*70)

# Load training examples for comparison
train_examples = embeddingVars.CVS_INTENT_EXAMPLES

for actual, predicted in CONFUSION_PAIRS:
    test_rows = tdf[tdf['Intent'] == actual]
    print(f"\n--- Labeled as '{actual}' (test has {len(test_rows)} queries) ---")
    print(f"    Model predicts → '{predicted}'")
    
    # Show training examples for ACTUAL intent
    if actual in train_examples:
        print(f"\n    Training examples for '{actual}' (first 3):")
        for ex in train_examples[actual][:3]:
            print(f"      TRAIN: \"{ex}\"")
    
    # Show training examples for PREDICTED intent
    if predicted in train_examples:
        print(f"\n    Training examples for '{predicted}' (first 3):")
        for ex in train_examples[predicted][:3]:
            print(f"      TRAIN: \"{ex}\"")
    
    # Show actual test queries that are labeled as 'actual' 
    # but look like they should be 'predicted'
    print(f"\n    Test queries labeled '{actual}' (first 5):")
    for _, row in test_rows.head(5).iterrows():
        print(f"      TEST:  \"{row['Prompt'][:100]}\"")
    
    print()

# Summary: How many test intents have training examples that look very different?
print("\n" + "="*70)
print("  LABEL MISMATCH CHECK")
print("  Test queries that semantically match a DIFFERENT intent")
print("="*70)

# Load embeddings and build model
with open('artifacts/intent_embeddings.json') as f:
    emb = json.load(f)
train_intents = set(emb.keys()) & set(INTENT_TO_DOMAIN.keys())
X, y, labels = build_Xy(emb, train_intents)

pipe = IntentPipeline(n_pca=200)
pipe.fit(X, y, labels)

# For test queries, check if the model's prediction SEMANTICALLY makes more sense
print(f"\n  Checking if 'wrong' predictions are actually correct labels...")
print(f"  (i.e., the test data has mislabeled examples)")
print()
