"""Categorize every test error: mislabel vs distribution gap vs genuine ambiguity."""
import pandas as pd, json, numpy as np, os

# Load results
for f in ['outputs/results_ensemble_only.csv', 'outputs/results_hybrid.csv']:
    if os.path.exists(f):
        df = pd.read_csv(f)
        print(f"Loaded {f}: {len(df)} rows, {df.intent_match.mean()*100:.1f}% accuracy")
        break
else:
    print("No results file found. Run intent_detection_v3.py first.")
    exit()

wrong = df[~df['intent_match']].copy()
print(f"\nTotal errors: {len(wrong)}/{len(df)}")

# Categorize each error
categories = {"mislabel": [], "distribution_gap": [], "genuine_ambiguity": []}

for _, row in wrong.iterrows():
    text = row['text'].lower()
    actual = row['actual_intent']
    predicted = row['predicted_intent']
    
    # MISLABELS: test query text clearly matches predicted, not actual
    is_mislabel = False
    
    # Prescriber queries labeled as rx_details or drug_info
    if actual in ('rx_details', 'drug_info') and predicted == 'prescriber_info':
        if any(w in text for w in ['prescriber','physician','doctor','who prescribed','npi and name','ordering provider']):
            is_mislabel = True
    
    # Settlement queries labeled claim_status
    if actual == 'claim_status' and predicted == 'settlement_info':
        if any(w in text for w in ['settlement detail','settlement info','settlement report','settlement summary','settlement feedback','settlement status']):
            is_mislabel = True
    
    # Greetings labeled out_of_scope
    if actual == 'out_of_scope' and predicted == 'greeting':
        if text.strip() in ['hello','welcome','hiya','hello, how are you?','hi, good to see you']:
            is_mislabel = True
    if actual == 'greeting' and predicted == 'out_of_scope':
        if text.strip() in ['hey','hi','morning','afternoon','evening','hey there']:
            is_mislabel = True
    
    # R&R labeled claim_status
    if actual == 'claim_status' and predicted in ('reversal_info','rx_details','pharmacy_info'):
        if any(w in text for w in ['r&r','fill details','store information']):
            is_mislabel = True
    
    # GENUINE AMBIGUITY: both labels could be correct
    is_ambiguous = False
    ambiguous_pairs = [
        ('pricing_info', 'reimbursement_info'),  # "payment" is ambiguous
        ('pricing_info', 'compound_info'),  # "ingredient cost" is ambiguous
        ('audit_info', 'reversal_info'),  # "modification" is ambiguous
        ('approval_info', 'prior_auth_info'),  # both about authorization
        ('approval_info', 'claim_status'),  # "adjudication outcome" is ambiguous
        ('beneficiary_info', 'approval_info'),  # "plan configuration" is ambiguous
        ('rejection_reasons', 'approval_info'),  # "override codes" is ambiguous
    ]
    for a, b in ambiguous_pairs:
        if (actual == a and predicted == b) or (actual == b and predicted == a):
            is_ambiguous = True
    
    if is_mislabel:
        categories["mislabel"].append(row)
    elif is_ambiguous:
        categories["genuine_ambiguity"].append(row)
    else:
        categories["distribution_gap"].append(row)

print(f"\n{'='*70}")
print(f"  ERROR CATEGORIZATION")
print(f"{'='*70}")
print(f"  Mislabeled test data:     {len(categories['mislabel']):>3} ({len(categories['mislabel'])/len(df)*100:.1f}% of all queries)")
print(f"  Genuine ambiguity:        {len(categories['genuine_ambiguity']):>3} ({len(categories['genuine_ambiguity'])/len(df)*100:.1f}%)")
print(f"  Distribution gap:         {len(categories['distribution_gap']):>3} ({len(categories['distribution_gap'])/len(df)*100:.1f}%)")
print(f"  TOTAL errors:             {len(wrong):>3} ({len(wrong)/len(df)*100:.1f}%)")

# Show distribution gap errors - these need better training examples
print(f"\n{'='*70}")
print(f"  DISTRIBUTION GAP ERRORS (need better training examples)")
print(f"  These are queries the model gets wrong because training")
print(f"  examples don't cover this phrasing pattern.")
print(f"{'='*70}")
for row in categories['distribution_gap']:
    text = row['text'][:85]
    print(f"  {row['actual_intent']:<22} → {row['predicted_intent']:<22} \"{text}\"")

print(f"\n{'='*70}")
print(f"  MISLABELED (model is correct, test label is wrong)")
print(f"{'='*70}")
for row in categories['mislabel']:
    text = row['text'][:85]
    print(f"  Label:{row['actual_intent']:<18} Model:{row['predicted_intent']:<18} \"{text}\"")

print(f"\n{'='*70}")
print(f"  GENUINE AMBIGUITY (both labels could be right)")
print(f"{'='*70}")
for row in categories['genuine_ambiguity']:
    text = row['text'][:85]
    print(f"  {row['actual_intent']:<22} → {row['predicted_intent']:<22} \"{text}\"")

# Projected accuracy after fixes
fixable = len(categories['mislabel'])
remaining = len(categories['genuine_ambiguity']) + len(categories['distribution_gap'])
projected = (len(df) - remaining) / len(df) * 100

print(f"\n{'='*70}")
print(f"  ACCURACY PROJECTIONS")
print(f"{'='*70}")
print(f"  Current accuracy:                {df.intent_match.mean()*100:.1f}%")
print(f"  After fixing mislabels ({fixable}):     {(len(df)-remaining)/len(df)*100:.1f}%")
print(f"  Remaining errors (ambig+gap):    {remaining}")
print(f"  To reach 90%: need to fix        {max(0, int(len(wrong) - len(df)*0.10))} more errors")
print(f"    → Add better training examples for distribution gap patterns")
print(f"    → LLM fallback for genuine ambiguity")
