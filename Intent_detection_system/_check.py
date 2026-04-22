import pandas as pd, os

print('Testdata.csv exists:', os.path.exists('Testdata.csv'))
print('Testdata_corrected.csv exists:', os.path.exists('Testdata_corrected.csv'))

df = pd.read_csv('Testdata.csv')
mislabels = 0

# prescriber queries labeled rx_details
for _, r in df[df['Intent']=='rx_details'].iterrows():
    p = r['Prompt'].lower()
    if any(w in p for w in ['prescriber','physician','doctor','npi','who prescribed']):
        prompt = r['Prompt'][:70]
        print(f'  MISLABEL rx_details->prescriber: {prompt}')
        mislabels += 1

# settlement labeled claim_status
for _, r in df[df['Intent']=='claim_status'].iterrows():
    p = r['Prompt'].lower()
    if 'settlement' in p:
        prompt = r['Prompt'][:70]
        print(f'  MISLABEL claim_status->settlement: {prompt}')
        mislabels += 1

# greeting labeled out_of_scope
for _, r in df[df['Intent']=='out_of_scope'].iterrows():
    p = r['Prompt'].lower().strip()
    if p in ['hello','welcome','hiya','hello, how are you?','hi, good to see you']:
        prompt = r['Prompt'][:70]
        print(f'  MISLABEL out_of_scope->greeting: {prompt}')
        mislabels += 1

# prescriber labeled drug_info
for _, r in df[df['Intent']=='drug_info'].iterrows():
    p = r['Prompt'].lower()
    if any(w in p for w in ['prescriber','physician','doctor','who prescribed','who ordered']):
        prompt = r['Prompt'][:70]
        print(f'  MISLABEL drug_info->prescriber: {prompt}')
        mislabels += 1

print(f'\nTotal obvious mislabels found: {mislabels}')

# Also show confidence stats from the last run
if os.path.exists('outputs/results_hybrid.csv'):
    res = pd.read_csv('outputs/results_hybrid.csv')
    ens = res[res['source']=='ensemble']
    llm = res[res['source']=='llm']
    print(f'\nEnsemble: {len(ens)} queries, {ens.intent_match.mean()*100:.1f}% accuracy')
    print(f'LLM:      {len(llm)} queries, {llm.intent_match.mean()*100:.1f}% accuracy')
    print(f'\nEnsemble is BETTER than LLM by {ens.intent_match.mean()*100 - llm.intent_match.mean()*100:.1f}%')
    print(f'LLM is HURTING overall accuracy!')
    
    # Show confidence distribution
    print(f'\nConfidence distribution (ensemble predictions):')
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        above = res[res['confidence'] >= t]
        print(f'  conf >= {t}: {len(above)} queries ({len(above)/len(res)*100:.0f}%), acc={above.intent_match.mean()*100:.1f}%')
elif os.path.exists('outputs/results_ensemble_only.csv'):
    res = pd.read_csv('outputs/results_ensemble_only.csv')
    print(f'\nEnsemble only results found')
    print(f'Accuracy: {res.intent_match.mean()*100:.1f}%')
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        above = res[res['confidence'] >= t]
        print(f'  conf >= {t}: {len(above)} queries ({len(above)/len(res)*100:.0f}%), acc={above.intent_match.mean()*100:.1f}%')
