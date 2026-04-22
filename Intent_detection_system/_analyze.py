import json, pandas as pd

df = pd.read_csv('Testdata.csv')
print(f'Test records: {len(df)}')
print(f'Unique intents in test: {df["Intent"].nunique()}')
print(f'Unique domains in test: {df["domain"].nunique()}')

with open('artifacts/intent_embeddings.json') as f:
    cached = json.load(f)

print(f'\nCached intents: {len(cached)}')
print(f'Embedding dim: {len(cached[list(cached.keys())[0]][0])}')

test_intents = set(df['Intent'].unique())
cached_intents = set(cached.keys())
print(f'Test intents missing from cache: {sorted(test_intents - cached_intents)}')
print(f'All test intents covered: {test_intents.issubset(cached_intents)}')

print(f'\nTraining samples per intent:')
for intent in sorted(cached.keys()):
    marker = ' <-- TEST' if intent in test_intents else ''
    print(f'  {intent}: {len(cached[intent])} examples{marker}')
print(f'\nTotal training vectors: {sum(len(v) for v in cached.values())}')

# Show test distribution
print(f'\nTest distribution:')
for intent, count in df['Intent'].value_counts().items():
    print(f'  {intent}: {count}')
