import pandas as pd

df = pd.read_csv('outputs/results_ensemble_only.csv')
print(f'Total: {len(df)}, Correct: {int(df.intent_match.sum())}, Wrong: {len(df)-int(df.intent_match.sum())}')
print(f'Intent Acc: {df.intent_match.mean()*100:.1f}%')
print()

wrong = df[~df['intent_match']]
print('=== TOP 20 CONFUSIONS ===')
conf = wrong.groupby(['actual_intent','predicted_intent']).size().sort_values(ascending=False)
for (a,p), c in conf.head(20).items():
    ex = wrong[(wrong.actual_intent==a)&(wrong.predicted_intent==p)].iloc[0]['text']
    ex_short = ex[:90] + '...' if len(ex) > 90 else ex
    print(f'  {a} -> {p}: {c}x')
    print(f'    e.g. "{ex_short}"')

print()
print('=== PER-INTENT ACCURACY ===')
for intent in sorted(df.actual_intent.unique()):
    s = df[df.actual_intent==intent]
    acc = s.intent_match.mean()*100
    n = len(s)
    marker = '  *** LOW' if acc < 80 else ''
    print(f'  {intent:<28} {acc:>5.1f}% ({int(s.intent_match.sum())}/{n}){marker}')

# Check confidence distribution for wrong predictions
print()
print('=== CONFIDENCE ON WRONG PREDICTIONS ===')
print(f'  Wrong predictions avg confidence: {wrong.confidence.mean():.3f}')
print(f'  Wrong predictions avg margin:     {wrong.margin.mean():.3f}')
correct = df[df.intent_match]
print(f'  Correct predictions avg confidence: {correct.confidence.mean():.3f}')
print(f'  Correct predictions avg margin:     {correct.margin.mean():.3f}')
