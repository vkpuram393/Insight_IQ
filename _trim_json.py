import json

with open('api-response.json', 'r') as f:
    data = json.load(f)

original_count = len(data['claims'])
data['claims'] = data['claims'][:50]

with open('api-response.json', 'w') as f:
    json.dump(data, f, indent=4)

print("Trimmed claims from " + str(original_count) + " to " + str(len(data['claims'])))
