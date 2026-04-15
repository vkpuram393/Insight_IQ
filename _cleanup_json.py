import json

with open('api-response.json', 'r') as f:
    data = json.load(f)

# Extract member from first claim that has it
member = None
for claim in data.get('claims', []):
    if 'member' in claim and claim['member'] is not None:
        member = claim['member']
        break

# Remove unwanted fields from each claim
fields_to_remove = ['messages', 'overrides', 'priorAuthorization', 'pricing', 'member']
for claim in data.get('claims', []):
    for field in fields_to_remove:
        claim.pop(field, None)

# Build new data with member at top level after totalCount
new_data = {}
for key in data:
    new_data[key] = data[key]
    if key == 'totalCount':
        new_data['member'] = member

# Ensure claims uses the cleaned version
new_data['claims'] = data['claims']

with open('api-response.json', 'w') as f:
    json.dump(new_data, f, indent=4)

num_claims = len(data['claims'])
print("Done. Member extracted to top level. Removed messages, overrides, priorAuthorization, pricing, member from " + str(num_claims) + " claims.")
