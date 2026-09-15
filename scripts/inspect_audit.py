import json
with open('results/phase7b/connectome_expansion_audit.json') as f:
    data = json.load(f)

types = {}
for n in data['candidate_neurons']:
    t = n['type']
    types[t] = types.get(t, 0) + 1

print('=== Candidate Cell Type Distribution ===')
for t, count in sorted(types.items(), key=lambda x: -x[1]):
    print(f'  {t}: {count}')

print(f'\nTotal candidates: {len(data["candidate_neurons"])}')
print(f'Total connections: {len(data["connections"])}')

# Count afferent vs efferent
aff = sum(1 for c in data['connections'] if c['direction'] == 'afferent')
eff = sum(1 for c in data['connections'] if c['direction'] == 'efferent')
print(f'Afferent connections (into circuit): {aff}')
print(f'Efferent connections (out of circuit): {eff}')

# Top connections by weight
print('\n=== Top 20 Strongest Connections ===')
top = sorted(data['connections'], key=lambda x: -x['weight'])[:20]
for c in top:
    print(f'  {c["source_id"]} ({c["source_type"]}) -> {c["target_id"]} ({c["target_type"]}): {c["weight"]} [{c["direction"]}]')
