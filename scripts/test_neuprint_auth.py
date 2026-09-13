import requests
import sys

sys.stdout.reconfigure(encoding='utf-8')

TOKEN = "ddbee0f25b59d207e1e5ede8108a823e8dfa5b9e3d3eb194239dba80cf44f3a4"
BASE = "https://neuprint.janelia.org"
DATASET = "hemibrain:v1.2.1"
HEADERS = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}

def cypher(query):
    resp = requests.post(
        f"{BASE}/api/custom/custom",
        headers=HEADERS,
        json={"cypher": query, "dataset": DATASET},
        timeout=60
    )
    resp.raise_for_status()
    return resp.json()

def show(r, limit=10):
    cols = r["columns"]
    for row in r["data"][:limit]:
        print(f"  {dict(zip(cols, row))}")

# ── Confirmed real type names from Hemibrain v1.2.1 ──
# EPG      = compass neurons (E-PG in literature)
# PEN_a    = angular velocity left (P-EN in literature)
# PEN_b    = angular velocity right
# Delta7   = global inhibitory ring
# PEG      = premotor output (P-EG)
# PFNv/d   = fan-shaped body navigation

print("=== 1. EPG compass neuron count and samples ===")
r = cypher("""
MATCH (n:`hemibrain_Neuron`)
WHERE n.type = 'EPG'
RETURN n.bodyId AS bodyId, n.type AS type, n.instance AS instance, n.status AS status
LIMIT 10
""")
print(f"  Found {len(r['data'])} EPG rows (limited to 10)")
show(r)

print("\n=== 2. PEN_a and PEN_b counts ===")
r = cypher("""
MATCH (n:`hemibrain_Neuron`)
WHERE n.type IN ['PEN_a(PEN1)', 'PEN_b(PEN2)']
RETURN n.type AS type, count(n) AS count
""")
show(r)

print("\n=== 3. Full CX circuit neuron counts ===")
r = cypher("""
MATCH (n:`hemibrain_Neuron`)
WHERE n.type IN ['EPG', 'PEN_a(PEN1)', 'PEN_b(PEN2)', 'Delta7', 'PEG', 'PFNv', 'PFNd', 'EL', 'ER4m', 'ER4d', 'ER2a', 'ER2m', 'ER2r']
RETURN n.type AS type, count(n) AS count
ORDER BY count DESC
""")
show(r, 20)

print("\n=== 4. EPG -> PEN connections (compass -> integrator) ===")
r = cypher("""
MATCH (a:`hemibrain_Neuron`)-[c:ConnectsTo]->(b:`hemibrain_Neuron`)
WHERE a.type = 'EPG' AND b.type IN ['PEN_a(PEN1)', 'PEN_b(PEN2)']
RETURN a.bodyId AS src, b.bodyId AS tgt, b.type AS tgt_type, c.weight AS synapses
ORDER BY synapses DESC LIMIT 8
""")
show(r)

print("\n=== 5. Delta7 -> EPG connections (inhibitory) ===")
r = cypher("""
MATCH (a:`hemibrain_Neuron`)-[c:ConnectsTo]->(b:`hemibrain_Neuron`)
WHERE a.type = 'Delta7' AND b.type = 'EPG'
RETURN a.bodyId AS src, b.bodyId AS tgt, c.weight AS synapses
ORDER BY synapses DESC LIMIT 5
""")
show(r)

print("\n[DONE] CX circuit topology confirmed accessible.")
