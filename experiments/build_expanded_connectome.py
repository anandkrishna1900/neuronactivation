"""Build the expanded 427-neuron connectome JSON from the audit data."""
import json
from pathlib import Path
from collections import defaultdict

ROOT = Path("C:/Users/anand/OneDrive/Desktop/Projects/Fly stuff")
DATA_DIR = ROOT / "data" / "processed"
AUDIT = ROOT / "results" / "phase7b" / "connectome_expansion_audit.json"
OUTPUT = DATA_DIR / "cx_heading_v1_expanded.json"

# Load existing 261-neuron circuit
with open(DATA_DIR / "cx_heading_v1.json") as f:
    original = json.load(f)

existing_ids = set(n["body_id"] for n in original["neurons"])
print(f"Original circuit: {len(existing_ids)} neurons, {len(original['connections'])} connections")

# Load audit
with open(AUDIT) as f:
    audit = json.load(f)

# Identify new neurons
candidates = audit["candidate_neurons"]
new_neurons = [n for n in candidates if n["body_id"] not in existing_ids]
new_ids = set(n["body_id"] for n in new_neurons)
print(f"New neurons from audit: {len(new_neurons)}")

# Select best candidates — prioritize FB and PFL neurons for temporal dynamics
# Take all FC3, PFL1/2/3, and top FB neurons
selected_new = []
selected_ids = set()

# Priority 1: PFL neurons (PB → FB connectors, crucial for temporal dynamics)
for n in new_neurons:
    if n["type"] in ("PFL1", "PFL2", "PFL3"):
        selected_new.append(n)
        selected_ids.add(n["body_id"])

# Priority 2: FC3 (fan-shaped body layer 3, columnar organization)
for n in new_neurons:
    if n["type"] == "FC3" and n["body_id"] not in selected_ids:
        selected_new.append(n)
        selected_ids.add(n["body_id"])

# Priority 3: FB interneurons (layers 1-6)
fb_types = [t for t in set(n["type"] for n in new_neurons) if t.startswith("FB")]
for n in new_neurons:
    if n["type"] in fb_types and n["body_id"] not in selected_ids:
        selected_new.append(n)
        selected_ids.add(n["body_id"])

# Priority 4: MBONs (mushroom body output, limited)
for n in new_neurons:
    if n["type"].startswith("MBON") and n["body_id"] not in selected_ids:
        selected_new.append(n)
        selected_ids.add(n["body_id"])

print(f"Selected new neurons: {len(selected_new)}")
all_ids = existing_ids | selected_ids
print(f"Total neurons in expanded circuit: {len(all_ids)}")

# Build neuron metadata for new neurons
neuron_meta = list(original["neurons"])  # keep all original neurons

# Build neurotransmitter map
NT_MAP = {
    "FC3": "cholinergic",
    "PFL1": "cholinergic",
    "PFL2": "cholinergic",
    "PFL3": "cholinergic",
    "FB1A": "cholinergic", "FB1B": "cholinergic", "FB1C": "cholinergic",
    "FB2A": "cholinergic", "FB2C": "cholinergic", "FB2D": "cholinergic", "FB2E": "cholinergic",
    "FB3A": "cholinergic", "FB3B": "gabaergic", "FB3C": "cholinergic",
    "FB3D": "cholinergic", "FB3E": "cholinergic",
    "FB4A": "cholinergic", "FB4B": "cholinergic", "FB4C": "cholinergic",
    "FB4D": "cholinergic", "FB4E": "cholinergic",
    "FB5A": "cholinergic", "FB5E": "cholinergic",
    "FB6B": "cholinergic",
    "MBON02": "cholinergic", "MBON05": "cholinergic", "MBON06": "cholinergic",
}

for n in selected_new:
    body_id = n["body_id"]
    cell_type = n["type"]
    instance = n.get("instance", f"{cell_type}_{body_id}")
    nt = NT_MAP.get(cell_type, "cholinergic")
    neuron_meta.append({
        "body_id": body_id,
        "cell_type": cell_type,
        "instance": instance,
        "roi": "CX",
        "neurotransmitter": nt,
        "extra": {
            "status": "Traced",
            "pre": n.get("pre", 0),
            "post": n.get("post", 0),
        },
    })

# Build connections — keep all original + add new connections involving selected neurons
all_connections = list(original["connections"])
conn_set = set()
for c in all_connections:
    conn_set.add((c["source_id"], c["target_id"]))

# Add connections from audit that involve selected new neurons
audit_conns = audit.get("connections", [])
added_conns = 0
for c in audit_conns:
    src = c["source_id"]
    tgt = c["target_id"]
    # Only add if both neurons are in our expanded set
    if src in all_ids and tgt in all_ids:
        key = (src, tgt)
        if key not in conn_set:
            all_connections.append({
                "source_id": src,
                "target_id": tgt,
                "weight": c["weight"],
                "neurotransmitter": c.get("neurotransmitter", None),
                "confidence": c.get("confidence", None),
                "extra": {},
            })
            conn_set.add(key)
            added_conns += 1

print(f"Added {added_conns} new connections")
print(f"Total connections: {len(all_connections)}")

# Count synapses per new cell type
from collections import Counter
new_type_counts = Counter(n["type"] for n in selected_new)
print("\nExpanded circuit cell type breakdown:")
all_type_counts = Counter(n["cell_type"] for n in neuron_meta)
for ct, count in all_type_counts.most_common():
    marker = " (NEW)" if ct in new_type_counts else ""
    print(f"  {ct}: {count}{marker}")

# Save expanded circuit
expanded = {
    "name": "cx_heading_v1_expanded",
    "num_neurons": len(neuron_meta),
    "num_synapses": len(all_connections),
    "neurons": neuron_meta,
    "connections": all_connections,
    "expansion_info": {
        "original_neurons": 261,
        "added_neurons": len(selected_new),
        "added_connections": added_conns,
        "selected_types": dict(new_type_counts),
        "source": "Janelia Hemibrain v1.2.1 via NeuPrint API",
    },
}

with open(OUTPUT, "w") as f:
    json.dump(expanded, f, indent=None, separators=(",", ":"))

print(f"\nSaved expanded connectome: {OUTPUT}")
print(f"  Size: {OUTPUT.stat().st_size / 1024:.1f} KB")
print(f"  Neurons: {expanded['num_neurons']}")
print(f"  Synapses: {expanded['num_synapses']}")
