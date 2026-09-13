"""Inspect connectome motor populations for Phase 6B."""
import json
from collections import Counter

with open("data/processed/cx_heading_v1.json") as f:
    data = json.load(f)

type_counts = Counter()
hemisphere_counts = Counter()
motor_neurons = []

for n in data["neurons"]:
    ct = n.get("cell_type", "unknown")
    inst = n.get("instance", "")
    bid = n["body_id"]
    type_counts[ct] += 1
    if "_L" in inst:
        hemisphere_counts[ct + "_L"] += 1
    elif "_R" in inst:
        hemisphere_counts[ct + "_R"] += 1
    if ct in ("PEG", "PEN_a(PEN1)", "PEN_b(PEN2)"):
        motor_neurons.append({"id": bid, "cell_type": ct, "instance": inst})

print("=== Neuron Types ===")
for ct, count in sorted(type_counts.items()):
    print(f"  {ct}: {count}")

print()
print("=== Hemispheres ===")
for k, v in sorted(hemisphere_counts.items()):
    print(f"  {k}: {v}")

print()
print("=== Motor Populations (PEG, PEN_a, PEN_b) ===")
for m in motor_neurons:
    print(f"  id={m['id']}, type={m['cell_type']}, instance={m['instance']}")
print(f"  Total motor neurons: {len(motor_neurons)}")

peg_ids = {m["id"] for m in motor_neurons if m["cell_type"] == "PEG"}
pen_a_ids = {m["id"] for m in motor_neurons if m["cell_type"] == "PEN_a(PEN1)"}
pen_b_ids = {m["id"] for m in motor_neurons if m["cell_type"] == "PEN_b(PEN2)"}
epg_ids = {n["body_id"] for n in data["neurons"] if n.get("cell_type") == "EPG"}
delta7_ids = {n["body_id"] for n in data["neurons"] if n.get("cell_type") == "Delta7"}

id_to_type = {n["body_id"]: n.get("cell_type", "unknown") for n in data["neurons"]}

print()
print("=== Incoming synapses to PEG ===")
incoming_counts = Counter()
for syn in data["connections"]:
    src = syn["source_id"]
    tgt = syn["target_id"]
    w = syn.get("weight", 1)
    if tgt in peg_ids:
        incoming_counts[id_to_type.get(src, "unknown")] += w
print("  Incoming by type:")
for t, c in sorted(incoming_counts.items(), key=lambda x: -x[1]):
    print(f"    {t}: {c} synapses")

print()
print("=== Outgoing synapses from PEG ===")
outgoing_counts = Counter()
for syn in data["connections"]:
    src = syn["source_id"]
    tgt = syn["target_id"]
    w = syn.get("weight", 1)
    if src in peg_ids:
        outgoing_counts[id_to_type.get(tgt, "unknown")] += w
print("  Outgoing by type:")
for t, c in sorted(outgoing_counts.items(), key=lambda x: -x[1]):
    print(f"    {t}: {c} synapses")

print()
print("=== Key pathways ===")
for name, src_set, tgt_set in [
    ("PEN_a -> PEG", pen_a_ids, peg_ids),
    ("PEN_b -> PEG", pen_b_ids, peg_ids),
    ("EPG -> PEG", epg_ids, peg_ids),
    ("EPG -> PEN_a", epg_ids, pen_a_ids),
    ("EPG -> PEN_b", epg_ids, pen_b_ids),
    ("Delta7 -> EPG", delta7_ids, epg_ids),
]:
    count = sum(syn.get("weight", 1) for syn in data["connections"]
                if syn["source_id"] in src_set and syn["target_id"] in tgt_set)
    print(f"  {name}: {count} synapses")

print()
print("=== PEG hemispheric L/R ===")
peg_L = [m for m in motor_neurons if m["cell_type"] == "PEG" and "_L" in m["instance"]]
peg_R = [m for m in motor_neurons if m["cell_type"] == "PEG" and "_R" in m["instance"]]
print(f"  PEG_L: {len(peg_L)} neurons")
print(f"  PEG_R: {len(peg_R)} neurons")
for m in peg_L:
    print(f"    {m['instance']}")
for m in peg_R:
    print(f"    {m['instance']}")

print()
print("=== PFN neuron counts ===")
pfnd = [n for n in data["neurons"] if n.get("cell_type") == "PFNd"]
pfnv = [n for n in data["neurons"] if n.get("cell_type") == "PFNv"]
print(f"  PFNd: {len(pfnd)}")
print(f"  PFNv: {len(pfnv)}")
pfnd_L = sum(1 for n in pfnd if "_L" in n["instance"])
pfnd_R = sum(1 for n in pfnd if "_R" in n["instance"])
pfnv_L = sum(1 for n in pfnv if "_L" in n["instance"])
pfnv_R = sum(1 for n in pfnv if "_R" in n["instance"])
print(f"  PFNd_L/R: {pfnd_L}/{pfnd_R}")
print(f"  PFNv_L/R: {pfnv_L}/{pfnv_R}")
