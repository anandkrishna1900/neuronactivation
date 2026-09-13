import json
from pathlib import Path

data_path = Path("data/processed/cx_heading_v1.json")
with open(data_path, "r", encoding="utf-8") as f:
    data = json.load(f)

neurons = data.get("neurons", [])
print(f"Total neurons: {len(neurons)}")
types = {}
for n in neurons:
    ct = n.get("cell_type", "unknown")
    types.setdefault(ct, []).append(n)

for ct, ns in sorted(types.items()):
    sample_instances = [x.get("instance") for x in ns[:4]]
    print(f"{ct:15s} count={len(ns):3d} | sample instances: {sample_instances}")
