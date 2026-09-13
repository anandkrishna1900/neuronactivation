"""
Phase 2: Extract the Central Complex heading direction circuit from Hemibrain v1.2.1
and save as a validated sparse graph to data/processed/cx_heading_v1.json.

Run once to populate the local cache. All downstream experiments use the cached file.

Usage:
    python scripts/extract_cx_circuit.py

Requires:
    NEUPRINT_APPLICATION_CREDENTIALS environment variable  (or .env file)
"""

import sys
import os
from pathlib import Path

# Allow running from project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Load .env if present
env_file = ROOT / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

from flymind.connectome.query import NeuPrintQueryClient, CX_HEADING_CIRCUIT_TYPES
from flymind.connectome.loader import ConnectomeLoader

OUTPUT_PATH = ROOT / "data" / "processed" / "cx_heading_v1.json"

def main():
    print("=" * 60)
    print("FlyMind Phase 2: CX Heading Circuit Extraction")
    print("Dataset: hemibrain:v1.2.1 (Scheffer et al. 2020)")
    print("=" * 60)

    token = os.environ.get("NEUPRINT_APPLICATION_CREDENTIALS", "")
    if not token:
        print("ERROR: NEUPRINT_APPLICATION_CREDENTIALS not set.")
        sys.exit(1)

    client = NeuPrintQueryClient(token=token)

    print("\n[0/4] Pinging neuprint.janelia.org...", flush=True)
    if not client.ping():
        print("ERROR: Cannot reach neuprint.janelia.org")
        sys.exit(1)
    print("      -> Connected OK", flush=True)

    print(f"\nTarget cell types: {CX_HEADING_CIRCUIT_TYPES}\n")

    graph = client.build_cx_heading_graph(min_synapse_weight=1)

    print(f"\n[4/4] Saving graph to {OUTPUT_PATH}...")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ConnectomeLoader.save_to_json(graph, OUTPUT_PATH)
    size_kb = OUTPUT_PATH.stat().st_size / 1024
    print(f"      -> Saved ({size_kb:.1f} KB)")

    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print(f"  Neurons  : {graph.num_neurons}")
    print(f"  Synapses : {graph.num_synapses}")
    print(f"  File     : {OUTPUT_PATH}")
    print("=" * 60)

    # Print type breakdown
    from collections import Counter
    type_counts = Counter(
        graph.nx_graph.nodes[n].get("cell_type", "unknown")
        for n in graph.nx_graph.nodes()
    )
    print("\nNeuron type breakdown:")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t:20s}: {c:4d}")


if __name__ == "__main__":
    main()
