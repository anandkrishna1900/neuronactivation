"""
Phase 3/4: Graph analysis and topology visualization of the CX heading circuit.

Usage:
    python scripts/analyze_cx_circuit.py

Outputs plots to: results/cx_analysis/
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from flymind.connectome.loader import ConnectomeLoader
from flymind.utils.visualization import (
    ConnectomeAnalyzer,
    plot_type_distribution,
    plot_weight_distribution,
    plot_type_connectivity_heatmap,
    plot_circuit_topology,
)

GRAPH_PATH = ROOT / "data" / "processed" / "cx_heading_v1.json"
OUT_DIR = ROOT / "results" / "cx_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 60)
    print("FlyMind Phase 3/4: CX Circuit Analysis")
    print("=" * 60)

    print("\nLoading graph...")
    graph = ConnectomeLoader.load_from_json(GRAPH_PATH)
    analyzer = ConnectomeAnalyzer(graph)

    # ── Summary stats ─────────────────────────────────────────
    print(f"\n  Neurons  : {graph.num_neurons}")
    print(f"  Synapses : {graph.num_synapses}")

    print("\n  Neuron type counts:")
    for t, c in analyzer.type_counts().items():
        nt = graph.nx_graph.nodes[next(
            n for n in graph.nx_graph.nodes()
            if graph.nx_graph.nodes[n].get("cell_type") == t
        )].get("neurotransmitter", "?")
        print(f"    {t:22s}: {c:4d}  [{nt}]")

    print("\n  Degree statistics:")
    for k, v in analyzer.degree_stats().items():
        print(f"    {k:22s}: {v:.1f}")

    print("\n  Synapse weight statistics:")
    for k, v in analyzer.weight_stats().items():
        print(f"    {k:22s}: {v}")

    print("\n  Top 10 strongest connections:")
    for src, tgt, w in analyzer.top_connections(10):
        src_type = graph.nx_graph.nodes[src].get("cell_type", "?")
        tgt_type = graph.nx_graph.nodes[tgt].get("cell_type", "?")
        print(f"    {src_type:20s} -> {tgt_type:20s}  [{w} synapses]")

    # ── Generate plots ─────────────────────────────────────────
    print("\nGenerating plots...")
    plot_type_distribution(analyzer, OUT_DIR / "01_neuron_type_counts.png")
    plot_weight_distribution(analyzer, OUT_DIR / "02_synapse_weight_distribution.png")
    plot_type_connectivity_heatmap(analyzer, OUT_DIR / "03_type_connectivity_heatmap.png")
    plot_circuit_topology(analyzer, OUT_DIR / "04_circuit_topology.png")

    print(f"\n[DONE] All plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
