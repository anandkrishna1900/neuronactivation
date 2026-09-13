"""
Graph topology analysis and visualization for the CX heading circuit.
"""

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional


# Cell-type color palette for visualization
CELL_TYPE_COLORS: Dict[str, str] = {
    "EPG":          "#4C9BE8",   # blue  — compass neurons
    "PEN_a(PEN1)":  "#E87B4C",   # orange — angular velocity left
    "PEN_b(PEN2)":  "#E8C04C",   # yellow — angular velocity right
    "Delta7":       "#D94F4F",   # red   — inhibitory ring
    "PEG":          "#7BE87B",   # green — premotor output
    "PFNd":         "#9B59B6",   # purple — fan-shaped body dorsal
    "PFNv":         "#C39BD3",   # light purple — fan-shaped body ventral
    "ER4d":         "#1ABC9C",   # teal  — visual landmark input
    "ER4m":         "#48C9B0",   # light teal — visual motion input
    "EL":           "#95A5A6",   # gray  — local interneurons
}


class ConnectomeAnalyzer:
    """Topological analysis of a ConnectomeGraph."""

    def __init__(self, graph):
        self.graph = graph
        self.nx_g = graph.nx_graph

    def type_counts(self) -> Dict[str, int]:
        counts = Counter(
            self.nx_g.nodes[n].get("cell_type", "unknown")
            for n in self.nx_g.nodes()
        )
        return dict(sorted(counts.items(), key=lambda x: -x[1]))

    def connection_density_by_type(self) -> Dict[str, Dict[str, int]]:
        """Total synapse weight between each pair of cell types."""
        matrix: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for u, v, data in self.nx_g.edges(data=True):
            src_type = self.nx_g.nodes[u].get("cell_type", "?")
            tgt_type = self.nx_g.nodes[v].get("cell_type", "?")
            matrix[src_type][tgt_type] += int(data.get("weight", 0))
        return {k: dict(v) for k, v in matrix.items()}

    def top_connections(self, n: int = 10):
        """Return the n strongest individual connections by synapse count."""
        edges = [
            (u, v, data.get("weight", 0))
            for u, v, data in self.nx_g.edges(data=True)
        ]
        return sorted(edges, key=lambda x: -x[2])[:n]

    def degree_stats(self):
        in_deg = [d for _, d in self.nx_g.in_degree()]
        out_deg = [d for _, d in self.nx_g.out_degree()]
        return {
            "mean_in_degree": float(np.mean(in_deg)),
            "max_in_degree": int(np.max(in_deg)),
            "mean_out_degree": float(np.mean(out_deg)),
            "max_out_degree": int(np.max(out_deg)),
        }

    def weight_stats(self):
        weights = [d.get("weight", 0) for _, _, d in self.nx_g.edges(data=True)]
        return {
            "total_synapses": int(sum(weights)),
            "mean_weight": float(np.mean(weights)),
            "median_weight": float(np.median(weights)),
            "max_weight": int(max(weights)),
            "min_weight": int(min(weights)),
        }


def plot_type_distribution(analyzer: ConnectomeAnalyzer, save_path: Path):
    counts = analyzer.type_counts()
    types = list(counts.keys())
    values = list(counts.values())
    colors = [CELL_TYPE_COLORS.get(t, "#AAAAAA") for t in types]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(types, values, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_title("CX Heading Circuit — Neuron Count by Cell Type\n(Hemibrain v1.2.1, Scheffer et al. 2020)",
                 fontsize=13, pad=12)
    ax.set_ylabel("Neuron Count")
    ax.set_xlabel("Cell Type (Hemibrain nomenclature)")
    ax.tick_params(axis="x", rotation=30)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(val), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_weight_distribution(analyzer: ConnectomeAnalyzer, save_path: Path):
    weights = [d.get("weight", 0) for _, _, d in analyzer.nx_g.edges(data=True)]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(weights, bins=60, color="#4C9BE8", edgecolor="white", linewidth=0.4)
    ax.set_yscale("log")
    ax.set_title("Synapse Count Distribution\n(CX Heading Circuit, 19,969 connections)",
                 fontsize=12)
    ax.set_xlabel("Synapse Count per Connection")
    ax.set_ylabel("Number of Connections (log scale)")
    ax.axvline(np.median(weights), color="#D94F4F", linewidth=1.5,
               linestyle="--", label=f"Median = {np.median(weights):.0f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_type_connectivity_heatmap(analyzer: ConnectomeAnalyzer, save_path: Path):
    matrix = analyzer.connection_density_by_type()
    types = sorted(CELL_TYPE_COLORS.keys())
    n = len(types)
    M = np.zeros((n, n))
    for i, src in enumerate(types):
        for j, tgt in enumerate(types):
            M[i, j] = matrix.get(src, {}).get(tgt, 0)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(M, cmap="Blues", aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(types, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(types, fontsize=8)
    ax.set_xlabel("Target Cell Type")
    ax.set_ylabel("Source Cell Type")
    ax.set_title("Total Synapse Count by Cell Type Pair\n(CX Heading Circuit, Hemibrain v1.2.1)",
                 fontsize=12)
    plt.colorbar(im, ax=ax, label="Total Synapse Count")

    # Annotate non-zero cells
    for i in range(n):
        for j in range(n):
            val = int(M[i, j])
            if val > 0:
                ax.text(j, i, str(val), ha="center", va="center",
                        fontsize=6, color="black" if val < M.max() / 2 else "white")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_circuit_topology(analyzer: ConnectomeAnalyzer, save_path: Path, max_nodes: int = 100):
    """Visualize a subgraph of the strongest connections."""
    G = analyzer.nx_g

    # Keep edges above weight threshold to make plot readable
    weights = sorted([d["weight"] for _, _, d in G.edges(data=True)], reverse=True)
    threshold = weights[min(500, len(weights) - 1)]

    subgraph = nx.DiGraph()
    for u, v, data in G.edges(data=True):
        if data["weight"] >= threshold:
            subgraph.add_edge(u, v, weight=data["weight"])

    node_colors = [
        CELL_TYPE_COLORS.get(G.nodes[n].get("cell_type", "?"), "#AAAAAA")
        for n in subgraph.nodes()
    ]
    edge_weights = [subgraph[u][v]["weight"] for u, v in subgraph.edges()]
    max_w = max(edge_weights) if edge_weights else 1
    edge_alphas = [0.3 + 0.7 * (w / max_w) for w in edge_weights]

    fig, ax = plt.subplots(figsize=(14, 10))
    pos = nx.spring_layout(subgraph, seed=42, k=0.4)
    nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors,
                           node_size=60, ax=ax, alpha=0.9)
    for (u, v), alpha in zip(subgraph.edges(), edge_alphas):
        nx.draw_networkx_edges(subgraph, pos, edgelist=[(u, v)],
                               ax=ax, alpha=alpha, edge_color="#555555",
                               arrows=True, arrowsize=8, width=0.8,
                               connectionstyle="arc3,rad=0.1")

    patches = [mpatches.Patch(color=c, label=t) for t, c in CELL_TYPE_COLORS.items()]
    ax.legend(handles=patches, loc="upper left", fontsize=7, ncol=2,
              title="Cell Type", title_fontsize=8)
    ax.set_title(f"CX Heading Circuit Topology (strongest {len(subgraph.edges())} connections)\n"
                 f"Hemibrain v1.2.1 — Scheffer et al. 2020",
                 fontsize=12)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")
