"""
Validation and sanity checks for connectome graphs.
"""

from typing import Dict, List, Tuple
import math
from flymind.connectome.graph import ConnectomeGraph


def validate_connectome_graph(graph: ConnectomeGraph) -> Dict[str, any]:
    """
    Validates structural invariants of a ConnectomeGraph.
    
    Checks:
    - No NaN or Inf weights.
    - No negative weights (weights represent raw synapse counts).
    - Checks for self-connections.
    - Node and edge counts.
    """
    nx_g = graph.nx_graph
    invalid_edges: List[Tuple[int, int, float]] = []
    self_connections: List[Tuple[int, float]] = []

    for u, v, data in nx_g.edges(data=True):
        weight = data.get("weight", 0.0)
        if math.isnan(weight) or math.isinf(weight) or weight < 0:
            invalid_edges.append((u, v, weight))
        if u == v:
            self_connections.append((u, weight))

    report = {
        "valid": len(invalid_edges) == 0,
        "num_neurons": nx_g.number_of_nodes(),
        "num_synaptic_connections": nx_g.number_of_edges(),
        "invalid_edges_count": len(invalid_edges),
        "invalid_edges": invalid_edges,
        "self_connections_count": len(self_connections),
        "self_connections": self_connections,
    }

    if not report["valid"]:
        raise ValueError(f"Connectome validation failed: {report}")

    return report
