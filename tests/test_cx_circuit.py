"""
Tests for the real extracted CX heading circuit graph (cx_heading_v1.json).
"""

import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GRAPH_PATH = ROOT / "data" / "processed" / "cx_heading_v1.json"

# Skip entire module if the extracted graph file doesn't exist yet
pytestmark = pytest.mark.skipif(
    not GRAPH_PATH.exists(),
    reason="cx_heading_v1.json not found; run scripts/extract_cx_circuit.py first"
)


def test_cx_graph_loads():
    from flymind.connectome.loader import ConnectomeLoader
    g = ConnectomeLoader.load_from_json(GRAPH_PATH)
    assert g.num_neurons == 261
    assert g.num_synapses > 15000


def test_cx_graph_validation():
    from flymind.connectome.loader import ConnectomeLoader
    from flymind.connectome.validation import validate_connectome_graph
    g = ConnectomeLoader.load_from_json(GRAPH_PATH)
    report = validate_connectome_graph(g)
    assert report["valid"] is True
    assert report["invalid_edges_count"] == 0


def test_cx_neuron_types_present():
    from flymind.connectome.loader import ConnectomeLoader
    g = ConnectomeLoader.load_from_json(GRAPH_PATH)
    types = {
        g.nx_graph.nodes[n].get("cell_type")
        for n in g.nx_graph.nodes()
    }
    required = {"EPG", "PEN_a(PEN1)", "PEN_b(PEN2)", "Delta7", "PEG"}
    assert required.issubset(types), f"Missing types: {required - types}"


def test_cx_epg_neurotransmitter():
    """EPG neurons must be annotated as cholinergic (excitatory)."""
    from flymind.connectome.loader import ConnectomeLoader
    g = ConnectomeLoader.load_from_json(GRAPH_PATH)
    for nid in g.nx_graph.nodes():
        node = g.nx_graph.nodes[nid]
        if node.get("cell_type") == "EPG":
            assert node.get("neurotransmitter") == "cholinergic"


def test_cx_delta7_neurotransmitter():
    """Delta7 neurons must be annotated as gabaergic (inhibitory)."""
    from flymind.connectome.loader import ConnectomeLoader
    g = ConnectomeLoader.load_from_json(GRAPH_PATH)
    for nid in g.nx_graph.nodes():
        node = g.nx_graph.nodes[nid]
        if node.get("cell_type") == "Delta7":
            assert node.get("neurotransmitter") == "gabaergic"


def test_cx_epg_pen_connections_exist():
    """There must be synaptic connections from EPG to PEN neurons."""
    from flymind.connectome.loader import ConnectomeLoader
    g = ConnectomeLoader.load_from_json(GRAPH_PATH)
    nx_g = g.nx_graph

    epg_to_pen = [
        (u, v) for u, v in nx_g.edges()
        if nx_g.nodes[u].get("cell_type") == "EPG"
        and "PEN" in (nx_g.nodes[v].get("cell_type") or "")
    ]
    assert len(epg_to_pen) > 0, "No EPG->PEN connections found"


def test_cx_no_nan_weights():
    """All synaptic weights must be finite positive numbers."""
    import math
    from flymind.connectome.loader import ConnectomeLoader
    g = ConnectomeLoader.load_from_json(GRAPH_PATH)
    for u, v, data in g.nx_graph.edges(data=True):
        w = data.get("weight", 0)
        assert not math.isnan(w), f"NaN weight on edge {u}->{v}"
        assert not math.isinf(w), f"Inf weight on edge {u}->{v}"
        assert w > 0, f"Non-positive weight {w} on edge {u}->{v}"
