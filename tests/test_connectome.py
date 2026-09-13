"""
Tests for connectome graph construction and validation invariants.
"""

import pytest
from flymind.connectome.graph import ConnectomeGraph, NeuronMetadata, SynapticConnection
from flymind.connectome.validation import validate_connectome_graph


def test_connectome_graph_construction():
    g = ConnectomeGraph(name="test_circuit")
    
    # Add neurons
    n1 = NeuronMetadata(body_id=101, cell_type="E-PG", neurotransmitter="cholinergic")
    n2 = NeuronMetadata(body_id=102, cell_type="P-EN", neurotransmitter="cholinergic")
    n3 = NeuronMetadata(body_id=103, cell_type="Delta7", neurotransmitter="gabaergic")
    
    g.add_neuron(n1)
    g.add_neuron(n2)
    g.add_neuron(n3)

    # Add synapses
    g.add_connection(SynapticConnection(source_id=101, target_id=102, weight=12.0))
    g.add_connection(SynapticConnection(source_id=102, target_id=101, weight=5.0))
    g.add_connection(SynapticConnection(source_id=103, target_id=101, weight=8.0))

    assert g.num_neurons == 3
    assert g.num_synapses == 3

    report = validate_connectome_graph(g)
    assert report["valid"] is True
    assert report["num_neurons"] == 3
    assert report["num_synaptic_connections"] == 3
    assert report["invalid_edges_count"] == 0


def test_connectome_validation_catches_invalid_weights():
    g = ConnectomeGraph(name="invalid_circuit")
    g.add_connection(SynapticConnection(source_id=1, target_id=2, weight=-5.0))

    with pytest.raises(ValueError, match="Connectome validation failed"):
        validate_connectome_graph(g)
