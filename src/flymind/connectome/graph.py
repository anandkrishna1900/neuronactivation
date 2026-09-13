"""
Directed weighted graph representation for connectome subcircuits.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import networkx as nx


@dataclass
class NeuronMetadata:
    """Represents biological neuron metadata extracted from connectome sources."""
    body_id: int
    cell_type: Optional[str] = None
    instance: Optional[str] = None
    roi: Optional[str] = None
    neurotransmitter: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SynapticConnection:
    """Represents a directed synaptic connection between two neurons."""
    source_id: int
    target_id: int
    weight: float  # Synapse count or scaled conductance
    neurotransmitter: Optional[str] = None
    confidence: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class ConnectomeGraph:
    """Directed weighted graph representation of a biological connectome."""

    def __init__(self, name: str = "drosophila_circuit"):
        self.name = name
        self._graph = nx.DiGraph()
        self._neurons: Dict[int, NeuronMetadata] = {}

    def add_neuron(self, neuron: NeuronMetadata) -> None:
        """Add a neuron with its metadata to the graph."""
        self._neurons[neuron.body_id] = neuron
        self._graph.add_node(
            neuron.body_id,
            cell_type=neuron.cell_type,
            instance=neuron.instance,
            roi=neuron.roi,
            neurotransmitter=neuron.neurotransmitter,
            extra=neuron.extra,
        )

    def add_connection(self, conn: SynapticConnection) -> None:
        """Add a directed synaptic connection to the graph."""
        if conn.source_id not in self._graph:
            self.add_neuron(NeuronMetadata(body_id=conn.source_id))
        if conn.target_id not in self._graph:
            self.add_neuron(NeuronMetadata(body_id=conn.target_id))

        self._graph.add_edge(
            conn.source_id,
            conn.target_id,
            weight=conn.weight,
            neurotransmitter=conn.neurotransmitter,
            confidence=conn.confidence,
            extra=conn.extra,
        )

    @property
    def num_neurons(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def num_synapses(self) -> int:
        return self._graph.number_of_edges()

    @property
    def nx_graph(self) -> nx.DiGraph:
        return self._graph

    def get_neuron(self, body_id: int) -> Optional[NeuronMetadata]:
        return self._neurons.get(body_id)
