"""
Orchestrates connectome topology, neuron models, and synaptic interactions.
"""

from typing import Dict, List, Optional
import numpy as np
import networkx as nx

from flymind.brain.neuron import NeuronModel, RateNeuron
from flymind.brain.synapse import SynapseModel, DEFAULT_NEUROTRANSMITTER_SIGNS
from flymind.connectome.graph import ConnectomeGraph


class NeuralNetwork:
    """Combines a connectome graph with dynamic neuron and synapse models."""

    def __init__(
        self,
        graph: ConnectomeGraph,
        neuron_model_cls=RateNeuron,
        synapse_scale: float = 0.01,
        dt: float = 1.0,
    ):
        self.graph = graph
        self.dt = dt
        self.neuron_ids: List[int] = list(graph.nx_graph.nodes())
        self.id_to_idx: Dict[int, int] = {nid: i for i, nid in enumerate(self.neuron_ids)}
        self.num_neurons = len(self.neuron_ids)

        # Build raw weight matrix W[i, j] (i -> j)
        adj_matrix = np.zeros((self.num_neurons, self.num_neurons), dtype=np.float64)
        signs = np.ones(self.num_neurons, dtype=np.float64)

        for u, v, data in graph.nx_graph.edges(data=True):
            i = self.id_to_idx[u]
            j = self.id_to_idx[v]
            adj_matrix[i, j] = float(data.get("weight", 1.0))

        for nid, i in self.id_to_idx.items():
            meta = graph.get_neuron(nid)
            nt = (meta.neurotransmitter or "").lower() if meta else ""
            if nt in DEFAULT_NEUROTRANSMITTER_SIGNS:
                signs[i] = DEFAULT_NEUROTRANSMITTER_SIGNS[nt]

        self.synapses = SynapseModel(adj_matrix, signs=signs, scale=synapse_scale)
        self.neurons: NeuronModel = neuron_model_cls(self.num_neurons)

    def step(self, external_current: Optional[np.ndarray] = None) -> np.ndarray:
        """Single simulation timestep."""
        if external_current is None:
            external_current = np.zeros(self.num_neurons, dtype=np.float64)

        synaptic_input = self.synapses.compute_postsynaptic_current(
            getattr(self.neurons, "activity", getattr(self.neurons, "spikes", np.zeros(self.num_neurons)))
        )
        total_current = external_current + synaptic_input
        activity = self.neurons.step(total_current, self.dt)
        return activity

    def reset(self) -> None:
        self.neurons.reset()
