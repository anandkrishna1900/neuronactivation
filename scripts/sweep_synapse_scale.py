"""
Systematic sweep of synapse_scale and drive strength for CX heading circuit.
Aims to find the dynamic operating regime where EPG activity stays in [0.1, 0.7].
"""

import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from flymind.connectome.loader import ConnectomeLoader
from flymind.brain.network import NeuralNetwork
from flymind.brain.neuron import RateNeuron
from flymind.brain.simulator import DiscreteSimulator

GRAPH_PATH = ROOT / "data" / "processed" / "cx_heading_v1.json"
graph = ConnectomeLoader.load_from_json(GRAPH_PATH)

scales = [0.0001, 0.0005, 0.001, 0.002, 0.005]
drives = [0.5, 1.0, 2.0, 3.0]

epg_indices = [i for i, nid in enumerate(graph.nx_graph.nodes())
               if graph.nx_graph.nodes[nid].get("cell_type") == "EPG"]
er4d_indices = [i for i, nid in enumerate(graph.nx_graph.nodes())
                if graph.nx_graph.nodes[nid].get("cell_type") == "ER4d"]
peg_indices = [i for i, nid in enumerate(graph.nx_graph.nodes())
               if graph.nx_graph.nodes[nid].get("cell_type") == "PEG"]

print(f"{'Scale':<10} {'Drive':<10} {'EPG Mean (t=50-100)':<25} {'PEG Mean (t=50-100)':<25}")
print("-" * 70)

for s in scales:
    for d in drives:
        net = NeuralNetwork(graph=graph, neuron_model_cls=RateNeuron, synapse_scale=s, dt=1.0)
        sim = DiscreteSimulator(net)
        inputs = np.zeros((100, net.num_neurons), dtype=np.float64)
        inputs[:, er4d_indices[:len(er4d_indices)//2]] = d
        trace = sim.run(num_steps=100, inputs=inputs)
        
        epg_m = trace[50:, epg_indices].mean()
        peg_m = trace[50:, peg_indices].mean()
        print(f"{s:<10.5f} {d:<10.2f} {epg_m:<25.4f} {peg_m:<25.4f}")
