"""
Connectome loader for deserializing cached subcircuit representations.
"""

import json
from pathlib import Path
from typing import Union
from flymind.connectome.graph import ConnectomeGraph, NeuronMetadata, SynapticConnection


class ConnectomeLoader:
    """Loads connectome data from processed local files or cache."""

    @staticmethod
    def load_from_json(path: Union[str, Path]) -> ConnectomeGraph:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Connectome file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        graph = ConnectomeGraph(name=data.get("name", "circuit"))

        for n in data.get("neurons", []):
            neuron = NeuronMetadata(
                body_id=n["body_id"],
                cell_type=n.get("cell_type"),
                instance=n.get("instance"),
                roi=n.get("roi"),
                neurotransmitter=n.get("neurotransmitter"),
                extra=n.get("extra", {}),
            )
            graph.add_neuron(neuron)

        for c in data.get("connections", []):
            conn = SynapticConnection(
                source_id=c["source_id"],
                target_id=c["target_id"],
                weight=float(c["weight"]),
                neurotransmitter=c.get("neurotransmitter"),
                confidence=c.get("confidence"),
                extra=c.get("extra", {}),
            )
            graph.add_connection(conn)

        return graph

    @staticmethod
    def save_to_json(graph: ConnectomeGraph, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        neurons = []
        for body_id, n in graph._neurons.items():
            neurons.append({
                "body_id": body_id,
                "cell_type": n.cell_type,
                "instance": n.instance,
                "roi": n.roi,
                "neurotransmitter": n.neurotransmitter,
                "extra": n.extra,
            })

        connections = []
        for u, v, data in graph.nx_graph.edges(data=True):
            connections.append({
                "source_id": u,
                "target_id": v,
                "weight": data.get("weight", 1.0),
                "neurotransmitter": data.get("neurotransmitter"),
                "confidence": data.get("confidence"),
                "extra": data.get("extra", {}),
            })

        payload = {
            "name": graph.name,
            "num_neurons": graph.num_neurons,
            "num_synapses": graph.num_synapses,
            "neurons": neurons,
            "connections": connections,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
