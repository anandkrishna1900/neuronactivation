"""
NeuPrint query interface for extracting biological subcircuits from Hemibrain.

Biological data source:
  Scheffer et al. (2020) A connectome and analysis of the adult Drosophila central brain.
  eLife 9:e57448. https://doi.org/10.7554/eLife.57448
  Accessed via: https://neuprint.janelia.org  (dataset: hemibrain:v1.2.1)

Cell type naming follows Hemibrain v1.2.1 conventions:
  EPG         = compass neurons (E-PG in literature)
  PEN_a(PEN1) = angular velocity integrators (P-EN1)
  PEN_b(PEN2) = angular velocity integrators (P-EN2)
  Delta7      = global inhibitory ring neurons
  PEG         = premotor steering output neurons (P-EG)
  PFNd/PFNv   = fan-shaped body navigation columns
  ER4d/ER4m   = visual ring input neurons

All claims backed by:
  Hulse et al. (2021) eLife 10:e66039
  Turner-Evans et al. (2020) Neuron 108:145-163
"""

import os
import time
import json
from pathlib import Path
from typing import List, Optional, Dict, Any

import requests

from flymind.connectome.graph import ConnectomeGraph, NeuronMetadata, SynapticConnection
from flymind.connectome.validation import validate_connectome_graph


# ── Neurotransmitter signs as per Drosophila CX biology ──────────────────────
# Source: Eckstein et al. (2024) Nature 634:153-162
# EPG, PEN, PEG, PFN = cholinergic (excitatory in Drosophila CNS)
# Delta7 = GABAergic (inhibitory)
# ER (ring neurons) = cholinergic
KNOWN_NEUROTRANSMITTERS: Dict[str, str] = {
    "EPG":          "cholinergic",
    "PEN_a(PEN1)":  "cholinergic",
    "PEN_b(PEN2)":  "cholinergic",
    "Delta7":       "gabaergic",
    "PEG":          "cholinergic",
    "PFNd":         "cholinergic",
    "PFNv":         "cholinergic",
    "ER4d":         "cholinergic",
    "ER4m":         "cholinergic",
    "ER2a":         "cholinergic",
    "EL":           "cholinergic",
}

# ── Central Complex V1 circuit definition ─────────────────────────────────────
# Selected based on heading direction and steering circuitry
# (Seelig & Jayaraman 2015; Hulse et al. 2021; Turner-Evans et al. 2020)
CX_HEADING_CIRCUIT_TYPES: List[str] = [
    "EPG",           # 46 neurons — heading compass ring attractor
    "PEN_a(PEN1)",   # 20 neurons — angular velocity, shifts bump left
    "PEN_b(PEN2)",   # 22 neurons — angular velocity, shifts bump right
    "Delta7",        # 42 neurons — long-range inhibitory ring stabilizer
    "PEG",           # 18 neurons — premotor steering output
    "PFNd",          # 40 neurons — fan-shaped body navigation columns
    "PFNv",          # 20 neurons — fan-shaped body navigation columns
    "ER4d",          # 25 neurons — visual landmark ring input
    "ER4m",          # 10 neurons — visual motion ring input
    "EL",            # 18 neurons — ellipsoid body local neurons
]


class NeuPrintQueryClient:
    """
    Abstracted query interface for the neuPrint connectome analysis service.
    Supports Hemibrain v1.2.1 and MANC v1.0 datasets.
    """

    def __init__(
        self,
        server: str = "neuprint.janelia.org",
        dataset: str = "hemibrain:v1.2.1",
        token: Optional[str] = None,
        timeout: int = 120,
    ):
        self.server = server
        self.dataset = dataset
        self.timeout = timeout
        self.token = token or os.environ.get("NEUPRINT_APPLICATION_CREDENTIALS", "")
        if not self.token:
            raise ValueError(
                "neuPrint API token required. Set NEUPRINT_APPLICATION_CREDENTIALS "
                "environment variable or pass token= to NeuPrintQueryClient."
            )
        self._base_url = f"https://{server}"
        self._headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    def cypher(self, query: str) -> Dict[str, Any]:
        """Execute a Cypher query against neuPrint and return raw response dict."""
        url = f"{self._base_url}/api/custom/custom"
        payload = {"cypher": query, "dataset": self.dataset}
        resp = requests.post(url, headers=self._headers, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def ping(self) -> bool:
        """Check connectivity to neuPrint server."""
        resp = requests.get(
            f"{self._base_url}/api/version",
            headers=self._headers,
            timeout=10,
        )
        return resp.status_code == 200

    def fetch_neurons_by_type(self, cell_types: List[str]) -> List[Dict]:
        """
        Fetch all neurons matching the given cell types.
        Returns list of dicts with keys: bodyId, type, instance, status.
        """
        types_str = ", ".join(f"'{t}'" for t in cell_types)
        query = f"""
        MATCH (n:`hemibrain_Neuron`)
        WHERE n.type IN [{types_str}]
        RETURN n.bodyId AS bodyId,
               n.type AS type,
               n.instance AS instance,
               n.status AS status,
               n.pre AS pre,
               n.post AS post
        ORDER BY n.type, n.instance
        """
        r = self.cypher(query)
        cols = r["columns"]
        return [dict(zip(cols, row)) for row in r["data"]]

    def fetch_adjacency_within(self, body_ids: List[int], min_weight: int = 1) -> List[Dict]:
        """
        Fetch all synaptic connections among a specified set of neurons.
        min_weight: minimum synapse count to include (filters noise).
        Returns list of dicts with keys: src, tgt, weight.
        """
        # neuPrint has a parameter limit; batch if needed
        ids_str = ", ".join(str(b) for b in body_ids)
        query = f"""
        MATCH (a:`hemibrain_Neuron`)-[c:ConnectsTo]->(b:`hemibrain_Neuron`)
        WHERE a.bodyId IN [{ids_str}]
          AND b.bodyId IN [{ids_str}]
          AND c.weight >= {min_weight}
        RETURN a.bodyId AS src, b.bodyId AS tgt, c.weight AS weight
        ORDER BY c.weight DESC
        """
        r = self.cypher(query)
        cols = r["columns"]
        return [dict(zip(cols, row)) for row in r["data"]]

    def build_cx_heading_graph(
        self,
        cell_types: Optional[List[str]] = None,
        min_synapse_weight: int = 1,
    ) -> ConnectomeGraph:
        """
        Full pipeline: fetch CX heading circuit neurons and adjacency,
        construct and validate a ConnectomeGraph.

        BIOLOGICAL DATA: neuron identities, types, and synaptic connectivity
          from Hemibrain v1.2.1 (Scheffer et al. 2020).
        MODEL ASSUMPTION: neurotransmitter sign assigned from KNOWN_NEUROTRANSMITTERS
          lookup based on Eckstein et al. 2024 predictions.
        """
        if cell_types is None:
            cell_types = CX_HEADING_CIRCUIT_TYPES

        print(f"[1/4] Fetching neurons for {len(cell_types)} cell types...")
        neurons = self.fetch_neurons_by_type(cell_types)
        print(f"      -> {len(neurons)} neurons found")

        graph = ConnectomeGraph(name="cx_heading_v1")
        body_ids = []

        for n in neurons:
            nt = KNOWN_NEUROTRANSMITTERS.get(n["type"], None)
            meta = NeuronMetadata(
                body_id=int(n["bodyId"]),
                cell_type=n["type"],
                instance=n["instance"],
                roi="CX",
                neurotransmitter=nt,
                extra={
                    "status": n.get("status"),
                    "pre": n.get("pre"),
                    "post": n.get("post"),
                },
            )
            graph.add_neuron(meta)
            body_ids.append(int(n["bodyId"]))

        print(f"[2/4] Fetching synaptic adjacency among {len(body_ids)} neurons "
              f"(min_weight={min_synapse_weight})...")
        connections = self.fetch_adjacency_within(body_ids, min_weight=min_synapse_weight)
        print(f"      -> {len(connections)} directed connections found")

        for c in connections:
            conn = SynapticConnection(
                source_id=int(c["src"]),
                target_id=int(c["tgt"]),
                weight=float(c["weight"]),
                # Neurotransmitter derived from presynaptic neuron type
                neurotransmitter=graph.get_neuron(int(c["src"])).neurotransmitter
                    if graph.get_neuron(int(c["src"])) else None,
            )
            graph.add_connection(conn)

        print("[3/4] Validating graph...")
        report = validate_connectome_graph(graph)
        print(f"      -> {report['num_neurons']} neurons, "
              f"{report['num_synaptic_connections']} synaptic connections — VALID")

        return graph
