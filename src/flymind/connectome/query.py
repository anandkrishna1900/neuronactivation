"""
Abstracted interface for querying remote connectome databases (neuPrint / FlyWire).
"""

from typing import List, Optional
import os
from flymind.connectome.graph import ConnectomeGraph, NeuronMetadata, SynapticConnection


class NeuPrintQueryClient:
    """Wrapper around neuprint-python client for querying Hemibrain and MANC."""

    def __init__(
        self,
        server: str = "neuprint.janelia.org",
        dataset: str = "hemibrain:v1.2.1",
        token: Optional[str] = None,
    ):
        self.server = server
        self.dataset = dataset
        self.token = token or os.environ.get("NEUPRINT_APPLICATION_CREDENTIALS")
        self._client = None

    def connect(self):
        """Lazy connection to neuPrint server."""
        if not self.token:
            raise ValueError(
                "neuPrint API token required. Set NEUPRINT_APPLICATION_CREDENTIALS "
                "or pass token to NeuPrintQueryClient."
            )
        from neuprint import Client
        self._client = Client(self.server, dataset=self.dataset, token=self.token)
        return self._client

    def fetch_subcircuit(
        self,
        rois: Optional[List[str]] = None,
        cell_types: Optional[List[str]] = None,
    ) -> ConnectomeGraph:
        """Query neurons and synaptic adjacencies for specified ROIs and cell types."""
        # Detailed query logic implemented in Phase 2
        raise NotImplementedError("Subcircuit querying will be activated in Phase 2.")
