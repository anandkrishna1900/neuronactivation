"""
Discrete-timestep neural simulation engine.
"""

from typing import List, Optional
import numpy as np
from flymind.brain.network import NeuralNetwork


class DiscreteSimulator:
    """Executes discrete-timestep simulation of a NeuralNetwork."""

    def __init__(self, network: NeuralNetwork):
        self.network = network

    def run(self, num_steps: int, inputs: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Run the network for num_steps.
        inputs: shape (num_steps, num_neurons) or None
        Returns: activity trace shape (num_steps, num_neurons)
        """
        history = []
        for t in range(num_steps):
            ext = inputs[t] if inputs is not None else None
            act = self.network.step(ext)
            history.append(act)
        return np.array(history)
