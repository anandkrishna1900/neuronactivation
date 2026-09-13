"""
Neuron models (Rate-based and Leaky Integrate-and-Fire).
Dynamics are strictly separated from connectome topology.
"""

from abc import ABC, abstractmethod
import numpy as np


class NeuronModel(ABC):
    """Abstract base class for neuron dynamics."""

    @abstractmethod
    def step(self, input_current: np.ndarray, dt: float) -> np.ndarray:
        """Advance neuron state by dt given input current."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state to resting baseline."""
        pass


class RateNeuron(NeuronModel):
    """Continuous rate-based neuron model with leaky integration and non-linear activation."""

    def __init__(
        self,
        num_neurons: int,
        tau: float = 10.0,
        baseline: float = 0.0,
        gain: float = 1.0,
    ):
        self.num_neurons = num_neurons
        self.tau = tau
        self.baseline = baseline
        self.gain = gain
        self.activity = np.full(num_neurons, baseline, dtype=np.float64)

    def step(self, input_current: np.ndarray, dt: float) -> np.ndarray:
        # Leaky rate dynamics: tau * dr/dt = -r + phi(I)
        target = np.tanh(np.maximum(0.0, self.gain * (input_current + self.baseline)))
        self.activity += (dt / self.tau) * (-self.activity + target)
        return self.activity.copy()

    def reset(self) -> None:
        self.activity.fill(self.baseline)


class LIFNeuron(NeuronModel):
    """Leaky Integrate-and-Fire (LIF) spiking neuron model."""

    def __init__(
        self,
        num_neurons: int,
        tau_m: float = 20.0,
        v_rest: float = -70.0,
        v_reset: float = -75.0,
        v_thresh: float = -50.0,
    ):
        self.num_neurons = num_neurons
        self.tau_m = tau_m
        self.v_rest = v_rest
        self.v_reset = v_reset
        self.v_thresh = v_thresh
        self.v = np.full(num_neurons, v_rest, dtype=np.float64)
        self.spikes = np.zeros(num_neurons, dtype=bool)

    def step(self, input_current: np.ndarray, dt: float) -> np.ndarray:
        # Reset spiked neurons
        self.v[self.spikes] = self.v_reset
        # Membrane voltage update
        dv = (dt / self.tau_m) * (-(self.v - self.v_rest) + input_current)
        self.v += dv
        # Threshold detection
        self.spikes = self.v >= self.v_thresh
        return self.spikes.astype(np.float64)

    def reset(self) -> None:
        self.v.fill(self.v_rest)
        self.spikes.fill(False)
