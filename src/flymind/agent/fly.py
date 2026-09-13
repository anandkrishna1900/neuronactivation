"""
Fly agent implementations binding sensors, brain, and motor actions.
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

from flymind.brain.network import NeuralNetwork
from flymind.environment.world import ArenaState
from flymind.environment.sensors import SectorVisualSensor
from flymind.environment.actions import ActionType


class BaseAgent(ABC):
    """Abstract agent interface."""

    @abstractmethod
    def act(self, state: ArenaState) -> int:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass


class RandomFlyAgent(BaseAgent):
    """Baseline agent selecting actions uniformly at random."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def act(self, state: ArenaState) -> int:
        # Bias slightly toward forward motion to avoid trivial standing still
        return int(self.rng.choice([1, 2, 3]))

    def reset(self) -> None:
        pass


class ConnectomeFlyAgent(BaseAgent):
    """
    Agent driven by a biological connectome-derived neural network.
    
    Sensors -> Input Ring Neurons -> Network Dynamics -> Motor Populations -> Action
    """

    def __init__(
        self,
        network: NeuralNetwork,
        sensor: Optional[SectorVisualSensor] = None,
        input_neuron_indices: Optional[np.ndarray] = None,
        motor_neuron_indices: Optional[np.ndarray] = None,
    ):
        self.network = network
        self.sensor = sensor or SectorVisualSensor()
        self.input_indices = input_neuron_indices
        self.motor_indices = motor_neuron_indices

    def act(self, state: ArenaState) -> int:
        visual_signals = self.sensor.sense(state)  # [left, center, right]

        # Prepare external current injection vector
        ext_current = np.zeros(self.network.num_neurons, dtype=np.float64)

        if self.input_indices is not None and len(self.input_indices) >= 3:
            # Map left, center, right visual sectors into assigned input populations
            ext_current[self.input_indices[0]] = visual_signals[0] * 10.0
            ext_current[self.input_indices[1]] = visual_signals[1] * 10.0
            ext_current[self.input_indices[2]] = visual_signals[2] * 10.0

        # Step brain dynamics
        activity = self.network.step(ext_current)

        # Motor decoding
        if self.motor_indices is not None and len(self.motor_indices) >= 3:
            # Populations correspond to [forward, turn_left, turn_right]
            motor_acts = activity[self.motor_indices]
            choice = int(np.argmax(motor_acts))
            # Map choice (0, 1, 2) -> (FORWARD=1, TURN_LEFT=2, TURN_RIGHT=3)
            return choice + 1

        # Default fallback: forward
        return int(ActionType.FORWARD)

    def reset(self) -> None:
        self.network.reset()
