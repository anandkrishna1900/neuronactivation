"""
Local biologically inspired plasticity mechanisms.
Explicitly avoids global backpropagation in the biological connectome network.
"""

from abc import ABC, abstractmethod
import numpy as np


class PlasticityRule(ABC):
    """Abstract base class for local synaptic plasticity."""

    @abstractmethod
    def update(
        self,
        weights: np.ndarray,
        pre_activity: np.ndarray,
        post_activity: np.ndarray,
        reward: float = 0.0,
    ) -> np.ndarray:
        pass


class RewardModulatedHebbian(PlasticityRule):
    """
    Reward-modulated Hebbian learning rule:
    dW_ij = eta * R * (pre_i * post_j)
    """

    def __init__(self, learning_rate: float = 0.001):
        self.learning_rate = learning_rate

    def update(
        self,
        weights: np.ndarray,
        pre_activity: np.ndarray,
        post_activity: np.ndarray,
        reward: float = 0.0,
    ) -> np.ndarray:
        outer = np.outer(pre_activity, post_activity)
        delta_w = self.learning_rate * reward * outer
        return weights + delta_w
