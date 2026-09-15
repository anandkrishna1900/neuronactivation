"""
Local biologically inspired plasticity mechanisms with eligibility traces and pathway masking.
Explicitly avoids global backpropagation in the biological connectome network.
"""

from abc import ABC, abstractmethod
from typing import Optional, Set, Tuple
import numpy as np


class PlasticityRule(ABC):
    """Abstract base class for local synaptic plasticity."""

    @abstractmethod
    def update(
        self,
        raw_weights: np.ndarray,
        pre_activity: np.ndarray,
        post_activity: np.ndarray,
        reward: float = 0.0,
    ) -> np.ndarray:
        pass


class RewardModulatedHebbian(PlasticityRule):
    """
    3-Factor Reward-Modulated Hebbian learning with Eligibility Traces:
    
    Eligibility Trace:
    e_ij(t) = gamma * e_ij(t-1) + pre_i(t) * post_j(t)
    
    Synaptic Weight Update:
    dW_ij = eta * e_ij(t) * R_t * mask_ij
    
    W_ij(t+1) = clip(W_ij(t) + dW_ij, W_min, W_max)
    """

    def __init__(
        self,
        learning_rate: float = 0.002,
        eligibility_decay: float = 0.85,
        weight_min: float = 0.0,
        weight_max: float = 250.0,
        plasticity_mask: Optional[np.ndarray] = None,
    ):
        self.learning_rate = learning_rate
        self.eligibility_decay = eligibility_decay
        self.weight_min = weight_min
        self.weight_max = weight_max
        self.plasticity_mask = plasticity_mask
        self.eligibility_trace: Optional[np.ndarray] = None

    def reset_traces(self, num_neurons: int) -> None:
        """Reset eligibility trace matrix at episode boundaries."""
        self.eligibility_trace = np.zeros((num_neurons, num_neurons), dtype=np.float64)

    def step_eligibility(self, pre_activity: np.ndarray, post_activity: np.ndarray) -> None:
        """Accumulate pre x post coincidence into eligibility traces."""
        if self.eligibility_trace is None or self.eligibility_trace.shape[0] != len(pre_activity):
            self.reset_traces(len(pre_activity))

        coincidence = np.outer(pre_activity, post_activity)
        self.eligibility_trace = (self.eligibility_decay * self.eligibility_trace) + coincidence

    def update(
        self,
        raw_weights: np.ndarray,
        pre_activity: np.ndarray,
        post_activity: np.ndarray,
        reward: float = 0.0,
    ) -> np.ndarray:
        """Apply dopamine/reward modulation strictly masked to designated plastic edges.

        Uses soft-bound plasticity (BCM-style): update magnitude scales with
        (1 - W/W_max), so weights approaching the ceiling naturally stop being
        modified.  This prevents monotonic drift while preserving all other
        biological properties of the 3-factor rule.
        """
        self.step_eligibility(pre_activity, post_activity)

        if abs(reward) < 1e-7:
            return raw_weights

        dW = self.learning_rate * reward * self.eligibility_trace

        # Soft-bound: as W -> W_max, the update shrinks to zero.
        # Biologically analogous to synaptic saturation / BCM-style bounds.
        if self.weight_max > 0:
            soft_bound = np.clip(1.0 - raw_weights / self.weight_max, 0.0, 1.0)
            dW = dW * soft_bound

        # Enforce anatomical pathway mask: non-mask connections strictly receive 0 update
        if self.plasticity_mask is not None:
            dW = dW * self.plasticity_mask
        else:
            dW = dW * (raw_weights > 0).astype(np.float64)

        updated_weights = np.clip(raw_weights + dW, self.weight_min, self.weight_max)
        return updated_weights
