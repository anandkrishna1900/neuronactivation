"""
Synapse models and neurotransmitter sign mapping.
"""

from typing import Dict, Optional
import numpy as np


# Sign mapping according to Drosophila neurotransmitter biology
# Acetylcholine: excitatory (+1)
# GABA: inhibitory (-1)
# Glutamate: inhibitory in Drosophila central brain via glutamate-gated chloride channels (GluCl) (-1)
# Dopamine / Serotonin / Octopamine: modulatory (default 0 or small +1 in V1)
DEFAULT_NEUROTRANSMITTER_SIGNS: Dict[str, float] = {
    "acetylcholine": 1.0,
    "cholinergic": 1.0,
    "gaba": -1.0,
    "gabaergic": -1.0,
    "glutamate": -1.0,
    "glutamatergic": -1.0,
    "dopamine": 0.0,
    "serotonin": 0.0,
    "octopamine": 0.0,
}


class SynapseModel:
    """Manages effective synaptic weights incorporating sign and scaling."""

    def __init__(
        self,
        weight_matrix: np.ndarray,
        signs: Optional[np.ndarray] = None,
        scale: float = 0.01,
    ):
        self.raw_weights = weight_matrix
        self.scale = scale
        if signs is None:
            self.signs = np.ones(weight_matrix.shape[0])
        else:
            self.signs = signs

        # Effective weight W_eff[i, j] = sign[i] * scale * raw_weight[i, j]
        self.effective_weights = (self.signs[:, None] * self.raw_weights) * self.scale

    def compute_postsynaptic_current(self, presynaptic_activity: np.ndarray) -> np.ndarray:
        """Matrix-vector product of presynaptic firing and effective weights."""
        return presynaptic_activity @ self.effective_weights
