"""
Phase 2 Connectome Agent with biological ring sensory input, EPG compass interaction,
motor population decoding, and 3-factor eligibility-trace plasticity.
"""

from typing import Optional, List, Tuple
import numpy as np

from flymind.brain.network import NeuralNetwork
from flymind.brain.plasticity import RewardModulatedHebbian
from flymind.environment.world import ArenaState
from flymind.environment.sensors import SectorVisualSensor
from flymind.agent.fly import BaseAgent


class PlasticCXAgent(BaseAgent):
    """
    Connectome-driven artificial Drosophila agent for Phase 2 navigation.
    
    Sensory Pipeline:
      Compound Eye Sector Sensor -> [Left, Center, Right] -> ER4d / ER4m Ring Neurons
    
    Central Circuit Pipeline:
      ER4d Ring Neurons -> EPG Compass Neurons -> Delta7 Inhibitory Ring / PEN Integrators -> PEG Motor Channels
    
    Motor Decoding:
      PEG / PEN sub-populations -> [Forward, Turn Left, Turn Right, Stop]
    
    Plasticity:
      Local 3-Factor Hebbian with Eligibility Traces on anatomically permitted synapses.
    """

    def __init__(
        self,
        network: NeuralNetwork,
        sensor: Optional[SectorVisualSensor] = None,
        learning_rate: float = 0.002,
        eligibility_decay: float = 0.85,
        sensory_drive: float = 2.0,
        enable_plasticity: bool = True,
    ):
        self.network = network
        self.sensor = sensor or SectorVisualSensor()
        self.sensory_drive = sensory_drive
        self.enable_plasticity = enable_plasticity

        # 1. Map sensory input populations (ER4d / ER4m ring neurons)
        self.er4d_indices = [
            self.network.id_to_idx[nid] for nid in self.network.neuron_ids
            if self.network.graph.nx_graph.nodes[nid].get("cell_type") in ("ER4d", "ER4m")
        ]
        n_in = len(self.er4d_indices)
        s1, s2 = n_in // 3, 2 * n_in // 3
        self.input_sectors = [
            self.er4d_indices[:s1],       # Left receptive sector
            self.er4d_indices[s1:s2],     # Center receptive sector
            self.er4d_indices[s2:],       # Right receptive sector
        ]

        # 2. Map motor output populations (PEG & PEN steering neurons)
        self.peg_indices = [
            self.network.id_to_idx[nid] for nid in self.network.neuron_ids
            if self.network.graph.nx_graph.nodes[nid].get("cell_type") in ("PEG", "PEN_a(PEN1)", "PEN_b(PEN2)")
        ]
        n_out = len(self.peg_indices)
        m1, m2 = n_out // 3, 2 * n_out // 3
        
        # Correctly aligned: Index 0 -> Action 1 (FORWARD), Index 1 -> Action 2 (TURN_LEFT), Index 2 -> Action 3 (TURN_RIGHT)
        self.motor_sectors = [
            self.peg_indices[:m1],        # Forward population (index 0 -> +1 = 1: FORWARD)
            self.peg_indices[m1:m2],      # Turn Left population (index 1 -> +1 = 2: TURN_LEFT)
            self.peg_indices[m2:],        # Turn Right population (index 2 -> +1 = 3: TURN_RIGHT)
        ]

        # 3. Setup anatomical plasticity mask (only existing non-zero connections may adapt)
        self.plasticity_mask = (self.network.synapses.raw_weights > 0).astype(np.float64)
        self.plasticity = RewardModulatedHebbian(
            learning_rate=learning_rate,
            eligibility_decay=eligibility_decay,
            plasticity_mask=self.plasticity_mask,
        )

        # State tracking
        self.prev_activity = np.zeros(self.network.num_neurons, dtype=np.float64)
        self.current_activity = np.zeros(self.network.num_neurons, dtype=np.float64)

    def act(self, state: ArenaState) -> int:
        """
        Sensory input -> Neural integration -> Motor population decoding -> Action
        """
        visual_signals = self.sensor.sense(state)  # [left, center, right]
        ext_current = np.zeros(self.network.num_neurons, dtype=np.float64)

        # Inject visual signals into corresponding ER4d ring neuron populations
        for sec_idx, neuron_grp in enumerate(self.input_sectors):
            for nid_idx in neuron_grp:
                ext_current[nid_idx] = visual_signals[sec_idx] * self.sensory_drive

        # Add small spontaneous background exploratory drive if no visual stimulus is present
        if np.sum(visual_signals) < 1e-4:
            # Baseline spontaneous activity in ring neurons to permit continuous steering
            ext_current[self.er4d_indices] += 0.15

        self.prev_activity = self.current_activity.copy()
        self.current_activity = self.network.step(ext_current)

        # Motor decoding: compare average firing rate across motor populations
        action_scores = []
        for m_grp in self.motor_sectors:
            if len(m_grp) > 0:
                action_scores.append(float(np.mean(self.current_activity[m_grp])))
            else:
                action_scores.append(0.0)

        # Action: 0 -> FORWARD (1), 1 -> TURN_LEFT (2), 2 -> TURN_RIGHT (3)
        choice = int(np.argmax(action_scores))
        return choice + 1

    def apply_reward(self, reward: float) -> None:
        """Apply 3-factor eligibility-trace Hebbian update if plasticity is enabled."""
        if not self.enable_plasticity:
            return

        updated_weights = self.plasticity.update(
            raw_weights=self.network.synapses.raw_weights,
            pre_activity=self.prev_activity,
            post_activity=self.current_activity,
            reward=reward,
        )
        self.network.synapses.raw_weights = updated_weights
        self.network.synapses.effective_weights = (
            self.network.synapses.scale * self.network.synapses.raw_weights * self.network.synapses.signs[:, np.newaxis]
        )

    def freeze_weights(self) -> None:
        """Freeze synaptic weights for held-out evaluation."""
        self.enable_plasticity = False

    def unfreeze_weights(self) -> None:
        """Enable synaptic plasticity for training."""
        self.enable_plasticity = True

    def reset(self) -> None:
        """Reset internal neural dynamics and eligibility traces."""
        self.network.reset()
        self.prev_activity = np.zeros(self.network.num_neurons, dtype=np.float64)
        self.current_activity = np.zeros(self.network.num_neurons, dtype=np.float64)
        self.plasticity.reset_traces(self.network.num_neurons)
