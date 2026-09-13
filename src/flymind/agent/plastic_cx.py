"""
Phase 4 Connectome Agent with Panoramic Visual Receptive Fields,
Pathway-Specific Plasticity Masking, and Real-Time Behavioral Tracking.
"""

from typing import Optional, List, Tuple, Union
import numpy as np

from flymind.brain.network import NeuralNetwork
from flymind.brain.plasticity import RewardModulatedHebbian
from flymind.environment.world import ArenaState
from flymind.environment.sensors import SectorVisualSensor, PanoramicCompoundEyeSensor, ActiveExplorationSensor
from flymind.agent.fly import BaseAgent


class PlasticCXAgent(BaseAgent):
    """
    Connectome-driven artificial Drosophila agent for Phase 4 navigation and behavioral variability.
    
    Sensory Pipeline:
      Supports SectorVisualSensor (3 channels), PanoramicCompoundEyeSensor (12 channels), or ActiveExplorationSensor.
      Receptive field signals are mapped into ER4d / ER4m ring neurons.
    
    Plasticity:
      Pathway-Specific (ER4d -> EPG & EPG -> PEG) or Global or Unplastic.
    """

    def __init__(
        self,
        network: NeuralNetwork,
        sensor: Optional[Union[SectorVisualSensor, PanoramicCompoundEyeSensor, ActiveExplorationSensor]] = None,
        learning_rate: float = 0.002,
        eligibility_decay: float = 0.85,
        sensory_drive: float = 2.0,
        enable_plasticity: bool = True,
        plasticity_mode: str = "pathway", # "pathway" or "global" or "none"
    ):
        self.network = network
        self.sensor = sensor or PanoramicCompoundEyeSensor()
        self.sensory_drive = sensory_drive
        self.enable_plasticity = enable_plasticity and (plasticity_mode != "none")
        self.plasticity_mode = plasticity_mode

        # 1. Map ER4d / ER4m ring neurons
        self.er4d_indices = [
            self.network.id_to_idx[nid] for nid in self.network.neuron_ids
            if self.network.graph.nx_graph.nodes[nid].get("cell_type") in ("ER4d", "ER4m")
        ]
        self.epg_indices = [
            self.network.id_to_idx[nid] for nid in self.network.neuron_ids
            if self.network.graph.nx_graph.nodes[nid].get("cell_type") == "EPG"
        ]
        self.peg_indices = [
            self.network.id_to_idx[nid] for nid in self.network.neuron_ids
            if self.network.graph.nx_graph.nodes[nid].get("cell_type") == "PEG"
        ]
        self.pen_a_indices = [
            self.network.id_to_idx[nid] for nid in self.network.neuron_ids
            if self.network.graph.nx_graph.nodes[nid].get("cell_type") == "PEN_a(PEN1)"
        ]
        self.pen_b_indices = [
            self.network.id_to_idx[nid] for nid in self.network.neuron_ids
            if self.network.graph.nx_graph.nodes[nid].get("cell_type") == "PEN_b(PEN2)"
        ]

        # 2. Setup Motor Sectors: Action 1 = FORWARD (PEG), Action 2 = TURN_LEFT (PEN_a), Action 3 = TURN_RIGHT (PEN_b)
        self.motor_sectors = [
            self.peg_indices,     # Index 0 -> Action 1 (FORWARD)
            self.pen_a_indices,   # Index 1 -> Action 2 (TURN_LEFT)
            self.pen_b_indices,   # Index 2 -> Action 3 (TURN_RIGHT)
        ]

        # 3. Setup Plasticity Mask
        num_n = self.network.num_neurons
        if plasticity_mode == "pathway":
            # Only ER4d -> EPG and EPG -> PEG
            self.plasticity_mask = np.zeros((num_n, num_n), dtype=np.float64)
            for u in self.er4d_indices:
                for v in self.epg_indices:
                    if self.network.synapses.raw_weights[u, v] > 0:
                        self.plasticity_mask[u, v] = 1.0
            for u in self.epg_indices:
                for v in self.peg_indices:
                    if self.network.synapses.raw_weights[u, v] > 0:
                        self.plasticity_mask[u, v] = 1.0
        elif plasticity_mode == "global":
            self.plasticity_mask = (self.network.synapses.raw_weights > 0).astype(np.float64)
        else: # none
            self.plasticity_mask = np.zeros((num_n, num_n), dtype=np.float64)

        self.plasticity = RewardModulatedHebbian(
            learning_rate=learning_rate,
            eligibility_decay=eligibility_decay,
            plasticity_mask=self.plasticity_mask,
        )

        self.prev_activity = np.zeros(num_n, dtype=np.float64)
        self.current_activity = np.zeros(num_n, dtype=np.float64)

    def act(self, state: ArenaState) -> int:
        sensory_signals = self.sensor.sense(state)
        ext_current = np.zeros(self.network.num_neurons, dtype=np.float64)

        n_signals = len(sensory_signals)
        n_er = len(self.er4d_indices)

        if n_signals == 3:
            # 3-sector mapping
            s1, s2 = n_er // 3, 2 * n_er // 3
            ext_current[self.er4d_indices[:s1]] = sensory_signals[0] * self.sensory_drive
            ext_current[self.er4d_indices[s1:s2]] = sensory_signals[1] * self.sensory_drive
            ext_current[self.er4d_indices[s2:]] = sensory_signals[2] * self.sensory_drive
        else:
            # Panoramic 12-channel mapping into 35 ring neurons
            for i, sig in enumerate(sensory_signals):
                er_idx = int((i / n_signals) * n_er)
                if er_idx < n_er:
                    ext_current[self.er4d_indices[er_idx]] += sig * self.sensory_drive

        # Spontaneous baseline drive
        if np.sum(sensory_signals) < 1e-4:
            ext_current[self.er4d_indices] += 0.15

        self.prev_activity = self.current_activity.copy()
        self.current_activity = self.network.step(ext_current)

        # Motor decoding: Forward (PEG), Turn Left (PEN_a), Turn Right (PEN_b)
        scores = [
            float(np.mean(self.current_activity[self.peg_indices])) if len(self.peg_indices) > 0 else 0.0,
            float(np.mean(self.current_activity[self.pen_a_indices])) if len(self.pen_a_indices) > 0 else 0.0,
            float(np.mean(self.current_activity[self.pen_b_indices])) if len(self.pen_b_indices) > 0 else 0.0,
        ]

        choice = int(np.argmax(scores))
        return choice + 1

    def apply_reward(self, reward: float) -> None:
        if not self.enable_plasticity:
            return

        updated = self.plasticity.update(
            raw_weights=self.network.synapses.raw_weights,
            pre_activity=self.prev_activity,
            post_activity=self.current_activity,
            reward=reward,
        )
        self.network.synapses.raw_weights = updated
        self.network.synapses.effective_weights = (
            self.network.synapses.scale * self.network.synapses.raw_weights * self.network.synapses.signs[:, np.newaxis]
        )

    def freeze_weights(self) -> None:
        self.enable_plasticity = False

    def unfreeze_weights(self) -> None:
        if self.plasticity_mode != "none":
            self.enable_plasticity = True

    def reset(self) -> None:
        self.network.reset()
        self.prev_activity = np.zeros(self.network.num_neurons, dtype=np.float64)
        self.current_activity = np.zeros(self.network.num_neurons, dtype=np.float64)
        self.plasticity.reset_traces(self.network.num_neurons)
