"""
Phase 5 Connectome Agent with Hemispheric Protocerebral Bridge Decoding,
Multi-Step Recurrent Settling, and Panoramic Visual Compass Drive.
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
    Connectome-driven Drosophila navigation agent with biological Protocerebral Bridge
    hemispheric steering, multi-step recurrent settling, and pathway-specific plasticity.
    """

    def __init__(
        self,
        network: NeuralNetwork,
        sensor: Optional[Union[SectorVisualSensor, PanoramicCompoundEyeSensor, ActiveExplorationSensor]] = None,
        learning_rate: float = 0.002,
        eligibility_decay: float = 0.85,
        sensory_drive: float = 2.0,
        enable_plasticity: bool = True,
        plasticity_mode: str = "pathway",  # "pathway" | "global" | "none"
        decoder_mode: str = "hemispheric",  # "hemispheric" | "softmax" | "argmax"
        temperature: float = 0.2,           # softmax temperature
        sub_steps: int = 10,                # internal ODE steps per behavioral action
        steer_gain: float = 50.0,           # gain on hemispheric asymmetry
    ):
        self.network = network
        self.sensor = sensor or PanoramicCompoundEyeSensor()
        self.sensory_drive = sensory_drive
        self.enable_plasticity = enable_plasticity and (plasticity_mode != "none")
        self.plasticity_mode = plasticity_mode
        self.decoder_mode = decoder_mode
        self.temperature = max(temperature, 1e-6)
        self.sub_steps = sub_steps
        self.steer_gain = steer_gain
        self._rng = np.random.default_rng(None)

        # 1. Map all major Central Complex cell classes
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

        # 2. Map Protocerebral Bridge Left vs Right Hemispheres
        def _get_hemi_indices(cell_type: str) -> Tuple[List[int], List[int]]:
            nodes = [nid for nid in self.network.neuron_ids if self.network.graph.nx_graph.nodes[nid].get("cell_type") == cell_type]
            lefts = [self.network.id_to_idx[nid] for nid in nodes if "_L" in self.network.graph.nx_graph.nodes[nid].get("instance", "")]
            rights = [self.network.id_to_idx[nid] for nid in nodes if "_R" in self.network.graph.nx_graph.nodes[nid].get("instance", "")]
            return lefts, rights

        self.pena_L, self.pena_R = _get_hemi_indices("PEN_a(PEN1)")
        self.penb_L, self.penb_R = _get_hemi_indices("PEN_b(PEN2)")
        self.peg_L, self.peg_R = _get_hemi_indices("PEG")

        # 3. Build ordered circular EPG compass mapping (L8..L1, R1..R8)
        ordered_gloms = [f"L{i}" for i in range(8, 0, -1)] + [f"R{i}" for i in range(1, 9)]
        epg_by_glom = {g: [] for g in ordered_gloms}
        for nid in self.network.neuron_ids:
            if self.network.graph.nx_graph.nodes[nid].get("cell_type") == "EPG":
                inst = self.network.graph.nx_graph.nodes[nid].get("instance", "")
                for g in ordered_gloms:
                    if f"_{g}" in inst:
                        epg_by_glom[g].append(self.network.id_to_idx[nid])
                        break
        self.epg_ordered_indices = [idx for g in ordered_gloms for idx in epg_by_glom[g]]

        # 4. Setup Plasticity Mask
        num_n = self.network.num_neurons
        if plasticity_mode == "pathway":
            # ER4d -> EPG and EPG -> PEG
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
        else:
            self.plasticity_mask = np.zeros((num_n, num_n), dtype=np.float64)

        self.plasticity = RewardModulatedHebbian(
            learning_rate=learning_rate,
            eligibility_decay=eligibility_decay,
            plasticity_mask=self.plasticity_mask,
        )

        self.prev_activity = np.zeros(num_n, dtype=np.float64)
        self.current_activity = np.zeros(num_n, dtype=np.float64)

        # 5. Calibrate baseline hemispheric asymmetry at 0 deg relative bearing
        self.hemi_offset = 0.0
        if self.pena_L and self.pena_R:
            self.network.neurons.reset()
            ref_state = ArenaState(
                agent_pos=np.array([50.0, 50.0], dtype=np.float64),
                agent_heading=0.0,
                target_pos=np.array([80.0, 50.0], dtype=np.float64),
                step_count=0,
                target_reached=False,
                collision_occurred=False,
            )
            ref_signals = self.sensor.sense(ref_state)
            ref_ext = self._build_sensory_current(ref_signals)
            for _ in range(self.sub_steps):
                act_ref = self.network.step(ref_ext)
            pen_l0 = float(np.mean(act_ref[self.pena_L]))
            pen_r0 = float(np.mean(act_ref[self.pena_R]))
            self.hemi_offset = pen_l0 - pen_r0
            self.network.neurons.reset()

    def _build_sensory_current(self, sensory_signals: np.ndarray) -> np.ndarray:
        ext_current = np.zeros(self.network.num_neurons, dtype=np.float64)
        n_signals = len(sensory_signals)
        n_epg = len(self.epg_ordered_indices)

        if n_signals == 3:
            # 3-sector mapping (left, center, right) into circular compass
            s1, s2 = n_epg // 3, 2 * n_epg // 3
            ext_current[self.epg_ordered_indices[:s1]] += sensory_signals[0] * self.sensory_drive
            ext_current[self.epg_ordered_indices[s1:s2]] += sensory_signals[1] * self.sensory_drive
            ext_current[self.epg_ordered_indices[s2:]] += sensory_signals[2] * self.sensory_drive
        elif n_epg > 0:
            # Panoramic 12-channel mapping into circular 16-wedge compass ring
            for i, sig in enumerate(sensory_signals):
                epg_i = int((i / n_signals) * n_epg)
                if epg_i < n_epg:
                    ext_current[self.epg_ordered_indices[epg_i]] += sig * self.sensory_drive
        else:
            # Fallback to ER4 mapping if EPG glom labels missing
            n_er = len(self.er4d_indices)
            for i, sig in enumerate(sensory_signals):
                er_idx = int((i / n_signals) * n_er)
                if er_idx < n_er:
                    ext_current[self.er4d_indices[er_idx]] += sig * self.sensory_drive

        # Spontaneous baseline drive if sensory input is dark
        if np.sum(sensory_signals) < 1e-4 and len(self.er4d_indices) > 0:
            ext_current[self.er4d_indices] += 0.15

        return ext_current

    def act(self, state: ArenaState) -> int:
        # Relax membrane activity before processing new sensory frame to prevent runaway saturation
        self.network.neurons.reset()

        sensory_signals = self.sensor.sense(state)
        ext_current = self._build_sensory_current(sensory_signals)

        # Execute internal multi-step neural integration to settle recurrent dynamics
        self.prev_activity = self.current_activity.copy()
        for _ in range(self.sub_steps):
            self.current_activity = self.network.step(ext_current)

        # Action Decoding
        if self.decoder_mode == "hemispheric":
            # Protocerebral Bridge Left vs Right hemisphere asymmetric steering
            pen_l = float(np.mean(self.current_activity[self.pena_L])) if self.pena_L else 0.0
            pen_r = float(np.mean(self.current_activity[self.pena_R])) if self.pena_R else 0.0
            
            # Asymmetry differential zero-centered around calibrated baseline
            asym = ((pen_l - pen_r) - self.hemi_offset) * self.steer_gain

            # Descending forward gating: forward when asymmetry is small (aligned)
            fwd_score = max(0.0, 1.0 - abs(asym))
            # Corrective saccades: Action 2 = TURN_LEFT (ccw), Action 3 = TURN_RIGHT (cw)
            left_score = max(0.0, -asym)
            right_score = max(0.0, asym)

            scores = np.array([fwd_score, left_score, right_score])
            logits = scores / self.temperature
            logits -= logits.max()
            probs = np.exp(logits)
            probs /= probs.sum()
            choice = int(self._rng.choice(3, p=probs))

        elif self.decoder_mode == "softmax":
            scores = np.array([
                float(np.mean(self.current_activity[self.peg_indices])) if self.peg_indices else 0.0,
                float(np.mean(self.current_activity[self.pen_a_indices])) if self.pen_a_indices else 0.0,
                float(np.mean(self.current_activity[self.pen_b_indices])) if self.pen_b_indices else 0.0,
            ])
            logits = scores / self.temperature
            logits -= logits.max()
            probs = np.exp(logits)
            probs /= probs.sum()
            choice = int(self._rng.choice(3, p=probs))
        else:  # argmax
            scores = np.array([
                float(np.mean(self.current_activity[self.peg_indices])) if self.peg_indices else 0.0,
                float(np.mean(self.current_activity[self.pen_a_indices])) if self.pen_a_indices else 0.0,
                float(np.mean(self.current_activity[self.pen_b_indices])) if self.pen_b_indices else 0.0,
            ])
            choice = int(np.argmax(scores))

        return choice + 1

    def get_motor_scores(self) -> np.ndarray:
        """Return raw motor scores [Forward, Turn Left, Turn Right]."""
        if self.pena_L and self.pena_R:
            pen_l = float(np.mean(self.current_activity[self.pena_L]))
            pen_r = float(np.mean(self.current_activity[self.pena_R]))
            asym = ((pen_l - pen_r) - getattr(self, "hemi_offset", 0.0)) * self.steer_gain
            return np.array([max(0.0, 1.0 - abs(asym)), max(0.0, -asym), max(0.0, asym)])
        return np.array([0.33, 0.33, 0.33])

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
