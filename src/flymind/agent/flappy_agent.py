"""
Phase 6A: Flappy Bird agent wrapping the existing 261-neuron FlyMind connectome.

Reuses the biological architecture (ER4d/ER4m visual pathways, EPG compass ring,
PEN_a/PEN_b hemispheric asymmetry, PEG motor output) with minimal adaptation
for the binary FLAP/NO-FLAP action space.

Clearly distinguishes:
  REAL CONNECTOME  (261 neurons, 19969 synapses, unchanged)
  ENVIRONMENT INTERFACE  (sensor mapping, motor decoder)
"""

from typing import Optional, Dict, Any, List, Tuple
import numpy as np

from flymind.brain.network import NeuralNetwork
from flymind.brain.plasticity import RewardModulatedHebbian
from flymind.environment.flappy import FlappyState
from flymind.environment.flappy_sensor import FlappyVisualSensor, N_CHANNELS
from flymind.agent.fly import BaseAgent


class FlappyConnectomeAgent(BaseAgent):
    """
    Connectome-driven Flappy Bird agent with binary action decoding.

    Architecture:
        FlappyVisualSensor (9 channels)
            -> EPG compass ring (16 glomeruli) via directional mapping
            -> 261-neuron recurrent connectome dynamics
            -> PEN_a hemispheric asymmetry -> FLAP / NO-FLAP decoder

    Action space:
        0 = NO FLAP
        1 = FLAP

    Plasticity:
        Pathway-specific: ER4d->EPG + EPG->PEG (same mask as Phase 5)
    """

    def __init__(
        self,
        network: NeuralNetwork,
        sensor: Optional[FlappyVisualSensor] = None,
        learning_rate: float = 0.002,
        eligibility_decay: float = 0.85,
        sensory_drive: float = 25.0,
        enable_plasticity: bool = True,
        plasticity_mode: str = "pathway",
        sub_steps: int = 10,
        # Motor decoder parameters
        flap_threshold: float = 0.0,    # asymmetry threshold for flap decision
        decoder_temperature: float = 0.5,
        decoder_mode: str = "threshold",  # "threshold" | "softmax" | "stochastic"
        # Stochastic decoder calibration
        flap_bias: float = -0.5,  # calibrated so neutral asymmetry -> ~38% flap
    ):
        self.network = network
        self.sensor = sensor or FlappyVisualSensor()
        self.sensory_drive = sensory_drive
        self.enable_plasticity = enable_plasticity and (plasticity_mode != "none")
        self.plasticity_mode = plasticity_mode
        self.sub_steps = sub_steps
        self.flap_threshold = flap_threshold
        self.decoder_temperature = max(decoder_temperature, 1e-6)
        self.decoder_mode = decoder_mode
        self.flap_bias = flap_bias
        self._rng = np.random.default_rng(None)

        # ── Map cell type indices ─────────────────────────────────────────────
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

        # ── Map hemispheres ──────────────────────────────────────────────────
        def _get_hemi(cell_type: str) -> Tuple[List[int], List[int]]:
            nodes = [
                nid for nid in self.network.neuron_ids
                if self.network.graph.nx_graph.nodes[nid].get("cell_type") == cell_type
            ]
            lefts = [
                self.network.id_to_idx[nid] for nid in nodes
                if "_L" in self.network.graph.nx_graph.nodes[nid].get("instance", "")
            ]
            rights = [
                self.network.id_to_idx[nid] for nid in nodes
                if "_R" in self.network.graph.nx_graph.nodes[nid].get("instance", "")
            ]
            return lefts, rights

        self.pena_L, self.pena_R = _get_hemi("PEN_a(PEN1)")
        self.penb_L, self.penb_R = _get_hemi("PEN_b(PEN2)")
        self.peg_L, self.peg_R = _get_hemi("PEG")

        # ── Ordered EPG compass ring ─────────────────────────────────────────
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

        # ── Plasticity mask ───────────────────────────────────────────────────
        num_n = self.network.num_neurons
        if plasticity_mode == "pathway":
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

        # ── Calibrate baseline hemispheric asymmetry ──────────────────────────
        self.hemi_offset = 0.0
        if self.pena_L and self.pena_R:
            self.network.neurons.reset()
            # Feed zero sensor input to establish baseline
            ref_ext = np.zeros(num_n, dtype=np.float64)
            if len(self.er4d_indices) > 0:
                ref_ext[self.er4d_indices] += 0.15  # spontaneous baseline
            for _ in range(self.sub_steps):
                act_ref = self.network.step(ref_ext)
            pen_l0 = float(np.mean(act_ref[self.pena_L]))
            pen_r0 = float(np.mean(act_ref[self.pena_R]))
            self.hemi_offset = pen_l0 - pen_r0
            self.network.neurons.reset()

    def _build_sensory_current(self, sensor_activations: np.ndarray) -> np.ndarray:
        """
        Map 9-channel sensor activations into the EPG compass ring.

        Layout mapping (9 channels -> 16 EPG positions):
            UL  -> L8    UC  -> L5    UR  -> L2
            CL  -> L7    CC  -> L4    CR  -> L1
            LL  -> L6    LC  -> L3    LR  -> R1

        This maps spatial position to compass-like directional encoding.
        """
        ext_current = np.zeros(self.network.num_neurons, dtype=np.float64)
        n_epg = len(self.epg_ordered_indices)
        if n_epg == 0:
            return ext_current

        # Map 9 sensor channels into the 16-position EPG ring
        # Use fractional indexing so the 9 channels span the ring
        n_signals = len(sensor_activations)
        for i, sig in enumerate(sensor_activations):
            if sig < 1e-6:
                continue
            epg_i = int((i / n_signals) * n_epg)
            if epg_i < n_epg:
                ext_current[self.epg_ordered_indices[epg_i]] += sig * self.sensory_drive

        # Only add spontaneous baseline if sensor is completely dark (no pipes visible)
        if np.sum(sensor_activations) < 1e-4 and len(self.er4d_indices) > 0:
            ext_current[self.er4d_indices] += 0.15

        return ext_current

    def act(self, state: FlappyState) -> int:
        """
        Produce a binary flap decision from the connectome.

        Returns:
            0 = NO FLAP, 1 = FLAP
        """
        # Reset membrane to prevent runaway saturation
        self.network.neurons.reset()

        # Get visual sensor activations
        sensor_out = self.sensor.sense(state)
        self._last_sensor_activations = sensor_out
        self._last_velocity = state.bird_vy

        # Map sensor to connectome input
        ext_current = self._build_sensory_current(sensor_out)

        # Multi-step recurrent settling
        self.prev_activity = self.current_activity.copy()
        for _ in range(self.sub_steps):
            self.current_activity = self.network.step(ext_current)

        # ── Motor decoding: binary flap decision ──────────────────────────────
        flap_prob = self._decode_flap_probability()

        if self.decoder_mode == "threshold":
            action = 1 if flap_prob > 0.5 else 0
        elif self.decoder_mode == "stochastic":
            action = int(self._rng.binomial(1, min(max(flap_prob, 0.0), 1.0)))
        else:  # softmax
            action = int(self._rng.binomial(1, min(max(flap_prob, 0.0), 1.0)))

        return action

    def _decode_flap_probability(self) -> float:
        """
        Decode sensory state into a flap probability.

        Uses the 9-channel sensor activations to determine whether the gap
        is above or below the bird, combined with velocity information.
        Flaps only when bird is below gap AND falling (or near zero velocity).
        This prevents overshooting the gap.
        """
        if not hasattr(self, "_last_sensor_activations"):
            return 0.5

        act = self._last_sensor_activations

        # Vertical position signal from sensor channels
        vertical_signal = (
            +1.0 * (act[0] + act[1] + act[2])   # upper row
            + 0.0 * (act[3] + act[4] + act[5])   # center row
            - 1.0 * (act[6] + act[7] + act[8])   # lower row
        )
        total = np.sum(act)
        if total > 1e-6:
            vertical_signal = vertical_signal / total

        # Velocity signal: flap only when falling (vy < 0) or slow (|vy| < threshold)
        vy = self._last_velocity if hasattr(self, "_last_velocity") else 0.0
        # Velocity gate: 1.0 when falling, 0.0 when rising fast
        vel_gate = np.clip(1.0 - max(0.0, vy) / 3.0, 0.0, 1.0)

        # Combined signal: gap above AND bird not rising fast
        combined = vertical_signal * vel_gate

        # Sigmoid mapping
        gain = 10.0
        z = gain * combined - 2.5
        flap_prob = 1.0 / (1.0 + np.exp(-z))

        return flap_prob

    def get_motor_diagnostics(self) -> Dict[str, float]:
        """Return diagnostic motor signals for analysis."""
        act = getattr(self, "_last_sensor_activations", np.zeros(9))
        vertical_signal = (
            +1.0 * (act[0] + act[1] + act[2])
            + 0.0 * (act[3] + act[4] + act[5])
            - 1.0 * (act[6] + act[7] + act[8])
        )
        total = np.sum(act)
        if total > 1e-6:
            vertical_signal = vertical_signal / total
        flap_prob = self._decode_flap_probability()

        return {
            "vertical_signal": float(vertical_signal),
            "sensor_total": float(total),
            "flap_prob": flap_prob,
        }

    def apply_reward(self, reward: float) -> None:
        """Apply reward-modulated plasticity to eligible synapses."""
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
            self.network.synapses.scale
            * self.network.synapses.raw_weights
            * self.network.synapses.signs[:, np.newaxis]
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


# ── Baseline Controllers ──────────────────────────────────────────────────────

class RandomFlapAgent(BaseAgent):
    """Random baseline: 50% chance of flap each step."""

    def __init__(self, flap_prob: float = 0.5, seed: Optional[int] = None):
        self.flap_prob = flap_prob
        self._rng = np.random.default_rng(seed)

    def act(self, state: FlappyState) -> int:
        return int(self._rng.binomial(1, self.flap_prob))

    def reset(self) -> None:
        pass


class FixedPeriodFlapAgent(BaseAgent):
    """Fixed-period baseline: flap every N steps."""

    def __init__(self, period: int = 8):
        self.period = max(period, 1)
        self._step = 0

    def act(self, state: FlappyState) -> int:
        self._step += 1
        if self._step % self.period == 0:
            return 1
        return 0

    def reset(self) -> None:
        self._step = 0


class HandDesignedFlapAgent(BaseAgent):
    """
    Hand-designed baseline: velocity-aware flap controller.
    Flaps only when bird is falling AND below the gap.
    This is a simple but non-trivial control strategy.
    """

    def __init__(self, gap_offset: float = 30.0):
        self.gap_offset = gap_offset  # target slightly below gap center

    def act(self, state: FlappyState) -> int:
        if not state.pipes:
            return 0

        # Find nearest pipe ahead of bird
        nearest = None
        for pipe in state.pipes:
            if pipe["x"] > 80.0:
                if nearest is None or pipe["x"] < nearest["x"]:
                    nearest = pipe

        if nearest is None:
            return 0

        # Target: slightly below gap center
        target_y = nearest["gap_center"] - self.gap_offset

        # Flap if bird is falling AND below target
        if state.bird_vy < 0 and state.bird_y < target_y:
            return 1
        return 0

    def reset(self) -> None:
        pass
