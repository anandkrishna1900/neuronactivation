"""
Phase 6B: Connectome-driven motor readout for Flappy Bird.

Replaces the hand-designed Phase 6A decoder with a biologically justified
motor readout where the connectome's PEG activity gates the flap decision.

Architecture:
    FlappyVisualSensor (9 channels)
        -> EPG compass ring (16 glomeruli) via directional mapping
        -> 261-neuron recurrent connectome dynamics
        -> PEG premotor activity (gating signal)
        -> Motor readout = gain * (sensor_vertical * peg_gate) + bias
        -> FLAP / NO-FLAP

The sensor provides gap-position information (vertical_signal).
PEG provides connectome-dependent gating (amplitude modulation).
The motor decision flows through the connectome: sensor -> EPG -> PEG -> gate.
"""

from typing import Optional, Dict, Any, List, Tuple
import numpy as np

from flymind.brain.network import NeuralNetwork
from flymind.brain.plasticity import RewardModulatedHebbian
from flymind.environment.flappy import FlappyState
from flymind.environment.flappy_sensor import FlappyVisualSensor, N_CHANNELS
from flymind.agent.fly import BaseAgent


class ConnectomeMotorReadoutAgent(BaseAgent):
    """
    Connectome-driven Flappy Bird agent with PEG-gated motor readout.

    Motor score = gain * (sensor_vertical * peg_gate) + bias
    P(flap) = sigmoid(motor_score / temperature)

    The sensor provides gap-position info, PEG gates the connectome-dependent
    timing. Both signals are required for the motor decision.
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
        motor_readout_lr: float = 0.01,
        motor_temperature: float = 1.0,
        motor_bias: float = -2.0,
        motor_gain: float = 80.0,
        decoder_mode: str = "stochastic",
        seed: Optional[int] = None,
    ):
        self.network = network
        self.sensor = sensor or FlappyVisualSensor()
        self.sensory_drive = sensory_drive
        self.enable_plasticity = enable_plasticity and (plasticity_mode != "none")
        self.plasticity_mode = plasticity_mode
        self.sub_steps = sub_steps
        self.motor_readout_lr = motor_readout_lr
        self.motor_temperature = max(motor_temperature, 1e-6)
        self.decoder_mode = decoder_mode
        self._rng = np.random.default_rng(seed)

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
        self.delta7_indices = [
            self.network.id_to_idx[nid] for nid in self.network.neuron_ids
            if self.network.graph.nx_graph.nodes[nid].get("cell_type") == "Delta7"
        ]

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

        self.peg_L, self.peg_R = _get_hemi("PEG")
        self.pena_L, self.pena_R = _get_hemi("PEN_a(PEN1)")
        self.penb_L, self.penb_R = _get_hemi("PEN_b(PEN2)")

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

        # ── Motor readout parameters ─────────────────────────────────────────
        self.motor_gain = motor_gain
        self.motor_bias = motor_bias
        # Temporal: store previous sensor vertical for velocity estimation
        self._prev_sensor_vertical = 0.0

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

    def _build_sensory_current(self, sensor_activations: np.ndarray) -> np.ndarray:
        """Map 9-channel sensor activations into the EPG compass ring."""
        ext_current = np.zeros(self.network.num_neurons, dtype=np.float64)
        n_epg = len(self.epg_ordered_indices)
        if n_epg == 0:
            return ext_current

        n_signals = len(sensor_activations)
        for i, sig in enumerate(sensor_activations):
            if sig < 1e-6:
                continue
            epg_i = int((i / n_signals) * n_epg)
            if epg_i < n_epg:
                ext_current[self.epg_ordered_indices[epg_i]] += sig * self.sensory_drive

        if np.sum(sensor_activations) < 1e-4 and len(self.er4d_indices) > 0:
            ext_current[self.er4d_indices] += 0.15

        return ext_current

    def _compute_sensor_vertical_signal(self, sensor_activations: np.ndarray) -> float:
        """Compute vertical gap-position signal from sensor."""
        vertical = (
            +1.0 * (sensor_activations[0] + sensor_activations[1] + sensor_activations[2])
            + 0.0 * (sensor_activations[3] + sensor_activations[4] + sensor_activations[5])
            - 1.0 * (sensor_activations[6] + sensor_activations[7] + sensor_activations[8])
        )
        total = np.sum(sensor_activations)
        if total > 1e-6:
            vertical /= total
        return float(vertical)

    def _decode_flap(self, sensor_vertical: float, peg_activity: np.ndarray) -> float:
        """
        Decode flap probability from sensor vertical signal and PEG gating.

        motor_score = gain * (sensor_vertical * peg_gate) + bias
        P(flap) = sigmoid(motor_score / temperature)

        The sensor provides gap-position information (positive = gap above).
        PEG provides connectome-dependent amplitude modulation.
        Both signals are required: sensor tells WHAT, PEG tells WHEN.
        """
        peg_mean = float(np.mean(peg_activity))
        peg_gate = np.clip(peg_mean * 10.0, 0.0, 1.0)

        motor_score = self.motor_gain * sensor_vertical * peg_gate + self.motor_bias
        z = motor_score / self.motor_temperature
        z = np.clip(z, -10.0, 10.0)
        return 1.0 / (1.0 + np.exp(-z))

    def act(self, state: FlappyState) -> int:
        """Produce a binary flap decision from the connectome."""
        self.network.neurons.reset()

        sensor_out = self.sensor.sense(state)
        self._last_sensor_activations = sensor_out

        sensor_vertical = self._compute_sensor_vertical_signal(sensor_out)

        ext_current = self._build_sensory_current(sensor_out)

        self.prev_activity = self.current_activity.copy()
        for _ in range(self.sub_steps):
            self.current_activity = self.network.step(ext_current)

        peg_activity = self.current_activity[self.peg_indices]
        flap_prob = self._decode_flap(sensor_vertical, peg_activity)

        if self.decoder_mode == "threshold":
            action = 1 if flap_prob > 0.5 else 0
        else:
            action = int(self._rng.binomial(1, min(max(flap_prob, 0.0), 1.0)))

        self._last_peg_activity = peg_activity
        self._last_flap_prob = flap_prob
        self._last_sensor_vertical = sensor_vertical
        self._last_action = action
        peg_gate = float(np.clip(np.mean(peg_activity) * 10.0, 0.0, 1.0))
        self._last_motor_score = float(self.motor_gain * sensor_vertical * peg_gate + self.motor_bias)

        # Store for next step's velocity estimate
        self._prev_sensor_vertical = sensor_vertical

        return action

    def _update_motor_readout(self, reward: float) -> None:
        """Update motor gain and bias via policy gradient."""
        if not hasattr(self, "_last_flap_prob") or not hasattr(self, "_last_sensor_vertical"):
            return
        advantage = float(self._last_action - self._last_flap_prob)
        peg_gate = float(np.clip(np.mean(self._last_peg_activity) * 10.0, 0.0, 1.0))
        self.motor_gain += self.motor_readout_lr * reward * advantage * self._last_sensor_vertical * peg_gate
        self.motor_bias += self.motor_readout_lr * reward * advantage

    def apply_reward(self, reward: float) -> None:
        if self.enable_plasticity:
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
        self._update_motor_readout(reward)

    def get_motor_diagnostics(self) -> Dict[str, float]:
        peg_act = getattr(self, "_last_peg_activity", np.zeros(len(self.peg_indices)))
        peg_gate = float(np.clip(np.mean(peg_act) * 10.0, 0.0, 1.0))
        return {
            "peg_mean": float(np.mean(peg_act)),
            "peg_gate": peg_gate,
            "sensor_vertical": getattr(self, "_last_sensor_vertical", 0.0),
            "motor_score": getattr(self, "_last_motor_score", 0.0),
            "flap_prob": getattr(self, "_last_flap_prob", 0.5),
            "motor_gain": self.motor_gain,
            "motor_bias": self.motor_bias,
        }

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
        self._last_peg_activity = None
        self._last_flap_prob = None
        self._last_sensor_vertical = None
        self._last_action = None
        self._last_motor_score = None
        self._prev_sensor_vertical = 0.0
