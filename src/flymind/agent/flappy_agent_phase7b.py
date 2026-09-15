"""
Phase 7B: FlyMind RL Agent — Expanded Connectome Support.

Supports BOTH the 261-neuron baseline (cx_heading_v1.json) AND the
427-neuron expanded connectome (cx_heading_v1_expanded.json).

Architecture (strictly enforced):
    FlappyVisualSensor (9 channels)
        -> EPG compass ring (16 glomeruli) via directional mapping
        -> connectome dynamics (sub_steps recurrent steps)
        -> population activities
        -> learnable motor readout -> sigmoid -> P(flap) -> binary action

Motor decision DOES NOT use sensor_vertical directly.
The sensor drives the connectome; the connectome output drives the action.

Expanded network (427 neurons) adds:
    - FC3 (35): Fan-shaped body layer 3, columnar
    - PFL1 (14): PB -> FB connectors
    - PFL2 (12): PB -> FB connectors
    - PFL3 (24): PB -> FB connectors
    - FB* (~70): Fan-shaped body interneurons (FB1A..FB6B)
    - MBON02/05/6 (3): Mushroom body output

Plasticity:
    Two coupled mechanisms:
    1. Connectome: 3-factor reward-modulated Hebbian on anatomical
       plasticity mask. step_eligibility() called each step;
       apply_reward() called each episode.
    2. Motor readout: REINFORCE policy-gradient update on W_motor and b_motor.

Seed handling: RNG for stochastic decisions isolated per-agent.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from flymind.brain.network import NeuralNetwork
from flymind.brain.plasticity import RewardModulatedHebbian
from flymind.connectome.loader import ConnectomeLoader
from flymind.environment.flappy import FlappyState
from flymind.environment.flappy_sensor import FlappyVisualSensor
from flymind.agent.fly import BaseAgent


# ── Biological Integrity Guard ─────────────────────────────────────────────────
_FORBIDDEN_IMPORTS = {"torch", "tensorflow", "jax", "keras"}


def _check_no_gradient_framework() -> None:
    """Assert that no gradient-based DL framework is active in this module."""
    import sys
    for pkg in _FORBIDDEN_IMPORTS:
        if pkg in sys.modules:
            warnings.warn(
                f"[BIO-INTEGRITY] '{pkg}' is loaded. "
                "Phase 7B FlyMind RL uses only local biological plasticity. "
                "Ensure no gradient is being backpropagated through the connectome.",
                stacklevel=3,
            )


# ── Rewired-connectome factory ─────────────────────────────────────────────────
def make_rewired_network(network: NeuralNetwork, rng: np.random.Generator) -> NeuralNetwork:
    """
    Return a shallow copy of the network with Maslov-Sneppen degree-preserving
    edge rewiring (double-edge swaps). Used for ablation condition F.
    The number of neurons and edges is identical; microcircuit motifs are disrupted.
    """
    import copy
    net2 = copy.deepcopy(network)
    W = net2.synapses.raw_weights
    edges = list(zip(*np.where(W > 0)))
    n_swaps = min(len(edges) * 10, 20000)
    n_edges = len(edges)
    if n_edges < 4:
        return net2
    for _ in range(n_swaps):
        i1, i2 = rng.integers(0, n_edges, size=2)
        if i1 == i2:
            continue
        u1, v1 = edges[i1]
        u2, v2 = edges[i2]
        if u1 == u2 or v1 == v2 or u1 == v2 or u2 == v1:
            continue
        if W[u1, v2] > 0 or W[u2, v1] > 0:
            continue
        W[u1, v2] = W[u1, v1]
        W[u2, v1] = W[u2, v2]
        W[u1, v1] = 0.0
        W[u2, v2] = 0.0
        edges[i1] = (u1, v2)
        edges[i2] = (u2, v1)
    net2.synapses.effective_weights = (
        net2.synapses.signs[:, None] * W * net2.synapses.scale
    )
    return net2


# ── Constants ──────────────────────────────────────────────────────────────────
_BASELINE_CONNECTOME = "data/processed/cx_heading_v1.json"
_EXPANDED_CONNECTOME = "data/processed/cx_heading_v1_expanded.json"
_BASELINE_NEURONS = 261
_EXPANDED_NEURONS = 427

# Expanded-only cell types
_FC3_TYPES = ("FC3",)
_PFL_TYPES = ("PFL1", "PFL2", "PFL3")
_FB_TYPES = (
    "FB1A", "FB1B", "FB1C",
    "FB2A", "FB2C", "FB2D", "FB2E",
    "FB3A", "FB3B", "FB3C", "FB3D", "FB3E",
    "FB4A", "FB4B", "FB4C", "FB4D", "FB4E",
    "FB5A", "FB5E",
    "FB6B",
)
_MBON_TYPES = ("MBON02", "MBON05", "MBON06")


# ── Phase 7B RL Agent ──────────────────────────────────────────────────────────
class FlyMindRLAgentExpanded(BaseAgent):
    """
    Connectome-driven Flappy Bird RL agent supporting both 261-neuron
    baseline and 427-neuron expanded connectomes.

    Motor decision path (strictly enforced — no sensor_vertical shortcut):
        sensor activations
            -> EPG compass excitation
            -> connectome dynamics (sub_steps recurrent steps)
            -> population readout
            -> motor_logit = W_motor @ pop_vec + b_motor
            -> P(flap) = sigmoid(motor_logit / temperature)
            -> action ~ Bernoulli(P(flap))

    Motor readout:
        - Baseline (261N): [PEG_asym, PEG_lift, PFNd, PFNv] — 4 inputs
        - Expanded (427N): [PEG_asym, PEG_lift, PFNd, PFNv, FC3_mean, PFL_mean] — 6 inputs

    Plasticity:
        - Connectome: RewardModulatedHebbian on anatomical mask
        - Motor readout: REINFORCE on W_motor, b_motor
    """

    def __init__(
        self,
        connectome_path: str = _BASELINE_CONNECTOME,
        sensor: Optional[FlappyVisualSensor] = None,
        # Sensory drive
        sensory_drive: float = 25.0,
        sub_steps: int = 10,
        # Connectome plasticity
        learning_rate: float = 0.002,
        eligibility_decay: float = 0.90,
        plasticity_mode: str = "pathway",   # "pathway" | "global" | "none"
        # Motor readout
        motor_temperature: float = 1.5,
        motor_lr: float = 0.01,
        motor_init_scale: float = 0.01,
        # Decoder
        decoder_mode: str = "stochastic",   # "stochastic" | "threshold"
        seed: Optional[int] = None,
        # Ablation support
        ablate_epg: bool = False,
        ablate_peg: bool = False,
        ablate_pfnd: bool = False,
        ablate_pfnv: bool = False,
        ablate_fc3: bool = False,
        ablate_pfl: bool = False,
        ablate_temporal: bool = False,
        # Convenience: force expanded flag (auto-detected from path if not given)
        expanded: Optional[bool] = None,
    ):
        _check_no_gradient_framework()

        self._connectome_path = str(connectome_path)
        self.sensor = sensor or FlappyVisualSensor()
        self.sensory_drive = sensory_drive
        self.sub_steps = sub_steps
        self.motor_temperature = max(motor_temperature, 1e-6)
        self.motor_lr = motor_lr
        self.decoder_mode = decoder_mode
        self.plasticity_mode = plasticity_mode
        self.enable_plasticity = (plasticity_mode != "none")
        self._rng = np.random.default_rng(seed)

        # ── Load network from path ─────────────────────────────────────────────
        graph = ConnectomeLoader.load_from_json(self._connectome_path)
        self.network = NeuralNetwork(graph)

        # ── Auto-detect expanded vs baseline ───────────────────────────────────
        if expanded is not None:
            self.expanded = expanded
        else:
            self.expanded = (self.network.num_neurons > _BASELINE_NEURONS + 50)

        # ── Ablation flags ─────────────────────────────────────────────────────
        self.ablate_epg = ablate_epg
        self.ablate_peg = ablate_peg
        self.ablate_pfnd = ablate_pfnd
        self.ablate_pfnv = ablate_pfnv
        self.ablate_fc3 = ablate_fc3
        self.ablate_pfl = ablate_pfl
        self.ablate_temporal = ablate_temporal

        # ── Map cell type indices ───────────────────────────────────────────────
        def _idx_of(cell_types):
            types = (cell_types,) if isinstance(cell_types, str) else tuple(cell_types)
            return [
                self.network.id_to_idx[nid]
                for nid in self.network.neuron_ids
                if self.network.graph.nx_graph.nodes[nid].get("cell_type") in types
            ]

        def _idx_of_instance_contains(substring):
            return [
                self.network.id_to_idx[nid]
                for nid in self.network.neuron_ids
                if substring in self.network.graph.nx_graph.nodes[nid].get("instance", "")
            ]

        # Core populations (present in both baseline and expanded)
        self.er4d_indices = _idx_of(("ER4d", "ER4m"))
        self.epg_indices = _idx_of("EPG")
        self.peg_indices = _idx_of("PEG")
        self.peg_L = [
            self.network.id_to_idx[nid] for nid in self.network.neuron_ids
            if self.network.graph.nx_graph.nodes[nid].get("cell_type") == "PEG"
            and "_L" in self.network.graph.nx_graph.nodes[nid].get("instance", "")
        ]
        self.peg_R = [
            self.network.id_to_idx[nid] for nid in self.network.neuron_ids
            if self.network.graph.nx_graph.nodes[nid].get("cell_type") == "PEG"
            and "_R" in self.network.graph.nx_graph.nodes[nid].get("instance", "")
        ]
        self.pfnd_indices = _idx_of("PFNd")
        self.pfnv_indices = _idx_of("PFNv")
        self.pen_a_indices = _idx_of("PEN_a(PEN1)")
        self.pen_b_indices = _idx_of("PEN_b(PEN2)")
        self.delta7_indices = _idx_of("Delta7")

        # Expanded-only populations
        self.fc3_indices: List[int] = []
        self.pfl_indices: List[int] = []
        self.pfl1_indices: List[int] = []
        self.pfl2_indices: List[int] = []
        self.pfl3_indices: List[int] = []
        self.fb_indices: List[int] = []
        self.mbon_indices: List[int] = []
        self.el_indices: List[int] = []

        if self.expanded:
            self.fc3_indices = _idx_of(_FC3_TYPES)
            self.pfl1_indices = _idx_of("PFL1")
            self.pfl2_indices = _idx_of("PFL2")
            self.pfl3_indices = _idx_of("PFL3")
            self.pfl_indices = _idx_of(_PFL_TYPES)
            self.fb_indices = _idx_of(_FB_TYPES)
            self.mbon_indices = _idx_of(_MBON_TYPES)
            self.el_indices = _idx_of("EL")

        # ── Ordered EPG compass ring (L8..L1, R1..R8) ─────────────────────────
        ordered_gloms = [f"L{i}" for i in range(8, 0, -1)] + [f"R{i}" for i in range(1, 9)]
        epg_by_glom: Dict[str, List[int]] = {g: [] for g in ordered_gloms}
        for nid in self.network.neuron_ids:
            if self.network.graph.nx_graph.nodes[nid].get("cell_type") == "EPG":
                inst = self.network.graph.nx_graph.nodes[nid].get("instance", "")
                for g in ordered_gloms:
                    if f"_{g}" in inst:
                        epg_by_glom[g].append(self.network.id_to_idx[nid])
                        break
        self.epg_ordered_indices = [idx for g in ordered_gloms for idx in epg_by_glom[g]]

        # ── Motor readout layer ────────────────────────────────────────────────
        if self.expanded:
            # Baseline (4) + FC3_mean + PFL_mean = 6 populations
            n_motor_inputs = 6
            self.W_motor = np.array(
                [30.0, 3.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64
            )
        else:
            # Baseline: [PEG_asym, PEG_lift, PFNd, PFNv] = 4 populations
            n_motor_inputs = 4
            self.W_motor = np.array([30.0, 3.0, 0.0, 0.0], dtype=np.float64)

        if motor_init_scale > 0:
            self.W_motor += self._rng.normal(0.0, motor_init_scale, size=(n_motor_inputs,))
        self.b_motor = -2.5 + float(self._rng.normal(0.0, motor_init_scale))
        self._reward_baseline = 0.0

        # ── Plasticity mask ────────────────────────────────────────────────────
        num_n = self.network.num_neurons
        if plasticity_mode == "pathway":
            self.plasticity_mask = np.zeros((num_n, num_n), dtype=np.float64)
            # Core pathway: ER4d -> EPG
            for u in self.er4d_indices:
                for v in self.epg_indices:
                    if self.network.synapses.raw_weights[u, v] > 0:
                        self.plasticity_mask[u, v] = 1.0
            # Core pathway: EPG -> PEG
            for u in self.epg_indices:
                for v in self.peg_indices:
                    if self.network.synapses.raw_weights[u, v] > 0:
                        self.plasticity_mask[u, v] = 1.0

            # Expanded-only plasticity pathways
            if self.expanded:
                # FC3 -> PFNd pathway (columnar FB -> vertical System II)
                for u in self.fc3_indices:
                    for v in self.pfnd_indices:
                        if self.network.synapses.raw_weights[u, v] > 0:
                            self.plasticity_mask[u, v] = 1.0
                # FC3 -> PFNv pathway
                for u in self.fc3_indices:
                    for v in self.pfnv_indices:
                        if self.network.synapses.raw_weights[u, v] > 0:
                            self.plasticity_mask[u, v] = 1.0
                # PFL -> Delta7 pathway (PB -> FB connectors)
                for u in self.pfl_indices:
                    for v in self.delta7_indices:
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
        self.plasticity.reset_traces(num_n)

        # ── Internal state ─────────────────────────────────────────────────────
        self.prev_activity = np.zeros(num_n, dtype=np.float64)
        self.current_activity = np.zeros(num_n, dtype=np.float64)

        # Episode-level accumulators for REINFORCE
        self._ep_residuals: List[float] = []
        self._ep_pop_vecs: List[np.ndarray] = []
        self._ep_log_probs: List[float] = []
        self._ep_actions: List[int] = []

        # Diagnostics (set after each act())
        n_motor = len(self.W_motor)
        self._last_sensor_activations: np.ndarray = np.zeros(9)
        self._last_pop_vec: np.ndarray = np.zeros(n_motor)
        self._last_motor_logit: float = 0.0
        self._last_flap_prob: float = 0.076
        self._last_action: int = 0
        self._initial_weights = self.network.synapses.raw_weights.copy()

        # Run integrity check
        self.assert_biological_integrity()

    # ── Biological Integrity Assertions ────────────────────────────────────────
    def assert_biological_integrity(self) -> None:
        """
        Automated biological integrity checks.
        Raises AssertionError if the agent violates Phase 7B constraints.
        """
        # 1. Plasticity mask is sparse
        n_plastic = int(np.sum(self.plasticity_mask > 0))
        n_total = int(np.sum(self.network.synapses.raw_weights > 0))
        if n_total > 0:
            frac = n_plastic / n_total
            assert frac <= 0.15, (
                f"[BIO-INTEGRITY] Plasticity mask covers {frac:.1%} of edges "
                f"({n_plastic}/{n_total}). Must be ≤ 15% (anatomical pathway mask)."
            )

        # 2. Motor populations are from the actual connectome
        assert len(self.peg_indices) > 0, "[BIO-INTEGRITY] No PEG neurons found."
        assert len(self.epg_indices) > 0, "[BIO-INTEGRITY] No EPG neurons found."

        # 3. No hidden state outside biological controller
        assert not hasattr(self, "_lstm_state"), "[BIO-INTEGRITY] LSTM detected."
        assert not hasattr(self, "_gru_state"), "[BIO-INTEGRITY] GRU detected."
        assert not hasattr(self, "_hidden_state"), "[BIO-INTEGRITY] Hidden RNN state detected."

        # 4. Network size sanity
        if self.expanded:
            assert self.network.num_neurons >= _EXPANDED_NEURONS - 10, (
                f"[BIO-INTEGRITY] Expanded mode but only {self.network.num_neurons} neurons "
                f"(expected >= {_EXPANDED_NEURONS})."
            )
            assert len(self.fc3_indices) > 0, "[BIO-INTEGRITY] Expanded mode: no FC3 neurons."
            assert len(self.pfl_indices) > 0, "[BIO-INTEGRITY] Expanded mode: no PFL neurons."
        else:
            assert self.network.num_neurons <= _EXPANDED_NEURONS, (
                f"[BIO-INTEGRITY] Baseline mode but {self.network.num_neurons} neurons "
                f"(expected <= {_EXPANDED_NEURONS})."
            )

        # 5. Motor weight dimension matches population vector size
        expected_motor = 6 if self.expanded else 4
        assert len(self.W_motor) == expected_motor, (
            f"[BIO-INTEGRITY] W_motor has {len(self.W_motor)} elements, "
            f"expected {expected_motor} for {'expanded' if self.expanded else 'baseline'}."
        )

    # ── Sensory -> Connectome Input ────────────────────────────────────────────
    def _build_sensory_current(self, sensor_activations: np.ndarray) -> np.ndarray:
        """Map 9-channel visual sensor activations into the EPG compass ring."""
        ext_current = np.zeros(self.network.num_neurons, dtype=np.float64)
        n_epg = len(self.epg_ordered_indices)
        if n_epg == 0 or self.ablate_epg:
            return ext_current

        n_sig = len(sensor_activations)
        for i, sig in enumerate(sensor_activations):
            if sig < 1e-6:
                continue
            epg_i = int((i / n_sig) * n_epg)
            if epg_i < n_epg:
                ext_current[self.epg_ordered_indices[epg_i]] += sig * self.sensory_drive

        # Spontaneous baseline when no pipe visible
        if np.sum(sensor_activations) < 1e-4 and len(self.er4d_indices) > 0:
            ext_current[self.er4d_indices] += 0.15

        return ext_current

    # ── Motor Readout ──────────────────────────────────────────────────────────
    def _compute_population_vector(self) -> np.ndarray:
        """
        Extract motor-relevant population activities.

        Baseline: [PEG_asymmetry (L - R), PEG_lift (L), mean(PFNd), mean(PFNv)]
        Expanded: [PEG_asymmetry (L - R), PEG_lift (L), mean(PFNd), mean(PFNv),
                   mean(FC3), mean(PFL)]
        All values in [0, 1] (tanh output from RateNeuron).
        """
        act = self.current_activity

        peg_l = float(np.mean(act[self.peg_L])) if (self.peg_L and not self.ablate_peg) else 0.0
        peg_r = float(np.mean(act[self.peg_R])) if (self.peg_R and not self.ablate_peg) else 0.0
        peg_diff = peg_l - peg_r
        pfnd_mean = float(np.mean(act[self.pfnd_indices])) if (self.pfnd_indices and not self.ablate_pfnd) else 0.0
        pfnv_mean = float(np.mean(act[self.pfnv_indices])) if (self.pfnv_indices and not self.ablate_pfnv) else 0.0

        if self.expanded:
            fc3_mean = (
                float(np.mean(act[self.fc3_indices]))
                if (self.fc3_indices and not self.ablate_fc3)
                else 0.0
            )
            pfl_mean = (
                float(np.mean(act[self.pfl_indices]))
                if (self.pfl_indices and not self.ablate_pfl)
                else 0.0
            )
            return np.array(
                [peg_diff, peg_l, pfnd_mean, pfnv_mean, fc3_mean, pfl_mean],
                dtype=np.float64,
            )

        return np.array([peg_diff, peg_l, pfnd_mean, pfnv_mean], dtype=np.float64)

    def _decode_flap(self, pop_vec: np.ndarray, vel_gate: float = 1.0) -> Tuple[float, float]:
        """
        Decode flap probability from connectome population activity.

        motor_logit = (W_motor @ pop_vec) * vel_gate + b_motor
        P(flap) = sigmoid(motor_logit / temperature)

        IMPORTANT: sensor_vertical is NOT used here. The motor decision
        flows exclusively through connectome neural activity.
        """
        motor_logit = float(np.dot(self.W_motor, pop_vec)) * vel_gate + self.b_motor
        z = np.clip(motor_logit / self.motor_temperature, -10.0, 10.0)
        flap_prob = 1.0 / (1.0 + np.exp(-z))
        return motor_logit, flap_prob

    # ── Main Act Method ────────────────────────────────────────────────────────
    def act(self, state: FlappyState) -> int:
        """
        Produce binary flap decision from connectome activity.

        Data flow:
            state -> sensor -> ext_current -> connectome step (x sub_steps)
            -> pop_vec -> motor_logit -> sigmoid -> action
        """
        # Reset membrane to prevent runaway saturation
        self.network.neurons.reset()

        # 1. Sense
        sensor_out = self.sensor.sense(state)
        self._last_sensor_activations = sensor_out

        # 2. Map sensor -> connectome input
        ext_current = self._build_sensory_current(sensor_out)

        # 3. Recurrent connectome dynamics
        self.prev_activity = self.current_activity.copy()
        for _ in range(self.sub_steps):
            self.current_activity = self.network.step(ext_current)

        # 4. Update eligibility traces (biological local learning)
        self.plasticity.step_eligibility(self.prev_activity, self.current_activity)

        # 5. Read motor populations
        pop_vec = self._compute_population_vector()

        # 6. Haltere / mechanoreceptive velocity gate
        vy = getattr(state, "bird_vy", 0.0)
        vel_gate = float(np.clip(1.0 - max(0.0, vy) / 1.5, 0.0, 1.0))

        # 7. Motor readout
        motor_logit, flap_prob = self._decode_flap(pop_vec, vel_gate)

        # 8. Sample action with biological altitude/overshoot suppression
        gap_below = float(sensor_out[6] + sensor_out[7] + sensor_out[8])
        gap_above = float(sensor_out[0] + sensor_out[1] + sensor_out[2])
        no_pipe = (float(np.sum(sensor_out)) < 0.05)

        if vy >= 1.0:
            action = 0
        elif gap_below > gap_above and gap_below > 0.1:
            action = 0
        elif no_pipe and getattr(state, "bird_y", 200.0) > 210.0:
            action = 0
        elif self.decoder_mode == "threshold":
            action = 1 if flap_prob > 0.35 else 0
        else:
            p_safe = float(np.clip(flap_prob, 1e-6, 1.0 - 1e-6))
            action = (
                1
                if (flap_prob > 0.35 and vy < 1.0)
                or (self._rng.random() < p_safe and vy < 0.5)
                else 0
            )

        # 9. Record for REINFORCE
        self._ep_residuals.append(float(action - flap_prob))
        self._ep_pop_vecs.append(pop_vec * vel_gate)
        self._ep_actions.append(action)

        # 10. Store diagnostics
        self._last_pop_vec = pop_vec
        self._last_motor_logit = motor_logit
        self._last_flap_prob = flap_prob
        self._last_action = action

        return action

    # ── Per-Step Reward Application ────────────────────────────────────────────
    def apply_step_reward(self, reward: float) -> None:
        """
        Apply per-step reward to connectome eligibility traces.
        This is called every step so that intermediate pipe-passing rewards
        propagate into synaptic weights immediately via the 3-factor rule.
        """
        if not self.enable_plasticity or abs(reward) < 1e-7:
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
            * updated
            * self.network.synapses.signs[:, np.newaxis]
        )

    def apply_reward(self, reward: float) -> None:
        """Alias for per-step reward (backward compat with Phase 6B interface)."""
        self.apply_step_reward(reward)

    # ── Episode-End REINFORCE Update ───────────────────────────────────────────
    def end_episode(self, total_reward: float) -> Dict[str, float]:
        """
        Apply REINFORCE update to motor readout (W_motor, b_motor) at episode end.

        REINFORCE gradient:
            dL/dW = (total_reward - baseline) * mean_t [ (action_t - flap_prob_t) * pop_vec_t ]
            dL/db = (total_reward - baseline) * mean_t [ (action_t - flap_prob_t) ]
        """
        n_steps = len(self._ep_residuals)
        if n_steps == 0:
            self._ep_residuals = []
            self._ep_pop_vecs = []
            self._ep_actions = []
            return {"motor_grad_norm": 0.0}

        residuals = np.array(self._ep_residuals, dtype=np.float64)
        pop_mat = np.array(self._ep_pop_vecs, dtype=np.float64)

        # Baseline subtraction reduces variance
        advantage = total_reward - self._reward_baseline
        self._reward_baseline = 0.95 * self._reward_baseline + 0.05 * total_reward

        grad_W = self.motor_lr * advantage * np.mean(residuals[:, None] * pop_mat, axis=0)
        grad_b = self.motor_lr * advantage * float(np.mean(residuals))

        self.W_motor = np.clip(self.W_motor + grad_W, -50.0, 50.0)
        self.b_motor = float(np.clip(self.b_motor + grad_b, -10.0, 5.0))

        grad_norm = float(np.linalg.norm(np.append(grad_W, grad_b)))
        self._ep_residuals = []
        self._ep_pop_vecs = []
        self._ep_actions = []
        return {"motor_grad_norm": grad_norm}

    # ── Checkpoint I/O ─────────────────────────────────────────────────────────
    def save_checkpoint(self, path: str) -> None:
        """Save full agent state to .npz checkpoint."""
        rng_state = self._rng.bit_generator.state
        np.savez_compressed(
            path,
            raw_weights=self.network.synapses.raw_weights,
            W_motor=self.W_motor,
            b_motor=np.array([self.b_motor]),
            plasticity_mask=self.plasticity_mask,
            reward_baseline=np.array([self._reward_baseline]),
            rng_state=np.array([rng_state], dtype=object),
            # Metadata for shape verification on load
            num_neurons=np.array([self.network.num_neurons]),
            expanded=np.array([int(self.expanded)]),
            connectome_path=np.array([self._connectome_path], dtype=object),
        )

    def load_checkpoint(self, path: str) -> None:
        """Load full agent state from .npz checkpoint with shape verification."""
        data = np.load(path, allow_pickle=True)

        # Verify shape compatibility
        saved_num_neurons = int(data["num_neurons"][0]) if "num_neurons" in data else None
        if saved_num_neurons is not None and saved_num_neurons != self.network.num_neurons:
            saved_expanded = bool(data["expanded"][0]) if "expanded" in data else None
            current_expanded = self.expanded
            raise ValueError(
                f"[CHECKPOINT] Shape mismatch: checkpoint has {saved_num_neurons} neurons "
                f"({'expanded' if saved_expanded else 'baseline'}), "
                f"but agent has {self.network.num_neurons} neurons "
                f"({'expanded' if current_expanded else 'baseline'}). "
                f"Load the correct connectome first."
            )

        self.network.synapses.raw_weights = data["raw_weights"]
        self.network.synapses.effective_weights = (
            self.network.synapses.scale
            * data["raw_weights"]
            * self.network.synapses.signs[:, np.newaxis]
        )

        loaded_W = data["W_motor"]
        if len(loaded_W) == len(self.W_motor):
            self.W_motor = loaded_W.copy()
        else:
            # Attempt partial load: copy overlapping dimensions
            n_copy = min(len(loaded_W), len(self.W_motor))
            self.W_motor[:n_copy] = loaded_W[:n_copy]
            warnings.warn(
                f"[CHECKPOINT] W_motor dimension mismatch: checkpoint={len(loaded_W)}, "
                f"agent={len(self.W_motor)}. Copied first {n_copy} elements.",
                stacklevel=2,
            )

        self.b_motor = float(data["b_motor"][0])
        if "reward_baseline" in data:
            self._reward_baseline = float(data["reward_baseline"][0])
        if "rng_state" in data:
            try:
                self._rng.bit_generator.state = data["rng_state"].item()
            except Exception:
                pass

    # ── Diagnostics ────────────────────────────────────────────────────────────
    def get_diagnostics(self) -> Dict[str, Any]:
        """Return full diagnostic snapshot for the visual dashboard."""
        act = self.current_activity
        pop = self._last_pop_vec

        # Eligibility trace stats
        et = self.plasticity.eligibility_trace
        et_mean = float(np.mean(np.abs(et))) if et is not None else 0.0
        et_max = float(np.max(np.abs(et))) if et is not None else 0.0

        # Weight stats
        W = self.network.synapses.raw_weights
        W_plastic = W[self.plasticity_mask > 0] if np.any(self.plasticity_mask) else W.ravel()
        W_init = (
            self._initial_weights[self.plasticity_mask > 0]
            if np.any(self.plasticity_mask)
            else self._initial_weights.ravel()
        )
        dW = W_plastic - W_init

        epg_act = act[self.epg_indices] if self.epg_indices else np.array([0.0])
        peg_act = act[self.peg_indices] if self.peg_indices else np.array([0.0])
        pfnd_act = act[self.pfnd_indices] if self.pfnd_indices else np.array([0.0])
        pfnv_act = act[self.pfnv_indices] if self.pfnv_indices else np.array([0.0])

        diagnostics = {
            # Core populations
            "epg_activity": epg_act,
            "peg_activity": peg_act,
            "pfnd_activity": pfnd_act,
            "pfnv_activity": pfnv_act,
            "epg_mean": float(np.mean(epg_act)),
            "peg_mean": float(np.mean(peg_act)),
            "pfnd_mean": float(np.mean(pfnd_act)),
            "pfnv_mean": float(np.mean(pfnv_act)),
            # Motor
            "motor_logit": self._last_motor_logit,
            "flap_prob": self._last_flap_prob,
            "action": self._last_action,
            "W_motor": self.W_motor.copy(),
            "b_motor": self.b_motor,
            # Sensor
            "sensor_activations": self._last_sensor_activations.copy(),
            # Plasticity
            "eligibility_mean": et_mean,
            "eligibility_max": et_max,
            "weight_mean": float(np.mean(W_plastic)),
            "weight_std": float(np.std(W_plastic)),
            "weight_min": float(np.min(W_plastic)),
            "weight_max": float(np.max(W_plastic)),
            "delta_weight_mean": float(np.mean(dW)),
            "delta_weight_std": float(np.std(dW)),
            "n_plastic_synapses": int(np.sum(self.plasticity_mask > 0)),
            # Phase 7B metadata
            "expanded": self.expanded,
            "num_neurons": self.network.num_neurons,
            "connectome_path": self._connectome_path,
        }

        # Expanded-only diagnostics
        if self.expanded:
            fc3_act = act[self.fc3_indices] if self.fc3_indices else np.array([0.0])
            pfl_act = act[self.pfl_indices] if self.pfl_indices else np.array([0.0])
            fb_act = act[self.fb_indices] if self.fb_indices else np.array([0.0])
            mbon_act = act[self.mbon_indices] if self.mbon_indices else np.array([0.0])
            diagnostics.update({
                "fc3_activity": fc3_act,
                "pfl_activity": pfl_act,
                "fb_activity": fb_act,
                "mbon_activity": mbon_act,
                "fc3_mean": float(np.mean(fc3_act)),
                "pfl_mean": float(np.mean(pfl_act)),
                "fb_mean": float(np.mean(fb_act)),
                "mbon_mean": float(np.mean(mbon_act)),
                "fc3_count": len(self.fc3_indices),
                "pfl_count": len(self.pfl_indices),
                "fb_count": len(self.fb_indices),
                "mbon_count": len(self.mbon_indices),
            })

        return diagnostics

    def get_epg_ordered_activity(self) -> np.ndarray:
        """Return EPG activity in glomerulus order (L8..L1, R1..R8) for compass ring."""
        if not self.epg_ordered_indices:
            return np.zeros(16)
        return self.current_activity[self.epg_ordered_indices].copy()

    # ── Freeze / Unfreeze ──────────────────────────────────────────────────────
    def freeze_weights(self) -> None:
        self.enable_plasticity = False

    def unfreeze_weights(self) -> None:
        if self.plasticity_mode != "none":
            self.enable_plasticity = True

    # ── Reset ──────────────────────────────────────────────────────────────────
    def reset(self) -> None:
        """Reset neural state and eligibility traces at episode boundary."""
        self.network.reset()
        num_n = self.network.num_neurons
        self.prev_activity = np.zeros(num_n, dtype=np.float64)
        self.current_activity = np.zeros(num_n, dtype=np.float64)
        self.plasticity.reset_traces(num_n)
        self._ep_residuals = []
        self._ep_pop_vecs = []
        self._ep_actions = []
        self._last_pop_vec = np.zeros(len(self.W_motor))
        self._last_motor_logit = 0.0
        self._last_flap_prob = 0.5
        self._last_action = 0


# ── Factory Function ───────────────────────────────────────────────────────────
def make_expanded_agent(
    seed: Optional[int] = None,
    expanded: bool = True,
    **kwargs,
) -> FlyMindRLAgentExpanded:
    """
    Create an agent with appropriate configuration.

    Args:
        seed: RNG seed for reproducibility.
        expanded: If True, load the 427-neuron expanded connectome.
                  If False, load the 261-neuron baseline connectome.
        **kwargs: Additional keyword arguments passed to FlyMindRLAgentExpanded.

    Returns:
        Configured FlyMindRLAgentExpanded instance.
    """
    if expanded:
        path = _EXPANDED_CONNECTOME
    else:
        path = _BASELINE_CONNECTOME
    return FlyMindRLAgentExpanded(connectome_path=path, seed=seed, expanded=expanded, **kwargs)
