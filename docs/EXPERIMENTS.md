# FlyMind Experimental Protocols & Metrics

## 1. Experimental Objectives

The goal of the FlyMind experimental suite is to rigorously evaluate whether the **real topological structure** of a biological connectome imparts functional advantages in sensorimotor control and spatial navigation when compared against:
1. **Random agents** (chance performance baseline).
2. **Conventional artificial neural networks** (MLP/RL standard architectures).
3. **Topological ablations** (shuffled weights, randomized adjacency, lesioned cell types).

---

## 2. Standardized Agent Baselines

Every benchmark run compares three distinct agent classes under identical arena conditions and random seeds:

### Baseline A: Random Agent
* Selects valid actions uniformly at random or through brownian motion.
* Establishes the empirical floor for target interception time, path length, and success rate.

### Baseline B: Conventional Neural Network (MLP)
* Standard feedforward network with equivalent parameter count to the biological circuit.
* Evaluated both in naive/uninitialized state and trained using standard reinforcement learning (PPO / DQN / REINFORCE) under the same sensory-motor interface.

### Baseline C: Connectome Agent (FlyMind)
* Uses the real Drosophila Central Complex graph topology extracted from Hemibrain v1.2.1.
* Synaptic weights derive from biological synapse counts and predicted neurotransmitter polarities.
* Tested with both static pre-wired connectivity and biologically plausible local plasticity (Hebbian / STDP / reward modulation).

---

## 3. Quantitative Evaluation Metrics

All experiments record and output the following metrics per episode across $N \ge 100$ independent trials:

1. **Success Rate ($\%$ Reach Target)**:
   $$\text{Success Rate} = \frac{N_{\text{reached}}}{N_{\text{total}}} \times 100$$
2. **Time to Target (Steps)**: Number of simulation timesteps required to reach the target within threshold distance $\epsilon$.
3. **Path Efficiency Ratio ($\eta$)**:
   $$\eta = \frac{D_{\text{euclidean}}(P_{\text{start}}, P_{\text{target}})}{\sum_{t=1}^T \|P_t - P_{t-1}\|_2}$$
   where $\eta \in (0, 1]$, and $\eta = 1$ denotes a direct straight line.
4. **Cumulative Episodic Reward**: Sum of step penalties and arrival rewards.
5. **Heading Stability & Angular Drift**: Mean squared deviation between internal compass representation (E-PG activity bump phase) and ground-truth physical heading $\theta_{\text{fly}}$.
6. **Computational Latency**: Mean wall-clock time per decision step (ms/step).

---

## 4. Ablation & Null Hypothesis Testing

To prove that observed behavioral competence derives specifically from biological wiring rather than arbitrary recurrent dynamics, FlyMind executes four systematic ablations:

| Ablation ID | Manipulation | Biological Question Tested |
|---|---|---|
| **ABL-1: Shuffled Weights** | Topology preserved, synapse counts randomly permuted among existing edges | Does the specific quantitative distribution of synaptic strength matter? |
| **ABL-2: Degree-Preserving Randomization** | Edge re-wiring keeping in- and out-degree distributions constant (Maslov-Sneppen rewiring) | Does the specific micro-circuit loop motif (e.g. ring attractor) provide navigation advantages over random graphs with equal degree? |
| **ABL-3: Random Erdős–Rényi Graph** | Completely random directed graph with matching $N$ (nodes) and $M$ (edges) | Is biological connectivity superior to arbitrary recurrent connectivity? |
| **ABL-4: Cell-Type Lesions** | Selective zeroing of specific neuron populations (e.g. silencing `Delta7` inhibitory neurons or `P-EN` integrator neurons) | Does eliminating known biological functional groups produce the behavioral deficits predicted by neuroscience literature? |

---

## 5. Result Storage & Artifact Schema

Every experiment automatically creates a versioned directory in `results/`:
```
results/
└── exp_YYYYMMDD_HHMMSS_run_name/
    ├── config.json          # Complete serializable experiment configuration
    ├── metrics.csv          # Episode-by-episode quantitative metrics
    ├── trajectory_log.h5    # Fly trajectories and neural activation traces (optional)
    ├── plots/
    │   ├── learning_curve.png
    │   ├── path_trajectories.png
    │   └── neural_bump_tracking.png
    └── summary.txt          # Statistical summary and significance tests
```

---

## 6. Phase 6 Flappy Bird Visual & Telemetry Suite

Phase 6 extends the navigation benchmark into a 2D side-scrolling obstacle environment:
* **Phase 6A**: Initial Flappy Bird environment setup, 9-channel visual sensor, and PEN_a asymmetry motor decoder.
* **Phase 6B**: Connectome motor readout using PEG premotor gating.
* **Phase 6C**: Real-time visual gameplay engine, 16-glomeruli polar EPG ring dashboard, side-by-side comparison mode, compressed replay serialization (`.npz`), video export (`MP4`/`PNG`), zero-divergence passivity validation, and composite 6-panel publication figure generation.

