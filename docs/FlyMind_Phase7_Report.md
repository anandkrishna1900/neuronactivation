# FlyMind Phase 7: Reinforcement Learning & Visual Mastery — Final Report

**Date**: 2026-09-14  
**Connectome**: 261 neurons, 19,969 synapses (Janelia Hemibrain v1.2.1 Central Complex)  
**Environment**: Flappy-Bird-style 2D side-scrolling aerodynamic sandbox  
**Core Innovation**: Connectome-Driven Policy with Reward-Modulated Local Plasticity (No Backpropagation)  

---

## 1. Executive Summary & Objective

In Phase 6B, the agent utilized a premotor readout gate where the sensory error (`sensor_vertical`) was multiplied directly into the flap decision. While effective as a biological proof-of-concept, it relied on a sensory bypass shortcut:
$$\text{motor\_score} = \text{gain} \times (\text{sensor\_vertical} \times \text{peg\_gate}) + \text{bias}$$

**Phase 7 eliminates this sensory shortcut entirely.** The Phase 7 agent (`FlyMindRLAgent`) routes all sensory information exclusively into the Central Complex neuropils (EB/PB/FB). The action decision is derived **strictly from the population firing rates of premotor and descending neuropil neurons** (PEG, PFNd, PFNv). Synaptic adaptation occurs locally via a biologically plausible **three-factor reward-modulated Hebbian learning rule** with eligibility traces, coupled with a minimal descending motor readout layer updated via policy gradients.

### Key Architectural Pillars:
1. **Zero Sensory Shortcut**: The decision to flap is driven $100\%$ by connectome population states ($W_{\text{motor}} \cdot [\overline{\text{PEG}}, \overline{\text{PFNd}}, \overline{\text{PFNv}}]^T + b$).
2. **Biological Plasticity Mask**: Plasticity is strictly confined to 1,888 biologically permissible synapses identified in `docs/PLASTICITY_MASK.md` (EPG $\to$ PEG, PFN $\to$ hDelta, etc.). Fixed connectome backbones remain locked.
3. **Reward Shaping & Temporal Credit Assignment**: FlappyRewardShaper introduces survival incentives ($+0.001$/step), pipe clearance rewards ($+1.0$), and collision penalties ($-1.0$), combined with exponential eligibility trace decays ($\gamma = 0.95$).
4. **Curriculum Learning Pipeline**: 5-stage progressive curriculum from wide gaps ($160\text{px}$) and slow speeds down to standard and randomized challenging flight regimes.
5. **Real-Time Visual Learning Telemetry**: A high-framerate Pygame dashboard featuring live gameplay, 16-channel polar EPG compass, motor readout weights, population activity meters, plasticity weight delta histograms, and rolling learning curves.
6. **Zero-Divergence Passivity Guarantee**: Verified bit-for-bit equivalence ($\Delta = 0.0$) between headless simulation and visual rendering.

---

## 2. Connectome Architecture & Motor Readout Pathway

### Central Complex Sub-Populations (261 neurons total):
- **EPG (Compass / Heading Ring)**: 46 neurons encoding azimuthal orientation in the Ellipsoid Body.
- **PEG (Premotor Steering / Gating)**: 18 neurons projecting from PB to Gall/LAL.
- **PFNd (Noduli / Directional Steering)**: 40 neurons relaying vector flow and noduli directional signals.
- **PFNv (Ventral Noduli / Pitch / Thrust)**: 20 neurons mediating ventral pitch control.
- **Delta / hDelta / FC / FB**: 137 tangential and columnar neurons providing internal recurrence and coordinate transformation.

### Sensory Ingestion:
Visual sensory inputs from the 9-channel raycaster (`FlappyVisualSensor`) project into the wedge neurons and PB columnar inputs:
- Gap vertical offset $\Delta y \to$ EPG ring injection.
- Horizontal distance $d_x \to$ PB/FB tangential excitation.
- Vertical velocity $v_y \to$ Noduli/PFN dynamic modulation.

### Motor Readout:
$$\pi(\text{flap} \mid s) = \sigma\left(\frac{W_{\text{motor}} \cdot \mathbf{a}_{\text{motor}} + b}{\tau}\right)$$
where:
$$\mathbf{a}_{\text{motor}} = \begin{bmatrix} \overline{\text{PEG}} \\ \overline{\text{PFNd}} \\ \overline{\text{PFNv}} \end{bmatrix}, \quad W_{\text{motor}} \in \mathbb{R}^3$$
- $\overline{\text{PEG}}$: Mean firing rate across all 18 PEG neurons.
- $\overline{\text{PFNd}}$: Mean firing rate across all 40 PFNd neurons.
- $\overline{\text{PFNv}}$: Mean firing rate across all 20 PFNv neurons.
- $\tau$: Softmax/sigmoid temperature parameter (default $1.0$).

---

## 3. Local Biological Plasticity (3-Factor Rule)

Connectome synapses update via dopamine-modulated Hebbian plasticity:
$$\Delta W_{ij} = \eta \cdot M_{ij} \cdot R(t) \cdot e_{ij}(t)$$
where:
- $\eta$: Local learning rate ($\sim 10^{-4}$).
- $M_{ij} \in \{0, 1\}$: Binary plasticity mask ($1,888$ plastic synapses out of $19,969$).
- $R(t)$: Global neuromodulatory reward/dopamine signal (computed by `FlappyRewardShaper`).
- $e_{ij}(t)$: Synaptic eligibility trace updated at each step:
  $$e_{ij}(t) = \lambda \cdot e_{ij}(t-1) + r_i^{\text{pre}}(t) \cdot r_j^{\text{post}}(t)$$
- Dale's law and non-negativity are strictly enforced: all excitatory synapses remain $W \ge 0$, and inhibitory connections remain $W \le 0$.

---

## 4. Passivity & Integrity Verification

To verify that the Phase 7 agent strictly respects biology and introduces no simulation divergence:
1. **Biological Integrity**: Verified via `agent.assert_biological_integrity()`:
   - Synapse count: 19,969 intact.
   - Plastic synapses: Exactly 1,888 masked synapses.
   - Dale's principle: Sign preservation across all plastic synapses.
2. **Passivity Guarantee**: Run with `python experiments/phase7_flappy_rl.py --validate`:
   ```
   ======================================================================
   PHASE 7 PASSIVITY VALIDATION
   ======================================================================
   Seed   42: [PASS] Steps=14 vs 14 | ActionDelta=0 | MaxYDelta=0.00e+00
   Seed  100: [PASS] Steps=12 vs 12 | ActionDelta=0 | MaxYDelta=0.00e+00
   Seed  777: [PASS] Steps=14 vs 14 | ActionDelta=0 | MaxYDelta=0.00e+00
   ----------------------------------------------------------------------
   [SUCCESS] All seeds validated bit-for-bit! PASSIVITY GUARANTEE VERIFIED.
   ======================================================================
   ```

---

## 5. Available Scripts & User Execution Guide

All components are fully implemented, compiled, and verified. You can run any of the following commands in your shell to observe the execution and visual outputs directly:

### 1. Live Interactive Visual RL Training Dashboard
Opens a $1300 \times 900$ high-resolution dashboard with real-time gameplay, polar EPG compass, population meters, motor weights, and live reward plots:
```bash
python experiments/phase7_flappy_rl_visual.py --render-training --episodes 1000 --fps 60
```
- Add `--curriculum` to train with adaptive curriculum progression:
  ```bash
  python experiments/phase7_flappy_rl_visual.py --render-training --curriculum --episodes 1000 --fps 60
  ```

### 2. High-Speed Headless Training
For long-horizon multi-thousand episode training with periodic checkpoints and evaluations:
```bash
python experiments/phase7_flappy_rl.py --episodes 5000 --seed 42 --checkpoint-interval 250
```
- With curriculum:
  ```bash
  python experiments/phase7_flappy_rl.py --episodes 5000 --curriculum --seed 42
  ```
- To resume from a checkpoint:
  ```bash
  python experiments/phase7_flappy_rl.py --resume results/phase7/checkpoints/<checkpoint_name>.npz
  ```

### 3. Visual Checkpoint Evaluation & Twin Comparison
To evaluate an existing checkpoint visually:
```bash
python experiments/phase7_flappy_rl_visual.py --render-eval --checkpoint results/phase7/checkpoints/<checkpoint_name>.npz
```
To run side-by-side comparison between FlyMind RL and the Hand-Designed benchmark:
```bash
python experiments/phase7_flappy_rl_visual.py --compare
```

### 4. Frozen Evaluation & Generalization Suite
Runs a standardized 50-seed frozen evaluation across diverse wind, gap, and speed conditions:
```bash
python experiments/phase7_evaluate.py --checkpoint results/phase7/checkpoints/<checkpoint_name>.npz --generalization
```
To compare all controllers (FlyMind, Phase 6B, Hand, Random, Fixed):
```bash
python experiments/phase7_evaluate.py --compare-all
```

### 5. Biological Ablation Battery
To evaluate the necessity of specific Central Complex structures:
```bash
# Lesion EPG compass ring
python experiments/phase7_flappy_rl.py --ablation no_epg --episodes 1000

# Disable local plasticity (frozen weights)
python experiments/phase7_flappy_rl.py --ablation no_plasticity --episodes 1000

# Rewired null model (scrambled connectome topology)
python experiments/phase7_flappy_rl.py --ablation rewired --episodes 1000

# Run full automated ablation suite
python experiments/phase7_flappy_rl.py --run-ablations --episodes 500
```
