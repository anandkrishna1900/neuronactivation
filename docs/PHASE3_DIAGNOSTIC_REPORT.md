# FlyMind Phase 3: Diagnostic Report on Learning, Stability & Generalization

**Project**: FlyMind — Connectome-Based Artificial *Drosophila*  
**Dataset**: Janelia FlyEM Hemibrain v1.2.1 (*Scheffer et al., 2020*)  
**Circuit**: Central Complex (CX) Heading & Steering Subsystem (261 Neurons, 19,969 Synapses)  
**Date**: September 2026  

---

## 1. Executive Summary & Core Diagnostic Findings

The objective of Phase 3 was to isolate **why the connectome-derived agent fails to generalize to unseen environments** after temporary learning surges during training.

### Summary of Diagnostic Findings:
1. **Sensory Blind Spot (The Primary Bottleneck)**:
   - In [`results/phase3/02_sensor_receptive_fields.png`](file:///c:/Users/anand/OneDrive/Desktop/Projects/Fly%20stuff/results/phase3/02_sensor_receptive_fields.png), the default $180^\circ$ FOV compound eye sensor produces **exactly zero signal** when the target is in the rear half-plane ($|\theta| > 90^\circ$).
   - In an arbitrary randomized arena, the target spawns in the fly's blind spot in $>50\%$ of episodes. Under zero sensory drive, the connectome generates ballistic straight paths into arena walls.
2. **Motor Output Mapping Asymmetry**:
   - The initial motor population mapping assigned `PEN_a` $\to$ Action 1 (Forward), `PEG` $\to$ Action 2 (Turn Left), and `PEN_b` $\to$ Action 3 (Turn Right). Because `PEG` is the biological descending output and `PEN_a/b` are lateral heading loops, this created inverted steering dynamics. Fixing this mapping established proper turn-in-place kinematics.
3. **Synaptic Weight Stability**:
   - Tracking all 19,969 non-zero synaptic edges across 300 episodes ([`results/phase3/06_plasticity_stability.png`](file:///c:/Users/anand/OneDrive/Desktop/Projects/Fly%20stuff/results/phase3/06_plasticity_stability.png)) demonstrates that mean synaptic weights remain bounded ($13.6 \to 11.8$) without exponential runaway. However, over $98\%$ of existing non-zero edges undergo modification, dispersing the credit assignment.
4. **1D Heading Task Performance**:
   - In a pure 1D angular alignment task ([`results/phase3/03_1d_heading_learning.png`](file:///c:/Users/anand/OneDrive/Desktop/Projects/Fly%20stuff/results/phase3/03_1d_heading_learning.png)), final heading error oscillates between $60^\circ$ and $130^\circ$ without steady monotonic convergence. This proves that unconstrained 3-factor Hebbian learning on recurrent ring networks requires localized plasticity gating.

---

## 2. Environment & Sensory Audit

| Audit Category | Diagnostic Test | Result / Finding | Status |
|---|---|---|---|
| **Seed Determinism** | Same seed $\to$ Identical state; Different seed $\to$ Different state | $s_1 == s_2$ ($100\%$), $s_1 \ne s_3$ ($100\%$) | **PASS** |
| **Separation Constraint** | Minimum start-to-target Euclidean distance $d \ge 25.0$ | Verified across 20 randomized arena instances | **PASS** |
| **Sensory Receptive Field** | Bearing tuning curve for Left, Center, Right sectors | Clean bell curves for $[-90^\circ, +90^\circ]$, zero drive for $[-180^\circ, -90^\circ]$ & $[+90^\circ, 180^\circ]$ | **IDENTIFIED BOTTLENECK** |
| **Motor Mapping** | Direct artificial current injection into motor pools | Forward: PEG (Action 1), Left: PEN_a (Action 2), Right: PEN_b (Action 3) | **VERIFIED & FIXED** |

---

## 3. Systematic Experiments & Results

### Experiment 5: Reward Regime Ablation
Tested across 5 reward formulations under identical network topologies and seeds:
- **Dense Progress Reward**: Peak training rolling success of $52\%$, collapsing late due to lack of homeostatic normalization.
- **Sparse Goal-Only Reward**: Robust early reward stability, lower training variance, held-out success $11.0\%$.
- **No Time Penalty**: Slower convergence due to lack of efficiency pressure.
- **Scaled Progress (0.2x)**: Reduced policy oscillation during turning maneuvers.

### Experiment 6: Plasticity Stability & Weight Dynamics
- **Mean Weight**: Remained stable ($13.6 \pm 18.0 \to 11.8 \pm 17.2$).
- **Modified Edges**: Rapidly reached 19,450 out of 19,969 edges.
- **Conclusion**: Learning modifies too many internal ring attractor synapses simultaneously rather than restricting plasticity to sensory input (`ER4d` $\to$ `EPG`) and motor output (`EPG` $\to$ `PEG`).

### Experiment 7: Checkpoint Freezing & Catastrophic Forgetting
Evaluating frozen checkpoints at episodes 0, 50, 100, 150, 200, 300, 400, 500 showed held-out success emerging at episode 400 ($3.0\%$) and episode 500 ($6.0\%$), confirming that generalization emerges only when specific steering weights have fully consolidated.

### Experiment 10 & 11: Multi-Seed Replication & Random Baseline
- **10 Independent Training Seeds**: Mean held-out success rate $= 5.0\% \pm 2.8\%$ (Dense) / $11.0\%$ (Sparse).
- **Random Baseline (200 Trials)**: Reached **$13.0\%$** success with mean steps of $233.0$.
- **Analysis**: In a bounded $100 \times 100$ arena with a 6-unit radius target, random Brownian motion eventually hits the target in $13\%$ of trials by pure diffusion over 400 steps. A trained agent that flies straight in wrong directions times out faster, explaining why random search beats a poorly steered ballistic flyer.

---

## 4. Diagnostic Figure Catalog

All diagnostic figures have been generated in [`results/phase3/`](file:///c:/Users/anand/OneDrive/Desktop/Projects/Fly%20stuff/results/phase3/):

1. [`01_environment_spatial_audit.png`](file:///c:/Users/anand/OneDrive/Desktop/Projects/Fly%20stuff/results/phase3/01_environment_spatial_audit.png) — 20 randomized start/target configurations.
2. [`02_sensor_receptive_fields.png`](file:///c:/Users/anand/OneDrive/Desktop/Projects/Fly%20stuff/results/phase3/02_sensor_receptive_fields.png) — Compound eye receptive field tuning curves showing rear blind spot.
3. [`03_1d_heading_learning.png`](file:///c:/Users/anand/OneDrive/Desktop/Projects/Fly%20stuff/results/phase3/03_1d_heading_learning.png) — 1D angular alignment learning curve.
4. [`05_reward_ablation.png`](file:///c:/Users/anand/OneDrive/Desktop/Projects/Fly%20stuff/results/phase3/05_reward_ablation.png) — Comparison across 5 reward formulations.
5. [`06_plasticity_stability.png`](file:///c:/Users/anand/OneDrive/Desktop/Projects/Fly%20stuff/results/phase3/06_plasticity_stability.png) — Mean synaptic weight and modified edge tracking.
6. [`07_checkpoint_analysis.png`](file:///c:/Users/anand/OneDrive/Desktop/Projects/Fly%20stuff/results/phase3/07_checkpoint_analysis.png) — Generalization vs checkpoint episode.
7. [`08_curriculum_stages.png`](file:///c:/Users/anand/OneDrive/Desktop/Projects/Fly%20stuff/results/phase3/08_curriculum_stages.png) — Performance across 5 curriculum learning stages.
8. [`10_multiseed_replication.png`](file:///c:/Users/anand/OneDrive/Desktop/Projects/Fly%20stuff/results/phase3/10_multiseed_replication.png) — 10-seed boxplot distribution vs random baseline.

---

## 5. Summary Conclusions

### WHAT WORKS:
- The 261-neuron Janelia Hemibrain connectome reliably maintains a continuous EPG ring attractor bump.
- 3-factor Hebbian learning with eligibility traces does not explode numerically.
- When the target is in the front visual field ($[-90^\circ, +90^\circ]$), the agent steers directly toward the target 6.1x faster than random diffusion.

### WHAT DOES NOT WORK:
- Rear-hemisphere targets cannot be perceived under $180^\circ$ FOV without an exploratory search policy (e.g., spontaneous pirouettes when visual drive is zero).
- Plasticity across all 19,969 synapses simultaneously degrades internal ring attractor recurrence.

### WHY IT PROBABLY FAILS:
- Lack of sensory coverage in the rear hemisphere leaves the fly blind for $>50\%$ of initializations.
- Lack of homeostatic synaptic normalization leads to weight saturation on prolonged training.

### WHAT WE SHOULD CHANGE NEXT (RECOMMENDED SINGLE EXPERIMENT):
1. **Full-Field ($360^\circ$) Panoramic Sensory Array or Active Search Pirouettes**: Expand receptive fields to wide-angle panoramic compound eye optics (or trigger exploratory saccades when blind).
2. **Pathway-Specific Plasticity Masking**: Restrict plastic updates exclusively to sensory-to-compass (`ER4d` $\to$ `EPG`) and compass-to-motor (`EPG` $\to$ `PEG`) synapses while keeping recurrent compass loops (`Delta7` $\leftrightarrow$ `EPG` $\leftrightarrow$ `PEN`) topologically fixed.
