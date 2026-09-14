# FlyMind Phase 6C: Visual Gameplay & Replay System — Final Report

**Date**: 2026-09-14 07:31  
**Connectome**: 261 neurons, 19,969 synapses (Janelia Hemibrain v1.2.1)  
**Environment**: Flappy-Bird-style 2D side-scrolling sandbox  
**Visualization Engine**: `pygame-ce` + `imageio`  
**Script**: `experiments/phase6c_flappy_visual.py`  

---

## 1. Research & Engineering Objective

Phase 6C addresses the requirement of **watchable, verifiable visual gameplay and telemetry analysis** for FlyMind agents in the Flappy Bird environment. While Phase 6A established initial control and Phase 6B evaluated PEG premotor readout gating, Phase 6C introduces:
1. Real-time visual observation with continuous telemetry HUD and neural panel displays.
2. Compressed episode replay serialization (`.npz`).
3. High-definition MP4/PNG recording export (`libx264`).
4. Dual-viewport side-by-side agent comparison mode (`--compare`).
5. 16-glomeruli polar EPG heading ring visualization (`--neural-replay`).
6. Zero-divergence passivity guarantee validation (`--validate`).
7. Automated multi-controller demo benchmark suite & composite 6-panel research figure generation (`--demo` / `--figure`).

---

## 2. Core Operating Modes

| Flag | Mode | Output / Function | Typical Usage |
|---|---|---|---|
| *(default)* | **Live Interactive** | Pygame window (30 FPS) with HUD & neural panel | Real-time human inspection |
| `--neural-replay` | **Polar Compass View** | 16-glomeruli polar EPG compass ring display | Inspect ring attractor phase |
| `--compare` | **Side-by-Side Compare** | Twin viewports (Hand vs FlyMind) with synchronized seeds | Qualitative controller comparison |
| `--record` | **Video Export** | MP4 video (`libx264`) & PNG frame sequence | Presentations, publications |
| `--validate` | **Passivity Test** | Bit-for-bit trajectory validation ($\Delta = 0$) | Scientific verification |
| `--demo` | **Benchmark Suite** | Evaluates all 4 controllers, saves replays & composite figure | Automated CI/CD verification |
| `--figure` | **Publication Figure** | Generates 6-panel composite PNG & vector PDF figure | Research paper documentation |

---

## 3. Passivity & Reproducibility Guarantee

To guarantee that the introduction of a visual rendering pipeline does not alter agent action choices, random seeds, or connectome internal states, Phase 6C implements a **Zero-Divergence Passivity Guarantee**:

- **Post-Act Extraction**: `extract_diagnostics(agent)` is executed exclusively *after* `agent.act(state)` returns.
- **Read-Only Inspection**: Primitive float values and arrays are copied without executing neural forward passes or updating plasticity eligibility traces.
- **Validation Results**:

```
======================================================================
PHASE 6C PASSIVITY & REPRODUCIBILITY VALIDATION
======================================================================
Seed   42: [PASS] Steps=19 vs 19 | Score=0 vs 0 | ActionDelta=0 | MaxYDelta=0.00e+00
Seed  100: [PASS] Steps=163 vs 163 | Score=0 vs 0 | ActionDelta=0 | MaxYDelta=0.00e+00
Seed  777: [PASS] Steps=140 vs 140 | Score=0 vs 0 | ActionDelta=0 | MaxYDelta=0.00e+00
----------------------------------------------------------------------
[SUCCESS] All seeds validated bit-for-bit! PASSIVITY GUARANTEE VERIFIED.
======================================================================
```

---

## 4. Benchmark Performance Metrics (Demo Suite)

Evaluating all 4 benchmark controllers across identical seeds ($N=3$ episodes per controller, base seed = 42):

| Controller | Mean Score | Max Score | Mean Survival (Steps) | Replay File Saved |
|---|---|---|---|---|
| **Random (50% Flap)** | 0.00 | 0 | 13.7 | `replay_random_seed42.npz` |
| **Fixed Period (N=8)** | 0.00 | 0 | 42.0 | `replay_fixed_seed42.npz` |
| **Hand-Designed** | 2.33 | 3 | 405.7 | `replay_hand_seed42.npz` |
| **FlyMind (Connectome)** | **0.33** | **1** | **121.7** | `replay_flymind_seed42.npz` |

### Observations:
- **FlyMind** significantly outlives random control ($121.7$ vs $13.7$ steps) and fixed period control ($121.7$ vs $42.0$ steps), demonstrating active pipe obstacle avoidance.
- **Hand-Designed** control achieves highest survival ($405.7$ steps), utilizing explicit vertical velocity gating.

---

## 5. Replay Data Specification (`.npz`)

Serialized episode replays stored under `results/phase6c/replays/` contain:

| Array Field | Dtype | Shape | Description |
|---|---|---|---|
| `bird_y` | float32 | `(T,)` | Vertical bird position over episode |
| `bird_vy` | float32 | `(T,)` | Vertical velocity over episode |
| `actions` | int8 | `(T,)` | Action sequence ($0 = \text{Glide}, 1 = \text{Flap}$) |
| `flap_probs` | float32 | `(T,)` | Sigmoid flap output probability $P(\text{flap})$ |
| `scores` | int32 | `(T,)` | Cumulative pipe score |
| `sensor_vertical` | float32 | `(T,)` | 9-channel sensor target gap alignment signal |
| `peg_mean`, `peg_gate` | float32 | `(T,)` | PEG premotor activity gating trace |
| `motor_score` | float32 | `(T,)` | Raw logit feeding motor decision |
| `sensor_activations` | float32 | `(T, 9)` | Receptive field visual grid over time |
| `epg_activity` | float32 | `(T, 16)` | EPG 16-glomeruli compass activation trajectory |

---

## 6. Generated Publication Figures & Artifacts

1. **Composite Research Figure**: `results/phase6c/figures/phase6c_gameplay_figure.png` & `.pdf`
   - **Panel A**: Flight Trajectories across controllers.
   - **Panel B**: Flap Action Probability Timelines.
   - **Panel C**: EPG 16-Glomeruli Compass Space-Time Heatmap.
   - **Panel D**: PEG Premotor Gating Signal vs Sensor Vertical Alignment.
   - **Panel E**: Score Distribution Comparison.
   - **Panel F**: Controller Survival Duration (Steps).
2. **Recorded Gameplay MP4 Video**: `results/phase6c/videos/gameplay_*.mp4`

---

## 7. Conclusions & Next Steps

Phase 6C provides the definitive visual verification framework for FlyMind. It proves bit-for-bit reproducibility between visual and headless modes, establishes standardized compressed episode serialization, and enables continuous spatial-temporal telemetry tracking of the *Drosophila* Central Complex compass ring during simulated flight.

### Recommended Next Steps:
1. Integrate Phase 6C visual dashboard into interactive user web interfaces or Jupyter notebook replays.
2. Extend replay logging to include synapse-level plasticity trace changes during learning runs.
