# FlyMind Phase 6C: Visual Gameplay & Replay System

**Script**: `experiments/phase6c_flappy_visual.py`  
**Engine**: `pygame-ce` (Community Edition) + `imageio`  
**Connectome**: 261 neurons, 19,969 synapses (Janelia Hemibrain v1.2.1)  
**Environment**: Flappy-Bird-style 2D side-scrolling sandbox (deterministic physics)  

---

## 1. Executive Summary

Phase 6C fulfills the critical requirement of **watchable, verifiable visual gameplay** for FlyMind. Rather than relying solely on scalar metrics and end-of-run summaries, Phase 6C enables researchers to observe the fly brain navigate 2D flight dynamics in real time, record episodes to compressed replays, export high-definition videos, inspect internal compass and premotor representations, and validate experimental reproducibility with a zero-divergence passivity guarantee.

---

## 2. Core Capabilities & Operating Modes

| Mode | Flag | Description | Typical Use Case |
|---|---|---|---|
| **Live Interactive** | *(default)* | Live 30 FPS Pygame window with real-time HUD and neural overlay panel | Human observation, real-time debugging |
| **Neural Replay** | `--neural-replay` | Expanded dashboard featuring a 16-glomeruli polar EPG compass ring | Biological inspection of ring attractor heading |
| **Side-by-Side Compare** | `--compare` | Two synchronized side-by-side viewports with shared seeds and pipe layouts | Direct controller comparison (e.g. Hand vs FlyMind) |
| **Video & Frame Capture** | `--record` | Automated export to MP4 (via libx264) or PNG frame sequence | Presentation videos, publications |
| **Passivity Validator** | `--validate` | Bit-for-bit trajectory verification comparing instrumented vs pure runs | Rigorous reproducibility assurance |
| **Demo Suite** | `--demo` | Automated headless run of all 4 benchmark controllers with replays & figure | CI/CD testing, benchmark verification |
| **Publication Figure** | `--figure` | Generates a 6-panel composite research figure in PNG and vector PDF format | Paper submission, documentation figures |

---

## 3. Interactive Keyboard Controls

When running in live mode (non-headless), the following interactive controls are available:

- `SPACE`: **Pause / Resume** simulation. Pauses physics and agent stepping while keeping the window responsive.
- `R`: **Restart Episode**. Resets the environment and agent state to test a fresh trial.
- `ESC` or Window Close: **Exit**. Gracefully closes display surfaces and flushes open recordings.

> [!NOTE]
> **No Manual Flap Key**: Consistent with the autonomous nature of FlyMind agents, manual control keys are strictly disabled. The agent autonomously drives flight decisions via sensory-motor connectome dynamics.

---

## 4. The Passivity Guarantee

A fundamental scientific concern when introducing visualization tools into closed-loop neural simulations is that the observation layer might inadvertently modify internal agent states, alter random seeds, or induce side-effects in weight tensors.

Phase 6C implements a **Zero-Divergence Passivity Guarantee**:

1. **Post-Act Observation Only**: The diagnostic reader `extract_diagnostics(agent)` is strictly called **after** `agent.act(state)` has concluded its execution cycle.
2. **Read-Only Access**: Diagnostics are extracted by copying primitive float values and numpy arrays without re-triggering network forward passes or updating synaptic plasticity eligibility traces.
3. **Automated Verification**: Running `python experiments/phase6c_flappy_visual.py --validate` tests multiple random seeds across hundreds of steps comparing:
   - Run A: Pure headless baseline without visualization or diagnostic calls.
   - Run B: Fully instrumented visual rendering pipeline.
   - **Result**: Exactly zero divergence ($\Delta = 0.000000$) in all actions, vertical positions, and cumulative scores across all seeds.

---

## 5. Architectural Data Flow

```
+-------------------------------------------------------------+
|                      FlappyEnvironment                      |
|         (bird_x, bird_y, bird_vy, pipes, score, alive)      |
+------------------------------+------------------------------+
                               |
                        FlappyState
                               |
                               v
+-------------------------------------------------------------+
|               ConnectomeMotorReadoutAgent                   |
|  1. FlappyVisualSensor (9 channels)                         |
|  2. EPG compass ring excitation (16 glomeruli)              |
|  3. 261-neuron recurrent connectome dynamics                |
|  4. PEG premotor activity gating                            |
|  5. Motor readout: Gain * (Sensor_V * PEG_Gate) + Bias      |
|  6. Decision: action in {0, 1}                              |
+------------------------------+------------------------------+
                               |
                       (Action & State)
                               |
                               v
+-------------------------------------------------------------+
|                Phase 6C Visualization Layer                 |
|  - GameRenderer: Sky gradient, scrolling pipes, bird wing   |
|  - NeuralOverlayPanel: 3x3 sensor view, readout gauges      |
|  - EPGRingRenderer: 16-glomeruli polar heading ring         |
|  - ActionTimeline: Rolling 60-step action/prob strip        |
|  - VideoExporter: MP4 / PNG sequence writer                 |
|  - EpisodeReplay: Compressed .npz serialization             |
+-------------------------------------------------------------+
```

---

## 6. Output Artifacts & Directories

All visual outputs and replays are saved under `results/phase6c/`:

```
results/phase6c/
├── replays/
│   ├── replay_flymind_seed42.npz
│   ├── replay_hand_seed42.npz
│   ├── replay_fixed_seed42.npz
│   └── replay_random_seed42.npz
├── videos/
│   └── gameplay_YYYYMMDD_HHMMSS.mp4
├── frames/
│   └── gameplay_YYYYMMDD_HHMMSS/
│       ├── frame_00000.png
│       └── ...
└── figures/
    ├── phase6c_gameplay_figure.png
    └── phase6c_gameplay_figure.pdf
```

### Replay Data Format (`.npz`)

Each saved episode replay contains compressed arrays:
- `bird_y`, `bird_vy`: Float32 arrays of bird coordinates over time.
- `actions`: Int8 array of decisions (`0` = Glide, `1` = Flap).
- `flap_probs`: Float32 sigmoid output $P(\text{flap})$.
- `scores`: Int32 cumulative score over time.
- `sensor_vertical`: Float32 vertical gap alignment metric.
- `peg_mean`, `peg_gate`: Float32 premotor gating signals.
- `motor_score`: Float32 raw logit input to the motor sigmoid.
- `sensor_activations`: Shape `(T, 9)` matrix of visual sensor activations.
- `epg_activity`: Shape `(T, 16)` matrix of compass ring glomeruli.

---

## 7. CLI Usage Examples

```bash
# 1. Live interactive window with FlyMind controller
python experiments/phase6c_flappy_visual.py

# 2. Expanded neural dashboard with polar EPG ring
python experiments/phase6c_flappy_visual.py --neural-replay

# 3. Side-by-side comparison: Hand-designed vs FlyMind
python experiments/phase6c_flappy_visual.py --compare

# 4. Record gameplay video to MP4
python experiments/phase6c_flappy_visual.py --record --episodes 3

# 5. Run passivity guarantee test
python experiments/phase6c_flappy_visual.py --validate

# 6. Run headless demo for all 4 controllers with replays
python experiments/phase6c_flappy_visual.py --demo

# 7. Generate publication-quality 6-panel research figure
python experiments/phase6c_flappy_visual.py --figure
```
