# FlyMind Phase 4: Implementation Plan & Architectural Audit

**Project**: FlyMind — Connectome-Based Artificial *Drosophila*  
**Dataset**: Janelia FlyEM Hemibrain v1.2.1 (*Scheffer et al., 2020*)  
**Circuit**: Central Complex (CX) Heading & Steering Subsystem (261 Neurons, 19,969 Synapses)  
**Date**: September 2026  

---

## 1. Executive Overview & Phase 3 Audit

In Phase 3, systematic diagnostics isolated the two primary bottlenecks responsible for poor generalization:
1. **Sensory Blind Spot**: The $180^\circ$ FOV compound eye sensor produces zero signal when targets spawn in the rear hemisphere ($|\theta| > 90^\circ$). Under zero drive, the connectome fly moves ballistically straight into arena walls.
2. **Plasticity Dispersion Across Recurrent Microcircuits**: Unconstrained Hebbian learning was modifying $>98\%$ of all 19,969 synapses, degrading the intrinsic compass dynamics of the `Delta7` $\leftrightarrow$ `EPG` $\leftrightarrow$ `PEN` recurrent ring.

### Target Fixes in Phase 4:
- **Fix #1: Panoramic 360° Vision & Active Saccades**:
  - Implement a 12-ommatidia circular panoramic sensor (azimuthal receptive fields at $0^\circ, 30^\circ, \dots, 330^\circ$) eliminating rear blind spots.
  - Test active saccadic exploration when sensory drive falls below baseline.
- **Fix #2: Pathway-Specific Synaptic Masking**:
  - Confine 3-factor Hebbian plasticity strictly to `ER4d -> EPG` (1,148 synapses) and `EPG -> PEG` (280 synapses), representing **7.15% (1,428 edges)** of the network.
  - Freeze all other 18,541 internal recurrent compass and lateral inhibition edges.
- **Fix #3: Behavioral Variability & Non-Scripted Control**:
  - Implement a multi-target sandbox arena with quantitative behavioral metrics (trajectory diversity, action entropy, dwell time, turning distributions).

---

## 2. File Modification & Architectural Matrix

| Module / File Path | Status | Planned Modifications & Responsibilities |
|---|---|---|
| `src/flymind/environment/sensors.py` | **MODIFY** | Add `PanoramicCompoundEyeSensor` (12 receptive fields across 360°) and `ActiveExplorationSensor` (with exploratory saccades). |
| `src/flymind/brain/plasticity.py` | **MODIFY** | Implement `PathwaySpecificPlasticity` with runtime validation that frozen synapses strictly undergo $\Delta W_{ij} \equiv 0$. |
| `src/flymind/agent/plastic_cx.py` | **MODIFY** | Support 360° panoramic input mappings into ER4 ring neurons and configurable plasticity masks. |
| `src/flymind/utils/behavioral_metrics.py` | **NEW** | Quantitative metrics: trajectory entropy, turning angle distributions, dwell times, behavioral consistency. |
| `docs/PLASTICITY_MASK.md` | **NEW** | Formal anatomical specification of the 1,428 plastic edges vs 18,541 frozen edges. |
| `experiments/phase4_suite.py` | **NEW** | Complete 8-condition ablation suite (A..H) across 10 independent seeds. |
| `experiments/phase4_behavioral_variability.py` | **NEW** | Multi-target choice sandbox measuring structured adaptive variability. |
| `docs/FlyMind_Phase4_Report.md` | **NEW** | Comprehensive final Phase 4 scientific report. |

---

## 3. Systematic Phase 4 Experiment Matrix

1. **Experiment 1: Sensor Receptive Field Audit & Calibration**:
   - 13 target bearings ($-180^\circ$ to $+180^\circ$ in $30^\circ$ increments) verifying full 360° response symmetry without privileged coordinate leakage.
2. **Experiment 2: Pathway-Specific Plasticity vs Global Stability**:
   - Tracking synaptic weight drift, bump coherence, and edge modification rates across 300 episodes.
3. **Experiment 3: 1D Heading Alignment Benchmark**:
   - 10-seed alignment benchmark comparing No Plasticity, Global Plasticity, and Pathway-Specific Plasticity.
4. **Experiment 4: Full 2D Navigation Benchmark (8-Condition Ablation A..H)**:
   - Exp A: 180° + Global Plasticity
   - Exp B: 360° + Global Plasticity
   - Exp C: 180° + Pathway Plasticity
   - Exp D: 360° + Pathway Plasticity
   - Exp E: 180° + Saccades + Pathway Plasticity
   - Exp F: 360° + Unplastic
   - Exp G: Rewired Connectome + Pathway Plasticity
   - Exp H: Random / Brownian Baseline
5. **Experiment 5: Behavioral Variability & Non-Scripted Multi-Target Sandbox**:
   - Measuring trajectory entropy, reaction latency, turning distributions, and context-dependent adaptation under sudden target shifts.
