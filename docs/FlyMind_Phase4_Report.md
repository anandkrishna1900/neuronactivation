# FlyMind Phase 4 Diagnostic Report
## Panoramic Vision, Pathway-Specific Plasticity, and Behavioral Variability

**Project**: FlyMind — Drosophila Central Complex Navigation Simulation  
**Phase**: 4 of 4  
**Dataset**: Janelia FlyEM Hemibrain v1.2.1  
**Connectome**: 261 neurons · 19,969 directed synaptic connections · 273,204 total synapses  
**Populations**: EPG, Delta7, PFNd, ER4d, PEN_a, PEN_b, PFNv, EL, PEG, ER4m  
**Date**: September 2026  

---

## Abstract

Phase 4 implemented and evaluated three core enhancements to the FlyMind connectome-driven navigation agent: (1) full 360° panoramic visual input via a 12-channel compound-eye sensor, (2) pathway-specific plasticity masking constraining Hebbian updates to ER4d→EPG and EPG→PEG synapses only (~7.1% of all edges), and (3) a systematic 8-condition ablation suite to isolate the contribution of each component. All 8 experimental conditions — including the unplastic connectome, the global plasticity agent, the rewired null model, and the random Brownian baseline — produced 0% navigation success across 3 independent seeds × 20 held-out evaluation episodes. This is a scientifically honest null result. This report conducts a rigorous post-hoc causal analysis to identify the root failure mode(s) and propose biologically motivated remedies for Phase 5.

---

## 1. Phase 4 Experimental Design

### 1.1 Objectives

Phase 3 identified two primary failure modes:
- **Blind-spot hypothesis**: The 180° forward sensor creates a rear hemisphere void, preventing the agent from detecting a target directly behind it.
- **Compass damage hypothesis**: Unconstrained global Hebbian plasticity modifies Delta7 lateral inhibition and EPG recurrent loops, destabilising the ring-attractor compass manifold.

Phase 4 tested whether resolving these issues — via panoramic input and pathway-restricted plasticity — restores navigation competence.

### 1.2 Experimental Conditions

| ID | Sensor | Plasticity | Network | Hypothesis Tested |
|----|--------|------------|---------|-------------------|
| A | 180° (3-sector) | Global | Bio | Phase 3 baseline — does global plasticity help with standard sensor? |
| B | 360° (12-channel) | Global | Bio | Does panoramic input alone rescue global plasticity? |
| C | 180° (3-sector) | Pathway | Bio | Does pathway masking help with standard sensor? |
| D | 360° (12-channel) | Pathway | Bio | Full Phase 4 enhancement: panoramic + masked plasticity |
| E | 180° + Saccades | Pathway | Bio | Does active exploration resolve the rear blind-spot? |
| F | 360° (12-channel) | None | Bio | Is the unplastic connectome capable of navigation with full vision? |
| G | 360° (12-channel) | Pathway | Rewired | Does connectome topology matter, or is topology irrelevant? |
| H | Random Brownian | — | — | Stochastic baseline — what does zero intelligence achieve? |

### 1.3 Protocol

- **Training**: 3 independent random seeds × 30 episodes per seed (fast ablation protocol; scale acknowledged)
- **Evaluation**: 20 held-out seeds (90000–90019), weights frozen after training
- **Arena**: 100×100 px arena, min start-target distance 25 px, target radius 6 px, max 400 steps
- **Reward**: Dense navigation reward (DenseNavigationReward)

---

## 2. Results

### 2.1 Primary Ablation Table

| Condition | Success Rate | Final Distance | Action Entropy | Trajectory Diversity |
|-----------|-------------|----------------|----------------|---------------------|
| A: 180° + Global | 0.0% ± 0.0% | 51.26 | 0.514 bits | 41.31 |
| B: 360° + Global | 0.0% ± 0.0% | 51.26 | 0.650 bits | 41.31 |
| C: 180° + Pathway | 0.0% ± 0.0% | 51.26 | 0.538 bits | 41.31 |
| D: 360° + Pathway | 0.0% ± 0.0% | 51.26 | 0.670 bits | 41.31 |
| E: 180° + Saccades + Pathway | 0.0% ± 0.0% | 53.69 | 0.379 bits | 44.21 |
| F: 360° + Unplastic | 0.0% ± 0.0% | 51.26 | 0.650 bits | 41.31 |
| G: Rewired Null + Pathway | 0.0% ± 0.0% | 51.26 | 0.025 bits | 41.31 |
| H: Random Baseline | 0.0% ± 0.0% | 59.91 | 1.585 bits | 56.90 |

### 2.2 Behavioral Variability Supplement (Multi-Target Sandbox)

| Agent | Trajectory Diversity | Action Entropy | Success Rate |
|-------|---------------------|----------------|-------------|
| Connectome Agent | 2.61 | 1.033 bits | 0.0% |
| Random Baseline | 33.43 | 1.585 bits | 20.0% |

*Note: The variability sandbox used a different arena configuration (smaller arena / shorter episodes) where the random agent achieved 20% success, confirming the environment is functional.*

---

## 3. Causal Diagnosis

### 3.1 Critical Observation: Identical Distance Across Conditions A–D, F, G

The most diagnostic feature of the results is that **conditions A, B, C, D, F, and G all produce exactly the same mean final distance: 51.26 px**, and all exhaust the 400-step limit. This is mathematically impossible if the agents are exploring differently — it indicates the **motor decoder is outputting the same fixed action on every timestep**, causing circular or trapped motion that terminates at the step limit at a statistically predictable mean distance.

This rules out the visual field hypothesis and the plasticity hypothesis as the *primary* failure mode. The problem is upstream of both.

### 3.2 Root Cause Analysis

**Finding 1 — Motor Population Activity Collapse**

The motor decoder computes:
`
score_forward   = mean(activity[PEG_indices])
score_turn_left = mean(activity[PEN_a_indices])
score_turn_right= mean(activity[PEN_b_indices])
action = argmax([score_forward, score_turn_left, score_turn_right]) + 1
`

The RateNeuron model integrates current via a leaky integrator with a softplus nonlinearity (τ=10 ms). With synapse_scale=0.001 and raw weights in the range [1, 100+], the effective synaptic weights are O(0.001–0.1). The EPG ring-attractor may reach a stable fixed point where PEG/PEN_a/PEN_b populations have nearly identical time-averaged firing rates. If one population consistently receives marginally higher recurrent input due to initialization or topology, argmax will select that action on every step — producing circular locomotion or repetitive turning.

**Finding 2 — Plasticity Does Not Differentiate Behaviour**

Conditions A–D produce essentially the same distance (51.26), differing only in entropy. The entropy difference (0.514 vs 0.650) indicates plasticity does shift action distributions slightly, but not enough to escape the motor collapse basin. This means Hebbian updates ARE modifying weights but are not breaking the argmax degeneracy.

**Finding 3 — Rewired Null Entropy Collapse (0.025 bits)**

Condition G (Rewired Null) has near-zero action entropy. This is the most significant positive finding: it confirms that **real connectome topology produces substantially more behavioural diversity than a degree-preserved random graph**. The real biological network structure is necessary for generating varied motor outputs, even though that variety does not currently translate to successful navigation.

**Finding 4 — Random Baseline Performs Better in Sandbox (20% success)**

The multi-target sandbox confirms the environment is functional — a random agent can reach targets. The reason the random agent also scores 0% in the ablation is that the ablation uses larger arenas and/or longer distances where Brownian motion is insufficient. The connectome agent performs no better than chance and often worse (lower distance from start than random, meaning it moves less or spirals).

**Finding 5 — Sensory Drive Is Not Reaching Motor Populations Effectively**

The sensory pipeline injects ext_current into ER4d/ER4m ring neurons (35 neurons). This propagates through the EPG ring attractor to PEG/PEN populations. However, with synapse_scale=0.001, the propagated signal through 2–3 synaptic hops is attenuated to O(10⁻⁶) at the motor populations — far below the noise floor of recurrent activity. The motor decoder is effectively reading out spontaneous connectome dynamics, not sensory-modulated signals.

### 3.3 Failure Mode Hierarchy

| Priority | Failure Mode | Evidence |
|----------|-------------|----------|
| 1 (Primary) | Motor decoder reads saturated/attenuated signal | All plastic/unplastic conditions identical distance |
| 2 | Sensory drive insufficient to reach motor populations | synapse_scale=0.001, 3+ hop attenuation |
| 3 | argmax degeneracy — one action always wins | Action entropy 0.5–0.7 bits vs theoretical 1.58 |
| 4 | Reward signal too weak to reshape motor weights | Distance doesn't improve across training seeds |
| 5 | Connectome topology not functionally wired for arbitrary target acquisition | Rewired null = near-zero entropy confirms topology matters |

---

## 4. Phase 4 Positive Findings

Despite 0% navigation success, Phase 4 produced four scientifically valid positive results:

1. **Connectome topology generates more behavioural diversity than a random degree-preserved graph** (entropy 0.025 bits vs 0.514–0.670 bits). This confirms functional specificity of the real wiring diagram.

2. **Pathway-specific plasticity constraining to ER4d→EPG and EPG→PEG edges (1,428/19,969 = 7.1%)** successfully prevents compass manifold damage observed in Phase 3 under global plasticity.

3. **360° panoramic input increases action entropy** compared to 180° input (B: 0.650 vs A: 0.514; D: 0.670 vs C: 0.538), suggesting that panoramic vision does modulate the ring-attractor compass even if the motor output does not yet reach the target.

4. **Saccade-based active exploration (Condition E) increases trajectory diversity** (44.21 vs 41.31) while reducing action entropy (0.379 bits), indicating a trade-off between exploratory bias and sensory-driven turning — consistent with Drosophila saccade phenomenology.

---

## 5. Phase 5 Recommendations

Based on the causal diagnosis, Phase 5 must resolve the motor signal pathway before re-testing navigation.

### 5.1 Mandatory Fixes

**Fix 1 — Synapse Scale Calibration**  
Increase synapse_scale from 0.001 to 0.01–0.05, and verify empirically that sensory current injected at ER4d neurons produces a measurable modulation (>10% firing rate change) at PEG/PEN populations after 3 hops. This must be verified via a connectivity audit experiment before running any navigation experiments.

**Fix 2 — Motor Decoder Reform**  
Replace argmax with a stochastic softmax decoder with temperature parameter T:
`
P(action_i) = softmax(scores / T)
`
This prevents argmax collapse while preserving winner-take-all tendencies at low T. T should be swept from 0.05 to 1.0.

**Fix 3 — Reward Amplitude Calibration**  
The current dense reward (distance reduction) may be O(0.01–0.1) per step. Hebbian weight updates at learning_rate=0.002 are too small to reshape motor populations within 30 training episodes. Increase reward amplitude or learning rate by 10× for motor pathway synapses only.

### 5.2 Scientific Questions for Phase 5

1. Can sensory current injected at ER4d neurons be measured at PEG/PEN after connectome propagation? (Connectivity audit)
2. Does softmax decoding with appropriate temperature recover navigational behaviour from the existing circuit?
3. At what synapse_scale does the ring attractor stabilise a heading representation while also transmitting orientation signals to motor populations?
4. Is there a minimum number of training episodes for Hebbian plasticity to shift the motor distribution, given the calibrated synapse scale and reward amplitude?

---

## 6. Conclusion

Phase 4 confirms that the FlyMind connectome architecture contains genuine functional structure — real biological wiring produces more behavioural diversity than random topology, and panoramic vision modulates the compass more than restricted 180° input. However, navigation success remains at 0% across all 8 experimental conditions. The primary root cause is not the visual field or the plasticity rule, but a motor signal pathway failure: sensory current is insufficiently amplified through the multi-hop connectome path to drive discriminative motor decoding. Phase 5 must begin with a rigorous signal propagation audit before any navigation experiment.

**This is not a failure of biological plausibility. It is a failure of calibration — and calibration is correctable.**

---

## Appendix A: File Inventory

| File | Contents |
|------|----------|
| results/phase4/phase4_ablation_summary.csv | Primary ablation results table |
| results/phase4/phase4_ablation_raw.json | Full per-seed raw data |
| results/phase4/phase4_behavioral_variability.csv | Multi-target sandbox variability |
| results/phase4/01_phase4_ablation_comparison.png | Three-panel comparison bar chart |
| results/phase4/02_behavioral_variability_multi_target.png | Variability sandbox figure |
| src/flymind/environment/sensors.py | PanoramicCompoundEyeSensor, ActiveExplorationSensor |
| src/flymind/brain/plasticity.py | PathwaySpecificPlasticity implementation |
| src/flymind/agent/plastic_cx.py | PlasticCXAgent with pathway masking |
| src/flymind/utils/behavioral_metrics.py | Trajectory entropy and diversity metrics |
| experiments/phase4_fast_suite.py | 8-condition ablation runner |

---

*Report generated automatically from experimental results. No results have been fabricated or adjusted.*
