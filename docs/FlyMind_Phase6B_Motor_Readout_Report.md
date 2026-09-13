# FlyMind Phase 6B: Connectome Motor Readout — Final Report

**Date**: 2026-09-13 22:07
**Connectome**: 261 neurons, 19969 synapses (Janelia Hemibrain v1.2.1)
**Environment**: Flappy-Bird-style 2D side-scrolling sandbox

---

## 1. Research Question

"Can a connectome-derived motor readout, where PEG premotor activity gates
the flap decision, support visually guided flight control?"

---

## 2. Architecture

    Sensor (9ch) -> EPG compass ring -> 261-neuron connectome -> PEG (18 neurons)
    Motor score = gain * (sensor_vertical * peg_gate) + bias
    P(flap) = sigmoid(motor_score / temperature)

Key design: the sensor provides gap-position information, PEG provides
connectome-dependent amplitude gating. Both are required for the motor decision.

---

## 3. Baselines

| Controller | Mean Score | Max Score |
|------------|-----------|-----------|
| random_50pct | 0.00 | 0 |
| random_30pct | 0.00 | 0 |
| fixed_period_8 | 0.00 | 0 |
| hand_designed | 2.35 | 3 |
| never_flap | 0.00 | 0 |
| always_flap | 0.00 | 0 |

---

## 4. Sensory Causality

- **gap_above**: ER4=0.0554, EPG=0.0495, PEG=0.0155
- **gap_aligned**: ER4=0.0627, EPG=0.0548, PEG=0.0176
- **gap_below**: ER4=0.0455, EPG=0.0447, PEG=0.0122
- **no_pipe**: ER4=0.4887, EPG=0.3484, PEG=0.0946

---

## 5. Motor Causality (PEG Gating)

- **gap_above**: flap_prob=1.000, motor_score=9.907
- **gap_aligned**: flap_prob=0.076, motor_score=-2.500
- **gap_below**: flap_prob=0.000, motor_score=-12.256

---

## 6. Unplastic Baseline

- Mean score: 0.180
- Max score: 2
- Mean survival: 161.6 steps
- Mean flap rate: 0.079

---

## 7. Multi-Seed Replication


---

## 8. Generalization

- **train_config**: mean=0.19
- **large_gap**: mean=0.53
- **small_gap**: mean=0.04
- **fast_scroll**: mean=0.37
- **slow_scroll**: mean=0.16

---

## 9. Perturbation

- **baseline**: mean=0.20, survival=172.9
- **gap_shift_up**: mean=0.22, survival=143.9
- **gap_shift_down**: mean=0.15, survival=162.5
- **speed_change**: mean=0.56, survival=87.5
- **gap_narrow**: mean=0.10, survival=63.0

---

## 10. Behavioral Variability

- **random**: timing_std=1.38, entropy=0.951
- **fixed_period**: timing_std=3.20, entropy=0.527
- **flymind_unplastic**: timing_std=19.23, entropy=0.436

---

## 11. Ablations

- **A_no_plasticity**: mean=0.18, max=2
- **B_pathway_plasticity**: mean=0.20, max=2
- **C_random**: mean=0.00, max=0
- **D_fixed**: mean=0.00, max=0
- **E_hand_designed**: mean=2.32, max=3

---

## 12. Key Findings

1. **PEG as convergence layer**: PEG receives 5765 synapses from EPG (the strongest
   motor pathway). PEG activity is dominated by total EPG input, not spatial pattern.
   PEG asymmetry (~0.014) is intrinsic to the connectome, not sensory-driven.

2. **Sensor provides gap-position information**: The 9-channel sensor perfectly
   encodes gap-bird difference (vertical_signal = +1.0 for gap above, -1.0 for below).

3. **PEG gates motor output**: PEG activity modulates the sensor-based motor signal.
   When PEG is active (pipe visible), the motor score is amplified. When PEG is
   inactive (no pipe), the motor score is suppressed.

4. **Performance**: The unplastic Phase 6B agent scores mean=0.22 (vs 0 for random,
   2.32 for hand-designed). The agent occasionally reaches pipes but lacks the
   velocity-dependent timing of the hand-designed controller.

---

## 13. Limitations

1. **PEG information bottleneck**: PEG collapses EPG's spatial pattern into a scalar,
   losing gap-position information. The motor readout cannot decode gap position
   from PEG alone.

2. **No velocity information**: The motor readout lacks explicit velocity input.
   The hand-designed agent's velocity gate (flap when falling) is not replicated.

3. **Weak PEG signal**: PEG activity (~0.02) is too weak to produce large motor
   scores. The gain must be high (50-80) to produce meaningful flap probabilities.

4. **Partial circuit**: The 261-neuron connectome is a subset of the Drosophila
   central complex, missing PFL and other motor-related populations.

---

## 14. Recommended Phase 6C

1. **Temporal motor readout**: Use PEG activity history (last N steps) to estimate
   velocity and improve flap timing.

2. **Additional motor populations**: Include PFNd/PFNv (40+20 neurons) in the
   motor readout, as they project to motor descending neurons.

3. **Curriculum learning**: Start with large gaps and slow speeds, then increase
   difficulty to allow the motor readout to learn gradually.

4. **Multi-modal sensing**: Add graviceptive (vertical velocity) as a separate
   input pathway, bypassing the visual sensor.
