# FlyMind Phase 2: Biological Audit & Methodology Documentation

**Project**: FlyMind — Connectome-Based Artificial *Drosophila*  
**Dataset**: Janelia FlyEM Hemibrain v1.2.1 (*Scheffer et al., 2020*)  
**Circuit**: Central Complex Heading & Steering Subsystem  
**Date**: September 2026  

---

## 1. Objective of this Audit

This document provides a strict, unambiguous audit of every component of **FlyMind Phase 2: Learned Heading Navigation**. In computational neuroscience, it is essential to distinguish empirical biological data from engineering assumptions and modeling approximations.

---

## 2. Empirical Biological Data (Ground Truth)

The following components are directly derived from published biological data and reconstructed electron microscopy datasets:

| Biological Component | Source / Citation | Status in FlyMind |
|---|---|---|
| **Neuron Identity & Counts** | Janelia FlyEM Hemibrain v1.2.1 (*Scheffer et al., 2020*) | **Direct Empirical Data** (261 verified neurons) |
| **Directed Synaptic Connections** | Janelia FlyEM Hemibrain v1.2.1 Cypher Database | **Direct Empirical Data** (19,969 verified directed connections) |
| **Synapse Counts & Weights** | EM Reconstruction Synapse Annotations | **Direct Empirical Data** (273,204 total counted synapses) |
| **Neurotransmitter Signs** | *Eckstein et al., 2024* (Optic lobe & Central Complex NT predictions) | **Direct Empirical Data** (EPG, PEN, PEG, ER4d = Cholinergic [+1]; Delta7 = GABAergic [-1]) |
| **Anatomical Glomerulus Structure** | Protocerebral Bridge (PB) columns $L1..L8, R1..R8$ (*Wolff et al., 2015; Green et al., 2017*) | **Direct Empirical Data** (Mapped from Hemibrain instance tags `EPG(PB08)_L3`, etc.) |
| **Ring Attractor Topology** | *Kim et al., 2017; Turner-Evans et al., 2020* | **Empirical Recurrent Wiring** (EPG $\leftrightarrow$ PEN and Delta7 lateral inhibition loops) |

---

## 3. Modeling Assumptions & Approximations

The following components are biologically plausible approximations of neural dynamics:

| Modeling Component | Assumption Made | Biological Justification |
|---|---|---|
| **Neuron Dynamics** | Rate Neuron model with continuous firing rate $r(t) = \tanh(\max(0, I_{\text{tot}}))$ | Captures population-level mean-field activity across millisecond timescales. |
| **Synaptic Conductance Scaling** | Linear scaling factor $s = 0.001$ applied to raw integer synapse counts | Maps physical synapse count to non-saturating conductance range. |
| **Integration Timestep** | Discrete simulation step $\Delta t = 1.0$ | Balances numerical stability with consumer CPU execution speed. |
| **Synaptic Edge Plasticity** | 3-factor eligibility-trace Hebbian updates strictly constrained to existing non-zero connections | Preserves underlying structural connectome topology; no de novo synapse creation. |

---

## 4. Engineering & Simulation Abstractions

The following components are engineering abstractions necessary to test embodied behavior in a simulated arena:

| Engineering Component | Nature of Abstraction | What is NOT Claimed |
|---|---|---|
| **2D Virtual Arena** | Continuous $100 \times 100$ unit bounded plane with forward/turn kinematics | Not a physical flight arena; simplified kinematic abstraction. |
| **Sensory Compound Eye** | 3-sector receptive field (Left, Center, Right) activated by relative bearing | Replaces full $700$-ommatidia optical array with a 3-channel visual drive. |
| **Motor Decoding** | Argmax over PEG/PEN population firing rates to select discrete actions | Replaces full descending nerve cord / VNC motor circuit with a discrete action decoder. |
| **Reward Function** | Dense distance-shaping reward and sparse goal arrival signal | Abstraction of dopaminergic reinforcement; not claimed to replicate exact mushroom body PAM/PPL1 firing rates. |

---

## 5. Non-Privileged Information Guarantee

To maintain research integrity, the neural network agent is **strictly isolated** from ground truth coordinates:
- The network **NEVER** receives target coordinates $(x_{\text{target}}, y_{\text{target}})$.
- The network **NEVER** receives agent coordinates $(x_{\text{agent}}, y_{\text{agent}})$.
- The network **NEVER** receives Euclidean distance or exact angular error.
- All spatial and reward calculations are executed internally by the environment.
- The brain receives only $[S_{\text{left}}, S_{\text{center}}, S_{\text{right}}]$ sensory drive into `ER4d`/`ER4m` ring neurons.

---

## 6. Audit Conclusion

FlyMind Phase 2 maintains scientific rigor by:
1. Grounding all neural connectivity and sign mapping in empirical Hemibrain data.
2. Restricting plasticity to biologically existing edges.
3. Explicitly distinguishing biological data from engineering abstractions.
