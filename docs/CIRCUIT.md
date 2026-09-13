# FlyMind V1 Circuit: Central Complex Heading Direction Loop

## 1. Circuit Selection Rationale

The **Central Complex (CX)** was selected as the first simulation target for the following reasons:

1. **Best-understood navigation circuit in any insect**: The CX heading direction system is among the most extensively characterized neural circuits in neuroscience, with computational models, physiological recordings, and behavioral experiments spanning three decades.
2. **Compact and simulatable on consumer hardware**: The core ring-attractor circuit comprises ~261 neurons and ~20,000 directed synaptic connections — small enough to simulate on a CPU without requiring GPU acceleration.
3. **Direct functional relevance to the task**: The CX generates and maintains an internal heading direction representation that directly drives steering output. It maps precisely onto the FlyMind agent's sensorimotor problem (visual landmark → internal compass → steering action).
4. **Dense, validated reconstruction in Hemibrain**: All 10 cell types are marked as **Traced** (fully proofread) in Hemibrain v1.2.1. Synapse counts have been independently validated.

---

## 2. Biological Background

### 2.1 The Heading Direction System

The Drosophila CX maintains an internal estimate of the fly's current heading (compass bearing) as a localized bump of neural activity that can be anchored to visual landmarks or maintained through angular integration during turns.

This was demonstrated by:
- Seelig & Jayaraman (2015): *Neural dynamics for landmark orientation and angular path integration.* Nature 521:186–191. [DOI: 10.1038/nature14446](https://doi.org/10.1038/nature14446)
- Kim et al. (2019): *Ring attractor dynamics in the Drosophila central brain.* Science 356:849–853.
- Turner-Evans et al. (2020): *The neuroanatomical ultrastructure and function of a heading direction circuit in Drosophila.* Neuron 108:145–163.

### 2.2 Ring Attractor Dynamics

The circuit implements a **ring attractor**: a continuous family of stable states where neural activity forms a single localized "bump" that can rotate around the ring. The angular position of the bump encodes the fly's heading direction.

```
                    Visual Input (ER4d/ER4m)
                           │
                   ┌───────▼───────┐
                   │  Ring Neurons  │   Landmark anchor
                   └───────┬───────┘
                           │
              ┌────────────▼────────────┐
              │       EPG Neurons       │  ← Heading compass bump
              │  (Ellipsoid Body ring)  │     ~46 cholinergic neurons
              └──┬──────────────────────┘
                 │  Reciprocal recurrence
              ┌──▼──────────────────────┐
              │     Delta7 Neurons      │  ← Global inhibition
              │  (PB long-range inhib)  │     ~42 GABAergic neurons
              └─────────────────────────┘
                    │           │
           ┌────────▼──┐   ┌───▼────────┐
           │  PEN_a     │   │  PEN_b     │  ← Angular velocity
           │ (PEN1, L)  │   │ (PEN2, R)  │    ~20 + 22 cholinergic
           └────┬───────┘   └────────────┘
                │
           ┌────▼───────────────────────┐
           │      PEG Neurons           │  ← Premotor steering output
           │  (Protocerebral Bridge)    │     ~18 cholinergic neurons
           └────────────────────────────┘
                │
           Motor commands → Turn Left / Forward / Turn Right
```

---

## 3. Extracted Neurons (Hemibrain v1.2.1)

### 3.1 Neuron Type Summary

All data from: Scheffer et al. (2020) eLife 9:e57448, accessed via neuprint.janelia.org.

| Cell Type | Count | Neuropil | Neurotransmitter | Biological Role |
|---|---|---|---|---|
| **EPG** | 46 | EB, PB, Gall | Cholinergic (excitatory) | Ring attractor bump; encodes heading direction |
| **Delta7** | 42 | PB | GABAergic (inhibitory) | Long-range global inhibition; enforces single-bump stability |
| **PFNd** | 40 | PB, FB, NO | Cholinergic (excitatory) | Fan-shaped body dorsal navigation columns |
| **ER4d** | 25 | BU, EB | Cholinergic (excitatory) | Visual landmark ring input to compass |
| **PEN_b(PEN2)** | 22 | PB, EB, NO | Cholinergic (excitatory) | Shifts bump rightward during clockwise turns |
| **PEN_a(PEN1)** | 20 | PB, EB, NO | Cholinergic (excitatory) | Shifts bump leftward during counter-clockwise turns |
| **PFNv** | 20 | PB, FB, NO | Cholinergic (excitatory) | Fan-shaped body ventral navigation columns |
| **EL** | 18 | EB | Cholinergic (excitatory) | Ellipsoid body local interneurons |
| **PEG** | 18 | PB, EB, Gall | Cholinergic (excitatory) | Premotor steering output (heading → action) |
| **ER4m** | 10 | BU, EB | Cholinergic (excitatory) | Visual motion ring input to compass |

**Total**: 261 neurons

### 3.2 Synaptic Connectivity

**Total directed connections**: 19,969  
**Minimum synapse count included**: 1 (all verified synaptic contacts)

Key verified pathways (from neuPrint query, Hemibrain v1.2.1):
- **EPG → PEN_b**: up to 64 synapses per pair (compass drives angular integrator)
- **EPG → PEN_a**: up to 51 synapses per pair
- **Delta7 → EPG**: up to 49 synapses per pair (GABAergic inhibition stabilizes bump)

---

## 4. Modeling Assumptions for this Circuit

| Component | Biological Source | Modeling Assumption | Justification |
|---|---|---|---|
| Neuron identities & count | Hemibrain v1.2.1 (Traced, proofread) | Used directly | Gold standard reconstruction |
| Synaptic topology | T-bar / PSD annotations in EM | Used directly | Direct physical measurement |
| Connection weights | Synapse count per directed pair | Weight ∝ synapse count | Validated proxy in Drosophila (Caron et al. 2013) |
| Neurotransmitter polarity | Eckstein et al. (Nature 2024) predictions | Cholinergic=+1, GABAergic=−1 | High-confidence prediction (>90% accuracy validated) |
| Neuron dynamics | Hodgkin-Huxley / LIF literature | Rate neuron or LIF point model | Multi-compartment biophysics underconstrained by available data |
| Sensory encoding | ER4d/ER4m physiology | Simplified 3-sector visual receptive field | Full optic lobe simulation deferred to V3 |
| Motor decoding | PEG/LAL descending physiology | Population argmax over PEG cluster activity | Full biomechanics deferred to V8 |

---

## 5. Known Limitations of V1

1. **No optic lobe**: Visual signals are injected directly into ER4 ring neurons with a simplified geometric encoding. The real visual transduction pathway (retina → lamina → medulla → AOTU → BU → EB) involves >80,000 neurons not included in V1.
2. **Spatial topology ignored**: Ring attractor dynamics depend on the precise angular layout of EPG-to-PEN connectivity. In V1, this is approximated from synapse counts without explicit angular coordinate constraints.
3. **No gap junctions**: Electrical synapses present in the CX are not detectable at EM resolution and are omitted.
4. **Male VNC not included**: Motor commands from PEG neurons travel via descending neurons to the ventral nerve cord (VNC/MANC). This final output stage is abstracted as a simple population vote.

---

## 6. References

1. Scheffer, L. K., et al. (2020). *A connectome and analysis of the adult Drosophila central brain.* eLife 9:e57448.
2. Hulse, B. K., et al. (2021). *A connectome of the Drosophila central complex reveals network motifs suitable for flexible navigation and contextual action selection.* eLife 10:e66039.
3. Seelig, J. D., & Jayaraman, V. (2015). *Neural dynamics for landmark orientation and angular path integration.* Nature 521:186–191.
4. Turner-Evans, D., et al. (2020). *The neuroanatomical ultrastructure and function of a heading direction circuit in Drosophila.* Neuron 108:145–163.
5. Kim, S. S., et al. (2019). *Ring attractor dynamics in the Drosophila central brain.* Science 356:849–853.
6. Eckstein, N., et al. (2024). *Neurotransmitter classification from electron microscopy images at synapse resolution.* Nature 634:153–162.
