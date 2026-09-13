# FlyMind: Anatomical Plasticity Mask Specification

**Dataset**: Janelia FlyEM Hemibrain v1.2.1 (*Scheffer et al., 2020*)  
**Circuit**: Central Complex Heading & Steering Subsystem (261 Neurons, 19,969 Synapses)  

---

## 1. Scientific Rationale for Pathway-Specific Masking

In biological *Drosophila*, the core heading direction network in the Protocerebral Bridge (PB) and Ellipsoid Body (EB) is a hardwired, highly conserved recurrent ring attractor (*Kim et al., 2017; Turner-Evans et al., 2020*). Lateral inhibition by `Delta7` neurons and mutual excitation between `EPG` and `PEN` neurons maintain the compass representation.

Applying global, unconstrained Hebbian learning across all 19,969 synaptic connections damages this recurrent ring attractor. Synaptic plasticity in the fly central brain during visual heading learning is localized specifically to:
1. **Sensory-to-Compass Mapping**: `ER4d -> EPG` ring synapses in the Ellipsoid Body.
2. **Compass-to-Motor Steering Mapping**: `EPG -> PEG` descending projection synapses.

All internal recurrent compass loops and lateral inhibitory channels are **topologically frozen**.

---

## 2. Quantitative Synaptic Breakdown

| Synaptic Pathway | Pre-Synaptic Cell Type | Post-Synaptic Cell Type | Edge Count | Plasticity Status | Neurotransmitter | Biological Role |
|---|---|---|---|---|---|---|
| **Visual Drive $\to$ Compass** | `ER4d` | `EPG` | **1,148** | **PLASTIC** | Cholinergic (+1) | Visual landmark alignment |
| **Compass $\to$ Motor Output** | `EPG` | `PEG` | **280** | **PLASTIC** | Cholinergic (+1) | Steering action selection |
| **Compass $\leftrightarrow$ Ring Attractor** | `Delta7`, `PEN_a`, `PEN_b` | `EPG`, `Delta7`, `PEN` | **18,541** | **FROZEN** | GABAergic / Cholinergic | Preserves recurrent compass manifold |

### Summary Metrics:
- **Total Network Synaptic Connections**: 19,969
- **Plastic Synaptic Connections**: **1,428 (7.15%)**
- **Frozen Synaptic Connections**: **18,541 (92.85%)**
- **Runtime Mask Enforcement**: Synapses outside the mask strictly undergo $\Delta W_{ij} \equiv 0.0$.
