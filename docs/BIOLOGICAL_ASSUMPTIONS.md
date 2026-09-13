# Biological Assumptions, Approximations, and Empirical Boundaries

## 1. Purpose of this Document

In accordance with scientific integrity guidelines, this document explicitly details the boundary between **empirically measured biological data** and **computational modeling assumptions** in FlyMind. 

We do not describe the model as a literal "simulated fly brain," but rather as a **connectome-derived artificial agent**.

---

## 2. Taxonomy of Components

| Component | Category | Biological Ground Truth Source | Modeling Assumption / Approximation |
|---|---|---|---|
| **Neuron Identification** | Biological Data | Janelia FlyEM Hemibrain v1.2.1 (`bodyId`, `type`, `instance`) | Neurons are assumed to act as point processes (isopotential compartments), ignoring spatial dendritic morphology in V1. |
| **Synaptic Topology** | Biological Data | T-bar and PSD electron microscopy annotations | Direction of chemical transmission is known; electrical gap junctions are omitted in V1 due to EM resolution limits. |
| **Connection Weight** | Biological Data | Integer count of individual synaptic contacts between neuron pairs | Synaptic weight is approximated as proportional to synapse count ($W_{ij} \propto \text{syn\_count}_{ij}$). |
| **Neurotransmitter Signs** | Biological / Inference | Eckstein et al. (2020/2024) predictions | Cholinergic = Excitatory (+1); GABAergic & Glutamatergic = Inhibitory (-1) (Drosophila GluCl receptors are chloride-gated). |
| **Neuron Dynamics** | Assumption | Hodgkin-Huxley / patch-clamp literature | Simplified Rate-based or Leaky Integrate-and-Fire (LIF) models; constant membrane capacitance and resistance across cell types. |
| **Sensory Encoding** | Approximation | Drosophila visual system literature (ommatidia → lamina → medulla → AOTU → BU → EB ring neurons) | Simplified geometric receptive fields (left, center, right visual activation sectors) injecting current directly into ring neurons. |
| **Motor Decoding** | Approximation | Lateral Accessory Lobe (LAL) descending steering literature | Population voting or softmax action selection over left/right turning premotor clusters; physical biomechanics abstracted to kinematic motion. |
| **Plasticity** | Modeling Hypothesis | Central complex dopamine/mushroom body plasticity studies | Modulated Hebbian or three-factor plasticity rules operating on designated subcircuit synapses; no global backpropagation. |

---

## 3. Detailed Assumptions for V1

### 3.1 Point Neurons vs. Multi-compartment Dendritic Trees
* **Biological Reality**: Drosophila neurons possess complex, arborized neurites where synaptic integration is active, non-linear, and compartmentalized.
* **V1 Approximation**: Single-compartment point neurons. Each neuron maintains a scalar activation state (or membrane potential $V(t)$).
* **Justification**: Multi-compartment biophysical simulation of hundreds of arborized neurons requires thousands of spatial compartments and extensive unknown ion-channel distributions, making simulation prohibitively slow on consumer hardware and under-constrained by empirical data.

### 3.2 Synapse Count as Functional Conductance
* **Biological Reality**: Physiological synaptic strength depends on presynaptic vesicle release probability, receptor density, receptor subunit composition, and distance from the spike initiation zone.
* **V1 Approximation**: Functional synaptic weight $W_{ij}$ is modeled as:
  $$W_{ij} = \text{sign}(NT_i) \cdot \alpha \cdot \text{SynapseCount}_{ij}$$
  where $\text{sign}(NT_i) \in \{+1, -1\}$ is determined by the presynaptic neurotransmitter and $\alpha$ is a global scaling factor.
* **Justification**: In Drosophila connectomics, synapse count has been shown in multiple physiological studies (e.g., Caron et al., 2013; Turner-Evans et al., 2020) to correlate strongly with post-synaptic potential amplitude.

### 3.3 Visual Transduction Abstraction
* **Biological Reality**: Drosophila compound eyes feature ~750-800 ommatidia per eye, projecting to complex retinotopic motion-detection columns in the lamina, medulla, and lobula before reaching the Central Complex via optic glomeruli.
* **V1 Approximation**: A simplified 3-sector visual field (left, center, right) maps visual contrast/target bearing into current injection vectors for ring neuron clusters.
* **Justification**: This provides an isolatable sensorimotor testbench without requiring the simulation of over 80,000 optic lobe neurons in the initial phase.

---

## 4. Empirical Sources and References
1. Scheffer, L. K., et al. (2020). *A connectome and analysis of the adult Drosophila central brain*. eLife, 9:e57448.
2. Hulse, B. K., et al. (2021). *A connectome of the Drosophila central complex reveals network motifs suitable for flexible navigation and contextual action selection*. eLife, 10:e66039.
3. Turner-Evans, D., et al. (2020). *The neuroanatomical ultrastructure and function of a heading direction circuit in Drosophila*. Neuron, 108(1), 145-163.
4. Seelig, J. D., & Jayaraman, V. (2015). *Neural dynamics for landmark orientation and angular path integration*. Nature, 521(7551), 186-191.
5. Eckstein, N., et al. (2024). *Neurotransmitter classification from electron microscopy images*. Nature, 634, 153-162.
