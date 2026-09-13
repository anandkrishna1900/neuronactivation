# Drosophila Melanogaster Connectome Datasets: Comprehensive Survey & V1 Selection

## 1. Executive Summary

This document provides a systematic evaluation of publicly available connectome datasets for *Drosophila melanogaster* (fruit fly) to establish an empirical, biologically grounded foundation for the **FlyMind** project.

To adhere to scientific integrity, the project requires:
1. Reconstructed biological neurons with unique identifiers and verified morphologies.
2. Direct synaptic connectivity (presynaptic active zones / T-bars and postsynaptic densities / PSDs).
3. Synapse weights (counts of verified synaptic contacts per connection).
4. Neurotransmitter identity (chemically confirmed or high-confidence deep-learning predictions).
5. Accessibility via standard open APIs or downloadable sparse representations.

Following this survey, **Janelia FlyEM Hemibrain (v1.2.1)** is selected as the primary dataset for **V1**, focused on the **Central Complex (CX)** navigation and steering loop.

---

## 2. Connectome Datasets Survey

### 2.1 Janelia FlyEM Hemibrain (v1.2.1)

* **Primary Citation**: Scheffer, L. K., Xu, C. S., Januszewski, M., Lu, Z., Takemura, S. Y., Hayworth, K. J., ... & Plaza, S. M. (2020). *A connectome and analysis of the adult Drosophila central brain*. eLife, 9, e57448. [DOI: 10.7554/eLife.57448](https://doi.org/10.7554/eLife.57448)
* **Specimen & Modality**: Adult female *Drosophila melanogaster*, focused ion beam scanning electron microscopy (FIB-SEM) at 8x8x8 nm voxel resolution.
* **Brain Coverage**: Central brain of one hemisphere, including full Central Complex (EB, PB, FB, NO, AB), Mushroom Body (MB), Antennal Lobe (AL), Lateral Horn (LH), and associated neuropils. Does not include outer optic lobes (medulla, lobula plate).
* **Neuron Count**: ~25,000 reconstructed neurons (~21,700 extensively proofread and named).
* **Synapse Count**: ~21.7 million chemical synapses (polyadic synapses with identified T-bars and corresponding postsynaptic partners).
* **Metadata Available**:
  * `bodyId`: Unique 64-bit integer identifier.
  * `type`: Anatomical cell type classification (e.g., `E-PG`, `P-EN`, `ER4d`, `Delta7`).
  * `instance`: Specific neuron instance name (e.g., `E-PG_L1`, `E-PG_R8`).
  * `status`: Proofreading status (`Traced`, `Roughly traced`, etc.).
  * `pre` / `post`: Total presynaptic and postsynaptic sites per neuron.
  * `roiInfo`: Exact synapse count breakdown per neuropil/ROI (Region of Interest).
* **Neurotransmitter Predictions**: Eckstein et al. (2020/2024) predictions mapped into neuPrint v1.2.1 metadata, categorizing acetylcholine (cholinergic, excitatory), GABA (GABAergic, inhibitory), glutamate (glutamatergic, typically inhibitory via GluClα in Drosophila), dopamine, serotonin, and octopamine.
* **Access Mechanism**:
  * **API**: neuPrint HTTP REST / Cypher query engine accessible via the official Python client `neuprint-python`.
  * **Server**: `https://neuprint.janelia.org` (dataset: `hemibrain:v1.2.1`).
  * **Flat Dumps**: Google Cloud Storage (`gs://flyem-hemibrain/v1.2.1/`) providing CSV/feather files of neurons, synapses, and connectivity matrices.
* **Licensing**: Creative Commons Attribution 4.0 International (CC-BY 4.0).

---

### 2.2 FlyWire / Whole-Brain Connectome (FAFB / Release 783)

* **Primary Citations**:
  * Dorkenwald, S., Li, P. H., Januszewski, M., ... & Seung, H. S. (2024). *Neuronal wiring diagram of an adult brain*. Nature, 634, 124–138.
  * Schlegel, P., Yin, Y., Bates, A. S., ... & Jefferis, G. S. (2024). *Whole-brain annotation and multi-connectome analysis of Drosophila*. Nature, 634, 139–152.
* **Specimen & Modality**: Full Adult Female Brain (FAFB), serial section transmission electron microscopy (ssTEM) at 4x4x40 nm voxel resolution, crowdsourced proofreading.
* **Brain Coverage**: Complete adult female brain, including both hemispheres, both optic lobes (lamina, medulla, lobula, lobula plate), subesophageal zone (SEZ), and central brain.
* **Neuron Count**: 139,255 proofread neurons (8,453 distinct cell types).
* **Synapse Count**: 54.5 million chemical synapses.
* **Metadata Available**:
  * `root_id`: Current segmentation identity.
  * Cell type annotations, hemispheric symmetry pairs, superclasses (sensory, interneuron, descending, ascending).
  * 3D spatial coordinates for proofread skeletons.
* **Neurotransmitter Predictions**: High-confidence machine-learning predictions per synapse and per cell (Eckstein et al., Nature 2024, >90% validation accuracy across 6 major neurotransmitters).
* **Access Mechanism**:
  * **API**: CAVEclient (`caveclient`), FlyWire Codex API (`codex.flywire.ai`).
  * **Bulk Downloads**: Princeton Data Commons & Zenodo flat Parquet/Feather files (synapse tables, connection matrices).
* **Licensing**: CC-BY 4.0.

---

### 2.3 Janelia MANC (Male Adult Nerve Cord v1.0)

* **Primary Citation**: Takemura, S. Y., Hayworth, K. J., Marin, E. C., ... & Berg, S. (2023). *A connectome of the male Drosophila ventral nerve cord*. bioRxiv / Nature.
* **Specimen & Modality**: Male adult ventral nerve cord (VNC), FIB-SEM at 8x8x8 nm.
* **Coverage**: Thoraco-abdominal ganglia (VNC), containing all leg, wing, and haltere motor neurons, sensory afferents from legs/wings, and descending/ascending projections to/from the brain.
* **Neuron Count**: ~14,600 reconstructed neurons.
* **Synapse Count**: ~6.5 million synapses.
* **Metadata Available**: Motor neuron targeting (e.g., specific leg muscle groups, steering muscles of the flight apparatus), descending neuron terminal zones.
* **Access Mechanism**: neuPrint (`manc:v1.0`) via `neuprint-python`, flat downloadable tables.
* **Licensing**: CC-BY 4.0.

---

### 2.4 Larval Drosophila Connectome

* **Primary Citation**: Winding, M., Pedigo, B. D., Barnes, C. L., ... & Zlatic, M. (2023). *The connectome of an insect brain*. Science, 379(6636), eadd9330.
* **Specimen & Modality**: First-instar larva (*Drosophila melanogaster*), serial section TEM.
* **Coverage**: Complete whole-organism central nervous system (brain and nerve cord).
* **Neuron Count**: 3,016 neurons.
* **Synapse Count**: 548,000 synaptic connections.
* **Characteristics**: Extremely compact, fully reconstructed with bilateral symmetry.
* **Limitation for V1**: Larval behavior consists primarily of peristaltic crawling, head sweeping, and larval chemotaxis/phototaxis. Adult Drosophila navigational dynamics (ring attractor heading compass, optic flow integration, visual landmark steering) reside in adult-specific circuits not present in the early larva.

---

## 3. Comparative Evaluation Matrix

| Criterion | Hemibrain v1.2.1 | FlyWire (FAFB 783) | MANC v1.0 | Larval CNS |
|---|---|---|---|---|
| **Neuron Count** | 25,000 | 139,255 | 14,600 | 3,016 |
| **Synapse Count** | 21,700,000 | 54,500,000 | 6,500,000 | 548,000 |
| **Navigation Circuit (CX)** | **Dense, gold standard** | Complete, both lobes | N/A (VNC only) | Incomplete/immature |
| **ROI Sub-Querying** | Highly granular (EB, PB, FB, NO, LAL) | CAVE/Codex ROIs | Leg/wing neuropils | Whole-brain only |
| **API Usability** | Excellent (`neuprint-python`) | Good (`caveclient`, Codex) | Excellent (`neuprint-python`) | Static tables |
| **Consumer PC Memory Footprint** | ~50 MB for CX subcircuit | Full brain > 10 GB | ~30 MB for motor | ~15 MB full |
| **Neurotransmitter Accuracy** | Chemically verified + Eckstein et al. | ML predicted (Eckstein 2024) | Predicted | Annotated |
| **Suitability for V1** | **Highest (Recommended)** | Ideal for V3+ visual expansion | Ideal for V8 motor control | Non-adult behavior |

---

## 4. Recommendation for V1: Janelia FlyEM Hemibrain

### 4.1 Justification
1. **Biological Specificity**: The Central Complex (CX) in Hemibrain has been exhaustively analyzed down to individual columnar cell types and single-synapse resolution (Hulse et al., eLife 2021). Its heading compass (ring attractor) and angular integration circuits are among the best understood biological neural networks in neuroscience.
2. **Computational Feasibility**: Rather than parsing millions of peripheral sensory neurons, we can query only the specific ROIs involved in heading, steering, and ring-attractor maintenance (`EB`: Ellipsoid Body, `PB`: Protocerebral Bridge, `NO`: Noduli, `LAL`: Lateral Accessory Lobe). The resulting graph contains approximately **400 to 800 neurons** and **15,000 to 35,000 synapses**.
3. **Reproducibility & Stability**: The Hemibrain v1.2.1 dataset is version-locked and permanently hosted on `neuprint.janelia.org`. Queries produce deterministic results.
4. **Standard Tooling**: The `neuprint-python` library provides straightforward, well-documented querying using Python and Cypher queries.

### 4.2 Target Subcircuit for Phase 3
The selected biological subcircuit will consist of the **Central Complex Navigation and Heading Ring Attractor**:
* **Visual Landmark & Sensory Input**: Ring neurons (`ER1`, `ER2`, `ER4d`) receiving visual feature inputs from the Anterior Optic Tubercle (AOTU) and bulb (BU), providing visual landmark cues to the compass.
* **Heading Direction Compass (Ring Attractor)**: `E-PG` (Ellipsoid body - Protocerebral bridge - Gall) neurons whose activity forms a single localized bump representing current orientation.
* **Angular Velocity & Self-Motion Integration**: `P-EN` (Protocerebral bridge - Ellipsoid body - Noduli) neurons that shift the compass bump left or right during angular movement.
* **Compass Recurrence & Global Inhibition**: `Delta7` neurons in the Protocerebral Bridge providing long-range inhibition that stabilizes the single-bump attractor dynamics.
* **Premotor & Steering Output**: `P-EG` and `P-FN` neurons projecting downstream to the Lateral Accessory Lobe (`LAL`) where descending steering signals are modulated.

---

## 5. API Access & Authentication Guide

### 5.1 Obtaining neuPrint Access
1. Navigate to [https://neuprint.janelia.org](https://neuprint.janelia.org).
2. Log in using any Google account.
3. Open your profile / account settings (top-right menu) to obtain your unique **API Token**.
4. Set the token in your local environment:
   ```bash
   # Linux / macOS
   export NEUPRINT_APPLICATION_CREDENTIALS="your_token_here"

   # Windows PowerShell
   $env:NEUPRINT_APPLICATION_CREDENTIALS="your_token_here"
   ```

### 5.2 Python Query Pattern
```python
from neuprint import Client, NeuronCriteria as NC, fetch_neurons, fetch_adjacencies

c = Client('neuprint.janelia.org', dataset='hemibrain:v1.2.1')

# Query compass neurons in the Ellipsoid Body
neurons_df, roi_counts_df = fetch_neurons(NC(type='E-PG.*', regex=True))

# Query synaptic connectivity between compass and integrator neurons
neuron_ids = neurons_df['bodyId'].tolist()
weights_df, _ = fetch_adjacencies(NC(bodyId=neuron_ids), NC(bodyId=neuron_ids))
```

### 5.3 Offline Caching & Fallback
To ensure that all unit tests, automated CI, and offline research function without requiring live internet access or personal authentication tokens, `flymind` will maintain a serialized, validated extraction of the target Central Complex navigation subcircuit in `data/processed/cx_heading_v1.json` and `data/processed/cx_heading_v1.parquet`.
