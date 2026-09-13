# FlyMind — Connectome-Based Artificial Fly

A research-oriented computational neuroscience platform that uses real **Drosophila melanogaster (fruit fly) connectome data** to create a biologically grounded artificial agent capable of learning and navigating a virtual environment.

---

## 🔬 Scientific Integrity & Scope

FlyMind evaluates whether the structural properties of a real biological neural network produce useful, adaptive behavior when situated inside a simulated sensorimotor environment. 

To maintain scientific rigor, we explicitly differentiate between what is empirical biological fact and what is computational modeling:

### 1. Real Biological Data (Empirical Ground Truth)
- **Connectome Topology**: Synapse-level directed graph extracted from electron-microscopy reconstructions (Janelia FlyEM Hemibrain v1.2.1; Scheffer et al., 2020).
- **Neuron Identifiers & Classes**: Empirical cell types (`E-PG`, `P-EN`, `Delta7`, `ER`, `P-EG`, etc.) from verified anatomical literature.
- **Synaptic Weights**: Derived from physical synapse counts (T-bars and postsynaptic densities).
- **Neurotransmitter Profiles**: Sign and modality based on published predictions and experimental validation (Eckstein et al., Nature 2024).

### 2. Model Assumptions & Approximations
- **Neuron Dynamics**: Point-neuron approximations (Rate-based or Leaky Integrate-and-Fire) rather than multi-compartment morphological models with spatial dendrites.
- **Sensory Encoding**: Simplified visual receptive fields mapping the environment into ring neuron current injections.
- **Motor Decoding**: Premotor population activity translated into kinematic actions (forward, turn left, turn right).
- **Plasticity Rules**: Local Hebbian, STDP, and reward-modulated plasticity hypotheses; **no global backpropagation** inside the primary biological model.

### 3. Engineering & Simulation Components
- **Virtual Arena**: 2D continuous/discrete simulation environment.
- **Baselines**: Random walk agent and conventional artificial neural network (MLP/RL) agent.
- **Ablation Suite**: Weight-shuffled, randomized graph, and lesion controls.
- **Visualization**: Dual-view synchronized environment and neural activation viewer.

> **Terminology Note**: We describe FlyMind as a **connectome-derived artificial agent**, not a "simulated fly brain."

---

## 📂 Project Architecture

```
Fly stuff/
├── README.md                  # Project overview & scientific scope
├── requirements.txt           # Python package dependencies
├── pyproject.toml             # Package metadata and configuration
│
├── docs/                      # Scientific documentation
│   ├── DATASETS.md            # Connectome datasets survey & V1 selection
│   ├── ARCHITECTURE.md        # Technical architecture & data flow
│   ├── BIOLOGICAL_ASSUMPTIONS.md # Explicit modeling approximations
│   └── EXPERIMENTS.md         # Baseline and ablation protocols
│
├── data/
│   ├── raw/                   # Raw API query extracts
│   ├── processed/             # Sparse graph representations
│   └── README.md              # Data provenance and policies
│
├── src/flymind/               # Core library
│   ├── connectome/            # Data loader, query engine, graph builder
│   ├── brain/                 # Point neuron models, synapses, simulation loop
│   ├── environment/           # 2D arena, sensors, action space
│   ├── agent/                 # Fly agent wrapper
│   └── utils/                 # Config, helpers, plotting
│
├── experiments/               # Experiment definitions (baseline, connectome, ablation)
├── notebooks/                 # Analysis and visualization notebooks
├── tests/                     # Automated test suite
└── results/                   # Reproducible experiment runs
```

---

## 🚀 Quickstart & Setup

### 1. Clone & Environment Setup
```bash
# Clone the repository and navigate to root
cd "Fly stuff"

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies in editable mode
pip install -e .
pip install -r requirements.txt
```

### 2. Connectome API Configuration (Optional for V1)
FlyMind uses the **Janelia FlyEM Hemibrain v1.2.1** dataset. You can obtain a free API key at [neuprint.janelia.org](https://neuprint.janelia.org).
```bash
# Set your API token (optional; offline cached circuit fixture provided)
export NEUPRINT_APPLICATION_CREDENTIALS="your_token_here"
```

### 3. Run Verification Tests
```bash
pytest
```

---

## 🛣️ Research Roadmap

* **Phase 0 (Current)**: Connectome dataset research, comparative analysis, repository scaffolding.
* **Phase 1**: Architecture and data contract definition.
* **Phase 2**: Connectome data pipeline (loader, sparse graph builder, validation checks).
* **Phase 3**: Extraction of Central Complex navigation circuit (~400–800 neurons).
* **Phase 4**: Graph representations and topological analysis.
* **Phase 5**: Neural simulator (discrete-timestep, rate and LIF dynamics).
* **Phase 6**: 2D virtual fly environment.
* **Phase 7 & 8**: Biologically grounded sensor encoding and motor decoding.
* **Phase 9**: Local biological plasticity (reward-modulated Hebbian / STDP).
* **Phase 10 & 11**: Baseline comparisons and ablation studies.
* **Phase 12 & 13**: Real-time visualization and experiment tracking.

---

## 📚 Key References
- Scheffer, L. K., et al. (2020). *A connectome and analysis of the adult Drosophila central brain*. eLife, 9:e57448.
- Hulse, B. K., et al. (2021). *A connectome of the Drosophila central complex reveals network motifs suitable for flexible navigation and contextual action selection*. eLife, 10:e66039.
- Turner-Evans, D., et al. (2020). *The neuroanatomical ultrastructure and function of a heading direction circuit in Drosophila*. Neuron, 108(1), 145-163.
- Dorkenwald, S., et al. (2024). *Neuronal wiring diagram of an adult brain*. Nature, 634, 124–138.
- Eckstein, N., et al. (2024). *Neurotransmitter classification from electron microscopy images*. Nature, 634, 153–162.
