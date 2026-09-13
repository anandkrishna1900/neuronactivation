# FlyMind System Architecture

## 1. Architectural Philosophy

**FlyMind** is designed as a modular, reproducible computational neuroscience and artificial agent framework. It enforces a strict separation between:
1. **Connectome Topology (Empirical Biology)**: Graph data, synaptic counts, neuronal identifiers, and neurotransmitter classifications sourced directly from EM reconstructions.
2. **Neural Dynamics (Biophysical Models)**: Mathematical formulations of neuron membrane potentials, firing thresholds, synaptic transmission delays, and plasticity mechanisms.
3. **Environment & Embodiment (Simulation Engine)**: The virtual arena, sensory encoders, motor decoders, and physics loop.
4. **Experimental Infrastructure**: Tracking, baseline models, ablation harnesses, and visualization tools.

---

## 2. Directory Layout & Module Boundaries

```
flymind/
├── README.md                  # Project overview, scientific integrity statement, quickstart
├── requirements.txt           # Core Python dependencies
├── pyproject.toml             # Packaging, metadata, pytest configuration
│
├── docs/                      # Scientific documentation and specs
│   ├── DATASETS.md            # Connectome datasets survey and API guide
│   ├── ARCHITECTURE.md        # System design, data flow, component interfaces
│   ├── BIOLOGICAL_ASSUMPTIONS.md # Explicit documentation of all approximations
│   └── EXPERIMENTS.md         # Protocol for baseline, connectome, and ablation runs
│
├── data/                      # Local data storage
│   ├── raw/                   # Unmodified raw queries/dumps from neuPrint/FlyWire
│   ├── processed/             # Cleaned, standardized subcircuit graphs (sparse formats)
│   └── README.md              # Data provenance, versioning, and download instructions
│
├── src/flymind/               # Core source code
│   ├── __init__.py
│   ├── connectome/            # Connectome data extraction and graph management
│   │   ├── __init__.py
│   │   ├── loader.py          # Local/remote dataset loading
│   │   ├── query.py           # NeuPrint/Cypher/Codex query abstractions
│   │   ├── graph.py           # Directed weighted graph representation (sparse adjacency)
│   │   └── validation.py      # Biological sanity checks (no NaNs, no orphan edges)
│   │
│   ├── brain/                 # Neural simulation engine
│   │   ├── __init__.py
│   │   ├── neuron.py          # Neuron models (Rate-based, Leaky Integrate-and-Fire)
│   │   ├── synapse.py         # Synaptic dynamics, neurotransmitter sign mapping
│   │   ├── network.py         # Network topology + state orchestration
│   │   ├── simulator.py       # Discrete-timestep simulation loop
│   │   └── plasticity.py      # Local learning rules (Hebbian, STDP, reward-modulated)
│   │
│   ├── environment/           # Simulation arena & embodiment
│   │   ├── __init__.py
│   │   ├── world.py           # 2D continuous/discrete virtual arena
│   │   ├── sensors.py         # Visual & sensory receptive fields (left/center/right)
│   │   ├── actions.py         # Discrete locomotion actions (forward, turn left, turn right)
│   │   └── rewards.py         # Goal-directed reward signals
│   │
│   ├── agent/                 # Embodied agent wrapper
│   │   ├── __init__.py
│   │   └── fly.py             # Agent binding brain, sensors, and actions to the world
│   │
│   └── utils/                 # Utilities and tracking
│       ├── __init__.py
│       ├── config.py          # Experiment configuration schema
│       └── visualization.py   # Topology and trajectory plotting
│
├── experiments/               # Standardized experiment protocols
│   ├── baseline/              # Random & standard MLP/RL baselines
│   ├── connectome/            # Primary biological connectome trials
│   └── ablation/              # Randomized, weight-shuffled, and lesion ablations
│
├── notebooks/                 # Exploratory research & analysis notebooks
├── tests/                     # Automated unit and integration test suite
└── results/                   # Reproducible experiment runs (metrics, logs, plots)
```

---

## 3. Data Flow Diagram

```
         ┌──────────────────────────────────────┐
         │         2D Virtual Environment       │
         │  (Target beacon, boundaries, agent)  │
         └──────────────────┬───────────────────┘
                            │ Environment State
                            ▼
         ┌──────────────────────────────────────┐
         │            Sensory Model             │
         │   (Left / Center / Right Sensors)    │
         └──────────────────┬───────────────────┘
                            │ Receptive Field Activation
                            ▼
 ┌───────────────────────────────────────────────────────────────┐
 │               FlyMind Neural Network (Brain)                  │
 │                                                               │
 │   ┌─────────────────┐       ┌─────────────────────────────┐   │
 │   │  Input Neurons  │ ────► │ Intermediate Neurons (CX)   │   │
 │   │  (e.g., ER / BU)│       │ (E-PG compass, P-EN steer)  │   │
 │   └─────────────────┘       └──────────────┬──────────────┘   │
 │                                            │                  │
 │                                            ▼                  │
 │                             ┌─────────────────────────────┐   │
 │                             │ Output / Premotor Neurons   │   │
 │                             │ (e.g., P-EG / LAL descent)  │   │
 │                             └──────────────┬──────────────┘   │
 └────────────────────────────────────────────┼──────────────────┘
                                              │ Firing Rate / Spike Count
                                              ▼
         ┌──────────────────────────────────────┐
         │            Motor Decoder             │
         │   (Population argmax or softmax)     │
         └──────────────────┬───────────────────┘
                            │ Action (Turn Left, Forward, Turn Right)
                            ▼
         ┌──────────────────────────────────────┐
         │           Environment Step           │
         │ (Update position, calculate reward)  │
         └──────────────────────────────────────┘
```

---

## 4. Key Design Invariants

1. **Sparse Memory Footprint**: Large dense adjacency matrices ($N \times N$) are prohibited. All graphs utilize adjacency lists (`networkx`) or coordinate/CSR sparse matrices (`scipy.sparse`).
2. **Deterministic Reproducibility**: Every simulation step and weight initialization is seeded using explicit pseudo-random number generator instances (no global state mutations).
3. **Hardware Independence**: The simulation core relies solely on vectorized NumPy/SciPy operations without requiring a GPU for V1.
