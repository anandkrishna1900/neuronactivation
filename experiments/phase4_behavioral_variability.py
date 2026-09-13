"""
Phase 4 Behavioral Variability Experiment:
Sandbox with multiple targets to measure whether connectome controllers exhibit structured,
state-dependent, and adaptive behavioral variability versus deterministic or purely random control.
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from flymind.connectome.loader import ConnectomeLoader
from flymind.brain.network import NeuralNetwork
from flymind.brain.neuron import RateNeuron
from flymind.environment.world import VirtualArena, ArenaState
from flymind.environment.sensors import PanoramicCompoundEyeSensor
from flymind.environment.rewards import DenseNavigationReward
from flymind.agent.plastic_cx import PlasticCXAgent
from flymind.agent.fly import RandomFlyAgent
from flymind.utils.behavioral_metrics import compute_behavioral_metrics, compute_turning_angle_distribution

GRAPH_PATH = ROOT / "data" / "processed" / "cx_heading_v1.json"
RESULTS_DIR = ROOT / "results" / "phase4"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class MultiTargetSandboxArena:
    """Multi-target environment allowing multiple valid goal trajectories."""

    def __init__(self, width=100.0, height=100.0, target_radius=6.0, step_size=2.0):
        self.width = width
        self.height = height
        self.target_radius = target_radius
        self.step_size = step_size
        self.agent_pos = np.array([50.0, 50.0], dtype=np.float64)
        self.agent_heading = 0.0
        # 3 equidistant target beacons (North, South-East, South-West)
        self.targets = [
            np.array([50.0, 85.0]),
            np.array([80.0, 25.0]),
            np.array([20.0, 25.0]),
        ]
        self.active_target_idx = 0
        self.step_count = 0

    def reset(self, initial_heading=0.0):
        self.agent_pos = np.array([50.0, 50.0], dtype=np.float64)
        self.agent_heading = float(initial_heading)
        self.step_count = 0
        return self.get_state()

    def get_state(self):
        # Senses nearest active target
        dists = [np.linalg.norm(self.agent_pos - t) for t in self.targets]
        self.active_target_idx = int(np.argmin(dists))
        target_pos = self.targets[self.active_target_idx]
        reached = bool(dists[self.active_target_idx] <= self.target_radius)
        return ArenaState(
            agent_pos=self.agent_pos.copy(),
            agent_heading=self.agent_heading,
            target_pos=target_pos.copy(),
            step_count=self.step_count,
            target_reached=reached,
            collision_occurred=False,
        )

    def step(self, action_id: int):
        self.step_count += 1
        if action_id == 1: # Forward
            self.agent_pos[0] = np.clip(self.agent_pos[0] + self.step_size * np.cos(self.agent_heading), 0, self.width)
            self.agent_pos[1] = np.clip(self.agent_pos[1] + self.step_size * np.sin(self.agent_heading), 0, self.height)
        elif action_id == 2: # Turn Left
            self.agent_heading = (self.agent_heading + np.pi/8.0) % (2*np.pi)
        elif action_id == 3: # Turn Right
            self.agent_heading = (self.agent_heading - np.pi/8.0) % (2*np.pi)

        st = self.get_state()
        done = st.target_reached or (self.step_count >= 150)
        return st, 0.0, done, {"target_reached": st.target_reached, "step": self.step_count, "chosen_target": self.active_target_idx}


def run_behavioral_variability_experiment():
    print("=" * 70)
    print("FlyMind Phase 4: Structured Behavioral Variability Experiment")
    print("=" * 70)

    bio_graph = ConnectomeLoader.load_from_json(GRAPH_PATH)
    arena = MultiTargetSandboxArena()
    sensor = PanoramicCompoundEyeSensor()

    # Agents to compare:
    # 1. Connectome with Pathway Plasticity
    net = NeuralNetwork(graph=bio_graph, neuron_model_cls=RateNeuron, synapse_scale=0.001)
    agent_connectome = PlasticCXAgent(network=net, sensor=sensor, plasticity_mode="pathway")

    # 2. Random Agent
    agent_random = RandomFlyAgent(seed=42)

    n_trials = 30
    cx_records = []
    rand_records = []

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Run Connectome Controller Trials
    for t in range(n_trials):
        st = arena.reset(initial_heading=(t / n_trials) * (2*np.pi))
        agent_connectome.reset()
        traj = [st.agent_pos.copy()]
        acts = []
        headings = [st.agent_heading]
        done = False

        while not done:
            a = agent_connectome.act(st)
            acts.append(a)
            st, _, done, info = arena.step(a)
            traj.append(st.agent_pos.copy())
            headings.append(st.agent_heading)

        traj = np.array(traj)
        cx_records.append({"trajectory": traj, "actions": acts, "headings": headings, "success": info["target_reached"]})
        ax1.plot(traj[:, 0], traj[:, 1], color="#2980B9", alpha=0.6, linewidth=1.5)

    # Run Random Controller Trials
    for t in range(n_trials):
        st = arena.reset(initial_heading=(t / n_trials) * (2*np.pi))
        traj = [st.agent_pos.copy()]
        acts = []
        headings = [st.agent_heading]
        done = False

        while not done:
            a = agent_random.act(st)
            acts.append(a)
            st, _, done, info = arena.step(a)
            traj.append(st.agent_pos.copy())
            headings.append(st.agent_heading)

        traj = np.array(traj)
        rand_records.append({"trajectory": traj, "actions": acts, "headings": headings, "success": info["target_reached"]})
        ax2.plot(traj[:, 0], traj[:, 1], color="#E67E22", alpha=0.6, linewidth=1.5)

    # Plot arena target beacons
    for target in arena.targets:
        c1 = plt.Circle(target, arena.target_radius, color="#27AE60", alpha=0.4)
        c2 = plt.Circle(target, arena.target_radius, color="#27AE60", alpha=0.4)
        ax1.add_patch(c1)
        ax2.add_patch(c2)

    ax1.set_title(f"Connectome Agent: Structured Variable Trajectories\n(30 Initial Headings -> 3 Multi-Goal Beacons)", fontsize=10, fontweight="bold")
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 100)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.set_title(f"Random Baseline: Unstructured Brownian Walks\n(30 Initial Headings -> 3 Multi-Goal Beacons)", fontsize=10, fontweight="bold")
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 100)
    ax2.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    plot_p = RESULTS_DIR / "02_behavioral_variability_multi_target.png"
    fig.savefig(plot_p, dpi=150)
    plt.close(fig)

    # Compute Metrics
    cx_metrics = compute_behavioral_metrics(cx_records)
    rand_metrics = compute_behavioral_metrics(rand_records)

    df_b = pd.DataFrame([
        {"agent": "Connectome Agent", **cx_metrics},
        {"agent": "Random Baseline", **rand_metrics},
    ])
    df_b.to_csv(RESULTS_DIR / "phase4_behavioral_variability.csv", index=False)
    print(f"  [SAVED] Behavioral variability plot to: {plot_p}")
    print(f"  [SAVED] Metrics to: {RESULTS_DIR / 'phase4_behavioral_variability.csv'}")


if __name__ == "__main__":
    run_behavioral_variability_experiment()
