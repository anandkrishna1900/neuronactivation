"""
Phase 6: Closed-loop simulation comparing ConnectomeFlyAgent vs RandomFlyAgent
in the 2D Virtual Arena.
"""

import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from flymind.connectome.loader import ConnectomeLoader
from flymind.brain.network import NeuralNetwork
from flymind.brain.neuron import RateNeuron
from flymind.environment.world import VirtualArena
from flymind.environment.sensors import SectorVisualSensor
from flymind.agent.fly import ConnectomeFlyAgent, RandomFlyAgent

GRAPH_PATH = ROOT / "data" / "processed" / "cx_heading_v1.json"


def run_trials(agent, arena, n_trials=100, seed_base=42):
    results = []
    for trial in range(n_trials):
        seed = seed_base + trial
        state = arena.reset(seed=seed)
        agent.reset()
        
        trajectory = [state.agent_pos.copy()]
        done = False
        total_reward = 0.0
        
        while not done:
            action = agent.act(state)
            state, reward, done, info = arena.step(action)
            trajectory.append(state.agent_pos.copy())
            total_reward += reward
            
        results.append({
            "trial": trial,
            "target_reached": state.target_reached,
            "steps": state.step_count,
            "final_distance": info["distance"],
            "total_reward": total_reward,
            "trajectory": np.array(trajectory)
        })
    return results


def main():
    print("=" * 60)
    print("FlyMind Phase 6: Closed-Loop Embodied Agent Navigation")
    print("Connectome: Hemibrain v1.2.1 CX Heading Circuit")
    print("=" * 60)

    # 1. Load connectome & build network
    graph = ConnectomeLoader.load_from_json(GRAPH_PATH)
    net = NeuralNetwork(
        graph=graph,
        neuron_model_cls=RateNeuron,
        synapse_scale=0.001,  # Optimal operating regime
        dt=1.0,
    )
    
    # 2. Extract input (ER4d/ER4m) and motor (PEG/PFNd) neuron indices
    er4d_indices = [net.id_to_idx[nid] for nid in net.neuron_ids
                    if graph.nx_graph.nodes[nid].get("cell_type") in ("ER4d", "ER4m")]
    peg_indices = [net.id_to_idx[nid] for nid in net.neuron_ids
                   if graph.nx_graph.nodes[nid].get("cell_type") in ("PEG", "PEN_a(PEN1)", "PEN_b(PEN2)")]

    # Split ER4d into 3 sectors: Left, Center, Right
    n_in = len(er4d_indices)
    s1, s2 = n_in // 3, 2 * n_in // 3
    input_sectors = [
        er4d_indices[:s1],       # Left
        er4d_indices[s1:s2],     # Center
        er4d_indices[s2:],       # Right
    ]

    # Split motor into 3 actions: Forward (center), Turn Left, Turn Right
    n_out = len(peg_indices)
    m1, m2 = n_out // 3, 2 * n_out // 3
    motor_sectors = [
        peg_indices[m1:m2],      # Forward
        peg_indices[:m1],        # Turn Left
        peg_indices[m2:],        # Turn Right
    ]

    print(f"Sensory input neurons: {len(er4d_indices)} across 3 visual sectors")
    print(f"Motor output neurons : {len(peg_indices)} across 3 motor actions")

    # Custom mapping Connectome Agent
    class EmbodiedConnectomeAgent(ConnectomeFlyAgent):
        def act(self, state):
            visual_signals = self.sensor.sense(state)  # [left, center, right]
            ext_current = np.zeros(self.network.num_neurons, dtype=np.float64)
            
            for sec_idx, neuron_grp in enumerate(input_sectors):
                for nid_idx in neuron_grp:
                    ext_current[nid_idx] = visual_signals[sec_idx] * 2.0
                    
            activity = self.network.step(ext_current)
            
            # Compute average activity for each motor action
            action_scores = []
            for m_grp in motor_sectors:
                if len(m_grp) > 0:
                    action_scores.append(float(np.mean(activity[m_grp])))
                else:
                    action_scores.append(0.0)
            
            # Action mapping: 0 -> Forward (1), 1 -> Turn Left (2), 2 -> Turn Right (3)
            choice = int(np.argmax(action_scores))
            return choice + 1

    arena = VirtualArena(width=100.0, height=100.0, target_radius=6.0, step_size=2.0, max_steps=400)
    sensor = SectorVisualSensor()

    connectome_agent = EmbodiedConnectomeAgent(network=net, sensor=sensor)
    random_agent = RandomFlyAgent(seed=42)

    n_trials = 50
    print(f"\nRunning {n_trials} navigation trials for Random Agent...")
    rand_results = run_trials(random_agent, arena, n_trials=n_trials)
    
    print(f"Running {n_trials} navigation trials for Connectome-Driven Agent...")
    cx_results = run_trials(connectome_agent, arena, n_trials=n_trials)

    rand_success = sum(r["target_reached"] for r in rand_results) / n_trials * 100
    cx_success = sum(r["target_reached"] for r in cx_results) / n_trials * 100

    rand_avg_steps = np.mean([r["steps"] for r in rand_results if r["target_reached"]] or [arena.max_steps])
    cx_avg_steps = np.mean([r["steps"] for r in cx_results if r["target_reached"]] or [arena.max_steps])

    rand_avg_dist = np.mean([r["final_distance"] for r in rand_results])
    cx_avg_dist = np.mean([r["final_distance"] for r in cx_results])

    print("\n" + "=" * 60)
    print(f"BENCHMARK RESULTS ({n_trials} Trials, Max Steps = {arena.max_steps})")
    print("=" * 60)
    print(f"{'Metric':<25} | {'Random Agent':<15} | {'Connectome Agent':<15}")
    print("-" * 60)
    print(f"{'Success Rate (%)':<25} | {rand_success:<15.1f} | {cx_success:<15.1f}")
    print(f"{'Avg Steps (Successes)':<25} | {rand_avg_steps:<15.1f} | {cx_avg_steps:<15.1f}")
    print(f"{'Final Target Dist (avg)':<25} | {rand_avg_dist:<15.2f} | {cx_avg_dist:<15.2f}")
    print("=" * 60)

    # Plot sample trajectories
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for i in range(min(15, n_trials)):
        traj = rand_results[i]["trajectory"]
        ax1.plot(traj[:, 0], traj[:, 1], alpha=0.6, linewidth=1.2)
    ax1.set_title(f"Random Agent (Success: {rand_success:.1f}%)")
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 100)
    ax1.grid(True, linestyle="--", alpha=0.5)

    for i in range(min(15, n_trials)):
        traj = cx_results[i]["trajectory"]
        color = "green" if cx_results[i]["target_reached"] else "red"
        ax2.plot(traj[:, 0], traj[:, 1], alpha=0.6, linewidth=1.2, color=color)
    ax2.set_title(f"Connectome Agent (Success: {cx_success:.1f}%)")
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 100)
    ax2.grid(True, linestyle="--", alpha=0.5)

    out_plot = ROOT / "results" / "cx_analysis" / "06_navigation_benchmark.png"
    fig.tight_layout()
    fig.savefig(out_plot, dpi=150)
    plt.close(fig)
    print(f"\nTrajectory benchmark plot saved to: {out_plot}")


if __name__ == "__main__":
    main()
