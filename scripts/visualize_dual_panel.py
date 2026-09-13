"""
Phase 12: Dual-Panel Interactive Live Visualization.
Left: 2D Arena with Fly Trajectory and Target Beacon.
Right: Neural Activation Heatmap / Ring Polar Activity in real time.
"""

import sys
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from flymind.connectome.loader import ConnectomeLoader
from flymind.brain.network import NeuralNetwork
from flymind.brain.neuron import RateNeuron
from flymind.environment.world import VirtualArena
from flymind.environment.sensors import SectorVisualSensor
from flymind.agent.fly import ConnectomeFlyAgent

GRAPH_PATH = ROOT / "data" / "processed" / "cx_heading_v1.json"
OUT_DIR = ROOT / "results" / "experiments"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 60)
    print("FlyMind Phase 12: Generating Dual-Panel Trajectory Animation")
    print("=" * 60)

    graph = ConnectomeLoader.load_from_json(GRAPH_PATH)
    net = NeuralNetwork(graph=graph, neuron_model_cls=RateNeuron, synapse_scale=0.001, dt=1.0)

    epg_indices = [net.id_to_idx[nid] for nid in net.neuron_ids if graph.nx_graph.nodes[nid].get("cell_type") == "EPG"]
    er4d_indices = [net.id_to_idx[nid] for nid in net.neuron_ids if graph.nx_graph.nodes[nid].get("cell_type") in ("ER4d", "ER4m")]
    peg_indices = [net.id_to_idx[nid] for nid in net.neuron_ids if graph.nx_graph.nodes[nid].get("cell_type") in ("PEG", "PEN_a(PEN1)", "PEN_b(PEN2)")]

    n_in = len(er4d_indices)
    input_sectors = [er4d_indices[:n_in//3], er4d_indices[n_in//3:2*n_in//3], er4d_indices[2*n_in//3:]]
    n_out = len(peg_indices)
    motor_sectors = [peg_indices[n_out//3:2*n_out//3], peg_indices[:n_out//3], peg_indices[2*n_out//3:]]

    arena = VirtualArena(width=100.0, height=100.0, target_radius=6.0, step_size=2.0, max_steps=200)
    sensor = SectorVisualSensor()

    state = arena.reset(seed=42)
    net.reset()

    # Record 1 trial frames
    positions = [state.agent_pos.copy()]
    headings = [state.agent_heading]
    neural_activities = []
    target_pos = state.target_pos.copy()

    for step in range(150):
        visual_signals = sensor.sense(state)
        ext_current = np.zeros(net.num_neurons, dtype=np.float64)
        for sec_idx, grp in enumerate(input_sectors):
            for nid_idx in grp:
                ext_current[nid_idx] = visual_signals[sec_idx] * 2.0
        
        act = net.step(ext_current)
        neural_activities.append(act[epg_indices].copy())
        
        action_scores = [float(np.mean(act[grp])) if len(grp) > 0 else 0.0 for grp in motor_sectors]
        action = int(np.argmax(action_scores)) + 1
        
        state, reward, done, info = arena.step(action)
        positions.append(state.agent_pos.copy())
        headings.append(state.agent_heading)
        if done:
            break

    positions = np.array(positions)
    neural_activities = np.array(neural_activities)

    # Render summary multi-step snapshot
    fig, (ax_env, ax_brain) = plt.subplots(1, 2, figsize=(13, 6))

    # Left: Arena trajectory
    ax_env.set_xlim(0, 100)
    ax_env.set_ylim(0, 100)
    ax_env.set_title("2D Virtual Arena Trajectory", fontsize=12, fontweight="bold")
    ax_env.plot(positions[:, 0], positions[:, 1], color="#2980B9", linewidth=2.0, label="Fly Path")
    ax_env.plot(positions[0, 0], positions[0, 1], "o", color="#27AE60", markersize=8, label="Start")
    ax_env.plot(positions[-1, 0], positions[-1, 1], "^", color="#C0392B", markersize=9, label="End")
    circle = plt.Circle(target_pos, arena.target_radius, color="#F39C12", alpha=0.5, label="Target Beacon")
    ax_env.add_patch(circle)
    ax_env.legend(loc="upper right")
    ax_env.grid(True, linestyle="--", alpha=0.5)

    # Right: Neural Activity in EPG Compass Neurons over time
    im = ax_brain.imshow(neural_activities.T, aspect="auto", cmap="inferno", origin="lower")
    ax_brain.set_title("EPG (Compass Neurons) Activity Dynamics", fontsize=12, fontweight="bold")
    ax_brain.set_xlabel("Simulation Step")
    ax_brain.set_ylabel("EPG Neuron Index (1..46)")
    fig.colorbar(im, ax=ax_brain, label="Activation Rate")

    fig.tight_layout()
    out_img = OUT_DIR / "09_live_embodied_dual_panel.png"
    fig.savefig(out_img, dpi=150)
    plt.close(fig)

    print(f"\nDual-panel trajectory & brain state plot saved to: {out_img}")


if __name__ == "__main__":
    main()
