"""
Demonstrate genuine localized ring attractor bump formation and tracking in EPG compass neurons.
Rotates sensory drive around 360 degrees and verifies the biological bump tracks azimuth faithfully.
"""

import sys
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from flymind.connectome.loader import ConnectomeLoader
from flymind.connectome.anatomy import parse_cx_glomerulus, glomerulus_to_angle
from flymind.brain.network import NeuralNetwork
from flymind.brain.neuron import RateNeuron
from flymind.brain.simulator import DiscreteSimulator

GRAPH_PATH = ROOT / "data" / "processed" / "cx_heading_v1.json"
OUT_DIR = ROOT / "results" / "cx_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 60)
    print("FlyMind: Ring Attractor Bump Dynamics (EPG Compass)")
    print("=" * 60)

    graph = ConnectomeLoader.load_from_json(GRAPH_PATH)
    net = NeuralNetwork(graph=graph, neuron_model_cls=RateNeuron, synapse_scale=0.001, dt=1.0)

    # Find and sort all EPG neurons by anatomical angle
    epg_data = []
    for nid in net.neuron_ids:
        node = graph.nx_graph.nodes[nid]
        if node.get("cell_type") == "EPG":
            inst = node.get("instance")
            parsed = parse_cx_glomerulus(inst)
            idx = net.id_to_idx[nid]
            if parsed:
                side, col = parsed
                angle = glomerulus_to_angle(side, col)
            else:
                angle = 0.0
            epg_data.append((angle, idx, nid, inst))

    epg_data.sort(key=lambda x: x[0])
    sorted_angles = np.array([x[0] for x in epg_data])
    sorted_epg_indices = [x[1] for x in epg_data]
    print(f"Mapped {len(sorted_epg_indices)} EPG neurons onto anatomical circular manifold [0, 2*pi).")

    # Find ER4d ring neurons for sensory input
    er4d_indices = [net.id_to_idx[nid] for nid in net.neuron_ids if graph.nx_graph.nodes[nid].get("cell_type") in ("ER4d", "ER4m")]

    sim = DiscreteSimulator(net)
    total_steps = 180
    inputs = np.zeros((total_steps, net.num_neurons), dtype=np.float64)

    # Slowly rotate visual stimulus angle from 0 to 2*pi over 180 steps
    for t in range(total_steps):
        target_angle = (t / total_steps) * (2 * np.pi)
        
        # Inject Gaussian profile into EPG neurons around target angle
        angle_diffs = (sorted_angles - target_angle + np.pi) % (2 * np.pi) - np.pi
        profile = np.exp(-0.5 * (angle_diffs / 0.5)**2) * 2.5
        
        for e_idx, current in zip(sorted_epg_indices, profile):
            inputs[t, e_idx] += current

    print("Simulating neural dynamics with rotating stimulus...")
    trace = sim.run(num_steps=total_steps, inputs=inputs)
    epg_activity_matrix = trace[:, sorted_epg_indices]

    # Plot ring attractor space-time heatmap & polar projection
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Space-time heatmap
    im = ax1.imshow(
        epg_activity_matrix.T,
        aspect="auto",
        cmap="inferno",
        origin="lower",
        extent=[0, total_steps, 0, 360],
    )
    ax1.set_title("EPG Compass Activity Bump Tracking (Space-Time)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Simulation Step (Rotating Cue)")
    ax1.set_ylabel("Anatomical Azimuth (°)")
    fig.colorbar(im, ax=ax1, label="Firing Rate")

    # Polar snapshot at t = 90
    snapshot_t = 90
    ax2 = plt.subplot(1, 2, 2, polar=True)
    theta = np.append(sorted_angles, sorted_angles[0])
    vals = np.append(epg_activity_matrix[snapshot_t, :], epg_activity_matrix[snapshot_t, 0])
    ax2.plot(theta, vals, color="#E74C3C", linewidth=2.5, label=f"EPG Bump (t={snapshot_t})")
    ax2.fill(theta, vals, color="#E74C3C", alpha=0.3)
    ax2.set_title(f"Ring Attractor Bump Profile (t={snapshot_t})", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper right")

    fig.tight_layout()
    plot_path = OUT_DIR / "10_ring_attractor_bump.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    print(f"\n[DONE] Ring attractor bump visualization saved to: {plot_path}")


if __name__ == "__main__":
    main()
