"""
Phase 5: Run the real CX heading circuit through the neural simulator.
Connects real Hemibrain topology to the FlyMind simulation engine.

Usage:
    python scripts/run_cx_simulation.py
"""

import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from flymind.connectome.loader import ConnectomeLoader
from flymind.brain.network import NeuralNetwork
from flymind.brain.neuron import RateNeuron
from flymind.brain.simulator import DiscreteSimulator

GRAPH_PATH = ROOT / "data" / "processed" / "cx_heading_v1.json"
OUT_DIR = ROOT / "results" / "cx_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 60)
    print("FlyMind Phase 5: CX Neural Simulation (Rate Neurons)")
    print("Connectome: Hemibrain v1.2.1, Scheffer et al. 2020")
    print("=" * 60)

    # ── Load the real connectome graph ────────────────────────────
    print("\n[1] Loading CX heading circuit...", flush=True)
    graph = ConnectomeLoader.load_from_json(GRAPH_PATH)
    print(f"    {graph.num_neurons} neurons, {graph.num_synapses} synapses", flush=True)

    # ── Build neural network ──────────────────────────────────────
    print("\n[2] Building NeuralNetwork from connectome topology...", flush=True)
    net = NeuralNetwork(
        graph=graph,
        neuron_model_cls=RateNeuron,
        synapse_scale=0.005,   # scales raw synapse counts to conductance
        dt=1.0,
    )
    print(f"    Network: {net.num_neurons} neurons", flush=True)

    # ── Identify neuron index groups by cell type ─────────────────
    epg_indices   = [net.id_to_idx[nid] for nid in net.neuron_ids
                     if graph.nx_graph.nodes[nid].get("cell_type") == "EPG"]
    pen_a_indices = [net.id_to_idx[nid] for nid in net.neuron_ids
                     if graph.nx_graph.nodes[nid].get("cell_type") == "PEN_a(PEN1)"]
    pen_b_indices = [net.id_to_idx[nid] for nid in net.neuron_ids
                     if graph.nx_graph.nodes[nid].get("cell_type") == "PEN_b(PEN2)"]
    peg_indices   = [net.id_to_idx[nid] for nid in net.neuron_ids
                     if graph.nx_graph.nodes[nid].get("cell_type") == "PEG"]
    er4d_indices  = [net.id_to_idx[nid] for nid in net.neuron_ids
                     if graph.nx_graph.nodes[nid].get("cell_type") == "ER4d"]

    print(f"    EPG (compass)        : {len(epg_indices)} neurons")
    print(f"    PEN_a (left integr.) : {len(pen_a_indices)} neurons")
    print(f"    PEN_b (right integr.): {len(pen_b_indices)} neurons")
    print(f"    PEG (steering out)   : {len(peg_indices)} neurons")
    print(f"    ER4d (visual input)  : {len(er4d_indices)} neurons")

    # ── Experiment 1: Spontaneous activity (no input) ─────────────
    print("\n[3] Experiment 1: Spontaneous ring dynamics (50 steps, no input)")
    sim = DiscreteSimulator(net)
    net.reset()
    trace_spontaneous = sim.run(num_steps=50)
    epg_mean = trace_spontaneous[:, epg_indices].mean(axis=1)
    print(f"    EPG mean activity: min={epg_mean.min():.4f}, max={epg_mean.max():.4f}")

    # ── Experiment 2: Inject visual signal into ER4d ──────────────
    print("\n[4] Experiment 2: Visual input via ER4d ring neurons (100 steps)")
    net.reset()
    inputs = np.zeros((100, net.num_neurons), dtype=np.float64)

    # Simulate a "target to the left" visual signal:
    # Inject excitatory current into left-sector ER4d neurons (first half)
    left_er4d = er4d_indices[:len(er4d_indices) // 2]
    for t in range(100):
        inputs[t, left_er4d] = 3.0  # sustained visual drive

    trace_visual = sim.run(num_steps=100, inputs=inputs)

    epg_trace  = trace_visual[:, epg_indices].mean(axis=1)
    peg_trace  = trace_visual[:, peg_indices].mean(axis=1)
    pen_a_trace = trace_visual[:, pen_a_indices].mean(axis=1)
    pen_b_trace = trace_visual[:, pen_b_indices].mean(axis=1)

    print(f"    EPG mean activity  (t=50-99): {epg_trace[50:].mean():.4f}")
    print(f"    PEG output activity (t=50-99): {peg_trace[50:].mean():.4f}")
    print(f"    PEN_a (left)  (t=50-99): {pen_a_trace[50:].mean():.4f}")
    print(f"    PEN_b (right) (t=50-99): {pen_b_trace[50:].mean():.4f}")

    # ── Determine steering output ─────────────────────────────────
    peg_final = trace_visual[-1, peg_indices]
    n_peg = len(peg_indices)
    third = n_peg // 3

    left_pop   = peg_final[:third].mean() if third > 0 else 0
    center_pop = peg_final[third:2*third].mean() if third > 0 else 0
    right_pop  = peg_final[2*third:].mean() if 2*third < n_peg else 0

    action_map = {0: "TURN LEFT", 1: "FORWARD", 2: "TURN RIGHT"}
    action = int(np.argmax([left_pop, center_pop, right_pop]))
    print(f"\n    PEG population activity: left={left_pop:.4f}, center={center_pop:.4f}, right={right_pop:.4f}")
    print(f"    Decoded motor action  : {action_map[action]}")

    # ── Save activity traces as plot ──────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    ax_epg, ax_peg, ax_pena, ax_penb = axes

    ax_epg.plot(epg_trace, color="#4C9BE8", linewidth=1.5, label="EPG mean activity")
    ax_epg.set_ylabel("EPG\n(Compass)", fontsize=9)
    ax_epg.legend(loc="upper right", fontsize=8)

    ax_peg.plot(peg_trace, color="#7BE87B", linewidth=1.5, label="PEG mean activity")
    ax_peg.set_ylabel("PEG\n(Steering out)", fontsize=9)
    ax_peg.legend(loc="upper right", fontsize=8)

    ax_pena.plot(pen_a_trace, color="#E87B4C", linewidth=1.5, label="PEN_a (left)")
    ax_pena.set_ylabel("PEN_a\n(Left integr.)", fontsize=9)
    ax_pena.legend(loc="upper right", fontsize=8)

    ax_penb.plot(pen_b_trace, color="#E8C04C", linewidth=1.5, label="PEN_b (right)")
    ax_penb.set_ylabel("PEN_b\n(Right integr.)", fontsize=9)
    ax_penb.set_xlabel("Simulation Timestep")
    ax_penb.legend(loc="upper right", fontsize=8)

    axes[0].set_title(
        "CX Heading Circuit — Neural Activity Traces\n"
        "Visual input: left ER4d injection (t=0 to t=99)\n"
        "Real Hemibrain v1.2.1 synaptic topology",
        fontsize=11
    )
    fig.tight_layout()
    out_path = OUT_DIR / "05_simulation_activity.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\n    Activity trace saved: {out_path}")

    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("The real Drosophila CX connectome is driving neural dynamics.")
    print("=" * 60)


if __name__ == "__main__":
    main()
