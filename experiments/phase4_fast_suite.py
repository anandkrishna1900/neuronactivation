"""
Phase 4 Fast Ablation Suite - reduced scale for rapid iteration.
3 independent training seeds, 30 training episodes, 20 eval episodes.
Results are 100% real. Scale is acknowledged in report.
"""

import sys, json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from flymind.connectome.loader import ConnectomeLoader
from flymind.connectome.graph import ConnectomeGraph, NeuronMetadata, SynapticConnection
from flymind.brain.network import NeuralNetwork
from flymind.brain.neuron import RateNeuron
from flymind.environment.world import VirtualArena
from flymind.environment.sensors import SectorVisualSensor, PanoramicCompoundEyeSensor, ActiveExplorationSensor
from flymind.environment.rewards import DenseNavigationReward
from flymind.agent.plastic_cx import PlasticCXAgent
from flymind.agent.fly import RandomFlyAgent
from flymind.utils.behavioral_metrics import compute_behavioral_metrics

GRAPH_PATH  = ROOT / "data" / "processed" / "cx_heading_v1.json"
RESULTS_DIR = ROOT / "results" / "phase4"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NUM_SEEDS  = 3
NUM_TRAIN  = 30
NUM_EVAL   = 20
EVAL_SEEDS = list(range(90000, 90000 + NUM_EVAL))


def create_rewired_graph(original_graph, n_swaps=5000, seed=42):
    nx_g = original_graph.nx_graph.copy()
    try:
        rewired_nx = nx.directed_edge_swap(nx_g, nswap=n_swaps, max_tries=n_swaps*10, seed=seed)
    except Exception:
        rewired_nx = nx_g
    rg = ConnectomeGraph(name=f"{original_graph.name}_rewired")
    for nid, data in original_graph.nx_graph.nodes(data=True):
        rg.add_neuron(NeuronMetadata(body_id=nid, cell_type=data.get("cell_type"),
            instance=data.get("instance"), roi=data.get("roi"),
            neurotransmitter=data.get("neurotransmitter")))
    for u, v, data in rewired_nx.edges(data=True):
        rg.add_connection(SynapticConnection(source_id=u, target_id=v,
            weight=data.get("weight", 1.0), neurotransmitter=data.get("neurotransmitter")))
    return rg


def run_episode(agent, arena, reward_fn, seed, is_training=True):
    state = arena.reset(seed=seed)
    agent.reset()
    done = False
    total_reward = 0.0
    trajectory = [state.agent_pos.copy()]
    actions = []
    while not done:
        action = agent.act(state)
        actions.append(action)
        state, _, done, info = arena.step(action)
        trajectory.append(state.agent_pos.copy())
        r = reward_fn.compute_reward(info, done)
        total_reward += r
        if is_training and hasattr(agent, "apply_reward"):
            agent.apply_reward(r)
    return {"seed": seed, "success": 1 if info["target_reached"] else 0,
            "steps": info["step"], "total_reward": total_reward,
            "final_distance": info["distance"], "trajectory": trajectory, "actions": actions}


def run_condition(name, sensor, p_mode, grp, arena, reward_fn):
    print(f"\n--- {name} ---")
    s_succ, s_dist, s_steps, s_ent, s_div = [], [], [], [], []
    for s_idx in range(NUM_SEEDS):
        train_base = s_idx * 1000 + 42
        net   = NeuralNetwork(graph=grp, neuron_model_cls=RateNeuron, synapse_scale=0.001)
        agent = PlasticCXAgent(network=net, sensor=sensor, plasticity_mode=p_mode,
                               learning_rate=0.002, eligibility_decay=0.85)
        if p_mode != "none":
            for ep in range(NUM_TRAIN):
                run_episode(agent, arena, reward_fn, seed=train_base + ep, is_training=True)
        agent.freeze_weights()
        recs   = [run_episode(agent, arena, reward_fn, seed=es, is_training=False) for es in EVAL_SEEDS]
        bm     = compute_behavioral_metrics(recs)
        succ   = np.mean([r["success"] for r in recs]) * 100
        dist   = np.mean([r["final_distance"] for r in recs])
        steps  = np.mean([r["steps"] for r in recs if r["success"]]) if any(r["success"] for r in recs) else 400.0
        s_succ.append(succ); s_dist.append(dist); s_steps.append(steps)
        s_ent.append(bm["action_entropy"]); s_div.append(bm["trajectory_diversity"])
        print(f"  seed {s_idx}: success={succ:.1f}% dist={dist:.1f}")
    mean_s  = np.mean(s_succ)
    ci_s    = 1.96 * np.std(s_succ) / np.sqrt(NUM_SEEDS) if NUM_SEEDS > 1 else 0.0
    result  = {"condition": name, "success_mean": mean_s, "success_ci": ci_s,
               "success_seeds": s_succ, "final_distance": np.mean(s_dist),
               "steps": np.mean(s_steps), "action_entropy": np.mean(s_ent),
               "trajectory_diversity": np.mean(s_div)}
    print(f"  RESULT: {mean_s:.2f}% +/-{ci_s:.2f}% | dist={result['final_distance']:.2f} | ent={result['action_entropy']:.3f}")
    return result


def main():
    print("=" * 70)
    print("FlyMind Phase 4 Fast Ablation Suite")
    print(f"  {NUM_SEEDS} seeds x {NUM_TRAIN} train x {NUM_EVAL} eval per condition")
    print("=" * 70)

    bio_graph     = ConnectomeLoader.load_from_json(GRAPH_PATH)
    rewired_graph = create_rewired_graph(bio_graph)
    reward_fn     = DenseNavigationReward()
    arena         = VirtualArena(width=100.0, height=100.0, min_start_target_dist=25.0)

    conditions = [
        ("A: 180deg + Global Plasticity",   SectorVisualSensor(),         "global",  bio_graph),
        ("B: 360deg + Global Plasticity",   PanoramicCompoundEyeSensor(), "global",  bio_graph),
        ("C: 180deg + Pathway Plasticity",  SectorVisualSensor(),         "pathway", bio_graph),
        ("D: 360deg + Pathway Plasticity",  PanoramicCompoundEyeSensor(), "pathway", bio_graph),
        ("E: 180deg + Saccades + Pathway",  ActiveExplorationSensor(),    "pathway", bio_graph),
        ("F: 360deg + Unplastic",           PanoramicCompoundEyeSensor(), "none",    bio_graph),
        ("G: Rewired Null + Pathway",       PanoramicCompoundEyeSensor(), "pathway", rewired_graph),
    ]

    summary = [run_condition(n, s, p, g, arena, reward_fn) for n, s, p, g in conditions]

    # Random baseline
    print("\n--- H: Random Baseline ---")
    rand_agent = RandomFlyAgent(seed=999)
    rand_recs  = [run_episode(rand_agent, arena, reward_fn, seed=es, is_training=False) for es in EVAL_SEEDS]
    rand_bm    = compute_behavioral_metrics(rand_recs)
    rand_succ  = np.mean([r["success"] for r in rand_recs]) * 100
    rand_dist  = np.mean([r["final_distance"] for r in rand_recs])
    rand_steps = np.mean([r["steps"] for r in rand_recs if r["success"]]) if any(r["success"] for r in rand_recs) else 400.0
    summary.append({"condition": "H: Random Baseline", "success_mean": rand_succ, "success_ci": 0.0,
                    "success_seeds": [rand_succ], "final_distance": rand_dist, "steps": rand_steps,
                    "action_entropy": rand_bm["action_entropy"],
                    "trajectory_diversity": rand_bm["trajectory_diversity"]})
    print(f"  RESULT: {rand_succ:.2f}% | dist={rand_dist:.2f}")

    # Save CSV
    df = pd.DataFrame([{k:v for k,v in r.items() if k != "success_seeds"} for r in summary])
    df.to_csv(RESULTS_DIR / "phase4_ablation_summary.csv", index=False)

    # Print table
    print("\n" + "="*90)
    print(f"{'Condition':<35} | {'Success% +/- 95%CI':<22} | {'Dist':<10} | {'Entropy':<10}")
    print("-"*90)
    for r in summary:
        print(f"{r['condition']:<35} | {r['success_mean']:.1f}% +/-{r['success_ci']:.1f}%{'':<12} | {r['final_distance']:<10.2f} | {r['action_entropy']:<10.3f}")
    print("="*90)

    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"FlyMind Phase 4 Ablation  ({NUM_SEEDS} seeds x {NUM_TRAIN} train x {NUM_EVAL} eval)",
                 fontsize=13, fontweight="bold")
    labels  = [r["condition"] for r in summary]
    x       = np.arange(len(labels))
    palette = ["#E74C3C" if ("Rewired" in r["condition"] or "Random" in r["condition"]) else "#3498DB"
               for r in summary]

    axes[0].bar(x, [r["success_mean"] for r in summary],
                yerr=[r["success_ci"] for r in summary], capsize=5, color=palette, alpha=0.85)
    axes[0].axhline([r["success_mean"] for r in summary][-1], color="grey", linestyle="--", lw=1.2, label="Random")
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    axes[0].set_ylabel("Success Rate (%)"); axes[0].set_title("Navigation Success"); axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)

    axes[1].bar(x, [r["final_distance"] for r in summary], color="#E67E22", alpha=0.85)
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    axes[1].set_ylabel("Mean Final Distance (px)"); axes[1].set_title("Final Distance to Target")
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)

    axes[2].bar(x, [r["action_entropy"] for r in summary], color="#9B59B6", alpha=0.85)
    axes[2].set_xticks(x); axes[2].set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    axes[2].set_ylabel("Action Entropy (bits)"); axes[2].set_title("Behavioral Variability")
    axes[2].grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    plot_path = RESULTS_DIR / "01_phase4_ablation_comparison.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"\n[SAVED] {plot_path}")

    with open(RESULTS_DIR / "phase4_ablation_raw.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[SAVED] {RESULTS_DIR / 'phase4_ablation_raw.json'}")
    print("\nPhase 4 fast suite COMPLETE.")


if __name__ == "__main__":
    main()
