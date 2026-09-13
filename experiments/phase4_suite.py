"""
Phase 4: Comprehensive 8-Condition Ablation Suite (A..H) across 10 Independent Seeds.
- Exp A: 180° + Global Plasticity
- Exp B: 360° + Global Plasticity
- Exp C: 180° + Pathway-Specific Plasticity
- Exp D: 360° + Pathway-Specific Plasticity
- Exp E: 180° + Saccades + Pathway Plasticity
- Exp F: 360° + Unplastic
- Exp G: Rewired Null + Pathway Plasticity
- Exp H: Random / Brownian Baseline
"""

import sys
import json
import csv
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
from flymind.environment.world import VirtualArena, ArenaState
from flymind.environment.sensors import SectorVisualSensor, PanoramicCompoundEyeSensor, ActiveExplorationSensor
from flymind.environment.rewards import DenseNavigationReward
from flymind.agent.plastic_cx import PlasticCXAgent
from flymind.agent.fly import RandomFlyAgent
from flymind.utils.behavioral_metrics import compute_behavioral_metrics

GRAPH_PATH = ROOT / "data" / "processed" / "cx_heading_v1.json"
RESULTS_DIR = ROOT / "results" / "phase4"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def create_rewired_graph(original_graph: ConnectomeGraph, n_swaps: int = 20000, seed: int = 42) -> ConnectomeGraph:
    nx_g = original_graph.nx_graph.copy()
    try:
        rewired_nx = nx.directed_edge_swap(nx_g, nswap=n_swaps, max_tries=n_swaps * 10, seed=seed)
    except Exception:
        rewired_nx = nx_g

    rewired_graph = ConnectomeGraph(name=f"{original_graph.name}_rewired")
    for nid, data in original_graph.nx_graph.nodes(data=True):
        rewired_graph.add_neuron(NeuronMetadata(
            body_id=nid, cell_type=data.get("cell_type"), instance=data.get("instance"),
            roi=data.get("roi"), neurotransmitter=data.get("neurotransmitter"),
        ))
    for u, v, data in rewired_nx.edges(data=True):
        rewired_graph.add_connection(SynapticConnection(
            source_id=u, target_id=v, weight=data.get("weight", 1.0),
            neurotransmitter=data.get("neurotransmitter"),
        ))
    return rewired_graph


def run_episode(agent, arena, reward_fn, seed: int, is_training: bool = True) -> Dict[str, Any]:
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

    return {
        "seed": seed,
        "success": 1 if info["target_reached"] else 0,
        "steps": info["step"],
        "total_reward": total_reward,
        "final_distance": info["distance"],
        "trajectory": trajectory,
        "actions": actions,
    }


def main():
    print("=" * 70)
    print("FlyMind Phase 4: Panoramic Vision & Pathway-Specific Plasticity Suite")
    print("=" * 70)

    bio_graph = ConnectomeLoader.load_from_json(GRAPH_PATH)
    rewired_graph = create_rewired_graph(bio_graph, n_swaps=20000, seed=42)

    reward_fn = DenseNavigationReward()
    arena = VirtualArena(width=100.0, height=100.0, min_start_target_dist=25.0)

    # 10 Independent Seeds for training, 100 held-out evaluation seeds
    num_train = 150
    eval_seeds = list(range(90000, 90100))

    conditions = [
        ("Exp A: 180° + Global Plasticity", SectorVisualSensor(), "global", bio_graph),
        ("Exp B: 360° + Global Plasticity", PanoramicCompoundEyeSensor(), "global", bio_graph),
        ("Exp C: 180° + Pathway Plasticity", SectorVisualSensor(), "pathway", bio_graph),
        ("Exp D: 360° + Pathway Plasticity", PanoramicCompoundEyeSensor(), "pathway", bio_graph),
        ("Exp E: 180° + Saccades + Pathway", ActiveExplorationSensor(), "pathway", bio_graph),
        ("Exp F: 360° + Unplastic", PanoramicCompoundEyeSensor(), "none", bio_graph),
        ("Exp G: Rewired Null + Pathway", PanoramicCompoundEyeSensor(), "pathway", rewired_graph),
    ]

    benchmark_summary = []

    for name, sensor, p_mode, grp in conditions:
        print(f"\n--- Running: {name} (10 Independent Training Seeds) ---")
        seed_successes = []
        seed_distances = []
        seed_steps = []
        seed_entropies = []
        seed_diversities = []

        for s_idx in range(10):
            train_base = s_idx * 1000 + 42
            net = NeuralNetwork(graph=grp, neuron_model_cls=RateNeuron, synapse_scale=0.001)
            agent = PlasticCXAgent(network=net, sensor=sensor, plasticity_mode=p_mode, learning_rate=0.002, eligibility_decay=0.85)

            if p_mode != "none":
                for ep in range(num_train):
                    run_episode(agent, arena, reward_fn, seed=train_base + ep, is_training=True)

            # Freeze weights for held-out evaluation
            agent.freeze_weights()
            eval_recs = [run_episode(agent, arena, reward_fn, seed=es, is_training=False) for es in eval_seeds]
            b_metrics = compute_behavioral_metrics(eval_recs)

            succ = np.mean([r["success"] for r in eval_recs]) * 100
            dist = np.mean([r["final_distance"] for r in eval_recs])
            step_succ = [r["steps"] for r in eval_recs if r["success"]]
            mean_step = np.mean(step_succ) if step_succ else 400.0

            seed_successes.append(succ)
            seed_distances.append(dist)
            seed_steps.append(mean_step)
            seed_entropies.append(b_metrics["action_entropy"])
            seed_diversities.append(b_metrics["trajectory_diversity"])

        mean_succ = np.mean(seed_successes)
        sem_succ = np.std(seed_successes) / np.sqrt(10)
        ci_succ = 1.96 * sem_succ

        mean_d = np.mean(seed_distances)
        mean_st = np.mean(seed_steps)
        mean_ent = np.mean(seed_entropies)
        mean_div = np.mean(seed_diversities)

        print(f"  [RESULT] Success: {mean_succ:.2f}% ± {ci_succ:.2f}% (95% CI) | Final Dist: {mean_d:.2f} | Entropy: {mean_ent:.3f}")
        benchmark_summary.append({
            "condition": name,
            "success_mean": mean_succ,
            "success_ci": ci_succ,
            "final_distance": mean_d,
            "steps": mean_st,
            "action_entropy": mean_ent,
            "trajectory_diversity": mean_div,
        })

    # Run Random Baseline (Exp H)
    print("\n--- Running: Exp H: Random / Brownian Baseline ---")
    rand_agent = RandomFlyAgent(seed=999)
    rand_recs = [run_episode(rand_agent, arena, reward_fn, seed=es, is_training=False) for es in eval_seeds]
    rand_b_metrics = compute_behavioral_metrics(rand_recs)
    rand_succ = np.mean([r["success"] for r in rand_recs]) * 100
    rand_d = np.mean([r["final_distance"] for r in rand_recs])
    rand_steps = np.mean([r["steps"] for r in rand_recs if r["success"]]) or 400.0
    benchmark_summary.append({
        "condition": "Exp H: Random Baseline",
        "success_mean": rand_succ,
        "success_ci": 0.0,
        "final_distance": rand_d,
        "steps": rand_steps,
        "action_entropy": rand_b_metrics["action_entropy"],
        "trajectory_diversity": rand_b_metrics["trajectory_diversity"],
    })

    # Save to CSV
    df_res = pd.DataFrame(benchmark_summary)
    df_res.to_csv(RESULTS_DIR / "phase4_ablation_summary.csv", index=False)

    print("\n" + "=" * 85)
    print("PHASE 4 BENCHMARK SUMMARY (10 Independent Seeds x 100 Held-Out Evaluations)")
    print("=" * 85)
    print(f"{'Condition':<35} | {'Success (%) ± 95% CI':<22} | {'Final Dist':<11} | {'Action Entropy':<14}")
    print("-" * 85)
    for r in benchmark_summary:
        succ_str = f"{r['success_mean']:.1f}% ± {r['success_ci']:.1f}%"
        print(f"{r['condition']:<35} | {succ_str:<22} | {r['final_distance']:<11.2f} | {r['action_entropy']:<14.3f}")
    print("=" * 85)

    # Plot Comparison Bar Chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    cond_names = [r["condition"].replace("Exp ", "") for r in benchmark_summary]
    s_means = [r["success_mean"] for r in benchmark_summary]
    s_errs = [r["success_ci"] for r in benchmark_summary]
    d_means = [r["final_distance"] for r in benchmark_summary]

    x = np.arange(len(cond_names))
    ax1.bar(x, s_means, yerr=s_errs, capsize=5, color="#2ECC71", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(cond_names, rotation=35, ha="right", fontsize=8.5)
    ax1.set_ylabel("Held-Out Success Rate (%) ± 95% CI")
    ax1.set_title("Held-Out Navigation Success (10 Seeds x 100 Eval)", fontsize=11, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    ax2.bar(x, d_means, color="#E67E22", alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(cond_names, rotation=35, ha="right", fontsize=8.5)
    ax2.set_ylabel("Final Distance to Beacon (px)")
    ax2.set_title("Mean Final Target Distance (Lower is Closer)", fontsize=11, fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()
    plot_p = RESULTS_DIR / "01_phase4_ablation_comparison.png"
    fig.savefig(plot_p, dpi=150)
    plt.close(fig)
    print(f"\n[SAVED] Benchmark comparison plot saved to: {plot_p}")


if __name__ == "__main__":
    main()
