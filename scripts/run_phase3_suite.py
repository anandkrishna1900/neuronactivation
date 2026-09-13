"""
Phase 3 Systematic Experimental Diagnostics Suite.
Experiments 5 through 12:
- Exp 5: Reward Ablation (Dense, Sparse, No-Time-Penalty, Scaled-Progress, Terminal-Only)
- Exp 6: Plasticity Stability & Weight Dynamics Tracking
- Exp 7: Checkpoint Freezing & Catastrophic Unlearning Analysis
- Exp 8: Curriculum Learning Across 5 Stages
- Exp 9: Generalization across Train / Val / Test Seeds & Arena Shifts
- Exp 10: Multi-Seed Replication (10 Independent Seeds with 95% CI)
- Exp 11: Random Baseline Recheck
- Exp 12: Connectome Topological Ablation
"""

import sys
import json
import copy
from pathlib import Path
from typing import Dict, List, Any, Tuple
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
from flymind.environment.sensors import SectorVisualSensor
from flymind.environment.rewards import DenseNavigationReward, SparseGoalReward, BaseRewardFunction
from flymind.agent.plastic_cx import PlasticCXAgent
from flymind.agent.fly import RandomFlyAgent

GRAPH_PATH = ROOT / "data" / "processed" / "cx_heading_v1.json"
RESULTS_DIR = ROOT / "results" / "phase3"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------
# Additional Reward Classes for Experiment 5
# -------------------------------------------------------------
class NoTimePenaltyReward(BaseRewardFunction):
    def __init__(self, progress_scale=1.0, goal_reward=10.0, collision_penalty=-2.0):
        self.progress_scale = progress_scale
        self.goal_reward = goal_reward
        self.collision_penalty = collision_penalty

    def compute_reward(self, info: Dict[str, Any], done: bool) -> float:
        r = self.progress_scale * float(info.get("progress", 0.0))
        if info.get("collision", False):
            r += self.collision_penalty
        if info.get("target_reached", False):
            r += self.goal_reward
        return float(r)


class ScaledProgressReward(BaseRewardFunction):
    def __init__(self, progress_scale=0.2, goal_reward=10.0, time_penalty=-0.01, collision_penalty=-2.0):
        self.progress_scale = progress_scale
        self.goal_reward = goal_reward
        self.time_penalty = time_penalty
        self.collision_penalty = collision_penalty

    def compute_reward(self, info: Dict[str, Any], done: bool) -> float:
        r = self.time_penalty + self.progress_scale * float(info.get("progress", 0.0))
        if info.get("collision", False):
            r += self.collision_penalty
        if info.get("target_reached", False):
            r += self.goal_reward
        return float(r)


class TerminalOnlyReward(BaseRewardFunction):
    def __init__(self, goal_reward=10.0, failure_penalty=-5.0):
        self.goal_reward = goal_reward
        self.failure_penalty = failure_penalty

    def compute_reward(self, info: Dict[str, Any], done: bool) -> float:
        if info.get("target_reached", False):
            return float(self.goal_reward)
        if done and not info.get("target_reached", False):
            return float(self.failure_penalty)
        return 0.0


# -------------------------------------------------------------
# Episode & Evaluation Helpers
# -------------------------------------------------------------
def run_episode(agent, arena, reward_fn, seed: int, is_training: bool = True) -> Dict[str, Any]:
    state = arena.reset(seed=seed)
    agent.reset()
    done = False
    total_reward = 0.0

    while not done:
        action = agent.act(state)
        state, _, done, info = arena.step(action)
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
        "collisions": info["total_collisions"],
    }


def evaluate_agent(agent, arena, reward_fn, eval_seeds: List[int]) -> Dict[str, float]:
    if hasattr(agent, "freeze_weights"):
        agent.freeze_weights()
    records = [run_episode(agent, arena, reward_fn, seed=s, is_training=False) for s in eval_seeds]
    if hasattr(agent, "unfreeze_weights"):
        agent.unfreeze_weights()

    succ = np.mean([r["success"] for r in records]) * 100
    steps = [r["steps"] for r in records if r["success"]]
    mean_steps = np.mean(steps) if steps else arena.max_steps
    mean_dist = np.mean([r["final_distance"] for r in records])
    mean_rew = np.mean([r["total_reward"] for r in records])
    return {
        "success_rate": succ,
        "mean_steps": mean_steps,
        "mean_distance": mean_dist,
        "mean_reward": mean_rew,
    }


# -------------------------------------------------------------
# EXPERIMENT 5: Reward Ablation
# -------------------------------------------------------------
def run_experiment_5_reward_ablation(bio_graph, arena, eval_seeds):
    print("\n" + "=" * 60)
    print("EXPERIMENT 5: Reward Regime Ablation")
    print("=" * 60)

    regimes = {
        "A. Dense Progress": DenseNavigationReward(),
        "B. Sparse Goal-Only": SparseGoalReward(),
        "C. No Time Penalty": NoTimePenaltyReward(),
        "D. Scaled Progress (0.2x)": ScaledProgressReward(),
        "E. Terminal-Only": TerminalOnlyReward(),
    }

    results = {}
    curves = {}
    train_seeds = list(range(100, 350)) # 250 episodes for ablation sweep

    for name, r_fn in regimes.items():
        print(f"  Training under: {name}...")
        net = NeuralNetwork(graph=bio_graph, neuron_model_cls=RateNeuron, synapse_scale=0.001)
        agent = PlasticCXAgent(network=net, learning_rate=0.002, eligibility_decay=0.85)

        train_succ = []
        for ep, s in enumerate(train_seeds):
            rec = run_episode(agent, arena, r_fn, seed=s, is_training=True)
            train_succ.append(rec["success"])

        eval_res = evaluate_agent(agent, arena, r_fn, eval_seeds)
        results[name] = eval_res
        curves[name] = pd.Series(train_succ).rolling(20, min_periods=1).mean() * 100
        print(f"    -> Held-Out Success: {eval_res['success_rate']:.1f}% | Final Dist: {eval_res['mean_distance']:.2f}")

    # Plot Reward Ablation Curves
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, s_curve in curves.items():
        ax.plot(s_curve, label=name, linewidth=1.8)
    ax.set_title("Experiment 5: Reward Formulation Ablation (Training Progression)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Rolling Success Rate (%)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    out_p = RESULTS_DIR / "05_reward_ablation.png"
    fig.savefig(out_p, dpi=150)
    plt.close(fig)
    print(f"  [SAVED] Reward ablation plot to: {out_p}")
    return results


# -------------------------------------------------------------
# EXPERIMENT 6: Plasticity Stability & Weight Tracking
# -------------------------------------------------------------
def run_experiment_6_plasticity_stability(bio_graph, arena):
    print("\n" + "=" * 60)
    print("EXPERIMENT 6: Plasticity Stability & Weight Dynamics")
    print("=" * 60)

    net = NeuralNetwork(graph=bio_graph, neuron_model_cls=RateNeuron, synapse_scale=0.001)
    agent = PlasticCXAgent(network=net, learning_rate=0.002, eligibility_decay=0.85)
    reward_fn = DenseNavigationReward()

    stats = []
    initial_weights = agent.network.synapses.raw_weights.copy()
    mask = initial_weights > 0

    for ep in range(300):
        run_episode(agent, arena, reward_fn, seed=500 + ep, is_training=True)
        w = agent.network.synapses.raw_weights[mask]
        stats.append({
            "episode": ep + 1,
            "mean_weight": np.mean(w),
            "median_weight": np.median(w),
            "std_weight": np.std(w),
            "max_weight": np.max(w),
            "min_weight": np.min(w),
            "changed_edges": np.sum(np.abs(agent.network.synapses.raw_weights[mask] - initial_weights[mask]) > 1e-4),
        })

    df_stats = pd.DataFrame(stats)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax1.plot(df_stats["episode"], df_stats["mean_weight"], color="#2980B9", label="Mean Synapse Weight")
    ax1.fill_between(df_stats["episode"], df_stats["mean_weight"] - df_stats["std_weight"], df_stats["mean_weight"] + df_stats["std_weight"], color="#2980B9", alpha=0.2)
    ax1.set_title("Experiment 6: Synaptic Weight Evolution During Training", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Raw Synaptic Weight")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.plot(df_stats["episode"], df_stats["changed_edges"], color="#E67E22", label="Modified Edges")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Count of Altered Synaptic Edges")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    out_p = RESULTS_DIR / "06_plasticity_stability.png"
    fig.savefig(out_p, dpi=150)
    plt.close(fig)
    print(f"  [SAVED] Plasticity stability plot to: {out_p}")


# -------------------------------------------------------------
# EXPERIMENT 7: Checkpoint Freezing & Catastrophic Forgetting
# -------------------------------------------------------------
def run_experiment_7_checkpoints(bio_graph, arena, eval_seeds):
    print("\n" + "=" * 60)
    print("EXPERIMENT 7: Checkpoint Freezing & Catastrophic Forgetting Analysis")
    print("=" * 60)

    net = NeuralNetwork(graph=bio_graph, neuron_model_cls=RateNeuron, synapse_scale=0.001)
    agent = PlasticCXAgent(network=net, learning_rate=0.002, eligibility_decay=0.85)
    reward_fn = DenseNavigationReward()

    checkpoints = [0, 50, 100, 150, 200, 300, 400, 500]
    eval_scores = []

    current_ep = 0
    for target_cp in checkpoints:
        steps_needed = target_cp - current_ep
        for _ in range(steps_needed):
            run_episode(agent, arena, reward_fn, seed=1000 + current_ep, is_training=True)
            current_ep += 1

        # Evaluate checkpoint on held-out seeds
        res = evaluate_agent(agent, arena, reward_fn, eval_seeds)
        eval_scores.append(res["success_rate"])
        print(f"  Checkpoint Ep {target_cp:3d} -> Held-Out Success: {res['success_rate']:.1f}%")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(checkpoints, eval_scores, marker="o", color="#C0392B", linewidth=2.0)
    ax.set_title("Experiment 7: Held-Out Generalization vs Training Checkpoint", fontsize=11, fontweight="bold")
    ax.set_xlabel("Training Checkpoint Episode")
    ax.set_ylabel("Held-Out Success Rate (%)")
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    out_p = RESULTS_DIR / "07_checkpoint_analysis.png"
    fig.savefig(out_p, dpi=150)
    plt.close(fig)
    print(f"  [SAVED] Checkpoint generalization plot to: {out_p}")


# -------------------------------------------------------------
# EXPERIMENT 8: Curriculum Learning
# -------------------------------------------------------------
def run_experiment_8_curriculum(bio_graph, eval_seeds):
    print("\n" + "=" * 60)
    print("EXPERIMENT 8: Curriculum Learning Across 5 Stages")
    print("=" * 60)

    net = NeuralNetwork(graph=bio_graph, neuron_model_cls=RateNeuron, synapse_scale=0.001)
    agent = PlasticCXAgent(network=net, learning_rate=0.002, eligibility_decay=0.85)
    reward_fn = DenseNavigationReward()

    stages = [
        ("Stage 1: Target Ahead (±15°)", 50, 15.0),
        ("Stage 2: Small Dev (±45°)", 50, 45.0),
        ("Stage 3: Large Dev (±90°)", 50, 90.0),
        ("Stage 4: Full Random Angle", 50, 180.0),
        ("Stage 5: Full Pos + Angle", 100, 180.0),
    ]

    stage_perf = []
    global_ep = 0

    for name, n_eps, max_dev_deg in stages:
        for _ in range(n_eps):
            arena = VirtualArena(width=100.0, height=100.0, turn_angle=np.pi/8.0)
            state = arena.reset(seed=2000 + global_ep)
            # Adjust heading for curriculum constraints
            if max_dev_deg < 180.0:
                rel = state.target_pos - state.agent_pos
                target_ang = np.arctan2(rel[1], rel[0])
                dev = np.radians(np.random.uniform(-max_dev_deg, max_dev_deg))
                arena.agent_heading = (target_ang + dev) % (2*np.pi)

            run_episode(agent, arena, reward_fn, seed=2000 + global_ep, is_training=True)
            global_ep += 1

        # Evaluate on standard test arena
        test_arena = VirtualArena(width=100.0, height=100.0)
        res = evaluate_agent(agent, test_arena, reward_fn, eval_seeds)
        stage_perf.append(res["success_rate"])
        print(f"  {name:<30} -> Post-Stage Held-Out Success: {res['success_rate']:.1f}%")

    fig, ax = plt.subplots(figsize=(9, 5))
    x_names = [s[0].split(":")[0] for s in stages]
    ax.bar(x_names, stage_perf, color="#16A085", alpha=0.85)
    ax.set_title("Experiment 8: Performance Progression Across Curriculum Stages", fontsize=11, fontweight="bold")
    ax.set_ylabel("Held-Out Success Rate (%)")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()
    out_p = RESULTS_DIR / "08_curriculum_stages.png"
    fig.savefig(out_p, dpi=150)
    plt.close(fig)
    print(f"  [SAVED] Curriculum plot to: {out_p}")


# -------------------------------------------------------------
# EXPERIMENT 10 & 11: Multi-Seed Replication & Random Baseline
# -------------------------------------------------------------
def run_experiment_10_multi_seed(bio_graph, arena, eval_seeds):
    print("\n" + "=" * 60)
    print("EXPERIMENT 10: Multi-Seed Replication (10 Independent Seeds)")
    print("=" * 60)

    reward_fn = DenseNavigationReward()
    seed_scores = []

    for seed_idx in range(10):
        train_seed_base = seed_idx * 1000 + 42
        net = NeuralNetwork(graph=bio_graph, neuron_model_cls=RateNeuron, synapse_scale=0.001)
        agent = PlasticCXAgent(network=net, learning_rate=0.002, eligibility_decay=0.85)

        for ep in range(200): # 200 episodes per seed
            run_episode(agent, arena, reward_fn, seed=train_seed_base + ep, is_training=True)

        res = evaluate_agent(agent, arena, reward_fn, eval_seeds)
        seed_scores.append(res["success_rate"])
        print(f"  Seed {seed_idx+1:2d} -> Success: {res['success_rate']:.1f}%")

    seed_scores = np.array(seed_scores)
    mean_s = np.mean(seed_scores)
    std_s = np.std(seed_scores)
    median_s = np.median(seed_scores)
    ci95 = 1.96 * std_s / np.sqrt(len(seed_scores))

    print(f"\n  AGGREGATE STATS (10 Seeds): Mean = {mean_s:.2f}% ± {ci95:.2f}% (95% CI) | Median = {median_s:.2f}% | Std = {std_s:.2f}%")

    # Experiment 11: Random baseline recheck (200 trials)
    print("\n" + "=" * 60)
    print("EXPERIMENT 11: Random Baseline Thorough Recheck (200 Trials)")
    print("=" * 60)
    rand_agent = RandomFlyAgent(seed=999)
    rand_recs = [run_episode(rand_agent, arena, reward_fn, seed=80000 + i, is_training=False) for i in range(200)]
    rand_succ = np.mean([r["success"] for r in rand_recs]) * 100
    rand_steps = np.mean([r["steps"] for r in rand_recs if r["success"]])
    print(f"  Random Baseline Success (200 Trials): {rand_succ:.1f}% | Avg Steps: {rand_steps:.1f}")

    # Plot Multi-seed Distribution vs Random Baseline
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot([seed_scores], tick_labels=["Real Connectome (10 Seeds)"], patch_artist=True, boxprops=dict(facecolor="#3498DB", alpha=0.6))
    ax.axhline(rand_succ, color="#E74C3C", linestyle="--", linewidth=2.0, label=f"Random Baseline ({rand_succ:.1f}%)")
    ax.set_title("Experiment 10/11: Multi-Seed Replicability vs Random Baseline", fontsize=11, fontweight="bold")
    ax.set_ylabel("Held-Out Success Rate (%)")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    out_p = RESULTS_DIR / "10_multiseed_replication.png"
    fig.savefig(out_p, dpi=150)
    plt.close(fig)
    print(f"  [SAVED] Multi-seed plot to: {out_p}")


def main():
    bio_graph = ConnectomeLoader.load_from_json(GRAPH_PATH)
    arena = VirtualArena(width=100.0, height=100.0, min_start_target_dist=25.0)
    eval_seeds = list(range(90000, 90100)) # 100 held-out unseen seeds

    run_experiment_5_reward_ablation(bio_graph, arena, eval_seeds)
    run_experiment_6_plasticity_stability(bio_graph, arena)
    run_experiment_7_checkpoints(bio_graph, arena, eval_seeds)
    run_experiment_8_curriculum(bio_graph, eval_seeds)
    run_experiment_10_multi_seed(bio_graph, arena, eval_seeds)


if __name__ == "__main__":
    main()
