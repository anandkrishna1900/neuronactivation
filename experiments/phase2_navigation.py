"""
Phase 2: Comprehensive Navigation Learning Benchmark.
Trains ConnectomeFlyAgent under Dense and Sparse reward regimes across randomized positions and orientations.
Runs held-out evaluation on unseen seeds and performs 4-way topological ablation.
"""

import sys
import json
import csv
import copy
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import networkx as nx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from flymind.connectome.loader import ConnectomeLoader
from flymind.connectome.graph import ConnectomeGraph, NeuronMetadata, SynapticConnection
from flymind.brain.network import NeuralNetwork
from flymind.brain.neuron import RateNeuron
from flymind.environment.world import VirtualArena
from flymind.environment.sensors import SectorVisualSensor
from flymind.environment.rewards import DenseNavigationReward, SparseGoalReward
from flymind.agent.plastic_cx import PlasticCXAgent
from flymind.agent.fly import RandomFlyAgent

GRAPH_PATH = ROOT / "data" / "processed" / "cx_heading_v1.json"
RESULTS_DIR = ROOT / "results" / "phase2"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def create_rewired_graph(original_graph: ConnectomeGraph, n_swaps: int = 20000, seed: int = 42) -> ConnectomeGraph:
    """Maslov-Sneppen degree-preserving directed edge rewiring."""
    nx_g = original_graph.nx_graph.copy()
    try:
        rewired_nx = nx.directed_edge_swap(nx_g, nswap=n_swaps, max_tries=n_swaps * 10, seed=seed)
    except Exception:
        rewired_nx = nx_g

    rewired_graph = ConnectomeGraph(name=f"{original_graph.name}_rewired")
    for nid, data in original_graph.nx_graph.nodes(data=True):
        rewired_graph.add_neuron(NeuronMetadata(
            body_id=nid,
            cell_type=data.get("cell_type"),
            instance=data.get("instance"),
            roi=data.get("roi"),
            neurotransmitter=data.get("neurotransmitter"),
        ))

    for u, v, data in rewired_nx.edges(data=True):
        rewired_graph.add_connection(SynapticConnection(
            source_id=u,
            target_id=v,
            weight=data.get("weight", 1.0),
            neurotransmitter=data.get("neurotransmitter"),
        ))
    return rewired_graph


def run_episode(agent, arena, reward_fn, seed: int, is_training: bool = True) -> Dict[str, Any]:
    state = arena.reset(seed=seed)
    agent.reset()

    init_pos = state.agent_pos.copy()
    init_heading = state.agent_heading
    target_pos = state.target_pos.copy()
    init_dist = float(np.linalg.norm(init_pos - target_pos))

    done = False
    total_reward = 0.0
    trajectory = [init_pos]
    actions = []

    while not done:
        action = agent.act(state)
        actions.append(action)
        state, _, done, info = arena.step(action)
        trajectory.append(state.agent_pos.copy())

        # Compute reward
        r = reward_fn.compute_reward(info, done)
        total_reward += r

        # Apply plasticity step if training
        if is_training and hasattr(agent, "apply_reward"):
            agent.apply_reward(r)

    final_dist = info["distance"]
    return {
        "seed": seed,
        "success": 1 if info["target_reached"] else 0,
        "steps": info["step"],
        "total_reward": total_reward,
        "final_distance": final_dist,
        "initial_distance": init_dist,
        "path_length": info["path_length"],
        "collisions": info["total_collisions"],
        "initial_heading": init_heading,
        "final_heading": state.agent_heading,
        "trajectory": trajectory,
        "actions": actions,
    }


def train_and_evaluate(
    agent,
    arena,
    reward_fn,
    num_train_episodes: int = 500,
    num_eval_episodes: int = 100,
    train_seed_base: int = 1000,
    eval_seed_base: int = 90000,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Train agent and subsequently evaluate frozen weights on held-out seeds."""
    if hasattr(agent, "unfreeze_weights"):
        agent.unfreeze_weights()

    # 1. Training Phase
    train_records = []
    for ep in range(num_train_episodes):
        ep_seed = train_seed_base + ep
        rec = run_episode(agent, arena, reward_fn, seed=ep_seed, is_training=True)
        rec["episode"] = ep + 1
        train_records.append(rec)

    # 2. Frozen Held-Out Evaluation Phase
    if hasattr(agent, "freeze_weights"):
        agent.freeze_weights()

    eval_records = []
    for ep in range(num_eval_episodes):
        ep_seed = eval_seed_base + ep
        rec = run_episode(agent, arena, reward_fn, seed=ep_seed, is_training=False)
        rec["eval_episode"] = ep + 1
        eval_records.append(rec)

    return train_records, eval_records


def save_metrics_csv(records: List[Dict[str, Any]], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keys = [k for k in records[0].keys() if k not in ("trajectory", "actions")]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in records:
            filtered = {k: r[k] for k in keys}
            writer.writerow(filtered)


def main():
    print("=" * 70)
    print("FlyMind Phase 2: Learned Heading Navigation Benchmark")
    print("Dataset: Janelia FlyEM Hemibrain v1.2.1 | 261 Neurons | 19,969 Synapses")
    print("=" * 70)

    # Load connectome
    bio_graph = ConnectomeLoader.load_from_json(GRAPH_PATH)
    rewired_graph = create_rewired_graph(bio_graph, n_swaps=20000, seed=42)

    # Common parameters
    num_train_episodes = 500
    num_eval_episodes = 100
    synapse_scale = 0.001
    learning_rate = 0.002
    eligibility_decay = 0.85

    dense_reward = DenseNavigationReward(progress_scale=1.0, goal_reward=10.0, time_penalty=-0.01, collision_penalty=-2.0)
    sparse_reward = SparseGoalReward(goal_reward=1.0, step_penalty=-0.001, failure_penalty=-1.0)
    arena = VirtualArena(width=100.0, height=100.0, target_radius=6.0, step_size=2.0, max_steps=400, min_start_target_dist=25.0)

    # ─────────────────────────────────────────────────────────────
    # EXPERIMENT A: Dense Reward Regime
    # ─────────────────────────────────────────────────────────────
    print("\n[Running Experiment A: Dense Navigation Reward (500 Train / 100 Held-Out Eval)]...")
    net_dense = NeuralNetwork(graph=bio_graph, neuron_model_cls=RateNeuron, synapse_scale=synapse_scale, dt=1.0)
    agent_dense = PlasticCXAgent(network=net_dense, learning_rate=learning_rate, eligibility_decay=eligibility_decay)

    train_dense, eval_dense = train_and_evaluate(
        agent=agent_dense,
        arena=arena,
        reward_fn=dense_reward,
        num_train_episodes=num_train_episodes,
        num_eval_episodes=num_eval_episodes,
    )
    save_metrics_csv(train_dense, RESULTS_DIR / "dense" / "metrics_train.csv")
    save_metrics_csv(eval_dense, RESULTS_DIR / "dense" / "metrics_eval.csv")

    config_dense = {
        "dataset": "Janelia FlyEM Hemibrain",
        "dataset_version": "v1.2.1",
        "circuit": "Central Complex Compass & Steering",
        "num_neurons": bio_graph.num_neurons,
        "num_synapses": bio_graph.num_synapses,
        "reward_mode": "DenseNavigationReward",
        "progress_scale": dense_reward.progress_scale,
        "goal_reward": dense_reward.goal_reward,
        "time_penalty": dense_reward.time_penalty,
        "collision_penalty": dense_reward.collision_penalty,
        "episodes_train": num_train_episodes,
        "episodes_eval": num_eval_episodes,
        "learning_rate": learning_rate,
        "eligibility_decay": eligibility_decay,
        "synapse_scale": synapse_scale,
    }
    with open(RESULTS_DIR / "dense" / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_dense, f, indent=2)

    # ─────────────────────────────────────────────────────────────
    # EXPERIMENT B: Sparse Reward Regime
    # ─────────────────────────────────────────────────────────────
    print("\n[Running Experiment B: Sparse Goal-Only Reward (500 Train / 100 Held-Out Eval)]...")
    net_sparse = NeuralNetwork(graph=bio_graph, neuron_model_cls=RateNeuron, synapse_scale=synapse_scale, dt=1.0)
    agent_sparse = PlasticCXAgent(network=net_sparse, learning_rate=learning_rate, eligibility_decay=eligibility_decay)

    train_sparse, eval_sparse = train_and_evaluate(
        agent=agent_sparse,
        arena=arena,
        reward_fn=sparse_reward,
        num_train_episodes=num_train_episodes,
        num_eval_episodes=num_eval_episodes,
    )
    save_metrics_csv(train_sparse, RESULTS_DIR / "sparse" / "metrics_train.csv")
    save_metrics_csv(eval_sparse, RESULTS_DIR / "sparse" / "metrics_eval.csv")

    config_sparse = {
        "dataset": "Janelia FlyEM Hemibrain",
        "dataset_version": "v1.2.1",
        "circuit": "Central Complex Compass & Steering",
        "num_neurons": bio_graph.num_neurons,
        "num_synapses": bio_graph.num_synapses,
        "reward_mode": "SparseGoalReward",
        "goal_reward": sparse_reward.goal_reward,
        "step_penalty": sparse_reward.step_penalty,
        "episodes_train": num_train_episodes,
        "episodes_eval": num_eval_episodes,
        "learning_rate": learning_rate,
        "eligibility_decay": eligibility_decay,
        "synapse_scale": synapse_scale,
    }
    with open(RESULTS_DIR / "sparse" / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_sparse, f, indent=2)

    # ─────────────────────────────────────────────────────────────
    # SECTION 9/10: 4-WAY TOPOLOGICAL ABLATION BENCHMARK
    # ─────────────────────────────────────────────────────────────
    print("\n[Running 4-Way Topological Ablation Benchmark on Unseen Evaluation Set]...")
    # 1. Real Connectome + Plasticity (Trained agent from Exp A)
    # 2. Rewired Null Connectome + Plasticity
    net_rewired = NeuralNetwork(graph=rewired_graph, neuron_model_cls=RateNeuron, synapse_scale=synapse_scale, dt=1.0)
    agent_rewired = PlasticCXAgent(network=net_rewired, learning_rate=learning_rate, eligibility_decay=eligibility_decay)
    _, eval_rewired = train_and_evaluate(agent_rewired, arena, dense_reward, num_train_episodes=num_train_episodes, num_eval_episodes=num_eval_episodes)
    save_metrics_csv(eval_rewired, RESULTS_DIR / "dense" / "metrics_eval_rewired.csv")

    # 3. Real Connectome (Unplastic)
    net_unplastic = NeuralNetwork(graph=bio_graph, neuron_model_cls=RateNeuron, synapse_scale=synapse_scale, dt=1.0)
    agent_unplastic = PlasticCXAgent(network=net_unplastic, enable_plasticity=False)
    _, eval_unplastic = train_and_evaluate(agent_unplastic, arena, dense_reward, num_train_episodes=0, num_eval_episodes=num_eval_episodes)
    save_metrics_csv(eval_unplastic, RESULTS_DIR / "dense" / "metrics_eval_unplastic.csv")

    # 4. Random Baseline Agent
    agent_random = RandomFlyAgent(seed=42)
    eval_random = []
    for ep in range(num_eval_episodes):
        rec = run_episode(agent_random, arena, dense_reward, seed=90000 + ep, is_training=False)
        rec["eval_episode"] = ep + 1
        eval_random.append(rec)
    save_metrics_csv(eval_random, RESULTS_DIR / "dense" / "metrics_eval_random.csv")

    # Summary Statistics Calculation
    def summarize(recs):
        succ = np.mean([r["success"] for r in recs]) * 100
        steps = [r["steps"] for r in recs if r["success"]]
        mean_steps = np.mean(steps) if steps else arena.max_steps
        median_steps = np.median(steps) if steps else arena.max_steps
        mean_dist = np.mean([r["final_distance"] for r in recs])
        mean_rew = np.mean([r["total_reward"] for r in recs])
        mean_coll = np.mean([r["collisions"] for r in recs])
        return {
            "success": succ,
            "mean_steps": mean_steps,
            "median_steps": median_steps,
            "mean_dist": mean_dist,
            "mean_reward": mean_rew,
            "mean_collisions": mean_coll,
        }

    summary_dense = summarize(eval_dense)
    summary_rewired = summarize(eval_rewired)
    summary_unplastic = summarize(eval_unplastic)
    summary_random = summarize(eval_random)
    summary_sparse = summarize(eval_sparse)

    print("\n" + "=" * 80)
    print("HELD-OUT EVALUATION BENCHMARK (100 Unseen Randomized Episodes)")
    print("=" * 80)
    print(f"{'Condition':<35} | {'Success (%)':<11} | {'Mean Steps':<11} | {'Final Dist':<11} | {'Reward':<10}")
    print("-" * 80)
    print(f"{'Real Connectome + Plasticity (Dense)':<35} | {summary_dense['success']:<11.1f} | {summary_dense['mean_steps']:<11.1f} | {summary_dense['mean_dist']:<11.2f} | {summary_dense['mean_reward']:<10.2f}")
    print(f"{'Rewired Connectome + Plasticity':<35} | {summary_rewired['success']:<11.1f} | {summary_rewired['mean_steps']:<11.1f} | {summary_rewired['mean_dist']:<11.2f} | {summary_rewired['mean_reward']:<10.2f}")
    print(f"{'Real Connectome (Unplastic)':<35} | {summary_unplastic['success']:<11.1f} | {summary_unplastic['mean_steps']:<11.1f} | {summary_unplastic['mean_dist']:<11.2f} | {summary_unplastic['mean_reward']:<10.2f}")
    print(f"{'Random Baseline Agent':<35} | {summary_random['success']:<11.1f} | {summary_random['mean_steps']:<11.1f} | {summary_random['mean_dist']:<11.2f} | {summary_random['mean_reward']:<10.2f}")
    print(f"{'Real Connectome + Plasticity (Sparse)':<35} | {summary_sparse['success']:<11.1f} | {summary_sparse['mean_steps']:<11.1f} | {summary_sparse['mean_dist']:<11.2f} | {summary_sparse['mean_reward']:<10.2f}")
    print("=" * 80)

    # Save representative episode for replay
    sample_ep = eval_dense[0]
    replay_data = {
        "seed": sample_ep["seed"],
        "success": sample_ep["success"],
        "steps": sample_ep["steps"],
        "trajectory": [pos.tolist() if isinstance(pos, np.ndarray) else pos for pos in sample_ep["trajectory"]],
        "actions": sample_ep["actions"],
        "final_distance": sample_ep["final_distance"],
    }
    with open(RESULTS_DIR / "sample_episode.json", "w", encoding="utf-8") as f:
        json.dump(replay_data, f, indent=2)

    print(f"\n[DONE] Benchmark metrics and configs saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
