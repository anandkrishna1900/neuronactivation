"""Phase 5 Step 10: 2D Closed-Loop Navigation Benchmark with Motor Signal Calibration."""
import sys
from pathlib import Path
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
from flymind.agent.plastic_cx import PlasticCXAgent
from flymind.agent.fly import RandomFlyAgent
from flymind.environment.world import VirtualArena
from flymind.environment.sensors import PanoramicCompoundEyeSensor
from flymind.environment.rewards import DenseNavigationReward
from flymind.utils.behavioral_metrics import compute_behavioral_metrics

GRAPH_PATH  = ROOT / "data" / "processed" / "cx_heading_v1.json"
RESULTS_DIR = ROOT / "results" / "phase5"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NUM_SEEDS   = 5
NUM_TRAIN   = 50
NUM_EVAL    = 30
EVAL_SEEDS  = list(range(93000, 93000 + NUM_EVAL))

def get_scale_and_temp():
    sp = RESULTS_DIR / "recommended_scale.txt"
    tp = RESULTS_DIR / "recommended_temperature.txt"
    sc = float(sp.read_text()) if sp.exists() else 0.005
    T  = float(tp.read_text()) if tp.exists() else 0.2
    return sc, T

def create_rewired(bio, n_swaps=5000, seed=42):
    nx_g = bio.nx_graph.copy()
    try:
        rw = nx.directed_edge_swap(nx_g, nswap=n_swaps, max_tries=n_swaps*10, seed=seed)
    except Exception:
        rw = nx_g
    rg = ConnectomeGraph(name="rewired")
    for nid, d in bio.nx_graph.nodes(data=True):
        rg.add_neuron(NeuronMetadata(body_id=nid, cell_type=d.get("cell_type"),
            instance=d.get("instance"), roi=d.get("roi"), neurotransmitter=d.get("neurotransmitter")))
    for u, v, d in rw.edges(data=True):
        rg.add_connection(SynapticConnection(source_id=u, target_id=v,
            weight=d.get("weight", 1.0), neurotransmitter=d.get("neurotransmitter")))
    return rg

def run_episode(agent, arena, reward_fn, seed, train=True):
    state = arena.reset(seed=seed)
    agent.reset()
    done = False; total_r = 0.0; trajectory = [state.agent_pos.copy()]; actions = []
    while not done:
        action = agent.act(state)
        actions.append(action)
        state, _, done, info = arena.step(action)
        trajectory.append(state.agent_pos.copy())
        r = reward_fn.compute_reward(info, done)
        total_r += r
        if train and hasattr(agent, "apply_reward"):
            agent.apply_reward(r)
    return dict(seed=seed, success=1 if info["target_reached"] else 0,
                steps=info["step"], total_reward=total_r,
                final_distance=info["distance"], trajectory=trajectory, actions=actions)

def run_2d_condition(label, agent_factory, graph, scale, T, arena, reward_fn):
    all_eval_recs = []
    learning_curves = np.zeros((NUM_SEEDS, NUM_TRAIN))
    
    for s in range(NUM_SEEDS):
        net = NeuralNetwork(graph=graph, neuron_model_cls=RateNeuron, synapse_scale=scale)
        agent = agent_factory(net, T)
        for ep in range(NUM_TRAIN):
            rec = run_episode(agent, arena, reward_fn, seed=s*1000 + 42 + ep, train=True)
            learning_curves[s, ep] = rec["success"]
        
        if hasattr(agent, "freeze_weights"):
            agent.freeze_weights()
            
        for es in EVAL_SEEDS:
            rec = run_episode(agent, arena, reward_fn, seed=es, train=False)
            all_eval_recs.append(rec)
            
    bm = compute_behavioral_metrics(all_eval_recs)
    succ_rate = np.mean([r["success"] for r in all_eval_recs]) * 100
    mean_dist = np.mean([r["final_distance"] for r in all_eval_recs])
    mean_steps = np.mean([r["steps"] for r in all_eval_recs])
    
    path_effs = []
    for rec in all_eval_recs:
        traj = np.array(rec["trajectory"])
        path_len = float(np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))) if len(traj) > 1 else 0.0
        straight = float(np.linalg.norm(traj[-1] - traj[0])) if len(traj) > 1 else 0.0
        path_effs.append(straight / (path_len + 1e-6))
    mean_eff = float(np.mean(path_effs))
    
    return dict(
        label=label,
        success_rate=succ_rate,
        final_distance=mean_dist,
        mean_steps=mean_steps,
        path_efficiency=mean_eff,
        action_entropy=bm["action_entropy"],
        forward_ratio=bm.get("action_distribution", {}).get("FORWARD", 0.0),
        turn_ratio=bm.get("action_distribution", {}).get("TURN_LEFT", 0.0) + bm.get("action_distribution", {}).get("TURN_RIGHT", 0.0),
        learning_curve=learning_curves.mean(axis=0).tolist()
    )

def main():
    scale, T = get_scale_and_temp()
    print("=" * 60)
    print(f"Phase 5 Step 10: 2D Closed-Loop Navigation Benchmark")
    print(f"Calibrated Parameters: scale={scale}, temperature={T}")
    print("=" * 60)
    
    bio = ConnectomeLoader.load_from_json(GRAPH_PATH)
    rewired = create_rewired(bio)
    arena = VirtualArena(width=100.0, height=100.0, min_start_target_dist=25.0)
    reward_fn = DenseNavigationReward()
    sensor = PanoramicCompoundEyeSensor()
    
    conditions = [
        ("Bio + Pathway Plasticity (Calibrated)",
         lambda net, t: PlasticCXAgent(net, sensor, plasticity_mode="pathway",
                                       learning_rate=0.002, eligibility_decay=0.85,
                                       decoder_mode="hemispheric", temperature=t), bio),
        ("Bio + Unplastic Baseline (Hemispheric)",
         lambda net, t: PlasticCXAgent(net, sensor, plasticity_mode="none",
                                       decoder_mode="hemispheric", temperature=t), bio),
        ("Rewired Null + Pathway Plasticity",
         lambda net, t: PlasticCXAgent(net, sensor, plasticity_mode="pathway",
                                       learning_rate=0.002, eligibility_decay=0.85,
                                       decoder_mode="hemispheric", temperature=t), rewired),
    ]
    
    rows = []
    for label, factory, graph in conditions:
        print(f"  Running: {label} ...", end=" ", flush=True)
        r = run_2d_condition(label, factory, graph, scale, T, arena, reward_fn)
        rows.append(r)
        print(f"succ={r['success_rate']:.1f}% dist={r['final_distance']:.2f} eff={r['path_efficiency']:.3f} ent={r['action_entropy']:.3f}")
        
    # Random Baseline
    print("  Running: Random Exploration Baseline ...", end=" ", flush=True)
    all_rand_recs = []
    for s in range(NUM_SEEDS):
        rand_agent = RandomFlyAgent(seed=s*1000+42)
        for es in EVAL_SEEDS:
            rec = run_episode(rand_agent, arena, reward_fn, seed=es, train=False)
            all_rand_recs.append(rec)
    bm_rand = compute_behavioral_metrics(all_rand_recs)
    path_effs_rand = []
    for rec in all_rand_recs:
        traj = np.array(rec["trajectory"])
        path_len = float(np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))) if len(traj) > 1 else 0.0
        straight = float(np.linalg.norm(traj[-1] - traj[0])) if len(traj) > 1 else 0.0
        path_effs_rand.append(straight / (path_len + 1e-6))
    
    rows.append(dict(
        label="Random Baseline",
        success_rate=float(np.mean([r["success"] for r in all_rand_recs])) * 100,
        final_distance=float(np.mean([r["final_distance"] for r in all_rand_recs])),
        mean_steps=float(np.mean([r["steps"] for r in all_rand_recs])),
        path_efficiency=float(np.mean(path_effs_rand)),
        action_entropy=bm_rand["action_entropy"],
        forward_ratio=bm_rand.get("action_distribution", {}).get("FORWARD", 0.0),
        turn_ratio=bm_rand.get("action_distribution", {}).get("TURN_LEFT", 0.0) + bm_rand.get("action_distribution", {}).get("TURN_RIGHT", 0.0),
        learning_curve=[0.0] * NUM_TRAIN
    ))
    print(f"succ={rows[-1]['success_rate']:.1f}% dist={rows[-1]['final_distance']:.2f} eff={rows[-1]['path_efficiency']:.3f} ent={rows[-1]['action_entropy']:.3f}")
    
    # Save CSV
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "learning_curve"} for r in rows])
    df.to_csv(RESULTS_DIR / "navigation_2d_benchmark.csv", index=False)
    print("[SAVED] navigation_2d_benchmark.csv")
    print("\n" + df[["label", "success_rate", "final_distance", "path_efficiency", "action_entropy"]].to_string(index=False))
    
    # Plot benchmark results
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"FlyMind Phase 5 -- 2D Navigation Benchmark (scale={scale}, T={T})", fontsize=12, fontweight="bold")
    
    labels = [r["label"] for r in rows]
    succs = [r["success_rate"] for r in rows]
    dists = [r["final_distance"] for r in rows]
    effs = [r["path_efficiency"] for r in rows]
    colors = ["#2ECC71", "#3498DB", "#E67E22", "#95A5A6"]
    x = np.arange(len(labels))
    
    axes[0].bar(x, succs, color=colors, alpha=0.85, edgecolor="k")
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    axes[0].set_ylabel("Success Rate (%)"); axes[0].set_title("Navigation Success")
    axes[0].grid(axis="y", alpha=0.3)
    
    axes[1].bar(x, dists, color=colors, alpha=0.85, edgecolor="k")
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    axes[1].set_ylabel("Final Distance to Target"); axes[1].set_title("Mean Distance")
    axes[1].grid(axis="y", alpha=0.3)
    
    axes[2].bar(x, effs, color=colors, alpha=0.85, edgecolor="k")
    axes[2].set_xticks(x); axes[2].set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    axes[2].set_ylabel("Path Efficiency (Straight / Total)"); axes[2].set_title("Path Efficiency")
    axes[2].grid(axis="y", alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "navigation_2d_benchmark.png", dpi=150)
    plt.close(fig)
    print("[SAVED] navigation_2d_benchmark.png")
    print("Phase 5 Step 10 COMPLETE.")

if __name__ == "__main__":
    main()
