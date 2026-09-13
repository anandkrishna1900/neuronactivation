
import sys
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
from flymind.environment.world import VirtualArena
from flymind.environment.sensors import PanoramicCompoundEyeSensor
from flymind.environment.rewards import DenseNavigationReward
from flymind.agent.plastic_cx import PlasticCXAgent
from flymind.agent.fly import RandomFlyAgent
from flymind.utils.behavioral_metrics import compute_behavioral_metrics

GRAPH_PATH  = ROOT / "data" / "processed" / "cx_heading_v1.json"
RESULTS_DIR = ROOT / "results" / "phase5"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TEMPERATURES  = [0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
NUM_SEEDS     = 3
NUM_TRAIN     = 30
NUM_EVAL      = 20
EVAL_SEEDS    = list(range(91000, 91000 + NUM_EVAL))

def get_scale():
    p = RESULTS_DIR / "recommended_scale.txt"
    return float(p.read_text().strip()) if p.exists() else 0.005

def run_episode(agent, arena, reward_fn, seed, train=True):
    state = arena.reset(seed=seed)
    agent.reset()
    done = False; total_r = 0.0; trajectory = [state.agent_pos.copy()]; actions = []
    while not done:
        action = agent.act(state); actions.append(action)
        state, _, done, info = arena.step(action)
        trajectory.append(state.agent_pos.copy())
        r = reward_fn.compute_reward(info, done); total_r += r
        if train and hasattr(agent, "apply_reward"): agent.apply_reward(r)
    return dict(seed=seed, success=1 if info["target_reached"] else 0,
                steps=info["step"], total_reward=total_r,
                final_distance=info["distance"], trajectory=trajectory, actions=actions)

def run_temp(T, bio, scale, arena, reward_fn):
    sensor = PanoramicCompoundEyeSensor()
    succs, dists, entropies, effs = [], [], [], []
    for s in range(NUM_SEEDS):
        net   = NeuralNetwork(graph=bio, neuron_model_cls=RateNeuron, synapse_scale=scale)
        agent = PlasticCXAgent(network=net, sensor=sensor, plasticity_mode="pathway",
                               learning_rate=0.002, eligibility_decay=0.85,
                               decoder_mode="hemispheric", temperature=T)
        for ep in range(NUM_TRAIN):
            run_episode(agent, arena, reward_fn, seed=s*1000+42+ep, train=True)
        agent.freeze_weights()
        recs = [run_episode(agent, arena, reward_fn, seed=es, train=False) for es in EVAL_SEEDS]
        bm   = compute_behavioral_metrics(recs)
        succ = np.mean([r["success"] for r in recs]) * 100
        dist = np.mean([r["final_distance"] for r in recs])
        path_effs = []
        for rec in recs:
            traj = np.array(rec["trajectory"])
            path_len = float(np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))) if len(traj) > 1 else 0.0
            straight = float(np.linalg.norm(traj[-1] - traj[0])) if len(traj) > 1 else 0.0
            path_effs.append(straight / (path_len + 1e-6))
        succs.append(succ); dists.append(dist)
        entropies.append(bm["action_entropy"]); effs.append(float(np.mean(path_effs)))
    return dict(temperature=T,
                success_mean=float(np.mean(succs)), success_ci=1.96*float(np.std(succs))/np.sqrt(NUM_SEEDS),
                final_distance=float(np.mean(dists)), action_entropy=float(np.mean(entropies)),
                path_efficiency=float(np.mean(effs)))

def main():
    scale = get_scale()
    print("Phase 5 Step 5: Softmax Temperature Sweep (scale=" + str(scale) + ")")
    bio = ConnectomeLoader.load_from_json(GRAPH_PATH)
    arena = VirtualArena(width=100.0, height=100.0, min_start_target_dist=25.0)
    reward_fn = DenseNavigationReward()
    rows = []
    for T in TEMPERATURES:
        print("  T=" + str(T) + " ...", end=" ", flush=True)
        r = run_temp(T, bio, scale, arena, reward_fn)
        rows.append(r)
        print("succ=" + str(round(r["success_mean"],1)) + "% ent=" + str(round(r["action_entropy"],3)))
    # Argmax baseline
    print("  T=argmax ...", end=" ", flush=True)
    sensor = PanoramicCompoundEyeSensor()
    succs, entropies = [], []
    for s in range(NUM_SEEDS):
        net   = NeuralNetwork(graph=bio, neuron_model_cls=RateNeuron, synapse_scale=scale)
        agent = PlasticCXAgent(network=net, sensor=sensor, plasticity_mode="pathway",
                               learning_rate=0.002, eligibility_decay=0.85,
                               decoder_mode="argmax")
        for ep in range(NUM_TRAIN):
            run_episode(agent, arena, reward_fn, seed=s*1000+42+ep, train=True)
        agent.freeze_weights()
        recs = [run_episode(agent, arena, reward_fn, seed=es, train=False) for es in EVAL_SEEDS]
        bm   = compute_behavioral_metrics(recs)
        succs.append(np.mean([r["success"] for r in recs]) * 100)
        entropies.append(bm["action_entropy"])
    rows.append(dict(temperature=-1, success_mean=float(np.mean(succs)),
                     success_ci=0.0, final_distance=0.0,
                     action_entropy=float(np.mean(entropies)), path_efficiency=0.0))
    print("succ=" + str(round(rows[-1]["success_mean"],1)) + "% ent=" + str(round(rows[-1]["action_entropy"],3)))

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "softmax_sweep.csv", index=False)
    print("[SAVED] softmax_sweep.csv")

    # Best T by success, break ties by entropy
    df_sm = df[df["temperature"] > 0].copy()
    if df_sm["success_mean"].max() > 0:
        best_T = float(df_sm.loc[df_sm["success_mean"].idxmax(), "temperature"])
    else:
        best_T = float(df_sm.loc[df_sm["action_entropy"].sub(1.0).abs().idxmin(), "temperature"])
    with open(RESULTS_DIR / "recommended_temperature.txt", "w") as f: f.write(str(best_T))
    print("Recommended temperature = " + str(best_T))

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"FlyMind Phase 5 -- Softmax Temperature Sweep (scale={scale})", fontsize=12, fontweight="bold")
    df_plot = df[df["temperature"] > 0]
    Ts = df_plot["temperature"].tolist()
    for ax, col, lbl, c in zip(axes,
                                ["success_mean", "action_entropy", "path_efficiency"],
                                ["Success Rate (%)", "Action Entropy (bits)", "Path Efficiency"],
                                ["#2ECC71", "#3498DB", "#E67E22"]):
        ax.plot(Ts, df_plot[col], "o-", color=c, lw=2, ms=8)
        if col == "success_mean":
            ax.errorbar(Ts, df_plot["success_mean"], yerr=df_plot["success_ci"], fmt="none", color=c, capsize=5)
            ax.axhline(rows[-1]["success_mean"], color="grey", ls="--", lw=1.2, label="argmax")
            ax.legend(fontsize=9)
        if col == "action_entropy":
            ax.axhline(np.log2(3), color="red", ls="--", lw=1, label="max entropy")
            ax.axhline(rows[-1]["action_entropy"], color="grey", ls="--", lw=1.2, label="argmax")
            ax.legend(fontsize=9)
        ax.axvline(best_T, color="gold", ls="--", lw=1.5, label="best T")
        ax.set_xlabel("Temperature T"); ax.set_ylabel(lbl)
        ax.set_title(lbl + " vs Temperature")
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "softmax_sweep.png", dpi=150); plt.close(fig)
    print("[SAVED] softmax_sweep.png")
    print("Step 5 COMPLETE.")

if __name__ == "__main__":
    main()
