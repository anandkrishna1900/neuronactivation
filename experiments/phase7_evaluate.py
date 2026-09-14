"""
Phase 7: Frozen Evaluation, Generalization Tests, and Ablation Suite.

Usage:
  # Evaluate a specific checkpoint on test seeds
  python experiments/phase7_evaluate.py --checkpoint results/phase7/checkpoints/<name>.npz

  # Generalization test table
  python experiments/phase7_evaluate.py --checkpoint <path> --generalization

  # Full ablation report (reads all checkpoint files in results/phase7/checkpoints/)
  python experiments/phase7_evaluate.py --ablation-report

  # Compare all modes side-by-side
  python experiments/phase7_evaluate.py --compare-all --episodes 100
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from flymind.brain.network import NeuralNetwork
from flymind.connectome.loader import ConnectomeLoader
from flymind.environment.flappy import FlappyEnvironment
from flymind.environment.flappy_reward import FlappyRewardShaper
from flymind.agent.flappy_agent_phase7 import FlyMindRLAgent, make_rewired_network
from flymind.agent.flappy_agent import HandDesignedFlapAgent, RandomFlapAgent, FixedPeriodFlapAgent
from flymind.agent.flappy_agent_phase6b import ConnectomeMotorReadoutAgent

CONNECTOME_PATH = ROOT / "data" / "processed" / "cx_heading_v1.json"
RESULTS_DIR     = ROOT / "results" / "phase7"
EVAL_DIR        = RESULTS_DIR / "metrics"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

TEST_SEED_OFFSET = 2000

_CACHED_GRAPH = None
def get_graph():
    global _CACHED_GRAPH
    if _CACHED_GRAPH is None:
        _CACHED_GRAPH = ConnectomeLoader.load_from_json(CONNECTOME_PATH)
    return _CACHED_GRAPH

def make_network():
    return NeuralNetwork(get_graph(), synapse_scale=0.01)


# ── Single episode runner ─────────────────────────────────────────────────────
def run_episode(agent, env: FlappyEnvironment, shaper: FlappyRewardShaper,
                seed: int, max_steps: int = 5000) -> Dict:
    state = env.reset(seed=seed)
    agent.reset()
    total_reward = 0.0
    step_count   = 0
    done         = False
    while not done and step_count < max_steps:
        action = agent.act(state)
        next_state, env_reward, done, info = env.step(action)
        shaped_r = shaper.shape(env_reward, info, done)
        total_reward += shaped_r
        state = next_state
        step_count += 1
    return {"score": state.score, "steps": step_count, "reward": total_reward}


# ── Evaluation on N seeds ─────────────────────────────────────────────────────
def evaluate_agent(
    agent,
    n_seeds: int = 50,
    seed_offset: int = TEST_SEED_OFFSET,
    max_steps: int = 5000,
    env_kwargs: Optional[Dict] = None,
    shaper: Optional[FlappyRewardShaper] = None,
) -> Dict:
    if shaper is None:
        shaper = FlappyRewardShaper()
    if env_kwargs is None:
        env_kwargs = {}
    agent.freeze_weights()
    scores, survivals, rewards = [], [], []
    for i in range(n_seeds):
        seed = seed_offset + i
        env = FlappyEnvironment(seed=seed, max_steps=max_steps, **env_kwargs)
        r = run_episode(agent, env, shaper, seed=seed, max_steps=max_steps)
        scores.append(r["score"])
        survivals.append(r["steps"])
        rewards.append(r["reward"])
    return {
        "mean_score":    float(np.mean(scores)),
        "median_score":  float(np.median(scores)),
        "std_score":     float(np.std(scores)),
        "max_score":     int(np.max(scores)),
        "min_score":     int(np.min(scores)),
        "mean_survival": float(np.mean(survivals)),
        "mean_reward":   float(np.mean(rewards)),
        "success_rate":  float(np.mean([s > 0 for s in scores])),
        "n_seeds":       n_seeds,
        "scores":        scores,
    }


# ── Make agent from checkpoint ────────────────────────────────────────────────
def make_flymind_agent(checkpoint_path: Optional[str] = None, seed: int = 0) -> FlyMindRLAgent:
    net = make_network()
    agent = FlyMindRLAgent(
        network=net, sensory_drive=25.0, sub_steps=10,
        learning_rate=0.002, eligibility_decay=0.90,
        plasticity_mode="pathway", motor_temperature=1.5,
        motor_lr=0.01, decoder_mode="stochastic", seed=seed,
    )
    if checkpoint_path and Path(checkpoint_path).exists():
        agent.load_checkpoint(checkpoint_path)
        print(f"  [Loaded] {checkpoint_path}")
    else:
        print(f"  [No checkpoint] Using random initialization")
    return agent


# ── Generalization table ──────────────────────────────────────────────────────
GENERALIZATION_CONDITIONS = [
    ("Standard",        {},                                              "Baseline test distribution"),
    ("LargeGap",        {"gap_size": 220.0},                            "Gap=220 (easier)"),
    ("SmallGap",        {"gap_size": 110.0},                            "Gap=110 (harder)"),
    ("FastScroll",      {"horizontal_speed": 2.5},                      "Speed=2.5 (faster)"),
    ("SlowScroll",      {"horizontal_speed": 0.8},                      "Speed=0.8 (slower)"),
    ("WideSpacing",     {"pipe_spacing": 280.0},                        "Pipes spaced wider"),
    ("NarrowSpacing",   {"pipe_spacing": 150.0},                        "Pipes more frequent"),
    ("LargeGapFast",    {"gap_size": 200.0, "horizontal_speed": 2.2},   "Large gap + fast (compound)"),
    ("SmallGapSlow",    {"gap_size": 120.0, "horizontal_speed": 1.0},   "Small gap + slow"),
]


def run_generalization(checkpoint_path: Optional[str], n_seeds: int = 30) -> None:
    print("\n" + "=" * 80)
    print("PHASE 7 GENERALIZATION TABLE")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_path or 'random init'}")
    print(f"Seeds per condition: {n_seeds} (test seeds {TEST_SEED_OFFSET}–{TEST_SEED_OFFSET+n_seeds-1})")
    print("-" * 80)
    print(f"{'Condition':<20} {'Mean Score':>12} {'Median':>8} {'Std':>8} {'Max':>6} {'SuccRate':>10}  Description")
    print("-" * 80)

    agent = make_flymind_agent(checkpoint_path)
    results = []

    for cond_name, env_kw, desc in GENERALIZATION_CONDITIONS:
        stats = evaluate_agent(agent, n_seeds=n_seeds, env_kwargs=env_kw)
        results.append({
            "condition": cond_name,
            "description": desc,
            **{k: v for k, v in stats.items() if k != "scores"},
        })
        print(
            f"{cond_name:<20} {stats['mean_score']:>12.3f} "
            f"{stats['median_score']:>8.1f} {stats['std_score']:>8.3f} "
            f"{stats['max_score']:>6d} {stats['success_rate']:>10.1%}  {desc}"
        )

    print("-" * 80)
    out_path = EVAL_DIR / "generalization_table.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved] {out_path}")


# ── Ablation comparisons ──────────────────────────────────────────────────────
ABLATION_AGENTS = [
    ("FlyMind_RL",       "FlyMind + local plasticity (Phase 7)",     "flymind_rl"),
    ("Frozen_CX",        "Frozen biological connectome",             "frozen"),
    ("No_Plasticity",    "No plasticity (ablation)",                 "no_plasticity"),
    ("Random_Agent",     "Random 50% flap baseline",                 "random"),
    ("Fixed_Period",     "Fixed-period flap (N=8) baseline",         "fixed"),
    ("Hand_Designed",    "Hand-designed velocity-aware controller",   "hand"),
    ("Phase6B_Agent",    "Phase 6B PEG-gated motor readout",         "phase6b"),
]


def run_comparison(
    checkpoints: Dict[str, str],
    n_seeds: int = 50,
    max_steps: int = 5000,
    seed: int = 42,
) -> None:
    print("\n" + "=" * 90)
    print("PHASE 7 CONTROLLER COMPARISON")
    print("=" * 90)
    print(f"{'Agent':<22} {'Mean Score':>12} {'Median':>8} {'Std':>8} {'Max':>6} {'SuccRate':>10} {'MeanSurv':>10}")
    print("-" * 90)
    shaper = FlappyRewardShaper()

    summary = []
    for name, desc, key in ABLATION_AGENTS:
        ckpt = checkpoints.get(key)

        if key == "random":
            agent = RandomFlapAgent(seed=seed)
        elif key == "fixed":
            agent = FixedPeriodFlapAgent(period=8)
        elif key == "hand":
            agent = HandDesignedFlapAgent()
        elif key == "phase6b":
            net = make_network()
            agent = ConnectomeMotorReadoutAgent(network=net, enable_plasticity=False, seed=seed)
        elif key == "frozen":
            net = make_network()
            agent = FlyMindRLAgent(network=net, plasticity_mode="none", seed=seed)
        else:
            agent = make_flymind_agent(ckpt, seed=seed)

        stats = evaluate_agent(agent, n_seeds=n_seeds, max_steps=max_steps, shaper=shaper)
        summary.append({"agent": name, "description": desc, **{k: v for k, v in stats.items() if k != "scores"}})
        print(
            f"{name:<22} {stats['mean_score']:>12.3f} {stats['median_score']:>8.1f} "
            f"{stats['std_score']:>8.3f} {stats['max_score']:>6d} "
            f"{stats['success_rate']:>10.1%} {stats['mean_survival']:>10.0f}"
        )

    print("=" * 90)
    out_path = EVAL_DIR / "comparison_table.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Saved] {out_path}")


# ── Generate matplotlib learning curves ───────────────────────────────────────
def generate_learning_curves(csv_path: str) -> None:
    """Generate learning curve plots from training CSV log."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Warning] matplotlib not available — skipping figure generation")
        return

    FIGURES_DIR = RESULTS_DIR / "figures"
    FIGURES_DIR.mkdir(exist_ok=True)

    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        print("[Warning] Empty CSV — no learning curve to generate")
        return

    episodes   = [int(r["episode"]) for r in rows]
    scores     = [float(r["score"]) for r in rows]
    steps      = [float(r["steps"]) for r in rows]
    mean_s50   = [float(r["mean_score_50"]) for r in rows if r["mean_score_50"]]
    eval_scores= [float(r["eval_mean_score"]) for r in rows if r.get("eval_mean_score", "") != ""]
    eval_eps   = [int(r["episode"]) for r in rows if r.get("eval_mean_score", "") != ""]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor="#0E1216")
    fig.suptitle("FlyMind Phase 7 — Learning Curves", fontsize=16, color="#F5F8FF", y=0.98)

    panel_bg  = "#111820"
    accent    = "#00D7FF"
    dim       = "#8A99B5"

    def setup_ax(ax, title):
        ax.set_facecolor(panel_bg)
        ax.set_title(title, color=accent, fontsize=11, fontweight="bold", pad=6)
        ax.tick_params(colors=dim)
        ax.spines[["bottom","left","top","right"]].set_edgecolor("#2D3A55")
        ax.grid(True, color="#2D3A55", alpha=0.35)
        ax.xaxis.label.set_color(dim)
        ax.yaxis.label.set_color(dim)

    # Plot 1: Score vs episode
    axes[0, 0].scatter(episodes, scores, s=3, alpha=0.4, color="#3498DB", label="Raw score")
    if len(mean_s50) == len(episodes):
        axes[0, 0].plot(episodes, mean_s50, color=accent, lw=2, label="MA-50")
    setup_ax(axes[0, 0], "Score vs Episode")
    axes[0, 0].legend(facecolor="#111820", edgecolor="#2D3A55", fontsize=8)

    # Plot 2: Survival vs episode
    axes[0, 1].scatter(episodes, steps, s=3, alpha=0.4, color="#2ECC71")
    if len(steps) > 50:
        ma = [np.mean(steps[max(0, i-50):i+1]) for i in range(len(steps))]
        axes[0, 1].plot(episodes, ma, color=accent, lw=2)
    setup_ax(axes[0, 1], "Survival (Steps) vs Episode")

    # Plot 3: Validation scores at checkpoints
    if eval_eps:
        axes[0, 2].plot(eval_eps, eval_scores, color="#E74C3C", lw=2, marker="o", ms=4)
    setup_ax(axes[0, 2], "Validation Score (20 seeds)")

    # Plot 4: Moving average score
    window = 100
    ma100 = [np.mean(scores[max(0, i-window):i+1]) for i in range(len(scores))]
    axes[1, 0].plot(episodes, ma100, color="#9B59B6", lw=2)
    setup_ax(axes[1, 0], f"Score MA-{window}")

    # Plot 5: Score distribution
    axes[1, 1].hist(scores, bins=30, color=accent, alpha=0.7, edgecolor="#0E1216")
    axes[1, 1].axvline(np.mean(scores), color="#E74C3C", lw=2, ls="--", label=f"Mean={np.mean(scores):.2f}")
    axes[1, 1].legend(facecolor="#111820", edgecolor="#2D3A55", fontsize=8)
    setup_ax(axes[1, 1], "Score Distribution")

    # Plot 6: Survival distribution
    axes[1, 2].hist(steps, bins=30, color="#2ECC71", alpha=0.7, edgecolor="#0E1216")
    axes[1, 2].axvline(np.mean(steps), color="#E74C3C", lw=2, ls="--", label=f"Mean={np.mean(steps):.0f}")
    axes[1, 2].legend(facecolor="#111820", edgecolor="#2D3A55", fontsize=8)
    setup_ax(axes[1, 2], "Survival Distribution")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = FIGURES_DIR / "phase7_learning_curves.png"
    plt.savefig(out, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"[Saved] {out}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def build_parser():
    p = argparse.ArgumentParser(description="Phase 7: Frozen evaluation, generalization, and ablation report")
    p.add_argument("--checkpoint", type=str, default=None, help="Path to agent checkpoint")
    p.add_argument("--seeds", type=int, default=50, help="Number of test seeds (default: 50)")
    p.add_argument("--max-steps", type=int, default=5000, help="Max steps per episode")
    p.add_argument("--seed", type=int, default=42, help="Base RNG seed")
    p.add_argument("--generalization", action="store_true", help="Run generalization condition table")
    p.add_argument("--compare-all", action="store_true", help="Compare all agent types")
    p.add_argument("--learning-curves", type=str, default=None, help="CSV training log to plot")
    p.add_argument("--flymind-checkpoint",  type=str, default=None)
    p.add_argument("--rewired-checkpoint",  type=str, default=None)
    return p


def main():
    args = build_parser().parse_args()

    if args.learning_curves:
        generate_learning_curves(args.learning_curves)

    if args.generalization:
        run_generalization(args.checkpoint, n_seeds=args.seeds)

    if args.compare_all:
        checkpoints = {
            "flymind_rl": args.flymind_checkpoint or args.checkpoint,
            "rewired":    args.rewired_checkpoint,
        }
        run_comparison(checkpoints, n_seeds=args.seeds, max_steps=args.max_steps, seed=args.seed)

    if not any([args.generalization, args.compare_all, args.learning_curves]):
        # Default: run frozen eval on checkpoint
        print("\n" + "=" * 70)
        print("PHASE 7 FROZEN EVALUATION")
        print("=" * 70)
        agent = make_flymind_agent(args.checkpoint, seed=args.seed)
        stats = evaluate_agent(agent, n_seeds=args.seeds, seed_offset=TEST_SEED_OFFSET, max_steps=args.max_steps)
        for k, v in stats.items():
            if k != "scores":
                print(f"  {k}: {v}")
        print("=" * 70)


if __name__ == "__main__":
    main()
