"""
Phase 2: Generate learning curves, statistical confidence intervals, and ablation comparison plots.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

RESULTS_DIR = ROOT / "results" / "phase2"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_learning_curves():
    # Load training logs
    dense_train = pd.read_csv(RESULTS_DIR / "dense" / "metrics_train.csv")
    sparse_train = pd.read_csv(RESULTS_DIR / "sparse" / "metrics_train.csv")

    window = 25
    dense_train["succ_smooth"] = dense_train["success"].rolling(window, min_periods=1).mean() * 100
    dense_train["rew_smooth"] = dense_train["total_reward"].rolling(window, min_periods=1).mean()
    dense_train["steps_smooth"] = dense_train["steps"].rolling(window, min_periods=1).mean()
    dense_train["dist_smooth"] = dense_train["final_distance"].rolling(window, min_periods=1).mean()

    sparse_train["succ_smooth"] = sparse_train["success"].rolling(window, min_periods=1).mean() * 100
    sparse_train["rew_smooth"] = sparse_train["total_reward"].rolling(window, min_periods=1).mean()

    # 1. Success Rate vs Episode
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(dense_train["episode"], dense_train["succ_smooth"], color="#27AE60", linewidth=2.0, label="Dense Progress Reward")
    ax.plot(sparse_train["episode"], sparse_train["succ_smooth"], color="#E67E22", linewidth=2.0, linestyle="--", label="Sparse Goal-Only Reward")
    ax.set_title("Navigation Success Rate vs Training Episode (500 Episodes)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Rolling Success Rate (%)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "01_success_vs_episode.png", dpi=150)
    plt.close(fig)

    # 2. Episodic Return vs Episode
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax1.plot(dense_train["episode"], dense_train["rew_smooth"], color="#2980B9", linewidth=2.0)
    ax1.set_title("Dense Regime: Episodic Return vs Training Episode", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Shaped Return")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.plot(sparse_train["episode"], sparse_train["rew_smooth"], color="#8E44AD", linewidth=2.0)
    ax2.set_title("Sparse Regime: Episodic Return vs Training Episode", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Sparse Return")
    ax2.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "02_reward_vs_episode.png", dpi=150)
    plt.close(fig)

    # 3. Steps & Final Distance Progression
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax1.plot(dense_train["episode"], dense_train["steps_smooth"], color="#D35400", linewidth=2.0)
    ax1.set_title("Steps to Episode Termination", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Steps")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.plot(dense_train["episode"], dense_train["dist_smooth"], color="#C0392B", linewidth=2.0)
    ax2.set_title("Mean Final Target Distance (px)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Distance (px)")
    ax2.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "03_steps_and_distance.png", dpi=150)
    plt.close(fig)


def plot_ablation_comparison():
    dense_eval = pd.read_csv(RESULTS_DIR / "dense" / "metrics_eval.csv")
    rewired_eval = pd.read_csv(RESULTS_DIR / "dense" / "metrics_eval_rewired.csv")
    unplastic_eval = pd.read_csv(RESULTS_DIR / "dense" / "metrics_eval_unplastic.csv")
    random_eval = pd.read_csv(RESULTS_DIR / "dense" / "metrics_eval_random.csv")

    conditions = [
        "Real + Plasticity",
        "Rewired + Plasticity",
        "Real (Unplastic)",
        "Random Baseline"
    ]
    dfs = [dense_eval, rewired_eval, unplastic_eval, random_eval]

    succ_rates = [df["success"].mean() * 100 for df in dfs]
    succ_errs = [np.std(df["success"]) / np.sqrt(len(df)) * 100 for df in dfs]

    final_dists = [df["final_distance"].mean() for df in dfs]
    final_errs = [df["final_distance"].std() / np.sqrt(len(df)) for df in dfs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    x = np.arange(len(conditions))
    colors = ["#2ECC71", "#E67E22", "#3498DB", "#95A5A6"]

    ax1.bar(x, succ_rates, yerr=succ_errs, capsize=5, color=colors, alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(conditions, rotation=15, ha="right", fontsize=9)
    ax1.set_ylabel("Success Rate (%) ± SEM", fontsize=10)
    ax1.set_title("Held-Out Evaluation Success Rate (100 Unseen Seeds)", fontsize=11, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    ax2.bar(x, final_dists, yerr=final_errs, capsize=5, color=colors, alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(conditions, rotation=15, ha="right", fontsize=9)
    ax2.set_ylabel("Final Distance to Goal (px) ± SEM", fontsize=10)
    ax2.set_title("Held-Out Final Proximity (Lower is Closer)", fontsize=11, fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "04_ablation_heldout_benchmark.png", dpi=150)
    plt.close(fig)


def main():
    print("Generating Phase 2 publication-grade figures...")
    plot_learning_curves()
    plot_ablation_comparison()
    print(f"All plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
