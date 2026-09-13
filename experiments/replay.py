"""
Replay script for saved navigation episodes.
Loads saved trajectory and action history, displaying step-by-step agent motion.
"""

import sys
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent


def replay_episode(json_path: Path, out_path: Optional[Path] = None):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    traj = np.array(data["trajectory"])
    seed = data["seed"]
    success = bool(data["success"])
    steps = data["steps"]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_title(f"Replay Episode (Seed {seed}) | Steps: {steps} | Success: {success}", fontsize=11, fontweight="bold")

    ax.plot(traj[:, 0], traj[:, 1], color="#2980B9", linewidth=2.0, label="Path")
    ax.plot(traj[0, 0], traj[0, 1], "o", color="#27AE60", markersize=9, label="Start")
    ax.plot(traj[-1, 0], traj[-1, 1], "^", color="#C0392B", markersize=10, label="Final Pos")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)

    if out_path is None:
        out_path = json_path.with_suffix(".png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Replay trajectory saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay saved FlyMind episode.")
    parser.add_argument("--episode", type=str, default="results/phase2/sample_episode.json", help="Path to episode JSON file.")
    parser.add_argument("--output", type=str, default=None, help="Path to output PNG.")
    args = parser.parse_args()

    replay_episode(Path(args.episode), Path(args.output) if args.output else None)
