"""
Phase 4 Behavioral Metrics Module.
Calculates trajectory entropy, turning angle distributions, dwell times,
and measures whether variability is structured/adaptive rather than pure random noise.
"""

from typing import List, Dict, Any
import numpy as np


def compute_trajectory_diversity(trajectories: List[np.ndarray]) -> float:
    """Computes mean pairwise Frechet/Euclidean distance across trajectory paths."""
    if len(trajectories) < 2:
        return 0.0
    dists = []
    for i in range(len(trajectories)):
        for j in range(i + 1, len(trajectories)):
            t1, t2 = trajectories[i], trajectories[j]
            min_len = min(len(t1), len(t2))
            if min_len > 0:
                dist = np.mean(np.linalg.norm(t1[:min_len] - t2[:min_len], axis=1))
                dists.append(dist)
    return float(np.mean(dists)) if dists else 0.0


def compute_action_entropy(actions: List[int]) -> float:
    """Computes Shannon entropy of action selection distribution."""
    if not actions:
        return 0.0
    counts = np.bincount(actions)
    probs = counts[counts > 0] / len(actions)
    entropy = -np.sum(probs * np.log2(probs))
    return float(entropy)


def compute_turning_angle_distribution(headings: List[float]) -> Dict[str, float]:
    """Computes turning rate mean, variance, and directional bias."""
    if len(headings) < 2:
        return {"mean_turn_rad": 0.0, "std_turn_rad": 0.0, "entropy": 0.0}
    diffs = np.diff(headings)
    diffs = (diffs + np.pi) % (2.0 * np.pi) - np.pi
    return {
        "mean_turn_rad": float(np.mean(diffs)),
        "std_turn_rad": float(np.std(diffs)),
        "abs_mean_turn_rad": float(np.mean(np.abs(diffs))),
    }


def compute_behavioral_metrics(episode_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregates multi-trial behavioral variability metrics."""
    trajectories = [np.array(r["trajectory"]) for r in episode_records if "trajectory" in r]
    all_actions = [a for r in episode_records for a in r.get("actions", [])]
    
    div = compute_trajectory_diversity(trajectories)
    ent = compute_action_entropy(all_actions)
    succ = np.mean([r.get("success", 0) for r in episode_records]) * 100
    steps = np.mean([r.get("steps", 0) for r in episode_records])

    total_acts = len(all_actions) if all_actions else 1
    action_counts = {1: 0, 2: 0, 3: 0}
    for a in all_actions:
        if a in action_counts:
            action_counts[a] += 1

    action_dist = {
        "FORWARD": action_counts[1] / total_acts,
        "TURN_LEFT": action_counts[2] / total_acts,
        "TURN_RIGHT": action_counts[3] / total_acts,
    }
    
    return {
        "trajectory_diversity": div,
        "action_entropy": ent,
        "success_rate": succ,
        "mean_steps": steps,
        "action_distribution": action_dist,
    }
