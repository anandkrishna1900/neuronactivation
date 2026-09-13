"""
Phase 6A: Flappy Bird — Complete Experiment Runner.

Executes all 12 steps of the Phase 6A protocol:
  Step 1:  Validate sandbox with baseline controllers
  Step 2:  Validate visual sensor (sensor audit)
  Step 3:  Validate sensory -> connectome causality
  Step 4:  Validate connectome -> motor causality
  Step 5:  Run unplastic baseline
  Step 6:  Run learning experiment
  Step 7:  Run multi-seed replication
  Step 8:  Run generalization test
  Step 9:  Run perturbation experiment
  Step 10: Run behavioral variability analysis
  Step 11: Run ablations
  Step 12: Generate final report

Usage:
    python -m experiments.phase6_flappy
    python -m experiments.phase6_flappy --step 1
    python -m experiments.phase6_flappy --step 6 --seeds 5 --episodes 200
"""

import argparse
import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from flymind.connectome.loader import ConnectomeLoader
from flymind.brain.network import NeuralNetwork
from flymind.environment.flappy import FlappyEnvironment, FlappyState, summarize_physics
from flymind.environment.flappy_sensor import FlappyVisualSensor, N_CHANNELS
from flymind.agent.flappy_agent import (
    FlappyConnectomeAgent,
    RandomFlapAgent,
    FixedPeriodFlapAgent,
    HandDesignedFlapAgent,
    BaseAgent,
)

RESULTS_DIR = ROOT / "results" / "phase6" / "flappy_bird"
CONNECTOME_PATH = ROOT / "data" / "processed" / "cx_heading_v1.json"
FIGDIR = RESULTS_DIR / "figures"

# ── Load connectome once ─────────────────────────────────────────────────────
_graph = ConnectomeLoader.load_from_json(CONNECTOME_PATH)


def _make_network(synapse_scale: float = 0.005) -> NeuralNetwork:
    """Create a fresh NeuralNetwork from the cached connectome."""
    return NeuralNetwork(_graph, synapse_scale=synapse_scale)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Validate sandbox with baseline controllers
# ══════════════════════════════════════════════════════════════════════════════

def run_step1(n_episodes: int = 200, seed: int = 42) -> Dict[str, Any]:
    """Validate the Flappy sandbox with simple baseline controllers."""
    print("\n" + "=" * 70)
    print("STEP 1: Validate Flappy Sandbox with Baseline Controllers")
    print("=" * 70)

    env = FlappyEnvironment(seed=seed)
    physics = summarize_physics()
    print(f"Physics: {json.dumps(physics, indent=2)}")

    controllers = {
        "random_50pct": RandomFlapAgent(flap_prob=0.5, seed=seed),
        "random_30pct": RandomFlapAgent(flap_prob=0.3, seed=seed),
        "fixed_period_8": FixedPeriodFlapAgent(period=8),
        "fixed_period_6": FixedPeriodFlapAgent(period=6),
        "fixed_period_10": FixedPeriodFlapAgent(period=10),
        "hand_designed": HandDesignedFlapAgent(gap_offset=30.0),
        "hand_designed_0": HandDesignedFlapAgent(gap_offset=0.0),
        "never_flap": RandomFlapAgent(flap_prob=0.0, seed=seed),
        "always_flap": RandomFlapAgent(flap_prob=1.0, seed=seed),
    }

    results = {}
    for name, agent in controllers.items():
        scores = []
        survival = []
        for ep in range(n_episodes):
            state = env.reset(seed=seed + ep)
            agent.reset()
            done = False
            while not done:
                action = agent.act(state)
                state, reward, done, info = env.step(action)
            scores.append(info["score"])
            survival.append(info["step"])
        results[name] = {
            "mean_score": float(np.mean(scores)),
            "median_score": float(np.median(scores)),
            "max_score": int(np.max(scores)),
            "std_score": float(np.std(scores)),
            "mean_survival": float(np.mean(survival)),
            "scores": scores,
            "survival": survival,
        }
        print(f"  {name:25s}  mean={results[name]['mean_score']:.2f}  "
              f"max={results[name]['max_score']}  "
              f"survival={results[name]['mean_survival']:.1f} steps")

    # Save results
    summary = {k: {kk: vv for kk, vv in v.items() if kk not in ("scores", "survival")}
               for k, v in results.items()}
    with open(RESULTS_DIR / "step1_baseline_summary.json", "w") as f:
        json.dump({"physics": physics, "results": summary}, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    names = list(results.keys())
    mean_scores = [results[n]["mean_score"] for n in names]
    max_scores = [results[n]["max_score"] for n in names]
    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))

    axes[0].barh(names, mean_scores, color=colors)
    axes[0].set_xlabel("Mean Score")
    axes[0].set_title("Mean Score by Controller")
    axes[1].barh(names, max_scores, color=colors)
    axes[1].set_xlabel("Max Score")
    axes[1].set_title("Max Score by Controller")
    plt.tight_layout()
    plt.savefig(FIGDIR / "step1_baseline_comparison.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {RESULTS_DIR / 'step1_baseline_summary.json'}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Validate visual sensor (sensor audit)
# ══════════════════════════════════════════════════════════════════════════════

def run_step2(seed: int = 42) -> Dict[str, Any]:
    """
    Sensor audit: move gap through different vertical positions,
    measure neural sensory input, verify distinguishability.
    """
    print("\n" + "=" * 70)
    print("STEP 2: Visual Sensor Audit")
    print("=" * 70)

    sensor = FlappyVisualSensor()
    env = FlappyEnvironment(seed=seed)

    # Fixed bird position
    bird_y_values = np.linspace(60.0, 340.0, 15)
    gap_centers = np.linspace(80.0, 320.0, 15)
    horizontal_distances = [50.0, 100.0, 150.0]

    audit_data = []

    for gap_center in gap_centers:
        for bird_y in bird_y_values:
            # Construct a synthetic state with one pipe at known position
            for dx in horizontal_distances:
                state = FlappyState(
                    bird_y=bird_y,
                    bird_vy=0.0,
                    pipes=[{"x": 80.0 + dx, "gap_center": gap_center}],
                    score=0,
                    step_count=0,
                    alive=True,
                )
                activations = sensor.sense(state)

                # Classify relative position
                if bird_y < gap_center - 60.0:
                    rel_pos = "below_gap"
                elif bird_y > gap_center + 60.0:
                    rel_pos = "above_gap"
                else:
                    rel_pos = "aligned"

                audit_data.append({
                    "gap_center": gap_center,
                    "bird_y": bird_y,
                    "dx": dx,
                    "rel_pos": rel_pos,
                    "activations": activations.tolist(),
                    "total_activation": float(np.sum(activations)),
                    "max_channel": int(np.argmax(activations)),
                })

    # Analysis: can we distinguish gap_above, gap_aligned, gap_below?
    df = pd.DataFrame(audit_data)
    categories = ["above_gap", "aligned", "below_gap"]

    print("\nSensor activation by relative position:")
    for cat in categories:
        subset = df[df["rel_pos"] == cat]
        if len(subset) == 0:
            continue
        acts = np.array(subset["activations"].tolist())
        print(f"  {cat:15s}: mean_total={np.mean(acts.sum(axis=1)):.3f}  "
              f"max_ch_mean={np.mean(np.argmax(acts, axis=1)):.1f}")

    # Discriminability test
    print("\nDiscriminability test (one-way ANOVA proxy):")
    groups = []
    for cat in categories:
        subset = df[df["rel_pos"] == cat]
        if len(subset) > 0:
            acts = np.array(subset["activations"].tolist())
            groups.append(acts.sum(axis=1))
    if len(groups) == 3:
        all_totals = np.concatenate(groups)
        grand_mean = np.mean(all_totals)
        ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
        ss_within = sum(np.sum((g - np.mean(g))**2) for g in groups)
        f_stat = (ss_between / 2) / (ss_within / (len(all_totals) - 3)) if ss_within > 0 else 0
        print(f"  F-statistic = {f_stat:.2f} (larger = more distinguishable)")

    # Plot sensor heatmap
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    sensor_labels = sensor.get_channel_labels()
    for ch in range(N_CHANNELS):
        ax = axes[ch // 3, ch % 3]
        # Create heatmap: bird_y vs gap_center for this channel at dx=100
        subset = df[df["dx"] == 100.0]
        vals = np.zeros((len(bird_y_values), len(gap_centers)))
        for _, row in subset.iterrows():
            yi = np.argmin(np.abs(bird_y_values - row["bird_y"]))
            gi = np.argmin(np.abs(gap_centers - row["gap_center"]))
            vals[yi, gi] = row["activations"][ch]
        im = ax.imshow(vals, origin="lower", aspect="auto",
                       extent=[gap_centers[0], gap_centers[-1], bird_y_values[0], bird_y_values[-1]],
                       cmap="viridis")
        ax.set_title(f"Ch {ch}: {sensor_labels[ch]}")
        ax.set_xlabel("Gap Center")
        ax.set_ylabel("Bird Y")
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle("Sensor Channel Activations (dx=100)", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGDIR / "step2_sensor_audit_heatmap.png", dpi=150)
    plt.close()

    # Save
    df.to_csv(RESULTS_DIR / "step2_sensor_audit.csv", index=False)
    print(f"\nResults saved to {RESULTS_DIR / 'step2_sensor_audit.csv'}")

    return {"f_statistic": f_stat, "n_samples": len(df)}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Validate sensory -> connectome causality
# ══════════════════════════════════════════════════════════════════════════════

def run_step3(seed: int = 42) -> Dict[str, Any]:
    """Verify that visual input causally affects neural activity."""
    print("\n" + "=" * 70)
    print("STEP 3: Sensory -> Connectome Causality")
    print("=" * 70)

    network = _make_network()
    agent = FlappyConnectomeAgent(network, enable_plasticity=False)
    sensor = agent.sensor

    # Test conditions: gap above, aligned, below
    conditions = {
        "gap_above": {"bird_y": 100.0, "gap_center": 300.0},
        "gap_aligned": {"bird_y": 200.0, "gap_center": 200.0},
        "gap_below": {"bird_y": 300.0, "gap_center": 100.0},
        "no_pipe": {"bird_y": 200.0, "gap_center": 200.0, "pipes": []},
    }

    results = {}
    for cond_name, params in conditions.items():
        pipes = params.get("pipes", [{"x": 150.0, "gap_center": params["gap_center"]}])
        state = FlappyState(
            bird_y=params["bird_y"],
            bird_vy=0.0,
            pipes=pipes,
            score=0,
            step_count=0,
            alive=True,
        )

        # Run 10 trials to get mean activity
        trial_activities = []
        for trial in range(10):
            agent.reset()
            sensor_out = sensor.sense(state)
            ext_current = agent._build_sensory_current(sensor_out)
            for _ in range(agent.sub_steps):
                activity = network.step(ext_current)
            trial_activities.append(activity.copy())

        mean_activity = np.mean(trial_activities, axis=0)

        # Key populations
        er4_mean = float(np.mean(mean_activity[agent.er4d_indices])) if agent.er4d_indices else 0.0
        epg_mean = float(np.mean(mean_activity[agent.epg_indices])) if agent.epg_indices else 0.0
        pen_l = float(np.mean(mean_activity[agent.pena_L])) if agent.pena_L else 0.0
        pen_r = float(np.mean(mean_activity[agent.pena_R])) if agent.pena_R else 0.0

        results[cond_name] = {
            "er4_mean": er4_mean,
            "epg_mean": epg_mean,
            "pen_l": pen_l,
            "pen_r": pen_r,
            "asym": pen_l - pen_r,
            "sensor_total": float(np.sum(sensor_out)),
        }
        print(f"  {cond_name:15s}: ER4={er4_mean:.4f}  EPG={epg_mean:.4f}  "
              f"PEN_L={pen_l:.4f}  PEN_R={pen_r:.4f}  asym={pen_l - pen_r:.4f}")

    # Check causality: does neural activity differ across conditions?
    er4_vals = [results[c]["er4_mean"] for c in conditions]
    epg_vals = [results[c]["epg_mean"] for c in conditions]
    er4_range = max(er4_vals) - min(er4_vals)
    epg_range = max(epg_vals) - min(epg_vals)

    causality_pass = er4_range > 0.001 or epg_range > 0.001
    print(f"\n  Causality check: ER4 range={er4_range:.6f}, EPG range={epg_range:.6f}")
    print(f"  PASS: {causality_pass}")

    with open(RESULTS_DIR / "step3_sensory_causality.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Validate connectome -> motor causality
# ══════════════════════════════════════════════════════════════════════════════

def run_step4(seed: int = 42) -> Dict[str, Any]:
    """Verify that neural activity causally affects flap decisions."""
    print("\n" + "=" * 70)
    print("STEP 4: Connectome -> Motor Causality")
    print("=" * 70)

    network = _make_network()
    agent = FlappyConnectomeAgent(
        network, enable_plasticity=False, decoder_mode="threshold"
    )
    sensor = agent.sensor

    conditions = {
        "gap_above": {"bird_y": 100.0, "gap_center": 300.0},
        "gap_aligned": {"bird_y": 200.0, "gap_center": 200.0},
        "gap_below": {"bird_y": 300.0, "gap_center": 100.0},
    }

    results = {}
    for cond_name, params in conditions.items():
        flap_counts = []
        flap_probs_list = []
        motor_diags = []

        for trial in range(50):
            agent.reset()
            state = FlappyState(
                bird_y=params["bird_y"],
                bird_vy=0.0,
                pipes=[{"x": 150.0, "gap_center": params["gap_center"]}],
                score=0,
                step_count=0,
                alive=True,
            )
            action = agent.act(state)
            diag = agent.get_motor_diagnostics()
            flap_counts.append(action)
            flap_probs_list.append(diag["flap_prob"])
            motor_diags.append(diag)

        flap_rate = np.mean(flap_counts)
        mean_prob = np.mean(flap_probs_list)
        results[cond_name] = {
            "flap_rate": float(flap_rate),
            "mean_flap_prob": float(mean_prob),
            "std_flap_prob": float(np.std(flap_probs_list)),
            "mean_asym": float(np.mean([d["asym"] for d in motor_diags])),
        }
        print(f"  {cond_name:15s}: flap_rate={flap_rate:.3f}  "
              f"mean_prob={mean_prob:.3f}  asym={results[cond_name]['mean_asym']:.4f}")

    # Motor causality: flap rate should differ across conditions
    rates = [results[c]["flap_rate"] for c in conditions]
    rate_range = max(rates) - min(rates)
    motor_pass = rate_range > 0.05
    print(f"\n  Motor causality: flap rate range = {rate_range:.3f}")
    print(f"  PASS: {motor_pass}")

    with open(RESULTS_DIR / "step4_motor_causality.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: Run unplastic baseline
# ══════════════════════════════════════════════════════════════════════════════

def run_step5(n_episodes: int = 200, seed: int = 42) -> Dict[str, Any]:
    """Run the connectome agent without plasticity as baseline."""
    print("\n" + "=" * 70)
    print("STEP 5: FlyMind Unplastic Baseline")
    print("=" * 70)

    env = FlappyEnvironment(seed=seed)
    network = _make_network()
    agent = FlappyConnectomeAgent(network, enable_plasticity=False)

    scores = []
    survivals = []
    flap_frequencies = []
    action_entropies = []
    trajectories = []

    for ep in range(n_episodes):
        state = env.reset(seed=seed + ep)
        agent.reset()
        done = False
        ep_actions = []
        ep_trajectory = []

        while not done:
            action = agent.act(state)
            ep_actions.append(action)
            ep_trajectory.append(state.bird_y)
            state, reward, done, info = env.step(action)

        scores.append(info["score"])
        survivals.append(info["step"])
        flap_freq = np.mean(ep_actions) if ep_actions else 0.0
        flap_frequencies.append(flap_freq)

        # Action entropy
        p1 = max(flap_freq, 1e-10)
        p0 = max(1 - flap_freq, 1e-10)
        entropy = -(p0 * np.log2(p0) + p1 * np.log2(p1))
        action_entropies.append(entropy)
        trajectories.append(ep_trajectory)

        if (ep + 1) % 50 == 0:
            print(f"  Episode {ep+1}/{n_episodes}: score={info['score']}  "
                  f"survival={info['step']}  flap_rate={flap_freq:.3f}")

    results = {
        "mean_score": float(np.mean(scores)),
        "median_score": float(np.median(scores)),
        "max_score": int(np.max(scores)),
        "std_score": float(np.std(scores)),
        "mean_survival": float(np.mean(survivals)),
        "mean_flap_freq": float(np.mean(flap_frequencies)),
        "mean_entropy": float(np.mean(action_entropies)),
        "scores": scores,
        "survivals": survivals,
        "flap_frequencies": flap_frequencies,
    }

    print(f"\n  Mean score: {results['mean_score']:.3f}")
    print(f"  Max score:  {results['max_score']}")
    print(f"  Mean survival: {results['mean_survival']:.1f} steps")
    print(f"  Mean flap frequency: {results['mean_flap_freq']:.3f}")

    # Save
    with open(RESULTS_DIR / "step5_unplastic_baseline.json", "w") as f:
        json.dump({k: v for k, v in results.items() if k not in ("scores", "survivals", "flap_frequencies")}, f, indent=2)
    np.save(RESULTS_DIR / "step5_scores.npy", np.array(scores))
    np.save(RESULTS_DIR / "step5_trajectories.npy", np.array([t[:100] for t in trajectories], dtype=object), allow_pickle=True)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(scores, bins=range(0, max(scores) + 2), edgecolor="black")
    axes[0].set_xlabel("Score")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Score Distribution (Unplastic)")
    axes[1].plot(survivals[:50], alpha=0.7)
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Survival (steps)")
    axes[1].set_title("Survival Over Episodes")
    axes[2].plot(flap_frequencies[:50], alpha=0.7)
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Flap Frequency")
    axes[2].set_title("Flap Frequency Over Episodes")
    plt.tight_layout()
    plt.savefig(FIGDIR / "step5_unplastic_baseline.png", dpi=150)
    plt.close()

    return results


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: Run learning experiment
# ══════════════════════════════════════════════════════════════════════════════

def run_step6(
    n_episodes: int = 500,
    n_eval_episodes: int = 100,
    seed: int = 42,
    checkpoints: Optional[List[int]] = None,
    reward_mode: str = "sparse",
) -> Dict[str, Any]:
    """Run the learning experiment with pathway-specific plasticity."""
    print("\n" + "=" * 70)
    print(f"STEP 6: Learning Experiment (seed={seed}, reward={reward_mode})")
    print("=" * 70)

    if checkpoints is None:
        checkpoints = [0, 50, 100, 200, 300, 400, 500]

    env = FlappyEnvironment(seed=seed)
    network = _make_network()
    agent = FlappyConnectomeAgent(
        network,
        enable_plasticity=True,
        plasticity_mode="pathway",
        learning_rate=0.002,
        eligibility_decay=0.85,
    )

    train_scores = []
    train_survivals = []
    train_flap_rates = []
    checkpoint_data = {}

    for ep in range(n_episodes):
        state = env.reset(seed=seed + ep + 10000)
        agent.reset()
        done = False
        ep_actions = []

        while not done:
            action = agent.act(state)
            ep_actions.append(action)
            state, reward, done, info = env.step(action)

            # Apply reward
            if reward_mode == "sparse":
                agent.apply_reward(reward)
            else:  # sparse + survival
                survival_reward = 0.01 if not done else 0.0
                agent.apply_reward(reward + survival_reward)

        train_scores.append(info["score"])
        train_survivals.append(info["step"])
        train_flap_rates.append(np.mean(ep_actions) if ep_actions else 0.0)

        # Evaluate at checkpoints
        if ep in checkpoints:
            eval_results = _evaluate_agent(agent, env, n_eval_episodes, seed + 50000)
            checkpoint_data[ep] = eval_results
            print(f"  Checkpoint {ep:4d}: train_score={info['score']:3d}  "
                  f"eval_mean={eval_results['mean_score']:.2f}  "
                  f"eval_max={eval_results['max_score']}")

        if (ep + 1) % 100 == 0:
            recent = train_scores[-100:]
            print(f"  Episode {ep+1}/{n_episodes}: "
                  f"last100_mean={np.mean(recent):.2f}  "
                  f"last100_max={int(np.max(recent))}")

    # Final evaluation
    final_eval = _evaluate_agent(agent, env, n_eval_episodes, seed + 60000)
    checkpoint_data["final"] = final_eval

    results = {
        "seed": seed,
        "reward_mode": reward_mode,
        "n_episodes": n_episodes,
        "train_scores": train_scores,
        "train_survivals": train_survivals,
        "train_flap_rates": train_flap_rates,
        "checkpoints": {str(k): {kk: vv for kk, vv in v.items() if kk != "scores"}
                        for k, v in checkpoint_data.items()},
        "final_eval": final_eval,
    }

    print(f"\n  Final eval: mean={final_eval['mean_score']:.2f}  max={final_eval['max_score']}")

    # Save
    with open(RESULTS_DIR / f"step6_learning_seed{seed}.json", "w") as f:
        json.dump({
            k: v for k, v in results.items()
            if k not in ("train_scores", "train_survivals", "train_flap_rates")
        }, f, indent=2)
    np.save(RESULTS_DIR / f"step6_train_scores_seed{seed}.npy", np.array(train_scores))

    # Plot learning curve
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    # Training scores (smoothed)
    window = 20
    if len(train_scores) >= window:
        smoothed = np.convolve(train_scores, np.ones(window)/window, mode="valid")
        axes[0].plot(smoothed, color="blue")
    axes[0].plot(train_scores, alpha=0.3, color="blue")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Score")
    axes[0].set_title(f"Training Score (seed={seed})")

    # Eval scores at checkpoints
    ckpts = sorted([k for k in checkpoint_data.keys() if isinstance(k, int)])
    eval_means = [checkpoint_data[c]["mean_score"] for c in ckpts]
    eval_maxs = [checkpoint_data[c]["max_score"] for c in ckpts]
    axes[1].plot(ckpts, eval_means, "o-", label="mean")
    axes[1].plot(ckpts, eval_maxs, "s-", label="max")
    axes[1].set_xlabel("Training Episode")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Evaluation Performance")
    axes[1].legend()

    # Flap rate
    if len(train_flap_rates) >= window:
        smoothed_flap = np.convolve(train_flap_rates, np.ones(window)/window, mode="valid")
        axes[2].plot(smoothed_flap, color="green")
    axes[2].plot(train_flap_rates, alpha=0.3, color="green")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Flap Frequency")
    axes[2].set_title("Flap Frequency Over Training")

    plt.tight_layout()
    plt.savefig(FIGDIR / f"step6_learning_curve_seed{seed}.png", dpi=150)
    plt.close()

    return results


def _evaluate_agent(agent: BaseAgent, env: FlappyEnvironment, n_episodes: int, base_seed: int) -> Dict[str, Any]:
    """Evaluate an agent for n episodes and return summary statistics."""
    scores = []
    survivals = []
    flap_rates = []
    was_plastic = agent.enable_plasticity if hasattr(agent, "enable_plasticity") else False
    if hasattr(agent, "freeze_weights"):
        agent.freeze_weights()
    for ep in range(n_episodes):
        state = env.reset(seed=base_seed + ep)
        agent.reset()
        done = False
        ep_actions = []
        while not done:
            action = agent.act(state)
            ep_actions.append(action)
            state, reward, done, info = env.step(action)
        scores.append(info["score"])
        survivals.append(info["step"])
        flap_rates.append(np.mean(ep_actions) if ep_actions else 0.0)
    if was_plastic and hasattr(agent, "unfreeze_weights"):
        agent.unfreeze_weights()
    return {
        "mean_score": float(np.mean(scores)),
        "median_score": float(np.median(scores)),
        "max_score": int(np.max(scores)),
        "std_score": float(np.std(scores)),
        "mean_survival": float(np.mean(survivals)),
        "mean_flap_rate": float(np.mean(flap_rates)),
        "scores": scores,
    }


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7: Multi-seed replication
# ══════════════════════════════════════════════════════════════════════════════

def run_step7(n_seeds: int = 10, n_episodes: int = 500, base_seed: int = 1000) -> Dict[str, Any]:
    """Run learning with multiple independent seeds."""
    print("\n" + "=" * 70)
    print(f"STEP 7: Multi-Seed Replication ({n_seeds} seeds, {n_episodes} episodes each)")
    print("=" * 70)

    all_results = {}
    for i in range(n_seeds):
        seed = base_seed + i * 100
        print(f"\n  --- Seed {seed} ({i+1}/{n_seeds}) ---")
        result = run_step6(n_episodes=n_episodes, seed=seed, reward_mode="sparse")
        all_results[seed] = {
            "final_eval": result["final_eval"],
            "train_scores": result["train_scores"],
        }

    # Aggregate
    final_means = [all_results[s]["final_eval"]["mean_score"] for s in all_results]
    final_maxs = [all_results[s]["final_eval"]["max_score"] for s in all_results]

    aggregate = {
        "n_seeds": n_seeds,
        "n_episodes": n_episodes,
        "mean_of_means": float(np.mean(final_means)),
        "std_of_means": float(np.std(final_means)),
        "ci_95_low": float(np.mean(final_means) - 1.96 * np.std(final_means) / np.sqrt(n_seeds)),
        "ci_95_high": float(np.mean(final_means) + 1.96 * np.std(final_means) / np.sqrt(n_seeds)),
        "mean_of_maxes": float(np.mean(final_maxs)),
        "per_seed": {str(s): {"mean_score": all_results[s]["final_eval"]["mean_score"],
                              "max_score": all_results[s]["final_eval"]["max_score"]}
                     for s in all_results},
    }

    print(f"\n  Aggregate: mean={aggregate['mean_of_means']:.3f} "
          f"std={aggregate['std_of_means']:.3f} "
          f"95%CI=[{aggregate['ci_95_low']:.3f}, {aggregate['ci_95_high']:.3f}]")

    with open(RESULTS_DIR / "step7_multiseed.json", "w") as f:
        json.dump(aggregate, f, indent=2)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    seeds = sorted(all_results.keys())
    means = [all_results[s]["final_eval"]["mean_score"] for s in seeds]
    maxes = [all_results[s]["final_eval"]["max_score"] for s in seeds]
    x = range(len(seeds))
    ax.bar([i - 0.15 for i in x], means, width=0.3, label="Mean", color="steelblue")
    ax.bar([i + 0.15 for i in x], maxes, width=0.3, label="Max", color="coral")
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(s) for s in seeds], rotation=45)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Score")
    ax.set_title("Multi-Seed Replication")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGDIR / "step7_multiseed.png", dpi=150)
    plt.close()

    return aggregate


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8: Generalization test
# ══════════════════════════════════════════════════════════════════════════════

def run_step8(seed: int = 42, n_episodes: int = 200) -> Dict[str, Any]:
    """Generalization: train on one distribution, evaluate on unseen configs."""
    print("\n" + "=" * 70)
    print("STEP 8: Generalization Test")
    print("=" * 70)

    # Train on default distribution
    env_train = FlappyEnvironment(seed=seed, gap_size=120.0, horizontal_speed=2.5)
    network = _make_network()
    agent = FlappyConnectomeAgent(network, enable_plasticity=True, plasticity_mode="pathway")

    print("  Training on default distribution (gap=120, speed=2.5)...")
    for ep in range(300):
        state = env_train.reset(seed=seed + ep + 20000)
        agent.reset()
        done = False
        while not done:
            action = agent.act(state)
            state, reward, done, info = env_train.step(action)
            agent.apply_reward(reward)

    agent.freeze_weights()

    # Evaluate on different distributions
    configs = {
        "train_config": {"gap_size": 120.0, "horizontal_speed": 2.5},
        "large_gap": {"gap_size": 160.0, "horizontal_speed": 2.5},
        "small_gap": {"gap_size": 90.0, "horizontal_speed": 2.5},
        "fast_scroll": {"gap_size": 120.0, "horizontal_speed": 3.5},
        "slow_scroll": {"gap_size": 120.0, "horizontal_speed": 1.5},
        "high_start": {"gap_size": 120.0, "horizontal_speed": 2.5, "initial_bird_y": 350.0},
        "low_start": {"gap_size": 120.0, "horizontal_speed": 2.5, "initial_bird_y": 50.0},
    }

    results = {}
    for name, cfg in configs.items():
        env_eval = FlappyEnvironment(seed=seed + 70000, **cfg)
        eval_res = _evaluate_agent(agent, env_eval, n_episodes, seed + 80000)
        results[name] = eval_res
        print(f"  {name:20s}: mean={eval_res['mean_score']:.2f}  max={eval_res['max_score']}")

    with open(RESULTS_DIR / "step8_generalization.json", "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != "scores"}
                   for k, v in results.items()}, f, indent=2)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    names = list(results.keys())
    means = [results[n]["mean_score"] for n in names]
    colors = ["steelblue" if n == "train_config" else "coral" for n in names]
    ax.barh(names, means, color=colors)
    ax.set_xlabel("Mean Score")
    ax.set_title("Generalization Performance")
    plt.tight_layout()
    plt.savefig(FIGDIR / "step8_generalization.png", dpi=150)
    plt.close()

    return results


# ══════════════════════════════════════════════════════════════════════════════
# STEP 9: Perturbation experiment
# ══════════════════════════════════════════════════════════════════════════════

def run_step9(seed: int = 42, n_episodes: int = 100) -> Dict[str, Any]:
    """Introduce unexpected perturbations and measure adaptation."""
    print("\n" + "=" * 70)
    print("STEP 9: Perturbation Experiment")
    print("=" * 70)

    env = FlappyEnvironment(seed=seed)
    network = _make_network()
    agent = FlappyConnectomeAgent(network, enable_plasticity=False)

    perturbations = {
        "baseline": {"perturb_type": "none"},
        "gap_shift_up": {"perturb_type": "gap_shift", "shift": 50.0},
        "gap_shift_down": {"perturb_type": "gap_shift", "shift": -50.0},
        "speed_change": {"perturb_type": "speed_change", "new_speed": 4.0},
        "gap_narrow": {"perturb_type": "gap_narrow", "new_gap": 80.0},
    }

    results = {}
    for perturb_name, perturb_cfg in perturbations.items():
        scores = []
        survivals = []
        for ep in range(n_episodes):
            state = env.reset(seed=seed + ep + 30000)
            agent.reset()

            # Apply perturbation mid-episode
            if perturb_cfg["perturb_type"] == "gap_shift":
                if len(env.pipes) > 1:
                    env.pipes[1]["gap_center"] += perturb_cfg["shift"]
            elif perturb_cfg["perturb_type"] == "speed_change":
                env.horizontal_speed = perturb_cfg["new_speed"]
            elif perturb_cfg["perturb_type"] == "gap_narrow":
                env.gap_size = perturb_cfg["new_gap"]

            done = False
            while not done:
                action = agent.act(state)
                state, reward, done, info = env.step(action)
            scores.append(info["score"])
            survivals.append(info["step"])

        results[perturb_name] = {
            "mean_score": float(np.mean(scores)),
            "max_score": int(np.max(scores)),
            "mean_survival": float(np.mean(survivals)),
        }
        print(f"  {perturb_name:20s}: mean={results[perturb_name]['mean_score']:.2f}  "
              f"survival={results[perturb_name]['mean_survival']:.1f}")

    with open(RESULTS_DIR / "step9_perturbation.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# STEP 10: Behavioral variability analysis
# ══════════════════════════════════════════════════════════════════════════════

def run_step10(n_episodes: int = 200, seed: int = 42) -> Dict[str, Any]:
    """Measure behavioral variability across different controllers."""
    print("\n" + "=" * 70)
    print("STEP 10: Behavioral Variability Analysis")
    print("=" * 70)

    env = FlappyEnvironment(seed=seed)

    controllers = {
        "random": RandomFlapAgent(flap_prob=0.5, seed=seed),
        "fixed_period": FixedPeriodFlapAgent(period=8),
        "flymind_unplastic": FlappyConnectomeAgent(_make_network(), enable_plasticity=False),
    }

    results = {}
    for name, agent in controllers.items():
        flap_timings = []
        trajectories = []
        entropies = []

        for ep in range(n_episodes):
            state = env.reset(seed=seed + ep + 40000)
            agent.reset()
            done = False
            ep_actions = []
            ep_traj = []
            last_flap_step = -1

            while not done:
                action = agent.act(state)
                ep_actions.append(action)
                ep_traj.append(state.bird_y)
                if action == 1:
                    flap_timings.append(state.step_count - last_flap_step if last_flap_step >= 0 else 0)
                    last_flap_step = state.step_count
                state, reward, done, info = env.step(action)

            p1 = np.mean(ep_actions)
            p0 = 1 - p1
            entropy = -(max(p0, 1e-10) * np.log2(max(p0, 1e-10)) + max(p1, 1e-10) * np.log2(max(p1, 1e-10)))
            entropies.append(entropy)
            trajectories.append(ep_traj[:100])

        results[name] = {
            "timing_std": float(np.std(flap_timings)) if flap_timings else 0.0,
            "timing_mean": float(np.mean(flap_timings)) if flap_timings else 0.0,
            "trajectory_var": float(np.mean([np.std(t) for t in trajectories])) if trajectories else 0.0,
            "mean_entropy": float(np.mean(entropies)),
            "std_entropy": float(np.std(entropies)),
        }
        print(f"  {name:20s}: timing_std={results[name]['timing_std']:.2f}  "
              f"traj_var={results[name]['trajectory_var']:.2f}  "
              f"entropy={results[name]['mean_entropy']:.3f}")

    with open(RESULTS_DIR / "step10_variability.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# STEP 11: Ablations
# ══════════════════════════════════════════════════════════════════════════════

def run_step11(n_episodes: int = 200, seed: int = 42) -> Dict[str, Any]:
    """Run ablation experiments: different connectome/controller configurations."""
    print("\n" + "=" * 70)
    print("STEP 11: Ablation Experiments")
    print("=" * 70)

    env = FlappyEnvironment(seed=seed)

    configs = {
        "A_real_no_plasticity": {
            "agent_fn": lambda: FlappyConnectomeAgent(
                _make_network(), enable_plasticity=False, plasticity_mode="none"
            ),
        },
        "B_real_pathway_plasticity": {
            "agent_fn": lambda: FlappyConnectomeAgent(
                _make_network(), enable_plasticity=True, plasticity_mode="pathway"
            ),
        },
        "C_random_controller": {
            "agent_fn": lambda: RandomFlapAgent(flap_prob=0.5, seed=seed),
        },
        "D_fixed_controller": {
            "agent_fn": lambda: FixedPeriodFlapAgent(period=8),
        },
        "E_hand_designed": {
            "agent_fn": lambda: HandDesignedFlapAgent(gap_offset=30.0),
        },
        "F_real_global_plasticity": {
            "agent_fn": lambda: FlappyConnectomeAgent(
                _make_network(), enable_plasticity=True, plasticity_mode="global"
            ),
        },
    }

    results = {}
    for name, cfg in configs.items():
        agent = cfg["agent_fn"]()
        scores = []
        survivals = []
        for ep in range(n_episodes):
            state = env.reset(seed=seed + ep + 60000)
            agent.reset()
            done = False
            while not done:
                action = agent.act(state)
                state, reward, done, info = env.step(action)
                if hasattr(agent, "apply_reward"):
                    agent.apply_reward(reward)
            scores.append(info["score"])
            survivals.append(info["step"])

        results[name] = {
            "mean_score": float(np.mean(scores)),
            "median_score": float(np.median(scores)),
            "max_score": int(np.max(scores)),
            "mean_survival": float(np.mean(survivals)),
        }
        print(f"  {name:40s}: mean={results[name]['mean_score']:.2f}  "
              f"max={results[name]['max_score']}")

    with open(RESULTS_DIR / "step11_ablations.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    names = list(results.keys())
    means = [results[n]["mean_score"] for n in names]
    maxes = [results[n]["max_score"] for n in names]
    x = range(len(names))
    ax.barh(names, means, color="steelblue", label="Mean")
    for i, m in enumerate(maxes):
        ax.plot(m, i, "r|", markersize=20)
    ax.set_xlabel("Score")
    ax.set_title("Ablation Results")
    plt.tight_layout()
    plt.savefig(FIGDIR / "step11_ablations.png", dpi=150)
    plt.close()

    return results


# ══════════════════════════════════════════════════════════════════════════════
# STEP 12: Generate final report
# ══════════════════════════════════════════════════════════════════════════════

def run_step12() -> None:
    """Generate the final Phase 6A report."""
    print("\n" + "=" * 70)
    print("STEP 12: Generating Final Report")
    print("=" * 70)

    # Load all results
    def _load(name):
        path = RESULTS_DIR / name
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return {}

    step1 = _load("step1_baseline_summary.json")
    step3 = _load("step3_sensory_causality.json")
    step4 = _load("step4_motor_causality.json")
    step5 = _load("step5_unplastic_baseline.json")
    step7 = _load("step7_multiseed.json")
    step8 = _load("step8_generalization.json")
    step9 = _load("step9_perturbation.json")
    step10 = _load("step10_variability.json")
    step11 = _load("step11_ablations.json")

    physics = step1.get("physics", {})

    report = f"""# FlyMind Phase 6A: Flappy Bird — Final Report

**Date**: {time.strftime('%Y-%m-%d %H:%M')}
**Connectome**: 261 neurons, 19969 synapses (Janelia Hemibrain v1.2.1)
**Environment**: Flappy-Bird-style 2D side-scrolling sandbox

---

## 1. Research Question

"Can a connectome-derived neural controller acquire useful behavior in a new
visually guided, timing-dependent environment requiring continuous state
estimation, timing, and motor control?"

---

## 2. Environment

| Parameter | Value |
|-----------|-------|
| Gravity | {physics.get('gravity', 'N/A')} |
| Flap Impulse | {physics.get('flap_impulse', 'N/A')} |
| Horizontal Speed | {physics.get('horizontal_speed', 'N/A')} |
| Gap Size | {physics.get('gap_size', 'N/A')} |
| Pipe Spacing | {physics.get('pipe_spacing', 'N/A')} |
| Bird Radius | {physics.get('bird_radius', 'N/A')} |
| World Height | {physics.get('world_height', 'N/A')} |
| Max Steps | {physics.get('max_steps', 'N/A')} |

Action space: 2 actions (NO FLAP=0, FLAP=1).

---

## 3. Sensory Interface

9-channel directional receptive field sensor (3x3 grid):
- upper-left, upper-center, upper-right
- center-left, center-center, center-right
- lower-left, lower-center, lower-right

Encodes obstacle proximity and gap structure without exposing
privileged game coordinates.

---

## 4. Connectome Interface

**REAL CONNECTOME**: 261 neurons, 19969 synapses — unchanged from Phase 5.

**ENGINEERED ENVIRONMENT INTERFACE**:
- Sensor 9-channel mapping to EPG compass ring (16 glomeruli)
- PEN_a hemispheric asymmetry -> binary FLAP/NO-FLAP decoder

---

## 5. Motor Decoder

Binary decoder based on PEN_a hemispheric asymmetry:
- Positive asymmetry -> flap
- Negative asymmetry -> no flap
- Sigmoid mapping with calibrated temperature

---

## 6. Learning Rule

3-Factor Reward-Modulated Hebbian with Eligibility Traces:
- Pathway-specific mask: ER4d->EPG + EPG->PEG (7.15% of edges)
- Learning rate: 0.002
- Eligibility decay: 0.85

---

## 7. Training Protocol

10 independent seeds, 500 episodes per seed.
Checkpoints at 0, 50, 100, 200, 300, 400, 500.
Frozen evaluation at each checkpoint.

---

## 8. Baselines

| Controller | Mean Score | Max Score |
|------------|-----------|-----------|
"""

    if "results" in step1:
        for name, vals in step1["results"].items():
            report += f"| {name} | {vals.get('mean_score', 'N/A'):.2f} | {vals.get('max_score', 'N/A')} |\n"

    report += f"""
---

## 9. Results

### Sensory Causality (Step 3)

"""

    if step3:
        for cond, vals in step3.items():
            report += f"- **{cond}**: ER4={vals.get('er4_mean', 'N/A'):.4f}, EPG={vals.get('epg_mean', 'N/A'):.4f}\n"

    report += f"""
### Motor Causality (Step 4)

"""
    if step4:
        for cond, vals in step4.items():
            report += f"- **{cond}**: flap_rate={vals.get('flap_rate', 'N/A'):.3f}\n"

    report += f"""
### Unplastic Baseline (Step 5)

"""
    if step5:
        report += f"- Mean score: {step5.get('mean_score', 'N/A'):.3f}\n"
        report += f"- Max score: {step5.get('max_score', 'N/A')}\n"
        report += f"- Mean survival: {step5.get('mean_survival', 'N/A'):.1f} steps\n"

    report += f"""
### Multi-Seed Replication (Step 7)

"""
    if step7:
        report += f"- Seeds: {step7.get('n_seeds', 'N/A')}\n"
        report += f"- Mean of means: {step7.get('mean_of_means', 'N/A'):.3f}\n"
        report += f"- 95% CI: [{step7.get('ci_95_low', 'N/A'):.3f}, {step7.get('ci_95_high', 'N/A'):.3f}]\n"

    report += f"""
---

## 10. Generalization

"""
    if step8:
        for name, vals in step8.items():
            report += f"- **{name}**: mean={vals.get('mean_score', 'N/A'):.2f}\n"

    report += f"""
---

## 11. Perturbation Experiments

"""
    if step9:
        for name, vals in step9.items():
            report += f"- **{name}**: mean={vals.get('mean_score', 'N/A'):.2f}, survival={vals.get('mean_survival', 'N/A'):.1f}\n"

    report += f"""
---

## 12. Behavioral Variability

"""
    if step10:
        for name, vals in step10.items():
            report += f"- **{name}**: timing_std={vals.get('timing_std', 'N/A'):.2f}, entropy={vals.get('mean_entropy', 'N/A'):.3f}\n"

    report += f"""
---

## 13. Ablations

"""
    if step11:
        for name, vals in step11.items():
            report += f"- **{name}**: mean={vals.get('mean_score', 'N/A'):.2f}, max={vals.get('max_score', 'N/A')}\n"

    report += """
---

## 14. Failure Cases

(To be populated based on actual results)

---

## 15. Limitations

- The 261-neuron connectome is a partial circuit, not the full Drosophila brain.
- The visual sensor is a simplified abstraction, not a realistic optic lobe model.
- The motor decoder is an engineered interface, not a biological motor circuit.
- Plasticity is restricted to 7.15% of synapses (pathway mask).

---

## 16. Biological Interpretation

(To be populated based on actual results)

---

## 17. Conclusion

(To be populated based on actual results)

---

## 18. Recommended Phase 6B

(To be populated based on actual results)
"""

    report_path = ROOT / "docs" / "FlyMind_Phase6A_FlappyBird_Report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Report saved to {report_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Phase 6A: Flappy Bird Experiments")
    parser.add_argument("--step", type=int, default=None, help="Run specific step (1-12)")
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds for multi-seed")
    parser.add_argument("--episodes", type=int, default=500, help="Episodes per seed")
    parser.add_argument("--seed", type=int, default=42, help="Base seed")
    args = parser.parse_args()

    FIGDIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  PHASE 6A: FLYMIND PLAYS FLAPPY BIRD")
    print("=" * 70)

    t0 = time.time()

    steps = {
        1: lambda: run_step1(n_episodes=args.episodes, seed=args.seed),
        2: lambda: run_step2(seed=args.seed),
        3: lambda: run_step3(seed=args.seed),
        4: lambda: run_step4(seed=args.seed),
        5: lambda: run_step5(n_episodes=args.episodes, seed=args.seed),
        6: lambda: run_step6(n_episodes=args.episodes, seed=args.seed),
        7: lambda: run_step7(n_seeds=args.seeds, n_episodes=args.episodes),
        8: lambda: run_step8(seed=args.seed, n_episodes=args.episodes),
        9: lambda: run_step9(seed=args.seed, n_episodes=min(args.episodes, 100)),
        10: lambda: run_step10(n_episodes=args.episodes, seed=args.seed),
        11: lambda: run_step11(n_episodes=args.episodes, seed=args.seed),
        12: lambda: run_step12(),
    }

    if args.step is not None:
        if args.step in steps:
            steps[args.step]()
        else:
            print(f"Invalid step: {args.step}. Choose 1-12.")
            sys.exit(1)
    else:
        # Run all steps sequentially
        for step_num in range(1, 13):
            try:
                steps[step_num]()
            except Exception as e:
                print(f"\n  *** STEP {step_num} FAILED: {e} ***")
                print("  STOPPING — diagnose before continuing.")
                import traceback
                traceback.print_exc()
                sys.exit(1)

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"  Phase 6A complete. Total time: {elapsed:.1f}s")
    print(f"  Results: {RESULTS_DIR}")
    print(f"  Figures: {FIGDIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
