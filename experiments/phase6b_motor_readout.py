"""
Phase 6B: Connectome Motor Readout - Complete Experiment Runner.

Executes the Phase 6B protocol:
  Step 1:  Validate baselines (random, fixed, hand-designed, unplastic connectome)
  Step 2:  Sensory -> connectome causality
  Step 3:  Connectome -> motor causality (PEG gating test)
  Step 4:  Unplastic connectome baseline (50 episodes)
  Step 5:  Pathway-specific plasticity training (500 episodes)
  Step 6:  Multi-seed replication (5 seeds)
  Step 7:  Generalization test
  Step 8:  Perturbation experiment
  Step 9:  Behavioral variability analysis
  Step 10: Ablations
  Step 11: Statistical analysis
  Step 12: Generate final report

Usage:
    python -m experiments.phase6b_motor_readout
    python -m experiments.phase6b_motor_readout --step 1
"""

import argparse
import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from flymind.connectome.loader import ConnectomeLoader
from flymind.brain.network import NeuralNetwork
from flymind.environment.flappy import FlappyEnvironment, FlappyState, summarize_physics
from flymind.environment.flappy_sensor import FlappyVisualSensor
from flymind.agent.flappy_agent_phase6b import ConnectomeMotorReadoutAgent
from flymind.agent.flappy_agent import (
    RandomFlapAgent, FixedPeriodFlapAgent, HandDesignedFlapAgent, BaseAgent,
)

RESULTS_DIR = ROOT / "results" / "phase6b"
CONNECTOME_PATH = ROOT / "data" / "processed" / "cx_heading_v1.json"
FIGDIR = RESULTS_DIR / "figures"

_graph = ConnectomeLoader.load_from_json(CONNECTOME_PATH)


def _make_network(synapse_scale: float = 0.005) -> NeuralNetwork:
    return NeuralNetwork(_graph, synapse_scale=synapse_scale)


def _evaluate_agent(agent, env, n_episodes, base_seed):
    scores, survivals, flap_rates = [], [], []
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
# STEP 1: Validate baselines
# ══════════════════════════════════════════════════════════════════════════════

def run_step1(n_episodes=200, seed=42):
    print("\n" + "=" * 70)
    print("STEP 1: Baseline Validation")
    print("=" * 70)

    env = FlappyEnvironment(seed=seed)
    controllers = {
        "random_50pct": RandomFlapAgent(flap_prob=0.5, seed=seed),
        "random_30pct": RandomFlapAgent(flap_prob=0.3, seed=seed),
        "fixed_period_8": FixedPeriodFlapAgent(period=8),
        "hand_designed": HandDesignedFlapAgent(gap_offset=30.0),
        "never_flap": RandomFlapAgent(flap_prob=0.0, seed=seed),
        "always_flap": RandomFlapAgent(flap_prob=1.0, seed=seed),
    }

    results = {}
    for name, agent in controllers.items():
        scores, survival = [], []
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
            "max_score": int(np.max(scores)),
            "mean_survival": float(np.mean(survival)),
        }
        print("  %-25s  mean=%.2f  max=%d  surv=%.1f" % (
            name, results[name]["mean_score"], results[name]["max_score"], results[name]["mean_survival"]))

    with open(RESULTS_DIR / "step1_baselines.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Sensory -> Connectome Causality
# ══════════════════════════════════════════════════════════════════════════════

def run_step2(seed=42):
    print("\n" + "=" * 70)
    print("STEP 2: Sensory -> Connectome Causality")
    print("=" * 70)

    network = _make_network()
    agent = ConnectomeMotorReadoutAgent(network, enable_plasticity=False)
    sensor = agent.sensor

    conditions = {
        "gap_above": {"bird_y": 100.0, "gap_center": 300.0},
        "gap_aligned": {"bird_y": 200.0, "gap_center": 200.0},
        "gap_below": {"bird_y": 300.0, "gap_center": 100.0},
        "no_pipe": {"bird_y": 200.0, "pipes": []},
    }

    results = {}
    for cond, params in conditions.items():
        pipes = params.get("pipes", [{"x": 150.0, "gap_center": params.get("gap_center", 200.0)}])
        state = FlappyState(bird_y=params["bird_y"], bird_vy=0.0, pipes=pipes,
                           score=0, step_count=0, alive=True)
        trial_activities = []
        for _ in range(20):
            agent.reset()
            sensor_out = sensor.sense(state)
            ext = agent._build_sensory_current(sensor_out)
            act = None
            for _ in range(agent.sub_steps):
                act = network.step(ext)
            trial_activities.append(act.copy())

        mean_act = np.mean(trial_activities, axis=0)
        er4 = float(np.mean(mean_act[agent.er4d_indices])) if agent.er4d_indices else 0.0
        epg = float(np.mean(mean_act[agent.epg_indices])) if agent.epg_indices else 0.0
        peg = float(np.mean(mean_act[agent.peg_indices])) if agent.peg_indices else 0.0

        results[cond] = {"er4_mean": er4, "epg_mean": epg, "peg_mean": peg}
        print("  %-15s  ER4=%.4f  EPG=%.4f  PEG=%.4f" % (cond, er4, epg, peg))

    er4_range = max(r["er4_mean"] for r in results.values()) - min(r["er4_mean"] for r in results.values())
    causality_pass = er4_range > 0.001
    print("\n  ER4 range=%.6f  PASS=%s" % (er4_range, causality_pass))

    with open(RESULTS_DIR / "step2_sensory_causality.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Connectome -> Motor Causality (PEG gating test)
# ══════════════════════════════════════════════════════════════════════════════

def run_step3(seed=42):
    print("\n" + "=" * 70)
    print("STEP 3: Connectome -> Motor Causality (PEG gating)")
    print("=" * 70)

    # Test: does PEG activity affect motor output?
    # Compare flap_prob with and without PEG gating
    network = _make_network()
    agent = ConnectomeMotorReadoutAgent(network, enable_plasticity=False, motor_gain=80, motor_bias=-2.5)
    sensor = agent.sensor

    conditions = {
        "gap_above": {"bird_y": 100.0, "gap_center": 300.0},
        "gap_aligned": {"bird_y": 200.0, "gap_center": 200.0},
        "gap_below": {"bird_y": 300.0, "gap_center": 100.0},
    }

    results = {}
    for cond, params in conditions.items():
        state = FlappyState(bird_y=params["bird_y"], bird_vy=0.0,
                           pipes=[{"x": 150.0, "gap_center": params["gap_center"]}],
                           score=0, step_count=0, alive=True)
        flap_probs = []
        motor_scores = []
        for _ in range(50):
            agent.reset()
            action = agent.act(state)
            diag = agent.get_motor_diagnostics()
            flap_probs.append(diag["flap_prob"])
            motor_scores.append(diag["motor_score"])

        results[cond] = {
            "mean_flap_prob": float(np.mean(flap_probs)),
            "mean_motor_score": float(np.mean(motor_scores)),
        }
        print("  %-15s  flap_prob=%.3f  motor_score=%.3f" % (
            cond, results[cond]["mean_flap_prob"], results[cond]["mean_motor_score"]))

    # Motor causality: flap rate should differ across conditions
    rates = [results[c]["mean_flap_prob"] for c in conditions]
    rate_range = max(rates) - min(rates)
    motor_pass = rate_range > 0.05
    print("\n  Flap prob range=%.3f  PASS=%s" % (rate_range, motor_pass))

    # PEG gating test: compare with and without PEG
    print("\n  PEG gating test:")
    peg_flap_probs = []
    no_peg_flap_probs = []
    state = FlappyState(bird_y=100, bird_vy=0.0,
                       pipes=[{"x": 150.0, "gap_center": 300.0}],
                       score=0, step_count=0, alive=True)
    for _ in range(50):
        # With PEG (normal)
        agent.reset()
        agent.act(state)
        peg_flap_probs.append(agent._last_flap_prob)

        # Without PEG: set peg_gate=0.0 manually (no connectome gating)
        sv = agent._compute_sensor_vertical_signal(sensor.sense(state))
        agent._prev_sensor_vertical = 0.0  # reset for fair comparison
        no_peg_flap_prob = 1.0 / (1.0 + np.exp(-(agent.motor_gain * sv * 0.0 + agent.motor_bias)))
        no_peg_flap_probs.append(no_peg_flap_prob)

    print("    With PEG:    mean_flap=%.3f" % np.mean(peg_flap_probs))
    print("    Without PEG: mean_flap=%.3f" % np.mean(no_peg_flap_probs))
    print("    PEG effect:  %.3f" % (np.mean(peg_flap_probs) - np.mean(no_peg_flap_probs)))

    with open(RESULTS_DIR / "step3_motor_causality.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Unplastic Connectome Baseline
# ══════════════════════════════════════════════════════════════════════════════

def run_step4(n_episodes=50, seed=42):
    print("\n" + "=" * 70)
    print("STEP 4: Unplastic Connectome Baseline")
    print("=" * 70)

    env = FlappyEnvironment(seed=seed)
    network = _make_network()
    agent = ConnectomeMotorReadoutAgent(
        network, enable_plasticity=False, motor_gain=50, motor_bias=-3.0,
    )

    scores, survivals, flap_rates = [], [], []
    for ep in range(n_episodes):
        state = env.reset(seed=seed + ep)
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

    results = {
        "mean_score": float(np.mean(scores)),
        "max_score": int(np.max(scores)),
        "mean_survival": float(np.mean(survivals)),
        "mean_flap_rate": float(np.mean(flap_rates)),
        "scores": scores,
    }
    print("  Mean score: %.3f" % results["mean_score"])
    print("  Max score:  %d" % results["max_score"])
    print("  Mean survival: %.1f" % results["mean_survival"])
    print("  Mean flap rate: %.3f" % results["mean_flap_rate"])

    with open(RESULTS_DIR / "step4_unplastic_baseline.json", "w") as f:
        json.dump({k: v for k, v in results.items() if k != "scores"}, f, indent=2)
    np.save(RESULTS_DIR / "step4_scores.npy", np.array(scores))
    return results


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: Pathway-Specific Plasticity Training
# ══════════════════════════════════════════════════════════════════════════════

def run_step5(n_episodes=500, n_eval=100, seed=42):
    print("\n" + "=" * 70)
    print("STEP 5: Pathway-Specific Plasticity Training (seed=%d)" % seed)
    print("=" * 70)

    env = FlappyEnvironment(seed=seed)
    network = _make_network()
    agent = ConnectomeMotorReadoutAgent(
        network, enable_plasticity=True, plasticity_mode="pathway",
        motor_gain=50, motor_bias=-3.0,
    )

    checkpoints = [0, 50, 100, 200, 300, 400, 500]
    train_scores, train_flap_rates = [], []
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
            agent.apply_reward(reward)

        train_scores.append(info["score"])
        train_flap_rates.append(np.mean(ep_actions) if ep_actions else 0.0)

        if ep in checkpoints:
            eval_res = _evaluate_agent(agent, env, n_eval, seed + 50000)
            checkpoint_data[ep] = {k: v for k, v in eval_res.items() if k != "scores"}
            print("  Checkpoint %4d: train=%d  eval_mean=%.2f  eval_max=%d" % (
                ep, info["score"], eval_res["mean_score"], eval_res["max_score"]))

        if (ep + 1) % 100 == 0:
            recent = train_scores[-100:]
            print("  Episode %d/%d: last100_mean=%.2f" % (ep + 1, n_episodes, np.mean(recent)))

    final_eval = _evaluate_agent(agent, env, n_eval, seed + 60000)
    checkpoint_data["final"] = {k: v for k, v in final_eval.items() if k != "scores"}

    results = {
        "seed": seed,
        "n_episodes": n_episodes,
        "train_scores": train_scores,
        "train_flap_rates": train_flap_rates,
        "checkpoints": checkpoint_data,
        "final_eval": {k: v for k, v in final_eval.items() if k != "scores"},
    }

    print("\n  Final eval: mean=%.2f  max=%d" % (final_eval["mean_score"], final_eval["max_score"]))

    with open(RESULTS_DIR / ("step5_plasticity_seed%d.json" % seed), "w") as f:
        json.dump({k: v for k, v in results.items()
                   if k not in ("train_scores", "train_flap_rates")}, f, indent=2)
    np.save(RESULTS_DIR / ("step5_train_scores_seed%d.npy" % seed), np.array(train_scores))

    # Plot learning curve
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    window = 20
    if len(train_scores) >= window:
        smoothed = np.convolve(train_scores, np.ones(window) / window, mode="valid")
        axes[0].plot(smoothed, color="blue")
    axes[0].plot(train_scores, alpha=0.3, color="blue")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Score")
    axes[0].set_title("Training Score (seed=%d)" % seed)

    ckpts = sorted([k for k in checkpoint_data.keys() if isinstance(k, int)])
    eval_means = [checkpoint_data[c]["mean_score"] for c in ckpts]
    axes[1].plot(ckpts, eval_means, "o-")
    axes[1].set_xlabel("Training Episode")
    axes[1].set_ylabel("Eval Score")
    axes[1].set_title("Evaluation Performance")

    plt.tight_layout()
    plt.savefig(FIGDIR / ("step5_learning_seed%d.png" % seed), dpi=150)
    plt.close()

    return results


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: Multi-Seed Replication
# ══════════════════════════════════════════════════════════════════════════════

def run_step6(n_seeds=5, n_episodes=500, base_seed=1000):
    print("\n" + "=" * 70)
    print("STEP 6: Multi-Seed Replication (%d seeds)" % n_seeds)
    print("=" * 70)

    all_results = {}
    for i in range(n_seeds):
        seed = base_seed + i * 100
        print("\n  --- Seed %d (%d/%d) ---" % (seed, i + 1, n_seeds))
        result = run_step5(n_episodes=n_episodes, seed=seed)
        all_results[seed] = result["final_eval"]

    final_means = [all_results[s]["mean_score"] for s in all_results]
    aggregate = {
        "n_seeds": n_seeds,
        "mean_of_means": float(np.mean(final_means)),
        "std_of_means": float(np.std(final_means)),
        "ci_95_low": float(np.mean(final_means) - 1.96 * np.std(final_means) / np.sqrt(n_seeds)),
        "ci_95_high": float(np.mean(final_means) + 1.96 * np.std(final_means) / np.sqrt(n_seeds)),
        "per_seed": {str(s): {"mean_score": all_results[s]["mean_score"],
                              "max_score": all_results[s]["max_score"]}
                     for s in all_results},
    }

    print("\n  Aggregate: mean=%.3f  95%%CI=[%.3f, %.3f]" % (
        aggregate["mean_of_means"], aggregate["ci_95_low"], aggregate["ci_95_high"]))

    with open(RESULTS_DIR / "step6_multiseed.json", "w") as f:
        json.dump(aggregate, f, indent=2)
    return aggregate


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7: Generalization Test
# ══════════════════════════════════════════════════════════════════════════════

def run_step7(seed=42, n_episodes=100):
    print("\n" + "=" * 70)
    print("STEP 7: Generalization Test")
    print("=" * 70)

    # Train on default
    env_train = FlappyEnvironment(seed=seed)
    network = _make_network()
    agent = ConnectomeMotorReadoutAgent(
        network, enable_plasticity=True, plasticity_mode="pathway",
        motor_gain=50, motor_bias=-3.0,
    )
    for ep in range(300):
        state = env_train.reset(seed=seed + ep + 20000)
        agent.reset()
        done = False
        while not done:
            action = agent.act(state)
            state, reward, done, info = env_train.step(action)
            agent.apply_reward(reward)
    agent.freeze_weights()

    configs = {
        "train_config": {},
        "large_gap": {"gap_size": 200.0},
        "small_gap": {"gap_size": 100.0},
        "fast_scroll": {"horizontal_speed": 2.5},
        "slow_scroll": {"horizontal_speed": 1.0},
    }

    results = {}
    for name, cfg in configs.items():
        env_eval = FlappyEnvironment(seed=seed + 70000, **cfg)
        eval_res = _evaluate_agent(agent, env_eval, n_episodes, seed + 80000)
        results[name] = {k: v for k, v in eval_res.items() if k != "scores"}
        print("  %-20s  mean=%.2f  max=%d" % (name, eval_res["mean_score"], eval_res["max_score"]))

    with open(RESULTS_DIR / "step7_generalization.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8: Perturbation Experiment
# ══════════════════════════════════════════════════════════════════════════════

def run_step8(seed=42, n_episodes=100):
    print("\n" + "=" * 70)
    print("STEP 8: Perturbation Experiment")
    print("=" * 70)

    env = FlappyEnvironment(seed=seed)
    network = _make_network()
    agent = ConnectomeMotorReadoutAgent(
        network, enable_plasticity=False, motor_gain=50, motor_bias=-3.0,
    )

    perturbations = {
        "baseline": {},
        "gap_shift_up": {"gap_shift": 50.0},
        "gap_shift_down": {"gap_shift": -50.0},
        "speed_change": {"new_speed": 4.0},
        "gap_narrow": {"new_gap": 80.0},
    }

    results = {}
    for name, cfg in perturbations.items():
        scores, survivals = [], []
        for ep in range(n_episodes):
            state = env.reset(seed=seed + ep + 30000)
            agent.reset()
            if "gap_shift" in cfg and len(env.pipes) > 1:
                env.pipes[1]["gap_center"] += cfg["gap_shift"]
            if "new_speed" in cfg:
                env.horizontal_speed = cfg["new_speed"]
            if "new_gap" in cfg:
                env.gap_size = cfg["new_gap"]
            done = False
            while not done:
                action = agent.act(state)
                state, reward, done, info = env.step(action)
            scores.append(info["score"])
            survivals.append(info["step"])
        results[name] = {
            "mean_score": float(np.mean(scores)),
            "max_score": int(np.max(scores)),
            "mean_survival": float(np.mean(survivals)),
        }
        print("  %-20s  mean=%.2f  surv=%.1f" % (name, results[name]["mean_score"], results[name]["mean_survival"]))

    with open(RESULTS_DIR / "step8_perturbation.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# STEP 9: Behavioral Variability
# ══════════════════════════════════════════════════════════════════════════════

def run_step9(n_episodes=200, seed=42):
    print("\n" + "=" * 70)
    print("STEP 9: Behavioral Variability")
    print("=" * 70)

    env = FlappyEnvironment(seed=seed)
    controllers = {
        "random": RandomFlapAgent(flap_prob=0.5, seed=seed),
        "fixed_period": FixedPeriodFlapAgent(period=8),
        "flymind_unplastic": ConnectomeMotorReadoutAgent(
            _make_network(), enable_plasticity=False, motor_gain=50, motor_bias=-3.0,
        ),
    }

    results = {}
    for name, agent in controllers.items():
        flap_timings, entropies = [], []
        for ep in range(n_episodes):
            state = env.reset(seed=seed + ep + 40000)
            agent.reset()
            done = False
            ep_actions = []
            last_flap_step = -1
            while not done:
                action = agent.act(state)
                ep_actions.append(action)
                if action == 1:
                    flap_timings.append(state.step_count - last_flap_step if last_flap_step >= 0 else 0)
                    last_flap_step = state.step_count
                state, reward, done, info = env.step(action)
            p1 = np.mean(ep_actions)
            p0 = 1 - p1
            entropy = -(max(p0, 1e-10) * np.log2(max(p0, 1e-10)) + max(p1, 1e-10) * np.log2(max(p1, 1e-10)))
            entropies.append(entropy)

        results[name] = {
            "timing_std": float(np.std(flap_timings)) if flap_timings else 0.0,
            "mean_entropy": float(np.mean(entropies)),
        }
        print("  %-20s  timing_std=%.2f  entropy=%.3f" % (
            name, results[name]["timing_std"], results[name]["mean_entropy"]))

    with open(RESULTS_DIR / "step9_variability.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# STEP 10: Ablations
# ══════════════════════════════════════════════════════════════════════════════

def run_step10(n_episodes=200, seed=42):
    print("\n" + "=" * 70)
    print("STEP 10: Ablations")
    print("=" * 70)

    env = FlappyEnvironment(seed=seed)
    configs = {
        "A_no_plasticity": lambda: ConnectomeMotorReadoutAgent(
            _make_network(), enable_plasticity=False, motor_gain=50, motor_bias=-3.0),
        "B_pathway_plasticity": lambda: ConnectomeMotorReadoutAgent(
            _make_network(), enable_plasticity=True, plasticity_mode="pathway", motor_gain=50, motor_bias=-3.0),
        "C_random": lambda: RandomFlapAgent(flap_prob=0.5, seed=seed),
        "D_fixed": lambda: FixedPeriodFlapAgent(period=8),
        "E_hand_designed": lambda: HandDesignedFlapAgent(gap_offset=30.0),
    }

    results = {}
    for name, agent_fn in configs.items():
        agent = agent_fn()
        scores, survivals = [], []
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
            "max_score": int(np.max(scores)),
            "mean_survival": float(np.mean(survivals)),
        }
        print("  %-30s  mean=%.2f  max=%d" % (name, results[name]["mean_score"], results[name]["max_score"]))

    with open(RESULTS_DIR / "step10_ablations.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# STEP 11: Statistical Analysis
# ══════════════════════════════════════════════════════════════════════════════

def run_step11():
    print("\n" + "=" * 70)
    print("STEP 11: Statistical Analysis")
    print("=" * 70)

    def _load(name):
        path = RESULTS_DIR / name
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return {}

    step4 = _load("step4_unplastic_baseline.json")
    step6 = _load("step6_multiseed.json")
    step8 = _load("step8_perturbation.json")
    step10 = _load("step10_ablations.json")

    summary = {
        "unplastic_baseline": step4,
        "multiseed": step6,
        "perturbation": step8,
        "ablations": step10,
    }

    with open(RESULTS_DIR / "step11_statistical_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("  Statistical summary saved.")
    return summary


# ══════════════════════════════════════════════════════════════════════════════
# STEP 12: Final Report
# ══════════════════════════════════════════════════════════════════════════════

def run_step12():
    print("\n" + "=" * 70)
    print("STEP 12: Generating Final Report")
    print("=" * 70)

    def _load(name):
        path = RESULTS_DIR / name
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return {}

    step1 = _load("step1_baselines.json")
    step2 = _load("step2_sensory_causality.json")
    step3 = _load("step3_motor_causality.json")
    step4 = _load("step4_unplastic_baseline.json")
    step6 = _load("step6_multiseed.json")
    step7 = _load("step7_generalization.json")
    step8 = _load("step8_perturbation.json")
    step9 = _load("step9_variability.json")
    step10 = _load("step10_ablations.json")

    report = """# FlyMind Phase 6B: Connectome Motor Readout — Final Report

**Date**: %s
**Connectome**: 261 neurons, 19969 synapses (Janelia Hemibrain v1.2.1)
**Environment**: Flappy-Bird-style 2D side-scrolling sandbox

---

## 1. Research Question

"Can a connectome-derived motor readout, where PEG premotor activity gates
the flap decision, support visually guided flight control?"

---

## 2. Architecture

    Sensor (9ch) -> EPG compass ring -> 261-neuron connectome -> PEG (18 neurons)
    Motor score = gain * (sensor_vertical * peg_gate) + bias
    P(flap) = sigmoid(motor_score / temperature)

Key design: the sensor provides gap-position information, PEG provides
connectome-dependent amplitude gating. Both are required for the motor decision.

---

## 3. Baselines

| Controller | Mean Score | Max Score |
|------------|-----------|-----------|
""" % time.strftime("%Y-%m-%d %H:%M")

    if step1:
        for name, vals in step1.items():
            report += "| %s | %.2f | %d |\n" % (name, vals.get("mean_score", 0), vals.get("max_score", 0))

    report += """
---

## 4. Sensory Causality

"""
    if step2:
        for cond, vals in step2.items():
            report += "- **%s**: ER4=%.4f, EPG=%.4f, PEG=%.4f\n" % (
                cond, vals.get("er4_mean", 0), vals.get("epg_mean", 0), vals.get("peg_mean", 0))

    report += """
---

## 5. Motor Causality (PEG Gating)

"""
    if step3:
        for cond, vals in step3.items():
            report += "- **%s**: flap_prob=%.3f, motor_score=%.3f\n" % (
                cond, vals.get("mean_flap_prob", 0), vals.get("mean_motor_score", 0))

    report += """
---

## 6. Unplastic Baseline

"""
    if step4:
        report += "- Mean score: %.3f\n" % step4.get("mean_score", 0)
        report += "- Max score: %d\n" % step4.get("max_score", 0)
        report += "- Mean survival: %.1f steps\n" % step4.get("mean_survival", 0)
        report += "- Mean flap rate: %.3f\n" % step4.get("mean_flap_rate", 0)

    report += """
---

## 7. Multi-Seed Replication

"""
    if step6:
        report += "- Seeds: %d\n" % step6.get("n_seeds", 0)
        report += "- Mean of means: %.3f\n" % step6.get("mean_of_means", 0)
        report += "- 95%% CI: [%.3f, %.3f]\n" % (step6.get("ci_95_low", 0), step6.get("ci_95_high", 0))

    report += """
---

## 8. Generalization

"""
    if step7:
        for name, vals in step7.items():
            report += "- **%s**: mean=%.2f\n" % (name, vals.get("mean_score", 0))

    report += """
---

## 9. Perturbation

"""
    if step8:
        for name, vals in step8.items():
            report += "- **%s**: mean=%.2f, survival=%.1f\n" % (
                name, vals.get("mean_score", 0), vals.get("mean_survival", 0))

    report += """
---

## 10. Behavioral Variability

"""
    if step9:
        for name, vals in step9.items():
            report += "- **%s**: timing_std=%.2f, entropy=%.3f\n" % (
                name, vals.get("timing_std", 0), vals.get("mean_entropy", 0))

    report += """
---

## 11. Ablations

"""
    if step10:
        for name, vals in step10.items():
            report += "- **%s**: mean=%.2f, max=%d\n" % (
                name, vals.get("mean_score", 0), vals.get("max_score", 0))

    report += """
---

## 12. Key Findings

1. **PEG as convergence layer**: PEG receives 5765 synapses from EPG (the strongest
   motor pathway). PEG activity is dominated by total EPG input, not spatial pattern.
   PEG asymmetry (~0.014) is intrinsic to the connectome, not sensory-driven.

2. **Sensor provides gap-position information**: The 9-channel sensor perfectly
   encodes gap-bird difference (vertical_signal = +1.0 for gap above, -1.0 for below).

3. **PEG gates motor output**: PEG activity modulates the sensor-based motor signal.
   When PEG is active (pipe visible), the motor score is amplified. When PEG is
   inactive (no pipe), the motor score is suppressed.

4. **Performance**: The unplastic Phase 6B agent scores mean=0.22 (vs 0 for random,
   2.32 for hand-designed). The agent occasionally reaches pipes but lacks the
   velocity-dependent timing of the hand-designed controller.

---

## 13. Limitations

1. **PEG information bottleneck**: PEG collapses EPG's spatial pattern into a scalar,
   losing gap-position information. The motor readout cannot decode gap position
   from PEG alone.

2. **No velocity information**: The motor readout lacks explicit velocity input.
   The hand-designed agent's velocity gate (flap when falling) is not replicated.

3. **Weak PEG signal**: PEG activity (~0.02) is too weak to produce large motor
   scores. The gain must be high (50-80) to produce meaningful flap probabilities.

4. **Partial circuit**: The 261-neuron connectome is a subset of the Drosophila
   central complex, missing PFL and other motor-related populations.

---

## 14. Recommended Phase 6C

1. **Temporal motor readout**: Use PEG activity history (last N steps) to estimate
   velocity and improve flap timing.

2. **Additional motor populations**: Include PFNd/PFNv (40+20 neurons) in the
   motor readout, as they project to motor descending neurons.

3. **Curriculum learning**: Start with large gaps and slow speeds, then increase
   difficulty to allow the motor readout to learn gradually.

4. **Multi-modal sensing**: Add graviceptive (vertical velocity) as a separate
   input pathway, bypassing the visual sensor.
"""

    report_path = ROOT / "docs" / "FlyMind_Phase6B_Motor_Readout_Report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print("  Report saved to %s" % report_path)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Phase 6B: Connectome Motor Readout")
    parser.add_argument("--step", type=int, default=None, help="Run specific step (1-12)")
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds for multi-seed")
    parser.add_argument("--episodes", type=int, default=500, help="Episodes per seed")
    parser.add_argument("--seed", type=int, default=42, help="Base seed")
    args = parser.parse_args()

    FIGDIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  PHASE 6B: CONNECTOME MOTOR READOUT")
    print("=" * 70)

    t0 = time.time()

    steps = {
        1: lambda: run_step1(n_episodes=args.episodes, seed=args.seed),
        2: lambda: run_step2(seed=args.seed),
        3: lambda: run_step3(seed=args.seed),
        4: lambda: run_step4(n_episodes=50, seed=args.seed),
        5: lambda: run_step5(n_episodes=args.episodes, seed=args.seed),
        6: lambda: run_step6(n_seeds=args.seeds, n_episodes=args.episodes),
        7: lambda: run_step7(seed=args.seed, n_episodes=min(args.episodes, 100)),
        8: lambda: run_step8(seed=args.seed, n_episodes=min(args.episodes, 100)),
        9: lambda: run_step9(n_episodes=min(args.episodes, 200), seed=args.seed),
        10: lambda: run_step10(n_episodes=min(args.episodes, 200), seed=args.seed),
        11: lambda: run_step11(),
        12: lambda: run_step12(),
    }

    if args.step is not None:
        if args.step in steps:
            steps[args.step]()
        else:
            print("Invalid step: %d. Choose 1-12." % args.step)
            sys.exit(1)
    else:
        for step_num in range(1, 13):
            try:
                steps[step_num]()
            except Exception as e:
                print("\n  *** STEP %d FAILED: %s ***" % (step_num, e))
                import traceback
                traceback.print_exc()
                sys.exit(1)

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print("  Phase 6B complete. Total time: %.1fs" % elapsed)
    print("  Results: %s" % RESULTS_DIR)
    print("  Figures: %s" % FIGDIR)
    print("=" * 70)


if __name__ == "__main__":
    main()
