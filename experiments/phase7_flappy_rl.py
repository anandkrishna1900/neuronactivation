"""
Phase 7: FlyMind Reinforcement Learning — Long-Horizon Headless Trainer.

This script runs the complete Phase 7 RL experiment:
    - FlyMind connectome agent with reward-modulated local plasticity
    - Optional curriculum learning
    - Checkpointing + resume
    - Periodic frozen evaluation on validation seeds
    - PPO baseline (pure-numpy fallback)
    - Ablation experiments

Usage examples:
    # Standard FlyMind RL training
    python experiments/phase7_flappy_rl.py --episodes 10000 --seed 42

    # With curriculum
    python experiments/phase7_flappy_rl.py --episodes 5000 --curriculum --seed 42

    # Resume from checkpoint
    python experiments/phase7_flappy_rl.py --resume results/phase7/checkpoints/ep_1000.npz

    # PPO baseline
    python experiments/phase7_flappy_rl.py --mode ppo --episodes 5000 --seed 42

    # Ablation (no plasticity)
    python experiments/phase7_flappy_rl.py --ablation no_plasticity --episodes 2000

    # Run all ablations
    python experiments/phase7_flappy_rl.py --run-ablations --episodes 1000
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Path setup
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# FlyMind imports
from flymind.brain.network import NeuralNetwork
from flymind.connectome.loader import ConnectomeLoader
from flymind.environment.flappy import FlappyEnvironment, GAP_SIZE, HORIZONTAL_SPEED, PIPE_SPACING
from flymind.environment.flappy_sensor import FlappyVisualSensor
from flymind.environment.flappy_reward import FlappyRewardShaper
from flymind.environment.curriculum import CurriculumManager
from flymind.agent.flappy_agent_phase7 import FlyMindRLAgent, make_rewired_network
from flymind.agent.flappy_agent import (
    HandDesignedFlapAgent, RandomFlapAgent, FixedPeriodFlapAgent
)
from flymind.agent.flappy_agent_phase6b import ConnectomeMotorReadoutAgent

# Paths
CONNECTOME_PATH = ROOT / "data" / "processed" / "cx_heading_v1.json"
RESULTS_DIR     = ROOT / "results" / "phase7"
CKPT_DIR        = RESULTS_DIR / "checkpoints"
METRICS_DIR     = RESULTS_DIR / "metrics"
FIGURES_DIR     = RESULTS_DIR / "figures"
LOGS_DIR        = RESULTS_DIR / "logs"
REPLAY_DIR      = RESULTS_DIR / "replays"
VIDEO_DIR       = RESULTS_DIR / "videos"

for d in [CKPT_DIR, METRICS_DIR, FIGURES_DIR, LOGS_DIR, REPLAY_DIR, VIDEO_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Training seeds: 0–999 | Validation: 1000–1999 | Test: 2000–2999
TRAIN_SEED_OFFSET = 0
VALID_SEED_OFFSET = 1000
TEST_SEED_OFFSET  = 2000


# ── Connectome factory ────────────────────────────────────────────────────────
_CACHED_GRAPH = None

def get_connectome_graph():
    global _CACHED_GRAPH
    if _CACHED_GRAPH is None:
        if not CONNECTOME_PATH.exists():
            raise FileNotFoundError(f"Connectome not found: {CONNECTOME_PATH}")
        _CACHED_GRAPH = ConnectomeLoader.load_from_json(CONNECTOME_PATH)
    return _CACHED_GRAPH


def make_network(seed: Optional[int] = None) -> NeuralNetwork:
    """Create a fresh NeuralNetwork from the connectome graph."""
    graph = get_connectome_graph()
    return NeuralNetwork(graph, synapse_scale=0.01)


# ── PPO Baseline (pure-numpy, no external RL library) ─────────────────────────
class PPOBaseline:
    """
    Minimal pure-numpy PPO baseline for benchmarking.

    This is explicitly NOT the FlyMind biological agent.
    It uses a conventional MLP policy with backpropagation.
    Its purpose is to answer: 'How does biological plasticity compare
    with conventional RL on this task?'
    """

    def __init__(
        self,
        obs_dim: int = 9,
        hidden_dim: int = 64,
        lr: float = 3e-4,
        gamma: float = 0.99,
        clip_eps: float = 0.2,
        n_epochs: int = 4,
        seed: Optional[int] = None,
    ):
        rng = np.random.default_rng(seed)
        # MLP: obs -> hidden -> hidden -> value+logit
        self.W1 = rng.normal(0, 0.1, (obs_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.normal(0, 0.1, (hidden_dim, hidden_dim))
        self.b2 = np.zeros(hidden_dim)
        self.Wp = rng.normal(0, 0.01, (hidden_dim, 1))  # policy logit
        self.Wv = rng.normal(0, 0.01, (hidden_dim, 1))  # value head
        self.bp = np.zeros(1)
        self.bv = np.zeros(1)
        self.lr = lr
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.n_epochs = n_epochs
        self._rng = rng

        # Buffers
        self._obs_buf: List[np.ndarray] = []
        self._act_buf: List[int] = []
        self._rew_buf: List[float] = []
        self._lp_buf: List[float] = []
        self._val_buf: List[float] = []

    @property
    def name(self) -> str:
        return "PPO_Baseline (non-biological)"

    def _forward(self, obs: np.ndarray):
        h1 = np.tanh(obs @ self.W1 + self.b1)
        h2 = np.tanh(h1 @ self.W2 + self.b2)
        logit = float(h2 @ self.Wp + self.bp)
        value = float(h2 @ self.Wv + self.bv)
        prob = 1.0 / (1.0 + np.exp(-logit))
        return prob, value, h2

    def act(self, state) -> int:
        from flymind.environment.flappy_sensor import FlappyVisualSensor
        if not hasattr(self, "_sensor"):
            self._sensor = FlappyVisualSensor()
        obs = self._sensor.sense(state)
        prob, value, _ = self._forward(obs)
        p_safe = float(np.clip(prob, 1e-6, 1.0 - 1e-6))
        action = int(self._rng.binomial(1, p_safe))
        lp = np.log(p_safe) if action == 1 else np.log(1.0 - p_safe)
        self._obs_buf.append(obs)
        self._act_buf.append(action)
        self._lp_buf.append(float(lp))
        self._val_buf.append(value)
        return action

    def apply_step_reward(self, reward: float) -> None:
        self._rew_buf.append(reward)

    def apply_reward(self, reward: float) -> None:
        self.apply_step_reward(reward)

    def end_episode(self, total_reward: float) -> Dict[str, float]:
        n = len(self._rew_buf)
        if n == 0:
            return {}

        # Compute discounted returns
        returns = np.zeros(n)
        G = 0.0
        for t in reversed(range(n)):
            G = self._rew_buf[t] + self.gamma * G
            returns[t] = G

        # Normalize
        ret_mean, ret_std = returns.mean(), returns.std() + 1e-8
        returns = (returns - ret_mean) / ret_std

        obs_arr = np.array(self._obs_buf)
        act_arr = np.array(self._act_buf)
        old_lp  = np.array(self._lp_buf)

        # Mini PPO update (n_epochs over the episode buffer)
        for _ in range(self.n_epochs):
            for t in range(n):
                obs = obs_arr[t]
                act = act_arr[t]
                ret = returns[t]
                adv = ret - float(self._val_buf[t])

                prob, val, h2 = self._forward(obs)
                p_safe = float(np.clip(prob, 1e-6, 1.0 - 1e-6))
                new_lp = np.log(p_safe) if act == 1 else np.log(1.0 - p_safe)

                # PPO clip ratio
                ratio = np.exp(new_lp - old_lp[t])
                clip_ratio = np.clip(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                policy_loss = -min(ratio * adv, clip_ratio * adv)
                value_loss  = (val - ret) ** 2

                # Backprop (standard gradient through MLP — NOT through connectome)
                # Policy head gradient
                d_logit = (act - prob) * self.lr * adv * clip_ratio / max(ratio, 1e-6)
                d_Wp = np.outer(h2, [d_logit])
                d_bp = np.array([d_logit])

                # Value head gradient
                d_val = -2.0 * (val - ret) * self.lr
                d_Wv  = np.outer(h2, [d_val])
                d_bv  = np.array([d_val])

                # Hidden gradients
                d_h2 = (d_logit * self.Wp.ravel() + d_val * self.Wv.ravel()) * (1 - h2**2)
                d_W2 = np.outer(self.W2 @ np.zeros_like(h2), d_h2)  # simplified
                h1 = np.tanh(obs @ self.W1 + self.b1)
                d_W2 = np.outer(h1, d_h2) * self.lr
                d_b2 = d_h2 * self.lr
                d_h1 = (d_h2 @ self.W2.T) * (1 - h1**2)
                d_W1 = np.outer(obs, d_h1) * self.lr
                d_b1 = d_h1 * self.lr

                self.Wp += d_Wp
                self.bp += d_bp
                self.Wv += d_Wv
                self.bv += d_bv
                self.W2 += d_W2
                self.b2 += d_b2
                self.W1 += d_W1
                self.b1 += d_b1

        self._obs_buf.clear()
        self._act_buf.clear()
        self._rew_buf.clear()
        self._lp_buf.clear()
        self._val_buf.clear()
        return {}

    def reset(self) -> None:
        self._obs_buf.clear()
        self._act_buf.clear()
        self._rew_buf.clear()
        self._lp_buf.clear()
        self._val_buf.clear()

    def freeze_weights(self) -> None:
        pass

    def get_diagnostics(self) -> Dict[str, Any]:
        return {"note": "PPO Baseline — non-biological MLP"}

    def save_checkpoint(self, path: str) -> None:
        np.savez_compressed(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
                            Wp=self.Wp, bp=self.bp, Wv=self.Wv, bv=self.bv)

    def load_checkpoint(self, path: str) -> None:
        d = np.load(path)
        self.W1, self.b1 = d["W1"], d["b1"]
        self.W2, self.b2 = d["W2"], d["b2"]
        self.Wp, self.bp = d["Wp"], d["bp"]
        self.Wv, self.bv = d["Wv"], d["bv"]


# ── Agent factory ─────────────────────────────────────────────────────────────
def make_agent(mode: str, seed: int, ablation: str = "none", args=None):
    """Create the appropriate agent for the given training mode."""
    if mode == "ppo":
        return PPOBaseline(seed=seed)

    if mode in ("hand", "hand_designed"):
        return HandDesignedFlapAgent()

    if mode == "random":
        return RandomFlapAgent(seed=seed)

    if mode == "fixed":
        return FixedPeriodFlapAgent(period=8)

    if mode == "frozen_connectome":
        net = make_network()
        return FlyMindRLAgent(
            network=net,
            plasticity_mode="none",
            seed=seed,
        )

    if mode == "phase6b":
        net = make_network()
        return ConnectomeMotorReadoutAgent(network=net, enable_plasticity=False, seed=seed)

    # Phase 7 FlyMind RL agent — ablation variants
    net = make_network()

    ablation_kwargs = {}
    if ablation == "no_plasticity":
        ablation_kwargs["plasticity_mode"] = "none"
    elif ablation == "global_plasticity":
        ablation_kwargs["plasticity_mode"] = "global"
    elif ablation == "rewired":
        rng = np.random.default_rng(seed)
        net = make_rewired_network(net, rng)
        ablation_kwargs["plasticity_mode"] = "pathway"
    elif ablation == "no_epg":
        ablation_kwargs["ablate_epg"] = True
    elif ablation == "no_peg":
        ablation_kwargs["ablate_peg"] = True
    elif ablation == "no_pfnd":
        ablation_kwargs["ablate_pfnd"] = True
    elif ablation == "no_pfnv":
        ablation_kwargs["ablate_pfnv"] = True
    elif ablation == "no_temporal":
        ablation_kwargs["ablate_temporal"] = True

    lr   = getattr(args, "lr", 0.002) if args else 0.002
    elr  = getattr(args, "eligibility_decay", 0.90) if args else 0.90
    temp = getattr(args, "motor_temperature", 1.5) if args else 1.5
    pm   = ablation_kwargs.pop("plasticity_mode", "pathway")

    return FlyMindRLAgent(
        network=net,
        sensory_drive=25.0,
        sub_steps=10,
        learning_rate=lr,
        eligibility_decay=elr,
        plasticity_mode=pm,
        motor_temperature=temp,
        motor_lr=0.01,
        decoder_mode="stochastic",
        seed=seed,
        **ablation_kwargs,
    )


# ── Single episode runner ─────────────────────────────────────────────────────
def run_episode(
    agent,
    env: FlappyEnvironment,
    shaper: FlappyRewardShaper,
    seed: int,
    max_steps: int = 10000,
    training: bool = True,
    save_replay: bool = False,
    replay_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run one episode of Flappy Bird.

    Returns dict with episode metrics.
    """
    state = env.reset(seed=seed)
    agent.reset()

    total_reward = 0.0
    ep_steps     = 0
    done         = False

    # Replay buffers
    if save_replay:
        rbuf: Dict[str, List] = {k: [] for k in [
            "bird_y", "bird_vy", "actions", "flap_probs", "rewards",
            "scores", "sensor_activations", "epg_activity",
            "peg_mean", "pfnd_mean", "pfnv_mean", "motor_logit",
            "pipe_x_nearest", "pipe_gap_nearest",
        ]}

    while not done and ep_steps < max_steps:
        action = agent.act(state)
        next_state, env_reward, done, info = env.step(action)

        shaped_r = shaper.shape(env_reward, info, done)
        total_reward += shaped_r

        if training:
            agent.apply_step_reward(shaped_r)

        if save_replay:
            diag = agent.get_diagnostics() if hasattr(agent, "get_diagnostics") else {}
            rbuf["bird_y"].append(state.bird_y)
            rbuf["bird_vy"].append(state.bird_vy)
            rbuf["actions"].append(action)
            rbuf["flap_probs"].append(diag.get("flap_prob", 0.5))
            rbuf["rewards"].append(shaped_r)
            rbuf["scores"].append(state.score)
            rbuf["sensor_activations"].append(diag.get("sensor_activations", np.zeros(9)))
            epg_act = diag.get("epg_activity", np.zeros(1))
            rbuf["epg_activity"].append(np.mean(epg_act))
            rbuf["peg_mean"].append(diag.get("peg_mean", 0.0))
            rbuf["pfnd_mean"].append(diag.get("pfnd_mean", 0.0))
            rbuf["pfnv_mean"].append(diag.get("pfnv_mean", 0.0))
            rbuf["motor_logit"].append(diag.get("motor_logit", 0.0))
            # Nearest pipe (environment state — analysis only, not agent input)
            nearest_pipe = min(state.pipes, key=lambda p: abs(p["x"] - 80.0)) if state.pipes else {"x": 0, "gap_center": 200}
            rbuf["pipe_x_nearest"].append(nearest_pipe["x"])
            rbuf["pipe_gap_nearest"].append(nearest_pipe["gap_center"])

        state = next_state
        ep_steps += 1

    if training:
        reinforce_info = agent.end_episode(total_reward)
    else:
        reinforce_info = {}

    if save_replay and replay_path:
        np.savez_compressed(
            replay_path,
            bird_y=np.array(rbuf["bird_y"], dtype=np.float32),
            bird_vy=np.array(rbuf["bird_vy"], dtype=np.float32),
            actions=np.array(rbuf["actions"], dtype=np.int8),
            flap_probs=np.array(rbuf["flap_probs"], dtype=np.float32),
            rewards=np.array(rbuf["rewards"], dtype=np.float32),
            scores=np.array(rbuf["scores"], dtype=np.int32),
            sensor_activations=np.array(rbuf["sensor_activations"], dtype=np.float32),
            epg_activity=np.array(rbuf["epg_activity"], dtype=np.float32),
            peg_mean=np.array(rbuf["peg_mean"], dtype=np.float32),
            pfnd_mean=np.array(rbuf["pfnd_mean"], dtype=np.float32),
            pfnv_mean=np.array(rbuf["pfnv_mean"], dtype=np.float32),
            motor_logit=np.array(rbuf["motor_logit"], dtype=np.float32),
            pipe_x_nearest=np.array(rbuf["pipe_x_nearest"], dtype=np.float32),
            pipe_gap_nearest=np.array(rbuf["pipe_gap_nearest"], dtype=np.float32),
            seed=np.array([seed]),
        )

    return {
        "score":        state.score,
        "steps":        ep_steps,
        "total_reward": total_reward,
        "done":         done,
        **reinforce_info,
    }


# ── Evaluation on held-out seeds ──────────────────────────────────────────────
def evaluate(
    agent,
    n_seeds: int = 20,
    seed_offset: int = VALID_SEED_OFFSET,
    max_steps: int = 5000,
    shaper: Optional[FlappyRewardShaper] = None,
) -> Dict[str, float]:
    """Run frozen evaluation on n_seeds held-out seeds."""
    if shaper is None:
        shaper = FlappyRewardShaper()

    agent.freeze_weights()
    scores, survivals, rewards = [], [], []
    for i in range(n_seeds):
        seed = seed_offset + i
        env = FlappyEnvironment(seed=seed, max_steps=max_steps)
        result = run_episode(agent, env, shaper, seed=seed, max_steps=max_steps, training=False)
        scores.append(result["score"])
        survivals.append(result["steps"])
        rewards.append(result["total_reward"])

    if hasattr(agent, "unfreeze_weights"):
        agent.unfreeze_weights()

    return {
        "eval_mean_score":    float(np.mean(scores)),
        "eval_median_score":  float(np.median(scores)),
        "eval_std_score":     float(np.std(scores)),
        "eval_max_score":     float(np.max(scores)),
        "eval_min_score":     float(np.min(scores)),
        "eval_mean_survival": float(np.mean(survivals)),
        "eval_mean_reward":   float(np.mean(rewards)),
        "eval_success_rate":  float(np.mean([s > 0 for s in scores])),
        "eval_n_seeds":       n_seeds,
    }


# ── Training Loop ─────────────────────────────────────────────────────────────
def _save_train_state(path: str, **kwargs) -> None:
    """Save training state dict to .npz."""
    np.savez_compressed(path, **{k: np.array(v) for k, v in kwargs.items()})


def _load_train_state(path: str) -> dict:
    """Load training state dict from .npz."""
    data = np.load(path, allow_pickle=True)
    return {k: data[k].item() if data[k].ndim == 0 else data[k] for k in data.files}


def train(args) -> None:
    """Main training loop with full pause/resume support."""
    run_name = (
        f"{args.mode}_{args.ablation}_s{args.seed}_"
        f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    # If resuming, derive run_name from the checkpoint path
    start_episode = 1
    if args.resume:
        resume_path = Path(args.resume)
        # Extract run_name from checkpoint filename: <run_name>_ep<N>.npz or <run_name>_final.npz
        stem = resume_path.stem
        for suffix in ["_best", "_final"]:
            if stem.endswith(suffix):
                run_name = stem[: -len(suffix)]
                break
        else:
            # Try to find the training state file
            state_path = resume_path.parent / f"{stem}_train_state.npz"
            if state_path.exists():
                run_name = stem.replace("_ep", "_").rsplit("_", 1)[0]
            else:
                run_name = stem

    print(f"\n{'='*70}")
    print(f"PHASE 7 FLYMIND RL — {run_name}")
    print(f"Mode: {args.mode} | Ablation: {args.ablation} | Episodes: {args.episodes}")
    print(f"{'='*70}")

    # Reward shaper
    shaper = FlappyRewardShaper(
        pipe_reward=args.pipe_reward,
        survival_reward=args.survival_reward,
        collision_penalty=args.collision_penalty,
    )

    # Curriculum
    curriculum = CurriculumManager(
        enabled=args.curriculum,
        start_stage=args.curriculum_start_stage,
    )

    # Agent
    agent = make_agent(args.mode, args.seed, args.ablation, args)

    # Resume from checkpoint
    best_score = 0
    best_ckpt_path = None
    if args.resume:
        print(f"[Resume] Loading checkpoint: {args.resume}")
        agent.load_checkpoint(args.resume)
        # Load training state
        state_path = str(Path(args.resume).with_suffix("").as_posix()) + "_train_state.npz"
        if Path(state_path).exists():
            ts = _load_train_state(state_path)
            start_episode = int(ts.get("episode", 0)) + 1
            best_score = int(ts.get("best_score", 0))
            best_ckpt_path = ts.get("best_ckpt_path", "")
            if isinstance(best_ckpt_path, np.ndarray):
                best_ckpt_path = str(best_ckpt_path)
            # Restore score history for moving averages
            saved_scores = ts.get("all_scores", np.array([]))
            saved_survivals = ts.get("all_survivals", np.array([]))
            saved_rewards = ts.get("all_rewards", np.array([]))
            print(f"[Resume] Resuming from episode {start_episode}, best_score={best_score}")
        else:
            print(f"[Resume] No training state found at {state_path}, starting fresh from loaded weights")

    agent_name = getattr(agent, "name", type(agent).__name__)
    print(f"Agent: {agent_name}")
    print(f"Reward: {shaper}")
    print(f"Curriculum: {'enabled' if args.curriculum else 'disabled'}")
    print(f"Starting episode: {start_episode}")

    # Metrics log — append if resuming
    log_path = METRICS_DIR / f"{run_name}_training.csv"
    csv_fields = [
        "episode", "seed", "score", "steps", "total_reward",
        "mean_score_50", "mean_survival_50",
        "eval_mean_score", "eval_mean_survival", "eval_success_rate",
        "curriculum_stage", "wall_time_s",
    ]
    write_header = not log_path.exists() or start_episode == 1
    log_f = open(log_path, "a" if not write_header else "w", newline="")
    writer = csv.DictWriter(log_f, fieldnames=csv_fields)
    if write_header:
        writer.writeheader()

    # Score tracking
    score_deque    = deque(maxlen=50)
    survival_deque = deque(maxlen=50)
    all_scores     = []
    all_survivals  = []
    all_rewards    = []
    if args.resume:
        try:
            state_path = str(Path(args.resume).with_suffix("").as_posix()) + "_train_state.npz"
            if Path(state_path).exists():
                ts = _load_train_state(state_path)
                all_scores = list(ts.get("all_scores", []))
                all_survivals = list(ts.get("all_survivals", []))
                all_rewards = list(ts.get("all_rewards", []))
        except Exception:
            pass
    # Rebuild deques from saved history
    for s in all_scores[-50:]:
        score_deque.append(s)
    for s in all_survivals[-50:]:
        survival_deque.append(s)

    t0 = time.time()

    import itertools
    ep_iter = range(start_episode, args.episodes + 1) if args.episodes > 0 else itertools.count(start_episode)

    # Save a helper checkpoint immediately on resume so user always has a restart point
    def _save_everything(ckpt_tag: str, ep: int):
        nonlocal best_ckpt_path
        ckpt_path = str(CKPT_DIR / f"{run_name}_{ckpt_tag}.npz")
        agent.save_checkpoint(ckpt_path)
        # Save training state
        state = {
            "episode":       ep,
            "seed":          args.seed,
            "best_score":    best_score,
            "best_ckpt_path": best_ckpt_path or "",
            "all_scores":    np.array(all_scores[-2000:]),  # keep last 2000 for resume
            "all_survivals": np.array(all_survivals[-2000:]),
            "all_rewards":   np.array(all_rewards[-2000:]),
            "run_name":      run_name,
        }
        state_path = str(CKPT_DIR / f"{run_name}_{ckpt_tag}_train_state.npz")
        _save_train_state(state_path, **state)
        return ckpt_path

    try:
        for ep in ep_iter:
            ep_seed = TRAIN_SEED_OFFSET + args.seed + ep
            env_kwargs = curriculum.get_env_kwargs(ep)
            env = FlappyEnvironment(seed=ep_seed, max_steps=args.max_steps, **env_kwargs)

            # Save replay at checkpoint intervals
            save_rep = (ep % args.checkpoint_interval == 0)
            rep_path = str(REPLAY_DIR / f"{run_name}_ep{ep:06d}.npz") if save_rep else None

            result = run_episode(
                agent, env, shaper, seed=ep_seed,
                max_steps=args.max_steps,
                training=(args.mode not in ("hand", "random", "fixed")),
                save_replay=save_rep,
                replay_path=rep_path,
            )

            score    = result["score"]
            steps    = result["steps"]
            ep_reward = result["total_reward"]

            score_deque.append(score)
            survival_deque.append(steps)
            all_scores.append(score)
            all_survivals.append(steps)
            all_rewards.append(ep_reward)

            # Curriculum progression
            stage_advanced = curriculum.record_episode(score)
            if stage_advanced:
                print(f"  [Curriculum] Advanced to: {curriculum.stage_name}")

            mean_s50  = float(np.mean(score_deque))
            mean_sv50 = float(np.mean(survival_deque))

            # Best model checkpoint
            if score > best_score:
                best_score = score
                best_ckpt_path = str(CKPT_DIR / f"{run_name}_best.npz")
                agent.save_checkpoint(best_ckpt_path)
                _save_train_state(
                    str(CKPT_DIR / f"{run_name}_best_train_state.npz"),
                    episode=ep, seed=args.seed, best_score=best_score,
                    best_ckpt_path=best_ckpt_path,
                    all_scores=np.array(all_scores[-2000:]),
                    all_survivals=np.array(all_survivals[-2000:]),
                    all_rewards=np.array(all_rewards[-2000:]),
                    run_name=run_name,
                )

            # Periodic checkpoint (also saves full state for resume)
            eval_stats = {}
            if ep % args.checkpoint_interval == 0:
                ckpt_path = _save_everything(f"ep{ep:06d}", ep)

                # Validation evaluation
                eval_stats = evaluate(
                    agent, n_seeds=args.eval_seeds,
                    seed_offset=VALID_SEED_OFFSET,
                    max_steps=args.max_steps,
                    shaper=shaper,
                )

                wall_t = time.time() - t0
                print(
                    f"  [Eval ep={ep:>5d}] "
                    f"mean_score={eval_stats['eval_mean_score']:.3f} "
                    f"success_rate={eval_stats['eval_success_rate']:.1%} "
                    f"mean_survival={eval_stats['eval_mean_survival']:.0f} "
                    f"wall={wall_t:.0f}s"
                )

            # Progress print
            if ep % max(1, args.episodes // 100) == 0 or ep <= 10:
                wall_t = time.time() - t0
                print(
                    f"  ep={ep:>6d} | score={score} | steps={steps} | "
                    f"reward={ep_reward:+.3f} | "
                    f"mean50={mean_s50:.2f} | "
                    f"best={best_score} | "
                    f"stage={curriculum.stage_name if args.curriculum else '-'} | "
                    f"t={wall_t:.0f}s"
                )

            # CSV log
            row = {
                "episode":           ep,
                "seed":              ep_seed,
                "score":             score,
                "steps":             steps,
                "total_reward":      ep_reward,
                "mean_score_50":     mean_s50,
                "mean_survival_50":  mean_sv50,
                "eval_mean_score":   eval_stats.get("eval_mean_score", ""),
                "eval_mean_survival": eval_stats.get("eval_mean_survival", ""),
                "eval_success_rate": eval_stats.get("eval_success_rate", ""),
                "curriculum_stage":  curriculum.current_stage_idx,
                "wall_time_s":       f"{time.time() - t0:.1f}",
            }
            writer.writerow(row)
            log_f.flush()
    except KeyboardInterrupt:
        print("\n[Ctrl+C] Saving progress before exit...")
    finally:
        # Always save latest checkpoint on exit (normal, error, or Ctrl+C)
        try:
            _save_everything("latest", ep if 'ep' in dir() else start_episode)
            print(f"[Saved] Latest checkpoint for resume.")
        except Exception:
            pass
        log_f.close()

    # Final evaluation
    print(f"\n{'='*70}")
    print("FINAL EVALUATION (validation seeds)")
    final_eval = evaluate(
        agent, n_seeds=50,
        seed_offset=VALID_SEED_OFFSET,
        max_steps=args.max_steps,
        shaper=shaper,
    )
    for k, v in final_eval.items():
        print(f"  {k}: {v}")

    # Save final checkpoint
    final_ckpt = str(CKPT_DIR / f"{run_name}_final.npz")
    if hasattr(agent, "save_checkpoint"):
        agent.save_checkpoint(final_ckpt)
        print(f"[Saved] Final checkpoint: {final_ckpt}")

    # Save summary JSON
    summary = {
        "run_name":          run_name,
        "mode":              args.mode,
        "ablation":          args.ablation,
        "seed":              args.seed,
        "episodes":          args.episodes,
        "training_episodes": len(all_scores),
        "train_mean_score":  float(np.mean(all_scores)),
        "train_median_score": float(np.median(all_scores)),
        "train_std_score":   float(np.std(all_scores)),
        "train_max_score":   int(np.max(all_scores)) if all_scores else 0,
        "best_score":        best_score,
        "best_checkpoint":   best_ckpt_path,
        "final_checkpoint":  final_ckpt,
        "final_eval":        final_eval,
        "curriculum_enabled": args.curriculum,
        "reward_config": {
            "pipe_reward":       args.pipe_reward,
            "survival_reward":   args.survival_reward,
            "collision_penalty": args.collision_penalty,
        },
    }
    summary_path = METRICS_DIR / f"{run_name}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Saved] Summary: {summary_path}")
    print(f"[Saved] Training log: {log_path}")
    print(f"\nFinal results:")
    print(f"  Train mean score (last 50): {float(np.mean(list(score_deque))):.3f}")
    print(f"  Val mean score (50 seeds):  {final_eval['eval_mean_score']:.3f}")
    print(f"  Val success rate:           {final_eval['eval_success_rate']:.1%}")
    print(f"  Best score achieved:        {best_score}")
    print(f"{'='*70}")


# ── Ablation suite ────────────────────────────────────────────────────────────
ABLATIONS = [
    "none",
    "no_plasticity",
    "global_plasticity",
    "rewired",
    "no_epg",
    "no_peg",
    "no_pfnd",
    "no_pfnv",
    "no_temporal",
]


def run_ablations(args) -> None:
    """Run all ablation conditions sequentially."""
    print(f"\n{'='*70}")
    print(f"PHASE 7 ABLATION SUITE — {args.episodes} episodes each")
    print(f"{'='*70}")
    results = {}
    for abl in ABLATIONS:
        print(f"\nAblation: {abl}")
        args_copy = argparse.Namespace(**vars(args))
        args_copy.ablation = abl
        args_copy.mode = "flymind"
        train(args_copy)
        results[abl] = {"done": True}

    print("\nAblation suite complete.")


# ── Passivity Validation ───────────────────────────────────────────────────────
def run_validate(args) -> bool:
    """Verify headless and instrumented runs produce identical trajectories."""
    print("\n" + "=" * 70)
    print("PHASE 7 PASSIVITY VALIDATION")
    print("=" * 70)
    shaper = FlappyRewardShaper()
    seeds = [42, 100, 777]
    all_pass = True

    for seed in seeds:
        # Run A: pure headless
        agent_a = make_agent("flymind", seed)
        env_a   = FlappyEnvironment(seed=seed, max_steps=500)
        result_a = run_episode(agent_a, env_a, shaper, seed=seed, max_steps=500, training=False)

        # Run B: same but with get_diagnostics called after each act
        agent_b = make_agent("flymind", seed)
        env_b   = FlappyEnvironment(seed=seed, max_steps=500)
        state_b = env_b.reset(seed=seed)
        agent_b.reset()
        actions_b, ys_b = [], []
        done_b = False
        steps_b = 0
        while not done_b and steps_b < 500:
            act_b = agent_b.act(state_b)
            _ = agent_b.get_diagnostics()   # <-- instrumented call
            state_b, _, done_b, info_b = env_b.step(act_b)
            actions_b.append(act_b)
            ys_b.append(state_b.bird_y)
            steps_b += 1

        # Run A actions
        agent_a2 = make_agent("flymind", seed)
        env_a2   = FlappyEnvironment(seed=seed, max_steps=500)
        state_a2 = env_a2.reset(seed=seed)
        agent_a2.reset()
        actions_a, ys_a = [], []
        done_a2 = False
        steps_a2 = 0
        while not done_a2 and steps_a2 < 500:
            act_a2 = agent_a2.act(state_a2)
            state_a2, _, done_a2, _ = env_a2.step(act_a2)
            actions_a.append(act_a2)
            ys_a.append(state_a2.bird_y)
            steps_a2 += 1

        min_len = min(len(actions_a), len(actions_b))
        action_delta = sum(a != b for a, b in zip(actions_a[:min_len], actions_b[:min_len]))
        max_y_delta  = max(abs(a - b) for a, b in zip(ys_a[:min_len], ys_b[:min_len])) if min_len else 0.0
        passed = (action_delta == 0 and max_y_delta < 1e-9)
        status = "[PASS]" if passed else "[FAIL]"
        print(f"Seed {seed:>4d}: {status} Steps={steps_a2} vs {steps_b} | "
              f"ActionDelta={action_delta} | MaxYDelta={max_y_delta:.2e}")
        if not passed:
            all_pass = False

    print("-" * 70)
    if all_pass:
        print("[SUCCESS] All seeds validated bit-for-bit! PASSIVITY GUARANTEE VERIFIED.")
    else:
        print("[FAILURE] Passivity guarantee VIOLATED. Check visualization layer.")
    print("=" * 70)
    return all_pass


# ── CLI ───────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase 7: FlyMind RL — Long-Horizon Biological Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python experiments/phase7_flappy_rl.py --episodes 10000 --seed 42
  python experiments/phase7_flappy_rl.py --episodes 5000 --curriculum
  python experiments/phase7_flappy_rl.py --mode ppo --episodes 5000
  python experiments/phase7_flappy_rl.py --ablation no_plasticity --episodes 2000
  python experiments/phase7_flappy_rl.py --run-ablations --episodes 500
  python experiments/phase7_flappy_rl.py --validate
  python experiments/phase7_flappy_rl.py --resume results/phase7/checkpoints/ep_1000.npz
        """,
    )
    # Mode
    p.add_argument("--mode", default="flymind",
                   choices=["flymind", "ppo", "hand", "random", "fixed", "frozen_connectome", "phase6b"],
                   help="Agent mode (default: flymind)")
    p.add_argument("--ablation", default="none",
                   choices=ABLATIONS,
                   help="Ablation condition (default: none)")
    # Training
    p.add_argument("--episodes", type=int, default=2000, help="Total training episodes")
    p.add_argument("--max-steps", type=int, default=10000, help="Max steps per episode")
    p.add_argument("--seed", type=int, default=42, help="Base random seed")
    # Checkpointing
    p.add_argument("--checkpoint-interval", type=int, default=200,
                   help="Episodes between checkpoints and evaluations")
    p.add_argument("--eval-seeds", type=int, default=20,
                   help="Number of validation seeds for evaluation")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    # Curriculum
    p.add_argument("--curriculum", action="store_true", help="Enable curriculum learning")
    p.add_argument("--curriculum-start-stage", type=int, default=0, help="Start curriculum stage")
    # Reward
    p.add_argument("--pipe-reward", type=float, default=1.0, help="Reward per pipe passed")
    p.add_argument("--survival-reward", type=float, default=0.001, help="Survival reward per step")
    p.add_argument("--collision-penalty", type=float, default=-1.0, help="Collision penalty")
    # Agent hyperparameters
    p.add_argument("--lr", type=float, default=0.002, help="Connectome plasticity learning rate")
    p.add_argument("--eligibility-decay", type=float, default=0.90, help="Eligibility trace decay")
    p.add_argument("--motor-temperature", type=float, default=1.5, help="Motor sigmoid temperature")
    # Ablations
    p.add_argument("--run-ablations", action="store_true", help="Run full ablation suite")
    # Validation
    p.add_argument("--validate", action="store_true", help="Run passivity validation")
    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.validate:
        success = run_validate(args)
        sys.exit(0 if success else 1)

    if args.run_ablations:
        run_ablations(args)
        return

    train(args)


if __name__ == "__main__":
    main()
