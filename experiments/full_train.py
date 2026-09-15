"""
Phase 7B: Full Training Script
Runs 20 seeds x 10,000 episodes with checkpoint/resume support.
Designed for unattended long-run execution.
"""

import sys
import time
import json
import csv
import gc
import signal
import os
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

RESULTS = ROOT / "results" / "phase7b"
CHECKPOINTS = RESULTS / "checkpoints"
VALIDATION = RESULTS / "validation"
METRICS = RESULTS / "metrics"
LOGS = RESULTS / "logs"

for d in [RESULTS, CHECKPOINTS, VALIDATION, METRICS, LOGS]:
    d.mkdir(parents=True, exist_ok=True)

# Globals for graceful shutdown
_INTERRUPTED = False
_CURRENT_AGENT = None
_CURRENT_STATE = {}


def signal_handler(sig, frame):
    global _INTERRUPTED
    _INTERRUPTED = True
    print("\n[INTERRUPT] Caught signal, finishing current episode then saving...")
    if _CURRENT_AGENT and _CURRENT_STATE:
        try:
            save_checkpoint(_CURRENT_AGENT, _CURRENT_STATE)
            print("[INTERRUPT] Checkpoint saved.")
        except Exception as e:
            print(f"[INTERRUPT] Failed to save: {e}")


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def get_checkpoint_path(seed: int, episode: int, tag: str) -> Path:
    return CHECKPOINTS / f"seed{seed}_ep{episode:06d}_{tag}.npz"


def get_latest_checkpoint(seed: int, tag: str) -> Optional[Path]:
    pattern = f"seed{seed}_ep*_{tag}.npz"
    ckpts = sorted(CHECKPOINTS.glob(pattern))
    if not ckpts:
        return None
    best = None
    best_ep = -1
    for c in ckpts:
        try:
            parts = c.stem.split("_")
            ep_str = [p for p in parts if p.startswith("ep")][0]
            ep_num = int(ep_str[2:])
            if ep_num > best_ep:
                best_ep = ep_num
                best = c
        except (IndexError, ValueError):
            continue
    return best


def save_checkpoint(agent, state: dict) -> None:
    seed = state["seed"]
    episode = state["episode"]
    tag = state.get("tag", "expanded")
    path = get_checkpoint_path(seed, episode, tag)
    agent.save_checkpoint(str(path))
    meta_path = path.with_suffix(".json")
    meta = {
        "seed": seed,
        "episode": episode,
        "tag": tag,
        "expanded": state.get("expanded", False),
        "connectome_path": state.get("connectome_path", ""),
        "num_neurons": agent.network.num_neurons,
        "best_score": state.get("best_score", 0),
        "scores": state.get("scores", [])[-100:],  # Keep last 100 only
        "timestamps": state.get("timestamps", []),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f)


def load_checkpoint(agent, path: Path) -> dict:
    agent.load_checkpoint(str(path))
    meta_path = path.with_suffix(".json")
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return {}


def run_validation(agent, n_episodes: int = 20, max_steps: int = 2000,
                   seed_start: int = 50000) -> dict:
    from flymind.environment.flappy import FlappyEnvironment
    from flymind.environment.flappy_reward import FlappyRewardShaper

    shaper = FlappyRewardShaper()
    scores = []
    steps_list = []
    flap_rates = []
    action_entropies = []
    neural_stats = []

    for i in range(n_episodes):
        seed = seed_start + i
        env = FlappyEnvironment(seed=seed, max_steps=max_steps)
        agent.reset()
        total_reward = 0.0
        actions = []
        flap_count = 0
        step_count = 0

        for step in range(max_steps):
            state = env.get_state()
            action = agent.act(state)
            actions.append(action)
            if action == 1:
                flap_count += 1
            next_state, env_reward, done, info = env.step(action)
            shaped = shaper.shape(env_reward, info, done)
            total_reward += shaped
            step_count = step + 1
            if done:
                break

        scores.append(state.score)
        steps_list.append(step_count)
        flap_rates.append(flap_count / max(step_count, 1))

        if actions:
            p_flap = np.mean(actions)
            p_glide = 1 - p_flap
            entropy = 0.0
            if p_flap > 0:
                entropy -= p_flap * np.log(p_flap + 1e-10)
            if p_glide > 0:
                entropy -= p_glide * np.log(p_glide + 1e-10)
            action_entropies.append(entropy)

        diag = agent.get_diagnostics()
        neural_stats.append({
            "epg_mean": diag.get("epg_mean", 0),
            "peg_mean": diag.get("peg_mean", 0),
            "pfnd_mean": diag.get("pfnd_mean", 0),
            "pfnv_mean": diag.get("pfnv_mean", 0),
        })

    return {
        "mean_score": float(np.mean(scores)),
        "median_score": float(np.median(scores)),
        "max_score": int(np.max(scores)),
        "std_score": float(np.std(scores)),
        "mean_steps": float(np.mean(steps_list)),
        "mean_flap_rate": float(np.mean(flap_rates)),
        "mean_entropy": float(np.mean(action_entropies)) if action_entropies else 0,
        "scores": [int(s) for s in scores],
        "mean_epg": float(np.mean([s["epg_mean"] for s in neural_stats])),
        "mean_peg": float(np.mean([s["peg_mean"] for s in neural_stats])),
        "mean_pfnd": float(np.mean([s["pfnd_mean"] for s in neural_stats])),
        "mean_pfnv": float(np.mean([s["pfnv_mean"] for s in neural_stats])),
    }


def train_seed(seed: int, n_episodes: int, expanded: bool, connectome_path: str,
               max_steps: int = 2000, validation_interval: int = 500,
               checkpoint_episodes: List[int] = None) -> dict:
    global _CURRENT_AGENT, _CURRENT_STATE, _INTERRUPTED

    from flymind.agent.flappy_agent_phase7b import FlyMindRLAgentExpanded
    from flymind.environment.flappy import FlappyEnvironment
    from flymind.environment.flappy_reward import FlappyRewardShaper

    if checkpoint_episodes is None:
        checkpoint_episodes = [0, 100, 250, 500, 1000, 2000, 5000, 7500, 10000]

    tag = "expanded" if expanded else "baseline"
    shaper = FlappyRewardShaper()

    # Try to resume from latest checkpoint
    start_episode = 0
    agent = None
    state = {
        "seed": seed,
        "episode": 0,
        "expanded": expanded,
        "connectome_path": connectome_path,
        "tag": tag,
        "best_score": 0,
        "scores": [],
        "timestamps": [],
    }

    latest = get_latest_checkpoint(seed, tag)
    if latest:
        try:
            agent = FlyMindRLAgentExpanded(connectome_path=connectome_path, seed=seed, expanded=expanded)
            meta = load_checkpoint(agent, latest)
            start_episode = meta.get("episode", 0) + 1
            state.update(meta)
            print(f"  [RESUME] Seed {seed}: resuming from episode {start_episode}")
        except Exception as e:
            print(f"  [WARN] Could not resume seed {seed}: {e}")
            agent = None

    if agent is None:
        agent = FlyMindRLAgentExpanded(connectome_path=connectome_path, seed=seed, expanded=expanded)
        print(f"  [NEW] Seed {seed}: starting from episode 0")

    _CURRENT_AGENT = agent
    _CURRENT_STATE = state

    # Metrics CSV
    metrics_path = METRICS / f"seed{seed}_{tag}.csv"
    write_header = not metrics_path.exists() or start_episode == 0
    metrics_file = open(metrics_path, "a", newline="")
    writer = csv.writer(metrics_file)
    if write_header:
        writer.writerow(["episode", "score", "steps", "total_reward", "flap_rate",
                         "elapsed_s", "eps_per_sec", "gpu_memory_mb"])

    # Validation results
    val_path = VALIDATION / f"seed{seed}_{tag}.json"
    val_results = []
    if val_path.exists():
        with open(val_path) as f:
            val_results = json.load(f)

    scores_window = []
    rolling_mean = 0.0
    t_train_start = time.perf_counter()
    total_eps_done = 0

    for episode in range(start_episode, n_episodes):
        if _INTERRUPTED:
            break

        env = FlappyEnvironment(seed=seed + episode, max_steps=max_steps)
        agent.reset()
        total_reward = 0.0
        step_count = 0
        flap_count = 0

        for step in range(max_steps):
            state_obj = env.get_state()
            action = agent.act(state_obj)
            if action == 1:
                flap_count += 1
            next_state, env_reward, done, info = env.step(action)
            shaped = shaper.shape(env_reward, info, done)
            total_reward += shaped
            if agent.enable_plasticity:
                agent.apply_step_reward(shaped)
            step_count = step + 1
            if done:
                break

        agent.end_episode(total_reward)
        total_eps_done += 1

        score = state_obj.score
        flap_rate = flap_count / max(step_count, 1)
        scores_window.append(score)
        if len(scores_window) > 100:
            scores_window.pop(0)
        rolling_mean = float(np.mean(scores_window))

        state["scores"].append(score)
        state["episode"] = episode

        # Timing
        elapsed = time.perf_counter() - t_train_start
        eps_per_sec = total_eps_done / max(elapsed, 1e-6)

        # GPU memory
        gpu_mem = 0.0
        try:
            import torch
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.max_memory_allocated() / 1e6
        except ImportError:
            pass

        # Log to CSV
        writer.writerow([episode, score, step_count, total_reward, flap_rate,
                         f"{elapsed:.1f}", f"{eps_per_sec:.2f}", f"{gpu_mem:.1f}"])
        metrics_file.flush()

        # Console log every 500 episodes
        if episode % 500 == 0 or episode == n_episodes - 1:
            eta_hours = (n_episodes - episode - 1) / max(eps_per_sec, 1e-6) / 3600
            best = max(state.get("best_score", 0), score)
            state["best_score"] = best
            print(f"  [{tag}] seed={seed} ep={episode:5d}/{n_episodes} "
                  f"score={score:3d} roll={rolling_mean:.1f} best={best} "
                  f"flap={flap_rate:.2f} {eps_per_sec:.1f}ep/s ETA={eta_hours:.1f}h")

        # Periodic validation
        if episode > 0 and episode % validation_interval == 0:
            agent.freeze_weights()
            val = run_validation(agent, n_episodes=20, max_steps=max_steps,
                                seed_start=50000 + seed % 1000)
            val["episode"] = episode
            val_results.append(val)
            agent.unfreeze_weights()

            with open(val_path, "w") as f:
                json.dump(val_results, f, indent=2)

            print(f"    [VAL] ep={episode} mean_score={val['mean_score']:.1f} "
                  f"max={val['max_score']} mean_steps={val['mean_steps']:.0f}")

        # Checkpoint at specified episodes
        if episode in checkpoint_episodes:
            state["timestamps"].append(time.time())
            save_checkpoint(agent, state)
            print(f"    [CKPT] Saved checkpoint at episode {episode}")

        # Periodic checkpoint every 1000 episodes (for crash recovery)
        if episode > 0 and episode % 1000 == 0 and episode not in checkpoint_episodes:
            state["timestamps"].append(time.time())
            save_checkpoint(agent, state)

        # NaN/Inf detection
        W = agent.network.synapses.raw_weights
        if np.any(np.isnan(W)) or np.any(np.isinf(W)):
            print(f"    [ERROR] NaN/Inf detected in weights at episode {episode}! Stopping.")
            save_checkpoint(agent, state)
            break

        # Weight explosion guard
        w_max = float(np.max(np.abs(W)))
        if w_max > 100.0:
            print(f"    [WARN] Weight explosion (max={w_max:.1f}) at ep={episode}. Clipping.")
            np.clip(W, -100.0, 100.0, out=W)
            agent.network.synapses.effective_weights = (
                agent.network.synapses.scale * W * agent.network.synapses.signs[:, None]
            )

    # Final save
    if episode >= n_episodes - 1:
        state["timestamps"].append(time.time())
        save_checkpoint(agent, state)

    metrics_file.close()

    result = {
        "seed": seed,
        "episodes_done": total_eps_done,
        "best_score": state.get("best_score", 0),
        "final_rolling_mean": rolling_mean,
        "total_time_s": time.perf_counter() - t_train_start,
        "validation_results": val_results,
    }

    _CURRENT_AGENT = None
    _CURRENT_STATE = {}
    gc.collect()
    return result


def main():
    global _INTERRUPTED

    import argparse
    parser = argparse.ArgumentParser(description="FlyMind Phase 7B Full Training")
    parser.add_argument("--expanded", action="store_true", help="Train expanded network")
    parser.add_argument("--baseline", action="store_true", help="Train baseline network")
    parser.add_argument("--seeds", type=int, default=20, help="Number of seeds")
    parser.add_argument("--episodes", type=int, default=10000, help="Episodes per seed")
    args = parser.parse_args()

    if args.expanded or (not args.baseline):
        connectome_path = str(ROOT / "data" / "processed" / "cx_heading_v1_expanded.json")
        expanded = True
    else:
        connectome_path = str(ROOT / "data" / "processed" / "cx_heading_v1.json")
        expanded = False

    tag = "expanded" if expanded else "baseline"
    num_neurons = 427 if expanded else 261

    # Print startup info
    print(f"\n{'='*60}")
    print(f"FlyMind Phase 7B")
    print(f"{'='*60}")
    print(f"Model:")
    print(f"  {'Expanded' if expanded else 'Baseline'}: {num_neurons} neurons")
    print(f"  Synapses: 24104" if expanded else f"  Synapses: 19969")

    try:
        import torch
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA: available")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("GPU: CPU (CUDA unavailable)")
    except ImportError:
        print("GPU: CPU (torch not available)")

    print(f"Batch: CPU (sequential)")
    print(f"Training:")
    print(f"  {args.seeds} seeds")
    print(f"  {args.episodes} episodes/seed")
    print(f"  {args.seeds * args.episodes} total episodes")
    print(f"{'='*60}\n")

    seeds = list(range(42, 42 + args.seeds))
    all_results = []

    for i, seed in enumerate(seeds):
        if _INTERRUPTED:
            print("[INTERRUPT] Stopping training loop.")
            break

        print(f"\n{'-'*40}")
        print(f"Seed {seed} ({i+1}/{args.seeds})")
        print(f"{'-'*40}")

        res = train_seed(
            seed=seed,
            n_episodes=args.episodes,
            expanded=expanded,
            connectome_path=connectome_path,
            max_steps=2000,
            validation_interval=500,
        )
        all_results.append(res)

        # Save aggregate results after each seed
        aggregate = {
            "tag": tag,
            "n_seeds": args.seeds,
            "n_episodes": args.episodes,
            "results": all_results,
            "mean_best_score": float(np.mean([r["best_score"] for r in all_results])),
            "mean_rolling": float(np.mean([r["final_rolling_mean"] for r in all_results])),
        }
        with open(RESULTS / f"training_results_{tag}.json", "w") as f:
            json.dump(aggregate, f, indent=2)

    # Final summary
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Tag: {tag}")
    print(f"Seeds: {len(all_results)}")
    print(f"Mean best score: {np.mean([r['best_score'] for r in all_results]):.1f}")
    print(f"Mean rolling mean: {np.mean([r['final_rolling_mean'] for r in all_results]):.1f}")

    return all_results


if __name__ == "__main__":
    main()
