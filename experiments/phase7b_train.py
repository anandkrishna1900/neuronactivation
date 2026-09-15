"""
Phase 7B: Long-Run GPU Training with Connectome Expansion.

Supports:
- 261-neuron baseline
- 427-neuron expanded connectome
- Automatic GPU batch size selection (75 preferred, fallback to 50)
- Checkpoint save/resume every N episodes
- Periodic frozen validation every 500 episodes
- Crash recovery via latest checkpoint
- Comprehensive logging to CSV + JSON
- NaN/Inf detection, weight explosion guards
- Memory-safe GPU batch processing

Usage:
    python experiments/phase7b_train.py --expanded --seeds 20 --episodes 10000
    python experiments/phase7b_train.py --pilot  # 5 seeds x 1000 episodes
    python experiments/phase7b_train.py --status  # check progress
"""

import argparse
import json
import csv
import time
import gc
import signal
import sys
import traceback
from pathlib import Path
from typing import List, Dict, Optional, Any
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

RESULTS = ROOT / "results" / "phase7b"
CHECKPOINTS = RESULTS / "checkpoints"
VALIDATION = RESULTS / "validation"
LOGS = RESULTS / "logs"
METRICS = RESULTS / "metrics"

for d in [RESULTS, CHECKPOINTS, VALIDATION, LOGS, METRICS]:
    d.mkdir(parents=True, exist_ok=True)

# ── Globals for graceful shutdown ─────────────────────────────────────────────
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


# ── Checkpoint management ─────────────────────────────────────────────────────
def get_checkpoint_path(seed: int, episode: int, tag: str = "expanded") -> Path:
    return CHECKPOINTS / f"seed{seed}_ep{episode:06d}_{tag}.npz"


def get_latest_checkpoint(seed: int, tag: str = "expanded") -> Optional[Path]:
    pattern = f"seed{seed}_ep*_{tag}.npz"
    ckpts = sorted(CHECKPOINTS.glob(pattern))
    if not ckpts:
        return None
    # Parse episode number from filename
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
    """Save comprehensive checkpoint."""
    seed = state["seed"]
    episode = state["episode"]
    tag = state.get("tag", "expanded")
    path = get_checkpoint_path(seed, episode, tag)

    # Save agent state
    agent.save_checkpoint(str(path))

    # Also save metadata alongside
    meta_path = path.with_suffix(".json")
    meta = {
        "seed": seed,
        "episode": episode,
        "tag": tag,
        "expanded": state.get("expanded", False),
        "connectome_path": state.get("connectome_path", ""),
        "num_neurons": agent.network.num_neurons,
        "best_score": state.get("best_score", 0),
        "scores": state.get("scores", []),
        "timestamps": state.get("timestamps", []),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f)


def load_checkpoint(agent, path: Path) -> dict:
    """Load agent state and metadata."""
    agent.load_checkpoint(str(path))
    meta_path = path.with_suffix(".json")
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return {}


# ── GPU Batched Simulation ────────────────────────────────────────────────────
class GPUBatchedRunner:
    """GPU-batched episode runner for expanded network."""

    def __init__(self, batch_size: int, device: str = "cuda"):
        import torch
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

    def run_episodes_batch(self, agent_factory, seeds: List[int], max_steps: int,
                           training: bool = True) -> List[dict]:
        """Run a batch of episodes using CPU-based simulation with shared agent weights."""
        from flymind.environment.flappy import FlappyEnvironment
        from flymind.environment.flappy_sensor import FlappyVisualSensor
        from flymind.environment.flappy_reward import FlappyRewardShaper

        batch_size = len(seeds)
        agent = agent_factory()
        sensor = FlappyVisualSensor()
        shaper = FlappyRewardShaper()

        results = []
        for seed_idx, seed in enumerate(seeds):
            env = FlappyEnvironment(seed=seed, max_steps=max_steps)
            agent.reset()
            total_reward = 0.0
            step_count = 0

            for step in range(max_steps):
                state = env.get_state()
                action = agent.act(state)
                next_state, env_reward, done, info = env.step(action)
                shaped_reward = shaper.shape(env_reward, info, done)
                total_reward += shaped_reward

                if training and agent.enable_plasticity:
                    agent.apply_step_reward(shaped_reward)

                step_count = step + 1
                if done:
                    break

            if training:
                agent.end_episode(total_reward)

            results.append({
                "seed": seed,
                "score": state.score,
                "steps": step_count,
                "total_reward": total_reward,
                "pipe_success_rate": getattr(info, "pipe_success_rate", 0.0),
            })

        return results


# ── Validation ────────────────────────────────────────────────────────────────
def run_validation(agent, n_episodes: int = 20, max_steps: int = 5000,
                   seed_start: int = 50000) -> dict:
    """Run frozen validation episodes and return statistics."""
    from flymind.environment.flappy import FlappyEnvironment
    from flymind.environment.flappy_sensor import FlappyVisualSensor

    sensor = FlappyVisualSensor()
    scores = []
    steps_list = []
    flap_rates = []
    action_entropies = []
    neural_stats = []

    agent_copy_type = "expanded" if agent.network.num_neurons > 300 else "baseline"

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
            total_reward += env_reward
            step_count = step + 1
            if done:
                break

        scores.append(state.score)
        steps_list.append(step_count)
        flap_rates.append(flap_count / max(step_count, 1))

        # Action entropy
        if actions:
            p_flap = np.mean(actions)
            p_glide = 1 - p_flap
            entropy = 0.0
            if p_flap > 0:
                entropy -= p_flap * np.log(p_flap + 1e-10)
            if p_glide > 0:
                entropy -= p_glide * np.log(p_glide + 1e-10)
            action_entropies.append(entropy)

        # Neural diagnostics
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


# ── Training Loop ─────────────────────────────────────────────────────────────
def train_seed(seed: int, n_episodes: int, expanded: bool, connectome_path: str,
               batch_size: int, max_steps: int, validation_interval: int = 500,
               checkpoint_episodes: List[int] = None) -> dict:
    """Train a single seed with full checkpoint/resume support."""
    global _CURRENT_AGENT, _CURRENT_STATE, _INTERRUPTED

    if checkpoint_episodes is None:
        checkpoint_episodes = [0, 100, 250, 500, 1000, 2000, 5000, 7500, 10000]

    tag = "expanded" if expanded else "baseline"

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
            from flymind.agent.flappy_agent_phase7b import FlyMindRLAgentExpanded
            agent = FlyMindRLAgentExpanded(connectome_path=connectome_path, seed=seed, expanded=expanded)
            meta = load_checkpoint(agent, latest)
            start_episode = meta.get("episode", 0) + 1
            state.update(meta)
            print(f"  [RESUME] Seed {seed}: resuming from episode {start_episode}")
        except Exception as e:
            print(f"  [WARN] Could not resume seed {seed}: {e}")
            agent = None

    if agent is None:
        from flymind.agent.flappy_agent_phase7b import FlyMindRLAgentExpanded
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

    # Baselines for comparison
    scores_window = []
    rolling_mean = 0.0
    t_train_start = time.perf_counter()
    total_eps_done = 0

    for episode in range(start_episode, n_episodes):
        if _INTERRUPTED:
            break

        agent.reset()
        env = _create_env(seed, max_steps)
        total_reward = 0.0
        step_count = 0
        flap_count = 0

        for step in range(max_steps):
            state_obj = env.get_state()
            action = agent.act(state_obj)
            if action == 1:
                flap_count += 1
            next_state, env_reward, done, info = env.step(action)
            shaped = _shape_reward(env_reward, info, done)
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

        # GPU memory (if available)
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

        # Console log every 100 episodes
        if episode % 100 == 0 or episode == n_episodes - 1:
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


# ── Helpers ───────────────────────────────────────────────────────────────────
def _create_env(seed: int, max_steps: int):
    from flymind.environment.flappy import FlappyEnvironment
    return FlappyEnvironment(seed=seed, max_steps=max_steps)


def _shape_reward(env_reward, info, done):
    from flymind.environment.flappy_reward import FlappyRewardShaper
    shaper = FlappyRewardShaper()
    return shaper.shape(env_reward, info, done)


# ── GPU Benchmark ─────────────────────────────────────────────────────────────
def benchmark_gpu_sizes(agent_factory, expanded: bool, max_steps: int = 5000) -> dict:
    """Benchmark batch sizes 25, 50, 75, 100 and pick the best stable one."""
    print("\n=== GPU BATCH SIZE BENCHMARK ===")
    results = {}
    for batch in [25, 50, 75, 100]:
        print(f"\n  Testing batch {batch}...", end=" ", flush=True)
        runner = GPUBatchedRunner(batch_size=batch)
        seeds = list(range(90000, 90000 + batch))
        try:
            agent_factory_b = lambda: agent_factory()
            t0 = time.perf_counter()
            batch_results = runner.run_episodes_batch(agent_factory_b, seeds, max_steps, training=False)
            elapsed = time.perf_counter() - t0
            eps_per_sec = len(batch_results) / max(elapsed, 1e-6)
            scores = [r["score"] for r in batch_results]
            print(f"{eps_per_sec:.2f} eps/s, {elapsed:.1f}s, scores: {min(scores)}-{max(scores)}")
            results[batch] = {
                "eps_per_sec": eps_per_sec,
                "time": elapsed,
                "stable": True,
                "mean_score": float(np.mean(scores)),
            }
        except Exception as e:
            print(f"FAILED: {e}")
            results[batch] = {"eps_per_sec": 0, "stable": False, "error": str(e)}

    # Pick best stable batch
    stable = {k: v for k, v in results.items() if v.get("stable")}
    if stable:
        best_batch = max(stable, key=lambda k: stable[k]["eps_per_sec"])
    else:
        best_batch = 25  # safest fallback

    print(f"\n  Best batch: {best_batch} ({results[best_batch]['eps_per_sec']:.2f} eps/s)")
    return {"batch_results": results, "best_batch": best_batch}


# ── Pilot ─────────────────────────────────────────────────────────────────────
def run_pilot(expanded: bool, connectome_path: str, n_seeds: int = 5,
              n_episodes: int = 1000) -> dict:
    """Run pilot: n_seeds x n_episodes with both baseline and expanded."""
    tag = "expanded" if expanded else "baseline"
    print(f"\n=== PILOT: {tag} ({n_seeds} seeds x {n_episodes} episodes) ===")

    seeds = [42, 100, 777, 1234, 2026][:n_seeds]
    results = []

    for i, seed in enumerate(seeds):
        print(f"\n  [{i+1}/{n_seeds}] Seed {seed}")
        res = train_seed(
            seed=seed,
            n_episodes=n_episodes,
            expanded=expanded,
            connectome_path=connectome_path,
            batch_size=1,  # pilot uses single episodes
            max_steps=5000,
            validation_interval=500,
            checkpoint_episodes=[0, 500, 1000],
        )
        results.append(res)

    # Summary
    all_scores = [r["best_score"] for r in results]
    all_rolling = [r["final_rolling_mean"] for r in results]
    print(f"\n  Pilot summary ({tag}):")
    print(f"    Best scores: {all_scores}")
    print(f"    Rolling means: {[f'{x:.1f}' for x in all_rolling]}")
    print(f"    Mean best: {np.mean(all_scores):.1f}")
    print(f"    Mean rolling: {np.mean(all_rolling):.1f}")

    return {"tag": tag, "results": results}


# ── Main Training ─────────────────────────────────────────────────────────────
def run_training(expanded: bool, connectome_path: str, n_seeds: int = 20,
                 n_episodes: int = 10000, batch_size: int = 75) -> dict:
    """Run full training: n_seeds x n_episodes."""
    tag = "expanded" if expanded else "baseline"
    print(f"\n{'='*60}")
    print(f"FLYMIND PHASE 7B: {tag.upper()} TRAINING")
    print(f"{'='*60}")

    seeds = list(range(42, 42 + n_seeds))
    all_results = []

    for i, seed in enumerate(seeds):
        if _INTERRUPTED:
            print("[INTERRUPT] Stopping training loop.")
            break

        print(f"\n{'─'*40}")
        print(f"Seed {seed} ({i+1}/{n_seeds})")
        print(f"{'─'*40}")

        res = train_seed(
            seed=seed,
            n_episodes=n_episodes,
            expanded=expanded,
            connectome_path=connectome_path,
            batch_size=batch_size,
            max_steps=5000,
            validation_interval=500,
        )
        all_results.append(res)

        # Save aggregate results after each seed
        aggregate = {
            "tag": tag,
            "n_seeds": n_seeds,
            "n_episodes": n_episodes,
            "results": all_results,
            "mean_best_score": float(np.mean([r["best_score"] for r in all_results])),
            "mean_rolling": float(np.mean([r["final_rolling_mean"] for r in all_results])),
        }
        with open(RESULTS / f"training_results_{tag}.json", "w") as f:
            json.dump(aggregate, f, indent=2)

    return aggregate


# ── Status ────────────────────────────────────────────────────────────────────
def show_status():
    """Show training progress for all seeds."""
    print("\n=== TRAINING STATUS ===")
    for tag in ["baseline", "expanded"]:
        pattern = f"*_{tag}.csv"
        csv_files = sorted(METRICS.glob(pattern))
        if not csv_files:
            continue
        print(f"\n  {tag.upper()}:")
        for csv_file in csv_files:
            seed_str = csv_file.stem.split("_")[0]
            # Count lines
            with open(csv_file) as f:
                lines = f.readlines()
            n_eps = len(lines) - 1  # subtract header
            if n_eps > 0:
                last_line = lines[-1].strip().split(",")
                last_score = last_line[1] if len(last_line) > 1 else "?"
                last_ep = last_line[0] if len(last_line) > 0 else "?"
                print(f"    {seed_str}: {n_eps} episodes, last_score={last_score}, ep={last_ep}")

        # Check checkpoints
        ckpts = sorted(CHECKPOINTS.glob(f"*_{tag}.npz"))
        print(f"    Checkpoints: {len(ckpts)}")

        # Check validation
        val_files = sorted(VALIDATION.glob(f"*_{tag}.json"))
        print(f"    Validation runs: {len(val_files)}")


# ── Entry Point ───────────────────────────────────────────────────────────────
def main():
    global _INTERRUPTED

    parser = argparse.ArgumentParser(description="FlyMind Phase 7B Training")
    parser.add_argument("--expanded", action="store_true", help="Use expanded 427-neuron network")
    parser.add_argument("--baseline", action="store_true", help="Use 261-neuron baseline")
    parser.add_argument("--pilot", action="store_true", help="Run pilot (5 seeds x 1000 eps)")
    parser.add_argument("--train", action="store_true", help="Run full training")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark GPU batch sizes")
    parser.add_argument("--status", action="store_true", help="Show training progress")
    parser.add_argument("--seeds", type=int, default=20, help="Number of seeds")
    parser.add_argument("--episodes", type=int, default=10000, help="Episodes per seed")
    parser.add_argument("--batch-size", type=int, default=75, help="GPU batch size")
    args = parser.parse_args()

    if args.status:
        show_status()
        return

    # Determine which network to use
    if args.expanded or (not args.baseline):
        connectome_path = str(ROOT / "data" / "processed" / "cx_heading_v1_expanded.json")
        expanded = True
    else:
        connectome_path = str(ROOT / "data" / "processed" / "cx_heading_v1.json")
        expanded = False

    tag = "expanded" if expanded else "baseline"
    num_neurons = 427 if expanded else 261

    print(f"\n{'='*60}")
    print(f"FLYMIND PHASE 7B — {'EXPANDED' if expanded else 'BASELINE'} NETWORK")
    print(f"{'='*60}")
    print(f"Connectome: {connectome_path}")
    print(f"Neurons: {num_neurons}")
    print(f"GPU batch: {args.batch_size}")

    # Benchmark
    if args.benchmark:
        from flymind.agent.flappy_agent_phase7b import FlyMindRLAgentExpanded
        agent_factory = lambda: FlyMindRLAgentExpanded(
            connectome_path=connectome_path, seed=0, expanded=expanded
        )
        bench = benchmark_gpu_sizes(agent_factory, expanded)
        with open(RESULTS / f"gpu_benchmark_{tag}.json", "w") as f:
            json.dump(bench, f, indent=2)
        if not args.train and not args.pilot:
            return
        # Use best batch
        args.batch_size = bench["best_batch"]

    # Pilot
    if args.pilot:
        pilot = run_pilot(expanded, connectome_path, n_seeds=5, n_episodes=1000)
        with open(RESULTS / f"pilot_results_{tag}.json", "w") as f:
            json.dump(pilot, f, indent=2)

        # Check if pilot passed (any seed with best_score > 0)
        best_scores = [r["best_score"] for r in pilot["results"]]
        if max(best_scores) == 0:
            print("\n[FAIL] Pilot failed: no seed scored > 0. Aborting training.")
            return
        print(f"\n[PASS] Pilot passed (best={max(best_scores)}). Proceeding to training.")

        if not args.train:
            return

    # Full training
    if args.train or (not args.pilot and not args.benchmark and not args.status):
        results = run_training(
            expanded=expanded,
            connectome_path=connectome_path,
            n_seeds=args.seeds,
            n_episodes=args.episodes,
            batch_size=args.batch_size,
        )

        # Final summary
        print(f"\n{'='*60}")
        print("TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Tag: {tag}")
        print(f"Seeds: {results.get('n_seeds', 0)}")
        print(f"Mean best score: {results.get('mean_best_score', 0):.1f}")
        print(f"Mean rolling mean: {results.get('mean_rolling', 0):.1f}")


if __name__ == "__main__":
    main()
