"""
Phase 7 CPU/GPU/Hybrid Benchmark Suite.

Benchmarks the real FlyMind simulation across CPU serial, CPU multiprocessing,
GPU batched, and hybrid configurations. Includes correctness validation.

Usage:
    python experiments/phase7_cpu_gpu_benchmark.py --full-suite
    python experiments/phase7_cpu_gpu_benchmark.py --device cpu --cpu-workers 8 --episodes 20 --max-steps 5000
    python experiments/phase7_cpu_gpu_benchmark.py --device cuda --batch-size 50 --episodes 20 --max-steps 5000
    python experiments/phase7_cpu_gpu_benchmark.py --validate-correctness
"""

from __future__ import annotations
import argparse
import gc
import json
import os
import sys
import time
from collections import defaultdict
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from flymind.brain.network import NeuralNetwork
from flymind.connectome.loader import ConnectomeLoader
from flymind.environment.flappy import FlappyEnvironment
from flymind.environment.flappy_sensor import FlappyVisualSensor
from flymind.environment.flappy_reward import FlappyRewardShaper
from flymind.agent.flappy_agent_phase7 import FlyMindRLAgent

CONNECTOME_PATH = ROOT / "data" / "processed" / "cx_heading_v1.json"
RESULTS_DIR = ROOT / "results" / "phase7" / "benchmark"
FIGURES_DIR = RESULTS_DIR / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ── Connectome factory ────────────────────────────────────────────────────────
_graph_cache = None

def get_graph():
    global _graph_cache
    if _graph_cache is None:
        _graph_cache = ConnectomeLoader.load_from_json(CONNECTOME_PATH)
    return _graph_cache


def make_network():
    return NeuralNetwork(get_graph(), synapse_scale=0.01)


def make_agent(seed=42):
    net = make_network()
    return FlyMindRLAgent(
        network=net, sensory_drive=25.0, sub_steps=10,
        learning_rate=0.002, eligibility_decay=0.90,
        plasticity_mode="pathway", motor_temperature=1.5,
        motor_lr=0.01, decoder_mode="stochastic", seed=seed,
    )


# ── CPU single-seed episode runner ────────────────────────────────────────────
def run_cpu_episode(agent, seed, max_steps, training=False):
    """Run one episode on CPU (existing code path)."""
    env = FlappyEnvironment(seed=seed, max_steps=max_steps)
    shaper = FlappyRewardShaper()
    state = env.reset(seed=seed)
    agent.reset()
    total_reward = 0.0
    done = False
    steps = 0
    while not done and steps < max_steps:
        action = agent.act(state)
        state, env_reward, done, info = env.step(action)
        shaped_r = shaper.shape(env_reward, info, done)
        total_reward += shaped_r
        if training:
            agent.apply_step_reward(shaped_r)
        steps += 1
    if training:
        agent.end_episode(total_reward)
    return {"score": state.score, "steps": steps, "total_reward": total_reward}


# ── CPU parallel worker ──────────────────────────────────────────────────────
def _cpu_worker(worker_id, seeds, max_steps, training, result_queue):
    """Worker process for CPU multiprocessing benchmark."""
    agent = make_agent(seed=worker_id * 1000)
    results = []
    for seed in seeds:
        r = run_cpu_episode(agent, seed, max_steps, training=training)
        results.append(r)
    result_queue.put((worker_id, results))


# ── GPU batched simulation ───────────────────────────────────────────────────
class GPUBatchedSimulation:
    """
    GPU-batched FlyMind simulation using PyTorch.

    Faithfully replicates the CPU agent computation:
    - Same connectome weights
    - Same sensory -> EPG mapping
    - Same 10 sub-steps of recurrent dynamics
    - Same motor readout (PEG asymmetry, PEG lift, PFNd, PFNv)
    - Same sigmoid decoder

    Environments still run on CPU (they're cheap).
    Neural computation is batched on GPU.
    """

    def __init__(self, batch_size: int, device: str = "cuda"):
        import torch
        self.device = torch.device(device)
        self.batch_size = batch_size

        # Build reference agent to extract exact weights and indices
        agent = make_agent(seed=0)
        W_raw = agent.network.synapses.raw_weights.copy()
        signs = agent.network.synapses.signs.copy()
        scale = agent.network.synapses.scale
        W_eff = (signs[:, None] * W_raw) * scale

        self.num_neurons = W_raw.shape[0]

        # Convert to torch (float32 for GPU speed)
        self.W_eff_t = torch.tensor(W_eff, dtype=torch.float32, device=self.device)

        # Motor readout
        self.W_motor_t = torch.tensor(agent.W_motor, dtype=torch.float32, device=self.device)
        self.b_motor = float(agent.b_motor)

        # Neuron indices (exact same as CPU agent)
        self.epg_ordered_indices = list(agent.epg_ordered_indices)
        self.peg_L = list(agent.peg_L)
        self.peg_R = list(agent.peg_R)
        self.pfnd_indices = list(agent.pfnd_indices)
        self.pfnv_indices = list(agent.pfnv_indices)
        self.er4d_indices = list(agent.er4d_indices)

        # Sensory drive
        self.sensory_drive = 25.0
        self.sensor = FlappyVisualSensor()
        self.motor_temperature = 1.5

        # Batched neuron state
        self.activity = torch.zeros(batch_size, self.num_neurons, dtype=torch.float32, device=self.device)
        self.prev_activity = torch.zeros(batch_size, self.num_neurons, dtype=torch.float32, device=self.device)

        del agent

    def reset_all(self):
        self.activity.zero_()
        self.prev_activity.zero_()

    def _build_sensory_current_batch(self, sensor_batch: np.ndarray) -> torch.Tensor:
        """Map batch of 9-channel sensor outputs to connectome input currents."""
        import torch
        batch_size = sensor_batch.shape[0]
        ext_current = np.zeros((batch_size, self.num_neurons), dtype=np.float32)
        n_epg = len(self.epg_ordered_indices)

        for b in range(batch_size):
            for i, sig in enumerate(sensor_batch[b]):
                if sig < 1e-6:
                    continue
                epg_i = int((i / 9) * n_epg)
                if epg_i < n_epg:
                    ext_current[b, self.epg_ordered_indices[epg_i]] += sig * self.sensory_drive
            # Spontaneous baseline when no pipe visible
            if np.sum(sensor_batch[b]) < 1e-4 and self.er4d_indices:
                for idx in self.er4d_indices:
                    ext_current[b, idx] += 0.15

        return torch.tensor(ext_current, device=self.device)

    def _compute_motor_batch(self):
        """Compute motor logit and flap probability for all batch elements."""
        import torch
        act = self.activity

        # PEG asymmetry
        if self.peg_L and self.peg_R:
            peg_l = act[:, self.peg_L].mean(dim=1)
            peg_r = act[:, self.peg_R].mean(dim=1)
            peg_diff = peg_l - peg_r
            peg_lift = peg_l
        else:
            peg_diff = torch.zeros(self.batch_size, device=self.device)
            peg_lift = torch.zeros(self.batch_size, device=self.device)

        # PFNd, PFNv
        pfnd_mean = act[:, self.pfnd_indices].mean(dim=1) if self.pfnd_indices else torch.zeros(self.batch_size, device=self.device)
        pfnv_mean = act[:, self.pfnv_indices].mean(dim=1) if self.pfnv_indices else torch.zeros(self.batch_size, device=self.device)

        pop_vec = torch.stack([peg_diff, peg_lift, pfnd_mean, pfnv_mean], dim=1)
        motor_logit = (pop_vec @ self.W_motor_t) + self.b_motor
        z = torch.clamp(motor_logit / self.motor_temperature, -10.0, 10.0)
        flap_prob = torch.sigmoid(z)
        return flap_prob.cpu().numpy()

    def step_batch(self, ext_current_t):
        """One simulation step for all batch elements (10 sub-steps)."""
        import torch
        self.prev_activity = self.activity.clone()
        for _ in range(10):
            synaptic = torch.matmul(self.activity, self.W_eff_t)
            total = ext_current_t + synaptic
            target = torch.tanh(torch.relu(total))
            self.activity = self.activity + 0.1 * (-self.activity + target)

    def run_episodes(self, seeds: List[int], max_steps: int):
        """Run batch_size independent episodes simultaneously."""
        import torch
        batch_size = len(seeds)
        envs = [FlappyEnvironment(seed=s, max_steps=max_steps) for s in seeds]
        shaper = FlappyRewardShaper()

        self.activity.zero_()
        self.prev_activity.zero_()

        scores = [0] * batch_size
        steps_arr = [0] * batch_size
        rewards = [0.0] * batch_size
        dones = [False] * batch_size

        for step in range(max_steps):
            if all(dones):
                break

            # Sense
            sensor_batch = np.zeros((batch_size, 9), dtype=np.float64)
            for i in range(batch_size):
                if not dones[i]:
                    sensor_batch[i] = self.sensor.sense(envs[i].get_state())

            # Build external current
            ext_t = self._build_sensory_current_batch(sensor_batch)

            # Neural simulation (10 sub-steps)
            self.step_batch(ext_t)

            # Motor readout
            flap_probs = self._compute_motor_batch()

            # Actions (same heuristic overrides as CPU agent)
            actions = np.zeros(batch_size, dtype=int)
            for i in range(batch_size):
                if dones[i]:
                    continue
                state = envs[i].get_state()
                vy = state.bird_vy
                fp = float(flap_probs[i])
                sensor = sensor_batch[i]
                gap_below = float(sensor[6] + sensor[7] + sensor[8])
                gap_above = float(sensor[0] + sensor[1] + sensor[2])
                no_pipe = float(np.sum(sensor)) < 0.05

                if vy >= 1.0:
                    actions[i] = 0
                elif gap_below > gap_above and gap_below > 0.1:
                    actions[i] = 0
                elif no_pipe and state.bird_y > 210.0:
                    actions[i] = 0
                else:
                    p_safe = float(np.clip(fp, 1e-6, 1.0 - 1e-6))
                    rng = np.random.default_rng(seeds[i] * 10000 + step)
                    actions[i] = 1 if (fp > 0.35 and vy < 1.0) or (rng.random() < p_safe and vy < 0.5) else 0

            # Step environments
            for i in range(batch_size):
                if dones[i]:
                    continue
                state, env_r, done, info = envs[i].step(int(actions[i]))
                shaped_r = shaper.shape(env_r, info, done)
                rewards[i] += shaped_r
                if done:
                    dones[i] = True
                    scores[i] = state.score
                steps_arr[i] = step + 1

        return [
            {"score": scores[i], "steps": steps_arr[i], "total_reward": rewards[i]}
            for i in range(batch_size)
        ]


# ── Benchmark runners ─────────────────────────────────────────────────────────
def benchmark_cpu_serial(episodes, max_steps, warmup=3):
    """Benchmark CPU serial execution."""
    results = []
    # Warmup
    for i in range(warmup):
        agent = make_agent(seed=i)
        run_cpu_episode(agent, seed=10000 + i, max_steps=max_steps)

    # Timed runs
    times = []
    for ep in range(episodes):
        agent = make_agent(seed=ep)
        t0 = time.perf_counter()
        r = run_cpu_episode(agent, seed=20000 + ep, max_steps=max_steps)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        results.append(r)

    return {
        "times": times,
        "results": results,
        "mean_time": float(np.mean(times)),
        "median_time": float(np.median(times)),
        "eps_per_sec": 1.0 / np.mean(times),
        "total_time": sum(times),
    }


def benchmark_cpu_parallel(workers, episodes, max_steps, warmup=1):
    """Benchmark CPU parallel execution with subprocess workers."""
    import subprocess

    # Split episodes across workers
    eps_per_worker = episodes // workers
    extra = episodes % workers
    assignments = []
    base_seed = 30000
    for w in range(workers):
        count = eps_per_worker + (1 if w < extra else 0)
        assignments.append((base_seed + w * 100000, count))

    # Create worker script
    worker_script = ROOT / "experiments" / "_benchmark_worker.py"
    worker_script.write_text(f"""
import sys, time, json
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from experiments.phase7_cpu_gpu_benchmark import make_agent, run_cpu_episode

worker_id = int(sys.argv[1])
base_seed = int(sys.argv[2])
count = int(sys.argv[3])
max_steps = int(sys.argv[4])

agent = make_agent(seed=worker_id * 1000)
results = []
t0 = time.perf_counter()
for i in range(count):
    seed = base_seed + i
    r = run_cpu_episode(agent, seed, max_steps)
    results.append(r)
t1 = time.perf_counter()
print(json.dumps({{"worker": worker_id, "time": t1 - t0, "episodes": count, "results": results}}))
""".replace("ROOT", repr(str(ROOT))))

    # Warmup
    for i in range(min(warmup, workers)):
        subprocess.run(
            [sys.executable, str(worker_script), str(i), "99999", "1", str(max_steps)],
            capture_output=True, timeout=120, cwd=str(ROOT)
        )

    # Timed run
    t0 = time.perf_counter()
    procs = []
    for w, (base_seed, count) in enumerate(assignments):
        p = subprocess.Popen(
            [sys.executable, str(worker_script), str(w), str(base_seed), str(count), str(max_steps)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=str(ROOT)
        )
        procs.append(p)

    all_results = []
    for p in procs:
        stdout, _ = p.communicate(timeout=300)
        try:
            data = json.loads(stdout.decode().strip().split("\n")[-1])
            all_results.extend(data.get("results", []))
        except Exception:
            pass

    t1 = time.perf_counter()
    total_time = t1 - t0

    # Cleanup
    worker_script.unlink(missing_ok=True)

    return {
        "workers": workers,
        "episodes": len(all_results),
        "total_time": total_time,
        "eps_per_sec": len(all_results) / total_time if total_time > 0 else 0,
        "results": all_results,
    }


def benchmark_gpu_batch(batch_size, episodes, max_steps, warmup=3):
    """Benchmark GPU batched execution."""
    import torch

    actual_batch = min(batch_size, episodes)
    sim = GPUBatchedSimulation(actual_batch, device="cuda")

    # Warmup
    for i in range(warmup):
        seeds = [50000 + i * actual_batch + j for j in range(actual_batch)]
        sim.run_episodes(seeds, max_steps=min(max_steps, 100))
        torch.cuda.synchronize()

    # GC before timing
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Timed runs — run all episodes in batches
    times = []
    all_results = []
    ep = 0
    while ep < episodes:
        this_batch = min(actual_batch, episodes - ep)
        seeds = [60000 + ep + j for j in range(this_batch)]

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if this_batch == actual_batch:
            batch_results = sim.run_episodes(seeds, max_steps)
        else:
            # Smaller batch at the end
            small_sim = GPUBatchedSimulation(this_batch, device="cuda")
            batch_results = small_sim.run_episodes(seeds, max_steps)
            del small_sim
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        times.append((t1 - t0, this_batch))
        all_results.extend(batch_results)
        ep += this_batch

    total_time = sum(t for t, _ in times)
    total_eps = len(all_results)
    vram = torch.cuda.max_memory_allocated() / 1e6  # MB

    return {
        "batch_size": actual_batch,
        "episodes": total_eps,
        "total_time": total_time,
        "eps_per_sec": total_eps / total_time if total_time > 0 else 0,
        "times_per_batch": [(t, b) for t, b in times],
        "vram_mb": vram,
        "results": all_results,
    }


# ── Correctness validation ───────────────────────────────────────────────────
def validate_correctness(max_steps=500, seeds=None):
    """Compare CPU vs GPU outputs on identical seeds."""
    if seeds is None:
        seeds = [42, 100, 777, 1234, 5678]

    print("\n=== CORRECTNESS VALIDATION ===")
    print(f"Seeds: {seeds}")
    print(f"Max steps: {max_steps}")

    import torch

    all_match = True
    for seed in seeds:
        # CPU run
        agent_cpu = make_agent(seed=seed)
        cpu_result = run_cpu_episode(agent_cpu, seed=seed, max_steps=max_steps)

        # GPU run (batch=1)
        sim_gpu = GPUBatchedSimulation(1, device="cuda")
        gpu_results = sim_gpu.run_episodes([seed], max_steps)
        gpu_result = gpu_results[0]

        # Compare
        score_match = cpu_result["score"] == gpu_result["score"]
        steps_match = abs(cpu_result["steps"] - gpu_result["steps"]) <= 2
        reward_match = abs(cpu_result["total_reward"] - gpu_result["total_reward"]) < 0.5

        status = "PASS" if (score_match and steps_match and reward_match) else "FAIL"
        if status == "FAIL":
            all_match = False

        print(f"  Seed {seed}: {status}")
        print(f"    CPU: score={cpu_result['score']} steps={cpu_result['steps']} reward={cpu_result['total_reward']:.3f}")
        print(f"    GPU: score={gpu_result['score']} steps={gpu_result['steps']} reward={gpu_result['total_reward']:.3f}")

        del agent_cpu, sim_gpu
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nOverall: {'ALL MATCH' if all_match else 'MISMATCH DETECTED'}")
    return all_match


# ── Full suite runner ─────────────────────────────────────────────────────────
def run_full_suite(args):
    """Run the complete benchmark suite."""
    import torch

    print(f"\n{'='*70}")
    print("PHASE 7 CPU/GPU/HYBRID BENCHMARK SUITE")
    print(f"{'='*70}")
    print(f"Device: {args.device}")
    print(f"Episodes per config: {args.episodes}")
    print(f"Max steps: {args.max_steps}")
    print(f"Warmup: {args.warmup}")
    print(f"Measured runs: {args.runs}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"{'='*70}")

    all_results = {}

    # ── 1. CPU Serial ──────────────────────────────────────────────────────
    print("\n--- CPU Serial ---")
    cpu_serial = benchmark_cpu_serial(args.episodes, args.max_steps, warmup=args.warmup)
    print(f"  Mean: {cpu_serial['mean_time']:.4f}s/ep | {cpu_serial['eps_per_sec']:.2f} eps/s | "
          f"{cpu_serial['eps_per_sec']*3600:.0f} eps/hr")
    all_results["cpu_serial"] = {
        "mean_time": cpu_serial["mean_time"],
        "eps_per_sec": cpu_serial["eps_per_sec"],
        "eps_per_hour": cpu_serial["eps_per_sec"] * 3600,
        "total_time": cpu_serial["total_time"],
    }

    # ── 2. CPU Parallel ────────────────────────────────────────────────────
    for workers in [2, 4, 6, 8]:
        if workers > os.cpu_count():
            continue
        print(f"\n--- CPU Parallel ({workers} workers) ---")
        cpu_par = benchmark_cpu_parallel(workers, args.episodes, args.max_steps, warmup=1)
        print(f"  Mean: {cpu_par['total_time']:.2f}s total | {cpu_par['eps_per_sec']:.2f} eps/s | "
              f"{cpu_par['eps_per_sec']*3600:.0f} eps/hr")
        all_results[f"cpu_par_{workers}"] = {
            "workers": workers,
            "total_time": cpu_par["total_time"],
            "eps_per_sec": cpu_par["eps_per_sec"],
            "eps_per_hour": cpu_par["eps_per_sec"] * 3600,
        }

    # ── 3. GPU Batched ─────────────────────────────────────────────────────
    if torch.cuda.is_available():
        for batch in [1, 2, 5, 10, 20, 25, 50, 75, 100]:
            print(f"\n--- GPU Batch {batch} ---")
            try:
                torch.cuda.reset_peak_memory_stats()
                gpu_res = benchmark_gpu_batch(batch, args.episodes, args.max_steps, warmup=args.warmup)
                print(f"  Mean: {gpu_res['total_time']:.2f}s total | {gpu_res['eps_per_sec']:.2f} eps/s | "
                      f"{gpu_res['eps_per_sec']*3600:.0f} eps/hr | VRAM: {gpu_res['vram_mb']:.0f}MB")
                all_results[f"gpu_batch_{batch}"] = {
                    "batch_size": batch,
                    "total_time": gpu_res["total_time"],
                    "eps_per_sec": gpu_res["eps_per_sec"],
                    "eps_per_hour": gpu_res["eps_per_sec"] * 3600,
                    "vram_mb": gpu_res["vram_mb"],
                }
            except torch.cuda.OutOfMemoryError:
                print(f"  OUT OF MEMORY at batch={batch}")
                torch.cuda.empty_cache()
                all_results[f"gpu_batch_{batch}"] = {"error": "OOM"}
                break
            except Exception as e:
                print(f"  ERROR: {e}")
                all_results[f"gpu_batch_{batch}"] = {"error": str(e)}
                torch.cuda.empty_cache()
    else:
        print("\n--- GPU: CUDA not available, skipping ---")

    # ── Save results ───────────────────────────────────────────────────────
    summary_path = RESULTS_DIR / "benchmark_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[Saved] {summary_path}")

    # ── Find optimal ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")

    best_cpu_key = max(
        [(k, v) for k, v in all_results.items() if k.startswith("cpu") and "eps_per_sec" in v],
        key=lambda x: x[1]["eps_per_sec"],
        default=None,
    )
    best_gpu_key = max(
        [(k, v) for k, v in all_results.items() if k.startswith("gpu") and "eps_per_sec" in v],
        key=lambda x: x[1]["eps_per_sec"],
        default=None,
    )

    if best_cpu_key:
        k, v = best_cpu_key
        print(f"\nBEST CPU: {k}")
        print(f"  {v['eps_per_sec']:.2f} eps/s = {v['eps_per_hour']:.0f} eps/hr")

    if best_gpu_key:
        k, v = best_gpu_key
        print(f"\nBEST GPU: {k}")
        print(f"  {v['eps_per_sec']:.2f} eps/s = {v['eps_per_hour']:.0f} eps/hr")
        print(f"  VRAM: {v.get('vram_mb', 0):.0f} MB")

    if best_cpu_key and best_gpu_key:
        speedup = best_gpu_key[1]["eps_per_sec"] / best_cpu_key[1]["eps_per_sec"]
        print(f"\nGPU speedup: {speedup:.1f}x")

    # ETA projections
    print(f"\n{'='*70}")
    print("ETA PROJECTIONS (from measured throughput)")
    print(f"{'='*70}")
    best_eps_hr = max(
        [v.get("eps_per_hour", 0) for v in all_results.values() if isinstance(v, dict)],
        default=1,
    )
    for label, total in [
        ("10 seeds x 1,000 eps", 10000),
        ("10 seeds x 10,000 eps", 100000),
        ("20 seeds x 10,000 eps", 200000),
        ("50 seeds x 10,000 eps", 500000),
        ("100 seeds x 10,000 eps", 1000000),
    ]:
        hours = total / best_eps_hr
        print(f"  {label:30s}: {hours:>8.1f} hours")

    print(f"\n{'='*70}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*70}")
    return all_results


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Phase 7 CPU/GPU Benchmark")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--cpu-workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--full-suite", action="store_true")
    parser.add_argument("--validate-correctness", action="store_true")
    args = parser.parse_args()

    if args.validate_correctness:
        validate_correctness(max_steps=args.max_steps)
        return

    if args.full_suite:
        run_full_suite(args)
        return

    # Single benchmark
    if args.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            print("ERROR: CUDA not available")
            sys.exit(1)
        result = benchmark_gpu_batch(args.batch_size, args.episodes, args.max_steps, warmup=args.warmup)
    else:
        if args.cpu_workers > 1:
            result = benchmark_cpu_parallel(args.cpu_workers, args.episodes, args.max_steps, warmup=args.warmup)
        else:
            result = benchmark_cpu_serial(args.episodes, args.max_steps, warmup=args.warmup)

    print(f"\nResult: {json.dumps(result, indent=2, default=str)}")


if __name__ == "__main__":
    main()
