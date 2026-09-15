"""
9-hour Phase 7 training configuration.

Seeds:       42-51 (10 seeds)
Episodes:    14,000 per seed
Max steps:   5,000 per episode
Parallel:    5 seeds simultaneously
Checkpoint:  every 500 episodes
Resume:      auto from latest checkpoint

Estimated time: ~9 hours

To START training:
    python tools/train_9hr.py

To PAUSE (Ctrl+C saves progress automatically):
    Press Ctrl+C

To RESUME after pause:
    python tools/train_9hr.py

To check progress:
    python tools/train_9hr.py --status
"""

import subprocess
import sys
import time
import signal
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = ROOT / "results" / "phase7" / "checkpoints"
LOG_DIR = ROOT / "results" / "phase7" / "logs"

# ── Configuration ─────────────────────────────────────────────────────────────
SEEDS = list(range(42, 52))        # 10 seeds
EPISODES = 14000                   # per seed (~8.1h training + eval = ~9h total)
MAX_STEPS = 5000                   # per episode
CHECKPOINT_INTERVAL = 500          # save every 500 episodes
PARALLEL_JOBS = 5                  # seeds in parallel
EVAL_SEEDS = 20                    # validation seeds per eval
# ──────────────────────────────────────────────────────────────────────────────


def find_latest_ckpt(seed: int) -> str:
    """Find latest checkpoint for a seed."""
    if not CKPT_DIR.exists():
        return ""
    pattern = f"*_s{seed}_*_latest_train_state.npz"
    matches = sorted(CKPT_DIR.glob(pattern), reverse=True)
    if matches:
        ckpt = str(matches[0]).replace("_train_state.npz", ".npz")
        if Path(ckpt).exists():
            return ckpt
    return ""


def get_seed_progress(seed: int) -> dict:
    """Check how far a seed has trained."""
    if not CKPT_DIR.exists():
        return {"trained": False, "episode": 0}
    pattern = f"*_s{seed}_*_latest_train_state.npz"
    matches = sorted(CKPT_DIR.glob(pattern), reverse=True)
    if not matches:
        return {"trained": False, "episode": 0}
    try:
        import numpy as np
        data = np.load(matches[0], allow_pickle=True)
        ep = int(data.get("episode", 0))
        best = int(data.get("best_score", 0))
        return {"trained": True, "episode": ep, "best_score": best, "path": str(matches[0])}
    except Exception:
        return {"trained": False, "episode": 0}


def show_status():
    """Print status of all seeds."""
    print(f"\n{'='*60}")
    print("PHASE 7 TRAINING STATUS")
    print(f"{'='*60}")
    print(f"{'Seed':>6} | {'Episode':>8} | {'Progress':>10} | {'Best':>5}")
    print("-" * 45)
    for s in SEEDS:
        p = get_seed_progress(s)
        if p["trained"]:
            pct = p["episode"] / EPISODES * 100
            print(f"{s:>6} | {p['episode']:>8} | {pct:>8.1f}% | {p.get('best_score', '?')}")
        else:
            print(f"{s:>6} | {'--':>8} | {'not started':>10} | --")
    print(f"{'='*60}")
    total_eps = sum(get_seed_progress(s).get("episode", 0) for s in SEEDS)
    print(f"Total episodes trained: {total_eps} / {EPISODES * len(SEEDS)}")
    print(f"{'='*60}")


def launch_seed(seed: int, resume: bool = True) -> subprocess.Popen:
    """Launch training for a single seed."""
    cmd = [
        sys.executable,
        str(ROOT / "experiments" / "phase7_flappy_rl.py"),
        "--episodes", str(EPISODES),
        "--seed", str(seed),
        "--checkpoint-interval", str(CHECKPOINT_INTERVAL),
        "--max-steps", str(MAX_STEPS),
        "--eval-seeds", str(EVAL_SEEDS),
    ]

    if resume:
        ckpt = find_latest_ckpt(seed)
        if ckpt:
            cmd.extend(["--resume", ckpt])

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    f_out = open(LOG_DIR / f"run9h_s{seed}.out", "w")
    f_err = open(LOG_DIR / f"run9h_s{seed}.err", "w")

    return subprocess.Popen(cmd, stdout=f_out, stderr=f_err, cwd=str(ROOT))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="9-hour Phase 7 training")
    parser.add_argument("--status", action="store_true", help="Show training progress")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh (ignore checkpoints)")
    parser.add_argument("--dry-run", action="store_true", help="Show commands only")
    args = parser.parse_args()

    if args.status:
        show_status()
        return

    print(f"\n{'='*70}")
    print("PHASE 7 FLYMIND RL — 9-HOUR TRAINING RUN")
    print(f"{'='*70}")
    print(f"Seeds:           {SEEDS[0]}-{SEEDS[-1]} ({len(SEEDS)} seeds)")
    print(f"Episodes/seed:   {EPISODES}")
    print(f"Max steps/ep:    {MAX_STEPS}")
    print(f"Checkpoint every:{CHECKPOINT_INTERVAL} episodes")
    print(f"Parallel jobs:   {PARALLEL_JOBS}")
    print(f"Estimated time:  ~9 hours")
    print(f"Resume:          {'auto' if not args.no_resume else 'disabled'}")
    print(f"{'='*70}")

    # Check existing progress
    if not args.no_resume:
        any_progress = False
        for s in SEEDS:
            p = get_seed_progress(s)
            if p["trained"] and p["episode"] > 0:
                if not any_progress:
                    print("\nExisting progress found:")
                    any_progress = True
                print(f"  Seed {s}: episode {p['episode']}/{EPISODES} (best={p.get('best_score', '?')})")
        if any_progress:
            print("  Resuming from where each seed left off.\n")

    if args.dry_run:
        for s in SEEDS:
            ckpt = find_latest_ckpt(s) if not args.no_resume else ""
            resume_flag = f" --resume {ckpt}" if ckpt else ""
            print(f"  python experiments/phase7_flappy_rl.py --episodes {EPISODES} --seed {s} --max-steps {MAX_STEPS} --checkpoint-interval {CHECKPOINT_INTERVAL}{resume_flag}")
        return

    # Launch seeds in batches
    all_procs = []
    t0 = time.time()

    def cleanup(sig, frame):
        print(f"\n[Ctrl+C] Stopping {len(all_procs)} processes (progress saved automatically)...")
        for s, p in all_procs:
            try:
                p.terminate()
            except Exception:
                pass
        for s, p in all_procs:
            try:
                p.wait(timeout=10)
            except Exception:
                p.kill()
        show_status()
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)

    try:
        for batch_start in range(0, len(SEEDS), PARALLEL_JOBS):
            batch = SEEDS[batch_start:batch_start + PARALLEL_JOBS]
            batch_procs = []
            for s in batch:
                p = launch_seed(s, resume=not args.no_resume)
                batch_procs.append((s, p))
                all_procs.append((s, p))
                print(f"  [Seed {s}] Started (PID {p.pid})")

            # Wait for batch
            for s, p in batch_procs:
                p.wait()
                rc = p.returncode
                status = "DONE" if rc == 0 else f"FAIL(rc={rc})"
                elapsed = time.time() - t0
                print(f"  [Seed {s}] {status} ({elapsed:.0f}s elapsed)")

        total = time.time() - t0
        print(f"\n{'='*70}")
        print(f"TRAINING COMPLETE — {total:.0f}s ({total/3600:.1f} hours)")
        print(f"{'='*70}")
        show_status()

    except SystemExit:
        pass
    except Exception as e:
        print(f"\nError: {e}")
        show_status()


if __name__ == "__main__":
    main()
