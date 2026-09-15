"""
Phase 7B Pilot with soft-bound plasticity fix.
Runs 5 seeds x 1000 episodes for both baseline and expanded.
"""

import sys, io, time, json, gc, csv
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, 'src')

import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "phase7b"
CHECKPOINTS = RESULTS / "checkpoints"
VALIDATION = RESULTS / "validation"
METRICS = RESULTS / "metrics"

for d in [RESULTS, CHECKPOINTS, VALIDATION, METRICS]:
    d.mkdir(parents=True, exist_ok=True)

PILOT_SEEDS = [42, 100, 777, 1234, 2026]
N_EPISODES = 1000
MAX_STEPS = 2000


def run_pilot_for_network(expanded, connectome_path, tag):
    from flymind.agent.flappy_agent_phase7b import FlyMindRLAgentExpanded
    from flymind.environment.flappy import FlappyEnvironment
    from flymind.environment.flappy_reward import FlappyRewardShaper

    print()
    print('=' * 60)
    print(f'PILOT: {tag.upper()} NETWORK (soft-bound fix)')
    print('=' * 60)

    shaper = FlappyRewardShaper()
    plastic_mask = None
    n_plastic = 0
    results = []

    for seed_idx, seed in enumerate(PILOT_SEEDS):
        print(f'\n  [{seed_idx+1}/5] Seed {seed}')
        agent = FlyMindRLAgentExpanded(
            connectome_path=connectome_path,
            seed=seed,
            expanded=expanded,
        )
        if plastic_mask is None:
            plastic_mask = agent.plasticity_mask
            n_plastic = int(np.sum(plastic_mask > 0))

        env = FlappyEnvironment(seed=seed, max_steps=MAX_STEPS)
        t0 = time.time()
        scores = []
        steps_list = []
        flap_counts = []
        best_score = 0
        weight_stats = []

        for ep in range(N_EPISODES):
            env.reset(seed=ep)
            agent.reset()
            total_reward = 0.0
            flap_count = 0

            for step in range(MAX_STEPS):
                state = env.get_state()
                action = agent.act(state)
                if action == 1:
                    flap_count += 1
                next_state, env_reward, done, info = env.step(action)
                shaped = shaper.shape(env_reward, info, done)
                total_reward += shaped
                if agent.enable_plasticity:
                    agent.apply_step_reward(shaped)
                if done:
                    break

            agent.end_episode(total_reward)
            score = state.score
            scores.append(score)
            steps_list.append(step + 1)
            flap_counts.append(flap_count / max(step + 1, 1))
            best_score = max(best_score, score)

            # Record weight stats every 100 episodes
            if ep % 100 == 0:
                W = agent.network.synapses.raw_weights
                W_p = W[plastic_mask > 0]
                weight_stats.append({
                    'episode': ep,
                    'mean_plast': float(np.mean(W_p)),
                    'max_plast': float(np.max(W_p)),
                    'sat_count': int(np.sum(W_p >= 99.0)),
                })

            if ep % 200 == 0:
                elapsed = time.time() - t0
                eps_per_sec = (ep + 1) / max(elapsed, 1e-6)
                rolling = float(np.mean(scores[-100:])) if scores else 0
                W = agent.network.synapses.raw_weights
                W_p = W[plastic_mask > 0]
                print(f'    ep={ep:4d}/{N_EPISODES} score={score:3d} roll={rolling:.1f} '
                      f'best={best_score} w_max={np.max(W_p):.0f} {eps_per_sec:.1f} ep/s')

        elapsed = time.time() - t0
        eps_per_sec = N_EPISODES / max(elapsed, 1e-6)

        W = agent.network.synapses.raw_weights
        W_p = W[plastic_mask > 0]

        result = {
            'seed': seed,
            'episodes': N_EPISODES,
            'best_score': best_score,
            'mean_score': float(np.mean(scores)),
            'median_score': float(np.median(scores)),
            'std_score': float(np.std(scores)),
            'mean_steps': float(np.mean(steps_list)),
            'mean_flap_rate': float(np.mean(flap_counts)),
            'eps_per_sec': eps_per_sec,
            'elapsed_s': elapsed,
            'final_mean_plast': float(np.mean(W_p)),
            'final_max_plast': float(np.max(W_p)),
            'final_sat_count': int(np.sum(W_p >= 99.0)),
            'n_plastic': n_plastic,
            'scores': scores,
            'weight_stats': weight_stats,
        }
        results.append(result)

        print(f'    DONE: mean={result["mean_score"]:.1f} best={best_score} '
              f'w_max={np.max(W_p):.0f} sat={int(np.sum(W_p >= 99.0))}/{n_plastic} '
              f'{eps_per_sec:.1f} ep/s')

        del agent
        gc.collect()

    # Summary
    all_best = [r['best_score'] for r in results]
    all_mean = [r['mean_score'] for r in results]
    all_sat = [r['final_sat_count'] for r in results]

    summary = {
        'tag': tag,
        'n_seeds': len(PILOT_SEEDS),
        'n_episodes': N_EPISODES,
        'results': results,
        'mean_best': float(np.mean(all_best)),
        'mean_mean': float(np.mean(all_mean)),
        'max_best': int(np.max(all_best)),
        'mean_saturation': float(np.mean(all_sat)),
    }

    print(f'\n  PILOT SUMMARY ({tag}, soft-bound):')
    print(f'    Best scores: {all_best}')
    print(f'    Mean scores: {[f"{x:.1f}" for x in all_mean]}')
    print(f'    Mean of means: {np.mean(all_mean):.1f}')
    print(f'    Max best: {np.max(all_best)}')
    print(f'    Final saturation: {all_sat} (mean={np.mean(all_sat):.0f}/{n_plastic})')

    return summary


def main():
    print('=' * 60)
    print('FLYMIND PHASE 7B - PILOT (SOFT-BOUND FIX)')
    print('=' * 60)

    # Baseline
    baseline = run_pilot_for_network(
        expanded=False,
        connectome_path='data/processed/cx_heading_v1.json',
        tag='baseline',
    )
    with open(RESULTS / 'pilot_results_baseline_softbound.json', 'w') as f:
        json.dump(baseline, f, indent=2)

    # Expanded
    expanded = run_pilot_for_network(
        expanded=True,
        connectome_path='data/processed/cx_heading_v1_expanded.json',
        tag='expanded',
    )
    with open(RESULTS / 'pilot_results_expanded_softbound.json', 'w') as f:
        json.dump(expanded, f, indent=2)

    # Comparison
    print()
    print('=' * 60)
    print('PILOT COMPARISON')
    print('=' * 60)
    print(f'  Baseline (261N): mean_best={baseline["mean_best"]:.1f}, max_best={baseline["max_best"]}, sat={baseline["mean_saturation"]:.0f}/1888')
    print(f'  Expanded (427N): mean_best={expanded["mean_best"]:.1f}, max_best={expanded["max_best"]}, sat={expanded["mean_saturation"]:.0f}/1888')

    baseline_pass = max(r['best_score'] for r in baseline['results']) > 0
    expanded_pass = max(r['best_score'] for r in expanded['results']) > 0
    # Also check that saturation is low
    sat_ok = baseline['mean_saturation'] < 100 and expanded['mean_saturation'] < 100

    print(f'\n  Baseline pilot: {"PASS" if baseline_pass else "FAIL"}')
    print(f'  Expanded pilot: {"PASS" if expanded_pass else "FAIL"}')
    print(f'  Saturation OK:  {"PASS" if sat_ok else "FAIL"} ({baseline["mean_saturation"]:.0f} and {expanded["mean_saturation"]:.0f} saturated)')

    if baseline_pass or expanded_pass:
        print('\n  [OK] Pilot passed. Ready for full training.')
    else:
        print('\n  [FAIL] Pilot failed.')


if __name__ == '__main__':
    main()
