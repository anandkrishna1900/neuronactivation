# Phase 7 CPU/GPU/HYBRID Benchmark Report

**Date:** September 14, 2026  
**Status:** COMPLETE — RL training NOT started  
**Purpose:** Find fastest stable configuration for large-scale RL training

---

## Executive Summary

| Metric | CPU Serial | GPU Batch 50 | Speedup |
|--------|-----------|-------------|---------|
| Throughput | 6.44 eps/s | 11.20 eps/s | **1.74x** |
| Hourly | 23,179 eps/hr | 40,336 eps/hr | — |
| VRAM | N/A | 9.2 MB | — |
| VRAM Headroom | — | 99.8% free | — |

**Recommendation:** Use **GPU batch size 50** for RL training. It provides 1.74x speedup over CPU serial with negligible VRAM usage (9 MB of 4,290 MB available).

---

## Hardware Profile

| Component | Spec |
|-----------|------|
| GPU | NVIDIA GeForce RTX 2050 (Laptop) |
| VRAM | 4.29 GB GDDR6 |
| CUDA Cores | 2,048 |
| SM Count | 16 |
| Compute Capability | 8.6 |
| CPU | Intel Core i5-12450H (6P+4E, 10C/12T) |
| RAM | 16.3 GB |
| OS | Windows 11 |
| Python | 3.14.6 |
| PyTorch | 2.11.0+cu128 |

---

## Connectome Profile

| Parameter | Value |
|-----------|-------|
| Total neurons | 261 |
| EPG neurons | 12 (3 per wedge) |
| PEG neurons | 12 (3L + 3R) |
| PFNd/PFNv neurons | 12 |
| Motor output | 1 (flap) |
| Total synapses | 19,969 |
| Plastic synapses | 1,888 |
| Sub-steps per timestep | 10 |
| Weight matrix | 261 × 261 float32 |

---

## Benchmark Results

### Throughput by Configuration

```
Configuration      eps/s     eps/hr     VRAM
─────────────────────────────────────────────
CPU Serial          6.44     23,179     N/A
GPU B=1             0.67      2,428     8.8 MB
GPU B=2             1.12      4,044     8.8 MB
GPU B=5             2.90     10,442     8.8 MB
GPU B=10            4.37     15,718     8.9 MB
GPU B=20            5.39     19,403     9.2 MB
GPU B=25            7.25     26,092     9.0 MB
GPU B=50           11.20     40,336     9.2 MB  ← SWEET SPOT
GPU B=75           10.72     38,601     9.2 MB
GPU B=100          10.34     37,226     9.2 MB
```

### Key Observations

1. **GPU saturates at batch 50** — throughput plateaus at ~11 eps/s, with diminishing returns beyond batch 50.

2. **GPU is slower than CPU for small batches** — GPU batch 1 (0.67 eps/s) is 9.6x slower than CPU serial (6.44 eps/s) due to CUDA kernel launch overhead on a tiny 261×261 matrix.

3. **CPU serial is the clear CPU winner** — parallel subprocess overhead is prohibitive for this workload size. The connectome is only 261 neurons; per-episode wall time is ~0.15s, leaving no room for multiprocessing overhead.

4. **VRAM is trivial** — the 261×261 weight matrix consumes only ~9 MB. The RTX 2050's 4.29 GB is 99.8% unused, meaning much larger networks or batch sizes could be supported if needed.

5. **CPU vs GPU bottleneck breakdown** — Neural computation (261×261 matmul × 10 sub-steps) is fast on both. The dominant cost is Python-level environment stepping (sensing, collision detection, pipe scrolling), which runs on CPU in both modes.

### GPU Speedup Curve

```
Batch Size    Speedup vs CPU
─────────────────────────────
    1            0.10x
    2            0.17x
    5            0.45x
   10            0.68x
   20            0.84x
   25            1.13x
   50            1.74x  ← Best
   75            1.66x
  100            1.60x
```

---

## Training Time Projections

Based on measured throughput of **11.20 eps/s** (GPU B=50):

| Scale | Total Episodes | Estimated Time | 9-Hour Target |
|-------|---------------|----------------|---------------|
| 10 seeds × 10k | 100,000 | **2.5 hours** | ✅ Fits |
| 20 seeds × 10k | 200,000 | **5.0 hours** | ✅ Fits |
| 50 seeds × 10k | 500,000 | **12.4 hours** | ❌ Exceeds |
| 100 seeds × 10k | 1,000,000 | **24.8 hours** | ❌ Exceeds |

**For 9-hour training window:** Run 20 seeds × 10,000 episodes each (fits in ~5 hours with room for checkpointing overhead).

---

## Correctness Validation

CPU vs GPU outputs match exactly across 5 test seeds (42, 100, 777, 1234, 5678) at 200 timesteps. The GPU implementation faithfully replicates:
- Sensory → EPG mapping
- 10 sub-steps of recurrent dynamics
- Motor readout (PEG asymmetry, PFNd/PFNv)
- Sigmoid decoder with temperature = 1.5

---

## Recommendations

1. **Use GPU batch size 50** for training — 1.74x speedup, trivial VRAM.
2. **Run 20 seeds × 10k episodes** for the 9-hour window — fits comfortably.
3. **Do NOT use CPU parallel** — subprocess overhead negates any parallelism gain for this workload.
4. **Consider hybrid** if environment complexity increases — CPU handles environments, GPU handles neural batch.
5. **Future optimization**: If training needs to scale beyond 20 seeds, consider:
   - Porting environment stepping to C/Rust
   - Using larger batch sizes on a bigger GPU
   - Batching multiple timesteps into a single GPU kernel launch

---

## Files Generated

```
results/phase7/benchmark/
├── benchmark_summary.json          # Raw benchmark data
├── system_info.json                # Hardware/software profile
└── figures/
    └── benchmark_throughput.png    # Throughput/speedup/VRAM/ETA charts
```

---

*Report generated by Phase 7 benchmark suite. No RL training was performed.*
