"""Generate benchmark figures from benchmark_summary.json."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS = Path("results/phase7/benchmark")
FIGURES = RESULTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

with open(RESULTS / "benchmark_summary.json") as f:
    data = json.load(f)

cpu_eps = data["cpu_serial"]["eps_per_sec"]
gpu_configs = []
for name, val in data.items():
    if name.startswith("gpu_batch_"):
        batch = int(name.split("_")[-1])
        gpu_configs.append({
            "batch": batch,
            "eps_per_sec": val["eps_per_sec"],
            "vram_mb": val.get("vram_mb", 0),
        })
gpu_configs.sort(key=lambda x: x["batch"])

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("FlyMind Phase 7 — CPU vs GPU Benchmark\n(RTX 2050, 8-core CPU, 261-neuron connectome)", fontsize=13, fontweight="bold")

# 1. Throughput comparison (bar chart)
ax = axes[0, 0]
labels = ["CPU\nSerial"] + [f"GPU\nB={g['batch']}" for g in gpu_configs]
throughputs = [cpu_eps] + [g["eps_per_sec"] for g in gpu_configs]
colors = ["#4472C4"] + ["#ED7D31"] * len(gpu_configs)
bars = ax.bar(labels, throughputs, color=colors, edgecolor="white", linewidth=0.5)
ax.set_ylabel("Throughput (episodes/sec)")
ax.set_title("Throughput by Configuration")
for bar, val in zip(bars, throughputs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f"{val:.1f}", ha="center", va="bottom", fontsize=8)
ax.set_ylim(0, max(throughputs) * 1.15)

# 2. GPU speedup vs batch size
ax = axes[0, 1]
batches = [g["batch"] for g in gpu_configs]
speedups = [g["eps_per_sec"] / cpu_eps for g in gpu_configs]
ax.plot(batches, speedups, "o-", color="#ED7D31", linewidth=2, markersize=6)
ax.axhline(y=1.0, color="#4472C4", linestyle="--", linewidth=1, label="CPU serial baseline")
ax.set_xlabel("GPU Batch Size")
ax.set_ylabel("Speedup vs CPU Serial")
ax.set_title("GPU Speedup by Batch Size")
ax.set_xscale("log", base=2)
ax.set_xticks(batches)
ax.set_xticklabels([str(b) for b in batches])
ax.legend()
ax.grid(True, alpha=0.3)

# 3. VRAM usage
ax = axes[1, 0]
vrams = [g["vram_mb"] for g in gpu_configs]
ax.bar([f"B={g['batch']}" for g in gpu_configs], vrams, color="#70AD47", edgecolor="white")
ax.axhline(y=4290, color="red", linestyle="--", linewidth=1, label="RTX 2050 max (4.29 GB)")
ax.set_ylabel("VRAM Used (MB)")
ax.set_title("GPU Memory Usage")
ax.legend()
for i, v in enumerate(vrams):
    ax.text(i, v + 0.05, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
ax.set_ylim(0, 4500)

# 4. ETA projections for 9-hour training
ax = axes[1, 1]
seeds_range = [10, 20, 50, 100]
best_eps = max(g["eps_per_sec"] for g in gpu_configs)
cpu_etas = []
gpu_etas = []
for s in seeds_range:
    total_eps = s * 10000
    cpu_etas.append(total_eps / cpu_eps / 3600)
    gpu_etas.append(total_eps / best_eps / 3600)

x_pos = range(len(seeds_range))
width = 0.35
ax.bar([x - width/2 for x in x_pos], cpu_etas, width, label="CPU Serial", color="#4472C4")
ax.bar([x + width/2 for x in x_pos], gpu_etas, width, label=f"GPU B=50", color="#ED7D31")
ax.axhline(y=9.0, color="green", linestyle="--", linewidth=1.5, label="9-hour target")
ax.set_xticks(list(x_pos))
ax.set_xticklabels([f"{s} seeds\nx10k eps" for s in seeds_range])
ax.set_ylabel("Estimated Time (hours)")
ax.set_title("Training Time Projection")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(FIGURES / "benchmark_throughput.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"Figures saved to {FIGURES}")
print(f"  benchmark_throughput.png")
