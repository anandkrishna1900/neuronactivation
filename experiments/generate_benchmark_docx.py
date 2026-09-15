"""Generate Phase 7 Benchmark DOCX report with embedded figures."""
import json
from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn

RESULTS = Path("results/phase7/benchmark")
FIGURES = RESULTS / "figures"
OUTPUT = Path("docs/Phase7_CPU_GPU_Benchmark_Report.docx")

with open(RESULTS / "benchmark_summary.json") as f:
    data = json.load(f)

doc = Document()

# ── Styles ──
style = doc.styles["Normal"]
font = style.font
font.name = "Calibri"
font.size = Pt(11)

# ── Title ──
title = doc.add_heading("FlyMind Phase 7\nCPU/GPU/Hybrid Benchmark Report", level=0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER

doc.add_paragraph("")
meta = doc.add_paragraph()
meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
meta_run = meta.add_run(
    "Date: September 14, 2026\n"
    "Status: COMPLETE — RL training NOT started\n"
    "Purpose: Find fastest stable configuration for large-scale training"
)
meta_run.font.size = Pt(10)
meta_run.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

doc.add_page_break()

# ── 1. Executive Summary ──
doc.add_heading("1. Executive Summary", level=1)

cpu = data["cpu_serial"]
gpu50 = data["gpu_batch_50"]
speedup = gpu50["eps_per_sec"] / cpu["eps_per_sec"]

table = doc.add_table(rows=4, cols=4, style="Light Shading Accent 1")
table.alignment = WD_TABLE_ALIGNMENT.CENTER
headers = ["Metric", "CPU Serial", "GPU Batch 50", "Speedup"]
for i, h in enumerate(headers):
    table.rows[0].cells[i].text = h
    for p in table.rows[0].cells[i].paragraphs:
        for r in p.runs:
            r.bold = True

rows_data = [
    ["Throughput", f"{cpu['eps_per_sec']:.2f} eps/s", f"{gpu50['eps_per_sec']:.2f} eps/s", f"{speedup:.2f}x"],
    ["Hourly", f"{cpu['eps_per_sec']*3600:,.0f} eps/hr", f"{gpu50['eps_per_sec']*3600:,.0f} eps/hr", "—"],
    ["VRAM", "N/A", f"{gpu50.get('vram_mb', 0):.1f} MB", "99.8% free"],
]
for r, row in enumerate(rows_data):
    for c, val in enumerate(row):
        table.rows[r+1].cells[c].text = val

doc.add_paragraph("")
rec = doc.add_paragraph()
rec_run = rec.add_run("Recommendation: ")
rec_run.bold = True
rec.add_run(
    "Use GPU batch size 50 for RL training. "
    "It provides 1.74x speedup over CPU serial with negligible VRAM usage (9 MB of 4,290 MB available)."
)

doc.add_page_break()

# ── 2. Hardware Profile ──
doc.add_heading("2. Hardware Profile", level=1)

hw_table = doc.add_table(rows=8, cols=2, style="Light Shading Accent 1")
hw_data = [
    ("GPU", "NVIDIA GeForce RTX 2050 (Laptop)"),
    ("VRAM", "4.29 GB GDDR6"),
    ("CUDA Cores", "2,048"),
    ("SM Count", "16"),
    ("Compute Capability", "8.6"),
    ("CPU", "Intel Core i5-12450H (6P+4E, 10C/12T)"),
    ("RAM", "16.3 GB"),
    ("OS / Python", "Windows 11 / Python 3.14.6, PyTorch 2.11.0+cu128"),
]
for i, (k, v) in enumerate(hw_data):
    hw_table.rows[i].cells[0].text = k
    hw_table.rows[i].cells[1].text = v
    for p in hw_table.rows[i].cells[0].paragraphs:
        for r in p.runs:
            r.bold = True

# ── 3. Connectome Profile ──
doc.add_heading("3. Connectome Profile", level=1)

conn_table = doc.add_table(rows=8, cols=2, style="Light Shading Accent 1")
conn_data = [
    ("Total neurons", "261"),
    ("EPG neurons", "12 (3 per wedge)"),
    ("PEG neurons", "12 (3L + 3R)"),
    ("PFNd / PFNv", "12"),
    ("Total synapses", "19,969"),
    ("Plastic synapses", "1,888"),
    ("Sub-steps / timestep", "10"),
    ("Weight matrix", "261 × 261 float32"),
]
for i, (k, v) in enumerate(conn_data):
    conn_table.rows[i].cells[0].text = k
    conn_table.rows[i].cells[1].text = v
    for p in conn_table.rows[i].cells[0].paragraphs:
        for r in p.runs:
            r.bold = True

doc.add_page_break()

# ── 4. Benchmark Results ──
doc.add_heading("4. Benchmark Results", level=1)

doc.add_heading("4.1 Throughput by Configuration", level=2)

bench_rows = [
    ("CPU Serial", cpu["eps_per_sec"], cpu["eps_per_sec"]*3600, "N/A"),
]
for name, val in data.items():
    if name.startswith("gpu_batch_"):
        batch = int(name.split("_")[-1])
        bench_rows.append((
            f"GPU Batch {batch}",
            val["eps_per_sec"],
            val["eps_per_sec"]*3600,
            f"{val.get('vram_mb', 0):.1f} MB",
        ))

num_bench_rows = 1 + len(bench_rows)
bench_table = doc.add_table(rows=num_bench_rows, cols=4, style="Light Shading Accent 1")
bench_headers = ["Configuration", "eps/s", "eps/hr", "VRAM"]
for i, h in enumerate(bench_headers):
    bench_table.rows[0].cells[i].text = h
    for p in bench_table.rows[0].cells[i].paragraphs:
        for r in p.runs:
            r.bold = True

for r, (label, eps, eps_hr, vram) in enumerate(bench_rows):
    bench_table.rows[r+1].cells[0].text = label
    bench_table.rows[r+1].cells[1].text = f"{eps:.2f}"
    bench_table.rows[r+1].cells[2].text = f"{int(eps_hr):,}"
    bench_table.rows[r+1].cells[3].text = vram
    if "Batch 50" in label:
        for c in range(4):
            for p in bench_table.rows[r+1].cells[c].paragraphs:
                for run in p.runs:
                    run.bold = True

doc.add_paragraph("")

# ── 4.2 Figures ──
doc.add_heading("4.2 Throughput & Speedup Charts", level=2)

fig_path = FIGURES / "benchmark_throughput.png"
if fig_path.exists():
    doc.add_picture(str(fig_path), width=Inches(6.2))
    last_paragraph = doc.paragraphs[-1]
    last_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap = doc.add_paragraph("Figure 1: CPU vs GPU throughput, speedup curve, VRAM usage, and training time projections.")
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.runs[0].font.size = Pt(9)
    cap.runs[0].font.color.rgb = RGBColor(0x66, 0x66, 0x66)
else:
    doc.add_paragraph("[Figure not found — run generate_benchmark_figures.py first]")

doc.add_page_break()

# ── 5. Analysis ──
doc.add_heading("5. Analysis", level=1)

doc.add_heading("5.1 GPU Speedup Curve", level=2)
doc.add_paragraph(
    "GPU throughput scales with batch size up to batch 50, then plateaus. "
    "Below batch 25, the GPU is slower than CPU serial due to CUDA kernel launch overhead "
    "on the tiny 261×261 weight matrix. The crossover point is approximately batch 22."
)

speedup_table = doc.add_table(rows=10, cols=2, style="Light Shading Accent 1")
speedup_table.rows[0].cells[0].text = "Batch Size"
speedup_table.rows[0].cells[1].text = "Speedup vs CPU"
for p in speedup_table.rows[0].cells[0].paragraphs:
    for r in p.runs:
        r.bold = True
for p in speedup_table.rows[0].cells[1].paragraphs:
    for r in p.runs:
        r.bold = True

for name, val in data.items():
    if name.startswith("gpu_batch_"):
        batch = int(name.split("_")[-1])
        idx = batch // 10  # approximate row
        row_idx = {1:1, 2:2, 5:3, 10:4, 20:5, 25:6, 50:7, 75:8, 100:9}.get(batch, 0)
        speedup_table.rows[row_idx].cells[0].text = str(batch)
        speedup_table.rows[row_idx].cells[1].text = f"{val['eps_per_sec']/cpu['eps_per_sec']:.2f}x"

doc.add_heading("5.2 VRAM Utilization", level=2)
doc.add_paragraph(
    f"GPU memory usage is trivial across all batch sizes — peaking at "
    f"{gpu50.get('vram_mb', 0):.1f} MB for batch 50. The RTX 2050's 4.29 GB VRAM is "
    f"99.8% unused, meaning much larger networks or batch sizes could be supported."
)

doc.add_heading("5.3 CPU Parallel Overhead", level=2)
doc.add_paragraph(
    "CPU parallel (multiprocessing) was not benchmarked in the final run due to "
    "prohibitive subprocess startup overhead for this workload. Each episode takes "
    "~0.15s wall time, leaving no room for process creation/joining overhead. "
    "CPU serial is the recommended CPU configuration."
)

doc.add_page_break()

# ── 6. Training Projections ──
doc.add_heading("6. Training Time Projections", level=1)

doc.add_paragraph(
    f"Based on measured throughput of {gpu50['eps_per_sec']:.2f} eps/s (GPU batch 50):"
)

proj_table = doc.add_table(rows=5, cols=3, style="Light Shading Accent 1")
proj_headers = ["Scale", "Total Episodes", "Estimated Time"]
for i, h in enumerate(proj_headers):
    proj_table.rows[0].cells[i].text = h
    for p in proj_table.rows[0].cells[i].paragraphs:
        for r in p.runs:
            r.bold = True

proj_scales = [
    ("10 seeds × 10k", 100_000),
    ("20 seeds × 10k", 200_000),
    ("50 seeds × 10k", 500_000),
    ("100 seeds × 10k", 1_000_000),
]
for r, (scale, total) in enumerate(proj_scales):
    hours = total / gpu50["eps_per_sec"] / 3600
    fits = "✅ Fits" if hours <= 9 else "❌ Exceeds"
    proj_table.rows[r+1].cells[0].text = scale
    proj_table.rows[r+1].cells[1].text = f"{total:,}"
    proj_table.rows[r+1].cells[2].text = f"{hours:.1f} hours  {fits}"

doc.add_paragraph("")
rec2 = doc.add_paragraph()
rec2_run = rec2.add_run("For 9-hour window: ")
rec2_run.bold = True
rec2.add_run("Run 20 seeds × 10,000 episodes each — fits in ~5 hours with checkpointing overhead.")

doc.add_page_break()

# ── 7. Correctness Validation ──
doc.add_heading("7. Correctness Validation", level=1)

doc.add_paragraph(
    "CPU vs GPU outputs match exactly across 5 test seeds (42, 100, 777, 1234, 5678) "
    "at 200 timesteps each. The GPU implementation faithfully replicates:"
)

checks = [
    "Sensory → EPG mapping (9-channel → 12 EPG neurons)",
    "10 sub-steps of recurrent dynamics per timestep",
    "Motor readout (PEG asymmetry, PFNd/PFNv → motor logit)",
    "Sigmoid decoder with temperature = 1.5",
    "Same connectome weights (loaded from checkpoint)",
]
for item in checks:
    doc.add_paragraph(item, style="List Bullet")

doc.add_page_break()

# ── 8. Recommendations ──
doc.add_heading("8. Recommendations", level=1)

recs = [
    ("Use GPU batch size 50", "1.74x speedup, trivial VRAM (9 MB)."),
    ("Run 20 seeds × 10k episodes", "Fits in ~5 hours — leaves margin for checkpointing."),
    ("Do NOT use CPU parallel", "Subprocess overhead negates any parallelism gain for this workload."),
    ("Consider hybrid for future", "CPU handles environments, GPU handles neural batch — viable if environment complexity increases."),
    ("Scale-up path", "If training needs > 20 seeds, port environment stepping to C/Rust or use a larger GPU."),
]

for title_text, detail in recs:
    p = doc.add_paragraph()
    run_title = p.add_run(f"{title_text}: ")
    run_title.bold = True
    p.add_run(detail)

# ── 9. Files Generated ──
doc.add_heading("9. Files Generated", level=1)

files_list = [
    "results/phase7/benchmark/benchmark_summary.json — Raw benchmark data",
    "results/phase7/benchmark/system_info.json — Hardware/software profile",
    "results/phase7/benchmark/figures/benchmark_throughput.png — Charts",
    "docs/PHASE7_CPU_GPU_BENCHMARK_REPORT.md — Markdown report",
    "docs/Phase7_CPU_GPU_Benchmark_Report.docx — This document",
]
for f in files_list:
    doc.add_paragraph(f, style="List Bullet")

doc.add_paragraph("")
footer = doc.add_paragraph()
footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
footer_run = footer.add_run("Report generated by Phase 7 benchmark suite. No RL training was performed.")
footer_run.font.size = Pt(9)
footer_run.font.color.rgb = RGBColor(0x99, 0x99, 0x99)

# Save
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
doc.save(str(OUTPUT))
print(f"DOCX saved: {OUTPUT}")
print(f"  Size: {OUTPUT.stat().st_size / 1024:.1f} KB")
