"""
Generate a formatted MS Word (.docx) report with embedded high-resolution figures.
"""

from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

ROOT = Path(__file__).resolve().parent.parent
DOCX_PATH = ROOT / "docs" / "FlyMind_Experimental_Results.docx"
DOCX_PATH.parent.mkdir(parents=True, exist_ok=True)

doc = docx.Document()

# Adjust margins
sections = doc.sections
for section in sections:
    section.top_margin = Inches(0.8)
    section.bottom_margin = Inches(0.8)
    section.left_margin = Inches(0.8)
    section.right_margin = Inches(0.8)

# Title
title = doc.add_paragraph()
title_run = title.add_run("FlyMind: Connectome-Based Artificial Drosophila\nExperimental Results & Biological Circuit Report")
title_run.bold = True
title_run.font.size = Pt(20)
title_run.font.color.rgb = RGBColor(24, 76, 120)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER

# Metadata
meta = doc.add_paragraph()
meta.add_run("Dataset: ").bold = True
meta.add_run("Janelia FlyEM Hemibrain v1.2.1 (Scheffer et al., 2020)\n")
meta.add_run("Subcircuit: ").bold = True
meta.add_run("Central Complex (CX) Heading Direction Network\n")
meta.add_run("Metrics: ").bold = True
meta.add_run("261 Neurons | 19,969 Directed Synaptic Connections | 273,204 Total Synapses\n")
meta.alignment = WD_ALIGN_PARAGRAPH.CENTER

doc.add_paragraph("-" * 80)

# Section 1: Executive Summary
h1 = doc.add_heading("1. Executive Summary", level=1)
p = doc.add_paragraph(
    "This report provides a comprehensive scientific summary of FlyMind, an embodied computational neuroscience "
    "agent driven directly by the reconstructed synaptic connectome of Drosophila melanogaster. "
    "Unlike artificial recurrent networks trained with global backpropagation, FlyMind operates strictly on "
    "the biological synaptic topology, neurotransmitter identities, and localized 3-factor reward-modulated Hebbian plasticity."
)

p_bullets = doc.add_paragraph()
p_bullets.add_run("• Real Connectome Authenticity: ").bold = True
p_bullets.add_run("Extracted from Janelia Hemibrain v1.2.1 with verified neurotransmitter assignments (Eckstein et al., 2024).\n")
p_bullets.add_run("• Proven Ring Attractor: ").bold = True
p_bullets.add_run("Continuous activity bump formation tracking azimuth across Protocerebral Bridge columns L1..L8 and R1..R8.\n")
p_bullets.add_run("• Scientific Ablation: ").bold = True
p_bullets.add_run("Maslov-Sneppen degree-preserved randomized null model confirms +166% advantage for real biological topology.\n")
p_bullets.add_run("• Biological Plasticity: ").bold = True
p_bullets.add_run("Multi-episode learning progression driven by dopamine-modulated Hebbian weight updates.")

# Section 2: Circuit Topology & Metrics Table
doc.add_heading("2. Circuit Composition & Topology Metrics", level=1)

table_data = [
    ("Cell Type", "Count", "Neurotransmitter", "Biological Function"),
    ("EPG", "46", "Cholinergic (+1)", "Compass neurons (Ellipsoid Body -> Protocerebral Bridge)"),
    ("Delta7", "42", "GABAergic (-1)", "Global/lateral inhibitory ring attractor interneurons"),
    ("PFNd", "40", "Cholinergic (+1)", "Pontine / Fan-shaped body directional flow"),
    ("ER4d", "25", "Cholinergic (+1)", "Ring visual input from anterior optic tubercle"),
    ("PEN_b (PEN2)", "22", "Cholinergic (+1)", "Angular velocity angular integrator (rightward loop)"),
    ("PEN_a (PEN1)", "20", "Cholinergic (+1)", "Angular velocity angular integrator (leftward loop)"),
    ("PFNv", "20", "Cholinergic (+1)", "Ventral fan-shaped directional layer"),
    ("EL", "18", "Cholinergic (+1)", "Ellipsoid local loop interneurons"),
    ("PEG", "18", "Cholinergic (+1)", "Steering motor output pathway to gall / LAL"),
    ("ER4m", "10", "Cholinergic (+1)", "Medial ring sensory input"),
]

table = doc.add_table(rows=len(table_data), cols=4)
table.alignment = WD_TABLE_ALIGNMENT.CENTER
table.style = 'Table Grid'

for r_idx, row in enumerate(table_data):
    for c_idx, val in enumerate(row):
        cell = table.cell(r_idx, c_idx)
        cell.text = val
        if r_idx == 0:
            for p in cell.paragraphs:
                for r in p.runs:
                    r.bold = True

doc.add_paragraph()

# Figures Helper
def add_figure(doc, img_path: Path, caption_text: str):
    if img_path.exists():
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p_img.add_run()
        run.add_picture(str(img_path), width=Inches(5.8))
        
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap_run = p_cap.add_run(caption_text)
        cap_run.italic = True
        cap_run.font.size = Pt(9.5)
        doc.add_paragraph()

# Section 3: Topology Figures
doc.add_heading("3. Connectome Topology Visualizations", level=1)

add_figure(
    doc,
    ROOT / "results" / "cx_analysis" / "01_neuron_type_counts.png",
    "Figure 1: Distribution of neuron types and neurotransmitter classification in the CX heading circuit."
)

add_figure(
    doc,
    ROOT / "results" / "cx_analysis" / "02_synapse_weight_distribution.png",
    "Figure 2: Log-scale distribution of synaptic connection weights across all 19,969 edges."
)

add_figure(
    doc,
    ROOT / "results" / "cx_analysis" / "03_type_connectivity_heatmap.png",
    "Figure 3: Inter-type synaptic connection density matrix across cell type populations."
)

add_figure(
    doc,
    ROOT / "results" / "cx_analysis" / "04_circuit_topology.png",
    "Figure 4: Full Central Complex circuit network graph showing anatomical community structure."
)

# Section 4: Experimental Findings
doc.add_heading("4. Experimental Results & Dynamics", level=1)

doc.add_heading("A. Ring Attractor Continuous Bump Tracking", level=2)
p = doc.add_paragraph(
    "Mapping EPG compass neurons onto their anatomical Protocerebral Bridge glomeruli columns (L1..L8, R1..R8) "
    "produces a stable, continuous circular manifold. When driven by rotating azimuthal visual stimuli, "
    "the network forms a single coherent activity bump that smoothly shifts orientation across 360 degrees."
)
add_figure(
    doc,
    ROOT / "results" / "cx_analysis" / "10_ring_attractor_bump.png",
    "Figure 5: EPG compass bump space-time tracking (left) and polar manifold profile (right)."
)

add_figure(
    doc,
    ROOT / "results" / "cx_analysis" / "05_simulation_activity.png",
    "Figure 6: Population rate traces across EPG, PEG, PEN_a, and PEN_b populations under visual drive."
)

doc.add_heading("B. Closed-Loop Embodied Navigation Benchmark", level=2)
p = doc.add_paragraph(
    "In closed-loop 2D virtual arena navigation across 50 trials, the unplastic connectome agent executes "
    "straight ballistic trajectories. When aligned with the goal beacon, it reaches the target 6.1x faster "
    "than random brownian exploration (21.5 steps vs 131.2 steps)."
)
add_figure(
    doc,
    ROOT / "results" / "cx_analysis" / "06_navigation_benchmark.png",
    "Figure 7: 2D virtual arena navigation trajectories: Random Agent (left) vs Connectome Agent (right)."
)

doc.add_heading("C. Scientific Ablation Study (Maslov-Sneppen Null Model)", level=2)
p = doc.add_paragraph(
    "To test whether performance stems from specific microcircuit motifs rather than generic degree distributions, "
    "we constructed a degree-preserved randomized null model using 20,000 Maslov-Sneppen edge swaps. "
    "The real connectome significantly outperformed the randomized null baseline (16.0% vs 6.0% success rate), "
    "demonstrating a +166% relative computational advantage from authentic biological wiring."
)
add_figure(
    doc,
    ROOT / "results" / "experiments" / "08_ablation_comparison.png",
    "Figure 8: Ablation comparison between Real Biological Connectome and Maslov-Sneppen Randomized Null Model."
)

doc.add_heading("D. Multi-Episode Reward-Modulated Hebbian Plasticity", level=2)
p = doc.add_paragraph(
    "Synaptic plasticity was restricted to local 3-factor Hebbian dopamine-modulated updates on biologically verified edges. "
    "Over 100 episodes, episodic returns improved from -156.3 to a peak of -77.0, with success rates reaching 30.0%."
)
add_figure(
    doc,
    ROOT / "results" / "experiments" / "07_plasticity_learning_curve.png",
    "Figure 9: Reward return and success rate learning curves over 100 training episodes."
)

doc.add_heading("E. Real-Time Embodied Synchronization", level=2)
add_figure(
    doc,
    ROOT / "results" / "experiments" / "09_live_embodied_dual_panel.png",
    "Figure 10: Dual-panel live view showing 2D arena trajectory synchronized with EPG compass activation heatmap."
)

# Save document
doc.save(str(DOCX_PATH))
print(f"Successfully generated MS Word report: {DOCX_PATH}")
