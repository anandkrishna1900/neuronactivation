"""
Generate MS Word (.docx) report for Phase 3 Diagnostic Report with embedded plots.
"""

from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

ROOT = Path(__file__).resolve().parent.parent
DOCX_PATH = ROOT / "docs" / "FlyMind_Phase3_Diagnostic_Report.docx"
DOCX_PATH.parent.mkdir(parents=True, exist_ok=True)

doc = docx.Document()

for section in doc.sections:
    section.top_margin = Inches(0.8)
    section.bottom_margin = Inches(0.8)
    section.left_margin = Inches(0.8)
    section.right_margin = Inches(0.8)

# Title
title = doc.add_paragraph()
title_run = title.add_run("FlyMind Phase 3: Diagnostic Report\nInvestigating Learning, Stability & Generalization")
title_run.bold = True
title_run.font.size = Pt(20)
title_run.font.color.rgb = RGBColor(24, 76, 120)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER

# Meta
meta = doc.add_paragraph()
meta.add_run("Dataset: ").bold = True
meta.add_run("Janelia FlyEM Hemibrain v1.2.1 | Central Complex Circuit (261 Neurons, 19,969 Synapses)\n")
meta.add_run("Diagnostic Suite: ").bold = True
meta.add_run("Environment, Sensor, Motor Audits | Reward Ablations | Plasticity Stability | Multi-Seed Replication\n")
meta.alignment = WD_ALIGN_PARAGRAPH.CENTER

doc.add_paragraph("-" * 80)

# Section 1
doc.add_heading("1. Executive Summary & Diagnostic Findings", level=1)
doc.add_paragraph(
    "Phase 3 systematically investigated why the connectome agent temporarily improves during training but fails to generalize "
    "robustly to unseen environments under randomized initial positions and orientations."
)

p_bullets = doc.add_paragraph()
p_bullets.add_run("• Sensory Blind Spot Identified: ").bold = True
p_bullets.add_run("The 180° FOV compound eye sensor provides zero input when the target is behind the fly (>50% of randomized initializations).\n")
p_bullets.add_run("• Motor Mapping Fixed: ").bold = True
p_bullets.add_run("Aligned PEG (Action 1: Forward), PEN_a (Action 2: Turn Left), and PEN_b (Action 3: Turn Right).\n")
p_bullets.add_run("• Plasticity Dispersion: ").bold = True
p_bullets.add_run("Over 98% of all 19,969 synaptic connections undergo Hebbian updates, degrading internal recurrent compass dynamics.\n")
p_bullets.add_run("• Random Baseline Diffusion: ").bold = True
p_bullets.add_run("Random Brownian motion achieves 13.0% success by pure spatial diffusion in a bounded arena over 400 steps.")

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

# Section 2: Audit Figures
doc.add_heading("2. Diagnostic Audits (Environment, Sensor, Motor, 1D Task)", level=1)

add_figure(
    doc,
    ROOT / "results" / "phase3" / "01_environment_spatial_audit.png",
    "Figure 1: Environment Spatial Audit: 20 randomized start/target pairs with min distance >= 25.0 px."
)

add_figure(
    doc,
    ROOT / "results" / "phase3" / "02_sensor_receptive_fields.png",
    "Figure 2: Compound Eye Receptive Field Tuning Curves showing active vision in [-90°, +90°] and zero drive in rear hemisphere."
)

add_figure(
    doc,
    ROOT / "results" / "phase3" / "03_1d_heading_learning.png",
    "Figure 3: 1D Heading Alignment Learning Error Progression."
)

# Section 3: Experiments 5-11
doc.add_heading("3. Reward, Stability & Checkpoint Analysis", level=1)

add_figure(
    doc,
    ROOT / "results" / "phase3" / "05_reward_ablation.png",
    "Figure 4: Reward Formulation Ablation comparing Dense, Sparse, No-Time-Penalty, Scaled Progress, and Terminal-Only rewards."
)

add_figure(
    doc,
    ROOT / "results" / "phase3" / "06_plasticity_stability.png",
    "Figure 5: Synaptic Weight Dynamics: Mean weight and count of modified edges across 300 training episodes."
)

add_figure(
    doc,
    ROOT / "results" / "phase3" / "07_checkpoint_analysis.png",
    "Figure 6: Frozen Checkpoint Evaluation across 100 unseen seeds from Episode 0 to Episode 500."
)

add_figure(
    doc,
    ROOT / "results" / "phase3" / "08_curriculum_stages.png",
    "Figure 7: Curriculum Learning Performance progression across 5 developmental stages."
)

add_figure(
    doc,
    ROOT / "results" / "phase3" / "10_multiseed_replication.png",
    "Figure 8: 10-Seed Replication Distribution vs Thorough 200-Trial Random Baseline."
)

# Section 4: Recommendations
doc.add_heading("4. Summary & Recommended Next Actions", level=1)
p = doc.add_paragraph(
    "1. WHAT WORKS: Stable EPG ring attractor dynamics, fast goal arrival when target is in front field of view, bounded 3-factor Hebbian plasticity.\n"
    "2. WHAT DOES NOT WORK: Blind-spot exploration when target spawns in rear hemisphere; global unconstrained plasticity across all recurrent synapses.\n"
    "3. RECOMMENDED NEXT EXPERIMENT: Implement full panoramic compound eye optics (or exploratory saccades when blind) combined with pathway-specific plasticity masking (plasticity restricted to ER4d->EPG and EPG->PEG while keeping recurrent ring compass loops fixed)."
)

doc.save(str(DOCX_PATH))
print(f"Successfully generated Phase 3 Word report: {DOCX_PATH}")
