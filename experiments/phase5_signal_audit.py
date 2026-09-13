"""Phase 5 Steps 2+3+6: Signal Propagation Audit, Synapse Scale Sweep, Ring Attractor Stability."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from flymind.connectome.loader import ConnectomeLoader
from flymind.brain.network import NeuralNetwork
from flymind.brain.neuron import RateNeuron

GRAPH_PATH  = ROOT / "data" / "processed" / "cx_heading_v1.json"
RESULTS_DIR = ROOT / "results" / "phase5"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SCALES = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
SETTLE, STIM, PULSE = 30, 50, 2.0

PTYPE_MAP = {
    "ER4d":   {"ER4d"},
    "ER4m":   {"ER4m"},
    "ER4":    {"ER4d", "ER4m"},
    "EPG":    {"EPG"},
    "Delta7": {"Delta7"},
    "PEG":    {"PEG"},
    "PEN_a":  {"PEN_a(PEN1)"},
    "PEN_b":  {"PEN_b(PEN2)"},
}

def get_pops(net):
    result = {}
    for label, types in PTYPE_MAP.items():
        result[label] = [net.id_to_idx[n] for n in net.neuron_ids
                         if net.graph.nx_graph.nodes[n].get("cell_type") in types]
    return result

def mstats(act, idx):
    if not idx: return 0.0, 0.0
    v = act[idx]
    return float(np.mean(v)), float(np.std(v))

def run_audit(scale, bio):
    net  = NeuralNetwork(graph=bio, neuron_model_cls=RateNeuron, synapse_scale=scale)
    pops = get_pops(net)
    net.neurons.reset()
    null = np.zeros(net.num_neurons)
    for _ in range(SETTLE): act = net.step(null)
    baseline = {p: mstats(act, idx) for p, idx in pops.items()}

    pulse = np.zeros(net.num_neurons)
    for i in pops["ER4"]: pulse[i] = PULSE
    sat = False
    for _ in range(STIM):
        act = net.step(pulse)
        if np.any(~np.isfinite(act)) or np.mean(act) > 0.98:
            sat = True; break
    peak = act

    res = {}
    for p, idx in pops.items():
        if not idx: continue
        bm, bs = baseline[p]; sm, ss = mstats(peak, idx); dv = sm - bm
        res[p] = dict(baseline_mean=bm, baseline_std=bs, stim_mean=sm, stim_std=ss,
                      abs_change=dv, pct_change=(dv/(bm+1e-9))*100, snr=dv/(bs+1e-9))

    mm = [mstats(peak, pops["PEG"])[0], mstats(peak, pops["PEN_a"])[0], mstats(peak, pops["PEN_b"])[0]]
    disc = float(max(mm) - min(mm))

    net.neurons.reset()
    for _ in range(SETTLE): net.step(null)
    eidx = pops["EPG"]; ne = len(eidx)
    bamps, bpos = [], []
    for t in range(60):
        tb = int((t / 60.0) * ne)
        ext = np.zeros(net.num_neurons)
        for k in range(3): ext[eidx[(tb + k) % ne]] = PULSE * 0.5
        a = net.step(ext); ea = a[eidx]
        angs = np.linspace(0, 2 * np.pi, ne, endpoint=False)
        cx, cy = np.sum(ea * np.cos(angs)), np.sum(ea * np.sin(angs))
        bpos.append(float(np.arctan2(cy, cx)))
        bamps.append(float(np.max(ea) - np.min(ea)))
    stable = np.std(bamps) < 0.25 and np.mean(bamps) > 0.01 and not sat
    return dict(scale=scale, pop_results=res, motor_disc=disc,
                bump_amp_mean=float(np.mean(bamps)), bump_amp_std=float(np.std(bamps)),
                bump_pos=bpos, ring_stable=bool(stable), saturation=sat, motor_means=mm)

def main():
    print("=" * 60)
    print("Phase 5: Signal Propagation Audit + Scale Sweep + Ring Stability")
    print("=" * 60)
    bio = ConnectomeLoader.load_from_json(GRAPH_PATH)
    rows, sd = [], []
    for sc in SCALES:
        print("\n[scale=" + str(sc) + "]")
        r = run_audit(sc, bio); sd.append(r)
        epg = r["pop_results"].get("EPG", {}); peg = r["pop_results"].get("PEG", {})
        print("  EPG stim=" + str(round(epg.get("stim_mean",0), 5)) +
              " SNR=" + str(round(epg.get("snr",0), 3)))
        print("  PEG stim=" + str(round(peg.get("stim_mean",0), 5)) +
              " SNR=" + str(round(peg.get("snr",0), 3)))
        print("  disc=" + str(round(r["motor_disc"], 6)) +
              " bump=" + str(round(r["bump_amp_mean"], 4)) +
              " stable=" + str(r["ring_stable"]) + " sat=" + str(r["saturation"]))
        for p, m in r["pop_results"].items():
            row = dict(synapse_scale=sc, population=p)
            row.update(m)
            row.update(motor_disc=r["motor_disc"], bump_amp_mean=r["bump_amp_mean"],
                       bump_amp_std=r["bump_amp_std"], ring_stable=r["ring_stable"],
                       saturation=r["saturation"])
            rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = RESULTS_DIR / "synapse_scale_audit.csv"
    df.to_csv(csv_path, index=False)
    print("\n[SAVED] " + str(csv_path))

    print("\nSCALE SELECTION:")
    rec = None
    for r in sd:
        snr = r["pop_results"].get("PEG", {}).get("snr", 0.0)
        disc = r["motor_disc"]
        ok = snr > 0.5 and disc > 0.001 and r["ring_stable"] and not r["saturation"]
        if ok and rec is None: rec = r["scale"]
        tag = ("RECOMMENDED" if (ok and rec == r["scale"]) else
               "SATURATED" if r["saturation"] else
               "UNSTABLE" if not r["ring_stable"] else "LOW_SNR")
        print("  scale=" + str(r["scale"]) + " PEG_SNR=" + str(round(snr, 4)) +
              " disc=" + str(round(disc, 6)) + " stable=" + str(r["ring_stable"]) + " -> " + tag)

    if rec is None:
        rec = 0.01
        print("\n[WARNING] No optimal scale; fallback to " + str(rec))
    else:
        print("\n[GATE PASS] Recommended synapse_scale = " + str(rec))
    with open(RESULTS_DIR / "recommended_scale.txt", "w") as f: f.write(str(rec))

    # ── Plots ──
    pc = {"ER4d":"#E74C3C","EPG":"#3498DB","PEG":"#2ECC71","PEN_a":"#F39C12","PEN_b":"#9B59B6"}
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("FlyMind Phase 5 -- Synapse Scale Audit", fontsize=13, fontweight="bold")

    ax = axes[0, 0]
    for p in ["ER4d","EPG","PEG","PEN_a","PEN_b"]:
        vals = [r["pop_results"].get(p, {}).get("stim_mean", 0) for r in sd]
        ax.plot(SCALES, vals, "o-", label=p, color=pc[p], lw=2, ms=6)
    ax.set_xscale("log"); ax.set_xlabel("synapse_scale"); ax.set_ylabel("Mean FR (stim)")
    ax.set_title("Post-Stimulus Firing Rate"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for p in ["PEG","PEN_a","PEN_b"]:
        snrs = [r["pop_results"].get(p, {}).get("snr", 0) for r in sd]
        ax.plot(SCALES, snrs, "s-", label=p, color=pc[p], lw=2, ms=7)
    ax.axhline(0.5, color="red", ls="--", lw=1.2, label="SNR=0.5")
    ax.set_xscale("log"); ax.set_xlabel("synapse_scale"); ax.set_ylabel("SNR")
    ax.set_title("Motor Population SNR"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    discs = [r["motor_disc"] for r in sd]
    bcols = ["#2ECC71" if r["ring_stable"] and not r["saturation"] else "#E74C3C" for r in sd]
    ax.bar([str(s) for s in SCALES], discs, color=bcols, alpha=0.85, edgecolor="k", lw=0.7)
    ax.axhline(0.001, color="red", ls="--", lw=1.2, label="min disc")
    ax.set_xlabel("synapse_scale"); ax.set_ylabel("max-min motor FR")
    ax.set_title("Motor Discrimination (green=stable, red=sat/unstable)")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    if rec is not None:
        rs = str(rec)
        sl = [str(s) for s in SCALES]
        if rs in sl:
            ri = sl.index(rs)
            ax.bar([rs], [discs[ri]], color="gold", edgecolor="k", lw=1.5, label="recommended", zorder=5)
            ax.legend(fontsize=9)

    ax = axes[1, 1]
    bm_ = [r["bump_amp_mean"] for r in sd]; bs_ = [r["bump_amp_std"] for r in sd]
    ax.errorbar(SCALES, bm_, yerr=bs_, fmt="D-", color="#3498DB", lw=2, ms=7, capsize=5)
    ax.axhline(0.01, color="orange", ls="--", lw=1.2, label="min bump")
    for r, v in zip(sd, bm_):
        ax.annotate("ok" if r["ring_stable"] else "X", xy=(r["scale"], v),
                    xytext=(0, 10), textcoords="offset points", ha="center", fontsize=10,
                    color="#2ECC71" if r["ring_stable"] else "#E74C3C")
    ax.set_xscale("log"); ax.set_xlabel("synapse_scale"); ax.set_ylabel("EPG Bump Amplitude")
    ax.set_title("Ring Attractor Stability"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "synapse_scale_audit.png", dpi=150); plt.close(fig)
    print("[SAVED] synapse_scale_audit.png")

    fig2, axes2 = plt.subplots(2, 4, figsize=(18, 8))
    fig2.suptitle("EPG Bump Position Tracking vs Stimulus Rotation", fontsize=12, fontweight="bold")
    for ai, (r, ax2) in enumerate(zip(sd, axes2.flat)):
        ax2.plot(r["bump_pos"], color="#3498DB", lw=1.5, label="actual")
        exp = np.linspace(-np.pi, np.pi, len(r["bump_pos"]))
        ax2.plot(exp, color="red", ls="--", lw=1, alpha=0.5, label="expected")
        ax2.set_title("scale=" + str(r["scale"]) + "\nbump=" + str(round(r["bump_amp_mean"], 3)) +
                      " stable=" + str(r["ring_stable"]), fontsize=9)
        ax2.set_ylim(-np.pi - 0.5, np.pi + 0.5); ax2.grid(True, alpha=0.3)
        if ai == 0: ax2.legend(fontsize=8)
    for ax2 in axes2.flat[len(sd):]: ax2.set_visible(False)
    fig2.tight_layout()
    fig2.savefig(RESULTS_DIR / "ring_attractor_stability.png", dpi=150); plt.close(fig2)
    print("[SAVED] ring_attractor_stability.png")
    print("\nSteps 2-3-6 COMPLETE. Recommended scale: " + str(rec))
    return rec

if __name__ == "__main__":
    main()
