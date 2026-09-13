"""Phase 5 Step 4: Motor Decoder Causal Audit."""
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
from flymind.environment.world import VirtualArena, ArenaState

GRAPH_PATH  = ROOT / "data" / "processed" / "cx_heading_v1.json"
RESULTS_DIR = ROOT / "results" / "phase5"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PTYPE_MAP = {
    "ER4":  {"ER4d","ER4m"},
    "EPG":  {"EPG"},
    "PEG":  {"PEG"},
    "PEN_a":{"PEN_a(PEN1)"},
    "PEN_b":{"PEN_b(PEN2)"},
}
SETTLE = 30; PROBE = 30; PULSE = 2.0

def get_pops(net):
    out = {}
    for k, types in PTYPE_MAP.items():
        out[k] = [net.id_to_idx[n] for n in net.neuron_ids
                  if net.graph.nx_graph.nodes[n].get("cell_type") in types]
    return out

def mstats(act, idx):
    if not idx: return 0.0, 0.0
    v = act[idx]; return float(np.mean(v)), float(np.std(v))

def build_ext(net, pops, bearing_rad, n_channels=12, pulse_amp=PULSE):
    ext = np.zeros(net.num_neurons)
    er4 = pops["ER4"]; n_er = len(er4)
    for i in range(n_channels):
        pref = -np.pi + (2*np.pi/n_channels)*i
        ang_err = (bearing_rad - pref + np.pi) % (2*np.pi) - np.pi
        act = float(np.exp(-0.5*(ang_err/(np.pi/6))**2))
        er_idx = int((i/n_channels)*n_er)
        if er_idx < n_er:
            ext[er4[er_idx]] += act * pulse_amp
    return ext

def probe_bearing(net, pops, bearing_rad, condition="normal"):
    net.neurons.reset()
    null = np.zeros(net.num_neurons)
    for _ in range(SETTLE): net.step(null)
    ext_base = build_ext(net, pops, bearing_rad)
    if condition == "normal":
        ext = ext_base
    elif condition == "removed":
        ext = np.zeros(net.num_neurons)
    elif condition == "inverted":
        ext = build_ext(net, pops, bearing_rad + np.pi)
    elif condition == "shuffled":
        ext = ext_base.copy()
        er4 = pops["ER4"]
        vals = ext[er4].copy(); np.random.shuffle(vals); ext[er4] = vals
    else:
        ext = ext_base
    acts = []
    for _ in range(PROBE): act = net.step(ext); acts.append(act)
    peak = acts[-1]
    return {
        "PEG":   mstats(peak, pops["PEG"])[0],
        "PEN_a": mstats(peak, pops["PEN_a"])[0],
        "PEN_b": mstats(peak, pops["PEN_b"])[0],
    }

def main():
    rec_path = RESULTS_DIR / "recommended_scale.txt"
    if rec_path.exists():
        scale = float(rec_path.read_text().strip())
    else:
        scale = 0.01
    print("=" * 60)
    print("Phase 5 Step 4: Motor Decoder Causal Audit")
    print("Using synapse_scale = " + str(scale))
    print("=" * 60)

    bio  = ConnectomeLoader.load_from_json(GRAPH_PATH)
    net  = NeuralNetwork(graph=bio, neuron_model_cls=RateNeuron, synapse_scale=scale)
    pops = get_pops(net)
    np.random.seed(42)

    bearings = [-np.pi, -np.pi/2, 0.0, np.pi/2, np.pi - 0.01]
    bearing_labels = ["-180", "-90", "0", "+90", "+180"]
    conditions = ["normal", "removed", "inverted", "shuffled"]
    rows = []

    # ── Part A: Bearing sweep ──
    print("\nPart A: Motor activity vs target bearing (normal condition)")
    for b, bl in zip(bearings, bearing_labels):
        res = probe_bearing(net, pops, b, "normal")
        action = ["FORWARD","TURN_LEFT","TURN_RIGHT"][int(np.argmax([res["PEG"],res["PEN_a"],res["PEN_b"]]))]
        print("  bearing=" + bl + " deg  PEG=" + str(round(res["PEG"],5)) +
              " PENa=" + str(round(res["PEN_a"],5)) + " PENb=" + str(round(res["PEN_b"],5)) +
              " -> " + action)
        rows.append(dict(bearing_deg=bl, condition="normal", **res,
                         chosen_action=action,
                         motor_discrimination=max(res.values())-min(res.values())))

    # ── Part B: Causal conditions at bearing=0 ──
    print("\nPart B: Causal intervention (bearing=0 deg)")
    for cond in conditions:
        res = probe_bearing(net, pops, 0.0, cond)
        action = ["FORWARD","TURN_LEFT","TURN_RIGHT"][int(np.argmax([res["PEG"],res["PEN_a"],res["PEN_b"]]))]
        print("  " + cond + ": PEG=" + str(round(res["PEG"],5)) +
              " PENa=" + str(round(res["PEN_a"],5)) + " PENb=" + str(round(res["PEN_b"],5)) +
              " -> " + action)
        rows.append(dict(bearing_deg="0", condition=cond, **res,
                         chosen_action=action,
                         motor_discrimination=max(res.values())-min(res.values())))

    df = pd.DataFrame(rows)
    csv_path = RESULTS_DIR / "motor_decoder_audit.csv"
    df.to_csv(csv_path, index=False)
    print("\n[SAVED] " + str(csv_path))

    # Gate check
    normal_rows = df[df["condition"]=="normal"]
    max_disc = float(normal_rows["motor_discrimination"].max())
    print("\nGATE: max motor discrimination across bearings = " + str(round(max_disc,6)))
    if max_disc < 0.001:
        print("[GATE FAIL] Motor populations do not discriminate sensory input.")
        print("  -> Sensory signal is NOT reaching motor populations causally.")
        print("  -> STOPPING Phase 5 here. Diagnose network propagation first.")
        with open(RESULTS_DIR/"gate_result.txt","w") as f:
            f.write("FAIL: motor_disc=" + str(max_disc))
        sys.exit(1)
    else:
        print("[GATE PASS] Sensory input causally modulates motor populations.")
        with open(RESULTS_DIR/"gate_result.txt","w") as f:
            f.write("PASS: motor_disc=" + str(max_disc))

    # ── Plots ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("FlyMind Phase 5 -- Motor Decoder Causal Audit\nscale=" + str(scale),
                 fontsize=12, fontweight="bold")

    nd = normal_rows
    x = np.arange(len(nd))
    w = 0.25
    ax1.bar(x-w, nd["PEG"],   w, label="PEG (Forward)",  color="#2ECC71", alpha=0.85, edgecolor="k", lw=0.7)
    ax1.bar(x,   nd["PEN_a"], w, label="PEN_a (L-turn)", color="#F39C12", alpha=0.85, edgecolor="k", lw=0.7)
    ax1.bar(x+w, nd["PEN_b"], w, label="PEN_b (R-turn)", color="#9B59B6", alpha=0.85, edgecolor="k", lw=0.7)
    ax1.set_xticks(x); ax1.set_xticklabels([bl+" deg" for bl in bearing_labels])
    ax1.set_xlabel("Target Bearing"); ax1.set_ylabel("Mean Firing Rate")
    ax1.set_title("Motor Population Activity vs Target Bearing")
    ax1.legend(fontsize=9); ax1.grid(axis="y", alpha=0.3)

    causal_rows = df[df["bearing_deg"]=="0"]
    y_peg  = causal_rows["PEG"].values
    y_pena = causal_rows["PEN_a"].values
    y_penb = causal_rows["PEN_b"].values
    cx = np.arange(len(conditions))
    ax2.bar(cx-w, y_peg,  w, label="PEG",   color="#2ECC71", alpha=0.85, edgecolor="k", lw=0.7)
    ax2.bar(cx,   y_pena, w, label="PEN_a", color="#F39C12", alpha=0.85, edgecolor="k", lw=0.7)
    ax2.bar(cx+w, y_penb, w, label="PEN_b", color="#9B59B6", alpha=0.85, edgecolor="k", lw=0.7)
    ax2.set_xticks(cx); ax2.set_xticklabels(conditions)
    ax2.set_xlabel("Sensory Condition"); ax2.set_ylabel("Mean Firing Rate")
    ax2.set_title("Causal Intervention: Bearing=0 deg")
    ax2.legend(fontsize=9); ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "motor_decoder_audit.png", dpi=150); plt.close(fig)
    print("[SAVED] motor_decoder_audit.png")
    print("\nStep 4 COMPLETE.")

if __name__ == "__main__":
    main()
