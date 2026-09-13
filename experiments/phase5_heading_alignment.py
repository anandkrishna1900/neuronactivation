
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from flymind.connectome.loader import ConnectomeLoader
from flymind.connectome.graph import ConnectomeGraph, NeuronMetadata, SynapticConnection
from flymind.brain.network import NeuralNetwork
from flymind.brain.neuron import RateNeuron
from flymind.agent.plastic_cx import PlasticCXAgent
from flymind.agent.fly import RandomFlyAgent
from flymind.environment.sensors import PanoramicCompoundEyeSensor

GRAPH_PATH  = ROOT / "data" / "processed" / "cx_heading_v1.json"
RESULTS_DIR = ROOT / "results" / "phase5"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NUM_TRAIN_SEEDS = 10
NUM_EVAL        = 50
EVAL_SEEDS      = list(range(92000, 92000 + NUM_EVAL))
MAX_TURNS       = 64   # max steps to align heading
TURN_ANG        = np.pi / 8.0  # 22.5 deg per step
SUCCESS_THRESH  = 0.2  # radians (~11.5 deg)

def get_scale_and_temp():
    sp = RESULTS_DIR / "recommended_scale.txt"
    tp = RESULTS_DIR / "recommended_temperature.txt"
    sc = float(sp.read_text()) if sp.exists() else 0.005
    T  = float(tp.read_text()) if tp.exists() else 0.2
    return sc, T

def create_rewired(bio, n_swaps=5000, seed=42):
    nx_g = bio.nx_graph.copy()
    try: rw = nx.directed_edge_swap(nx_g, nswap=n_swaps, max_tries=n_swaps*10, seed=seed)
    except: rw = nx_g
    rg = ConnectomeGraph(name="rewired")
    for nid, d in bio.nx_graph.nodes(data=True):
        rg.add_neuron(NeuronMetadata(body_id=nid, cell_type=d.get("cell_type"),
            instance=d.get("instance"), roi=d.get("roi"), neurotransmitter=d.get("neurotransmitter")))
    for u, v, d in rw.edges(data=True):
        rg.add_connection(SynapticConnection(source_id=u, target_id=v,
            weight=d.get("weight", 1.0), neurotransmitter=d.get("neurotransmitter")))
    return rg

class HeadingArena:
    """Minimal 1D heading task. Agent must align heading to target bearing."""
    def __init__(self, max_steps=MAX_TURNS):
        self.max_steps = max_steps; self.rng = np.random.default_rng()

    def reset(self, seed=None):
        if seed is not None: self.rng = np.random.default_rng(seed)
        self.heading = float(self.rng.uniform(0, 2*np.pi))
        self.target  = float(self.rng.uniform(0, 2*np.pi))
        self.steps   = 0
        self.init_err = abs((self.heading - self.target + np.pi) % (2*np.pi) - np.pi)
        return self._state()

    def _state(self):
        from flymind.environment.world import ArenaState
        # Encode heading as a virtual position/bearing for sensor
        agent_pos  = np.array([50.0, 50.0])
        target_ang = self.target
        target_pos = np.array([50.0 + 30*np.cos(target_ang), 50.0 + 30*np.sin(target_ang)])
        return ArenaState(agent_pos=agent_pos, agent_heading=self.heading,
                          target_pos=target_pos, step_count=self.steps,
                          target_reached=False, collision_occurred=False)

    def step(self, action):
        self.steps += 1
        if action == 2: self.heading = (self.heading + TURN_ANG) % (2*np.pi)
        elif action == 3: self.heading = (self.heading - TURN_ANG) % (2*np.pi)
        err = abs((self.heading - self.target + np.pi) % (2*np.pi) - np.pi)
        done = err <= SUCCESS_THRESH or self.steps >= self.max_steps
        return self._state(), 0.0, done, {"error": err, "success": err <= SUCCESS_THRESH, "steps": self.steps}

class HeadingReward:
    def compute_reward(self, info, done):
        return -info["error"] / np.pi + (1.0 if info["success"] else 0.0)

def run_heading_episode(agent, arena, reward_fn, seed, train=True):
    state = arena.reset(seed=seed)
    agent.reset(); done = False; total_r = 0.0
    init_err = arena.init_err; errs = [init_err]
    while not done:
        action = agent.act(state)
        state, _, done, info = arena.step(action)
        r = reward_fn.compute_reward(info, done); total_r += r
        errs.append(info["error"])
        if train and hasattr(agent, "apply_reward"): agent.apply_reward(r)
    return dict(success=int(info["success"]), steps=info["steps"],
                init_err=init_err, final_err=info["error"],
                min_err=float(np.min(errs)), total_reward=total_r)

def run_condition(label, agent_fn, bio_or_rewired, scale, T, arena, reward_fn):
    succs, final_errs, steps_list, init_errs = [], [], [], []
    for si in range(NUM_TRAIN_SEEDS):
        net   = NeuralNetwork(graph=bio_or_rewired, neuron_model_cls=RateNeuron, synapse_scale=scale)
        agent = agent_fn(net, T)
        for ep in range(30):
            run_heading_episode(agent, arena, reward_fn, seed=si*1000+42+ep, train=True)
        if hasattr(agent, "freeze_weights"): agent.freeze_weights()
        for es in EVAL_SEEDS:
            rec = run_heading_episode(agent, arena, reward_fn, seed=es, train=False)
            succs.append(rec["success"]); final_errs.append(rec["final_err"])
            steps_list.append(rec["steps"]); init_errs.append(rec["init_err"])
    return dict(label=label,
                success_rate=float(np.mean(succs))*100,
                mean_final_err_rad=float(np.mean(final_errs)),
                mean_init_err_rad=float(np.mean(init_errs)),
                improvement_rad=float(np.mean(init_errs))-float(np.mean(final_errs)),
                mean_steps=float(np.mean(steps_list)))

def main():
    scale, T = get_scale_and_temp()
    print("Phase 5 Step 8: 1D Heading Alignment")
    print("scale=" + str(scale) + " T=" + str(T))
    bio     = ConnectomeLoader.load_from_json(GRAPH_PATH)
    rewired = create_rewired(bio)
    arena   = HeadingArena()
    reward_fn = HeadingReward()
    sensor = PanoramicCompoundEyeSensor()

    conditions = [
        ("A: Bio + No Plasticity",
         lambda net, T_: PlasticCXAgent(net, sensor, plasticity_mode="none",
                                        decoder_mode="hemispheric", temperature=T_), bio),
        ("B: Bio + Pathway Plasticity",
         lambda net, T_: PlasticCXAgent(net, sensor, plasticity_mode="pathway",
                                        decoder_mode="hemispheric", temperature=T_), bio),
        ("C: Rewired + Pathway Plasticity",
         lambda net, T_: PlasticCXAgent(net, sensor, plasticity_mode="pathway",
                                        decoder_mode="hemispheric", temperature=T_), rewired),
    ]
    rows = []
    for label, agent_fn, graph in conditions:
        print("  " + label + " ...", end=" ", flush=True)
        r = run_condition(label, agent_fn, graph, scale, T, arena, reward_fn)
        rows.append(r)
        print("succ=" + str(round(r["success_rate"],1)) + "% impr=" + str(round(r["improvement_rad"],3)) + " rad")

    # Random baseline
    print("  D: Random Baseline ...", end=" ", flush=True)
    succs, ferrs, ierrs = [], [], []
    rng_agent = RandomFlyAgent(seed=42)
    for si in range(NUM_TRAIN_SEEDS):
        for es in EVAL_SEEDS:
            a2 = HeadingArena()
            state = a2.reset(seed=es); rng_agent.reset(); done = False
            init_e = a2.init_err
            while not done:
                act = rng_agent.act(state); state, _, done, info = a2.step(act)
            succs.append(info["success"]); ferrs.append(info["error"]); ierrs.append(init_e)
    rows.append(dict(label="D: Random Baseline",
                     success_rate=float(np.mean(succs))*100,
                     mean_final_err_rad=float(np.mean(ferrs)),
                     mean_init_err_rad=float(np.mean(ierrs)),
                     improvement_rad=float(np.mean(ierrs))-float(np.mean(ferrs)),
                     mean_steps=MAX_TURNS/2))
    print("succ=" + str(round(rows[-1]["success_rate"],1)) + "%")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "heading_alignment.csv", index=False)
    print("[SAVED] heading_alignment.csv")
    print(df[["label","success_rate","improvement_rad","mean_final_err_rad"]].to_string(index=False))

    # Gate check
    rand_succ = rows[-1]["success_rate"]
    best_bio  = max(rows[:3], key=lambda x: x["success_rate"])["success_rate"]
    print("\nGATE: best_bio_success=" + str(round(best_bio,1)) + "% random=" + str(round(rand_succ,1)) + "%")
    if best_bio <= rand_succ:
        print("[GATE FAIL] No connectome condition beats the random baseline on heading alignment.")
        print("  -> DO NOT PROCEED TO 2D NAVIGATION.")
        with open(RESULTS_DIR/"heading_gate.txt","w") as f: f.write("FAIL")
    else:
        print("[GATE PASS] Connectome surpasses random baseline. Proceeding to 2D navigation.")
        with open(RESULTS_DIR/"heading_gate.txt","w") as f: f.write("PASS")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("FlyMind Phase 5 -- 1D Heading Alignment", fontsize=12, fontweight="bold")
    labels = [r["label"] for r in rows]
    srs    = [r["success_rate"] for r in rows]
    impr   = [r["improvement_rad"] for r in rows]
    colors = ["#E74C3C" if "Random" in l or "Rewired" in l else "#3498DB" for l in labels]
    x = np.arange(len(labels))
    ax1.bar(x, srs, color=colors, alpha=0.85, edgecolor="k", lw=0.7)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax1.set_ylabel("Success Rate (%)"); ax1.set_title("Heading Alignment Success")
    ax1.grid(axis="y", alpha=0.3)
    ax2.bar(x, impr, color=colors, alpha=0.85, edgecolor="k", lw=0.7)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax2.set_ylabel("Angular Error Improvement (rad)"); ax2.set_title("Mean Heading Error Reduction")
    ax2.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "heading_alignment.png", dpi=150); plt.close(fig)
    print("[SAVED] heading_alignment.png")
    print("Step 8 COMPLETE.")

if __name__ == "__main__":
    main()
