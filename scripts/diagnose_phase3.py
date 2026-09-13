"""
Phase 3 Diagnostics:
Experiment 1: Environment Audit
Experiment 2: Sensor Receptive Field Audit & Polar Map
Experiment 3: Motor Output Mapping Audit
Experiment 4: 1D Heading Alignment Learning Task
"""

import sys
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from flymind.connectome.loader import ConnectomeLoader
from flymind.brain.network import NeuralNetwork
from flymind.brain.neuron import RateNeuron
from flymind.environment.world import VirtualArena, ArenaState
from flymind.environment.sensors import SectorVisualSensor
from flymind.agent.plastic_cx import PlasticCXAgent

GRAPH_PATH = ROOT / "data" / "processed" / "cx_heading_v1.json"
DIAG_DIR = ROOT / "results" / "phase3"
DIAG_DIR.mkdir(parents=True, exist_ok=True)


def audit_environment():
    """Experiment 1: Environment Audit - Check determinism and spatial sanity."""
    print("--- [Experiment 1: Environment Audit] ---")
    arena = VirtualArena(width=100.0, height=100.0, min_start_target_dist=25.0)
    
    # 1. Determinism test
    s1 = arena.reset(seed=42)
    s2 = arena.reset(seed=42)
    s3 = arena.reset(seed=43)
    assert np.allclose(s1.agent_pos, s2.agent_pos)
    assert np.isclose(s1.agent_heading, s2.agent_heading)
    assert not np.allclose(s1.agent_pos, s3.agent_pos)
    print("  [PASS] Seed determinism verified (identical seed -> identical initial state).")

    # 2. Visualize 20 randomized arena environments
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_title("Environment Audit: 20 Randomized Start/Target Pairs", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)

    sensor = SectorVisualSensor(fov_radians=2*np.pi) # Check visibility
    for seed in range(20):
        st = arena.reset(seed=seed * 10 + 5)
        dist = np.linalg.norm(st.agent_pos - st.target_pos)
        assert dist >= 25.0
        
        # Plot start
        ax.plot(st.agent_pos[0], st.agent_pos[1], "o", color="#27AE60", markersize=6)
        # Plot heading vector
        dx = 5.0 * np.cos(st.agent_heading)
        dy = 5.0 * np.sin(st.agent_heading)
        ax.arrow(st.agent_pos[0], st.agent_pos[1], dx, dy, head_width=2, color="#2980B9", alpha=0.7)
        # Plot target
        ax.plot(st.target_pos[0], st.target_pos[1], "*", color="#E67E22", markersize=8)

    ax.plot([], [], "o", color="#27AE60", label="Start Pos")
    ax.plot([], [], "-", color="#2980B9", label="Heading Dir")
    ax.plot([], [], "*", color="#E67E22", label="Target Beacon")
    ax.legend(loc="upper right")

    fig.tight_layout()
    env_plot = DIAG_DIR / "01_environment_spatial_audit.png"
    fig.savefig(env_plot, dpi=150)
    plt.close(fig)
    print(f"  [SAVED] Environment audit plot saved to: {env_plot}")


def audit_sensors():
    """Experiment 2: Sensor Receptive Field Audit & Polar Map."""
    print("\n--- [Experiment 2: Sensor Sanity Audit] ---")
    sensor = SectorVisualSensor(fov_radians=np.pi) # standard 180 deg FOV
    angles = np.linspace(-np.pi, np.pi, 360)
    left_acts, center_acts, right_acts = [], [], []

    for ang in angles:
        # Fly at (50, 50) heading 0 (pointing East), target at distance 30 at relative angle ang
        agent_pos = np.array([50.0, 50.0])
        target_pos = np.array([50.0 + 30.0 * np.cos(ang), 50.0 + 30.0 * np.sin(ang)])
        st = ArenaState(agent_pos=agent_pos, agent_heading=0.0, target_pos=target_pos, step_count=0, target_reached=False, collision_occurred=False)
        acts = sensor.sense(st)
        left_acts.append(acts[0])
        center_acts.append(acts[1])
        right_acts.append(acts[2])

    left_acts = np.array(left_acts)
    center_acts = np.array(center_acts)
    right_acts = np.array(right_acts)

    fig, ax = plt.subplots(figsize=(9, 5))
    deg = np.degrees(angles)
    ax.plot(deg, left_acts, color="#3498DB", linewidth=2.0, label="Left Sector")
    ax.plot(deg, center_acts, color="#2ECC71", linewidth=2.0, label="Center Sector")
    ax.plot(deg, right_acts, color="#E74C3C", linewidth=2.0, label="Right Sector")
    ax.set_title("Sensory Encoding: Receptive Field Activation vs Target Relative Bearing", fontsize=11, fontweight="bold")
    ax.set_xlabel("Relative Target Bearing (°)")
    ax.set_ylabel("Normalized Sensory Drive")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    sensor_plot = DIAG_DIR / "02_sensor_receptive_fields.png"
    fig.savefig(sensor_plot, dpi=150)
    plt.close(fig)
    print(f"  [SAVED] Sensor tuning curve saved to: {sensor_plot}")


def audit_motor():
    """Experiment 3: Motor Output Mapping Audit."""
    print("\n--- [Experiment 3: Motor Decoder Audit] ---")
    graph = ConnectomeLoader.load_from_json(GRAPH_PATH)
    net = NeuralNetwork(graph=graph, neuron_model_cls=RateNeuron, synapse_scale=0.001)
    agent = PlasticCXAgent(network=net)

    # Test what actions are decoded under artificial population drive
    results = {}
    pop_names = ["Left Steering (PEN_a)", "Forward (PEG)", "Right Steering (PEN_b)"]
    
    for i, pop in enumerate(agent.motor_sectors):
        # Reset and inject high current only into this motor population
        net.reset()
        current = np.zeros(net.num_neurons)
        current[pop] = 5.0
        act = net.step(current)
        scores = [float(np.mean(act[p])) for p in agent.motor_sectors]
        action = int(np.argmax(scores)) + 1
        results[pop_names[i]] = {"scores": scores, "action": action}
        print(f"  Drive to {pop_names[i]:<25} -> Action: {action} (Scores: {[round(s, 2) for s in scores]})")


def run_1d_heading_task():
    """Experiment 4: 1D Heading Alignment Learning Task."""
    print("\n--- [Experiment 4: 1D Heading Alignment Learning Task] ---")
    graph = ConnectomeLoader.load_from_json(GRAPH_PATH)
    net = NeuralNetwork(graph=graph, neuron_model_cls=RateNeuron, synapse_scale=0.001, dt=1.0)
    agent = PlasticCXAgent(network=net, learning_rate=0.005, eligibility_decay=0.8)
    sensor = SectorVisualSensor(fov_radians=2*np.pi) # full view for 1D task

    # 100 trials where fly is fixed at (50,50) and target is at relative angle (-90, -45, 0, 45, 90)
    n_episodes = 150
    heading_errors = []

    for ep in range(n_episodes):
        agent.reset()
        # Random initial target bearing in [-pi, pi]
        target_bearing = np.random.uniform(-np.pi, np.pi)
        agent_heading = 0.0
        target_pos = np.array([50.0 + 30.0 * np.cos(target_bearing), 50.0 + 30.0 * np.sin(target_bearing)])
        
        ep_errors = []
        for step in range(30):
            st = ArenaState(agent_pos=np.array([50.0, 50.0]), agent_heading=agent_heading, target_pos=target_pos, step_count=step, target_reached=False, collision_occurred=False)
            curr_error = abs((target_bearing - agent_heading + np.pi) % (2*np.pi) - np.pi)
            ep_errors.append(np.degrees(curr_error))

            action = agent.act(st)
            # Execute turn in place
            if action == 2: # Turn Left
                agent_heading = (agent_heading + np.pi/16.0) % (2*np.pi)
            elif action == 3: # Turn Right
                agent_heading = (agent_heading - np.pi/16.0) % (2*np.pi)

            new_error = abs((target_bearing - agent_heading + np.pi) % (2*np.pi) - np.pi)
            reward = float(curr_error - new_error) * 5.0 # reward reduction in angular error
            agent.apply_reward(reward)

        heading_errors.append(ep_errors[-1])

    fig, ax = plt.subplots(figsize=(8, 5))
    smooth_err = np.convolve(heading_errors, np.ones(10)/10, mode="valid")
    ax.plot(smooth_err, color="#8E44AD", linewidth=2.0)
    ax.set_title("1D Heading Alignment Learning (Error vs Episode)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Final Heading Error (°)")
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    h_plot = DIAG_DIR / "03_1d_heading_learning.png"
    fig.savefig(h_plot, dpi=150)
    plt.close(fig)
    print(f"  [SAVED] 1D heading learning plot saved to: {h_plot}")


def main():
    audit_environment()
    audit_sensors()
    audit_motor()
    run_1d_heading_task()


if __name__ == "__main__":
    main()
