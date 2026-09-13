"""
Phase 9: Multi-episode training with Reward-Modulated Hebbian Plasticity.
Updates sensory-to-compass and compass-to-motor weights based on proximity reward.
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
from flymind.brain.plasticity import RewardModulatedHebbian
from flymind.environment.world import VirtualArena
from flymind.environment.sensors import SectorVisualSensor
from flymind.agent.fly import BaseAgent

GRAPH_PATH = ROOT / "data" / "processed" / "cx_heading_v1.json"
OUT_DIR = ROOT / "results" / "experiments"
OUT_DIR.mkdir(parents=True, exist_ok=True)


class PlasticConnectomeFlyAgent(BaseAgent):
    """
    Connectome-driven agent with biologically constrained reward-modulated Hebbian plasticity.
    Only allows plasticity on specific sensory (ER4d) -> compass (EPG) or motor (PEG) pathways.
    """

    def __init__(
        self,
        network: NeuralNetwork,
        sensor: SectorVisualSensor,
        input_sectors: list,
        motor_sectors: list,
        learning_rate: float = 0.005,
    ):
        self.network = network
        self.sensor = sensor
        self.input_sectors = input_sectors
        self.motor_sectors = motor_sectors
        self.plasticity = RewardModulatedHebbian(learning_rate=learning_rate)
        
        # Track previous pre/post activations for plasticity update
        self.prev_activity = np.zeros(network.num_neurons, dtype=np.float64)
        self.current_activity = np.zeros(network.num_neurons, dtype=np.float64)

        # Create plasticity mask (only allow plastic updates on non-zero existing synapses)
        self.plasticity_mask = (self.network.synapses.raw_weights > 0).astype(np.float64)

    def act(self, state) -> int:
        visual_signals = self.sensor.sense(state)  # [left, center, right]
        ext_current = np.zeros(self.network.num_neurons, dtype=np.float64)

        for sec_idx, neuron_grp in enumerate(self.input_sectors):
            for nid_idx in neuron_grp:
                ext_current[nid_idx] = visual_signals[sec_idx] * 2.0

        self.prev_activity = self.current_activity.copy()
        self.current_activity = self.network.step(ext_current)

        # Compute motor output
        action_scores = []
        for m_grp in self.motor_sectors:
            if len(m_grp) > 0:
                action_scores.append(float(np.mean(self.current_activity[m_grp])))
            else:
                action_scores.append(0.0)

        # 0 -> Forward (1), 1 -> Turn Left (2), 2 -> Turn Right (3)
        choice = int(np.argmax(action_scores))
        return choice + 1

    def apply_reward(self, reward: float) -> None:
        """Apply 3-factor Hebbian update modulated by dopamine/reward."""
        if np.sum(self.prev_activity) > 0:
            dW = self.plasticity.learning_rate * reward * np.outer(self.prev_activity, self.current_activity)
            # Enforce structural mask: do not create nonexistent anatomical connections
            dW = dW * self.plasticity_mask
            
            # Update raw weights with lower bound 0
            new_weights = np.maximum(0.0, self.network.synapses.raw_weights + dW)
            self.network.synapses.raw_weights = new_weights
            self.network.synapses.effective_weights = (
                self.network.synapses.scale * self.network.synapses.raw_weights * self.network.synapses.signs[:, np.newaxis]
            )

    def reset(self) -> None:
        self.network.reset()
        self.prev_activity = np.zeros(self.network.num_neurons, dtype=np.float64)
        self.current_activity = np.zeros(self.network.num_neurons, dtype=np.float64)


def train_plastic_agent(num_episodes=100, max_steps_per_ep=300):
    graph = ConnectomeLoader.load_from_json(GRAPH_PATH)
    net = NeuralNetwork(
        graph=graph,
        neuron_model_cls=RateNeuron,
        synapse_scale=0.001,
        dt=1.0,
    )

    er4d_indices = [net.id_to_idx[nid] for nid in net.neuron_ids
                    if graph.nx_graph.nodes[nid].get("cell_type") in ("ER4d", "ER4m")]
    peg_indices = [net.id_to_idx[nid] for nid in net.neuron_ids
                   if graph.nx_graph.nodes[nid].get("cell_type") in ("PEG", "PEN_a(PEN1)", "PEN_b(PEN2)")]

    n_in = len(er4d_indices)
    s1, s2 = n_in // 3, 2 * n_in // 3
    input_sectors = [er4d_indices[:s1], er4d_indices[s1:s2], er4d_indices[s2:]]

    n_out = len(peg_indices)
    m1, m2 = n_out // 3, 2 * n_out // 3
    motor_sectors = [peg_indices[m1:m2], peg_indices[:m1], peg_indices[m2:]]

    sensor = SectorVisualSensor()
    agent = PlasticConnectomeFlyAgent(
        network=net,
        sensor=sensor,
        input_sectors=input_sectors,
        motor_sectors=motor_sectors,
        learning_rate=0.002,
    )
    arena = VirtualArena(width=100.0, height=100.0, target_radius=6.0, step_size=2.0, max_steps=max_steps_per_ep)

    episode_rewards = []
    success_history = []
    steps_history = []

    print(f"Training Plastic Connectome Fly over {num_episodes} episodes...")

    for ep in range(num_episodes):
        state = arena.reset(seed=ep * 7 + 13)
        agent.reset()
        prev_dist = np.linalg.norm(state.agent_pos - state.target_pos)
        
        ep_reward = 0.0
        done = False

        while not done:
            action = agent.act(state)
            state, base_reward, done, info = arena.step(action)
            
            # Dense proximity shaping + target arrival reward
            curr_dist = info["distance"]
            dist_delta = prev_dist - curr_dist  # positive if moved closer
            prev_dist = curr_dist
            
            # Step reward signal
            step_reward = (dist_delta * 2.0) + (50.0 if state.target_reached else -0.05)
            agent.apply_reward(step_reward)
            
            ep_reward += step_reward

        episode_rewards.append(ep_reward)
        success_history.append(1 if state.target_reached else 0)
        steps_history.append(state.step_count)

        if (ep + 1) % 20 == 0 or ep == 0:
            recent_success = np.mean(success_history[-20:]) * 100 if ep >= 19 else np.mean(success_history) * 100
            print(f"  Episode {ep+1:3d}/{num_episodes} | Avg Reward: {np.mean(episode_rewards[-20:]):.2f} | Success (last 20): {recent_success:.1f}%")

    return episode_rewards, success_history, steps_history


def main():
    print("=" * 60)
    print("FlyMind Phase 9: Reward-Modulated Plasticity Training")
    print("=" * 60)

    num_episodes = 100
    rewards, successes, steps = train_plastic_agent(num_episodes=num_episodes)

    # Plot learning curves
    window = 10
    success_smooth = np.convolve(successes, np.ones(window)/window, mode='valid') * 100
    reward_smooth = np.convolve(rewards, np.ones(window)/window, mode='valid')

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    ax1.plot(range(window, num_episodes + 1), success_smooth, color="#2ECC71", linewidth=2.0)
    ax1.set_ylabel("Success Rate (%)", fontsize=10)
    ax1.set_title("Plastic Connectome Fly — Learning Progression\n(Reward-Modulated Hebbian Plasticity on Real Hemibrain Circuit)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.plot(range(window, num_episodes + 1), reward_smooth, color="#3498DB", linewidth=2.0)
    ax2.set_xlabel("Episode", fontsize=10)
    ax2.set_ylabel("Smoothed Return", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    plot_path = OUT_DIR / "07_plasticity_learning_curve.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    print("\n" + "=" * 60)
    print(f"Training complete. Learning curve saved to: {plot_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
