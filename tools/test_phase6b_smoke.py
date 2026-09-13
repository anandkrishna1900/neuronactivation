"""Quick smoke test for the Phase 6B motor readout agent."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from flymind.connectome.loader import ConnectomeLoader
from flymind.brain.network import NeuralNetwork
from flymind.environment.flappy import FlappyEnvironment
from flymind.environment.flappy_sensor import FlappyVisualSensor
from flymind.agent.flappy_agent_phase6b import ConnectomeMotorReadoutAgent
from flymind.agent.flappy_agent import HandDesignedFlapAgent

ROOT = Path(__file__).resolve().parent.parent
CONNECTOME_PATH = ROOT / "data" / "processed" / "cx_heading_v1.json"

print("Loading connectome...")
graph = ConnectomeLoader.load_from_json(CONNECTOME_PATH)
print(f"  Neurons: {graph.num_neurons}, Synapses: {graph.num_synapses}")

print("Creating network...")
network = NeuralNetwork(graph, synapse_scale=0.005)
print(f"  Network: {network.num_neurons} neurons")

print("Creating Phase 6B agent...")
agent = ConnectomeMotorReadoutAgent(
    network,
    enable_plasticity=False,
    sub_steps=10,
    sensory_drive=25.0,
)
print(f"  PEG indices: {len(agent.peg_indices)}")
print(f"  PEG_L: {len(agent.peg_L)}, PEG_R: {len(agent.peg_R)}")
print(f"  Motor readout gain: {agent.motor_gain}")
print(f"  Motor readout bias: {agent.motor_bias}")

print("\nRunning 5 episodes without plasticity...")
env = FlappyEnvironment(seed=42)
scores = []
for ep in range(5):
    state = env.reset(seed=42 + ep)
    agent.reset()
    done = False
    while not done:
        action = agent.act(state)
        state, reward, done, info = env.step(action)
    scores.append(info["score"])
    diag = agent.get_motor_diagnostics()
    print(f"  Episode {ep}: score={info['score']}, survival={info['step']}, "
          f"peg_mean={diag['peg_mean']:.4f}, flap_prob={diag['flap_prob']:.3f}")

print(f"\nScores: {scores}")
print(f"Mean: {np.mean(scores):.2f}")

print("\nRunning 3 episodes WITH plasticity...")
agent2 = ConnectomeMotorReadoutAgent(
    NeuralNetwork(graph, synapse_scale=0.005),
    enable_plasticity=True,
    plasticity_mode="pathway",
    sub_steps=10,
    sensory_drive=25.0,
)
scores2 = []
for ep in range(3):
    state = env.reset(seed=42 + ep + 100)
    agent2.reset()
    done = False
    while not done:
        action = agent2.act(state)
        state, reward, done, info = env.step(action)
        agent2.apply_reward(reward)
    scores2.append(info["score"])
    print(f"  Episode {ep}: score={info['score']}, survival={info['step']}")

print(f"Scores with plasticity: {scores2}")
print(f"Mean: {np.mean(scores2):.2f}")
print("\nSmoke test PASSED!")
