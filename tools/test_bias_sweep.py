"""Test different motor biases to find good starting point."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from flymind.connectome.loader import ConnectomeLoader
from flymind.brain.network import NeuralNetwork
from flymind.environment.flappy import FlappyEnvironment
from flymind.agent.flappy_agent_phase6b import ConnectomeMotorReadoutAgent

ROOT = Path(__file__).resolve().parent.parent
CONNECTOME_PATH = ROOT / "data" / "processed" / "cx_heading_v1.json"

graph = ConnectomeLoader.load_from_json(CONNECTOME_PATH)

for bias in [-1.0, -1.5, -2.0, -2.5, -3.0]:
    scores = []
    survivals = []
    flap_rates = []
    for ep in range(20):
        network = NeuralNetwork(graph, synapse_scale=0.005)
        agent = ConnectomeMotorReadoutAgent(
            network, enable_plasticity=False, sub_steps=10,
        )
        agent.motor_readout_bias = bias
        
        env = FlappyEnvironment(seed=42)
        state = env.reset(seed=42 + ep)
        agent.reset()
        done = False
        actions = []
        while not done:
            action = agent.act(state)
            actions.append(action)
            state, reward, done, info = env.step(action)
        scores.append(info["score"])
        survivals.append(info["step"])
        flap_rates.append(np.mean(actions) if actions else 0)
    
    print(f"bias={bias:5.1f}: mean_score={np.mean(scores):.2f}, "
          f"mean_survival={np.mean(survivals):.1f}, "
          f"mean_flap_rate={np.mean(flap_rates):.3f}")
