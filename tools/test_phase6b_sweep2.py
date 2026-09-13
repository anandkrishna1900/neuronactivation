"""Parameter sweep for Phase 6B with temporal motor readout."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from flymind.connectome.loader import ConnectomeLoader
from flymind.brain.network import NeuralNetwork
from flymind.environment.flappy import FlappyEnvironment
from flymind.agent.flappy_agent_phase6b import ConnectomeMotorReadoutAgent

graph = ConnectomeLoader.load_from_json("data/processed/cx_heading_v1.json")

best = (0, 0, 0, 0)
for gain in [20, 40, 60, 80, 100, 150]:
    for bias in [-0.5, -1.0, -1.5, -2.0, -2.5]:
        scores = []
        survivals = []
        for ep in range(20):
            network = NeuralNetwork(graph, synapse_scale=0.005)
            agent = ConnectomeMotorReadoutAgent(
                network, enable_plasticity=False, sub_steps=10,
                motor_gain=gain, motor_bias=bias, seed=ep,
            )
            env = FlappyEnvironment(seed=42)
            state = env.reset(seed=42 + ep)
            agent.reset()
            done = False
            while not done:
                action = agent.act(state)
                state, reward, done, info = env.step(action)
            scores.append(info["score"])
            survivals.append(info["step"])
        ms = np.mean(scores)
        msurv = np.mean(survivals)
        if ms > best[0]:
            best = (ms, gain, bias, msurv)
        if ms > 0 or msurv > 200:
            print("gain=%3d bias=%4.1f: score=%.2f surv=%.0f" % (gain, bias, ms, msurv))

print()
print("BEST: score=%.2f gain=%d bias=%.1f surv=%.0f" % best)
