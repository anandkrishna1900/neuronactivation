"""Parameter sweep for Phase 6B motor readout."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from flymind.connectome.loader import ConnectomeLoader
from flymind.brain.network import NeuralNetwork
from flymind.environment.flappy import FlappyEnvironment
from flymind.agent.flappy_agent_phase6b import ConnectomeMotorReadoutAgent

graph = ConnectomeLoader.load_from_json("data/processed/cx_heading_v1.json")

best = (0, 0, 0, 0, 0)
all_results = []

for temp in [0.5, 1.0, 2.0]:
    for bias in [-1.0, -1.5, -2.0, -2.5, -3.0, -3.5]:
        for lr in [0.005, 0.01, 0.02]:
            scores = []
            survivals = []
            for ep in range(20):
                network = NeuralNetwork(graph, synapse_scale=0.005)
                agent = ConnectomeMotorReadoutAgent(
                    network, enable_plasticity=False, sub_steps=10,
                    motor_temperature=temp, motor_bias=bias, motor_readout_lr=lr,
                    seed=ep,
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
                best = (ms, temp, bias, lr, msurv)
            if ms > 0 or msurv > 200:
                all_results.append((ms, temp, bias, lr, msurv))

all_results.sort(key=lambda x: -x[0])
print("Top 10 parameter combinations:")
for ms, temp, bias, lr, msurv in all_results[:10]:
    print("  temp=%.1f bias=%4.1f lr=%.3f: score=%.2f surv=%.0f" % (temp, bias, lr, ms, msurv))

print()
print("BEST: score=%.2f temp=%.1f bias=%.1f lr=%.3f surv=%.0f" % best)
