"""Test velocity-aware decoder over full episode."""
import sys
sys.path.insert(0, "src")
import numpy as np
from flymind.environment.flappy import FlappyEnvironment
from flymind.agent.flappy_agent import FlappyConnectomeAgent
from flymind.brain.network import NeuralNetwork
from flymind.connectome.loader import ConnectomeLoader

graph = ConnectomeLoader.load_from_json("data/processed/cx_heading_v1.json")
net = NeuralNetwork(graph, synapse_scale=0.005)
agent = FlappyConnectomeAgent(net, enable_plasticity=False, sensory_drive=25.0)
env = FlappyEnvironment(seed=42, max_steps=2000)
state = env.reset(seed=42)
agent.reset()
for step in range(500):
    action = agent.act(state)
    state, reward, done, info = env.step(action)
    if step % 50 == 0 or done:
        diag = agent.get_motor_diagnostics()
        print(f"step={step:3d} action={action} bird_y={state.bird_y:.0f} vy={state.bird_vy:.2f} score={info['score']} prob={diag['flap_prob']:.3f}")
    if done:
        break
print(f"Final: score={info['score']} steps={info['step']}")
