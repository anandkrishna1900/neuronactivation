"""Debug: test sensor-based decoder in flight."""
import sys
sys.path.insert(0, "src")
import numpy as np
from flymind.environment.flappy import FlappyState, FlappyEnvironment
from flymind.agent.flappy_agent import FlappyConnectomeAgent
from flymind.brain.network import NeuralNetwork
from flymind.connectome.loader import ConnectomeLoader

graph = ConnectomeLoader.load_from_json("data/processed/cx_heading_v1.json")
net = NeuralNetwork(graph, synapse_scale=0.005)
agent = FlappyConnectomeAgent(net, enable_plasticity=False, sensory_drive=25.0)

env = FlappyEnvironment(seed=42)
state = env.reset(seed=42)
agent.reset()
for step in range(40):
    action = agent.act(state)
    diag = agent.get_motor_diagnostics()
    print(f"step={step:2d} action={action} vsig={diag['vertical_signal']:+.3f} prob={diag['flap_prob']:.4f} bird_y={state.bird_y:.1f} vy={state.bird_vy:.2f}")
    state, reward, done, info = env.step(action)
    if done:
        break
print(f"Final: score={info['score']} steps={info['step']}")
