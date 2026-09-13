"""Debug: compare isolated test vs flight loop with new decoder."""
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

# Isolated tests
print("=== Isolated tests ===")
for gy, gc, label in [(100, 300, "above"), (200, 200, "aligned"), (300, 100, "below")]:
    agent.reset()
    state = FlappyState(bird_y=gy, bird_vy=0, pipes=[{"x":150,"gap_center":gc}], score=0, step_count=0, alive=True)
    action = agent.act(state)
    diag = agent.get_motor_diagnostics()
    print(f"  {label:8s}: action={action} vert={diag['vertical']:+.3f} prob={diag['flap_prob']:.4f}")

# Flight loop test
print("\n=== Flight loop ===")
env = FlappyEnvironment(seed=42)
state = env.reset(seed=42)
agent.reset()
for step in range(20):
    action = agent.act(state)
    diag = agent.get_motor_diagnostics()
    print(f"  step={step:2d} action={action} vert={diag['vertical']:+.3f} prob={diag['flap_prob']:.4f} bird_y={state.bird_y:.1f}")
    state, reward, done, info = env.step(action)
    if done:
        break
print(f"  Final: score={info['score']} steps={info['step']}")
