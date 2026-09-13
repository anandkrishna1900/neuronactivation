"""Diagnose EPG peak-based decoder during flight."""
import sys
sys.path.insert(0, "src")
from flymind.environment.flappy import FlappyEnvironment
from flymind.agent.flappy_agent import FlappyConnectomeAgent
from flymind.brain.network import NeuralNetwork
from flymind.connectome.loader import ConnectomeLoader

graph = ConnectomeLoader.load_from_json("data/processed/cx_heading_v1.json")
net = NeuralNetwork(graph, synapse_scale=0.005)
agent = FlappyConnectomeAgent(net, enable_plasticity=False, sensory_drive=25.0)
env = FlappyEnvironment(seed=42)
state = env.reset(seed=42)
agent.reset()
for step in range(60):
    action = agent.act(state)
    diag = agent.get_motor_diagnostics()
    nearest = None
    for p in state.pipes:
        if p["x"] > 80.0:
            if nearest is None or p["x"] < nearest["x"]:
                nearest = p
    gc = nearest["gap_center"] if nearest else 0
    print(f"step={step:2d} action={action} peak={diag['epg_peak']:2d} vert={diag['vertical']:+.3f} prob={diag['flap_prob']:.4f} bird_y={state.bird_y:.0f} gap={gc:.0f}")
    state, reward, done, info = env.step(action)
    if done:
        break
print(f"Final: score={info['score']} steps={info['step']}")
