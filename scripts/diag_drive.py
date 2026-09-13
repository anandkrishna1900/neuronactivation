"""Test different sensory drive values."""
import sys
sys.path.insert(0, "src")
import numpy as np
from flymind.environment.flappy import FlappyState
from flymind.agent.flappy_agent import FlappyConnectomeAgent
from flymind.brain.network import NeuralNetwork
from flymind.connectome.loader import ConnectomeLoader

graph = ConnectomeLoader.load_from_json("data/processed/cx_heading_v1.json")

for drive in [25, 50, 100, 200, 500]:
    net = NeuralNetwork(graph, synapse_scale=0.005)
    agent = FlappyConnectomeAgent(net, enable_plasticity=False, sensory_drive=drive)
    results = []
    for gy, gc, label in [(100, 300, "above"), (200, 200, "aligned"), (300, 100, "below")]:
        agent.reset()
        state = FlappyState(bird_y=gy, bird_vy=0, pipes=[{"x":150,"gap_center":gc}], score=0, step_count=0, alive=True)
        action = agent.act(state)
        diag = agent.get_motor_diagnostics()
        results.append(f"{label}=peak{diag['epg_peak']}_p{diag['flap_prob']:.3f}")
    print(f"drive={drive:3d}: {' '.join(results)}")
