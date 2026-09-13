"""Debug act() method."""
import sys
sys.path.insert(0, "src")
import numpy as np
from flymind.environment.flappy import FlappyState
from flymind.agent.flappy_agent import FlappyConnectomeAgent
from flymind.brain.network import NeuralNetwork
from flymind.connectome.loader import ConnectomeLoader

graph = ConnectomeLoader.load_from_json("data/processed/cx_heading_v1.json")
net = NeuralNetwork(graph, synapse_scale=0.005)
agent = FlappyConnectomeAgent(net, enable_plasticity=False, sensory_drive=25.0)

state = FlappyState(bird_y=100, bird_vy=0, pipes=[{"x":150,"gap_center":300}], score=0, step_count=0, alive=True)
action = agent.act(state)
diag = agent.get_motor_diagnostics()
print(f"action={action} peak={diag['epg_peak']} vert={diag['vertical']:.3f} prob={diag['flap_prob']:.4f}")
epg_act = agent.current_activity[agent.epg_ordered_indices]
print(f"EPG peak: {np.argmax(epg_act)} range: {epg_act.max()-epg_act.min():.6f}")
print(f"EPG activity: {epg_act.round(4)}")
