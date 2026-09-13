"""Debug: check what state the flight loop sees."""
import sys
sys.path.insert(0, "src")
import numpy as np
from flymind.environment.flappy import FlappyState, FlappyEnvironment
from flymind.environment.flappy_sensor import FlappyVisualSensor
from flymind.agent.flappy_agent import FlappyConnectomeAgent
from flymind.brain.network import NeuralNetwork
from flymind.connectome.loader import ConnectomeLoader

graph = ConnectomeLoader.load_from_json("data/processed/cx_heading_v1.json")
net = NeuralNetwork(graph, synapse_scale=0.005)
sensor = FlappyVisualSensor()
agent = FlappyConnectomeAgent(net, enable_plasticity=False, sensory_drive=25.0)

env = FlappyEnvironment(seed=42)
state = env.reset(seed=42)

print(f"bird_y={state.bird_y}")
print("Pipes:")
for p in state.pipes:
    dx = p["x"] - 80.0
    print(f"  x={p['x']:.0f} gc={p['gap_center']:.0f} dx={dx:.0f}")

sensor_out = sensor.sense(state)
print(f"Sensor: {sensor_out.round(3)}")

# Now test isolated with same state
agent.reset()
action = agent.act(state)
diag = agent.get_motor_diagnostics()
print(f"Flight act: peak={diag['epg_peak']} prob={diag['flap_prob']:.4f}")

# Compare with isolated test using same bird_y but different pipe
agent.reset()
state2 = FlappyState(bird_y=200, bird_vy=0, pipes=[{"x":150,"gap_center":256}], score=0, step_count=0, alive=True)
sensor_out2 = sensor.sense(state2)
print(f"\nIsolated sensor: {sensor_out2.round(3)}")
action2 = agent.act(state2)
diag2 = agent.get_motor_diagnostics()
print(f"Isolated act: peak={diag2['epg_peak']} prob={diag2['flap_prob']:.4f}")
