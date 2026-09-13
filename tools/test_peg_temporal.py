"""Test temporal dynamics of PEG activity during an episode."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from flymind.connectome.loader import ConnectomeLoader
from flymind.brain.network import NeuralNetwork
from flymind.environment.flappy import FlappyEnvironment
from flymind.environment.flappy_sensor import FlappyVisualSensor
from flymind.agent.flappy_agent_phase6b import ConnectomeMotorReadoutAgent

graph = ConnectomeLoader.load_from_json("data/processed/cx_heading_v1.json")
sensor = FlappyVisualSensor()

network = NeuralNetwork(graph, synapse_scale=0.005)
agent = ConnectomeMotorReadoutAgent(network, enable_plasticity=False, sub_steps=10, motor_bias=-3.0)

env = FlappyEnvironment(seed=42)
state = env.reset(seed=42)
agent.reset()

peg_over_time = []
er4_over_time = []
epg_over_time = []
sensor_over_time = []
actions = []

for step in range(200):
    action = agent.act(state)
    peg_over_time.append(agent.current_activity[agent.peg_indices].copy())
    er4_over_time.append(agent.current_activity[agent.er4d_indices].copy())
    epg_over_time.append(agent.current_activity[agent.epg_indices].copy())
    sensor_over_time.append(sensor.sense(state).copy())
    actions.append(action)
    state, reward, done, info = env.step(action)
    if done:
        break

peg_arr = np.array(peg_over_time)
er4_arr = np.array(er4_over_time)
epg_arr = np.array(epg_over_time)
sensor_arr = np.array(sensor_over_time)

print("Episode length: %d steps" % len(actions))
print("Score: %d" % info["score"])
print("Final bird_y: %.1f" % state.bird_y)

print("\nPEG activity over time:")
print("  Mean per step: %.5f +/- %.5f" % (np.mean(peg_arr), np.std(np.mean(peg_arr, axis=1))))
print("  L-R asymmetry per step: %.5f +/- %.5f" % (
    np.mean(np.mean(peg_arr[:, :9], axis=1) - np.mean(peg_arr[:, 9:], axis=1)),
    np.std(np.mean(peg_arr[:, :9], axis=1) - np.mean(peg_arr[:, 9:], axis=1))
))

print("\nER4 activity over time:")
print("  Mean per step: %.5f +/- %.5f" % (np.mean(er4_arr), np.std(np.mean(er4_arr, axis=1))))

print("\nEPG activity over time:")
print("  Mean per step: %.5f +/- %.5f" % (np.mean(epg_arr), np.std(np.mean(epg_arr, axis=1))))

print("\nSensor total over time:")
print("  Mean per step: %.5f +/- %.5f" % (np.mean(np.sum(sensor_arr, axis=1)), np.std(np.sum(sensor_arr, axis=1))))

# Check if PEG changes when bird approaches pipe
# Find when pipe enters sensor range (sensor total > 0)
pipe_visible_steps = np.where(np.sum(sensor_arr, axis=1) > 0.1)[0]
if len(pipe_visible_steps) > 0:
    first_visible = pipe_visible_steps[0]
    print("\nPipe first visible at step %d" % first_visible)
    print("  PEG mean before: %.5f" % np.mean(peg_arr[:first_visible]))
    print("  PEG mean after:  %.5f" % np.mean(peg_arr[first_visible:]))
    print("  PEG L-R asym before: %.5f" % np.mean(np.mean(peg_arr[:first_visible, :9], axis=1) - np.mean(peg_arr[:first_visible, 9:], axis=1)))
    print("  PEG L-R asym after:  %.5f" % np.mean(np.mean(peg_arr[first_visible:, :9], axis=1) - np.mean(peg_arr[first_visible:, 9:], axis=1)))
