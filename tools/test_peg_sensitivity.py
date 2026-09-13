"""Test if PEG activity changes with different sensory inputs."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from flymind.connectome.loader import ConnectomeLoader
from flymind.brain.network import NeuralNetwork
from flymind.environment.flappy import FlappyState
from flymind.environment.flappy_sensor import FlappyVisualSensor
from flymind.agent.flappy_agent_phase6b import ConnectomeMotorReadoutAgent

graph = ConnectomeLoader.load_from_json("data/processed/cx_heading_v1.json")
sensor = FlappyVisualSensor()

# Test: does PEG activity change when sensor input changes?
network = NeuralNetwork(graph, synapse_scale=0.005)
agent = ConnectomeMotorReadoutAgent(network, enable_plasticity=False, sub_steps=10)

# Condition 1: strong sensor input (pipe visible)
state_with_pipe = FlappyState(bird_y=200, bird_vy=0.0, pipes=[{"x": 150, "gap_center": 200}], score=0, step_count=0, alive=True)
# Condition 2: no pipe (dark sensor)
state_no_pipe = FlappyState(bird_y=200, bird_vy=0.0, pipes=[], score=0, step_count=0, alive=True)

peg_with = []
peg_without = []
er4_with = []
er4_without = []
epg_with = []
epg_without = []

for _ in range(50):
    # With pipe
    agent.reset()
    s1 = sensor.sense(state_with_pipe)
    ext1 = agent._build_sensory_current(s1)
    act1 = None
    for _ in range(agent.sub_steps):
        act1 = network.step(ext1)
    peg_with.append(act1[agent.peg_indices])
    er4_with.append(act1[agent.er4d_indices])
    epg_with.append(act1[agent.epg_indices])
    
    # Without pipe
    agent.reset()
    s2 = sensor.sense(state_no_pipe)
    ext2 = agent._build_sensory_current(s2)
    act2 = None
    for _ in range(agent.sub_steps):
        act2 = network.step(ext2)
    peg_without.append(act2[agent.peg_indices])
    er4_without.append(act2[agent.er4d_indices])
    epg_without.append(act2[agent.epg_indices])

mean_peg_with = np.mean(peg_with, axis=0)
mean_peg_without = np.mean(peg_without, axis=0)
mean_er4_with = np.mean(er4_with, axis=0)
mean_er4_without = np.mean(er4_without, axis=0)
mean_epg_with = np.mean(epg_with, axis=0)
mean_epg_without = np.mean(epg_without, axis=0)

print("PEG activity:")
print("  With pipe:    mean=%.5f, L=%.5f, R=%.5f" % (np.mean(mean_peg_with), np.mean(mean_peg_with[:9]), np.mean(mean_peg_with[9:])))
print("  Without pipe: mean=%.5f, L=%.5f, R=%.5f" % (np.mean(mean_peg_without), np.mean(mean_peg_without[:9]), np.mean(mean_peg_without[9:])))
print("  Difference:   %.5f" % (np.mean(mean_peg_with) - np.mean(mean_peg_without)))

print("\nER4 activity:")
print("  With pipe:    mean=%.5f" % np.mean(mean_er4_with))
print("  Without pipe: mean=%.5f" % np.mean(mean_er4_without))
print("  Difference:   %.5f" % (np.mean(mean_er4_with) - np.mean(mean_er4_without)))

print("\nEPG activity:")
print("  With pipe:    mean=%.5f, std=%.5f" % (np.mean(mean_epg_with), np.std(mean_epg_with)))
print("  Without pipe: mean=%.5f, std=%.5f" % (np.mean(mean_epg_without), np.std(mean_epg_without)))
print("  Difference:   %.5f" % (np.mean(mean_epg_with) - np.mean(mean_epg_without)))

# Correlation: does PEG pattern change?
peg_corr = np.corrcoef(mean_peg_with, mean_peg_without)[0, 1]
epg_corr = np.corrcoef(mean_epg_with, mean_epg_without)[0, 1]
print("\nPattern correlation (with vs without pipe):")
print("  PEG: %.4f" % peg_corr)
print("  EPG: %.4f" % epg_corr)
