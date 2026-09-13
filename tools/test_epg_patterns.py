"""Diagnostic: check if EPG patterns differ across gap positions."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from flymind.connectome.loader import ConnectomeLoader
from flymind.brain.network import NeuralNetwork
from flymind.environment.flappy import FlappyState
from flymind.environment.flappy_sensor import FlappyVisualSensor
from flymind.agent.flappy_agent_phase6b import ConnectomeMotorReadoutAgent

ROOT = Path(__file__).resolve().parent.parent
CONNECTOME_PATH = ROOT / "data" / "processed" / "cx_heading_v1.json"

graph = ConnectomeLoader.load_from_json(CONNECTOME_PATH)
network = NeuralNetwork(graph, synapse_scale=0.005)
agent = ConnectomeMotorReadoutAgent(network, enable_plasticity=False, sub_steps=10)
sensor = FlappyVisualSensor()

conditions = {
    "gap_above": {"bird_y": 100.0, "gap_center": 300.0},
    "gap_aligned": {"bird_y": 200.0, "gap_center": 200.0},
    "gap_below": {"bird_y": 300.0, "gap_center": 100.0},
}

print("=== EPG Ring Pattern by Gap Position ===")
for cond, params in conditions.items():
    state = FlappyState(
        bird_y=params["bird_y"],
        bird_vy=0.0,
        pipes=[{"x": 150.0, "gap_center": params["gap_center"]}],
        score=0, step_count=0, alive=True,
    )
    
    # Run 20 trials
    epg_patterns = []
    for trial in range(20):
        agent.reset()
        sensor_out = sensor.sense(state)
        ext_current = agent._build_sensory_current(sensor_out)
        activity = None
        for _ in range(agent.sub_steps):
            activity = network.step(ext_current)
        epg_act = activity[agent.epg_ordered_indices]
        epg_patterns.append(epg_act)
    
    mean_epg = np.mean(epg_patterns, axis=0)
    # Find peak position
    peak_pos = np.argmax(mean_epg)
    peak_val = mean_epg[peak_pos]
    
    print(f"\n{cond}:")
    print(f"  Sensor: {sensor_out.round(3)}")
    print(f"  EPG peak at position {peak_pos}, value={peak_val:.4f}")
    print(f"  EPG activity (16 positions): {mean_epg.round(4)}")
    print(f"  EPG mean={np.mean(mean_epg):.4f}, std={np.std(mean_epg):.4f}")
    
    # Check if patterns differ
    if cond == "gap_above":
        above_epg = mean_epg.copy()
    elif cond == "gap_below":
        below_epg = mean_epg.copy()

# Compute correlation between above and below patterns
corr = np.corrcoef(above_epg, below_epg)[0, 1]
print(f"\nCorrelation between gap_above and gap_below EPG patterns: {corr:.4f}")
print(f"(1.0 = identical, 0.0 = uncorrelated)")
