"""Design Phase 6B motor readout: sensor-informed, connectome-gated."""
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

network = NeuralNetwork(graph, synapse_scale=0.005)
agent = ConnectomeMotorReadoutAgent(network, enable_plasticity=False, sub_steps=10)

# Test: what does the sensor tell us about gap position?
conditions = [
    ("gap_above", 100, 300),
    ("gap_aligned", 200, 200),
    ("gap_below", 300, 100),
]

for cond, by, gc in conditions:
    state = FlappyState(bird_y=by, bird_vy=0.0, pipes=[{"x": 150, "gap_center": gc}], score=0, step_count=0, alive=True)
    s = sensor.sense(state)
    
    # Vertical signal from sensor (same as Phase 6A hand-designed)
    vertical_signal = (
        +1.0 * (s[0] + s[1] + s[2])  # upper row: gap above
        + 0.0 * (s[3] + s[4] + s[5])  # center row: gap aligned
        - 1.0 * (s[6] + s[7] + s[8])  # lower row: gap below
    )
    total = np.sum(s)
    if total > 1e-6:
        vertical_signal /= total
    
    # Gap-bird difference (what the hand-designed agent uses)
    gap_bird_diff = gc - by
    
    # Sensor tells us: positive = gap above (need to flap), negative = gap below (don't flap)
    print("%s: vertical_signal=%.3f, gap_bird_diff=%.0f, sensor=%s" % (
        cond, vertical_signal, gap_bird_diff, s.round(2)))

print("\nKey insight: sensor vertical_signal = (gc - by) / scale")
print("The sensor ALREADY encodes gap-bird difference!")
print("The motor readout just needs to threshold this signal.")
print("The connectome pathway (EPG->PEG) processes it, but the DECISION")
print("can be made from the sensor signal routed through the connectome.")
