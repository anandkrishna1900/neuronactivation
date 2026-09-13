"""Diagnostic: trace signal flow from sensor through connectome to PEG."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from flymind.connectome.loader import ConnectomeLoader
from flymind.brain.network import NeuralNetwork
from flymind.environment.flappy import FlappyEnvironment, FlappyState
from flymind.environment.flappy_sensor import FlappyVisualSensor
from flymind.agent.flappy_agent_phase6b import ConnectomeMotorReadoutAgent

ROOT = Path(__file__).resolve().parent.parent
CONNECTOME_PATH = ROOT / "data" / "processed" / "cx_heading_v1.json"

graph = ConnectomeLoader.load_from_json(CONNECTOME_PATH)
network = NeuralNetwork(graph, synapse_scale=0.005)
agent = ConnectomeMotorReadoutAgent(network, enable_plasticity=False, sub_steps=10)
sensor = FlappyVisualSensor()

# Test conditions
conditions = {
    "gap_above": {"bird_y": 100.0, "gap_center": 300.0},
    "gap_aligned": {"bird_y": 200.0, "gap_center": 200.0},
    "gap_below": {"bird_y": 300.0, "gap_center": 100.0},
    "no_pipe": {"bird_y": 200.0, "pipes": []},
}

print("=== Sensor -> Connectome -> PEG Signal Flow ===")
for cond, params in conditions.items():
    pipes = params.get("pipes", [{"x": 150.0, "gap_center": params["gap_center"]}])
    state = FlappyState(
        bird_y=params["bird_y"],
        bird_vy=0.0,
        pipes=pipes,
        score=0,
        step_count=0,
        alive=True,
    )

    # Run 20 trials
    peg_activities = []
    er4_activities = []
    epg_activities = []
    sensor_outs = []
    motor_scores = []

    for trial in range(20):
        agent.reset()
        sensor_out = sensor.sense(state)
        ext_current = agent._build_sensory_current(sensor_out)
        activity = None
        for _ in range(agent.sub_steps):
            activity = network.step(ext_current)
        
        peg_act = activity[agent.peg_indices]
        er4_act = activity[agent.er4d_indices]
        epg_act = activity[agent.epg_indices]
        
        peg_activities.append(peg_act)
        er4_activities.append(er4_act)
        epg_activities.append(epg_act)
        sensor_outs.append(sensor_out)
        
        # Motor score with random weights
        motor_score = float(np.dot(agent.motor_readout_weights, peg_act - agent.peg_baseline))
        motor_scores.append(motor_score)

    mean_sensor = np.mean(sensor_outs, axis=0)
    mean_peg = np.mean(peg_activities, axis=0)
    mean_er4 = np.mean(er4_activities, axis=0)
    mean_epg = np.mean(epg_activities, axis=0)
    
    print(f"\n{cond}:")
    print(f"  Sensor: {mean_sensor.round(4)}")
    print(f"  Sensor total: {np.sum(mean_sensor):.4f}")
    print(f"  ER4 mean: {np.mean(mean_er4):.4f}, std: {np.std(mean_er4):.4f}")
    print(f"  EPG mean: {np.mean(mean_epg):.4f}, std: {np.std(mean_epg):.4f}")
    print(f"  PEG mean: {np.mean(mean_peg):.4f}, std: {np.std(mean_peg):.4f}")
    print(f"  PEG L/R: {np.mean(mean_peg[:9]):.4f} / {np.mean(mean_peg[9:]):.4f}")
    print(f"  Motor scores: {np.mean(motor_scores):.4f} +/- {np.std(motor_scores):.4f}")

# Now test: what if we use a hand-designed decoder based on sensor (like Phase 6A)?
print("\n\n=== Phase 6A hand-designed decoder for comparison ===")
agent6a = ConnectomeMotorReadoutAgent(
    NeuralNetwork(graph, synapse_scale=0.005), 
    enable_plasticity=False,
)
# Manually set sensor to test conditions
for cond, params in conditions.items():
    pipes = params.get("pipes", [{"x": 150.0, "gap_center": params["gap_center"]}])
    state = FlappyState(
        bird_y=params["bird_y"],
        bird_vy=0.0,
        pipes=pipes,
        score=0,
        step_count=0,
        alive=True,
    )
    sensor_out = sensor.sense(state)
    
    # Phase 6A style vertical signal
    vertical_signal = (
        +1.0 * (sensor_out[0] + sensor_out[1] + sensor_out[2])
        + 0.0 * (sensor_out[3] + sensor_out[4] + sensor_out[5])
        - 1.0 * (sensor_out[6] + sensor_out[7] + sensor_out[8])
    )
    total = np.sum(sensor_out)
    if total > 1e-6:
        vertical_signal /= total
    
    print(f"  {cond}: vertical_signal={vertical_signal:.4f}, total_sensor={total:.4f}")
