"""Test different sensory drive levels."""
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

for drive in [25, 50, 100, 200]:
    network = NeuralNetwork(graph, synapse_scale=0.005)
    agent = ConnectomeMotorReadoutAgent(network, enable_plasticity=False, sub_steps=10, sensory_drive=drive)
    
    results = {}
    for cond, by, gc in [("above", 100, 300), ("aligned", 200, 200), ("below", 300, 100)]:
        state = FlappyState(bird_y=by, bird_vy=0.0, pipes=[{"x": 150.0, "gap_center": gc}], score=0, step_count=0, alive=True)
        peg_acts = []
        for _ in range(20):
            agent.reset()
            s = sensor.sense(state)
            ext = agent._build_sensory_current(s)
            act = None
            for _ in range(agent.sub_steps):
                act = network.step(ext)
            peg_acts.append(act[agent.peg_indices])
        mean_peg = np.mean(peg_acts, axis=0)
        asym = float(np.mean(mean_peg[:9]) - np.mean(mean_peg[9:]))
        results[cond] = {"mean": float(np.mean(mean_peg)), "asym": asym}
    
    print("drive=%4d: above_asym=%.5f, aligned_asym=%.5f, below_asym=%.5f" % (
        drive, results["above"]["asym"], results["aligned"]["asym"], results["below"]["asym"]))
