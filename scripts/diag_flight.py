"""Debug full flight with step-by-step EPG tracking."""
import sys
sys.path.insert(0, "src")
import numpy as np
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

for step in range(5):
    # Before act
    net.neurons.reset()
    sensor_out = agent.sensor.sense(state)
    ext = agent._build_sensory_current(sensor_out)
    print(f"step={step} sensor={sensor_out.round(3)} ext_max={ext.max():.3f} ext_nonzero={np.count_nonzero(ext)}")
    for _ in range(agent.sub_steps):
        act = net.step(ext)
    epg_act = act[agent.epg_ordered_indices]
    peak = np.argmax(epg_act)
    print(f"  EPG peak={peak} range={epg_act.max()-epg_act.min():.6f}")

    # Now do it properly through act()
    agent.reset()
    action = agent.act(state)
    diag = agent.get_motor_diagnostics()
    print(f"  action={action} peak={diag['epg_peak']} prob={diag['flap_prob']:.4f}")

    state, reward, done, info = env.step(action)
    print(f"  bird_y={state.bird_y:.1f} vy={state.bird_vy:.2f}")
    if done:
        break
