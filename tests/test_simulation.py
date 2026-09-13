"""
Tests for neural simulation and 2D virtual arena environment.
"""

import numpy as np
from flymind.connectome.graph import ConnectomeGraph, NeuronMetadata, SynapticConnection
from flymind.brain.network import NeuralNetwork
from flymind.brain.neuron import RateNeuron
from flymind.environment.world import VirtualArena
from flymind.environment.sensors import SectorVisualSensor
from flymind.agent.fly import ConnectomeFlyAgent, RandomFlyAgent


def test_neural_network_step():
    g = ConnectomeGraph(name="mini_circuit")
    g.add_neuron(NeuronMetadata(body_id=1, cell_type="ER", neurotransmitter="cholinergic"))
    g.add_neuron(NeuronMetadata(body_id=2, cell_type="E-PG", neurotransmitter="cholinergic"))
    g.add_connection(SynapticConnection(source_id=1, target_id=2, weight=10.0))

    net = NeuralNetwork(graph=g, neuron_model_cls=RateNeuron)
    
    # Inject current into neuron 1
    inp = np.array([5.0, 0.0])
    act1 = net.step(inp)
    assert act1.shape == (2,)
    assert act1[0] > 0.0  # Neuron 1 activated

    # Step again, signal should propagate to neuron 2
    act2 = net.step(inp)
    assert act2[1] > 0.0  # Neuron 2 activated through synapse


def test_arena_environment_and_sensor():
    arena = VirtualArena(width=100.0, height=100.0, seed=42)
    state = arena.reset(seed=42)

    assert state.agent_pos.shape == (2,)
    assert 0.0 <= state.agent_heading < 2 * np.pi

    sensor = SectorVisualSensor()
    vis = sensor.sense(state)
    assert vis.shape == (3,)
    assert np.all(vis >= 0.0) and np.all(vis <= 1.0)

    # Step forward
    new_state, reward, done, info = arena.step(1)
    assert new_state.step_count == 1
    assert "distance" in info


def test_random_agent_trial():
    arena = VirtualArena(max_steps=20, seed=42)
    agent = RandomFlyAgent(seed=42)
    state = arena.reset(seed=42)

    done = False
    total_reward = 0.0
    while not done:
        action = agent.act(state)
        state, reward, done, _ = arena.step(action)
        total_reward += reward

    assert state.step_count <= 20
