"""
Unit tests for Phase 2: Learned Heading Navigation.
Validates randomized arena spawning, sensory receptive fields, reward modules, eligibility traces, and frozen evaluation.
"""

import numpy as np
from pathlib import Path

from flymind.connectome.loader import ConnectomeLoader
from flymind.brain.network import NeuralNetwork
from flymind.brain.neuron import RateNeuron
from flymind.brain.plasticity import RewardModulatedHebbian
from flymind.environment.world import VirtualArena
from flymind.environment.sensors import SectorVisualSensor
from flymind.environment.rewards import DenseNavigationReward, SparseGoalReward
from flymind.agent.plastic_cx import PlasticCXAgent

GRAPH_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "cx_heading_v1.json"


def test_randomized_arena_initialization():
    arena = VirtualArena(width=100.0, height=100.0, min_start_target_dist=20.0, seed=42)
    state = arena.reset(seed=42)

    # Validate coordinate bounds
    assert 8.0 <= state.agent_pos[0] <= 92.0
    assert 8.0 <= state.agent_pos[1] <= 92.0
    assert 0.0 <= state.agent_heading < 2.0 * np.pi

    # Validate minimum distance separation
    dist = np.linalg.norm(state.agent_pos - state.target_pos)
    assert dist >= 20.0

    # Test seed reproducibility
    state2 = arena.reset(seed=42)
    assert np.allclose(state.agent_pos, state2.agent_pos)
    assert np.isclose(state.agent_heading, state2.agent_heading)


def test_sensory_encoding_no_privileged_coords():
    sensor = SectorVisualSensor()
    arena = VirtualArena(seed=123)
    state = arena.reset(seed=123)

    signals = sensor.sense(state)
    assert signals.shape == (3,)
    assert np.all(signals >= 0.0) and np.all(signals <= 1.0)


def test_reward_regimes():
    dense = DenseNavigationReward(progress_scale=1.0, goal_reward=10.0, time_penalty=-0.01, collision_penalty=-2.0)
    sparse = SparseGoalReward(goal_reward=1.0, step_penalty=0.0)

    # Test progress step
    info = {"progress": 1.5, "collision": False, "target_reached": False}
    r_dense = dense.compute_reward(info, done=False)
    assert np.isclose(r_dense, 1.49)  # 1.5 - 0.01

    r_sparse = sparse.compute_reward(info, done=False)
    assert np.isclose(r_sparse, 0.0)

    # Test goal arrival
    info_goal = {"progress": 0.5, "collision": False, "target_reached": True}
    assert dense.compute_reward(info_goal, done=True) > 10.0
    assert np.isclose(sparse.compute_reward(info_goal, done=True), 1.0)


def test_plasticity_eligibility_and_mask():
    weights = np.array([
        [0.0, 5.0],
        [0.0, 0.0],
    ], dtype=np.float64)
    mask = (weights > 0).astype(np.float64)

    rule = RewardModulatedHebbian(learning_rate=0.01, eligibility_decay=0.5, plasticity_mask=mask)
    pre = np.array([1.0, 0.0])
    post = np.array([0.0, 1.0])

    # Update with positive reward
    w_new = rule.update(weights, pre, post, reward=10.0)
    
    # Non-existent edge (0, 0) and (1, 0) must remain 0
    assert w_new[0, 0] == 0.0
    assert w_new[1, 0] == 0.0
    assert w_new[1, 1] == 0.0
    # Existing edge (0, 1) should increase
    assert w_new[0, 1] > 5.0


def test_agent_freeze_weights():
    graph = ConnectomeLoader.load_from_json(GRAPH_PATH)
    net = NeuralNetwork(graph=graph, neuron_model_cls=RateNeuron)
    agent = PlasticCXAgent(network=net, learning_rate=0.05)

    initial_weights = agent.network.synapses.raw_weights.copy()
    arena = VirtualArena(seed=42)
    state = arena.reset(seed=42)

    # Act twice to populate both prev_activity and current_activity
    agent.act(state)
    agent.act(state)
    agent.apply_reward(10.0)
    assert not np.allclose(agent.network.synapses.raw_weights, initial_weights)

    # Freeze weights
    agent.freeze_weights()
    frozen_weights = agent.network.synapses.raw_weights.copy()
    agent.act(state)
    agent.apply_reward(50.0)
    # Weights must NOT change when frozen
    assert np.allclose(agent.network.synapses.raw_weights, frozen_weights)
