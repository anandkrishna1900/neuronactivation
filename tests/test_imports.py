"""
Smoke tests for verifying importability and package integrity.
"""

import flymind
from flymind.connectome import ConnectomeGraph, ConnectomeLoader, validate_connectome_graph
from flymind.brain import RateNeuron, LIFNeuron, NeuralNetwork, DiscreteSimulator
from flymind.environment import VirtualArena, SectorVisualSensor, ActionType
from flymind.agent import ConnectomeFlyAgent, RandomFlyAgent


def test_package_metadata():
    assert flymind.__version__ == "0.1.0"
    assert "FlyMind" in flymind.__doc__


def test_core_classes_exist():
    assert ConnectomeGraph is not None
    assert ConnectomeLoader is not None
    assert RateNeuron is not None
    assert LIFNeuron is not None
    assert VirtualArena is not None
    assert SectorVisualSensor is not None
    assert ConnectomeFlyAgent is not None
    assert RandomFlyAgent is not None
