"""
2D virtual arena environment, sensory models, actions, and rewards.
"""

from flymind.environment.world import VirtualArena, ArenaState
from flymind.environment.sensors import SectorVisualSensor
from flymind.environment.actions import ActionType, DiscreteActionSpace
from flymind.environment.rewards import TargetProximityReward

__all__ = [
    "VirtualArena",
    "ArenaState",
    "SectorVisualSensor",
    "ActionType",
    "DiscreteActionSpace",
    "TargetProximityReward",
]
