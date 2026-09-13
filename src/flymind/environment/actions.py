"""
Action definitions for locomotion.
"""

from enum import IntEnum
from typing import List


class ActionType(IntEnum):
    STOP = 0
    FORWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3


class DiscreteActionSpace:
    """Action space containing the discrete actions available to the agent."""

    def __init__(self):
        self.actions: List[ActionType] = [
            ActionType.STOP,
            ActionType.FORWARD,
            ActionType.TURN_LEFT,
            ActionType.TURN_RIGHT,
        ]

    def __len__(self) -> int:
        return len(self.actions)
