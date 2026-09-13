"""
Reward computation for target navigation.
"""

from flymind.environment.world import ArenaState


class TargetProximityReward:
    """Computes sparse and distance-based shaping rewards."""

    def __init__(self, target_arrival_reward: float = 100.0, step_penalty: float = -0.1):
        self.target_arrival_reward = target_arrival_reward
        self.step_penalty = step_penalty

    def calculate(self, state: ArenaState) -> float:
        if state.target_reached:
            return self.target_arrival_reward
        return self.step_penalty
