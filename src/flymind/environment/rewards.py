"""
Reward computation for navigation benchmarks.
Provides Experiment A (Dense Progress-Shaped Reward) and Experiment B (Sparse Goal-Only Reward).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseRewardFunction(ABC):
    """Abstract reward function interface."""

    @abstractmethod
    def compute_reward(self, info: Dict[str, Any], done: bool) -> float:
        pass


class DenseNavigationReward(BaseRewardFunction):
    """
    Experiment A: Dense Shaped Navigation Reward
    R_t = progress_scale * (prev_dist - curr_dist) + goal_reward + time_penalty + collision_penalty
    """

    def __init__(
        self,
        progress_scale: float = 1.0,
        goal_reward: float = 10.0,
        time_penalty: float = -0.01,
        collision_penalty: float = -2.0,
    ):
        self.progress_scale = progress_scale
        self.goal_reward = goal_reward
        self.time_penalty = time_penalty
        self.collision_penalty = collision_penalty

    def compute_reward(self, info: Dict[str, Any], done: bool) -> float:
        r = self.time_penalty
        r += self.progress_scale * float(info.get("progress", 0.0))

        if info.get("collision", False):
            r += self.collision_penalty

        if info.get("target_reached", False):
            r += self.goal_reward

        return float(r)


class SparseGoalReward(BaseRewardFunction):
    """
    Experiment B: Sparse Biological-Style Goal Reward
    R_t = +1.0 if target reached, 0.0 otherwise.
    Optional failure penalty if max steps exceeded.
    """

    def __init__(
        self,
        goal_reward: float = 1.0,
        step_penalty: float = 0.0,
        failure_penalty: float = 0.0,
    ):
        self.goal_reward = goal_reward
        self.step_penalty = step_penalty
        self.failure_penalty = failure_penalty

    def compute_reward(self, info: Dict[str, Any], done: bool) -> float:
        if info.get("target_reached", False):
            return float(self.goal_reward)
        if done and not info.get("target_reached", False):
            return float(self.failure_penalty)
        return float(self.step_penalty)


# Backward compatibility alias
class TargetProximityReward:
    """Legacy reward calculator for backward compatibility."""

    def __init__(self, target_arrival_reward: float = 100.0, step_penalty: float = -0.1):
        self.target_arrival_reward = target_arrival_reward
        self.step_penalty = step_penalty

    def calculate(self, state) -> float:
        if state.target_reached:
            return self.target_arrival_reward
        return self.step_penalty
