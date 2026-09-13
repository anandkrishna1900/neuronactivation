"""
2D continuous bounded virtual arena.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np


@dataclass
class ArenaState:
    agent_pos: np.ndarray  # [x, y]
    agent_heading: float   # angle in radians [0, 2*pi)
    target_pos: np.ndarray # [x, y]
    step_count: int
    target_reached: bool


class VirtualArena:
    """Bounded 2D rectangular arena with fly agent and goal beacon."""

    def __init__(
        self,
        width: float = 100.0,
        height: float = 100.0,
        target_radius: float = 5.0,
        step_size: float = 2.0,
        turn_angle: float = np.pi / 8.0,
        max_steps: int = 500,
        seed: Optional[int] = None,
    ):
        self.width = width
        self.height = height
        self.target_radius = target_radius
        self.step_size = step_size
        self.turn_angle = turn_angle
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

        self.agent_pos = np.zeros(2, dtype=np.float64)
        self.agent_heading = 0.0
        self.target_pos = np.zeros(2, dtype=np.float64)
        self.step_count = 0

    def reset(self, seed: Optional[int] = None) -> ArenaState:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.agent_pos = np.array([self.width / 2.0, self.height / 2.0], dtype=np.float64)
        self.agent_heading = self.rng.uniform(0.0, 2 * np.pi)

        # Randomize target position within arena bounds
        margin = 10.0
        self.target_pos = self.rng.uniform(
            [margin, margin],
            [self.width - margin, self.height - margin]
        )
        self.step_count = 0
        return self.get_state()

    def step(self, action_id: int) -> Tuple[ArenaState, float, bool, dict]:
        """
        Execute an action:
        0: Stop
        1: Move Forward
        2: Turn Left
        3: Turn Right
        """
        self.step_count += 1

        if action_id == 1:  # Forward
            dx = self.step_size * np.cos(self.agent_heading)
            dy = self.step_size * np.sin(self.agent_heading)
            self.agent_pos[0] = np.clip(self.agent_pos[0] + dx, 0.0, self.width)
            self.agent_pos[1] = np.clip(self.agent_pos[1] + dy, 0.0, self.height)
        elif action_id == 2:  # Turn Left
            self.agent_heading = (self.agent_heading + self.turn_angle) % (2 * np.pi)
        elif action_id == 3:  # Turn Right
            self.agent_heading = (self.agent_heading - self.turn_angle) % (2 * np.pi)

        dist = np.linalg.norm(self.agent_pos - self.target_pos)
        target_reached = bool(dist <= self.target_radius)
        done = target_reached or (self.step_count >= self.max_steps)

        reward = 100.0 if target_reached else -0.1
        info = {"distance": dist, "step": self.step_count}

        return self.get_state(), reward, done, info

    def get_state(self) -> ArenaState:
        dist = np.linalg.norm(self.agent_pos - self.target_pos)
        return ArenaState(
            agent_pos=self.agent_pos.copy(),
            agent_heading=self.agent_heading,
            target_pos=self.target_pos.copy(),
            step_count=self.step_count,
            target_reached=bool(dist <= self.target_radius),
        )
