"""
2D continuous bounded virtual arena with full position and heading randomization.
Tracks collisions, step counts, and distances without giving coordinates to the agent.
"""

from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
import numpy as np


@dataclass
class ArenaState:
    agent_pos: np.ndarray       # [x, y] - used only internally by env / sensor
    agent_heading: float        # angle in radians [0, 2*pi)
    target_pos: np.ndarray      # [x, y] - used only internally by env / sensor
    step_count: int
    target_reached: bool
    collision_occurred: bool


class VirtualArena:
    """
    Bounded 2D continuous arena with randomized fly agent and goal beacon positions.
    """

    def __init__(
        self,
        width: float = 100.0,
        height: float = 100.0,
        target_radius: float = 6.0,
        step_size: float = 2.0,
        turn_angle: float = np.pi / 8.0,
        max_steps: int = 400,
        min_start_target_dist: float = 25.0,
        wall_margin: float = 8.0,
        seed: Optional[int] = None,
    ):
        self.width = width
        self.height = height
        self.target_radius = target_radius
        self.step_size = step_size
        self.turn_angle = turn_angle
        self.max_steps = max_steps
        self.min_start_target_dist = min_start_target_dist
        self.wall_margin = wall_margin
        self.rng = np.random.default_rng(seed)

        self.agent_pos = np.zeros(2, dtype=np.float64)
        self.agent_heading = 0.0
        self.target_pos = np.zeros(2, dtype=np.float64)
        self.step_count = 0
        self.total_path_length = 0.0
        self.total_collisions = 0
        self.prev_distance = 0.0

    def reset(self, seed: Optional[int] = None) -> ArenaState:
        """
        Reset arena with fully randomized initial fly position, heading, and target position.
        Guarantees minimum distance between start and target to prevent trivial spawning.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        margin = self.wall_margin

        # Randomize agent position within bounded margins
        self.agent_pos = self.rng.uniform(
            [margin, margin],
            [self.width - margin, self.height - margin]
        ).astype(np.float64)

        # Randomize initial heading in [0, 2*pi)
        self.agent_heading = float(self.rng.uniform(0.0, 2.0 * np.pi))

        # Randomize target position ensuring minimum distance from start
        for _ in range(100):
            target_candidate = self.rng.uniform(
                [margin, margin],
                [self.width - margin, self.height - margin]
            ).astype(np.float64)
            dist = float(np.linalg.norm(self.agent_pos - target_candidate))
            if dist >= self.min_start_target_dist:
                self.target_pos = target_candidate
                break
        else:
            self.target_pos = target_candidate

        self.step_count = 0
        self.total_path_length = 0.0
        self.total_collisions = 0
        self.prev_distance = float(np.linalg.norm(self.agent_pos - self.target_pos))

        return self.get_state(collision=False)

    def step(self, action_id: int) -> Tuple[ArenaState, float, bool, Dict[str, Any]]:
        """
        Execute an action:
        0: Stop
        1: Move Forward
        2: Turn Left
        3: Turn Right
        """
        self.step_count += 1
        collision = False
        prev_pos = self.agent_pos.copy()

        if action_id == 1:  # Move Forward
            dx = self.step_size * np.cos(self.agent_heading)
            dy = self.step_size * np.sin(self.agent_heading)
            new_x = self.agent_pos[0] + dx
            new_y = self.agent_pos[1] + dy

            # Check arena wall boundary collision
            if new_x < 0.0 or new_x > self.width or new_y < 0.0 or new_y > self.height:
                collision = True
                self.total_collisions += 1

            self.agent_pos[0] = np.clip(new_x, 0.0, self.width)
            self.agent_pos[1] = np.clip(new_y, 0.0, self.height)
            self.total_path_length += float(np.linalg.norm(self.agent_pos - prev_pos))

        elif action_id == 2:  # Turn Left
            self.agent_heading = float((self.agent_heading + self.turn_angle) % (2.0 * np.pi))
        elif action_id == 3:  # Turn Right
            self.agent_heading = float((self.agent_heading - self.turn_angle) % (2.0 * np.pi))

        curr_distance = float(np.linalg.norm(self.agent_pos - self.target_pos))
        target_reached = bool(curr_distance <= self.target_radius)
        done = target_reached or (self.step_count >= self.max_steps)

        # Progress distance delta (positive when getting closer)
        progress = self.prev_distance - curr_distance
        self.prev_distance = curr_distance

        info = {
            "distance": curr_distance,
            "progress": progress,
            "step": self.step_count,
            "collision": collision,
            "total_collisions": self.total_collisions,
            "path_length": self.total_path_length,
            "target_reached": target_reached,
        }

        state = self.get_state(collision=collision)
        return state, 0.0, done, info

    def get_state(self, collision: bool = False) -> ArenaState:
        dist = float(np.linalg.norm(self.agent_pos - self.target_pos))
        return ArenaState(
            agent_pos=self.agent_pos.copy(),
            agent_heading=self.agent_heading,
            target_pos=self.target_pos.copy(),
            step_count=self.step_count,
            target_reached=bool(dist <= self.target_radius),
            collision_occurred=collision,
        )
