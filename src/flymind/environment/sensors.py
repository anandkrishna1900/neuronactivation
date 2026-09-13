"""
Sensory abstractions mapping environment state into receptive field activations.
No privileged ground truth coordinates given to the neural network.
"""

from typing import Tuple
import numpy as np
from flymind.environment.world import ArenaState


class SectorVisualSensor:
    """
    Biological abstraction of compound eye receptive field:
    Divided into Left, Center, and Right visual fields.
    Outputs normalized visual signal activations [left, center, right].
    """

    def __init__(self, fov_radians: float = np.pi):
        self.fov = fov_radians

    def sense(self, state: ArenaState) -> np.ndarray:
        """
        Compute relative bearing to target:
        Positive angle: target is to the Left
        Negative angle: target is to the Right
        """
        rel_vec = state.target_pos - state.agent_pos
        target_angle = np.arctan2(rel_vec[1], rel_vec[0])
        diff = (target_angle - state.agent_heading + np.pi) % (2 * np.pi) - np.pi

        # If outside field of view, no visual signal
        if abs(diff) > (self.fov / 2.0):
            return np.zeros(3, dtype=np.float64)

        # 3 receptive sectors: Left (diff > pi/6), Center (|diff| <= pi/6), Right (diff < -pi/6)
        left_act = np.clip(1.0 - abs(diff - np.pi / 3.0) / (np.pi / 3.0), 0.0, 1.0)
        center_act = np.clip(1.0 - abs(diff) / (np.pi / 6.0), 0.0, 1.0)
        right_act = np.clip(1.0 - abs(diff + np.pi / 3.0) / (np.pi / 3.0), 0.0, 1.0)

        return np.array([left_act, center_act, right_act], dtype=np.float64)
