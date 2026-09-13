"""
Sensory abstractions mapping environment state into receptive field activations.
Provides:
1. SectorVisualSensor (180 deg FOV, 3 sectors)
2. PanoramicCompoundEyeSensor (360 deg panoramic vision, 12 directional ommatidia channels)
3. ActiveExplorationSensor (180 deg visual field with exploratory saccadic bursts when blind)
"""

from typing import Tuple, Optional
import numpy as np
from flymind.environment.world import ArenaState


class SectorVisualSensor:
    """
    Biological abstraction of compound eye receptive field:
    Divided into Left, Center, and Right visual fields within fov_radians.
    """

    def __init__(self, fov_radians: float = np.pi):
        self.fov = fov_radians

    def sense(self, state: ArenaState) -> np.ndarray:
        rel_vec = state.target_pos - state.agent_pos
        target_angle = np.arctan2(rel_vec[1], rel_vec[0])
        diff = (target_angle - state.agent_heading + np.pi) % (2.0 * np.pi) - np.pi

        if abs(diff) > (self.fov / 2.0):
            return np.zeros(3, dtype=np.float64)

        left_act = np.clip(1.0 - abs(diff - np.pi / 3.0) / (np.pi / 3.0), 0.0, 1.0)
        center_act = np.clip(1.0 - abs(diff) / (np.pi / 6.0), 0.0, 1.0)
        right_act = np.clip(1.0 - abs(diff + np.pi / 3.0) / (np.pi / 3.0), 0.0, 1.0)

        return np.array([left_act, center_act, right_act], dtype=np.float64)


class PanoramicCompoundEyeSensor:
    """
    Full 360-degree panoramic compound eye receptive field array.
    12 directional channels placed at 0, 30, 60, ..., 330 degrees around the azimuth.
    Provides complete azimuthal coverage without giving away privileged Cartesian coordinates.
    """

    def __init__(self, num_ommatidia: int = 12, tuning_width_rad: float = np.pi / 6.0):
        self.num_ommatidia = num_ommatidia
        self.tuning_width = tuning_width_rad
        # Preferred azimuthal angles from -pi to +pi
        self.preferred_angles = np.linspace(-np.pi + (np.pi / num_ommatidia), np.pi - (np.pi / num_ommatidia), num_ommatidia)

    def sense(self, state: ArenaState) -> np.ndarray:
        rel_vec = state.target_pos - state.agent_pos
        target_angle = np.arctan2(rel_vec[1], rel_vec[0])
        # Relative bearing diff in [-pi, pi]
        diff = (target_angle - state.agent_heading + np.pi) % (2.0 * np.pi) - np.pi

        activations = np.zeros(self.num_ommatidia, dtype=np.float64)
        for i, pref_ang in enumerate(self.preferred_angles):
            ang_err = (diff - pref_ang + np.pi) % (2.0 * np.pi) - np.pi
            # Circular Gaussian / Von Mises receptive field profile
            act = np.exp(-0.5 * (ang_err / self.tuning_width)**2)
            activations[i] = float(act)

        # Normalize so max activation is 1.0
        max_v = np.max(activations)
        if max_v > 1e-6:
            activations = activations / max_v

        return activations


class ActiveExplorationSensor:
    """
    180-degree visual field with active exploratory saccade drive.
    When target is outside FOV (blind spot), injects an exploratory turn bias.
    """

    def __init__(self, fov_radians: float = np.pi):
        self.base_sensor = SectorVisualSensor(fov_radians=fov_radians)
        self.blind_counter = 0

    def sense(self, state: ArenaState) -> np.ndarray:
        base_act = self.base_sensor.sense(state)
        if np.sum(base_act) < 1e-4:
            self.blind_counter += 1
            # Signal active exploration: exploratory asymmetric saccade drive
            return np.array([0.0, 0.0, 0.6], dtype=np.float64)
        else:
            self.blind_counter = 0
            return base_act
