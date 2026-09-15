"""
Phase 6A: Lightweight Flappy-Bird-style 2D side-scrolling environment.
Deterministic physics given a seed. No dependency on original game assets.

Environment parameters calibrated for reasonable flight dynamics.
Physics are NOT optimized to make the neural controller succeed.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any, List
import numpy as np


# ── Calibrated physics constants ──────────────────────────────────────────────
# These values produce reasonable flappy-bird-like dynamics.
# Documented and fixed. NOT tuned for neural controller success.
GRAVITY = -0.25           # pixels/frame^2 downward acceleration (moderate)
FLAP_IMPULSE = +4.5       # pixels/frame upward velocity change on flap
HORIZONTAL_SPEED = 1.5    # pixels/frame world scroll speed (measured)
BIRD_X = 80.0            # bird fixed horizontal position (world scrolls)
BIRD_RADIUS = 8.0        # collision radius
PIPE_WIDTH = 48.0        # horizontal extent of each pipe column
PIPE_SPACING = 200.0     # horizontal distance between consecutive pipes
GAP_SIZE = 160.0         # vertical opening between top and bottom pipes (generous)
WORLD_HEIGHT = 400.0     # vertical extent
WORLD_WIDTH = 600.0      # horizontal extent (for rendering reference)
MAX_STEPS = 2000         # maximum steps per episode
CEILING_Y = WORLD_HEIGHT  # top boundary
FLOOR_Y = 0.0            # bottom boundary


@dataclass
class FlappyState:
    """Internal game state exposed to the agent only through the sensor."""
    bird_y: float
    bird_vy: float
    pipes: List[Dict[str, float]]   # list of {"x": float, "gap_center": float}
    score: int
    step_count: int
    alive: bool
    obstacle_passed_this_step: bool = False


@dataclass
class FlappyObservation:
    """
    What the agent receives: sensor activations only.
    Never contains bird_y, bird_vy, pipe_x, gap_center directly.
    """
    sensor_activations: np.ndarray  # shape (n_channels,)
    score: int
    alive: bool


class FlappyEnvironment:
    """
    Deterministic 2D side-scrolling Flappy-Bird-style sandbox.

    The bird is fixed at horizontal position BIRD_X; the world scrolls left.
    Pipes are generated at regular intervals with randomized gap vertical positions.

    Action space:
        0 = NO FLAP (gravity pulls bird down)
        1 = FLAP (impulse upward)
    """

    def __init__(
        self,
        gravity: float = GRAVITY,
        flap_impulse: float = FLAP_IMPULSE,
        horizontal_speed: float = HORIZONTAL_SPEED,
        gap_size: float = GAP_SIZE,
        pipe_spacing: float = PIPE_SPACING,
        bird_x: float = BIRD_X,
        bird_radius: float = BIRD_RADIUS,
        pipe_width: float = PIPE_WIDTH,
        world_height: float = WORLD_HEIGHT,
        max_steps: int = MAX_STEPS,
        seed: Optional[int] = None,
        # Curriculum support
        initial_bird_y: Optional[float] = None,
    ):
        self.gravity = gravity
        self.flap_impulse = flap_impulse
        self.horizontal_speed = horizontal_speed
        self.gap_size = gap_size
        self.pipe_spacing = pipe_spacing
        self.bird_x = bird_x
        self.bird_radius = bird_radius
        self.pipe_width = pipe_width
        self.world_height = world_height
        self.max_steps = max_steps
        self.initial_bird_y = initial_bird_y
        self.world_width = WORLD_WIDTH

        self.rng = np.random.default_rng(seed)
        self._reset_state()

    def _reset_state(self) -> None:
        self.bird_y = self.initial_bird_y if self.initial_bird_y is not None else (self.world_height / 2.0)
        self.bird_vy = 0.0
        self.pipes = []
        self.score = 0
        self.step_count = 0
        self.alive = True
        self._next_pipe_x = self.bird_x + 300.0  # first pipe well ahead
        self._generate_initial_pipes()

    def _gap_center_range(self) -> Tuple[float, float]:
        """Vertical range for gap center, keeping gap fully within world."""
        margin = self.gap_size / 2.0 + self.bird_radius + 10.0
        return (margin, self.world_height - margin)

    def _generate_initial_pipes(self) -> None:
        """Pre-generate pipes across the visible window."""
        lo, hi = self._gap_center_range()
        x = self._next_pipe_x
        while x < self.bird_x + self.world_width + 200.0:
            gap_center = float(self.rng.uniform(lo, hi))
            self.pipes.append({"x": x, "gap_center": gap_center})
            x += self.pipe_spacing
        self._next_pipe_x = x

    def reset(self, seed: Optional[int] = None) -> FlappyState:
        """Reset environment, optionally with a new seed."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._reset_state()
        return self.get_state()

    def get_state(self) -> FlappyState:
        return FlappyState(
            bird_y=self.bird_y,
            bird_vy=self.bird_vy,
            pipes=[dict(p) for p in self.pipes],
            score=self.score,
            step_count=self.step_count,
            alive=self.alive,
        )

    def step(self, action: int) -> Tuple[FlappyState, float, bool, Dict[str, Any]]:
        """
        Execute one timestep.

        Args:
            action: 0 = NO FLAP, 1 = FLAP

        Returns:
            (state, reward, done, info)
        """
        if not self.alive:
            return self.get_state(), 0.0, True, {"reason": "already_dead"}

        self.step_count += 1
        obstacle_passed = False

        # ── Action ────────────────────────────────────────────────────────────
        if action == 1:
            self.bird_vy += self.flap_impulse

        # ── Physics ───────────────────────────────────────────────────────────
        self.bird_vy += self.gravity
        self.bird_y += self.bird_vy

        # ── Scroll pipes ──────────────────────────────────────────────────────
        for pipe in self.pipes:
            pipe["x"] -= self.horizontal_speed

        # Scroll the next-pipe anchor so it stays in the same world-frame
        self._next_pipe_x -= self.horizontal_speed

        # Remove pipes that have scrolled past the bird
        while self.pipes and self.pipes[0]["x"] + self.pipe_width / 2.0 < self.bird_x - 30.0:
            self.pipes.pop(0)

        # Generate new pipes ahead
        lo, hi = self._gap_center_range()
        while self._next_pipe_x < self.bird_x + self.world_width + 200.0:
            gap_center = float(self.rng.uniform(lo, hi))
            self.pipes.append({"x": self._next_pipe_x, "gap_center": gap_center})
            self._next_pipe_x += self.pipe_spacing

        # ── Score: count passed pipes ─────────────────────────────────────────
        for pipe in self.pipes:
            pipe_right = pipe["x"] + self.pipe_width / 2.0
            if not pipe.get("counted", False) and pipe_right < self.bird_x:
                self.score += 1
                obstacle_passed = True
                pipe["counted"] = True

        # ── Collision detection ───────────────────────────────────────────────
        collision = False

        # Floor / ceiling
        if self.bird_y - self.bird_radius <= FLOOR_Y:
            self.bird_y = FLOOR_Y + self.bird_radius
            collision = True
        if self.bird_y + self.bird_radius >= CEILING_Y:
            self.bird_y = CEILING_Y - self.bird_radius
            collision = True

        # Pipe collision (AABB vs circle approximation)
        if not collision:
            for pipe in self.pipes:
                px = pipe["x"]
                gc = pipe["gap_center"]
                half_gap = self.gap_size / 2.0
                pipe_left = px - self.pipe_width / 2.0
                pipe_right = px + self.pipe_width / 2.0
                gap_top = gc + half_gap
                gap_bottom = gc - half_gap

                # Check if bird is within pipe horizontal extent
                if (self.bird_x + self.bird_radius > pipe_left) and (self.bird_x - self.bird_radius < pipe_right):
                    # Check if bird is outside the gap
                    if (self.bird_y + self.bird_radius > gap_top) or (self.bird_y - self.bird_radius < gap_bottom):
                        collision = True
                        break

        if collision:
            self.alive = False

        # ── Reward ────────────────────────────────────────────────────────────
        reward = 0.0
        if obstacle_passed:
            reward += 1.0
        if collision:
            reward -= 1.0

        # ── Done ──────────────────────────────────────────────────────────────
        done = collision or (self.step_count >= self.max_steps)

        info = {
            "score": self.score,
            "step": self.step_count,
            "collision": collision,
            "obstacle_passed": obstacle_passed,
            "bird_y": self.bird_y,
            "bird_vy": self.bird_vy,
        }

        return self.get_state(), reward, done, info

    def get_bird_state(self) -> Tuple[float, float]:
        """Return (bird_y, bird_vy) for diagnostic use only."""
        return self.bird_y, self.bird_vy


def summarize_physics() -> Dict[str, Any]:
    """Return calibrated physics parameters for documentation."""
    return {
        "gravity": GRAVITY,
        "flap_impulse": FLAP_IMPULSE,
        "horizontal_speed": HORIZONTAL_SPEED,
        "gap_size": GAP_SIZE,
        "pipe_spacing": PIPE_SPACING,
        "bird_x": BIRD_X,
        "bird_radius": BIRD_RADIUS,
        "pipe_width": PIPE_WIDTH,
        "world_height": WORLD_HEIGHT,
        "max_steps": MAX_STEPS,
    }
