"""
Curriculum Learning Manager for Phase 7 Flappy Bird RL.

Implements 5 difficulty stages with automatic progression based on
measured agent performance (moving-average score).

Stage 1: Large gaps, slow scroll — easy
Stage 2: Normal gaps, normal scroll — standard
Stage 3: Smaller gaps, faster scroll — hard
Stage 4: Randomized difficulty — mixed
Stage 5: Unseen test distribution — held-out

Progression is based on measurable performance thresholds,
NOT on visual inspection or manual intervention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import numpy as np


@dataclass
class CurriculumStage:
    """Physics parameters for a single curriculum stage."""
    name: str
    gap_size: float          # pixels (default 160)
    horizontal_speed: float  # pixels/frame (default 1.5)
    pipe_spacing: float      # pixels (default 200)
    advance_threshold: float  # moving-average score to advance
    window: int = 50          # episodes to average over for advancement


CURRICULUM_STAGES: List[CurriculumStage] = [
    CurriculumStage(
        name="Stage1_EasyLarge",
        gap_size=220.0,
        horizontal_speed=1.0,
        pipe_spacing=240.0,
        advance_threshold=2.0,
        window=30,
    ),
    CurriculumStage(
        name="Stage2_Normal",
        gap_size=160.0,
        horizontal_speed=1.5,
        pipe_spacing=200.0,
        advance_threshold=1.5,
        window=50,
    ),
    CurriculumStage(
        name="Stage3_Harder",
        gap_size=130.0,
        horizontal_speed=1.8,
        pipe_spacing=180.0,
        advance_threshold=1.0,
        window=80,
    ),
    CurriculumStage(
        name="Stage4_Randomized",
        gap_size=145.0,     # mean, randomized per episode
        horizontal_speed=1.6,
        pipe_spacing=190.0,
        advance_threshold=0.8,
        window=100,
    ),
    CurriculumStage(
        name="Stage5_Unseen",
        gap_size=140.0,
        horizontal_speed=1.9,
        pipe_spacing=175.0,
        advance_threshold=float("inf"),  # never auto-advance — final stage
        window=100,
    ),
]


class CurriculumManager:
    """
    Manages automatic curriculum progression for Phase 7 RL training.

    Tracks a rolling score window and advances to the next stage when
    the moving-average score exceeds the stage threshold.
    """

    def __init__(
        self,
        enabled: bool = True,
        start_stage: int = 0,
        stages: Optional[List[CurriculumStage]] = None,
        randomize_stage4: bool = True,
        seed: Optional[int] = None,
    ):
        self.enabled = enabled
        self.stages  = stages or CURRICULUM_STAGES
        self.current_stage_idx = max(0, min(start_stage, len(self.stages) - 1))
        self.randomize_stage4  = randomize_stage4
        self._rng = np.random.default_rng(seed)
        self._score_history: List[float] = []

    @property
    def current_stage(self) -> CurriculumStage:
        return self.stages[self.current_stage_idx]

    @property
    def stage_name(self) -> str:
        return self.current_stage.name

    def get_env_kwargs(self, episode: int = 0) -> Dict[str, Any]:
        """Return FlappyEnvironment constructor kwargs for the current stage."""
        stage = self.current_stage
        gap = stage.gap_size
        # Randomize gap ±15% in Stage 4
        if self.randomize_stage4 and stage.name.startswith("Stage4"):
            gap = float(self._rng.uniform(gap * 0.85, gap * 1.15))
        return {
            "gap_size": gap,
            "horizontal_speed": stage.horizontal_speed,
            "pipe_spacing": stage.pipe_spacing,
        }

    def record_episode(self, score: float) -> bool:
        """
        Record episode score and check for stage advancement.

        Returns:
            True if stage advanced, False otherwise.
        """
        if not self.enabled:
            return False

        self._score_history.append(score)
        stage = self.current_stage
        if len(self._score_history) < stage.window:
            return False

        recent = self._score_history[-stage.window:]
        moving_avg = float(np.mean(recent))

        if (self.current_stage_idx < len(self.stages) - 1
                and moving_avg >= stage.advance_threshold):
            self.current_stage_idx += 1
            return True
        return False

    def moving_average_score(self) -> float:
        stage = self.current_stage
        w = min(len(self._score_history), stage.window)
        if w == 0:
            return 0.0
        return float(np.mean(self._score_history[-w:]))

    def summary(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "stage_idx": self.current_stage_idx,
            "stage_name": self.stage_name,
            "total_episodes": len(self._score_history),
            "moving_avg_score": self.moving_average_score(),
        }
