"""
Phase 7: FlyMind RL — Live Visual Training Dashboard.

Shows actual gameplay + real-time learning telemetry.

Layout (1300 x 900 pygame window):
┌──────────────────────────────────┬─────────────────────────────────┐
│      GAMEPLAY (600x400)          │     RL STATUS PANEL (650x400)   │
│   bird + pipes + score HUD       │  Episode / Score / Reward       │
│   actual physics, not faked      │  Training progress bars         │
├──────────┬───────────────────────┼─────────────────────────────────┤
│  NEURAL  │  SENSORY              │  PLASTICITY PANEL               │
│  PANEL   │  9-channel grid       │  Weight stats / Eligibility     │
│ EPG ring │  PEG/PFNd/PFNv bars   │  Modified synapses count        │
├──────────┴───────────────────────┴─────────────────────────────────┤
│       LEARNING CURVE (episode score history, scrolling)            │
│       REWARD STRIP  (per-step reward, action timeline)             │
└────────────────────────────────────────────────────────────────────┘

Passivity guarantee:
  All visualization reads agent state AFTER act() has returned.
  Rendering never modifies weights, seeds, or action selection.

Usage:
  python experiments/phase7_flappy_rl_visual.py --render-training --episodes 500
  python experiments/phase7_flappy_rl_visual.py --render-eval --checkpoint <path>
  python experiments/phase7_flappy_rl_visual.py --compare --episodes 100
  python experiments/phase7_flappy_rl_visual.py --replay results/phase7/replays/ep_000200.npz
  python experiments/phase7_flappy_rl_visual.py --record --episodes 100
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from flymind.brain.network import NeuralNetwork
from flymind.connectome.loader import ConnectomeLoader
from flymind.environment.flappy import (
    BIRD_RADIUS, BIRD_X, CEILING_Y, FLOOR_Y, GAP_SIZE,
    HORIZONTAL_SPEED, PIPE_SPACING, PIPE_WIDTH, WORLD_HEIGHT, WORLD_WIDTH,
    FlappyEnvironment, FlappyState,
)
from flymind.environment.flappy_sensor import FlappyVisualSensor
from flymind.environment.flappy_reward import FlappyRewardShaper
from flymind.environment.curriculum import CurriculumManager
from flymind.agent.flappy_agent_phase7 import FlyMindRLAgent
from flymind.agent.flappy_agent import HandDesignedFlapAgent, RandomFlapAgent
from flymind.agent.flappy_agent_phase6b import ConnectomeMotorReadoutAgent

try:
    import pygame
    PYGAME_OK = True
except ImportError:
    PYGAME_OK = False
    pygame = None

try:
    import imageio
    IMAGEIO_OK = True
except ImportError:
    IMAGEIO_OK = False

# Paths
RESULTS_DIR = ROOT / "results" / "phase7"
CKPT_DIR    = RESULTS_DIR / "checkpoints"
VIDEO_DIR   = RESULTS_DIR / "videos"
REPLAY_DIR  = RESULTS_DIR / "replays"
for d in [RESULTS_DIR, CKPT_DIR, VIDEO_DIR, REPLAY_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CONNECTOME_PATH = ROOT / "data" / "processed" / "cx_heading_v1.json"

# ── Color Palette ─────────────────────────────────────────────────────────────
C = {
    "bg":          (10, 14, 22),
    "panel_bg":    (16, 22, 34),
    "card":        (22, 30, 46),
    "border":      (38, 50, 78),
    "text":        (230, 238, 255),
    "dim":         (110, 125, 160),
    "accent":      (0, 210, 255),
    "green":       (39, 210, 130),
    "red":         (240, 70, 70),
    "yellow":      (255, 205, 50),
    "purple":      (155, 90, 255),
    "orange":      (255, 145, 30),
    "sky_top":     (14, 22, 40),
    "sky_bot":     (22, 36, 60),
    "pipe":        (35, 160, 85),
    "pipe_dark":   (24, 115, 60),
    "pipe_cap":    (44, 195, 108),
    "bird":        (255, 200, 0),
    "bird_eye":    (255, 255, 255),
    "bird_pupil":  (20, 20, 20),
    "flap_mark":   (255, 70, 70),
    "epg_active":  (0, 220, 190),
    "epg_bg":      (30, 42, 60),
    "peg_color":   (155, 90, 255),
    "pfnd_color":  (255, 145, 30),
    "pfnv_color":  (39, 210, 130),
    "reward_pos":  (39, 210, 130),
    "reward_neg":  (240, 70, 70),
    "plasticity":  (255, 205, 50),
}

WIN_W, WIN_H = 1300, 900
GAME_W, GAME_H = 620, 400
PANEL_X = GAME_W + 20
PANEL_W = WIN_W - PANEL_X - 10
BOTTOM_Y = GAME_H + 20
BOTTOM_H = WIN_H - BOTTOM_Y - 10


# ── Connectome factory ────────────────────────────────────────────────────────
_CACHED_GRAPH = None
def get_graph():
    global _CACHED_GRAPH
    if _CACHED_GRAPH is None:
        _CACHED_GRAPH = ConnectomeLoader.load_from_json(CONNECTOME_PATH)
    return _CACHED_GRAPH

def make_network():
    return NeuralNetwork(get_graph(), synapse_scale=0.01)

def make_rl_agent(seed: int, checkpoint: Optional[str] = None) -> FlyMindRLAgent:
    net = make_network()
    agent = FlyMindRLAgent(
        network=net, sensory_drive=25.0, sub_steps=10,
        learning_rate=0.002, eligibility_decay=0.90,
        plasticity_mode="pathway", motor_temperature=1.5,
        motor_lr=0.01, decoder_mode="stochastic", seed=seed,
    )
    if checkpoint and Path(checkpoint).exists():
        agent.load_checkpoint(checkpoint)
        print(f"[Loaded checkpoint] {checkpoint}")
    return agent


# ── Drawing utilities ─────────────────────────────────────────────────────────
def draw_rect_aa(surf, color, rect, border_radius=4):
    pygame.draw.rect(surf, color, rect, border_radius=border_radius)

def draw_text(surf, text, x, y, font, color=None, anchor="topleft"):
    color = color or C["text"]
    img = font.render(text, True, color)
    r = img.get_rect()
    setattr(r, anchor, (x, y))
    surf.blit(img, r)

def lerp_color(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))

def draw_bar(surf, x, y, w, h, value, max_val, color, bg_color=None):
    bg = bg_color or C["card"]
    pygame.draw.rect(surf, bg, (x, y, w, h), border_radius=3)
    fill_w = int(w * max(0.0, min(1.0, value / max(max_val, 1e-6))))
    if fill_w > 0:
        pygame.draw.rect(surf, color, (x, y, fill_w, h), border_radius=3)


# ── Game renderer (same visual quality as Phase 6C) ──────────────────────────
class GameRenderer:
    """Renders the Flappy Bird game world to a pygame surface."""

    def __init__(self, width: int, height: int, scale_x: float, scale_y: float):
        self.surf   = pygame.Surface((width, height))
        self.w      = width
        self.h      = height
        self.sx     = scale_x   # world -> screen scale
        self.sy     = scale_y

    def _world_to_screen(self, wx: float, wy: float) -> Tuple[int, int]:
        sx = int(wx * self.sx)
        sy = int((WORLD_HEIGHT - wy) * self.sy)
        return sx, sy

    def draw(self, state: FlappyState, action: int, flap_prob: float, score: int, step: int):
        s = self.surf
        # Sky gradient
        for row in range(self.h):
            t   = row / self.h
            col = lerp_color(C["sky_top"], C["sky_bot"], t)
            pygame.draw.line(s, col, (0, row), (self.w, row))

        # Ground
        gy = int((WORLD_HEIGHT - FLOOR_Y) * self.sy)
        pygame.draw.rect(s, C["panel_bg"], (0, self.h - 18, self.w, 18))
        pygame.draw.line(s, C["border"], (0, self.h - 18), (self.w, self.h - 18), 1)

        # Pipes
        for pipe in state.pipes:
            px, gc = pipe["x"], pipe["gap_center"]
            half_gap = GAP_SIZE / 2.0
            gap_top    = gc + half_gap
            gap_bottom = gc - half_gap

            scr_x = int(pipe["x"] * self.sx)
            pw    = max(1, int(PIPE_WIDTH * self.sx))

            # Bottom pipe
            bot_top = int((WORLD_HEIGHT - gap_bottom) * self.sy)
            pygame.draw.rect(s, C["pipe"], (scr_x - pw // 2, bot_top, pw, self.h - bot_top))
            pygame.draw.rect(s, C["pipe_cap"], (scr_x - pw // 2 - 3, bot_top - 8, pw + 6, 8))
            # Pipe shadow
            pygame.draw.rect(s, C["pipe_dark"], (scr_x - pw // 2, bot_top, 4, self.h - bot_top))

            # Top pipe
            top_bot = int((WORLD_HEIGHT - gap_top) * self.sy)
            pygame.draw.rect(s, C["pipe"], (scr_x - pw // 2, 0, pw, max(0, top_bot)))
            pygame.draw.rect(s, C["pipe_cap"], (scr_x - pw // 2 - 3, max(0, top_bot), pw + 6, 8))
            pygame.draw.rect(s, C["pipe_dark"], (scr_x - pw // 2, 0, 4, max(0, top_bot)))

            # Gap indicator line
            gap_center_y = int((WORLD_HEIGHT - gc) * self.sy)
            pygame.draw.line(s, (0, 180, 255, 30), (scr_x - pw // 2, gap_center_y), (scr_x + pw // 2, gap_center_y), 1)

        # Bird
        bx, by = self._world_to_screen(BIRD_X, state.bird_y)
        r = max(4, int(BIRD_RADIUS * self.sx))

        # Wing flap animation
        wing_offset = -int(r * 0.6 * math.sin(step * 0.3)) if action == 1 else int(r * 0.3)
        pygame.draw.ellipse(s, C["bird"], (bx - r + 3, by - r + 3 + wing_offset, r * 2 - 4, int(r * 0.6)))

        # Body
        pygame.draw.circle(s, C["bird"], (bx, by), r)
        pygame.draw.circle(s, (255, 170, 0), (bx, by), r, 2)

        # Eye
        pygame.draw.circle(s, C["bird_eye"], (bx + r // 2, by - r // 3), r // 3)
        pygame.draw.circle(s, C["bird_pupil"], (bx + r // 2 + 1, by - r // 3 + 1), r // 5)

        # Flap indicator
        if action == 1:
            pygame.draw.circle(s, C["flap_mark"], (bx, by), r + 4, 2)

        # HUD overlay
        return s


# ── EPG Ring Renderer ─────────────────────────────────────────────────────────
def draw_epg_ring(surf, cx, cy, radius, epg_activity, font_small):
    """Draw the 16-glomeruli EPG compass ring."""
    n = len(epg_activity)
    for i, act in enumerate(epg_activity):
        angle = (i / n) * 2 * math.pi - math.pi / 2
        gx = cx + int(radius * math.cos(angle))
        gy = cy + int(radius * math.sin(angle))
        r_dot = max(3, int(8 * act))
        t = min(1.0, act * 5.0)
        col = lerp_color(C["epg_bg"], C["epg_active"], t)
        pygame.draw.circle(surf, col, (gx, gy), r_dot)
    # Ring outline
    pygame.draw.circle(surf, C["border"], (cx, cy), radius, 1)


# ── Score history plot ────────────────────────────────────────────────────────
def draw_learning_curve(surf, rect, scores, ma_window=50, font=None):
    x, y, w, h = rect
    pygame.draw.rect(surf, C["card"], rect, border_radius=4)
    if not scores:
        return
    n = len(scores)
    max_s = max(max(scores), 1)
    # Raw scores (dim)
    step_w = w / max(n, 1)
    for i, s in enumerate(scores):
        sx = x + int(i * step_w)
        sy = y + h - int((s / max_s) * (h - 4))
        if i > 0:
            sx_prev = x + int((i - 1) * step_w)
            sy_prev = y + h - int((scores[i-1] / max_s) * (h - 4))
            pygame.draw.line(surf, (60, 80, 120), (sx_prev, sy_prev), (sx, sy), 1)

    # Moving average
    if n >= ma_window:
        ma = [np.mean(scores[max(0, i - ma_window):i + 1]) for i in range(n)]
        for i in range(1, n):
            sx1 = x + int((i - 1) * step_w)
            sy1 = y + h - int((ma[i-1] / max_s) * (h - 4))
            sx2 = x + int(i * step_w)
            sy2 = y + h - int((ma[i] / max_s) * (h - 4))
            pygame.draw.line(surf, C["accent"], (sx1, sy1), (sx2, sy2), 2)

    # Latest value indicator
    if n > 0:
        last_y = y + h - int((scores[-1] / max_s) * (h - 4))
        pygame.draw.circle(surf, C["yellow"], (x + w - 2, last_y), 4)


# ── Main Visual RL Engine ────────────────────────────────────────────────────
class VisualRLEngine:
    """
    Live visual RL training dashboard.

    Renders actual game + all telemetry panels. All reads occur AFTER act().
    Rendering never modifies agent state (passivity guarantee preserved).
    """

    def __init__(
        self,
        agent: FlyMindRLAgent,
        seed: int = 42,
        fps: int = 30,
        max_episodes: int = 500,
        max_steps_per_ep: int = 5000,
        training: bool = True,
        record: bool = False,
        video_path: Optional[str] = None,
        headless: bool = False,
        sim_steps_per_frame: int = 1,
        curriculum: Optional[CurriculumManager] = None,
        checkpoint_dir: Optional[str] = None,
        checkpoint_interval: int = 100,
        compare_agent = None,
        compare_name: str = "Hand-Designed",
        shaper: Optional[FlappyRewardShaper] = None,
    ):
        self.agent              = agent
        self.seed               = seed
        self.fps                = fps
        self.max_episodes       = max_episodes
        self.max_steps          = max_steps_per_ep
        self.training           = training
        self.record             = record
        self.headless           = headless
        self.sim_steps_per_frame = sim_steps_per_frame
        self.curriculum         = curriculum or CurriculumManager(enabled=False)
        self.checkpoint_dir     = Path(checkpoint_dir) if checkpoint_dir else CKPT_DIR
        self.checkpoint_interval = checkpoint_interval
        self.compare_agent      = compare_agent
        self.compare_name       = compare_name
        self.shaper             = shaper or FlappyRewardShaper()

        if PYGAME_OK and not headless:
            pygame.init()
            self.screen = pygame.display.set_mode((WIN_W, WIN_H))
            pygame.display.set_caption("FlyMind Phase 7 — RL Training Dashboard")
            self.clock = pygame.time.Clock()
            self.font_lg  = pygame.font.SysFont("Consolas", 16, bold=True)
            self.font_md  = pygame.font.SysFont("Consolas", 13)
            self.font_sm  = pygame.font.SysFont("Consolas", 11)
            self.font_hd  = pygame.font.SysFont("Consolas", 20, bold=True)

            scale_x = GAME_W / WORLD_WIDTH
            scale_y = GAME_H / WORLD_HEIGHT
            self.game_renderer = GameRenderer(GAME_W, GAME_H, scale_x, scale_y)

        # Video writer
        self.video_writer = None
        if record and IMAGEIO_OK:
            vp = video_path or str(VIDEO_DIR / f"phase7_training_{int(time.time())}.mp4")
            self.video_writer = imageio.get_writer(vp, fps=fps, codec="libx264", quality=7)
            print(f"[Record] Writing video to {vp}")

        # Score history
        self.score_history:    List[float] = []
        self.survival_history: List[float] = []
        self.reward_history:   List[float] = []
        self.reward_strip:     deque       = deque(maxlen=200)
        self.action_strip:     deque       = deque(maxlen=200)
        self.best_score        = 0
        self.best_survival     = 0
        self.episode_count     = 0
        self.paused            = False

    def run(self) -> None:
        """Main training/rendering loop."""
        run_start = time.time()

        for ep in range(1, self.max_episodes + 1):
            self.episode_count = ep
            ep_seed = self.seed + ep

            env_kwargs = self.curriculum.get_env_kwargs(ep)
            env = FlappyEnvironment(seed=ep_seed, max_steps=self.max_steps, **env_kwargs)
            state = env.reset(seed=ep_seed)
            self.agent.reset()

            if self.compare_agent:
                compare_env = FlappyEnvironment(seed=ep_seed, max_steps=self.max_steps, **env_kwargs)
                compare_state = compare_env.reset(seed=ep_seed)
                self.compare_agent.reset()
                compare_score = 0

            total_reward = 0.0
            step_count   = 0
            done         = False
            diag: Dict[str, Any] = {}

            while not done and step_count < self.max_steps:
                # Handle events
                if PYGAME_OK and not self.headless:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            self._cleanup()
                            return
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_ESCAPE:
                                self._cleanup()
                                return
                            if event.key == pygame.K_SPACE:
                                self.paused = not self.paused
                            if event.key == pygame.K_r:
                                done = True  # restart episode

                if self.paused and not self.headless:
                    pygame.time.wait(50)
                    continue

                # Run sim_steps_per_frame simulation steps
                for _ in range(self.sim_steps_per_frame):
                    if done:
                        break
                    action = self.agent.act(state)
                    diag = self.agent.get_diagnostics()   # read-only — passivity preserved
                    next_state, env_reward, done, info = env.step(action)
                    shaped_r = self.shaper.shape(env_reward, info, done)
                    total_reward += shaped_r
                    if self.training:
                        self.agent.apply_step_reward(shaped_r)
                    self.reward_strip.append(shaped_r)
                    self.action_strip.append(action)
                    state = next_state
                    step_count += 1

                # Compare agent step
                if self.compare_agent and not done:
                    c_action = self.compare_agent.act(compare_state)
                    compare_state, _, c_done, _ = compare_env.step(c_action)
                    compare_score = compare_state.score
                    if c_done:
                        compare_state = compare_env.reset(seed=ep_seed)

                # Render
                if PYGAME_OK and not self.headless:
                    self._render(state, action, diag, ep, step_count, total_reward,
                                 compare_score if self.compare_agent else None)
                    if self.fps > 0:
                        self.clock.tick(self.fps)

            # Episode end
            if self.training:
                ep_stats = self.agent.end_episode(total_reward)
            self.curriculum.record_episode(state.score)

            self.score_history.append(state.score)
            self.survival_history.append(step_count)
            self.reward_history.append(total_reward)
            if state.score > self.best_score:
                self.best_score = state.score
            if step_count > self.best_survival:
                self.best_survival = step_count

            # Checkpoint
            if ep % self.checkpoint_interval == 0:
                ckpt = str(self.checkpoint_dir / f"visual_ep{ep:06d}.npz")
                self.agent.save_checkpoint(ckpt)
                n = len(self.score_history)
                mean50 = float(np.mean(self.score_history[-50:])) if n >= 50 else float(np.mean(self.score_history))
                print(f"  [ep={ep:>5d}] score={state.score} | mean50={mean50:.2f} | "
                      f"best={self.best_score} | survival={step_count} | "
                      f"stage={self.curriculum.stage_name}")

        self._cleanup()
        print(f"\nTraining complete: {self.max_episodes} episodes")
        print(f"  Mean score (all):  {np.mean(self.score_history):.3f}")
        print(f"  Mean score (last 100): {np.mean(self.score_history[-100:]):.3f}")
        print(f"  Best score: {self.best_score}")

    def _render(self, state: FlappyState, action: int, diag: Dict,
                episode: int, step: int, ep_reward: float, compare_score=None):
        """Render all dashboard panels. Read-only from agent state."""
        screen = self.screen
        screen.fill(C["bg"])

        flap_prob   = diag.get("flap_prob", 0.5)
        epg_ordered = self.agent.get_epg_ordered_activity()
        peg_act     = diag.get("peg_activity", np.zeros(1))
        pfnd_act    = diag.get("pfnd_activity", np.zeros(1))
        pfnv_act    = diag.get("pfnv_activity", np.zeros(1))
        sensor_act  = diag.get("sensor_activations", np.zeros(9))
        et_mean     = diag.get("eligibility_mean", 0.0)
        W_mean      = diag.get("weight_mean", 0.0)
        W_std       = diag.get("weight_std", 0.0)
        n_plastic   = diag.get("n_plastic_synapses", 0)
        dW_mean     = diag.get("delta_weight_mean", 0.0)
        motor_logit = diag.get("motor_logit", 0.0)
        n_ep        = len(self.score_history)
        mean_s50    = float(np.mean(self.score_history[-50:])) if n_ep >= 1 else 0.0

        # ── Panel: Gameplay ────────────────────────────────────────────────────
        game_surf = self.game_renderer.draw(state, action, flap_prob, state.score, step)
        screen.blit(game_surf, (10, 10))
        pygame.draw.rect(screen, C["border"], (10, 10, GAME_W, GAME_H), 1)

        # Score overlay on game
        draw_text(screen, f"Score: {state.score}", 20, 18, self.font_lg, C["accent"])
        draw_text(screen, f"Best: {self.best_score}", 120, 18, self.font_lg, C["yellow"])
        if compare_score is not None:
            draw_text(screen, f"{self.compare_name}: {compare_score}", 250, 18, self.font_lg, C["green"])

        # ── Panel: RL Status ──────────────────────────────────────────────────
        rx, ry, rw = PANEL_X, 10, PANEL_W
        pygame.draw.rect(screen, C["panel_bg"], (rx, ry, rw, 200), border_radius=6)
        pygame.draw.rect(screen, C["border"], (rx, ry, rw, 200), 1, border_radius=6)

        draw_text(screen, "── FLYMIND RL DASHBOARD ──", rx + 10, ry + 8, self.font_lg, C["accent"])
        y_off = ry + 30
        rl_lines = [
            (f"Episode:      {episode:>6d} / {self.max_episodes}", C["text"]),
            (f"Step:         {step:>6d}", C["dim"]),
            (f"Score:        {state.score:>6d}   Best: {self.best_score}", C["yellow"]),
            (f"Ep Reward:    {ep_reward:>+8.3f}", C["green"] if ep_reward >= 0 else C["red"]),
            (f"Mean Score50: {mean_s50:>8.3f}", C["accent"]),
            (f"Best Surv:    {self.best_survival:>6d}", C["dim"]),
            (f"Stage:        {self.curriculum.stage_name}", C["purple"]),
            (f"P(flap):      {flap_prob:>8.3f}", C["orange"]),
            (f"Motor logit:  {motor_logit:>+8.3f}", C["dim"]),
        ]
        for txt, col in rl_lines:
            draw_text(screen, txt, rx + 14, y_off, self.font_md, col)
            y_off += 18

        # P(flap) bar
        draw_text(screen, "P(flap)", rx + 14, y_off, self.font_sm, C["dim"])
        draw_bar(screen, rx + 80, y_off + 2, rw - 100, 12, flap_prob, 1.0, C["orange"])
        y_off += 18

        # ── Panel: Neural Activity ────────────────────────────────────────────
        ny, nh = BOTTOM_Y, BOTTOM_H
        nx = 10

        # EPG compass ring
        ring_cx, ring_cy = nx + 90, ny + 70
        ring_r = 55
        pygame.draw.rect(screen, C["panel_bg"], (nx, ny, 195, 150), border_radius=6)
        pygame.draw.rect(screen, C["border"], (nx, ny, 195, 150), 1, border_radius=6)
        draw_text(screen, "EPG Compass", nx + 10, ny + 5, self.font_sm, C["accent"])
        draw_epg_ring(screen, ring_cx, ring_cy, ring_r, epg_ordered, self.font_sm)

        # Population bars: PEG, PFNd, PFNv
        pop_x = nx + 205
        pop_y = ny + 5
        pop_w = 220
        pops = [
            ("PEG",  float(np.mean(peg_act)),  C["peg_color"]),
            ("PFNd", float(np.mean(pfnd_act)), C["pfnd_color"]),
            ("PFNv", float(np.mean(pfnv_act)), C["pfnv_color"]),
        ]
        pygame.draw.rect(screen, C["panel_bg"], (pop_x - 5, ny, pop_w + 10, 150), border_radius=6)
        pygame.draw.rect(screen, C["border"], (pop_x - 5, ny, pop_w + 10, 150), 1, border_radius=6)
        draw_text(screen, "Motor Populations", pop_x, ny + 5, self.font_sm, C["accent"])
        by_ = ny + 22
        for name, val, col in pops:
            draw_text(screen, f"{name:5s} {val:.3f}", pop_x, by_, self.font_sm, col)
            draw_bar(screen, pop_x + 80, by_ + 2, pop_w - 85, 10, val, 0.5, col)
            by_ += 20

        # Sensor grid (3x3)
        sg_x = pop_x + pop_w + 15
        sg_y = ny + 5
        sg_cell = 22
        pygame.draw.rect(screen, C["panel_bg"], (sg_x - 5, ny, 90, 85), border_radius=6)
        draw_text(screen, "Sensor", sg_x, ny + 5, self.font_sm, C["accent"])
        row_labels = ["Up", "Ctr", "Dn"]
        col_labels = ["Far", "Mid", "Cls"]
        for row in range(3):
            for col in range(3):
                ch = row * 3 + col
                val = float(sensor_act[ch]) if ch < len(sensor_act) else 0.0
                t = min(1.0, val * 2.0)
                cell_col = lerp_color((30, 38, 55), C["yellow"], t)
                cx_ = sg_x + col * sg_cell
                cy_ = sg_y + 18 + row * sg_cell
                pygame.draw.rect(screen, cell_col, (cx_, cy_, sg_cell - 2, sg_cell - 2), border_radius=3)
                if val > 0.05:
                    draw_text(screen, f"{val:.1f}", cx_ + 2, cy_ + 4, self.font_sm, C["text"])

        # ── Panel: Plasticity ─────────────────────────────────────────────────
        plx = sg_x + 100
        ply = ny
        plw = 240
        plh = 150
        pygame.draw.rect(screen, C["panel_bg"], (plx, ply, plw, plh), border_radius=6)
        pygame.draw.rect(screen, C["border"], (plx, ply, plw, plh), 1, border_radius=6)
        draw_text(screen, "── PLASTICITY ──", plx + 10, ply + 5, self.font_sm, C["plasticity"])
        pl_lines = [
            f"Plastic synapses: {n_plastic}",
            f"Weight mean:  {W_mean:+.4f}",
            f"Weight std:   {W_std:.4f}",
            f"ΔW mean:      {dW_mean:+.6f}",
            f"Eligibility:  {et_mean:.6f}",
            f"W_motor: [{', '.join(f'{w:.3f}' for w in self.agent.W_motor)}]",
            f"b_motor: {self.agent.b_motor:+.3f}",
        ]
        for i, txt in enumerate(pl_lines):
            draw_text(screen, txt, plx + 10, ply + 22 + i * 16, self.font_sm, C["dim"])

        # ── Panel: Learning curve ─────────────────────────────────────────────
        lc_x = plx + plw + 15
        lc_y = ny
        lc_w = WIN_W - lc_x - 10
        lc_h = 150
        draw_learning_curve(screen, (lc_x, lc_y, lc_w, lc_h), self.score_history, font=self.font_sm)
        draw_text(screen, "Learning Curve (score per episode)", lc_x + 4, lc_y + 4, self.font_sm, C["accent"])

        # ── Reward / Action strip ─────────────────────────────────────────────
        strip_y = ny + 155
        strip_h = WIN_H - strip_y - 5
        pygame.draw.rect(screen, C["card"], (10, strip_y, WIN_W - 20, strip_h), border_radius=4)
        draw_text(screen, "Step Reward (green=+, red=-) | Action strip (red=FLAP)", 15, strip_y + 2, self.font_sm, C["dim"])
        n_strip = len(self.reward_strip)
        strip_x_step = max(1, (WIN_W - 20) // max(n_strip, 1))
        for i, (r_val, a_val) in enumerate(zip(self.reward_strip, self.action_strip)):
            bx_ = 10 + i * strip_x_step
            if r_val >= 0:
                col = C["reward_pos"]
                bar_h = min(int(r_val * strip_h * 0.5), strip_h - 4)
                pygame.draw.rect(screen, col, (bx_, strip_y + strip_h - bar_h - 2, max(strip_x_step - 1, 1), bar_h))
            else:
                col = C["reward_neg"]
                bar_h = min(int(abs(r_val) * strip_h * 0.5), strip_h - 4)
                pygame.draw.rect(screen, col, (bx_, strip_y + 2, max(strip_x_step - 1, 1), bar_h))
            if a_val == 1:
                pygame.draw.line(screen, C["flap_mark"], (bx_, strip_y + 2), (bx_, strip_y + strip_h - 2), 1)

        # Training/paused indicator
        mode_str = "TRAINING" if self.training else "EVALUATING"
        if self.paused:
            mode_str = "PAUSED (SPACE to resume)"
        draw_text(screen, mode_str, PANEL_X, WIN_H - 18, self.font_sm, C["accent"])

        pygame.display.flip()

        # Record frame
        if self.video_writer:
            frame = pygame.surfarray.array3d(screen)
            frame = np.transpose(frame, (1, 0, 2))
            self.video_writer.append_data(frame)

    def _cleanup(self):
        if self.video_writer:
            self.video_writer.close()
            print("[Record] Video saved.")
        if PYGAME_OK and not self.headless:
            pygame.quit()


# ── Replay visualizer ─────────────────────────────────────────────────────────
def run_replay(replay_path: str, fps: int = 30) -> None:
    """Load and visualize a saved Phase 7 replay file."""
    if not PYGAME_OK:
        print("pygame not available — cannot replay visually.")
        return
    data = np.load(replay_path)
    bird_y   = data["bird_y"]
    bird_vy  = data["bird_vy"] if "bird_vy" in data else np.zeros_like(bird_y)
    actions  = data["actions"]
    rewards  = data["rewards"] if "rewards" in data else np.zeros_like(bird_y)
    scores   = data["scores"]  if "scores"  in data else np.zeros(len(bird_y), dtype=int)
    n_steps  = len(bird_y)
    print(f"[Replay] {Path(replay_path).name} | {n_steps} steps | Max score: {scores[-1] if len(scores) else 0}")

    pygame.init()
    screen = pygame.display.set_mode((WIN_W // 2, GAME_H + 100))
    pygame.display.set_caption(f"Phase 7 Replay — {Path(replay_path).name}")
    clock = pygame.time.Clock()
    font  = pygame.font.SysFont("Consolas", 14)

    scale_x = (WIN_W // 2) / WORLD_WIDTH
    scale_y = GAME_H / WORLD_HEIGHT
    renderer = GameRenderer(WIN_W // 2, GAME_H, scale_x, scale_y)

    for t in range(n_steps):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.quit()
                return
        # Rebuild a minimal state for rendering
        state = FlappyState(
            bird_y=float(bird_y[t]),
            bird_vy=float(bird_vy[t]),
            pipes=[],
            score=int(scores[t]),
            step_count=t,
            alive=True,
        )
        surf = renderer.draw(state, int(actions[t]), 0.5, int(scores[t]), t)
        screen.fill(C["bg"])
        screen.blit(surf, (0, 0))
        # Reward bar
        r = float(rewards[t])
        draw_text(screen, f"Step {t+1}/{n_steps} | Score {int(scores[t])} | Reward {r:+.3f}", 10, GAME_H + 5, font, C["accent"])
        pygame.display.flip()
        clock.tick(fps)

    pygame.time.wait(1000)
    pygame.quit()


# ── CLI ───────────────────────────────────────────────────────────────────────
def build_parser():
    p = argparse.ArgumentParser(description="Phase 7: FlyMind RL Visual Dashboard")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--render-training", action="store_true", help="Live visual RL training")
    mode.add_argument("--render-eval",     action="store_true", help="Visual evaluation of a checkpoint")
    mode.add_argument("--compare",         action="store_true", help="Visual comparison FlyMind vs Hand-designed")
    mode.add_argument("--replay",          type=str, default=None, help="Path to replay .npz file to visualize")

    p.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path")
    p.add_argument("--episodes",   type=int, default=500,  help="Total episodes")
    p.add_argument("--max-steps",  type=int, default=5000, help="Max steps per episode")
    p.add_argument("--seed",       type=int, default=42,   help="Base random seed")
    p.add_argument("--fps",        type=int, default=30,   help="Framerate (0=unlimited)")
    p.add_argument("--record",     action="store_true",    help="Record MP4 video")
    p.add_argument("--headless",   action="store_true",    help="Run without display (headless training)")
    p.add_argument("--sim-steps-per-frame", type=int, default=1, help="Sim steps per rendered frame")
    p.add_argument("--curriculum", action="store_true",    help="Enable curriculum")
    p.add_argument("--checkpoint-interval", type=int, default=100, help="Save checkpoint every N episodes")
    return p


def main():
    args = build_parser().parse_args()

    if args.replay:
        run_replay(args.replay, fps=args.fps or 30)
        return

    agent = make_rl_agent(args.seed, checkpoint=args.checkpoint)
    compare_agent = None
    compare_name  = ""

    if args.compare:
        compare_agent = HandDesignedFlapAgent()
        compare_name  = "Hand-Designed"

    curriculum = CurriculumManager(enabled=args.curriculum)
    training   = args.render_training or (not args.render_eval and not args.compare)

    engine = VisualRLEngine(
        agent=agent,
        seed=args.seed,
        fps=args.fps,
        max_episodes=args.episodes,
        max_steps_per_ep=args.max_steps,
        training=training,
        record=args.record,
        headless=args.headless,
        sim_steps_per_frame=args.sim_steps_per_frame,
        curriculum=curriculum,
        checkpoint_interval=args.checkpoint_interval,
        compare_agent=compare_agent,
        compare_name=compare_name,
    )
    engine.run()


if __name__ == "__main__":
    main()
