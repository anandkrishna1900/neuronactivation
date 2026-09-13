"""
Phase 6A: Biologically inspired visual sensor for Flappy Bird environment.

Version 3: Clean gap-position sensor with non-overlapping rows.

The sensor detects WHERE the gap is relative to the bird:
- If gap is ABOVE bird -> upper channels activate
- If gap is AROUND bird -> center channels activate
- If gap is BELOW bird -> lower channels activate

Each column represents a horizontal distance band:
- far (>150px): pipe is distant
- mid (50-150px): pipe is approaching
- close (<50px): pipe is imminent
"""

from typing import Optional
import numpy as np

from flymind.environment.flappy import FlappyState, PIPE_WIDTH


N_CHANNELS = 9
ROW_LABELS = ["upper", "center", "lower"]
COL_LABELS = ["far", "mid", "close"]


class FlappyVisualSensor:
    """
    Gap-position sensor with clean non-overlapping receptive fields.

    Divides the vertical space around the bird into 3 non-overlapping bands.
    For each band, measures whether the gap opening overlaps with that band.
    """

    def __init__(
        self,
        band_half_height: float = 60.0,
        col_boundaries: Optional[list] = None,
    ):
        """
        Args:
            band_half_height: Half-height of the center band.
                              Upper band: [bird_y + band_half_height, +inf)
                              Center band: [bird_y - band_half_height, bird_y + band_half_height]
                              Lower band: (-inf, bird_y - band_half_height]
            col_boundaries: Horizontal distance boundaries for columns.
        """
        self.band_half_height = band_half_height
        self.col_boundaries = col_boundaries or [50.0, 150.0]

    def sense(self, state: FlappyState) -> np.ndarray:
        """Encode the gap position relative to the bird into 9 channels."""
        bird_y = state.bird_y
        activations = np.zeros(N_CHANNELS, dtype=np.float64)

        for pipe in state.pipes:
            px = pipe["x"]
            gc = pipe["gap_center"]

            dx = px - 80.0  # bird_x = 80
            if dx < -PIPE_WIDTH:
                continue

            # Column
            if dx < self.col_boundaries[0]:
                col = 2  # close
            elif dx < self.col_boundaries[1]:
                col = 1  # mid
            else:
                col = 0  # far

            # Gap extent
            half_gap = 80.0  # half of GAP_SIZE (160)
            gap_bottom = gc - half_gap
            gap_top = gc + half_gap

            # Three non-overlapping vertical bands
            bands = [
                (bird_y + self.band_half_height, 1e6),   # upper: above bird
                (bird_y - self.band_half_height, bird_y + self.band_half_height),  # center: around bird
                (-1e6, bird_y - self.band_half_height),   # lower: below bird
            ]

            for row, (band_bottom, band_top) in enumerate(bands):
                # Overlap between gap and this band
                overlap_bottom = max(band_bottom, gap_bottom)
                overlap_top = min(band_top, gap_top)
                overlap = max(0.0, overlap_top - overlap_bottom)

                band_height = band_top - band_bottom
                if band_height <= 0:
                    continue

                # Activation = fraction of band that is gap
                gap_fraction = overlap / band_height

                # Boost if gap center is within this band
                if band_bottom <= gc <= band_top:
                    gap_fraction = max(gap_fraction, 0.8)

                channel = row * 3 + col
                activations[channel] = max(activations[channel], gap_fraction)

        activations = np.clip(activations, 0.0, 1.0)
        return activations

    def get_channel_labels(self):
        labels = []
        for r, row in enumerate(ROW_LABELS):
            for c, col in enumerate(COL_LABELS):
                labels.append(f"{row}-{col}")
        return labels
