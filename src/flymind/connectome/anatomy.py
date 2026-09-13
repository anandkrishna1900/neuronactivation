"""
Extract and parse anatomical glomerulus columns (L1..L8, R1..R8) for EPG, PEN, and PEG neurons.
Constructs ordered ring angles in [0, 2*pi) for faithful ring attractor dynamics.
"""

import re
from typing import Dict, Optional, Tuple
import numpy as np


def parse_cx_glomerulus(instance_name: Optional[str]) -> Optional[Tuple[str, int]]:
    """
    Parses instance string like 'EPG(PB08)_L3' or 'PEG(PB07)_R4' or 'PEN_a(PB06a)_L2'.
    Returns ('L'|'R', glomerulus_num: 1..8) or None.
    """
    if not instance_name:
        return None
    
    match = re.search(r'_(L|R)(\d+)', instance_name)
    if match:
        side = match.group(1)
        col = int(match.group(2))
        return side, col
    return None


def glomerulus_to_angle(side: str, col: int) -> float:
    """
    Maps PB glomeruli L1..L8 and R1..R8 to an angular position in [0, 2*pi).
    PB has 16-18 glomeruli spanning 360 degrees of azimuthal space.
    Standard mapping:
    R8..R1 -> [0, pi)
    L1..L8 -> [pi, 2*pi)
    """
    if side == 'R':
        # R8 is near 0, R1 is near pi
        idx = 8 - col
        return (idx / 16.0) * (2.0 * np.pi)
    else: # 'L'
        # L1 is near pi, L8 is near 2*pi
        idx = 8 + (col - 1)
        return (idx / 16.0) * (2.0 * np.pi)
