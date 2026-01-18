from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def euclid(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return float(math.hypot(p1[0] - p2[0], p1[1] - p2[1]))


def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def safe_div(num: float, den: float, default: float = 0.0) -> float:
    if den == 0:
        return float(default)
    return float(num / den)


def rolling_variance(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.var(x))
