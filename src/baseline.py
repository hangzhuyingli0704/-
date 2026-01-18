from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .config import BaselineConfig
from .math_utils import clamp


@dataclass
class BaselineResult:
    ear_mean: float
    ear_std: float
    mar_mean: float
    mar_std: float
    ear_close_thr: float
    mar_yawn_thr: float


def compute_baseline(ears: List[float], mars: List[float], cfg: BaselineConfig) -> Optional[BaselineResult]:
    """Compute baseline thresholds from initial calibration period.

    We clamp thresholds to keep them reasonable across users.
    """
    if len(ears) < 30 or len(mars) < 30:
        return None

    e = np.array([x for x in ears if x is not None and np.isfinite(x)], dtype=float)
    m = np.array([x for x in mars if x is not None and np.isfinite(x)], dtype=float)
    if e.size < 30 or m.size < 30:
        return None

    ear_mean = float(np.mean(e))
    ear_std = float(np.std(e) + 1e-6)
    mar_mean = float(np.mean(m))
    mar_std = float(np.std(m) + 1e-6)

    ear_close_thr = ear_mean - cfg.ear_k * ear_std
    mar_yawn_thr = mar_mean + cfg.mar_k * mar_std

    ear_close_thr = clamp(ear_close_thr, cfg.ear_min, cfg.ear_max)
    mar_yawn_thr = clamp(mar_yawn_thr, cfg.mar_min, cfg.mar_max)

    return BaselineResult(
        ear_mean=ear_mean,
        ear_std=ear_std,
        mar_mean=mar_mean,
        mar_std=mar_std,
        ear_close_thr=ear_close_thr,
        mar_yawn_thr=mar_yawn_thr,
    )
