from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WindowConfig:
    window_seconds: float = 30.0
    step_seconds: float = 5.0
    default_fps: float = 30.0


@dataclass
class ThresholdConfig:
    # Eye event detection
    ear_close_threshold: float = 0.21
    blink_min_frames: int = 2

    # Mouth event detection
    mar_yawn_threshold: float = 0.75
    yawn_min_frames: int = 6

    # Head nod detection using pitch (degrees)
    nod_down_threshold_deg: float = 18.0
    nod_up_threshold_deg: float = 10.0

    # Alert
    alert_cooldown_seconds: float = 90.0


@dataclass
class BaselineConfig:
    # Initial baseline calibration duration
    calibration_seconds: float = 60.0

    # Adaptive thresholding factors
    # ear_close_threshold = max(0.12, ear_mean - ear_k * ear_std)
    ear_k: float = 2.0

    # mar_yawn_threshold = mar_mean + mar_k * mar_std
    mar_k: float = 2.5

    # Clamp ranges to avoid crazy values
    ear_min: float = 0.12
    ear_max: float = 0.35
    mar_min: float = 0.30
    mar_max: float = 1.50
