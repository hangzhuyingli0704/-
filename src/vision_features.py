from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2

from .landmarks import (
    LEFT_EYE,
    RIGHT_EYE,
    MOUTH_INNER,
    MOUTH_CORNERS,
    NOSE_TIP,
    CHIN,
    LEFT_EYE_OUTER,
    RIGHT_EYE_OUTER,
    MOUTH_LEFT,
    MOUTH_RIGHT,
)
from .math_utils import euclid, safe_div


def _lm_xy(landmarks, idx: int, w: int, h: int) -> Tuple[float, float]:
    lm = landmarks[idx]
    return (lm.x * w, lm.y * h)


def compute_ear(landmarks, w: int, h: int) -> Tuple[Optional[float], Optional[float]]:
    """Compute EAR for left and right eyes. Returns (left_ear, right_ear)."""
    try:
        def ear_for(eye_idx_tuple):
            p1 = _lm_xy(landmarks, eye_idx_tuple[0], w, h)
            p2 = _lm_xy(landmarks, eye_idx_tuple[1], w, h)
            p3 = _lm_xy(landmarks, eye_idx_tuple[2], w, h)
            p4 = _lm_xy(landmarks, eye_idx_tuple[3], w, h)
            p5 = _lm_xy(landmarks, eye_idx_tuple[4], w, h)
            p6 = _lm_xy(landmarks, eye_idx_tuple[5], w, h)
            # EAR = (||p2-p6|| + ||p3-p5||) / (2*||p1-p4||)
            num = euclid(p2, p6) + euclid(p3, p5)
            den = 2.0 * euclid(p1, p4)
            return safe_div(num, den, default=0.0)

        return ear_for(LEFT_EYE), ear_for(RIGHT_EYE)
    except Exception:
        return None, None


def compute_mar(landmarks, w: int, h: int) -> Optional[float]:
    """Mouth aspect ratio: inner vertical / mouth width."""
    try:
        up = _lm_xy(landmarks, MOUTH_INNER[0], w, h)
        low = _lm_xy(landmarks, MOUTH_INNER[1], w, h)
        left = _lm_xy(landmarks, MOUTH_CORNERS[0], w, h)
        right = _lm_xy(landmarks, MOUTH_CORNERS[1], w, h)
        vertical = euclid(up, low)
        width = euclid(left, right)
        return safe_div(vertical, width, default=0.0)
    except Exception:
        return None


@dataclass
class HeadPose:
    pitch_deg: float
    yaw_deg: float
    roll_deg: float


def compute_head_pose(landmarks, frame_w: int, frame_h: int) -> Optional[HeadPose]:
    """Estimate head pose (pitch/yaw/roll in degrees) using solvePnP.

    This is an approximation but works well enough for fatigue/nodding features.
    """
    try:
        image_points = np.array(
            [
                _lm_xy(landmarks, NOSE_TIP, frame_w, frame_h),
                _lm_xy(landmarks, CHIN, frame_w, frame_h),
                _lm_xy(landmarks, LEFT_EYE_OUTER, frame_w, frame_h),
                _lm_xy(landmarks, RIGHT_EYE_OUTER, frame_w, frame_h),
                _lm_xy(landmarks, MOUTH_LEFT, frame_w, frame_h),
                _lm_xy(landmarks, MOUTH_RIGHT, frame_w, frame_h),
            ],
            dtype=np.float64,
        )

        # Generic 3D model points (mm-ish scale). These are widely used heuristics.
        model_points = np.array(
            [
                (0.0, 0.0, 0.0),
                (0.0, -330.0, -65.0),
                (-225.0, 170.0, -135.0),
                (225.0, 170.0, -135.0),
                (-150.0, -150.0, -125.0),
                (150.0, -150.0, -125.0),
            ],
            dtype=np.float64,
        )

        focal_length = float(frame_w)
        center = (frame_w / 2.0, frame_h / 2.0)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rvec, tvec = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return None

        rot_mat, _ = cv2.Rodrigues(rvec)
        # Convert rotation matrix to Euler angles.
        sy = np.sqrt(rot_mat[0, 0] ** 2 + rot_mat[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(rot_mat[2, 1], rot_mat[2, 2])
            y = np.arctan2(-rot_mat[2, 0], sy)
            z = np.arctan2(rot_mat[1, 0], rot_mat[0, 0])
        else:
            x = np.arctan2(-rot_mat[1, 2], rot_mat[1, 1])
            y = np.arctan2(-rot_mat[2, 0], sy)
            z = 0

        pitch = float(np.degrees(x))
        yaw = float(np.degrees(y))
        roll = float(np.degrees(z))
        return HeadPose(pitch_deg=pitch, yaw_deg=yaw, roll_deg=roll)
    except Exception:
        return None
