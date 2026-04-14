"""
PROCTIFY — Head pose (yaw, pitch) from landmark lists + cv2.solvePnP.
Works with MediaPipe Tasks FaceLandmarker landmarks (478 pts, same indices as legacy mesh).
"""

from __future__ import annotations

import math
from typing import Any, Optional

import cv2
import numpy as np

_NOSE_TIP = 1
_CHIN = 152
_LEFT_EYE_OUTER = 33
_RIGHT_EYE_OUTER = 263
_LEFT_MOUTH = 61
_RIGHT_MOUTH = 291

_MODEL_POINTS = np.array(
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


def _lm_xy(lm: Any, w: int, h: int, idx: int) -> tuple[float, float]:
    p = lm[idx]
    return (p.x * w, p.y * h)


def estimate_head_pose_from_landmarks(
    lm: Any, w: int, h: int
) -> tuple[Optional[float], Optional[float], str]:
    """lm: indexable sequence of landmarks with .x .y (single face)."""
    try:
        face_2d = np.array(
            [
                _lm_xy(lm, w, h, _NOSE_TIP),
                _lm_xy(lm, w, h, _CHIN),
                _lm_xy(lm, w, h, _LEFT_EYE_OUTER),
                _lm_xy(lm, w, h, _RIGHT_EYE_OUTER),
                _lm_xy(lm, w, h, _LEFT_MOUTH),
                _lm_xy(lm, w, h, _RIGHT_MOUTH),
            ],
            dtype=np.float64,
        )
    except (IndexError, AttributeError):
        return None, None, "No Mesh"

    focal = 1.0 * w
    center = (w / 2.0, h / 2.0)
    cam_matrix = np.array(
        [[focal, 0.0, center[0]], [0.0, focal, center[1]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist = np.zeros((4, 1))

    ok, rot_vec, _trans = cv2.solvePnP(
        _MODEL_POINTS,
        face_2d,
        cam_matrix,
        dist,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None, None, "PnP Failed"

    rmat, _ = cv2.Rodrigues(rot_vec)
    sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        pitch = math.degrees(math.atan2(rmat[2, 1], rmat[2, 2]))
        yaw = math.degrees(math.atan2(-rmat[2, 0], sy))
    else:
        pitch = math.degrees(math.atan2(-rmat[1, 2], rmat[1, 1]))
        yaw = math.degrees(math.atan2(-rmat[2, 0], sy))

    status = _classify_head(yaw, pitch)
    return float(yaw), float(pitch), status


def _classify_head(yaw: float, pitch: float) -> str:
    if pitch > 20:
        return "Looking Down"
    if pitch < -15:
        return "Looking Up"
    if abs(yaw) > 25:
        return "Looking Left" if yaw > 0 else "Looking Right"
    return "Forward"


def head_overlay_line(status: str) -> str:
    return "[Head: " + status + "]"
