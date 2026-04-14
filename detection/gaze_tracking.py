"""
PROCTIFY — Gaze from iris + eye corners (Tasks Face Landmarker, 478 landmarks).
"""

from __future__ import annotations

from typing import Any

_LEFT_IRIS_CENTER = 468
_RIGHT_IRIS_CENTER = 473
_LEFT_EYE_INNER = 133
_LEFT_EYE_OUTER = 33
_RIGHT_EYE_INNER = 362
_RIGHT_EYE_OUTER = 263


def _pt(lm: Any, w: int, h: int, idx: int) -> tuple[float, float]:
    p = lm[idx]
    return (p.x * w, p.y * h)


def estimate_gaze_from_landmarks(lm: Any, w: int, h: int) -> tuple[str, bool]:
    """Returns (label, ok). Requires iris indices (478-pt model)."""
    try:
        if len(lm) <= max(_RIGHT_IRIS_CENTER, _LEFT_IRIS_CENTER, _RIGHT_EYE_OUTER):
            return "Unknown", False
    except TypeError:
        return "Unknown", False

    lix, liy = _pt(lm, w, h, _LEFT_IRIS_CENTER)
    rix, riy = _pt(lm, w, h, _RIGHT_IRIS_CENTER)
    iris_x = (lix + rix) / 2.0
    iris_y = (liy + riy) / 2.0

    le_l = _pt(lm, w, h, _LEFT_EYE_OUTER)
    le_r = _pt(lm, w, h, _LEFT_EYE_INNER)
    re_l = _pt(lm, w, h, _RIGHT_EYE_INNER)
    re_r = _pt(lm, w, h, _RIGHT_EYE_OUTER)

    left_bound = (le_l[0] + re_l[0]) / 2.0
    right_bound = (le_r[0] + re_r[0]) / 2.0
    eye_width = max(1.0, right_bound - left_bound)

    top_ref = (le_l[1] + le_r[1] + re_l[1] + re_r[1]) / 4.0
    vert_span = max(15.0, h * 0.08)

    nx = (iris_x - left_bound) / eye_width
    ny = (iris_y - top_ref) / vert_span

    return _classify_gaze(nx, ny), True


def _classify_gaze(nx: float, ny: float) -> str:
    if ny > 0.55:
        return "Looking Down"
    if ny < -0.35:
        return "Looking Up"
    if nx < 0.38:
        return "Looking Left"
    if nx > 0.62:
        return "Looking Right"
    return "Center"


def gaze_overlay_line(gaze: str) -> str:
    return "[Eyes: " + gaze + "]"
