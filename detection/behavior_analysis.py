"""
PROCTIFY — Session behavior aggregation, trust score, violation counting, warnings.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BehaviorState:
    """Mutable per-exam tracking_data and trust score."""

    total_time: float = 0.0
    looking_forward_time: float = 0.0
    looking_away_time: float = 0.0
    looking_down_time: float = 0.0
    violations: int = 0
    phone_detected: int = 0
    multiple_faces: int = 0
    alerts: list[dict[str, Any]] = field(default_factory=list)

    _last_t: float = field(default_factory=time.time)
    _away_accum: float = 0.0
    _down_accum: float = 0.0
    _no_face_accum: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    trust_score: float = 100.0
    warning_level: int = 0
    terminated: bool = False
    terminate_reason: str | None = None

    AWAY_RATE: float = 2.0  # per second while condition holds
    DOWN_RATE: float = 2.0
    PHONE_PENALTY: float = 30.0  # one-shot per detection event
    MULTI_FACE_PENALTY: float = 50.0  # one-shot per frame with multiple faces

    SUSTAIN_AWAY: float = 4.0
    SUSTAIN_DOWN: float = 4.0
    SUSTAIN_NO_FACE: float = 3.0

    def reset_clock(self) -> None:
        with self._lock:
            self._last_t = time.time()

    def tick(
        self,
        dt: float,
        num_faces: int,
        head_status: str,
        gaze: str,
    ) -> None:
        """Update timers and continuous trust penalties for pose/gaze/no-face."""
        with self._lock:
            self.total_time += dt
            away = head_status in ("Looking Left", "Looking Right") or gaze in (
                "Looking Left",
                "Looking Right",
            )
            down = head_status == "Looking Down" or gaze == "Looking Down"

            forward = num_faces == 1 and not away and not down
            if forward:
                self.looking_forward_time += dt
                self._away_accum = 0.0
                self._down_accum = 0.0
            else:
                if away:
                    self.looking_away_time += dt
                    self._away_accum += dt
                    self.trust_score = max(0.0, self.trust_score - self.AWAY_RATE * dt)
                else:
                    self._away_accum = 0.0

                if down:
                    self.looking_down_time += dt
                    self._down_accum += dt
                    self.trust_score = max(0.0, self.trust_score - self.DOWN_RATE * dt)
                else:
                    self._down_accum = 0.0

            if num_faces == 0:
                self._no_face_accum += dt
                self.trust_score = max(0.0, self.trust_score - 3.0 * dt)
            else:
                self._no_face_accum = 0.0

            if self._away_accum >= self.SUSTAIN_AWAY:
                self._add_violation("looking_away_sustained")
                self._away_accum = 0.0
            if self._down_accum >= self.SUSTAIN_DOWN:
                self._add_violation("looking_down_sustained")
                self._down_accum = 0.0
            if self._no_face_accum >= self.SUSTAIN_NO_FACE:
                self._add_violation("no_face")
                self._no_face_accum = 0.0

    def apply_phone_penalty(self) -> None:
        """Call once when phone is detected in a frame (edge-deduped by caller)."""
        with self._lock:
            self.phone_detected += 1
            self.trust_score = max(0.0, self.trust_score - self.PHONE_PENALTY)
            self.alerts.append(
                {
                    "event": "phone_detected",
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "severity": "HIGH",
                }
            )

    def apply_multi_face_penalty(self) -> None:
        """Call once per frame when multiple faces detected."""
        with self._lock:
            self.multiple_faces += 1
            self.trust_score = max(0.0, self.trust_score - self.MULTI_FACE_PENALTY)
            self.alerts.append(
                {
                    "event": "multiple_faces",
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "severity": "HIGH",
                }
            )

    def _add_violation(self, kind: str) -> None:
        self.violations += 1
        self.alerts.append(
            {
                "event": kind,
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "severity": "MEDIUM",
            }
        )
        self._sync_warnings()

    def _sync_warnings(self) -> None:
        v = self.violations
        if v <= 0:
            self.warning_level = 0
        elif v == 1:
            self.warning_level = 1
        elif v == 2:
            self.warning_level = 2
        else:
            self.warning_level = 3

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "total_time": self.total_time,
                "looking_forward_time": self.looking_forward_time,
                "looking_away_time": self.looking_away_time,
                "looking_down_time": self.looking_down_time,
                "violations": self.violations,
                "phone_detected": self.phone_detected,
                "multiple_faces": self.multiple_faces,
                "alerts": list(self.alerts),
                "trust_score": round(self.trust_score, 1),
                "warning_level": self.warning_level,
                "terminated": self.terminated,
                "terminate_reason": self.terminate_reason,
            }

    def terminate(self, reason: str) -> None:
        with self._lock:
            self.terminated = True
            self.terminate_reason = reason


def final_status(phone_detected: int, violations: int) -> str:
    if phone_detected > 0 or violations >= 3:
        return "FAILED"
    return "PASSED"
