"""
PROCTIFY — Face detection: MediaPipe Tasks FaceLandmarker when available, else OpenCV Haar fallback.

On some Windows + Python 3.13 setups MediaPipe fails to load (e.g. AttributeError: function 'free' not found).
The fallback keeps proctoring usable with coarser head/gaze estimates from face-box position only.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np

from detection.gaze_tracking import estimate_gaze_from_landmarks
from detection.head_pose import estimate_head_pose_from_landmarks


@dataclass
class FaceAnalysis:
    """Per-frame face pipeline output."""

    num_faces: int
    boxes: list[tuple[int, int, int, int]]
    yaw: Optional[float]
    pitch: Optional[float]
    head_status: str
    gaze_label: str
    gaze_ok: bool


def _head_pose_from_bbox(
    x: int, y: int, bw: int, bh: int, fw: int, fh: int
) -> tuple[Optional[float], Optional[float], str]:
    """Coarse pose when only a face box is available (OpenCV fallback)."""
    cx = x + bw / 2.0
    cy = y + bh / 2.0
    nx = (cx - fw / 2.0) / max(fw / 2.0, 1.0)
    ny = (cy - fh / 2.0) / max(fh / 2.0, 1.0)
    if ny > 0.18:
        return 0.0, 25.0, "Looking Down"
    if nx > 0.2:
        return 30.0, 0.0, "Looking Left"
    if nx < -0.2:
        return -30.0, 0.0, "Looking Right"
    return 0.0, 0.0, "Forward"


class FaceDetector:
    """
    Primary: MediaPipe FaceLandmarker (VIDEO).
    Fallback: OpenCV Haar frontal face (no landmarks — bbox-based pose only).
    """

    def __init__(self) -> None:
        self._mode: str = "opencv"
        self._landmarker = None
        self._cascade: cv2.CascadeClassifier | None = None
        self._analyze_impl: Callable[[np.ndarray, int], FaceAnalysis]

        force_cv = os.environ.get("PROCTIFY_FORCE_OPENCV", "").lower() in (
            "1",
            "true",
            "yes",
        )
        if force_cv:
            print(
                "[PROCTIFY] PROCTIFY_FORCE_OPENCV set — skipping MediaPipe, using OpenCV only."
            )
            self._init_opencv()
            self._analyze_impl = self._analyze_opencv
            return

        try:
            self._init_mediapipe()
            self._mode = "mediapipe"
            self._analyze_impl = self._analyze_mediapipe
        except Exception as e:
            print(
                "[PROCTIFY] MediaPipe Tasks could not start "
                f"({type(e).__name__}: {e}). "
                "Using OpenCV Haar cascade fallback (reduced gaze/head accuracy)."
            )
            self._init_opencv()
            self._analyze_impl = self._analyze_opencv

    def _init_mediapipe(self) -> None:
        """Raises if native library or model init fails."""
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core import base_options as base_options_lib
        from mediapipe.tasks.python.vision.core.image import ImageFormat

        from utils.mp_model_cache import ensure_face_landmarker_model

        model_path = ensure_face_landmarker_model()
        opts = vision.FaceLandmarkerOptions(
            base_options=base_options_lib.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=4,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(opts)
        self._mp_Image = mp.Image
        self._ImageFormat = ImageFormat

    def _init_opencv(self) -> None:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(path)
        if self._cascade.empty():
            raise RuntimeError(f"Failed to load Haar cascade: {path}")

    def _analyze_mediapipe(self, image_rgb: np.ndarray, timestamp_ms: int) -> FaceAnalysis:
        h, w = image_rgb.shape[:2]
        if not image_rgb.flags["C_CONTIGUOUS"]:
            image_rgb = np.ascontiguousarray(image_rgb)

        mp_image = self._mp_Image(image_format=self._ImageFormat.SRGB, data=image_rgb)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        face_lms = result.face_landmarks
        n = len(face_lms)
        boxes: list[tuple[int, int, int, int]] = []

        for fl in face_lms:
            xs = [p.x * w for p in fl]
            ys = [p.y * h for p in fl]
            if not xs:
                continue
            x0, x1 = int(min(xs)), int(max(xs))
            y0, y1 = int(min(ys)), int(max(ys))
            boxes.append((x0, y0, max(1, x1 - x0), max(1, y1 - y0)))

        yaw = pitch = None
        head_status = "Unknown"
        gaze_label = "Unknown"
        gaze_ok = False

        if n == 1:
            lm0 = face_lms[0]
            yaw, pitch, head_status = estimate_head_pose_from_landmarks(lm0, w, h)
            gaze_label, gaze_ok = estimate_gaze_from_landmarks(lm0, w, h)
            if not gaze_ok:
                gaze_label = "Unknown"
        elif n == 0:
            head_status = "No Face"
        else:
            head_status = "Multiple Faces"

        return FaceAnalysis(
            num_faces=n,
            boxes=boxes,
            yaw=yaw,
            pitch=pitch,
            head_status=head_status,
            gaze_label=gaze_label,
            gaze_ok=gaze_ok,
        )

    def _analyze_opencv(self, image_rgb: np.ndarray, _timestamp_ms: int) -> FaceAnalysis:
        h, w = image_rgb.shape[:2]
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(48, 48),
        )
        n = len(faces)
        boxes = [(int(x), int(y), int(bw), int(bh)) for (x, y, bw, bh) in faces]

        yaw = pitch = None
        head_status = "Unknown"
        gaze_label = "Unknown"
        gaze_ok = False

        if n == 1:
            x, y, bw, bh = boxes[0]
            yaw, pitch, head_status = _head_pose_from_bbox(x, y, bw, bh, w, h)
            gaze_label = "Center"
            gaze_ok = False
        elif n == 0:
            head_status = "No Face"
        else:
            head_status = "Multiple Faces"

        return FaceAnalysis(
            num_faces=n,
            boxes=boxes,
            yaw=yaw,
            pitch=pitch,
            head_status=head_status,
            gaze_label=gaze_label,
            gaze_ok=gaze_ok,
        )

    def analyze(self, image_rgb: np.ndarray, timestamp_ms: int) -> FaceAnalysis:
        return self._analyze_impl(image_rgb, timestamp_ms)

    def detect(
        self, image_rgb: np.ndarray, timestamp_ms: Optional[int] = None
    ) -> tuple[int, list[tuple[int, int, int, int]]]:
        ts = timestamp_ms if timestamp_ms is not None else int(time.time() * 1000)
        a = self.analyze(image_rgb, ts)
        return a.num_faces, a.boxes

    def draw_boxes(
        self,
        frame_bgr: np.ndarray,
        boxes: list[tuple[int, int, int, int]],
        color: tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        out = frame_bgr.copy()
        for (x, y, bw, bh) in boxes:
            cv2.rectangle(out, (x, y), (x + bw, y + bh), color, thickness)
        return out

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()


def annotate_face_status(num_faces: int) -> str:
    if num_faces == 0:
        return "[Face: Not Detected]"
    if num_faces == 1:
        return "[Face: Detected]"
    return f"[Face: Multiple ({num_faces})]"
