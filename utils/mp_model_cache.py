"""
Download MediaPipe Tasks model files once into project models/ directory.
(Legacy mp.solutions is not available in mediapipe>=0.10.31 / Py3.13 wheels.)
"""

from __future__ import annotations

import os
import urllib.request

# Official Google-hosted task bundle
FACE_LANDMARKER_TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)


def get_models_dir(base_dir: str | None = None) -> str:
    root = base_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d = os.path.join(root, "models")
    os.makedirs(d, exist_ok=True)
    return d


def ensure_face_landmarker_model(base_dir: str | None = None) -> str:
    """Return path to face_landmarker.task, downloading if missing."""
    path = os.path.join(get_models_dir(base_dir), "face_landmarker.task")
    if os.path.isfile(path) and os.path.getsize(path) > 1_000_000:
        return path
    print("[PROCTIFY] Downloading face_landmarker.task (one-time, ~3MB)…")
    urllib.request.urlretrieve(FACE_LANDMARKER_TASK_URL, path)
    return path
