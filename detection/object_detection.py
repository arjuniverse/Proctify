"""
PROCTIFY — YOLOv8 object detection (Ultralytics).
Targets: person, book, cell phone (COCO names).
"""

from __future__ import annotations

import numpy as np

# Lazy singleton for model load
_yolo_model = None


def get_yolo():
    """Load YOLOv8n once; downloads weights on first run."""
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO

        _yolo_model = YOLO("yolov8n.pt")
    return _yolo_model


def detect_objects(
    frame_bgr: np.ndarray,
    conf: float = 0.35,
    run: bool = True,
) -> dict:
    """
    Run YOLO on frame. If run is False, skip (for frame skipping).

    Returns:
        {
          'phone': bool,
          'book': bool,
          'person_count': int,
          'labels': list[str],  # human-readable hits
          'annotated_frame': np.ndarray,
        }
    """
    result = {
        "phone": False,
        "book": False,
        "person_count": 0,
        "labels": [],
        "annotated_frame": frame_bgr.copy(),
    }
    if not run:
        return result

    model = get_yolo()
    # Ultralytics expects BGR or RGB — OpenCV BGR is fine
    r = model(frame_bgr, conf=conf, verbose=False)[0]

    names = r.names
    annotated = r.plot()  # BGR image with boxes

    if r.boxes is not None and len(r.boxes) > 0:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            name = names.get(cls_id, "").lower()
            if name == "cell phone":
                result["phone"] = True
                result["labels"].append("phone")
            elif name == "book":
                result["book"] = True
                result["labels"].append("book")
            elif name == "person":
                result["person_count"] += 1

    # Extra person boxes beyond the examinee (heuristic: count all persons)
    if result["person_count"] > 0:
        result["labels"].append(f"person x{result['person_count']}")

    result["annotated_frame"] = annotated
    return result


def object_overlay_line(phone: bool, book: bool, extra_persons: int) -> str:
    """Build [Object: ...] line; yellow-style hint when objects present."""
    parts = []
    if phone:
        parts.append("Phone Detected")
    if book:
        parts.append("Book")
    if extra_persons > 0:
        parts.append(f"Extra Person(s): {extra_persons}")
    if not parts:
        return "[Object: Clear]"
    return "[Object: " + ", ".join(parts) + "]"
