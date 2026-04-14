"""
PROCTIFY — Admin-style alerts: console + JSON lines log file.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

_lock = threading.Lock()


def log_alert(
    logs_dir: str,
    student_id: str,
    event: str,
    severity: str,
    extra: dict[str, Any] | None = None,
) -> None:
    """
    Print structured alert to console and append to logs/alerts.jsonl
    """
    record: dict[str, Any] = {
        "student_id": student_id,
        "event": event,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "severity": severity,
    }
    if extra:
        record["extra"] = extra

    line = json.dumps(record, ensure_ascii=False)
    print(f"[PROCTIFY ALERT] {line}")

    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(logs_dir, "alerts.jsonl")
    with _lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
