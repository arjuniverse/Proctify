"""Append session events to CSV under logs/."""

from __future__ import annotations

import csv
import os
import threading
from pathlib import Path

_lock = threading.Lock()


def log_row(logs_dir: str, session_id: str, row: dict[str, str]) -> None:
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(logs_dir, f"session_{session_id}.csv")
    fieldnames = ["timestamp", "event", "detail", "severity"]
    with _lock:
        new_file = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if new_file:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in fieldnames})
