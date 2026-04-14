"""
PROCTIFY — JSON + PDF reports (reportlab).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def _pct(part: float, total: float) -> float:
    if total <= 0:
        return 0.0
    return round(100.0 * part / total, 1)


def build_report_payload(
    name: str,
    student_id: str,
    exam_name: str,
    tracking: dict[str, Any],
    final_status: str,
    evidence_paths: list[str],
) -> dict[str, Any]:
    total = float(tracking.get("total_time", 0.0))
    fwd = float(tracking.get("looking_forward_time", 0.0))
    away = float(tracking.get("looking_away_time", 0.0))
    down = float(tracking.get("looking_down_time", 0.0))

    return {
        "name": name,
        "student_id": student_id,
        "exam": exam_name,
        "duration_seconds": round(total, 1),
        "pct_looking_forward": _pct(fwd, total),
        "pct_looking_away": _pct(away, total),
        "pct_looking_down": _pct(down, total),
        "violations": tracking.get("violations", 0),
        "trust_score": tracking.get("trust_score", 0),
        "phone_detected": tracking.get("phone_detected", 0),
        "multiple_faces_events": tracking.get("multiple_faces", 0),
        "final_status": final_status,
        "evidence_images": evidence_paths,
        "alerts": tracking.get("alerts", []),
    }


def save_json_report(reports_dir: str, payload: dict[str, Any], basename: str) -> str:
    Path(reports_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(reports_dir, f"{basename}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def save_pdf_report(reports_dir: str, payload: dict[str, Any], basename: str) -> str:
    Path(reports_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(reports_dir, f"{basename}.pdf")

    doc = SimpleDocTemplate(path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>PROCTIFY — Exam Report</b>", styles["Title"]))
    story.append(Spacer(1, 0.2 * inch))

    rows = [
        ["Candidate", payload.get("name", "")],
        ["Student ID", payload.get("student_id", "")],
        ["Exam", payload.get("exam", "")],
        ["Duration (s)", str(payload.get("duration_seconds", 0))],
        ["% Looking Forward", str(payload.get("pct_looking_forward", 0))],
        ["% Looking Away", str(payload.get("pct_looking_away", 0))],
        ["% Looking Down", str(payload.get("pct_looking_down", 0))],
        ["Violations", str(payload.get("violations", 0))],
        ["Trust Score", str(payload.get("trust_score", 0))],
        ["Phone Detected (count)", str(payload.get("phone_detected", 0))],
        ["Multi-face Events", str(payload.get("multiple_faces_events", 0))],
        ["Final Status", payload.get("final_status", "")],
    ]
    t = Table(rows, colWidths=[2.2 * inch, 3.5 * inch])
    t.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.HexColor("#f0f4f8"), colors.white]),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#1e3a5f")),
            ]
        )
    )
    story.append(t)
    story.append(Spacer(1, 0.25 * inch))
    story.append(Paragraph("<b>Evidence files</b>", styles["Heading2"]))
    for p in payload.get("evidence_images", [])[:20]:
        story.append(Paragraph(os.path.basename(str(p)), styles["Normal"]))

    doc.build(story)
    return path
