"""
PROCTIFY — AI Smart Exam Monitoring System
Flask application: routes, MJPEG exam stream, trust score, termination, reports.
"""

from __future__ import annotations

import base64
import os
import threading
import time
import uuid
from typing import Any

import cv2
import numpy as np
from flask import Flask, jsonify, redirect, render_template, request, session, url_for

from detection.behavior_analysis import BehaviorState, final_status
from detection.face_detection import FaceDetector, annotate_face_status
from detection.gaze_tracking import gaze_overlay_line
from detection.head_pose import head_overlay_line
from detection.object_detection import detect_objects, object_overlay_line
from utils.alert_system import log_alert
from utils.csv_logger import log_row
from utils.report_generator import build_report_payload, save_json_report, save_pdf_report

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "logs")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
EVIDENCE_DIR = os.path.join(BASE_DIR, "evidence")

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(EVIDENCE_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get("PROCTIFY_SECRET", "proctify-dev-secret-change-me")

# ---------------------------------------------------------------------------
# Shared runtime (single local exam at a time — demo mode)
# ---------------------------------------------------------------------------
_runtime_lock = threading.Lock()
_runtime: dict[str, Any] = {
    "behavior": None,  # BehaviorState
    "exam_active": False,
    "student": {},
    "evidence_files": [],
    "last_frame": None,
    "overlay": {},
    "fps": 0.0,
    "warning_log": [],
    "phone_prev": False,
    "yolo_frame": 0,
    "start_time": None,
    "cap": None,
    "last_v_for_warn": 0,
    "report_generated": False,
    "multi_face_fired": False,
}


def _reset_runtime() -> None:
    with _runtime_lock:
        _runtime["behavior"] = BehaviorState()
        _runtime["exam_active"] = True
        _runtime["evidence_files"] = []
        _runtime["last_frame"] = None
        _runtime["overlay"] = {}
        _runtime["fps"] = 0.0
        _runtime["warning_log"] = []
        _runtime["phone_prev"] = False
        _runtime["yolo_frame"] = 0
        _runtime["start_time"] = time.time()
        _runtime["last_v_for_warn"] = 0
        _runtime["report_generated"] = False
        _runtime["multi_face_fired"] = False
        _runtime["behavior"].reset_clock()


def _save_evidence(frame_bgr: np.ndarray, reason: str) -> str:
    name = f"{int(time.time() * 1000)}_{reason}.jpg"
    path = os.path.join(EVIDENCE_DIR, name)
    cv2.imwrite(path, frame_bgr)
    with _runtime_lock:
        _runtime["evidence_files"].append(path)
    return path


def _append_warning(text: str) -> None:
    with _runtime_lock:
        _runtime["warning_log"] = _runtime["warning_log"][-49:]
        _runtime["warning_log"].append(
            {"t": time.strftime("%H:%M:%S"), "text": text}
        )


def _terminate_exam(reason: str, student_id: str) -> None:
    beh = _runtime.get("behavior")
    if beh is None or beh.terminated:
        return
    beh.terminate(reason)
    log_alert(LOGS_DIR, student_id, f"exam_terminated:{reason}", "HIGH")
    with _runtime_lock:
        sid = _runtime.get("session_id_cache", "unknown")
    log_row(
        LOGS_DIR,
        sid,
        {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "event": "terminate",
            "detail": reason,
            "severity": "HIGH",
        },
    )
    _append_warning("🚫 Exam Terminated – Cheating Detected")
    do_report = False
    with _runtime_lock:
        if not _runtime.get("report_generated"):
            _runtime["report_generated"] = True
            do_report = True
    if do_report:
        try:
            _generate_reports()
        except Exception as ex:
            print(f"Report generation failed: {ex}")


def _generate_reports() -> dict[str, str]:
    """Build JSON/PDF using cached student info (safe from video thread)."""
    with _runtime_lock:
        stu = dict(_runtime.get("student_cache") or {})
        beh = _runtime.get("behavior")
        evidence = list(_runtime.get("evidence_files", []))
    if beh is None:
        return {}
    snap = beh.snapshot()
    status = final_status(snap["phone_detected"], snap["violations"])
    payload = build_report_payload(
        stu.get("name", ""),
        stu.get("student_id", ""),
        stu.get("exam_name", ""),
        snap,
        status,
        evidence,
    )
    base = f"report_{stu.get('student_id', 'unknown')}_{int(time.time())}"
    jp = save_json_report(REPORTS_DIR, payload, base)
    pp = save_pdf_report(REPORTS_DIR, payload, base)
    rec = {"json": jp, "pdf": pp, "payload": payload}
    with _runtime_lock:
        _runtime["last_report"] = rec
    return {"json": jp, "pdf": pp}


# Lazy detector (MediaPipe Tasks FaceLandmarker — single graph for mesh + count)
_face_detector: FaceDetector | None = None


def _get_face_detector() -> FaceDetector:
    global _face_detector
    if _face_detector is None:
        _face_detector = FaceDetector()
    return _face_detector


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        u = request.form.get("username", "").strip()
        p = request.form.get("password", "").strip()
        # Demo auth — replace with real auth in production
        if u and p:
            session["user"] = u
            session["session_id"] = str(uuid.uuid4())
            return redirect(url_for("details"))
    return render_template("login.html")


@app.route("/details", methods=["GET", "POST"])
def details():
    if "user" not in session:
        return redirect(url_for("login"))
    if request.method == "POST":
        session["student"] = {
            "name": request.form.get("name", "").strip(),
            "student_id": request.form.get("student_id", "").strip(),
            "exam_name": request.form.get("exam_name", "").strip(),
        }
        return redirect(url_for("camera_check"))
    return render_template("details.html", student=session.get("student"))


@app.route("/camera_check")
def camera_check():
    if "user" not in session or "student" not in session:
        return redirect(url_for("login"))
    return render_template("camera_check.html")


@app.route("/api/camera_verify", methods=["POST"])
def api_camera_verify():
    """Receive base64 JPEG; verify exactly one face."""
    data = request.get_json(silent=True) or {}
    b64 = data.get("image", "") or ""
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    if not b64.strip():
        return jsonify({"ok": False, "faces": -1, "message": "No image data received"})
    try:
        raw = base64.b64decode(b64)
        arr = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"ok": False, "faces": -1, "message": "Bad image"})
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        fd = _get_face_detector()
        n, boxes = fd.detect(rgb, int(time.time() * 1000))
        ok = n == 1
        msg = "Camera Verified ✅" if ok else (
            "No face detected" if n == 0 else "Multiple people detected"
        )
        return jsonify({"ok": ok, "faces": n, "message": msg})
    except Exception as e:
        return jsonify({"ok": False, "faces": -1, "message": str(e)})


@app.route("/exam")
def exam():
    if "user" not in session or "student" not in session:
        return redirect(url_for("login"))
    _reset_runtime()
    stu = session.get("student", {})
    with _runtime_lock:
        _runtime["student_cache"] = dict(stu)
        _runtime["session_id_cache"] = session.get("session_id", "unknown")
    log_alert(
        LOGS_DIR,
        stu.get("student_id", "unknown"),
        "exam_started",
        "INFO",
        {"exam": stu.get("exam_name")},
    )
    return render_template("exam.html")


@app.route("/api/exam_state")
def api_exam_state():
    """JSON for UI polling."""
    with _runtime_lock:
        beh = _runtime["behavior"]
        snap = beh.snapshot() if beh else {}
        ov = dict(_runtime["overlay"])
        warn_log = list(_runtime["warning_log"])
        fps = _runtime["fps"]
        evidence_count = len(_runtime["evidence_files"])
    trust = int(round(max(0, min(100, snap.get("trust_score", 100)))))
    wl = snap.get("warning_level", 0)
    warn_text = None
    if wl == 1:
        warn_text = "⚠️ Warning 1/3"
    elif wl == 2:
        warn_text = "⚠️ Warning 2/3"
    elif wl >= 3 and not snap.get("terminated"):
        warn_text = "🚫 Final Warning"

    return jsonify(
        {
            "trust_score": trust,
            "face_status": ov.get("face_line", ""),
            "eye_status": ov.get("eye_line", ""),
            "head_status": ov.get("head_line", ""),
            "object_status": ov.get("obj_line", ""),
            "alert": ov.get("alert"),
            "warnings": warn_text,
            "violation_count": snap.get("violations", 0),
            "terminated": snap.get("terminated", False),
            "terminate_reason": snap.get("terminate_reason"),
            "fps": round(fps, 1),
            "warning_log": warn_log,
            "evidence_count": evidence_count,
        }
    )


@app.route("/video_feed")
def video_feed():
    """MJPEG stream with overlays and detection."""
    from flask import Response

    fd = _get_face_detector()

    def gen():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                blank,
                "Camera not available",
                (80, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            _, jpg = cv2.imencode(".jpg", blank)
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
            )
            return

        prev_t = time.time()
        fps_smooth = 0.0
        frame_i = 0

        while True:
            with _runtime_lock:
                active = _runtime.get("exam_active", False)
                beh = _runtime["behavior"]
                terminated = beh and beh.terminated

            if not active or beh is None:
                cap.release()
                break

            ok, frame = cap.read()
            if not ok:
                break

            frame_i += 1
            now = time.time()
            dt = max(0.001, now - prev_t)
            prev_t = now
            fps_smooth = fps_smooth * 0.85 + (1.0 / dt) * 0.15

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ts_ms = int(now * 1000)

            fa = fd.analyze(rgb, ts_ms)
            n_faces = fa.num_faces
            boxes = fa.boxes
            display = fd.draw_boxes(frame, boxes)

            head_status = fa.head_status
            gaze_label = fa.gaze_label
            yaw, pitch = fa.yaw, fa.pitch
            phone = False
            book = False
            person_count = 0
            obj_line = "[Object: Clear]"
            alert_msg = None
            cheat_alert = False

            run_yolo = frame_i % 3 == 0
            if run_yolo:
                with _runtime_lock:
                    _runtime["yolo_frame"] = frame_i
                od = detect_objects(display, run=True)
            else:
                od = detect_objects(display, run=False)

            display = od["annotated_frame"]
            phone = od["phone"]
            book = od["book"]
            person_count = od["person_count"]

            extra_persons = max(0, person_count - 1) if n_faces >= 1 else person_count

            if not fa.gaze_ok and n_faces == 1:
                gaze_label = "Unknown"

            obj_line = object_overlay_line(phone, book, extra_persons)

            # Alerts: possible cheating
            if phone or n_faces > 1 or extra_persons > 0:
                cheat_alert = True
                alert_msg = "⚠️ ALERT: Possible Cheating"

            face_line = annotate_face_status(n_faces)
            head_line = head_overlay_line(head_status)
            eye_line = gaze_overlay_line(gaze_label)

            with _runtime_lock:
                student_id = (_runtime.get("student_cache") or {}).get(
                    "student_id", "unknown"
                )

            # Behavior / trust — skip if terminated (freeze last)
            if not terminated:
                beh.tick(dt, n_faces, head_status, gaze_label)

                # Phone edge — one-shot penalty + evidence + terminate
                with _runtime_lock:
                    prev_p = _runtime["phone_prev"]
                if phone and not prev_p:
                    beh.apply_phone_penalty()
                    _save_evidence(display, "phone")
                    log_alert(LOGS_DIR, student_id, "phone_detected", "HIGH")
                    _append_warning("Phone detected — HIGH violation")
                    _terminate_exam("Cheating Detected (phone)", student_id)
                with _runtime_lock:
                    _runtime["phone_prev"] = phone

                with _runtime_lock:
                    mf_done = _runtime.get("multi_face_fired", False)
                if n_faces > 1 and not mf_done:
                    beh.apply_multi_face_penalty()
                    _save_evidence(display, "multiple_faces")
                    log_alert(LOGS_DIR, student_id, "multiple_faces", "HIGH")
                    _append_warning("Multiple faces — HIGH violation")
                    with _runtime_lock:
                        _runtime["multi_face_fired"] = True
                    _terminate_exam("Cheating Detected (multiple faces)", student_id)

                snap = beh.snapshot()
                with _runtime_lock:
                    last_v = _runtime.get("last_v_for_warn", 0)
                vnow = snap["violations"]
                if vnow > last_v:
                    if vnow == 1:
                        _append_warning("⚠️ Warning 1/3")
                    elif vnow == 2:
                        _append_warning("⚠️ Warning 2/3")
                    elif vnow >= 3:
                        _append_warning("🚫 Final Warning")
                    with _runtime_lock:
                        _runtime["last_v_for_warn"] = vnow

                if snap["violations"] >= 3:
                    _save_evidence(display, "violations")
                    _terminate_exam("Too many violations", student_id)

                if cheat_alert and frame_i % 15 == 0:
                    _save_evidence(display, "alert")

            with _runtime_lock:
                _runtime["fps"] = fps_smooth
                _runtime["overlay"] = {
                    "face_line": face_line,
                    "head_line": head_line,
                    "eye_line": eye_line,
                    "obj_line": obj_line,
                    "alert": alert_msg,
                    "cheat": cheat_alert,
                }
                _runtime["last_frame"] = display

            # ---- Overlay panel (BGR)
            font = cv2.FONT_HERSHEY_SIMPLEX
            y0 = 28
            lines = [
                face_line,
                head_line,
                eye_line,
                obj_line,
            ]
            if alert_msg:
                lines.append(alert_msg)

            color_main = (0, 255, 0)
            color_alert = (0, 0, 255)
            color_yellow = (0, 255, 255)

            for i, line in enumerate(lines):
                col = color_alert if ("ALERT" in line or "Multiple" in line or "Not Detected" in line) else color_main
                if "Object:" in line and ("Phone" in line or "Book" in line or "Person" in line):
                    col = color_yellow
                cv2.putText(display, line, (12, y0 + i * 26), font, 0.55, col, 2)

            snap = beh.snapshot() if beh else {}
            trust = int(round(max(0, min(100, snap.get("trust_score", 100)))))
            cv2.putText(
                display,
                f"Trust Score: {trust}%",
                (w - 280, 36),
                font,
                0.7,
                (0, 200, 255),
                2,
            )
            cv2.putText(
                display,
                f"FPS: {fps_smooth:.1f}",
                (w - 280, 66),
                font,
                0.6,
                (200, 200, 200),
                2,
            )

            if snap.get("terminated"):
                cv2.rectangle(display, (0, 0), (w, h), (0, 0, 80), -1)
                cv2.putText(
                    display,
                    "Exam Terminated - Cheating Detected",
                    (40, h // 2 - 20),
                    font,
                    1.0,
                    (0, 0, 255),
                    3,
                )
                msg = snap.get("terminate_reason") or ""
                cv2.putText(display, msg, (40, h // 2 + 30), font, 0.7, (255, 255, 255), 2)

            _, jpg = cv2.imencode(".jpg", display)
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
            )

            with _runtime_lock:
                if _runtime.get("behavior") and _runtime["behavior"].terminated:
                    # Keep streaming frozen end frame a few times
                    pass

    return Response(
        gen(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/end_exam", methods=["POST"])
def end_exam():
    """Manual end — generate report if not terminated."""
    with _runtime_lock:
        _runtime["exam_active"] = False
        beh = _runtime.get("behavior")
        already = _runtime.get("report_generated", False)
    if beh and not beh.terminated:
        beh.terminate("Exam submitted by candidate")
    if not already:
        with _runtime_lock:
            _runtime["report_generated"] = True
        paths = _generate_reports()
    else:
        with _runtime_lock:
            lr = _runtime.get("last_report") or {}
        paths = {"json": lr.get("json"), "pdf": lr.get("pdf")}
    return jsonify({"ok": True, "reports": paths})


@app.route("/terminate_watch")
def terminate_watch():
    """Client poll after termination — report already built in _terminate_exam."""
    return jsonify({"done": True})


@app.route("/report")
def report():
    with _runtime_lock:
        rep = _runtime.get("last_report")
    payload = (rep or {}).get("payload")
    return render_template("report.html", payload=payload)


# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("PROCTIFY — Starting server at http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, threaded=True)
