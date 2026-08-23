"""Local attendance monitor UI and protected camera ingestion API."""

from __future__ import annotations

import csv
import atexit
import hmac
import io
import os
import secrets
import sqlite3
import subprocess
import sys
import threading
from collections import deque
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from flask import Flask, Response, abort, flash, jsonify, redirect, render_template, request, session, url_for

from attendance_db import AttendanceDatabase, DEFAULT_DATABASE

ROOT = Path(__file__).resolve().parent.parent
WEB_ROOT = ROOT / "web"
INSTANCE_ROOT = ROOT / "instance"


class LatestFrame:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._jpeg: bytes | None = None
        self._sequence = 0

    def put(self, jpeg: bytes) -> None:
        with self._condition:
            self._jpeg = jpeg
            self._sequence += 1
            self._condition.notify_all()

    def get(self) -> bytes | None:
        with self._condition:
            return self._jpeg

    def clear(self) -> None:
        with self._condition:
            self._jpeg = None
            self._sequence += 1
            self._condition.notify_all()

    def wait_for_next(self, previous_sequence: int) -> tuple[bytes | None, int]:
        with self._condition:
            self._condition.wait_for(
                lambda: self._sequence != previous_sequence, timeout=2.0
            )
            return self._jpeg, self._sequence


class CameraProcessManager:
    """Own exactly one background recognition process for the local dashboard."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None
        self._lines: deque[str] = deque(maxlen=30)
        self._requested_stop = False

    def _drain_output(self, process: subprocess.Popen[str]) -> None:
        if process.stdout is None:
            return
        for line in process.stdout:
            cleaned = line.strip()
            if cleaned:
                with self._lock:
                    self._lines.append(cleaned)

    def start(self, source: str = "0") -> dict:
        normalized = source.strip()
        if not normalized.isdecimal() or not 0 <= int(normalized) <= 9:
            raise ValueError("Camera source must be a number from 0 to 9")
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                return self.status_unlocked()
            self._lines.clear()
            self._requested_stop = False
            command = [
                sys.executable,
                str(ROOT / "src" / "realtime_recognition.py"),
                "--source", normalized,
                "--headless",
            ]
            self._process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            threading.Thread(
                target=self._drain_output, args=(self._process,), daemon=True
            ).start()
            return self.status_unlocked()

    def status_unlocked(self) -> dict:
        process = self._process
        if process is None:
            return {"state": "stopped", "message": "Camera is stopped", "recent_output": []}
        return_code = process.poll()
        if return_code is None:
            state = "running"
            message = "Recognition camera is running"
        elif self._requested_stop or return_code == 0:
            state = "stopped"
            message = "Camera is stopped"
        else:
            state = "error"
            message = self._lines[-1] if self._lines else f"Recognition process exited with code {return_code}"
        return {"state": state, "message": message, "recent_output": list(self._lines)[-5:]}

    def status(self) -> dict:
        with self._lock:
            return self.status_unlocked()

    def stop(self) -> dict:
        with self._lock:
            process = self._process
            self._requested_stop = True
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
        return self.status()


def local_secret(filename: str, environment_name: str) -> str:
    configured = os.environ.get(environment_name)
    if configured:
        return configured
    INSTANCE_ROOT.mkdir(parents=True, exist_ok=True)
    path = INSTANCE_ROOT / filename
    if not path.exists():
        path.write_text(secrets.token_urlsafe(32), encoding="utf-8")
    return path.read_text(encoding="utf-8").strip()


def configured_timezone(name: str):
    try:
        return ZoneInfo(name)
    except Exception:
        if name == "Asia/Kolkata":
            return timezone(timedelta(hours=5, minutes=30), name="Asia/Kolkata")
        return timezone.utc


def create_app(
    database_path: Path | str | None = None,
    camera_manager: CameraProcessManager | None = None,
) -> Flask:
    app = Flask(__name__, template_folder=str(WEB_ROOT / "templates"), static_folder=str(WEB_ROOT / "static"))
    app.config.update(
        SECRET_KEY=local_secret("flask_secret.key", "ATTENDANCE_SECRET_KEY"),
        DATABASE_PATH=str(database_path or os.environ.get("ATTENDANCE_DATABASE", DEFAULT_DATABASE)),
        DEVICE_TOKEN=local_secret("device_token.txt", "ATTENDANCE_DEVICE_TOKEN"),
        APP_TIMEZONE=os.environ.get("ATTENDANCE_TIMEZONE", "Asia/Kolkata"),
        MAX_CONTENT_LENGTH=2 * 1024 * 1024,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
    )
    database = AttendanceDatabase(app.config["DATABASE_PATH"])
    database.initialize()
    frames = LatestFrame()
    camera = camera_manager or CameraProcessManager()
    app.extensions["camera_manager"] = camera
    if camera_manager is None:
        atexit.register(camera.stop)

    def require_device() -> None:
        supplied = request.headers.get("X-Device-Token", "")
        if not supplied or not hmac.compare_digest(supplied, app.config["DEVICE_TOKEN"]):
            abort(401)

    @app.before_request
    def protect_forms() -> None:
        if request.method == "POST" and not request.path.startswith("/api/"):
            supplied = request.form.get("csrf_token", "")
            expected = session.get("csrf_token", "")
            if not supplied or not expected or not hmac.compare_digest(supplied, expected):
                abort(400, "Invalid or missing CSRF token")

    @app.context_processor
    def context() -> dict:
        return {"csrf_token": session.setdefault("csrf_token", secrets.token_urlsafe(24))}

    @app.template_filter("local_datetime")
    def local_datetime(value: str) -> str:
        if not value:
            return "—"
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.astimezone(configured_timezone(app.config["APP_TIMEZONE"])).strftime("%d %b %Y, %I:%M:%S %p")

    @app.get("/")
    def dashboard():
        return render_template(
            "dashboard.html", data=database.monitor_data(), camera=camera.status()
        )

    @app.post("/camera/start")
    def start_camera():
        try:
            status = camera.start(request.form.get("camera_source", "0"))
            flash(status["message"], "success" if status["state"] == "running" else "error")
        except (OSError, ValueError) as error:
            flash(f"Camera could not start: {error}", "error")
        return redirect(url_for("dashboard"))

    @app.post("/camera/stop")
    def stop_camera():
        camera.stop()
        frames.clear()
        flash("Camera stopped.", "success")
        return redirect(url_for("dashboard"))

    @app.post("/sessions/start")
    def start_session():
        try:
            local_start = datetime.strptime(request.form["starts_at"], "%Y-%m-%dT%H:%M")
            start = local_start.replace(tzinfo=configured_timezone(app.config["APP_TIMEZONE"]))
            database.start_session(
                request.form["title"], start,
                int(request.form["duration_minutes"]),
                int(request.form["checkpoint_interval_minutes"]),
                int(request.form["checkpoint_window_minutes"]),
            )
            flash("Attendance session and checkpoints started.", "success")
        except (KeyError, ValueError, sqlite3.IntegrityError) as error:
            flash(str(error), "error")
        return redirect(url_for("dashboard"))

    @app.post("/sessions/<int:session_id>/close")
    def close_session(session_id: int):
        database.close_session(session_id)
        flash("Attendance session closed.", "success")
        return redirect(url_for("dashboard"))

    @app.get("/export.csv")
    def export_csv():
        data = database.monitor_data(log_limit=0)
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Registration number", "Student", "Identity label", "Checkpoint", "Recognized at", "Similarity"])
        for row in data["attendance"]:
            registration = row["registration_number"]
            if registration and registration.startswith("AUTO-"):
                registration = ""
            writer.writerow([
                registration, row["display_name"], row["identity_label"],
                row["checkpoint_number"], row["recognized_at"], f"{row['similarity']:.4f}",
            ])
        return Response(output.getvalue(), mimetype="text/csv", headers={"Content-Disposition": "attachment; filename=attendance.csv"})

    @app.get("/api/monitor")
    def monitor_api():
        data = database.monitor_data()
        data["camera"] = camera.status()
        return jsonify(data)

    @app.post("/api/recognition-events")
    def camera_event():
        require_device()
        event = request.get_json(silent=True)
        if not isinstance(event, dict):
            return jsonify({"error": "JSON object required"}), 400
        try:
            result = database.record_recognition_event(event)
        except (TypeError, ValueError) as error:
            return jsonify({"error": str(error)}), 400
        return jsonify(asdict(result)), 400 if result.outcome == "rejected" else 200

    @app.route("/api/camera-frame", methods=["GET", "POST"])
    def camera_frame():
        if request.method == "POST":
            require_device()
            if request.mimetype != "image/jpeg" or not request.data:
                return jsonify({"error": "JPEG frame required"}), 400
            frames.put(bytes(request.data))
            return "", 204
        jpeg = frames.get()
        if jpeg is None:
            return "", 404
        return Response(jpeg, mimetype="image/jpeg", headers={"Cache-Control": "no-store, max-age=0"})

    @app.get("/camera-stream")
    def camera_stream():
        def generate():
            sequence = -1
            while True:
                jpeg, sequence = frames.wait_for_next(sequence)
                if jpeg is None:
                    continue
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii")
                    + jpeg
                    + b"\r\n"
                )

        return Response(
            generate(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store, max-age=0"},
        )

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
