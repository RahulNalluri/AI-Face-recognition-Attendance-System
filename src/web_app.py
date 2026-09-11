"""Local attendance monitor UI and protected camera ingestion API."""

from __future__ import annotations

import csv
import atexit
import hmac
import io
import json
import os
import secrets
import sqlite3
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from flask import Flask, Response, abort, flash, jsonify, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

from attendance_db import AttendanceDatabase, DEFAULT_DATABASE
from timetable import Timetable, TimetableScheduler, WEEKDAYS
from section_setup import SectionSetup, normalize_grid

ROOT = Path(__file__).resolve().parent.parent
WEB_ROOT = ROOT / "web"
INSTANCE_ROOT = ROOT / "instance"
DEFAULT_LABELS_PATH = ROOT / "models" / "sface" / "labels.json"
DEFAULT_VALIDATION_REPORT = ROOT / "artifacts" / "evaluation" / "recognition_validation" / "report.json"


class LatestFrame:
    def __init__(self, stale_after_seconds: float = 4.0, clock=time.monotonic) -> None:
        self._condition = threading.Condition()
        self._jpeg: bytes | None = None
        self._sequence = 0
        self._updated_at: float | None = None
        self._stale_after_seconds = stale_after_seconds
        self._clock = clock

    def put(self, jpeg: bytes) -> None:
        with self._condition:
            self._jpeg = jpeg
            self._sequence += 1
            self._updated_at = self._clock()
            self._condition.notify_all()

    def get(self, fresh_only: bool = False) -> bytes | None:
        with self._condition:
            if fresh_only and not self._is_fresh_unlocked():
                return None
            return self._jpeg

    def clear(self) -> None:
        with self._condition:
            self._jpeg = None
            self._sequence += 1
            self._updated_at = None
            self._condition.notify_all()

    def _is_fresh_unlocked(self) -> bool:
        return (
            self._jpeg is not None and self._updated_at is not None
            and self._clock() - self._updated_at <= self._stale_after_seconds
        )

    def status(self) -> dict:
        with self._condition:
            age = None if self._updated_at is None else max(0.0, self._clock() - self._updated_at)
            return {
                "has_frame": self._jpeg is not None,
                "fresh": self._is_fresh_unlocked(),
                "age_seconds": None if age is None else round(age, 1),
                "sequence": self._sequence,
            }

    def wait_for_next(self, previous_sequence: int) -> tuple[bytes | None, int]:
        with self._condition:
            changed = self._condition.wait_for(
                lambda: self._sequence != previous_sequence, timeout=2.0
            )
            return (self._jpeg if changed and self._is_fresh_unlocked() else None), self._sequence


class CameraProcessManager:
    """Own exactly one background recognition process for the local dashboard."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None
        self._lines: deque[str] = deque(maxlen=30)
        self._requested_stop = False
        self._phase = "stopped"
        self._source: str | None = None
        self._started_at: float | None = None
        self._reconnect_count = 0

    def _drain_output(self, process: subprocess.Popen[str]) -> None:
        if process.stdout is None:
            return
        for line in process.stdout:
            cleaned = line.strip()
            if cleaned:
                with self._lock:
                    self._lines.append(cleaned)
                    try:
                        payload = json.loads(cleaned)
                    except json.JSONDecodeError:
                        payload = None
                    if isinstance(payload, dict) and payload.get("camera_status"):
                        self._phase = str(payload["camera_status"])
                        if payload.get("source") is not None:
                            self._source = str(payload["source"])
                        self._reconnect_count = int(payload.get("reconnect_count", self._reconnect_count))

    def start(self, source: str = "auto") -> dict:
        normalized = source.strip()
        if normalized.lower() != "auto" and (
            not normalized.isdecimal() or not 0 <= int(normalized) <= 9
        ):
            raise ValueError("Camera source must be auto or a number from 0 to 9")
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                return self.status_unlocked()
            self._lines.clear()
            self._requested_stop = False
            self._phase = "starting"
            self._source = normalized
            self._started_at = time.monotonic()
            self._reconnect_count = 0
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
            return {
                "state": "stopped", "phase": "stopped", "message": "Camera is stopped",
                "recent_output": [], "source": self._source, "reconnect_count": self._reconnect_count,
            }
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
        return {
            "state": state, "phase": self._phase, "message": message,
            "recent_output": list(self._lines)[-5:], "source": self._source,
            "reconnect_count": self._reconnect_count, "pid": process.pid,
            "uptime_seconds": (
                None if self._started_at is None else round(time.monotonic() - self._started_at, 1)
            ),
        }

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
        with self._lock:
            self._phase = "stopped"
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


def model_labels(path: Path | str | None) -> list[str]:
    if path is None or not Path(path).exists():
        return []
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    values = list(payload.values()) if isinstance(payload, dict) else list(payload)
    return [str(value) for value in values]


def ensure_demo_logins(database: AttendanceDatabase, credentials_directory: Path) -> list[Path]:
    """Create local admin, faculty and Rahul accounts once with private credentials."""
    credentials_directory.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    def write_credentials(filename: str, role: str, username: str, password: str) -> None:
        path = credentials_directory / filename
        path.write_text(
            f"role={role}\nusername={username}\npassword={password}\n", encoding="utf-8"
        )
        created.append(path)

    for role, username, name in (
        ("admin", "admin", "System Administrator"),
        ("faculty", "faculty", "Faculty User"),
    ):
        if database.staff_account_by_username(username) is None:
            password = secrets.token_urlsafe(12)
            database.save_staff_account(username, name, role, generate_password_hash(password))
            write_credentials(f"{role}_demo_credentials.txt", role, username, password)

    if database.student_account_by_username("rahul") is None:
        student = database.student_by_identity("rahul")
        if student is not None:
            password = secrets.token_urlsafe(12)
            database.save_student_account(student["id"], "rahul", generate_password_hash(password))
            write_credentials("student_demo_credentials.txt", "student", "rahul", password)
    return created


def create_app(
    database_path: Path | str | None = None,
    camera_manager: CameraProcessManager | None = None,
    labels_path: Path | str | None = DEFAULT_LABELS_PATH,
    validation_report_path: Path | str | None = DEFAULT_VALIDATION_REPORT,
    enable_demo_logins: bool = False,
    demo_credentials_path: Path | str | None = None,
) -> Flask:
    app = Flask(__name__, template_folder=str(WEB_ROOT / "templates"), static_folder=str(WEB_ROOT / "static"))
    app.config.update(
        SECRET_KEY=local_secret("flask_secret.key", "ATTENDANCE_SECRET_KEY"),
        DATABASE_PATH=str(database_path or os.environ.get("ATTENDANCE_DATABASE", DEFAULT_DATABASE)),
        DEVICE_TOKEN=local_secret("device_token.txt", "ATTENDANCE_DEVICE_TOKEN"),
        APP_TIMEZONE=os.environ.get("ATTENDANCE_TIMEZONE", "Asia/Kolkata"),
        VALIDATION_REPORT_PATH=str(validation_report_path) if validation_report_path else "",
        MAX_CONTENT_LENGTH=2 * 1024 * 1024,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        AUTH_REQUIRED=True,
    )
    app.config["DEMO_CREDENTIAL_DIRECTORY"] = str(
        Path(demo_credentials_path) if demo_credentials_path
        else Path(app.config["DATABASE_PATH"]).parent
    )
    database = AttendanceDatabase(app.config["DATABASE_PATH"])
    database.initialize()
    database.sync_identities(model_labels(labels_path))
    database.backfill_missing_rosters()
    if enable_demo_logins:
        credential_directory = Path(demo_credentials_path or INSTANCE_ROOT)
        for credential_file in ensure_demo_logins(database, credential_directory):
            print(f"Demo login created. Credentials saved to {credential_file}")
    app.extensions["attendance_database"] = database
    timetable = Timetable(database)
    sections = SectionSetup(database)
    scheduler = TimetableScheduler(timetable)
    app.extensions["timetable"] = timetable
    app.extensions["timetable_scheduler"] = scheduler
    frames = LatestFrame()
    camera = camera_manager or CameraProcessManager()
    app.extensions["camera_manager"] = camera
    app.extensions["latest_camera_frame"] = frames
    if camera_manager is None:
        atexit.register(camera.stop)
    failed_logins: dict[str, deque[float]] = {}
    login_lock = threading.Lock()

    def camera_status() -> dict:
        process_status = dict(camera.status())
        preview = frames.status()
        process_status["preview"] = preview
        if process_status.get("state") != "running":
            return process_status
        phase = process_status.get("phase", "running")
        if preview["fresh"]:
            process_status["state"] = "running"
            process_status["message"] = "Camera and live preview are healthy"
        elif phase in {"degraded", "reconnecting"} or preview["has_frame"]:
            process_status["state"] = "recovering"
            age = preview["age_seconds"]
            process_status["message"] = (
                f"Camera is reconnecting; last preview frame was {age:.1f}s ago"
                if age is not None else "Camera is reconnecting"
            )
        else:
            process_status["state"] = "starting"
            process_status["message"] = "Camera process started; waiting for the first frame"
        return process_status

    def require_device() -> None:
        supplied = request.headers.get("X-Device-Token", "")
        if not supplied or not hmac.compare_digest(supplied, app.config["DEVICE_TOKEN"]):
            abort(401)

    def current_user() -> dict | None:
        account_id, role = session.get("account_id"), session.get("user_role")
        if not isinstance(account_id, int) or role not in {"admin", "faculty", "student"}:
            return None
        account = (
            database.student_account(account_id) if role == "student"
            else database.staff_account(account_id)
        )
        if account is None or (role != "student" and account["role"] != role):
            return None
        account = dict(account)
        account["role"] = role
        return account

    @app.before_request
    def protect_forms() -> None:
        if request.method == "POST" and not request.path.startswith("/api/"):
            supplied = request.form.get("csrf_token", "")
            expected = session.get("csrf_token", "")
            if not supplied or not expected or not hmac.compare_digest(supplied, expected):
                abort(400, "Invalid or missing CSRF token")

    @app.before_request
    def protect_pages() -> None:
        if not app.config["AUTH_REQUIRED"]:
            return None
        endpoint = request.endpoint or ""
        if endpoint in {"login", "legacy_student_login", "static", "health"}:
            return None
        if endpoint in {"camera_event"} or (endpoint == "camera_frame" and request.method == "POST"):
            return None
        user = current_user()
        if user is None:
            session.pop("account_id", None)
            session.pop("user_role", None)
            if request.path.startswith("/api/"):
                abort(401)
            return redirect(url_for("login"))
        if endpoint in {"logout", "legacy_student_logout", "change_password"}:
            return None
        if endpoint == "student_portal":
            if user["role"] != "student":
                return redirect(url_for("dashboard"))
            return None
        if user["role"] == "student":
            return redirect(url_for("student_portal"))
        return None

    @app.context_processor
    def context() -> dict:
        return {"csrf_token": session.setdefault("csrf_token", secrets.token_urlsafe(24)),
                "camera_section": timetable.camera_section(),
                "current_user": current_user()}

    @app.template_filter("local_datetime")
    def local_datetime(value: str) -> str:
        if not value:
            return "—"
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.astimezone(configured_timezone(app.config["APP_TIMEZONE"])).strftime("%d %b %Y, %I:%M:%S %p")

    @app.route("/login", methods=["GET", "POST"])
    def login():
        signed_in = current_user()
        if signed_in is not None:
            return redirect(url_for("student_portal" if signed_in["role"] == "student" else "dashboard"))
        if request.method == "POST":
            role = request.form.get("role", "").strip().lower()
            username = request.form.get("username", "").strip()
            password = request.form.get("password", "")
            key = f"{request.remote_addr or 'local'}:{role}:{username.casefold()}"
            now_tick = time.monotonic()
            with login_lock:
                attempts = failed_logins.setdefault(key, deque())
                while attempts and now_tick - attempts[0] > 300:
                    attempts.popleft()
                limited = len(attempts) >= 5
            account = None
            if not limited and role == "student":
                account = database.student_account_by_username(username)
            elif not limited and role in {"admin", "faculty"}:
                candidate = database.staff_account_by_username(username)
                account = candidate if candidate and candidate["role"] == role else None
            valid = bool(
                account and account["enabled"] and (role != "student" or account["active"])
                and check_password_hash(account["password_hash"], password)
            )
            if not valid:
                with login_lock:
                    if not limited:
                        failed_logins[key].append(now_tick)
                message = (
                    "Too many attempts. Please wait five minutes and try again."
                    if limited else "Invalid username or password."
                )
                return render_template(
                    "login.html", error=message, username=username, selected_role=role,
                ), 429 if limited else 401
            with login_lock:
                failed_logins.pop(key, None)
            csrf_token = session.get("csrf_token") or secrets.token_urlsafe(24)
            session.clear()
            session["csrf_token"] = csrf_token
            session["account_id"] = int(account["id"])
            session["user_role"] = role
            if role == "student":
                database.record_student_login(int(account["id"]))
            else:
                database.record_staff_login(int(account["id"]))
            return redirect(url_for("student_portal" if role == "student" else "dashboard"))
        return render_template("login.html", error=None, username="", selected_role="student")

    @app.get("/student/login")
    def legacy_student_login():
        return redirect(url_for("login"))

    @app.post("/logout")
    def logout():
        session.clear()
        session["csrf_token"] = secrets.token_urlsafe(24)
        flash("You have been signed out.", "success")
        return redirect(url_for("login"))

    @app.post("/student/logout")
    def legacy_student_logout():
        return logout()

    @app.route("/change-password", methods=["GET", "POST"])
    def change_password():
        user = current_user()
        if request.method == "POST":
            old_password = request.form.get("current_password", "")
            new_password = request.form.get("new_password", "")
            confirmation = request.form.get("confirm_password", "")
            error = None
            if not check_password_hash(user["password_hash"], old_password):
                error = "Current password is incorrect."
            elif len(new_password) < 8:
                error = "New password must contain at least 8 characters."
            elif new_password != confirmation:
                error = "New password and confirmation do not match."
            elif check_password_hash(user["password_hash"], new_password):
                error = "Choose a password different from your current password."
            if error:
                return render_template("change_password.html", error=error), 400
            password_hash = generate_password_hash(new_password)
            if user["role"] == "student":
                database.update_student_password(int(user["id"]), password_hash)
            else:
                database.update_staff_password(int(user["id"]), password_hash)
            credential_file = Path(app.config["DEMO_CREDENTIAL_DIRECTORY"]) / (
                "student_demo_credentials.txt" if user["role"] == "student"
                else f"{user['role']}_demo_credentials.txt"
            )
            credential_file.unlink(missing_ok=True)
            flash("Password changed successfully. The old demo credential file was removed.", "success")
            return redirect(url_for("student_portal" if user["role"] == "student" else "dashboard"))
        return render_template("change_password.html", error=None)

    @app.get("/student")
    def student_portal():
        student_account = current_user()
        report = database.student_portal_data(int(student_account["student_id"]))
        memberships = database.student_group_ids(int(student_account["student_id"]))
        upcoming = [row for row in timetable.upcoming() if int(row["group_id"]) in memberships]
        return render_template(
            "student_portal.html", account=student_account, report=report,
            upcoming=upcoming[:12],
        )

    @app.get("/")
    def dashboard():
        scheduler.check_due()
        return render_template(
            "dashboard.html", data=database.monitor_data(), camera=camera_status(),
            groups=database.class_groups(), selected_group_id=request.args.get("group_id", type=int) or timetable.camera_section()["group_id"],
        )

    @app.get("/classes")
    def class_groups_page():
        return render_template("classes.html", groups=database.class_groups())

    def class_editor(group_id: int | None = None):
        group = database.class_group(group_id) if group_id is not None else {
            "id": None, "name": "", "description": "", "student_ids": [],
        }
        if group is None:
            abort(404)
        profile = sections.profile(group_id, app.config["APP_TIMEZONE"])
        status_code = 200
        if request.method == "POST":
            try:
                selected = [int(value) for value in request.form.getlist("student_ids")]
                group.update({
                    "name": request.form.get("name", ""),
                    "description": request.form.get("description", ""), "student_ids": selected,
                })
                if "grid_json" in request.form:
                    raw = json.loads(request.form["grid_json"])
                    if not isinstance(raw, dict) or not isinstance(raw.get("periods"), list) or not isinstance(raw.get("subjects"), list):
                        raise ValueError("Invalid timetable grid")
                    profile.update({field: request.form.get(field, "") for field in ("department", "semester", "academic_year")})
                    profile["grid"] = normalize_grid(raw)
                    saved_id = sections.save(
                        name=group["name"], description=group["description"], student_ids=selected,
                        group_id=group_id, **profile,
                    )
                else:
                    saved_id = database.save_class_group(group["name"], group["description"], selected, group_id)
                scheduler.check_due(force=True)
                flash("Section and roster saved. Existing sessions are unchanged.", "success")
                return redirect(url_for("edit_class_group", group_id=saved_id))
            except (ValueError, sqlite3.IntegrityError) as error:
                flash(str(error), "error")
                status_code = 400
        return render_template("class_group.html", group=group, students=database.students(), profile=profile), status_code

    @app.route("/classes/new", methods=["GET", "POST"])
    def new_class_group():
        return class_editor()

    @app.route("/classes/<int:group_id>", methods=["GET", "POST"])
    def edit_class_group(group_id: int):
        return class_editor(group_id)

    @app.get("/timetable")
    def timetable_page():
        scheduler.check_due()
        return render_template(
            "timetable.html", entries=timetable.entries(), upcoming=timetable.upcoming(),
            runs=timetable.runs(), weekdays=WEEKDAYS, scheduler=scheduler.status(),
            groups=database.class_groups(),
        )

    @app.post("/camera/section")
    def select_camera_section():
        try:
            value = request.form.get("group_id", "")
            timetable.select_camera_section(int(value) if value else None)
            scheduler.check_due(force=True)
            flash("Camera section updated. Only its enabled timetable will run automatically.", "success")
        except (ValueError, sqlite3.IntegrityError) as error:
            flash(str(error), "error")
        return redirect(url_for("dashboard"))

    def timetable_editor(entry_id=None):
        managed_group = timetable.managed_group(entry_id) if entry_id is not None else None
        if managed_group is not None:
            return redirect(url_for("edit_class_group", group_id=managed_group, _anchor="weekly-grid"))
        entry = timetable.entry(entry_id) if entry_id is not None else {
            "id": None, "group_id": request.args.get("group_id", type=int), "title": "",
            "weekday": 0, "start_time": "09:30", "duration_minutes": 130,
            "checkpoint_interval_minutes": 65, "checkpoint_window_minutes": 10,
            "timezone": app.config["APP_TIMEZONE"], "enabled": True,
        }
        if entry is None:
            abort(404)
        status_code = 200
        if request.method == "POST":
            # Preserve submitted values on validation errors, including checkbox state.
            for field in ("title", "start_time", "group_id", "weekday", "duration_minutes",
                          "checkpoint_interval_minutes", "checkpoint_window_minutes"):
                entry[field] = request.form.get(field, "")
            entry["enabled"] = request.form.get("enabled") == "on"
            try:
                saved_id = timetable.save(
                    group_id=int(entry["group_id"]), title=entry["title"],
                    weekday=int(entry["weekday"]), start_time=entry["start_time"],
                    duration_minutes=int(entry["duration_minutes"]),
                    checkpoint_interval_minutes=int(entry["checkpoint_interval_minutes"]),
                    checkpoint_window_minutes=int(entry["checkpoint_window_minutes"]),
                    timezone_name=entry["timezone"], enabled=entry["enabled"], entry_id=entry_id,
                )
                scheduler.check_due(force=True)
                flash("Weekly timetable saved. Existing sessions and today's processed occurrence are unchanged.", "success")
                return redirect(url_for("edit_timetable", entry_id=saved_id))
            except (ValueError, sqlite3.IntegrityError) as error:
                flash(str(error), "error")
                status_code = 400
        return render_template(
            "timetable_entry.html", entry=entry, groups=database.class_groups(), weekdays=WEEKDAYS,
        ), status_code

    @app.route("/timetable/new", methods=["GET", "POST"])
    def new_timetable():
        return timetable_editor()

    @app.route("/timetable/<int:entry_id>", methods=["GET", "POST"])
    def edit_timetable(entry_id):
        return timetable_editor(entry_id)

    @app.post("/timetable/<int:entry_id>/toggle")
    def toggle_timetable(entry_id):
        if timetable.entry(entry_id) is None:
            abort(404)
        try:
            enabled = request.form.get("enabled")
            if enabled not in {"0", "1"}:
                raise ValueError("Choose pause or enable")
            timetable.set_enabled(entry_id, enabled == "1")
            scheduler.check_due(force=True)
            flash("Timetable updated. Already-created attendance sessions are unchanged.", "success")
        except (ValueError, sqlite3.IntegrityError) as error:
            flash(str(error), "error")
        return redirect(url_for("timetable_page"))

    @app.get("/sessions")
    def session_history():
        return render_template("history.html", history=database.session_history())

    @app.get("/validation")
    def recognition_validation():
        report_path = Path(app.config["VALIDATION_REPORT_PATH"]) if app.config["VALIDATION_REPORT_PATH"] else None
        report = None
        report_error = None
        if report_path and report_path.exists():
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                report_error = f"Validation report could not be read: {error}"
        return render_template(
            "validation.html", report=report, report_error=report_error,
            report_path=str(report_path) if report_path else "",
        )

    @app.get("/sessions/<int:session_id>")
    def session_detail(session_id: int):
        detail = database.session_detail(session_id)
        if detail is None:
            abort(404)
        return render_template("session_detail.html", detail=detail)

    @app.post("/camera/start")
    def start_camera():
        try:
            status = camera.start(request.form.get("camera_source", "auto"))
            if status["state"] == "running":
                flash("Camera startup requested. Waiting for the first live frame.", "success")
            else:
                flash(status["message"], "error")
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
                group_id=int(request.form["group_id"]) if request.form.get("group_id") else None,
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

    @app.post("/sessions/<int:session_id>/attendance/adjust")
    def adjust_session_attendance(session_id: int):
        try:
            status = database.adjust_attendance(
                session_id=session_id,
                checkpoint_id=int(request.form["checkpoint_id"]),
                student_id=int(request.form["student_id"]),
                new_status=request.form["status"],
                reason=request.form["reason"],
            )
            flash(f"Attendance corrected to {status} and added to the audit trail.", "success")
        except (KeyError, TypeError, ValueError) as error:
            flash(str(error), "error")
        return redirect(url_for("session_detail", session_id=session_id, _anchor="attendance-matrix"))

    @app.get("/export.csv")
    def export_csv():
        data = database.monitor_data(log_limit=0)
        if not data["session"]:
            return Response("No attendance session exists.\n", status=404, mimetype="text/plain")
        return redirect(url_for("export_session_csv", session_id=data["session"]["id"]))

    @app.get("/sessions/<int:session_id>/export.csv")
    def export_session_csv(session_id: int):
        detail = database.session_detail(session_id)
        if detail is None:
            abort(404)
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(
            ["Identity", "Name"]
            + [f"Checkpoint {item['checkpoint_number']}" for item in detail["checkpoints"]]
            + ["Attended", "Completed checkpoints", "Attendance percentage", "Class group"]
        )
        for student in detail["roster"]:
            writer.writerow(
                [student["identity_label_snapshot"], student["display_name_snapshot"]]
                + [
                    cell["status"] + (" (manual)" if cell["override"] else "")
                    for cell in student["cells"]
                ]
                + [
                    student["attended_count"], detail["completed_checkpoint_count"],
                    "" if student["attendance_percentage"] is None else student["attendance_percentage"],
                    # Prevent a class name from being interpreted as a spreadsheet formula.
                    "'" + detail["session"]["group_name_snapshot"]
                    if (detail["session"]["group_name_snapshot"] or "").lstrip().startswith(("=", "+", "-", "@"))
                    else detail["session"]["group_name_snapshot"] or "All enrolled identities",
                ]
            )
        safe_title = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in detail["session"]["title"]
        ).strip("_") or f"session_{session_id}"
        return Response(
            output.getvalue(), mimetype="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{safe_title}_attendance.csv"'},
        )

    @app.get("/api/monitor")
    def monitor_api():
        scheduler.check_due()
        data = database.monitor_data()
        data["camera"] = camera_status()
        data["camera_section"] = timetable.camera_section()
        return jsonify(data)

    @app.post("/api/recognition-events")
    def camera_event():
        require_device()
        event = request.get_json(silent=True)
        if not isinstance(event, dict):
            return jsonify({"error": "JSON object required"}), 400
        try:
            scheduler.check_due(force=True)
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
        jpeg = frames.get(fresh_only=True)
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
        return jsonify({"status": "ok", "camera": camera_status(), "timetable": scheduler.status()})

    return app


app = create_app(enable_demo_logins=__name__ == "__main__")

if __name__ == "__main__":
    scheduler = app.extensions["timetable_scheduler"]
    scheduler.start()
    try:
        app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
    finally:
        scheduler.stop()
