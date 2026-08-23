"""Integration tests for the login-free attendance monitor."""

from __future__ import annotations

import re
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from attendance_db import AttendanceDatabase  # noqa: E402
from web_app import create_app  # noqa: E402


class FakeCameraManager:
    def __init__(self) -> None:
        self.state = "stopped"
        self.source = None

    def status(self) -> dict:
        return {"state": self.state, "message": f"Camera is {self.state}", "recent_output": []}

    def start(self, source: str = "0") -> dict:
        self.source = source
        self.state = "running"
        return self.status()

    def stop(self) -> dict:
        self.state = "stopped"
        return self.status()


class AttendanceMonitorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary.name) / "attendance.db"
        self.database = AttendanceDatabase(self.database_path)
        self.database.initialize()
        self.start = datetime.now(timezone.utc) - timedelta(minutes=1)
        self.session_id = self.database.start_session("Artificial Intelligence", self.start, 130, 65, 10)
        self.camera = FakeCameraManager()
        self.app = create_app(self.database_path, camera_manager=self.camera)
        self.app.config.update(TESTING=True, DEVICE_TOKEN="camera-test-token")
        self.client = self.app.test_client()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def event(self, timestamp: datetime | None = None) -> dict:
        return {
            "event": "recognition_and_liveness_passed",
            "timestamp_utc": (timestamp or datetime.now(timezone.utc)).isoformat(),
            "identity": "student_one",
            "similarity": 0.91,
            "liveness": "passed",
        }

    def post_event(self, event: dict):
        return self.client.post(
            "/api/recognition-events", json=event,
            headers={"X-Device-Token": "camera-test-token"},
        )

    def test_dashboard_has_no_login_and_shows_camera_monitor(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Live attendance", response.data)
        self.assertIn(b"camera-feed", response.data)
        self.assertIn(b"Start Camera", response.data)
        self.assertNotIn(b"Register student", response.data)
        self.assertNotIn(b"Password", response.data)

    def test_attendance_is_marked_once_per_checkpoint(self) -> None:
        first = self.post_event(self.event())
        self.assertEqual(first.get_json()["outcome"], "marked")
        duplicate = self.post_event(self.event())
        self.assertEqual(duplicate.get_json()["outcome"], "already_marked")
        second_time = self.start + timedelta(minutes=66)
        second = self.post_event(self.event(second_time))
        self.assertEqual(second.get_json()["outcome"], "marked")
        self.assertNotEqual(first.get_json()["checkpoint_id"], second.get_json()["checkpoint_id"])

    def test_untrusted_and_unknown_events_are_blocked(self) -> None:
        self.assertEqual(self.client.post("/api/recognition-events", json=self.event()).status_code, 401)
        bad = self.event()
        bad["liveness"] = "failed"
        self.assertEqual(self.post_event(bad).get_json()["outcome"], "rejected")
        unknown = self.event()
        unknown["identity"] = "Unknown"
        self.assertEqual(self.post_event(unknown).get_json()["outcome"], "rejected")

    def test_live_log_api_contains_student_details(self) -> None:
        self.post_event(self.event())
        data = self.client.get("/api/monitor").get_json()
        self.assertEqual(data["logs"][0]["display_name"], "Student One")
        self.assertEqual(data["logs"][0]["registration_number"], "AUTO-student_one")
        self.assertEqual(len(data["attendance"]), 1)
        self.assertEqual(len(data["students"]), 1)

    def test_camera_frame_requires_device_token_and_is_available_to_ui(self) -> None:
        jpeg = b"\xff\xd8test-frame\xff\xd9"
        self.assertEqual(self.client.post("/api/camera-frame", data=jpeg, content_type="image/jpeg").status_code, 401)
        stored = self.client.post(
            "/api/camera-frame", data=jpeg, content_type="image/jpeg",
            headers={"X-Device-Token": "camera-test-token"},
        )
        self.assertEqual(stored.status_code, 204)
        fetched = self.client.get("/api/camera-frame")
        self.assertEqual(fetched.status_code, 200)
        self.assertEqual(fetched.data, jpeg)

    def test_camera_can_be_started_and_stopped_from_dashboard(self) -> None:
        page = self.client.get("/")
        token = re.search(rb'name="csrf_token" value="([^"]+)"', page.data).group(1).decode()
        started = self.client.post("/camera/start", data={"csrf_token": token, "camera_source": "1"})
        self.assertEqual(started.status_code, 302)
        self.assertEqual(self.camera.state, "running")
        self.assertEqual(self.camera.source, "1")
        stopped = self.client.post("/camera/stop", data={"csrf_token": token})
        self.assertEqual(stopped.status_code, 302)
        self.assertEqual(self.camera.state, "stopped")


if __name__ == "__main__":
    unittest.main()
