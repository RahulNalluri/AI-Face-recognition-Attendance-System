"""Integration tests for the login-free attendance monitor."""

from __future__ import annotations

import re
import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from attendance_db import AttendanceDatabase  # noqa: E402
from camera_reliability import connect_camera  # noqa: E402
from recognition_validation import summarize_records  # noqa: E402
from web_app import LatestFrame, create_app  # noqa: E402


class FakeCapture:
    def __init__(self, opened: bool, frame=None) -> None:
        self.opened = opened
        self.frame = frame
        self.released = False

    def isOpened(self) -> bool:
        return self.opened

    def read(self):
        return self.frame is not None, self.frame

    def set(self, *_args) -> bool:
        return True

    def release(self) -> None:
        self.released = True


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
        self.labels_path = Path(self.temporary.name) / "labels.json"
        self.labels_path.write_text(
            json.dumps({"0": "student_one", "1": "student_two"}), encoding="utf-8"
        )
        self.camera = FakeCameraManager()
        self.validation_report_path = Path(self.temporary.name) / "recognition_validation.json"
        self.app = create_app(
            self.database_path, camera_manager=self.camera, labels_path=self.labels_path,
            validation_report_path=self.validation_report_path,
        )
        self.app.config.update(TESTING=True, DEVICE_TOKEN="camera-test-token")
        self.client = self.app.test_client()
        self.start = datetime.now(timezone.utc) - timedelta(minutes=1)
        self.session_id = self.database.start_session("Artificial Intelligence", self.start, 130, 65, 10)

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
        self.assertIn(b"Auto detect", response.data)
        self.assertIn(b"Preview", response.data)
        self.assertIn(b"Recoveries", response.data)
        self.assertIn(b"Use 8-minute test preset", response.data)
        self.assertNotIn(b"Register student", response.data)
        self.assertNotIn(b"Password", response.data)

    def test_recognition_validation_metrics_and_dashboard(self) -> None:
        records = [
            {"expected": "student_one", "predicted": "student_one", "detected": True, "similarity": 0.82},
            {"expected": "student_two", "predicted": "student_two", "detected": True, "similarity": 0.79},
            {"expected": "Unknown", "predicted": "Unknown", "detected": True, "similarity": 0.31},
            {"expected": "Unknown", "predicted": "Unknown", "detected": True, "similarity": 0.28},
        ]
        report = summarize_records(records, ["student_one", "student_two"])
        self.assertEqual(report["status"], "passed")
        self.assertTrue(report["deployment_ready"])
        report.update({
            "model": "OpenCV SFace 2021dec", "classifier": "knn_1",
            "locked_unknown_threshold": 0.49,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        })
        self.validation_report_path.write_text(json.dumps(report), encoding="utf-8")
        page = self.client.get("/validation")
        self.assertEqual(page.status_code, 200)
        self.assertIn(b"Deployment ready", page.data)
        self.assertIn(b"Student One", page.data)

    def test_validation_is_incomplete_without_real_unknown_people(self) -> None:
        report = summarize_records(
            [{"expected": "student_one", "predicted": "student_one", "detected": True, "similarity": 0.8}],
            ["student_one"],
        )
        self.assertEqual(report["status"], "incomplete")
        self.assertFalse(report["deployment_ready"])
        self.assertIn("unknown-person images are missing", report["notices"][0].lower())

    def test_attendance_is_marked_once_per_checkpoint(self) -> None:
        first = self.post_event(self.event(self.start + timedelta(minutes=1)))
        self.assertEqual(first.get_json()["outcome"], "marked")
        duplicate = self.post_event(self.event(self.start + timedelta(minutes=2)))
        self.assertEqual(duplicate.get_json()["outcome"], "already_marked")
        gap = self.post_event(self.event(self.start + timedelta(minutes=20)))
        self.assertEqual(gap.get_json()["outcome"], "no_active_checkpoint")
        second_time = self.start + timedelta(minutes=66)
        second = self.post_event(self.event(second_time))
        self.assertEqual(second.get_json()["outcome"], "marked")
        second_duplicate = self.post_event(self.event(self.start + timedelta(minutes=67)))
        self.assertEqual(second_duplicate.get_json()["outcome"], "already_marked")
        self.assertNotEqual(first.get_json()["checkpoint_id"], second.get_json()["checkpoint_id"])
        data = self.database.monitor_data()
        self.assertEqual(len(data["attendance"]), 2)
        self.assertEqual(
            [row["outcome"] for row in reversed(data["logs"][:5])],
            ["marked", "already_marked", "no_active_checkpoint", "marked", "already_marked"],
        )

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
        self.assertEqual(len(data["students"]), 2)

    def test_model_labels_are_synchronized_and_snapshotted(self) -> None:
        self.assertEqual(
            [row["identity_label"] for row in self.database.students()],
            ["student_one", "student_two"],
        )
        detail = self.database.session_detail(self.session_id)
        self.assertEqual(len(detail["roster"]), 2)
        self.assertEqual(
            {row["identity_label_snapshot"] for row in detail["roster"]},
            {"student_one", "student_two"},
        )

    def test_session_history_matrix_and_checkpoint_csv(self) -> None:
        self.post_event(self.event())
        history = self.client.get("/sessions")
        self.assertEqual(history.status_code, 200)
        self.assertIn(b"Artificial Intelligence", history.data)
        detail = self.client.get(f"/sessions/{self.session_id}")
        self.assertEqual(detail.status_code, 200)
        self.assertIn(b"Student One", detail.data)
        self.assertIn(b"Student Two", detail.data)
        self.assertIn(b"present", detail.data)
        exported = self.client.get(f"/sessions/{self.session_id}/export.csv")
        self.assertEqual(exported.status_code, 200)
        self.assertIn(b"Checkpoint 1", exported.data)
        self.assertIn(b"student_one", exported.data)
        self.assertIn(b"student_two", exported.data)

    def test_closed_checkpoints_calculate_absence_and_percentage(self) -> None:
        past_start = datetime.now(timezone.utc) - timedelta(minutes=12)
        session_id = self.database.start_session("Completed class", past_start, 8, 3, 2)
        marked = self.post_event(self.event(past_start + timedelta(minutes=1)))
        self.assertEqual(marked.get_json()["outcome"], "marked")
        detail = self.database.session_detail(session_id)
        self.assertEqual(detail["completed_checkpoint_count"], 3)
        self.assertEqual(detail["attendance_percentage"], 16.7)
        first = next(row for row in detail["roster"] if row["identity_label_snapshot"] == "student_one")
        second = next(row for row in detail["roster"] if row["identity_label_snapshot"] == "student_two")
        self.assertEqual([cell["status"] for cell in first["cells"]], ["present", "absent", "absent"])
        self.assertEqual([cell["status"] for cell in second["cells"]], ["absent", "absent", "absent"])

    def test_manual_corrections_preserve_evidence_and_create_audit_trail(self) -> None:
        past_start = datetime.now(timezone.utc) - timedelta(minutes=12)
        session_id = self.database.start_session("Correction test", past_start, 8, 3, 2)
        marked = self.post_event(self.event(past_start + timedelta(minutes=1)))
        self.assertEqual(marked.get_json()["outcome"], "marked")
        detail = self.database.session_detail(session_id)
        student = next(row for row in detail["roster"] if row["identity_label_snapshot"] == "student_one")
        checkpoint = detail["checkpoints"][0]

        page = self.client.get(f"/sessions/{session_id}")
        token = re.search(rb'name="csrf_token" value="([^"]+)"', page.data).group(1).decode()
        corrected = self.client.post(
            f"/sessions/{session_id}/attendance/adjust",
            data={
                "csrf_token": token, "checkpoint_id": checkpoint["id"],
                "student_id": student["student_id"], "status": "absent",
                "reason": "Approved medical leave",
            },
            follow_redirects=True,
        )
        self.assertIn(b"Attendance corrected to absent", corrected.data)
        detail = self.database.session_detail(session_id)
        student = next(row for row in detail["roster"] if row["identity_label_snapshot"] == "student_one")
        self.assertEqual(student["cells"][0]["status"], "absent")
        self.assertEqual(student["cells"][0]["automatic_status"], "present")
        self.assertIsNotNone(student["cells"][0]["record"])
        self.assertEqual(detail["audit"][0]["reason"], "Approved medical leave")
        exported = self.client.get(f"/sessions/{session_id}/export.csv")
        self.assertIn(b"absent (manual)", exported.data)
        history_item = next(row for row in self.database.session_history() if row["id"] == session_id)
        self.assertEqual(history_item["attendance_percentage"], 0.0)

        restored = self.client.post(
            f"/sessions/{session_id}/attendance/adjust",
            data={
                "csrf_token": token, "checkpoint_id": checkpoint["id"],
                "student_id": student["student_id"], "status": "automatic",
                "reason": "Restored recognition result",
            },
            follow_redirects=True,
        )
        self.assertIn(b"Attendance corrected to present", restored.data)
        detail = self.database.session_detail(session_id)
        student = next(row for row in detail["roster"] if row["identity_label_snapshot"] == "student_one")
        self.assertEqual(student["cells"][0]["status"], "present")
        self.assertIsNone(student["cells"][0]["override"])
        self.assertEqual(len(detail["audit"]), 2)

    def test_open_checkpoint_cannot_be_manually_corrected(self) -> None:
        detail = self.database.session_detail(self.session_id)
        student = detail["roster"][0]
        with self.assertRaisesRegex(ValueError, "only after the checkpoint window closes"):
            self.database.adjust_attendance(
                self.session_id, detail["checkpoints"][0]["id"], student["student_id"],
                "present", "Manual early correction",
            )

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

    def test_camera_health_moves_from_starting_to_running_after_first_frame(self) -> None:
        self.camera.start("0")
        starting = self.client.get("/api/monitor").get_json()["camera"]
        self.assertEqual(starting["state"], "starting")
        self.assertFalse(starting["preview"]["fresh"])

        self.client.post(
            "/api/camera-frame", data=b"\xff\xd8fresh\xff\xd9", content_type="image/jpeg",
            headers={"X-Device-Token": "camera-test-token"},
        )
        running = self.client.get("/api/monitor").get_json()["camera"]
        self.assertEqual(running["state"], "running")
        self.assertTrue(running["preview"]["fresh"])
        health = self.client.get("/health").get_json()
        self.assertEqual(health["status"], "ok")
        self.assertTrue(health["camera"]["preview"]["fresh"])

    def test_stale_preview_is_not_served(self) -> None:
        now = [100.0]
        latest = LatestFrame(stale_after_seconds=4.0, clock=lambda: now[0])
        latest.put(b"jpeg")
        self.assertEqual(latest.get(fresh_only=True), b"jpeg")
        now[0] = 105.0
        self.assertIsNone(latest.get(fresh_only=True))
        self.assertFalse(latest.status()["fresh"])

    def test_camera_connection_retries_until_a_frame_arrives(self) -> None:
        captures = [FakeCapture(False), FakeCapture(True), FakeCapture(True, frame="frame")]
        delays = []

        def factory(_source):
            return captures.pop(0)

        capture, frame, attempt = connect_camera(
            0, attempts=3, delay_seconds=0.25,
            capture_factory=factory, sleeper=delays.append,
        )
        self.assertEqual(frame, "frame")
        self.assertEqual(attempt, 3)
        self.assertFalse(capture.released)
        self.assertEqual(delays, [0.25, 0.25])

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
