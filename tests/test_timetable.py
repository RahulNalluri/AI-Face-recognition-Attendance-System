"""Deterministic timetable checks use temporary databases, never real attendance."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
import re
import sqlite3
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from attendance_db import AttendanceDatabase
from timetable import Timetable, TimetableScheduler
from web_app import create_app


class TimetableTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.app = create_app(Path(self.temporary.name) / "test.db", labels_path=None)
        self.app.config.update(TESTING=True)
        self.db = self.app.extensions["attendance_database"]
        self.table = self.app.extensions["timetable"]
        self.db.sync_identities(["student_one", "student_two"])
        self.ids = {s["identity_label"]: s["id"] for s in self.db.students()}
        self.group = self.db.save_class_group("CSE-A", "", [self.ids["student_one"]])
        self.table.select_camera_section(self.group)
        # Monday 09:30 in India, independent of the machine's local timezone.
        self.start = datetime(2030, 1, 7, 4, 0, tzinfo=timezone.utc)

    def tearDown(self):
        self.temporary.cleanup()

    def save(self, **changes):
        values = dict(group_id=self.group, title="Artificial Intelligence", weekday=0,
                      start_time="09:30", timezone_name="Asia/Kolkata", duration_minutes=130,
                      checkpoint_interval_minutes=65, checkpoint_window_minutes=10)
        values.update(changes)
        return self.table.save(**values)

    def count(self, table):
        with self.db.session() as connection:
            return connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    def test_weekday_time_and_timezone_boundaries(self):
        self.save()
        self.assertEqual(self.table.run_due(self.start - timedelta(days=1)), [])
        self.assertEqual(self.table.run_due(self.start - timedelta(seconds=1)), [])
        ids = self.table.run_due(self.start)
        self.assertEqual(len(ids), 1)
        detail = self.db.session_detail(ids[0])
        self.assertEqual(detail["session"]["starts_at"], self.start.isoformat())
        self.assertEqual(len(detail["checkpoints"]), 2)
        self.assertEqual(detail["checkpoints"][1]["opens_at"], (self.start + timedelta(minutes=65)).isoformat())
        self.assertEqual(detail["session"]["group_name_snapshot"], "CSE-A")
        self.assertEqual([s["student_id"] for s in detail["roster"]], [self.ids["student_one"]])

    def test_duplicate_restart_close_and_next_week(self):
        entry_id = self.save()
        first = self.table.run_due(self.start)[0]
        restarted = Timetable(AttendanceDatabase(self.db.path))
        self.db.initialize()
        self.assertEqual(restarted.run_due(self.start + timedelta(seconds=20)), [])
        self.db.close_session(first)
        self.assertEqual(restarted.run_due(self.start + timedelta(minutes=1)), [])
        self.table.set_enabled(entry_id, False)
        self.table.set_enabled(entry_id, True)
        self.assertEqual(restarted.run_due(self.start + timedelta(minutes=2)), [])
        second = restarted.run_due(self.start + timedelta(days=7))[0]
        self.assertNotEqual(first, second)
        self.assertEqual(self.count("monitor_sessions"), 2)
        self.assertEqual(self.count("timetable_runs"), 2)

    def test_concurrent_runners_create_exactly_one_session(self):
        self.save()
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda _: Timetable(self.db).run_due(self.start), range(8)))
        self.assertEqual(sum(len(r) for r in results), 1)
        self.assertEqual(self.count("monitor_sessions"), 1)
        self.assertEqual(self.count("timetable_runs"), 1)

    def test_failure_rolls_back_session_and_claim_together(self):
        self.save()
        with self.db.session() as connection:
            connection.execute("""CREATE TRIGGER fail_run BEFORE INSERT ON timetable_runs
                               BEGIN SELECT RAISE(ABORT, 'injected failure'); END""")
        with self.assertRaises(sqlite3.IntegrityError):
            self.table.run_due(self.start)
        self.assertEqual(self.count("monitor_sessions"), 0)
        self.assertEqual(self.count("monitor_checkpoints"), 0)
        self.assertEqual(self.count("session_roster"), 0)
        with self.db.session() as connection:
            connection.execute("DROP TRIGGER fail_run")
        self.assertEqual(len(self.table.run_due(self.start)), 1)

    def test_edits_affect_future_occurrences_only(self):
        entry_id = self.save()
        first = self.table.run_due(self.start)[0]
        self.db.save_class_group("Renamed class", "", [self.ids["student_two"]], self.group)
        self.save(entry_id=entry_id, title="Networks", start_time="10:00", duration_minutes=90)
        self.assertEqual(self.table.run_due(self.start + timedelta(minutes=30)), [])
        second = self.table.run_due(self.start + timedelta(days=7, minutes=30))[0]
        old, new = self.db.session_detail(first), self.db.session_detail(second)
        self.assertEqual(old["session"]["title"], "Artificial Intelligence")
        self.assertEqual(old["session"]["group_name_snapshot"], "CSE-A")
        self.assertEqual(new["session"]["title"], "Networks")
        self.assertEqual(new["roster"][0]["student_id"], self.ids["student_two"])
        old_run = next(r for r in self.table.runs() if r["session_id"] == first)
        self.assertEqual(old_run["title_snapshot"], "Artificial Intelligence")
        self.assertEqual(old_run["group_name_snapshot"], "CSE-A")

    def test_pause_and_empty_roster(self):
        entry_id = self.save(enabled=False)
        self.assertEqual(self.table.run_due(self.start), [])
        self.table.set_enabled(entry_id, True)
        self.db.save_class_group("CSE-A", "", [], self.group)
        self.assertEqual(self.table.run_due(self.start), [])
        self.assertEqual(self.table.runs()[0]["outcome"], "empty_roster")
        self.assertEqual(self.count("monitor_sessions"), 0)
        self.db.save_class_group("CSE-A", "", [self.ids["student_one"]], self.group)
        self.assertEqual(self.table.run_due(self.start + timedelta(minutes=1)), [])

    def test_late_start_does_not_backfill_absences(self):
        self.save()
        self.assertEqual(self.table.run_due(self.start + timedelta(minutes=10)), [])
        self.assertEqual(self.table.runs()[0]["outcome"], "missed")
        self.assertEqual(self.count("monitor_sessions"), 0)
        self.assertEqual(self.count("monitor_attendance"), 0)
        self.assertEqual(self.table.run_due(self.start + timedelta(days=1)), [])
        self.assertEqual(self.count("timetable_runs"), 1)

    def test_restart_during_first_window_keeps_original_schedule(self):
        self.save()
        ids = self.table.run_due(self.start + timedelta(minutes=5))
        detail = self.db.session_detail(ids[0])
        self.assertEqual(detail["checkpoints"][0]["opens_at"], self.start.isoformat())
        self.assertEqual(detail["checkpoints"][0]["closes_at"], (self.start + timedelta(minutes=10)).isoformat())

    def test_conflicting_manual_session_is_preserved(self):
        manual = self.db.start_session("Manual class", self.start, 60, 30, 5, group_id=self.group)
        self.save()
        self.assertEqual(self.table.run_due(self.start), [])
        self.assertEqual(self.table.runs()[0]["outcome"], "conflict")
        self.assertEqual(self.count("monitor_sessions"), 1)
        with self.db.session() as connection:
            row = connection.execute("SELECT status FROM monitor_sessions WHERE id=?", (manual,)).fetchone()
        self.assertEqual(row["status"], "scheduled")

    def test_overlaps_invalid_inputs_and_adjacent_slots(self):
        self.save()
        for changes in [dict(start_time="10:00"), dict(start_time="08:30"),
                        dict(title=""), dict(weekday=7), dict(start_time="25:00"),
                        dict(start_time="23:00"), dict(duration_minutes=0),
                        dict(checkpoint_interval_minutes=131), dict(checkpoint_window_minutes=66),
                        dict(group_id=99999), dict(timezone_name="Bad/Timezone"),
                        dict(timezone_name="UTC")]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.save(**changes)
        self.save(title="Next subject", start_time="11:40")
        paused = self.save(enabled=False)
        with self.assertRaises(ValueError):
            self.table.set_enabled(paused, True)
        self.assertFalse(self.table.entry(paused)["enabled"])
        self.assertEqual(len(self.table.entries()), 3)

    def test_next_occurrences_and_timezone_date_boundary(self):
        self.save()
        next_start = self.table.upcoming(self.start - timedelta(minutes=1))[0]["next_start"]
        self.assertEqual(next_start, self.start.isoformat())
        self.table.run_due(self.start)
        self.assertEqual(self.table.upcoming(self.start)[0]["next_start"], (self.start + timedelta(days=7)).isoformat())
        self.save(title="Midnight class", start_time="00:00", duration_minutes=60, checkpoint_interval_minutes=30)
        # Monday midnight in India is still Sunday in UTC.
        utc_sunday = self.start - timedelta(hours=9, minutes=30)
        self.assertEqual(len(self.table.run_due(utc_sunday)), 1)

    def test_group_rejection_and_liveness_still_apply(self):
        self.save()
        session_id = self.table.run_due(self.start)[0]
        event = dict(event="recognition_and_liveness_passed", timestamp_utc=self.start.isoformat(),
                     identity="student_two", similarity=.9, liveness="passed")
        self.assertEqual(self.db.record_recognition_event(event).outcome, "not_in_roster")
        event.update(identity="student_one", liveness="failed")
        self.assertEqual(self.db.record_recognition_event(event).outcome, "rejected")
        event["liveness"] = "passed"
        self.assertEqual(self.db.record_recognition_event(event).outcome, "marked")
        self.assertEqual(len(self.db.session_detail(session_id)["attendance"]), 1)

    def test_worker_runs_without_browser_and_can_stop(self):
        checked = threading.Event()
        with patch.object(self.table, "run_due", side_effect=lambda: checked.set()):
            worker = TimetableScheduler(self.table, interval=.01)
            try:
                worker.start()
                worker.start()
                self.assertTrue(checked.wait(timeout=2))
                self.assertTrue(worker.status()["background_running"])
            finally:
                worker.stop()
            self.assertFalse(worker.status()["background_running"])

    def test_recognition_api_creates_due_session_before_marking(self):
        self.save()
        self.app.config["DEVICE_TOKEN"] = "test-device"
        client = self.app.test_client()
        event = dict(event="recognition_and_liveness_passed", timestamp_utc=self.start.isoformat(),
                     identity="student_one", similarity=.9, liveness="passed")
        with patch("timetable.utc_now", return_value=self.start):
            self.assertEqual(client.post("/api/recognition-events", json=event).status_code, 401)
            self.assertEqual(self.count("monitor_sessions"), 0)
            response = client.post("/api/recognition-events", json=event,
                                   headers={"X-Device-Token": "test-device"})
        self.assertEqual(response.get_json()["outcome"], "marked")
        self.assertEqual(self.count("monitor_sessions"), 1)
        self.assertEqual(self.count("monitor_attendance"), 1)

    def test_read_only_monitor_check_cannot_recreate_closed_occurrence(self):
        self.save()
        client = self.app.test_client()
        with patch("timetable.utc_now", return_value=self.start):
            initial = client.get("/api/monitor").get_json()
            self.assertEqual(initial["session"]["group_id"], self.group)
            self.db.close_session(initial["session"]["id"])
            self.app.extensions["timetable_scheduler"].check_due(force=True)
            again = client.get("/api/monitor").get_json()
        self.assertEqual(again["session"]["id"], initial["session"]["id"])
        self.assertEqual(again["session"]["status"], "closed")
        self.assertEqual(self.count("monitor_sessions"), 1)

    def test_worker_error_is_visible_and_retries(self):
        worker = TimetableScheduler(self.table)
        with patch.object(self.table, "run_due", side_effect=sqlite3.OperationalError("busy")):
            with self.assertLogs("timetable", level="ERROR"):
                worker.check_due(force=True)
        self.assertIsNotNone(worker.status()["last_error"])
        with patch.object(self.table, "run_due", return_value=[]):
            worker.check_due(force=True)
        self.assertIsNone(worker.status()["last_error"])

    def test_routes_csrf_edit_pause_and_validation(self):
        client = self.app.test_client()
        self.assertEqual(client.get("/timetable").status_code, 200)
        page = client.get("/timetable/new")
        token = re.search(rb'name="csrf_token" value="([^"]+)"', page.data).group(1).decode()
        form = dict(csrf_token=token, group_id=self.group, title="<b>AI</b>", weekday=0,
                    start_time="09:30", duration_minutes=130, checkpoint_interval_minutes=65,
                    checkpoint_window_minutes=10)
        self.assertEqual(client.post("/timetable/new", data={}).status_code, 400)
        response = client.post("/timetable/new", data=form)
        self.assertEqual(response.status_code, 302)
        self.assertIn(b"&lt;b&gt;AI&lt;/b&gt;", client.get(response.location).data)
        entry_id = self.table.entries()[0]["id"]
        self.assertFalse(self.table.entry(entry_id)["enabled"])
        form.update(title="Networks", enabled="on")
        self.assertEqual(client.post(f"/timetable/{entry_id}", data=form).status_code, 302)
        self.assertTrue(self.table.entry(entry_id)["enabled"])
        form["start_time"] = "bad"
        self.assertEqual(client.post(f"/timetable/{entry_id}", data=form).status_code, 400)
        self.assertEqual(self.table.entry(entry_id)["start_time"], "09:30")
        self.assertEqual(client.post(f"/timetable/{entry_id}/toggle", data={"enabled":"0"}).status_code, 400)
        client.post(f"/timetable/{entry_id}/toggle", data={"csrf_token":token,"enabled":"0"})
        self.assertFalse(self.table.entry(entry_id)["enabled"])
        self.assertEqual(client.get("/timetable/999999").status_code, 404)
        self.assertEqual(client.post("/timetable/999999/toggle", data={"csrf_token":token}).status_code, 404)

    def test_migration_preserves_manual_data(self):
        self.table.select_camera_section(None)
        old = self.db.start_session("Old", self.start)
        with self.db.session() as connection:
            connection.execute("DROP TABLE timetable_runs")
            connection.execute("DROP TABLE timetable_entries")
        self.db.initialize()
        self.db.initialize()
        self.assertEqual(len(self.db.session_detail(old)["roster"]), 2)
        self.assertEqual(self.table.entries(), [])


if __name__ == "__main__":
    unittest.main()
