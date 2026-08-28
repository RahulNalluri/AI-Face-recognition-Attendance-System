"""Class roster isolation, historical snapshots, migration, and UI safety."""

import csv
import io
import re
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from attendance_db import AttendanceDatabase
from web_app import create_app


class ClassGroupsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.app = create_app(Path(self.temporary.name) / "test.db", labels_path=None)
        self.app.config.update(TESTING=True, DEVICE_TOKEN="test-camera")
        self.client = self.app.test_client()
        self.db = self.app.extensions["attendance_database"]
        self.db.sync_identities(["student_one", "student_two", "student_three"])
        self.ids = {row["identity_label"]: row["id"] for row in self.db.students()}
        self.one, self.two = self.ids["student_one"], self.ids["student_two"]
        self.start = datetime.now(timezone.utc) - timedelta(minutes=1)

    def tearDown(self):
        self.temporary.cleanup()

    def group(self, name="CSE-A", members=None):
        return self.db.save_class_group(name, "Semester 7", [self.one] if members is None else members)

    def event(self, identity="student_one", when=None):
        return {"event": "recognition_and_liveness_passed", "identity": identity,
                "liveness": "passed", "similarity": .91,
                "timestamp_utc": (when or datetime.now(timezone.utc)).isoformat()}

    def csrf(self, url="/classes/new"):
        page = self.client.get(url)
        return re.search(rb'name="csrf_token" value="([^"]+)"', page.data).group(1).decode()

    def test_create_edit_and_shared_membership(self):
        group_id = self.group(members=[self.one, self.one, self.two])
        other_id = self.group("CSE-B")
        self.assertEqual(set(self.db.class_group(group_id)["student_ids"]), {self.one, self.two})
        self.assertEqual(self.db.class_groups()[0]["member_count"], 2)
        self.db.save_class_group(" CSE-A renamed ", " New description ", [self.two], group_id)
        updated = self.db.class_group(group_id)
        self.assertEqual(updated["name"], "CSE-A renamed")
        self.assertEqual(updated["description"], "New description")
        self.assertEqual(updated["student_ids"], [self.two])
        self.assertEqual(self.db.class_group(other_id)["student_ids"], [self.one])

    def test_invalid_roster_edits_roll_back(self):
        group_id = self.group()
        for name, description, members in [("cse-a", "", []), ("", "", []),
                                           ("X" * 81, "", []), ("Other", "X" * 301, []),
                                           ("Other", "", [999999])]:
            with self.assertRaises(ValueError):
                self.db.save_class_group(name, description, members)
        with self.assertRaises(ValueError):
            self.db.save_class_group("Changed", "", [999999], group_id)
        self.assertEqual(self.db.class_group(group_id)["name"], "CSE-A")
        self.assertEqual(self.db.class_group(group_id)["student_ids"], [self.one])
        self.assertEqual(len(self.db.class_groups()), 1)
        with self.assertRaises(ValueError):
            self.db.save_class_group("Missing", "", [], 999999)

    def test_wrong_class_is_logged_but_never_added_to_roster(self):
        group_id = self.group()
        self.group("CSE-B", [self.two])
        session_id = self.db.start_session("AI", self.start, group_id=group_id)
        for identity in ["student_two", "new_model_identity"]:
            result = self.db.record_recognition_event(self.event(identity))
            self.assertEqual(result.outcome, "not_in_roster")
            self.assertIsNotNone(result.checkpoint_id)
        detail = self.db.session_detail(session_id)
        self.assertEqual([s["student_id"] for s in detail["roster"]], [self.one])
        self.assertEqual(detail["attendance"], [])
        self.assertEqual(self.db.class_group(group_id)["student_ids"], [self.one])
        self.assertEqual(self.db.monitor_data()["logs"][0]["outcome"], "not_in_roster")
        self.assertEqual(self.db.record_recognition_event(self.event()).outcome, "marked")
        self.assertEqual(self.db.record_recognition_event(self.event()).outcome, "already_marked")

    def test_edits_only_affect_future_session_snapshots(self):
        group_id = self.group()
        old_id = self.db.start_session("AI", self.start, group_id=group_id)
        self.db.save_class_group("CSE-B", "", [self.two], group_id)
        self.assertEqual(self.db.record_recognition_event(self.event()).outcome, "marked")
        self.assertEqual(self.db.record_recognition_event(self.event("student_two")).outcome, "not_in_roster")
        new_id = self.db.start_session("Networks", self.start, group_id=group_id)
        old, new = self.db.session_detail(old_id), self.db.session_detail(new_id)
        self.assertEqual(old["session"]["group_name_snapshot"], "CSE-A")
        self.assertEqual(new["session"]["group_name_snapshot"], "CSE-B")
        self.assertEqual([r["student_id"] for r in old["roster"]], [self.one])
        self.assertEqual([r["student_id"] for r in new["roster"]], [self.two])
        self.assertEqual(len(old["attendance"]), 1)
        self.assertEqual(self.db.record_recognition_event(self.event()).outcome, "not_in_roster")
        self.assertEqual(self.db.record_recognition_event(self.event("student_two")).outcome, "marked")

    def test_invalid_class_does_not_close_existing_active_session(self):
        active_id = self.db.start_session("Existing", self.start)
        empty_id = self.group(members=[])
        for group_id in [empty_id, 999999]:
            with self.assertRaises(ValueError):
                self.db.start_session("Invalid", self.start, group_id=group_id)
            self.assertEqual(self.db.session_detail(active_id)["session"]["status"], "active")
            self.assertEqual(len(self.db.session_history()), 1)

    def test_inactive_students_cannot_join_or_start_a_class(self):
        group_id = self.group()
        with self.db.session() as connection:
            connection.execute("UPDATE students SET active=0 WHERE id=?", (self.one,))
        self.assertEqual(self.db.class_groups()[0]["member_count"], 0)
        with self.assertRaises(ValueError):
            self.db.start_session("AI", self.start, group_id=group_id)
        with self.assertRaises(ValueError):
            self.db.save_class_group("Other", "", [self.one])

    def test_backfill_never_expands_class_rosters(self):
        grouped = self.db.start_session("AI", self.start, group_id=self.group())
        ungrouped = self.db.start_session("Legacy", self.start)
        with self.db.session() as connection:
            connection.execute("DELETE FROM session_roster WHERE session_id IN (?,?)", (grouped, ungrouped))
        self.assertEqual(self.db.backfill_missing_rosters(), 1)
        self.assertEqual(self.db.session_detail(grouped)["roster"], [])
        self.assertEqual(len(self.db.session_detail(ungrouped)["roster"]), 3)

    def test_migration_preserves_old_records_and_is_idempotent(self):
        session_id = self.db.start_session("Legacy", self.start)
        self.db.record_recognition_event(self.event())
        # Reproduce the pre-class schema, with real-shaped attendance data in this temp DB.
        with self.db.session() as connection:
            for table in ["session_class_groups", "class_group_members", "class_groups"]:
                connection.execute(f"DROP TABLE {table}")
        self.db.initialize()
        self.db.initialize()
        self.db.backfill_missing_rosters()
        detail = self.db.session_detail(session_id)
        self.assertEqual(len(detail["attendance"]), 1)
        self.assertEqual(len(detail["roster"]), 3)
        self.assertIsNone(detail["session"]["group_id"])
        self.assertEqual(self.db.class_groups(), [])
        self.assertEqual(self.db.record_recognition_event(self.event("new_model_identity")).outcome, "marked")
        self.assertEqual(len(self.db.session_detail(session_id)["roster"]), 4)

    def test_class_forms_csrf_validation_and_escaping(self):
        self.assertEqual(self.client.post("/classes/new", data={"name": "Bad"}).status_code, 400)
        token = self.csrf()
        response = self.client.post("/classes/new", data={
            "csrf_token": token, "name": "<script>CSE-A</script>", "student_ids": [str(self.one)]})
        self.assertEqual(response.status_code, 302)
        group_id = self.db.class_groups()[0]["id"]
        page = self.client.get(response.location)
        self.assertIn(b"&lt;script&gt;CSE-A&lt;/script&gt;", page.data)
        self.assertNotIn(b"<script>CSE-A</script>", page.data)
        updated = self.client.post(f"/classes/{group_id}", data={
            "csrf_token": token, "name": "CSE-B", "student_ids": [str(self.two)]})
        self.assertEqual(updated.status_code, 302)
        self.assertEqual(self.db.class_group(group_id)["student_ids"], [self.two])
        self.assertEqual(self.client.post(f"/classes/{group_id}", data={
            "csrf_token": token, "name": "Bad", "student_ids": ["not-an-id"]}).status_code, 400)
        self.assertEqual(self.db.class_group(group_id)["name"], "CSE-B")
        self.assertEqual(self.client.get("/classes/999999").status_code, 404)
        self.assertIn(b"CSE-B", self.client.get("/classes").data)

    def test_selector_creates_class_session_and_exports_snapshot(self):
        group_id = self.group("=CSE-A")
        token = self.csrf(f"/?group_id={group_id}")
        self.app.config["APP_TIMEZONE"] = "UTC"
        response = self.client.post("/sessions/start", data={
            "csrf_token": token, "group_id": group_id, "title": "AI",
            "starts_at": self.start.strftime("%Y-%m-%dT%H:%M"),
            "duration_minutes": 130, "checkpoint_interval_minutes": 65, "checkpoint_window_minutes": 10})
        self.assertEqual(response.status_code, 302)
        current = self.client.get("/api/monitor").get_json()["session"]
        self.assertEqual(current["group_id"], group_id)
        self.db.save_class_group("Renamed", "", [self.two], group_id)
        report = self.client.get(f"/sessions/{current['id']}")
        self.assertIn(b"=CSE-A", report.data)
        self.assertNotIn(b"Student Two", report.data)
        self.assertIn(b"=CSE-A", self.client.get("/sessions").data)
        exported = self.client.get(f"/sessions/{current['id']}/export.csv")
        rows = list(csv.reader(io.StringIO(exported.get_data(as_text=True))))
        self.assertEqual(rows[0][-1], "Class group")
        self.assertEqual(rows[1][-1], "'=CSE-A")
        self.assertEqual(len(rows), 2)

    def test_class_only_report_denominator(self):
        start = datetime.now(timezone.utc) - timedelta(minutes=20)
        session_id = self.db.start_session("AI", start, 60, 30, 5, group_id=self.group())
        self.db.record_recognition_event(self.event(when=start + timedelta(minutes=1)))
        detail = self.db.session_detail(session_id)
        self.assertEqual(detail["completed_checkpoint_count"], 1)
        self.assertEqual(detail["attendance_percentage"], 100.0)
        self.assertEqual(self.db.session_history()[0]["attendance_percentage"], 100.0)
        self.assertEqual(self.db.session_history()[0]["roster_count"], 1)


if __name__ == "__main__":
    unittest.main()
