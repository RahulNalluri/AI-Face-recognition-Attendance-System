"""Section grid transactions and camera assignment isolation."""
import json
import re
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from section_setup import SectionSetup, default_grid, normalize_grid
from web_app import create_app


class SectionSetupTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.app = create_app(Path(self.temp.name) / "test.db", labels_path=None)
        self.app.config.update(TESTING=True)
        self.db = self.app.extensions["attendance_database"]
        self.table = self.app.extensions["timetable"]
        self.sections = SectionSetup(self.db)
        self.db.sync_identities(["student_one", "student_two"])
        self.students = [r["id"] for r in self.db.students()]
        self.start = datetime(2030, 1, 7, 4, tzinfo=timezone.utc)

    def tearDown(self):
        self.temp.cleanup()

    def grid(self):
        grid = default_grid()
        grid["enabled"] = True
        grid["periods"][0]["duration"] = 45
        grid["periods"][0]["break_after"] = 15
        grid["subjects"][0][0] = "AI"
        grid["subjects"][0][1] = "Networks"
        return grid

    def save(self, **kwargs):
        data = dict(name="CSE-A", description="Section A", student_ids=[self.students[0]],
                    department="CSE", semester="7", academic_year="2026-27", grid=self.grid())
        data.update(kwargs)
        return self.sections.save(**data)

    def test_grid_saves_section_metadata_periods_breaks_and_entries(self):
        group = self.save()
        profile = self.sections.profile(group)
        self.assertEqual(profile["academic_year"], "2026-27")
        self.assertEqual(profile["department"], "CSE")
        entries = self.table.entries()
        self.assertEqual(len(entries), 2)
        self.assertEqual([e["start_time"] for e in entries], ["09:30", "10:30"])
        self.assertEqual(entries[0]["duration_minutes"], 45)
        self.assertEqual(entries[0]["checkpoint_interval_minutes"], 45)
        self.assertEqual(entries[0]["checkpoint_window_minutes"], 10)
        self.assertEqual(profile["grid"]["subjects"][0][:2], ["AI", "Networks"])

    def test_blank_grid_and_paused_rosterless_section_are_allowed(self):
        group = self.save(grid=default_grid(), student_ids=[])
        self.assertEqual(self.table.entries(), [])
        self.assertEqual(self.db.class_group(group)["student_ids"], [])
        paused = self.grid()
        paused["enabled"] = False
        self.save(group_id=group, student_ids=[], grid=paused)
        self.assertFalse(any(row["enabled"] for row in self.table.entries()))

    def test_invalid_enabled_empty_roster_rolls_back_entire_new_section(self):
        with self.assertRaises(ValueError):
            self.save(student_ids=[])
        self.assertEqual(self.db.class_groups(), [])
        self.assertEqual(self.table.entries(), [])

    def test_invalid_grid_is_rejected(self):
        variations = []
        for field, value in [("duration", 0), ("break_after", -1), ("repeat", 100), ("window", 70), ("duration", 1.5)]:
            grid = self.grid()
            grid["periods"][0][field] = value
            variations.append(grid)
        for field, value in [("periods", []), ("subjects", [["AI"]]), ("day_start", "bad"),
                             ("day_start", "23:30"), ("enabled", "yes")]:
            grid = self.grid()
            grid[field] = value
            variations.append(grid)
        for grid in variations:
            with self.subTest(grid=grid), self.assertRaises(ValueError):
                normalize_grid(grid)
        self.assertEqual(self.db.class_groups(), [])

    def test_overlapping_sections_allowed_but_only_assigned_one_runs(self):
        first = self.save()
        second = self.save(name="CSE-B", student_ids=[self.students[1]])
        self.assertEqual(self.table.run_due(self.start), [])
        self.table.select_camera_section(second)
        session = self.table.run_due(self.start)[0]
        detail = self.db.session_detail(session)
        self.assertEqual(detail["session"]["group_id"], second)
        self.assertEqual(detail["roster"][0]["student_id"], self.students[1])
        self.assertEqual(len(self.table.runs()), 1)
        self.assertNotEqual(first, second)

    def test_standalone_overlap_checks_apply_within_section_only(self):
        group = self.save()
        other = self.db.save_class_group("CSE-B", "", [self.students[1]])
        values = dict(title="Other class", weekday=0, start_time="09:30", duration_minutes=45,
                      checkpoint_interval_minutes=45, checkpoint_window_minutes=10,
                      timezone_name="Asia/Kolkata")
        self.table.save(group_id=other, **values)
        with self.assertRaises(ValueError):
            self.table.save(group_id=group, **values)

    def test_conflict_rolls_back_metadata_roster_and_disabled_entries(self):
        group = self.save()
        self.table.save(group_id=group, title="Extra subject", weekday=0, start_time="12:00",
                        duration_minutes=60, checkpoint_interval_minutes=60,
                        checkpoint_window_minutes=10, timezone_name="Asia/Kolkata")
        changed = self.grid()
        changed["day_start"] = "11:00"
        with self.assertRaises(ValueError):
            self.save(group_id=group, name="Changed", student_ids=[self.students[1]], grid=changed)
        self.assertEqual(self.db.class_group(group)["name"], "CSE-A")
        self.assertEqual(self.db.class_group(group)["student_ids"], [self.students[0]])
        self.assertTrue(all(e["enabled"] for e in self.table.entries()))
        self.assertEqual(self.sections.profile(group)["grid"]["day_start"], "09:30")

    def test_edit_clear_and_restore_cell_preserves_id_and_today_claim(self):
        group = self.save()
        self.table.select_camera_section(group)
        session = self.table.run_due(self.start)[0]
        old_entry = self.table.entries()[0]["id"]
        grid = self.grid()
        grid["subjects"][0][0] = ""
        self.save(group_id=group, grid=grid)
        with self.db.session() as connection:
            self.assertEqual(connection.execute("SELECT enabled FROM timetable_entries WHERE id=?", (old_entry,)).fetchone()[0], 0)
        grid["subjects"][0][0] = "New subject"
        self.save(group_id=group, grid=grid, name="Renamed", student_ids=[self.students[1]])
        self.assertEqual(self.table.entries()[0]["id"], old_entry)
        self.assertEqual(self.table.run_due(self.start + timedelta(minutes=1)), [])
        detail = self.db.session_detail(session)
        self.assertEqual(detail["session"]["title"], "AI")
        self.assertEqual(detail["session"]["group_name_snapshot"], "CSE-A")
        self.assertEqual(detail["roster"][0]["student_id"], self.students[0])
        next_session = self.table.run_due(self.start + timedelta(days=7))[0]
        self.assertEqual(self.db.session_detail(next_session)["session"]["title"], "New subject")

    def test_removed_period_disables_entry_without_deleting_history(self):
        group = self.save()
        old_entries = self.table.entries()
        grid = self.grid()
        grid["periods"] = grid["periods"][:1]
        grid["subjects"] = [row[:1] for row in grid["subjects"]]
        self.save(group_id=group, grid=grid)
        with self.db.session() as connection:
            self.assertEqual(connection.execute("SELECT enabled FROM timetable_entries WHERE id=?", (old_entries[1]["id"],)).fetchone()[0], 0)

    def test_long_lab_repeated_checkpoints(self):
        grid = self.grid()
        grid["periods"][0].update(duration=90, repeat=45)
        group = self.save(grid=grid)
        self.table.select_camera_section(group)
        detail = self.db.session_detail(self.table.run_due(self.start)[0])
        self.assertEqual(len(detail["checkpoints"]), 2)

    def test_switch_during_active_session_blocked_and_manual_mismatch_blocked(self):
        first, second = self.save(), self.save(name="CSE-B")
        self.table.select_camera_section(first)
        now = datetime.now(timezone.utc)
        session = self.db.start_session("Active", now - timedelta(minutes=1), group_id=first)
        for target in [second, None]:
            with self.assertRaises(ValueError):
                self.table.select_camera_section(target)
        self.assertEqual(self.table.camera_section()["group_id"], first)
        with self.assertRaises(ValueError):
            self.db.start_session("Wrong section", now, group_id=second)
        self.db.close_session(session)
        self.table.select_camera_section(second)
        self.assertEqual(self.table.camera_section()["group_id"], second)

    def test_old_future_manual_session_cannot_mark_wrong_section(self):
        first, second = self.save(), self.save(name="CSE-B")
        self.db.start_session("Future manual", self.start, group_id=first)
        self.table.select_camera_section(second)
        result = self.db.record_recognition_event({"event":"recognition_and_liveness_passed",
            "identity":"student_one", "timestamp_utc":self.start.isoformat(), "similarity":.9,"liveness":"passed"})
        self.assertEqual(result.outcome, "wrong_camera_section")

    def test_managed_entries_cannot_be_edited_or_enabled_individually(self):
        group = self.save()
        entry = self.table.entries()[0]
        with self.assertRaises(ValueError):
            self.table.set_enabled(entry["id"], False)
        with self.assertRaises(ValueError):
            self.table.save(group_id=group, title="Bypass", weekday=0, start_time="09:30",
                duration_minutes=45,checkpoint_interval_minutes=45,checkpoint_window_minutes=10,
                timezone_name="Asia/Kolkata",entry_id=entry["id"])
        response = self.app.test_client().get(f"/timetable/{entry['id']}")
        self.assertIn(f"/classes/{group}", response.location)

    def test_migration_assigns_single_legacy_section_once(self):
        group = self.save()
        with self.db.session() as connection:
            connection.execute("DELETE FROM camera_section")
        self.db.initialize()
        self.assertEqual(self.table.camera_section()["group_id"], group)
        self.table.select_camera_section(None)
        self.db.initialize()
        self.assertIsNone(self.table.camera_section()["group_id"])

    def test_migration_with_multiple_sections_requires_explicit_choice(self):
        self.save()
        self.save(name="CSE-B")
        with self.db.session() as connection:
            connection.execute("DELETE FROM camera_section")
        self.db.initialize()
        self.assertIsNone(self.table.camera_section()["group_id"])

    def test_section_form_round_trip_and_camera_csrf(self):
        client = self.app.test_client()
        page = client.get("/classes/new")
        token = re.search(rb'name="csrf_token" value="([^"]+)"', page.data).group(1).decode()
        self.assertIn(b"Weekly timetable", page.data)
        form = dict(csrf_token=token,name="CSE-A",student_ids=[str(self.students[0])],
                    department="CSE",semester="7",academic_year="2026-27",grid_json=json.dumps(self.grid()))
        response = client.post("/classes/new", data=form)
        self.assertEqual(response.status_code, 302)
        self.assertIn(b"2026-27", client.get(response.location).data)
        group = self.db.class_groups()[0]["id"]
        self.assertEqual(client.post("/camera/section",data={"group_id":group}).status_code,400)
        client.post("/camera/section",data={"csrf_token":token,"group_id":group})
        self.assertEqual(self.table.camera_section()["group_id"], group)
        self.assertIn(b"CSE-A", client.get("/").data)
        form["grid_json"] = "{}"
        self.assertEqual(client.post(f"/classes/{group}",data=form).status_code,400)
        self.assertEqual(len(self.table.entries()),2)


if __name__ == "__main__":
    unittest.main()
