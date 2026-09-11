"""Student authentication and private attendance dashboard tests."""

from __future__ import annotations

import json
import re
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from werkzeug.security import check_password_hash, generate_password_hash

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from attendance_db import AttendanceDatabase, datetime_text  # noqa: E402
from timetable import Timetable  # noqa: E402
from web_app import create_app  # noqa: E402


class StudentPortalTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.database_path = root / "attendance.db"
        self.labels_path = root / "labels.json"
        self.labels_path.write_text(json.dumps({"0": "Rahul", "1": "Other Student"}), encoding="utf-8")
        self.app = create_app(self.database_path, labels_path=self.labels_path, validation_report_path=None)
        self.app.config.update(TESTING=True)
        self.client = self.app.test_client()
        self.database = AttendanceDatabase(self.database_path)
        rahul = self.database.student_by_identity("rahul")
        self.rahul_id = rahul["id"]
        self.database.save_student_account(self.rahul_id, "rahul", generate_password_hash("test-password"))
        self.database.save_staff_account(
            "admin", "Test Administrator", "admin", generate_password_hash("admin-password"),
        )
        self.faculty_id = self.database.save_staff_account(
            "faculty", "Test Faculty", "faculty", generate_password_hash("faculty-password"),
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def csrf(self, url="/login") -> str:
        page = self.client.get(url)
        return re.search(rb'name="csrf_token" value="([^"]+)"', page.data).group(1).decode()

    def login(self, password="test-password"):
        return self.client.post("/login", data={
            "csrf_token": self.csrf(), "role": "student",
            "username": "rahul", "password": password,
        })

    def staff_login(self):
        return self.client.post("/login", data={
            "csrf_token": self.csrf(), "role": "admin",
            "username": "admin", "password": "admin-password",
        })

    def faculty_login(self, password="faculty-password"):
        return self.client.post("/login", data={
            "csrf_token": self.csrf(), "role": "faculty",
            "username": "faculty", "password": password,
        })

    def add_completed_attendance(self) -> None:
        start = datetime.now(timezone.utc) - timedelta(hours=3)
        session_id = self.database.start_session("Artificial Intelligence", start, 120, 30, 5)
        with self.database.session() as connection:
            checkpoints = connection.execute(
                "SELECT * FROM monitor_checkpoints WHERE session_id=? ORDER BY checkpoint_number", (session_id,)
            ).fetchall()
            connection.execute(
                """INSERT INTO monitor_attendance(
                checkpoint_id,student_id,recognized_at,similarity,liveness_passed
                ) VALUES (?,?,?,?,1)""",
                (checkpoints[0]["id"], self.rahul_id, checkpoints[0]["opens_at"], 0.93),
            )
            connection.execute(
                """INSERT INTO monitor_attendance(
                checkpoint_id,student_id,recognized_at,similarity,liveness_passed
                ) VALUES (?,?,?,?,1)""",
                (checkpoints[1]["id"], self.rahul_id, checkpoints[1]["opens_at"], 0.94),
            )
            other_id = connection.execute(
                "SELECT id FROM students WHERE identity_label='Other Student'"
            ).fetchone()[0]
            connection.execute(
                """INSERT INTO monitor_attendance(
                checkpoint_id,student_id,recognized_at,similarity,liveness_passed
                ) VALUES (?,?,?,?,1)""",
                (checkpoints[2]["id"], other_id, checkpoints[2]["opens_at"], 0.91),
            )

    def test_login_is_required_and_bad_password_is_generic(self) -> None:
        response = self.client.get("/student")
        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.headers["Location"].endswith("/login"))
        denied = self.login("wrong")
        self.assertEqual(denied.status_code, 401)
        self.assertIn(b"Invalid username or password", denied.data)

    def test_portal_shows_only_rahul_effective_attendance(self) -> None:
        self.add_completed_attendance()
        self.assertEqual(self.login().status_code, 302)
        page = self.client.get("/student")
        self.assertEqual(page.status_code, 200)
        self.assertIn(b"Welcome, Rahul", page.data)
        self.assertIn(b"50.0%", page.data)
        self.assertIn(b"Artificial Intelligence", page.data)
        self.assertNotIn(b"Other Student", page.data)

    def test_upcoming_classes_are_filtered_by_membership(self) -> None:
        other = self.database.student_by_identity("Other Student")
        own_group = self.database.save_class_group("CSE A", "", [self.rahul_id])
        other_group = self.database.save_class_group("CSE B", "", [other["id"]])
        future = datetime.now(timezone.utc) + timedelta(hours=1)
        timetable = Timetable(self.database)
        for group_id, title in ((own_group, "Machine Learning"), (other_group, "Private Other Class")):
            timetable.save(
                group_id=group_id, title=title, weekday=future.weekday(),
                start_time=future.strftime("%H:%M"), duration_minutes=60,
                checkpoint_interval_minutes=20, checkpoint_window_minutes=10,
                timezone_name="UTC",
            )
        self.login()
        page = self.client.get("/student")
        self.assertIn(b"Machine Learning", page.data)
        self.assertNotIn(b"Private Other Class", page.data)

    def test_roles_are_routed_to_their_authorized_dashboard(self) -> None:
        self.assertTrue(self.client.get("/").headers["Location"].endswith("/login"))
        self.login()
        self.assertTrue(self.client.get("/").headers["Location"].endswith("/student"))
        token = self.csrf("/student")
        self.client.post("/logout", data={"csrf_token": token})
        self.assertTrue(self.staff_login().headers["Location"].endswith("/"))
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertTrue(self.client.get("/student").headers["Location"].endswith("/"))

    def test_student_can_change_password(self) -> None:
        self.login()
        token = self.csrf("/change-password")
        changed = self.client.post("/change-password", data={
            "csrf_token": token, "current_password": "test-password",
            "new_password": "new-password-123", "confirm_password": "new-password-123",
        })
        self.assertEqual(changed.status_code, 302)
        token = self.csrf("/student")
        self.client.post("/logout", data={"csrf_token": token})
        self.assertEqual(self.login("test-password").status_code, 401)
        self.assertEqual(self.login("new-password-123").status_code, 302)

    def test_staff_can_change_required_percentage_used_by_student_planner(self) -> None:
        self.staff_login()
        token = self.csrf("/")
        changed = self.client.post("/settings/attendance-minimum", data={
            "csrf_token": token, "minimum_percentage": "80",
        })
        self.assertEqual(changed.status_code, 302)
        self.assertEqual(self.database.attendance_minimum(), 80.0)
        token = self.csrf("/")
        self.client.post("/logout", data={"csrf_token": token})
        self.login()
        page = self.client.get("/student")
        self.assertIn(b"Required attendance: 80.0%", page.data)

    def test_admin_links_student_account_and_forces_first_password_change(self) -> None:
        self.staff_login()
        other = self.database.student_by_identity("Other Student")
        group_id = self.database.save_class_group("CSE Registration", "", [self.rahul_id])
        token = self.csrf("/accounts")
        created = self.client.post("/accounts/students", data={
            "csrf_token": token, "student_id": other["id"], "group_id": group_id,
            "username": "other.student",
            "temporary_password": "temporary-123", "confirm_password": "temporary-123",
        }, follow_redirects=True)
        self.assertIn(b"Student account created and linked", created.data)
        linked = self.database.student_account_by_username("other.student")
        self.assertEqual(linked["student_id"], other["id"])
        self.assertEqual(linked["password_change_required"], 1)
        self.assertIn(group_id, self.database.student_group_ids(other["id"]))
        self.client.post("/logout", data={"csrf_token": self.csrf("/accounts")})
        response = self.client.post("/login", data={
            "csrf_token": self.csrf(), "role": "student", "username": "other.student",
            "password": "temporary-123",
        })
        self.assertEqual(response.status_code, 302)
        self.assertTrue(self.client.get("/student").headers["Location"].endswith("/change-password"))
        token = self.csrf("/change-password")
        changed = self.client.post("/change-password", data={
            "csrf_token": token, "current_password": "temporary-123",
            "new_password": "student-owned-456", "confirm_password": "student-owned-456",
        })
        self.assertEqual(changed.status_code, 302)
        self.assertEqual(
            self.database.student_account_by_username("other.student")["password_change_required"], 0,
        )

    def test_faculty_cannot_manage_accounts_and_admin_can_reset_and_disable_faculty(self) -> None:
        self.faculty_login()
        self.assertEqual(self.client.get("/accounts").status_code, 403)
        self.client.post("/logout", data={"csrf_token": self.csrf("/")})
        self.staff_login()
        token = self.csrf("/accounts")
        reset = self.client.post(f"/accounts/faculty/{self.faculty_id}/reset-password", data={
            "csrf_token": token, "temporary_password": "replacement-123",
            "confirm_password": "replacement-123",
        }, follow_redirects=True)
        self.assertIn(b"Temporary password saved", reset.data)
        account = self.database.staff_account_by_username("faculty")
        self.assertEqual(account["password_change_required"], 1)
        self.assertTrue(check_password_hash(account["password_hash"], "replacement-123"))
        disabled = self.client.post(f"/accounts/faculty/{self.faculty_id}/toggle", data={
            "csrf_token": token, "enabled": "0",
        }, follow_redirects=True)
        self.assertIn(b"Account disabled", disabled.data)
        self.assertEqual(self.database.staff_account_by_username("faculty")["enabled"], 0)

    def test_admin_assigns_subject_and_faculty_sees_focused_dashboard(self) -> None:
        group_id = self.database.save_class_group("CSE A", "", [self.rahul_id])
        self.staff_login()
        token = self.csrf("/accounts")
        response = self.client.post("/accounts/faculty-assignments", data={
            "csrf_token": token, "staff_account_id": self.faculty_id,
            "group_id": group_id, "subject_title": "Artificial Intelligence",
        }, follow_redirects=True)
        self.assertIn(b"Faculty subject and section assigned", response.data)
        self.client.post("/logout", data={"csrf_token": self.csrf("/accounts")})
        self.faculty_login()
        dashboard = self.client.get("/faculty")
        self.assertEqual(dashboard.status_code, 200)
        self.assertIn(b"Artificial Intelligence", dashboard.data)
        self.assertIn(b"CSE A", dashboard.data)
        self.assertNotIn(b"Manage assignments", dashboard.data)

    def test_student_download_contains_only_own_hourly_report(self) -> None:
        self.add_completed_attendance()
        self.login()
        response = self.client.get("/student/report.csv")
        self.assertEqual(response.status_code, 200)
        self.assertIn("attachment", response.headers["Content-Disposition"])
        self.assertIn(b"Artificial Intelligence", response.data)
        self.assertIn(b"Overall attendance,50.0%", response.data)
        self.assertIn(b"Passed checkpoints", response.data)
        self.assertNotIn(b"Other Student", response.data)

    def test_demo_credentials_are_generated_once_and_password_is_hashed(self) -> None:
        root = Path(self.temporary.name)
        fresh_db = root / "demo.db"
        credentials = root / "credentials"
        create_app(
            fresh_db, labels_path=self.labels_path, validation_report_path=None,
            enable_demo_logins=True, demo_credentials_path=credentials,
        )
        student_credentials = credentials / "student_demo_credentials.txt"
        original = student_credentials.read_text(encoding="utf-8")
        create_app(
            fresh_db, labels_path=self.labels_path, validation_report_path=None,
            enable_demo_logins=True, demo_credentials_path=credentials,
        )
        self.assertEqual(student_credentials.read_text(encoding="utf-8"), original)
        self.assertTrue((credentials / "admin_demo_credentials.txt").exists())
        self.assertTrue((credentials / "faculty_demo_credentials.txt").exists())
        password = dict(line.split("=", 1) for line in original.strip().splitlines())["password"]
        account = AttendanceDatabase(fresh_db).student_account_by_username("rahul")
        self.assertNotEqual(account["password_hash"], password)
        self.assertTrue(check_password_hash(account["password_hash"], password))


if __name__ == "__main__":
    unittest.main()
