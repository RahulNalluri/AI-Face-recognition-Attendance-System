"""Verify repeated checkpoint rules using an isolated temporary database."""

from __future__ import annotations

import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from attendance_db import AttendanceDatabase  # noqa: E402


def event(timestamp: datetime) -> dict:
    return {
        "event": "recognition_and_liveness_passed",
        "timestamp_utc": timestamp.isoformat(),
        "identity": "checkpoint_test_student",
        "similarity": 0.93,
        "liveness": "passed",
    }


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="attendance-checkpoints-") as directory:
        database = AttendanceDatabase(Path(directory) / "attendance.db")
        database.initialize()
        database.sync_identities(["checkpoint_test_student"])

        start = datetime.now(timezone.utc) - timedelta(hours=3)
        session_id = database.start_session(
            "Repeated checkpoint verification", start,
            duration_minutes=130,
            checkpoint_interval_minutes=65,
            checkpoint_window_minutes=10,
        )
        cases = [
            ("checkpoint 1", 1, "marked"),
            ("checkpoint 1 duplicate", 2, "already_marked"),
            ("closed gap", 20, "no_active_checkpoint"),
            ("checkpoint 2", 66, "marked"),
            ("checkpoint 2 duplicate", 67, "already_marked"),
        ]

        print("Repeated checkpoint verification (temporary database)")
        print("-" * 68)
        for name, minute, expected in cases:
            result = database.record_recognition_event(event(start + timedelta(minutes=minute)))
            status = "PASS" if result.outcome == expected else "FAIL"
            print(f"{status:4}  {name:24} expected={expected:20} got={result.outcome}")
            if result.outcome != expected:
                raise AssertionError(f"{name}: expected {expected}, received {result.outcome}")

        detail = database.session_detail(session_id)
        if detail is None:
            raise AssertionError("Verification session was not found")
        present_cells = [
            cell for student in detail["roster"] for cell in student["cells"]
            if cell["status"] == "present"
        ]
        if len(present_cells) != 2:
            raise AssertionError(f"Expected exactly 2 saved attendance records, found {len(present_cells)}")
        checkpoint_ids = {cell["checkpoint"]["id"] for cell in present_cells}
        if len(checkpoint_ids) != 2:
            raise AssertionError("Attendance records were not stored in two distinct checkpoints")

        print("-" * 68)
        print("PASS  Exactly two attendance records were saved in distinct checkpoints.")
        print("PASS  The real instance/attendance_monitor.db was not opened or changed.")


if __name__ == "__main__":
    main()
