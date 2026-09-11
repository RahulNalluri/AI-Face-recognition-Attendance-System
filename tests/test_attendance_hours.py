"""Tests for converting checkpoint evidence into hourly attendance."""

from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from attendance_hours import build_hour_blocks, evaluate_hour  # noqa: E402


class AttendanceHoursTest(unittest.TestCase):
    def setUp(self) -> None:
        self.start = datetime(2026, 9, 11, 9, 30, tzinfo=timezone.utc)

    def session(self, minutes=120):
        return {"starts_at": self.start.isoformat(), "ends_at": (self.start + timedelta(minutes=minutes)).isoformat()}

    def checkpoints(self, interval=20, duration=120):
        rows = []
        for index, minute in enumerate(range(0, duration, interval), 1):
            opens = self.start + timedelta(minutes=minute)
            rows.append({"id": index, "opens_at": opens.isoformat(), "closes_at": (opens + timedelta(minutes=5)).isoformat()})
        return rows

    def test_two_passes_mark_hour_present_and_one_marks_absent(self):
        blocks = build_hour_blocks(
            self.session(), self.checkpoints(), self.start + timedelta(hours=3),
        )
        self.assertEqual(len(blocks), 2)
        self.assertEqual(evaluate_hour(blocks[0], {1, 2})["status"], "present")
        self.assertEqual(evaluate_hour(blocks[1], {4})["status"], "absent")

    def test_checkpoint_on_hour_boundary_belongs_to_next_hour(self):
        blocks = build_hour_blocks(
            self.session(), self.checkpoints(), self.start + timedelta(hours=3),
        )
        self.assertEqual(blocks[0]["checkpoint_ids"], [1, 2, 3])
        self.assertEqual(blocks[1]["checkpoint_ids"], [4, 5, 6])

    def test_short_trailing_fragment_is_not_an_attendance_hour(self):
        blocks = build_hour_blocks(
            self.session(130), self.checkpoints(duration=130), self.start + timedelta(hours=3),
        )
        self.assertEqual(len(blocks), 2)

    def test_hour_with_too_few_scheduled_checks_is_not_evaluable(self):
        blocks = build_hour_blocks(
            self.session(), self.checkpoints(interval=65), self.start + timedelta(hours=3),
        )
        self.assertEqual(evaluate_hour(blocks[0], {1})["status"], "insufficient-checks")


if __name__ == "__main__":
    unittest.main()
