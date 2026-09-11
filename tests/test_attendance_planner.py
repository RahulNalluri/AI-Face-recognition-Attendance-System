"""Unit tests for checkpoint-aware shortage advice."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from attendance_planner import attendance_position, build_shortage_plan  # noqa: E402


class AttendancePlannerTest(unittest.TestCase):
    def test_recovery_count_reaches_minimum_exactly(self) -> None:
        result = attendance_position(5, 10, 75)
        self.assertEqual(result["status"], "critical")
        self.assertEqual(result["needed"], 10)
        self.assertEqual((5 + result["needed"]) / (10 + result["needed"]), 0.75)

    def test_safe_absence_allowance_does_not_cross_minimum(self) -> None:
        result = attendance_position(9, 10, 75)
        self.assertEqual(result["status"], "safe")
        self.assertEqual(result["can_miss"], 2)
        self.assertGreaterEqual(9 / (10 + result["can_miss"]), 0.75)
        self.assertLess(9 / (11 + result["can_miss"]), 0.75)

    def test_warning_has_no_false_safe_absence(self) -> None:
        result = attendance_position(8, 10, 75)
        self.assertEqual(result["status"], "warning")
        self.assertEqual(result["can_miss"], 0)

    def test_no_data_does_not_report_zero_percent(self) -> None:
        result = attendance_position(0, 0, 75)
        self.assertEqual(result["status"], "no-data")
        self.assertIsNone(result["percentage"])

    def test_plan_estimates_full_classes_from_checkpoint_count(self) -> None:
        report = {
            "attended": 1, "completed": 2,
            "subjects": [{
                "title": "AI", "attended": 1, "completed": 2,
                "percentage": 50.0, "class_names": "CSE A",
            }],
        }
        upcoming = [{
            "title": "AI", "duration_minutes": 130,
            "checkpoint_interval_minutes": 65, "next_start": "2030-01-01T09:30:00+00:00",
        }]
        plan = build_shortage_plan(report, upcoming, 75)
        self.assertEqual(plan["subjects"][0]["needed"], 2)
        self.assertEqual(plan["subjects"][0]["estimated_classes"], 1)
        self.assertEqual(plan["upcoming"][0]["attendance_plan"]["status"], "critical")


if __name__ == "__main__":
    unittest.main()
