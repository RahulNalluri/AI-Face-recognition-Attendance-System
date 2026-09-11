"""Checkpoint-aware attendance shortage calculations."""

from __future__ import annotations

from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from math import ceil
from typing import Any


def attendance_position(attended: int, completed: int, minimum_percentage: float) -> dict[str, Any]:
    if attended < 0 or completed < 0 or attended > completed:
        raise ValueError("Attendance counts are invalid")
    target = Decimal(str(minimum_percentage)) / Decimal(100)
    if not Decimal("0") < target < Decimal("1"):
        raise ValueError("Minimum attendance must be between 0% and 100%")
    if completed == 0:
        return {
            "percentage": None, "status": "no-data", "needed": 0, "can_miss": 0,
            "message": "No completed teaching hours yet. Attend the first scheduled hour to build a safe record.",
        }

    present, total = Decimal(attended), Decimal(completed)
    percentage = float((present * Decimal(100) / total).quantize(Decimal("0.1")))
    if present / total < target:
        raw_needed = (target * total - present) / (Decimal(1) - target)
        needed = max(0, int(raw_needed.to_integral_value(rounding=ROUND_CEILING)))
        status, can_miss = "critical", 0
        message = (
            f"Attend the next {needed} teaching hour{'s' if needed != 1 else ''} without missing "
            f"to reach {minimum_percentage:g}%."
        )
    else:
        allowed = present / target - total
        can_miss = max(0, int(allowed.to_integral_value(rounding=ROUND_FLOOR)))
        needed = 0
        status = "safe" if percentage >= min(100.0, minimum_percentage + 10) else "warning"
        message = (
            f"You can miss {can_miss} upcoming teaching hour{'s' if can_miss != 1 else ''} "
            f"and remain at or above {minimum_percentage:g}%."
            if can_miss else
            f"Attend the next teaching hour to stay at or above {minimum_percentage:g}%."
        )
    return {
        "percentage": percentage, "status": status, "needed": needed,
        "can_miss": can_miss, "message": message,
    }


def build_shortage_plan(
    report: dict[str, Any], upcoming: list[dict[str, Any]], minimum_percentage: float,
) -> dict[str, Any]:
    future_by_subject: dict[str, list[dict[str, Any]]] = {}
    for occurrence in upcoming:
        item = dict(occurrence)
        item["planned_hours"] = int(item["duration_minutes"]) // 60
        future_by_subject.setdefault(item["title"].strip().casefold(), []).append(item)

    subjects = []
    for source in report["subjects"]:
        subject = dict(source)
        subject.update(attendance_position(
            int(subject["attended"]), int(subject["completed"]), minimum_percentage,
        ))
        matches = future_by_subject.get(subject["title"].strip().casefold(), [])
        subject["next_class"] = matches[0] if matches else None
        hours_per_class = matches[0]["planned_hours"] if matches else 1
        subject["estimated_classes"] = (
            ceil(subject["needed"] / hours_per_class)
            if subject["needed"] and hours_per_class else None
        )
        subjects.append(subject)
    rank = {"critical": 0, "warning": 1, "safe": 2, "no-data": 3}
    subjects.sort(key=lambda item: (rank[item["status"]], item["percentage"] or 0, item["title"].casefold()))

    planned_subjects = {item["title"].strip().casefold(): item for item in subjects}
    planned_upcoming = []
    for occurrence in upcoming:
        item = dict(occurrence)
        item["attendance_plan"] = planned_subjects.get(item["title"].strip().casefold())
        planned_upcoming.append(item)
    return {
        "minimum": minimum_percentage,
        "overall": attendance_position(report["attended"], report["completed"], minimum_percentage),
        "subjects": subjects,
        "upcoming": planned_upcoming,
        "shortage_count": sum(item["status"] == "critical" for item in subjects),
    }
