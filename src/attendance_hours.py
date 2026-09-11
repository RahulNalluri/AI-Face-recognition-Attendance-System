"""Convert checkpoint evidence into one attendance decision per teaching hour."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

REQUIRED_PASSED_CHECKPOINTS = 2
ATTENDANCE_HOUR_MINUTES = 60


def _parse(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Datetime values must include a timezone")
    return parsed.astimezone(timezone.utc)


def _text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds")


def build_hour_blocks(
    session: dict[str, Any], checkpoints: Iterable[dict[str, Any]], now,
) -> list[dict[str, Any]]:
    """Build complete 60-minute blocks; short trailing fragments are not attendance hours."""
    start, end = _parse(session["starts_at"]), _parse(session["ends_at"])
    checkpoint_rows = [dict(row) for row in checkpoints]
    blocks = []
    hour_start = start
    number = 1
    while hour_start + timedelta(minutes=ATTENDANCE_HOUR_MINUTES) <= end:
        hour_end = hour_start + timedelta(minutes=ATTENDANCE_HOUR_MINUTES)
        included = [
            row for row in checkpoint_rows
            if hour_start <= _parse(row["opens_at"]) < hour_end
        ]
        evaluation_time = max(
            [hour_end] + [_parse(row["closes_at"]) for row in included]
        )
        blocks.append({
            "hour_number": number,
            "starts_at": _text(hour_start),
            "ends_at": _text(hour_end),
            "evaluation_at": _text(evaluation_time),
            "checkpoint_ids": [int(row["id"]) for row in included],
            "checkpoint_count": len(included),
            "required_passes": REQUIRED_PASSED_CHECKPOINTS,
            "completed": evaluation_time <= now,
            "evaluable": len(included) >= REQUIRED_PASSED_CHECKPOINTS,
        })
        hour_start = hour_end
        number += 1
    return blocks


def evaluate_hour(block: dict[str, Any], passed_checkpoint_ids: set[int]) -> dict[str, Any]:
    result = dict(block)
    result["passed_checkpoints"] = sum(
        checkpoint_id in passed_checkpoint_ids for checkpoint_id in block["checkpoint_ids"]
    )
    if not block["completed"]:
        result["status"] = "upcoming"
    elif not block["evaluable"]:
        result["status"] = "insufficient-checks"
    elif result["passed_checkpoints"] >= block["required_passes"]:
        result["status"] = "present"
    else:
        result["status"] = "absent"
    return result
