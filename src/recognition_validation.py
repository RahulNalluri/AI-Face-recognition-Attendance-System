"""Pure recognition-validation metrics shared by the CLI and tests."""

from __future__ import annotations

from collections import Counter
from typing import Any


DEFAULT_GATES = {
    "known_detection_rate": 0.90,
    "known_identification_rate": 0.90,
    "unknown_detection_rate": 0.90,
    "unknown_rejection_rate": 0.95,
}


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def summarize_records(
    records: list[dict[str, Any]],
    enrolled_labels: list[str],
    gates: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Calculate deployment metrics without tuning the locked threshold."""
    limits = {**DEFAULT_GATES, **(gates or {})}
    known = [row for row in records if row["expected"] != "Unknown"]
    unknown = [row for row in records if row["expected"] == "Unknown"]
    known_detected = [row for row in known if row["detected"]]
    unknown_detected = [row for row in unknown if row["detected"]]
    known_correct = sum(row["predicted"] == row["expected"] for row in known)
    known_rejected = sum(
        not row["detected"] or row["predicted"] == "Unknown" for row in known
    )
    known_misidentified = sum(
        row["detected"] and row["predicted"] not in {row["expected"], "Unknown"}
        for row in known
    )
    unknown_rejected = sum(row["predicted"] == "Unknown" for row in unknown_detected)

    known_metrics = {
        "samples": len(known),
        "detected": len(known_detected),
        "detection_rate": _rate(len(known_detected), len(known)),
        "correct": known_correct,
        "identification_rate": _rate(known_correct, len(known)),
        "false_rejects": known_rejected,
        "false_reject_rate": _rate(known_rejected, len(known)),
        "misidentifications": known_misidentified,
        "misidentification_rate": _rate(known_misidentified, len(known)),
    }
    unknown_metrics = {
        "samples": len(unknown),
        "detected": len(unknown_detected),
        "detection_rate": _rate(len(unknown_detected), len(unknown)),
        "rejected": unknown_rejected,
        "rejection_rate": _rate(unknown_rejected, len(unknown_detected)),
    }

    per_identity = []
    for label in enrolled_labels:
        rows = [row for row in known if row["expected"] == label]
        similarities = [float(row["similarity"]) for row in rows if row.get("similarity") is not None]
        per_identity.append({
            "identity": label,
            "samples": len(rows),
            "detected": sum(bool(row["detected"]) for row in rows),
            "correct": sum(row["predicted"] == label for row in rows),
            "rejected": sum(not row["detected"] or row["predicted"] == "Unknown" for row in rows),
            "misidentified": sum(
                row["detected"] and row["predicted"] not in {label, "Unknown"} for row in rows
            ),
            "accuracy": _rate(sum(row["predicted"] == label for row in rows), len(rows)),
            "mean_similarity": round(sum(similarities) / len(similarities), 6) if similarities else None,
        })

    confusion: dict[str, dict[str, int]] = {}
    for row in records:
        confusion.setdefault(row["expected"], {})
        confusion[row["expected"]][row["predicted"]] = (
            confusion[row["expected"]].get(row["predicted"], 0) + 1
        )

    failures = []
    if not known:
        failures.append("No enrolled validation images were found")
    elif known_metrics["detection_rate"] < limits["known_detection_rate"]:
        failures.append("Enrolled face-detection rate is below the required gate")
    if known and known_metrics["identification_rate"] < limits["known_identification_rate"]:
        failures.append("Enrolled identification rate is below the required gate")
    if unknown:
        if unknown_metrics["detection_rate"] < limits["unknown_detection_rate"]:
            failures.append("Unknown face-detection rate is below the required gate")
        if (
            unknown_metrics["rejection_rate"] is None
            or unknown_metrics["rejection_rate"] < limits["unknown_rejection_rate"]
        ):
            failures.append("Unknown-person rejection rate is below the required gate")

    if failures:
        status = "failed"
    elif not unknown:
        status = "incomplete"
    else:
        status = "passed"
    notices = list(failures)
    if not unknown:
        notices.append(
            "Real unknown-person images are missing; deployment readiness cannot be claimed"
        )
    return {
        "status": status,
        "deployment_ready": status == "passed",
        "gates": limits,
        "known": known_metrics,
        "unknown": unknown_metrics,
        "per_identity": per_identity,
        "confusion": confusion,
        "notices": notices,
        "prediction_counts": dict(Counter(row["predicted"] for row in records)),
    }
