"""SQLite storage and checkpoint rules for the local attendance monitor."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATABASE = ROOT / "instance" / "attendance_monitor.db"

SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS students (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 identity_label TEXT NOT NULL UNIQUE COLLATE NOCASE,
 display_name TEXT NOT NULL,
 registration_number TEXT NOT NULL UNIQUE COLLATE NOCASE,
 active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0,1)),
 created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS monitor_sessions (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 title TEXT NOT NULL, starts_at TEXT NOT NULL, ends_at TEXT NOT NULL,
 checkpoint_interval_minutes INTEGER NOT NULL,
 checkpoint_window_minutes INTEGER NOT NULL,
 status TEXT NOT NULL DEFAULT 'scheduled' CHECK (status IN ('scheduled','active','closed')),
 created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS monitor_checkpoints (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 session_id INTEGER NOT NULL, checkpoint_number INTEGER NOT NULL,
 opens_at TEXT NOT NULL, closes_at TEXT NOT NULL,
 UNIQUE(session_id, checkpoint_number),
 FOREIGN KEY(session_id) REFERENCES monitor_sessions(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS session_roster (
 session_id INTEGER NOT NULL,
 student_id INTEGER NOT NULL,
 identity_label_snapshot TEXT NOT NULL,
 display_name_snapshot TEXT NOT NULL,
 added_at TEXT NOT NULL,
 PRIMARY KEY(session_id, student_id),
 FOREIGN KEY(session_id) REFERENCES monitor_sessions(id) ON DELETE CASCADE,
 FOREIGN KEY(student_id) REFERENCES students(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS monitor_attendance (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 checkpoint_id INTEGER NOT NULL, student_id INTEGER NOT NULL,
 recognized_at TEXT NOT NULL, similarity REAL NOT NULL,
 liveness_passed INTEGER NOT NULL CHECK (liveness_passed IN (0,1)),
 UNIQUE(checkpoint_id, student_id),
 FOREIGN KEY(checkpoint_id) REFERENCES monitor_checkpoints(id) ON DELETE CASCADE,
 FOREIGN KEY(student_id) REFERENCES students(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS monitor_recognition_log (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 event_type TEXT NOT NULL, identity_label TEXT, occurred_at TEXT NOT NULL,
 similarity REAL, liveness_passed INTEGER NOT NULL DEFAULT 0,
 checkpoint_id INTEGER, student_id INTEGER, outcome TEXT NOT NULL,
 message TEXT NOT NULL, details_json TEXT,
 FOREIGN KEY(checkpoint_id) REFERENCES monitor_checkpoints(id),
 FOREIGN KEY(student_id) REFERENCES students(id)
);
CREATE INDEX IF NOT EXISTS idx_monitor_checkpoints_time ON monitor_checkpoints(opens_at, closes_at);
CREATE INDEX IF NOT EXISTS idx_monitor_log_time ON monitor_recognition_log(occurred_at);
CREATE INDEX IF NOT EXISTS idx_session_roster_student ON session_roster(student_id);
"""


@dataclass(frozen=True)
class AttendanceResult:
    outcome: str
    message: str
    attendance_id: int | None = None
    checkpoint_id: int | None = None
    student_id: int | None = None


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def ensure_aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("Datetime values must include a timezone")
    return value.astimezone(timezone.utc)


def datetime_text(value: datetime) -> str:
    return ensure_aware(value).isoformat(timespec="seconds")


def parse_datetime(value: str) -> datetime:
    return ensure_aware(datetime.fromisoformat(value.replace("Z", "+00:00")))


class AttendanceDatabase:
    def __init__(self, path: Path | str = DEFAULT_DATABASE) -> None:
        self.path = Path(path)

    def connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    @contextmanager
    def session(self) -> Iterator[sqlite3.Connection]:
        connection = self.connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.session() as connection:
            connection.executescript(SCHEMA)

    def students(self) -> list[sqlite3.Row]:
        with self.session() as connection:
            return connection.execute("SELECT * FROM students WHERE active=1 ORDER BY display_name").fetchall()

    def sync_identities(self, labels: list[str]) -> dict[str, int]:
        normalized = []
        seen = set()
        for raw_label in labels:
            label = str(raw_label).strip()
            key = label.casefold()
            if not label or label.lower() == "unknown" or key in seen:
                continue
            seen.add(key)
            normalized.append(label)
        inserted = 0
        with self.session() as connection:
            for label in normalized:
                display_name = label.replace("_", " ").replace("-", " ").strip().title()
                cursor = connection.execute(
                    """INSERT OR IGNORE INTO students(
                    identity_label,display_name,registration_number,created_at
                    ) VALUES (?,?,?,?)""",
                    (label, display_name or label, f"AUTO-{label}", datetime_text(utc_now())),
                )
                inserted += max(cursor.rowcount, 0)
        return {"labels": len(normalized), "inserted": inserted}

    def backfill_missing_rosters(self) -> int:
        now = datetime_text(utc_now())
        with self.session() as connection:
            sessions = connection.execute(
                """SELECT s.id FROM monitor_sessions s
                WHERE NOT EXISTS (
                    SELECT 1 FROM session_roster r WHERE r.session_id=s.id
                )"""
            ).fetchall()
            for row in sessions:
                connection.execute(
                    """INSERT OR IGNORE INTO session_roster(
                    session_id,student_id,identity_label_snapshot,display_name_snapshot,added_at
                    ) SELECT ?,id,identity_label,display_name,? FROM students WHERE active=1""",
                    (row["id"], now),
                )
            return len(sessions)

    def start_session(
        self, title: str, starts_at: datetime, duration_minutes: int = 130,
        checkpoint_interval_minutes: int = 65, checkpoint_window_minutes: int = 10,
    ) -> int:
        if not title.strip():
            raise ValueError("Session title is required")
        if min(duration_minutes, checkpoint_interval_minutes, checkpoint_window_minutes) < 1:
            raise ValueError("Duration, interval, and window must be positive")
        start = ensure_aware(starts_at)
        end = start + timedelta(minutes=duration_minutes)
        now = utc_now()
        status = "active" if start <= now < end else "scheduled"
        with self.session() as connection:
            connection.execute("UPDATE monitor_sessions SET status='closed' WHERE status='active'")
            cursor = connection.execute(
                """INSERT INTO monitor_sessions(
                title,starts_at,ends_at,checkpoint_interval_minutes,
                checkpoint_window_minutes,status,created_at) VALUES (?,?,?,?,?,?,?)""",
                (title.strip(), datetime_text(start), datetime_text(end), checkpoint_interval_minutes,
                 checkpoint_window_minutes, status, datetime_text(now)),
            )
            session_id = int(cursor.lastrowid)
            checkpoint_time = start
            number = 1
            while checkpoint_time < end:
                connection.execute(
                    "INSERT INTO monitor_checkpoints(session_id,checkpoint_number,opens_at,closes_at) VALUES (?,?,?,?)",
                    (session_id, number, datetime_text(checkpoint_time),
                     datetime_text(min(checkpoint_time + timedelta(minutes=checkpoint_window_minutes), end))),
                )
                checkpoint_time += timedelta(minutes=checkpoint_interval_minutes)
                number += 1
            connection.execute(
                """INSERT INTO session_roster(
                session_id,student_id,identity_label_snapshot,display_name_snapshot,added_at
                ) SELECT ?,id,identity_label,display_name,? FROM students WHERE active=1""",
                (session_id, datetime_text(now)),
            )
            return session_id

    def refresh_statuses(self, now: datetime | None = None) -> None:
        current = datetime_text(now or utc_now())
        with self.session() as connection:
            connection.execute(
                "UPDATE monitor_sessions SET status='active' WHERE status='scheduled' AND starts_at<=? AND ends_at>?",
                (current, current),
            )
            connection.execute(
                "UPDATE monitor_sessions SET status='closed' WHERE status!='closed' AND ends_at<=?", (current,)
            )

    def close_session(self, session_id: int) -> None:
        with self.session() as connection:
            connection.execute("UPDATE monitor_sessions SET status='closed' WHERE id=?", (session_id,))

    def record_recognition_event(self, event: dict[str, Any]) -> AttendanceResult:
        event_type = str(event.get("event", ""))
        identity = str(event.get("identity", "")).strip()
        live = event.get("liveness") == "passed"
        similarity = float(event.get("similarity", 0.0))
        occurred = parse_datetime(event.get("timestamp_utc", datetime_text(utc_now())))
        occurred_text = datetime_text(occurred)
        details = json.dumps(event, ensure_ascii=False)

        def log(connection, result: AttendanceResult) -> None:
            connection.execute(
                """INSERT INTO monitor_recognition_log(
                event_type,identity_label,occurred_at,similarity,liveness_passed,
                checkpoint_id,student_id,outcome,message,details_json) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (event_type, identity or None, occurred_text, similarity, int(live),
                 result.checkpoint_id, result.student_id, result.outcome, result.message, details),
            )

        self.refresh_statuses(occurred)
        with self.session() as connection:
            if event_type != "recognition_and_liveness_passed" or not live:
                result = AttendanceResult("rejected", "Recognition and liveness are both required")
                log(connection, result)
                return result
            if not identity or identity.lower() == "unknown":
                result = AttendanceResult("rejected", "Unknown person - attendance blocked")
                log(connection, result)
                return result
            student = connection.execute(
                "SELECT * FROM students WHERE identity_label=? AND active=1", (identity,)
            ).fetchone()
            if not student:
                display_name = identity.replace("_", " ").replace("-", " ").strip().title()
                cursor = connection.execute(
                    """INSERT INTO students(
                    identity_label,display_name,registration_number,created_at
                    ) VALUES (?,?,?,?)""",
                    (identity, display_name or identity, f"AUTO-{identity}", datetime_text(occurred)),
                )
                student = connection.execute(
                    "SELECT * FROM students WHERE id=?", (int(cursor.lastrowid),)
                ).fetchone()
            student_id = int(student["id"])
            checkpoint = connection.execute(
                """SELECT c.id,c.session_id FROM monitor_checkpoints c JOIN monitor_sessions s ON s.id=c.session_id
                WHERE s.status='active' AND c.opens_at<=? AND c.closes_at>=?
                ORDER BY c.opens_at DESC LIMIT 1""", (occurred_text, occurred_text),
            ).fetchone()
            if not checkpoint:
                result = AttendanceResult(
                    "no_active_checkpoint", "Recognized, but no attendance checkpoint is open",
                    student_id=student_id,
                )
                log(connection, result)
                return result
            checkpoint_id = int(checkpoint["id"])
            connection.execute(
                """INSERT OR IGNORE INTO session_roster(
                session_id,student_id,identity_label_snapshot,display_name_snapshot,added_at
                ) VALUES (?,?,?,?,?)""",
                (
                    int(checkpoint["session_id"]), student_id, student["identity_label"],
                    student["display_name"], occurred_text,
                ),
            )
            try:
                cursor = connection.execute(
                    "INSERT INTO monitor_attendance(checkpoint_id,student_id,recognized_at,similarity,liveness_passed) VALUES (?,?,?,?,1)",
                    (checkpoint_id, student_id, occurred_text, similarity),
                )
                result = AttendanceResult(
                    "marked", f"Attendance marked for {student['display_name']}",
                    int(cursor.lastrowid), checkpoint_id, student_id,
                )
                log(connection, result)
                return result
            except sqlite3.IntegrityError:
                existing = connection.execute(
                    "SELECT id FROM monitor_attendance WHERE checkpoint_id=? AND student_id=?",
                    (checkpoint_id, student_id),
                ).fetchone()
                result = AttendanceResult(
                    "already_marked", f"{student['display_name']} is already marked for this checkpoint",
                    int(existing["id"]), checkpoint_id, student_id,
                )
                log(connection, result)
                return result

    def session_history(self) -> list[dict[str, Any]]:
        self.refresh_statuses()
        current = datetime_text(utc_now())
        with self.session() as connection:
            rows = connection.execute(
                """SELECT s.*,
                (SELECT COUNT(*) FROM session_roster r WHERE r.session_id=s.id) roster_count,
                (SELECT COUNT(*) FROM monitor_checkpoints c WHERE c.session_id=s.id) checkpoint_count,
                (SELECT COUNT(*) FROM monitor_checkpoints c
                    WHERE c.session_id=s.id AND c.closes_at<=?) completed_checkpoint_count,
                (SELECT COUNT(*) FROM monitor_attendance a JOIN monitor_checkpoints c
                    ON c.id=a.checkpoint_id WHERE c.session_id=s.id) attendance_count,
                (SELECT COUNT(*) FROM monitor_attendance a JOIN monitor_checkpoints c
                    ON c.id=a.checkpoint_id
                    WHERE c.session_id=s.id AND c.closes_at<=?) completed_attendance_count
                FROM monitor_sessions s ORDER BY s.starts_at DESC,s.id DESC""",
                (current, current),
            ).fetchall()
        history = []
        for row in rows:
            item = dict(row)
            possible = item["roster_count"] * item["completed_checkpoint_count"]
            item["attendance_percentage"] = (
                round(item["completed_attendance_count"] * 100 / possible, 1)
                if possible else None
            )
            history.append(item)
        return history

    def session_detail(self, session_id: int) -> dict[str, Any] | None:
        self.refresh_statuses()
        now = utc_now()
        now_text = datetime_text(now)
        with self.session() as connection:
            session_row = connection.execute(
                "SELECT * FROM monitor_sessions WHERE id=?", (session_id,)
            ).fetchone()
            if not session_row:
                return None
            checkpoints = [dict(row) for row in connection.execute(
                """SELECT c.*,COUNT(a.id) attendance_count FROM monitor_checkpoints c
                LEFT JOIN monitor_attendance a ON a.checkpoint_id=c.id
                WHERE c.session_id=? GROUP BY c.id ORDER BY c.checkpoint_number""",
                (session_id,),
            ).fetchall()]
            roster = [dict(row) for row in connection.execute(
                """SELECT r.*,s.registration_number FROM session_roster r
                JOIN students s ON s.id=r.student_id
                WHERE r.session_id=? ORDER BY r.display_name_snapshot""",
                (session_id,),
            ).fetchall()]
            attendance = [dict(row) for row in connection.execute(
                """SELECT a.*,c.checkpoint_number FROM monitor_attendance a
                JOIN monitor_checkpoints c ON c.id=a.checkpoint_id
                WHERE c.session_id=? ORDER BY c.checkpoint_number,a.recognized_at""",
                (session_id,),
            ).fetchall()]
        record_map = {(row["student_id"], row["checkpoint_id"]): row for row in attendance}
        completed_ids = {
            checkpoint["id"] for checkpoint in checkpoints
            if checkpoint["closes_at"] <= now_text
        }
        for student in roster:
            cells = []
            attended_completed = 0
            attended_total = 0
            for checkpoint in checkpoints:
                record = record_map.get((student["student_id"], checkpoint["id"]))
                if record:
                    status = "present"
                    attended_total += 1
                    if checkpoint["id"] in completed_ids:
                        attended_completed += 1
                elif checkpoint["closes_at"] <= now_text:
                    status = "absent"
                elif checkpoint["opens_at"] <= now_text <= checkpoint["closes_at"]:
                    status = "open"
                else:
                    status = "upcoming"
                cells.append({"checkpoint": checkpoint, "record": record, "status": status})
            student["cells"] = cells
            student["attended_count"] = attended_total
            student["attendance_percentage"] = (
                round(attended_completed * 100 / len(completed_ids), 1)
                if completed_ids else None
            )
        completed_possible = len(completed_ids) * len(roster)
        completed_present = sum(
            1 for row in attendance if row["checkpoint_id"] in completed_ids
        )
        return {
            "session": dict(session_row),
            "checkpoints": checkpoints,
            "roster": roster,
            "attendance": attendance,
            "completed_checkpoint_count": len(completed_ids),
            "attendance_percentage": (
                round(completed_present * 100 / completed_possible, 1)
                if completed_possible else None
            ),
        }

    def monitor_data(self, log_limit: int = 30) -> dict[str, Any]:
        self.refresh_statuses()
        with self.session() as connection:
            current = connection.execute(
                "SELECT * FROM monitor_sessions ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            checkpoints: list[sqlite3.Row] = []
            attendance: list[sqlite3.Row] = []
            if current:
                checkpoints = connection.execute(
                    """SELECT c.*,COUNT(a.id) attendance_count FROM monitor_checkpoints c
                    LEFT JOIN monitor_attendance a ON a.checkpoint_id=c.id
                    WHERE c.session_id=? GROUP BY c.id ORDER BY c.checkpoint_number""", (current["id"],),
                ).fetchall()
                attendance = connection.execute(
                    """SELECT a.*,s.display_name,s.registration_number,s.identity_label,c.checkpoint_number
                    FROM monitor_attendance a JOIN students s ON s.id=a.student_id
                    JOIN monitor_checkpoints c ON c.id=a.checkpoint_id
                    WHERE c.session_id=? ORDER BY a.recognized_at DESC""", (current["id"],),
                ).fetchall()
            logs = connection.execute(
                """SELECT l.*,s.display_name,s.registration_number FROM monitor_recognition_log l
                LEFT JOIN students s ON s.id=l.student_id
                ORDER BY l.occurred_at DESC,l.id DESC LIMIT ?""", (log_limit,),
            ).fetchall()
            students = connection.execute("SELECT * FROM students WHERE active=1 ORDER BY display_name").fetchall()
        return {"session": dict(current) if current else None,
                "checkpoints": [dict(row) for row in checkpoints],
                "attendance": [dict(row) for row in attendance],
                "logs": [dict(row) for row in logs],
                "students": [dict(row) for row in students],
                "history": self.session_history()[:5]}
