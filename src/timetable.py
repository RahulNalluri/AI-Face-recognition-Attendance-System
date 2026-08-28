"""Weekly, single-camera schedules with durable once-per-date execution records."""

from datetime import datetime, timedelta, timezone
import logging
import json
import threading
import time
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from contextlib import nullcontext

from attendance_db import datetime_text, ensure_aware, utc_now

WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


def timetable_timezone(name):
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError:
        if name in {"Asia/Kolkata", "Asia/Calcutta"}:
            return timezone(timedelta(hours=5, minutes=30))
        if name == "UTC":
            return timezone.utc
        raise ValueError("Timezone is unavailable; install tzdata or use Asia/Kolkata or UTC") from None


class Timetable:
    def __init__(self, database):
        self.database = database

    def entries(self):
        with self.database.session() as connection:
            rows = [dict(row) for row in connection.execute(
                """SELECT t.*,g.name group_name,m.group_id managed_group_id,
                m.weekday grid_weekday,m.period_index grid_period,p.grid_json,
                (SELECT COUNT(*) FROM class_group_members m JOIN students s ON s.id=m.student_id
                 WHERE m.group_id=t.group_id AND s.active=1) member_count
                FROM timetable_entries t JOIN class_groups g ON g.id=t.group_id
                LEFT JOIN section_grid_cells m ON m.entry_id=t.id
                LEFT JOIN section_profiles p ON p.group_id=m.group_id
                ORDER BY t.weekday,t.start_time,t.id"""
            )]
        visible = []
        for row in rows:
            raw = row.pop("grid_json")
            day, period = row.pop("grid_weekday"), row.pop("grid_period")
            if raw and row["managed_group_id"] is not None:
                subjects = json.loads(raw)["subjects"]
                if period >= len(subjects[day]) or not subjects[day][period]:
                    continue
            visible.append(row)
        return visible

    def entry(self, entry_id):
        return next((row for row in self.entries() if row["id"] == entry_id), None)

    def camera_section(self):
        with self.database.session() as connection:
            return dict(connection.execute(
                "SELECT c.group_id,g.name FROM camera_section c LEFT JOIN class_groups g ON g.id=c.group_id WHERE c.id=1"
            ).fetchone())

    def select_camera_section(self, group_id):
        with self.database.session() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute("SELECT group_id FROM camera_section WHERE id=1").fetchone()[0]
            if current == group_id:
                return
            if group_id is not None and not connection.execute(
                "SELECT 1 FROM class_groups WHERE id=?", (group_id,)
            ).fetchone():
                raise ValueError("Choose an existing section")
            now = datetime_text(utc_now())
            if connection.execute(
                """SELECT 1 FROM monitor_sessions WHERE status!='closed'
                AND starts_at<=? AND ends_at>? LIMIT 1""", (now, now)
            ).fetchone():
                raise ValueError("Close the current attendance session before switching the camera section")
            connection.execute("UPDATE camera_section SET group_id=? WHERE id=1", (group_id,))

    def managed_group(self, entry_id):
        with self.database.session() as connection:
            row = connection.execute("SELECT group_id FROM section_grid_cells WHERE entry_id=?", (entry_id,)).fetchone()
            return row[0] if row else None

    def save(self, *, group_id, title, weekday, start_time, duration_minutes,
             checkpoint_interval_minutes, checkpoint_window_minutes, timezone_name,
             enabled=True, entry_id=None, _connection=None):
        title = title.strip()
        if not title or len(title) > 100:
            raise ValueError("Subject/session title must contain 1 to 100 characters")
        if weekday not in range(7):
            raise ValueError("Choose a weekday from Monday to Sunday")
        try:
            parsed = datetime.strptime(start_time, "%H:%M")
        except (TypeError, ValueError):
            raise ValueError("Start time must use HH:MM format") from None
        start_time = parsed.strftime("%H:%M")
        start_minute = parsed.hour * 60 + parsed.minute
        if not 1 <= duration_minutes <= 1440 or start_minute + duration_minutes > 1440:
            raise ValueError("Duration must be positive and the class must end by midnight")
        if not 1 <= checkpoint_interval_minutes <= duration_minutes:
            raise ValueError("Repeat after must be between 1 minute and the class duration")
        if not 1 <= checkpoint_window_minutes <= checkpoint_interval_minutes:
            raise ValueError("Open for must be between 1 minute and Repeat after")
        timetable_timezone(timezone_name)
        now = datetime_text(utc_now())
        with (self.database.session() if _connection is None else nullcontext(_connection)) as connection:
            if _connection is None:
                connection.execute("BEGIN IMMEDIATE")
                if entry_id is not None and connection.execute(
                    "SELECT 1 FROM section_grid_cells WHERE entry_id=?", (entry_id,)
                ).fetchone():
                    raise ValueError("Edit this entry through its section timetable grid")
            if entry_id is not None and not connection.execute(
                "SELECT 1 FROM timetable_entries WHERE id=?", (entry_id,)
            ).fetchone():
                raise ValueError("Timetable entry does not exist")
            if not connection.execute("SELECT 1 FROM class_groups WHERE id=?", (group_id,)).fetchone():
                raise ValueError("Choose an existing class group")
            if enabled:
                if not connection.execute(
                    """SELECT 1 FROM class_group_members m JOIN students s ON s.id=m.student_id
                    WHERE m.group_id=? AND s.active=1 LIMIT 1""", (group_id,)
                ).fetchone():
                    raise ValueError("Add students to the class before enabling its timetable")
                others = connection.execute(
                    "SELECT * FROM timetable_entries WHERE enabled=1 AND group_id=? AND id!=?", (group_id, entry_id or -1)
                ).fetchall()
                for other in others:
                    if other["timezone"] != timezone_name:
                        raise ValueError("All enabled entries in a section must use the same timezone")
                    hour, minute = map(int, other["start_time"].split(":"))
                    other_start = hour * 60 + minute
                    if other["weekday"] == weekday and (
                        start_minute < other_start + other["duration_minutes"]
                        and other_start < start_minute + duration_minutes
                    ):
                        raise ValueError(f"Overlaps with '{other['title']}' in this section")
            values = (group_id, title, weekday, start_time, timezone_name, duration_minutes,
                      checkpoint_interval_minutes, checkpoint_window_minutes, int(enabled), now)
            if entry_id is None:
                cursor = connection.execute(
                    """INSERT INTO timetable_entries(group_id,title,weekday,start_time,timezone,
                    duration_minutes,checkpoint_interval_minutes,checkpoint_window_minutes,enabled,
                    updated_at,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)""", values + (now,)
                )
                return int(cursor.lastrowid)
            connection.execute(
                """UPDATE timetable_entries SET group_id=?,title=?,weekday=?,start_time=?,timezone=?,
                duration_minutes=?,checkpoint_interval_minutes=?,checkpoint_window_minutes=?,enabled=?,
                updated_at=? WHERE id=?""", values + (entry_id,)
            )
            return entry_id

    def set_enabled(self, entry_id, enabled):
        if self.managed_group(entry_id) is not None:
            raise ValueError("Edit this entry through its section timetable grid")
        # Pause is a single write; enabling goes through overlap/roster validation.
        if not enabled:
            with self.database.session() as connection:
                result = connection.execute(
                    "UPDATE timetable_entries SET enabled=0,updated_at=? WHERE id=?",
                    (datetime_text(utc_now()), entry_id),
                )
                if not result.rowcount:
                    raise ValueError("Timetable entry does not exist")
            return
        row = self.entry(entry_id)
        if row is None:
            raise ValueError("Timetable entry does not exist")
        self.save(group_id=row["group_id"], title=row["title"], weekday=row["weekday"],
                  start_time=row["start_time"], duration_minutes=row["duration_minutes"],
                  checkpoint_interval_minutes=row["checkpoint_interval_minutes"],
                  checkpoint_window_minutes=row["checkpoint_window_minutes"],
                  timezone_name=row["timezone"], enabled=True, entry_id=entry_id)

    @staticmethod
    def occurrence(entry, local_date):
        zone = timetable_timezone(entry["timezone"])
        wall = datetime.strptime(f"{local_date.isoformat()} {entry['start_time']}", "%Y-%m-%d %H:%M")
        start = wall.replace(tzinfo=zone).astimezone(timezone.utc)
        # Do not silently shift a nonexistent wall-clock time during a DST jump.
        if start.astimezone(zone).replace(tzinfo=None) != wall:
            return None
        return start

    def run_due(self, now=None):
        now = ensure_aware(now or utc_now())
        created = []
        # Serialize claims and session creation across threads/processes, not just this object.
        with self.database.session() as connection:
            connection.execute("BEGIN IMMEDIATE")
            entries = connection.execute(
                """SELECT t.*,g.name group_name FROM timetable_entries t
                JOIN class_groups g ON g.id=t.group_id
                WHERE t.enabled=1 AND t.group_id=(SELECT group_id FROM camera_section WHERE id=1)
                ORDER BY t.start_time,t.id"""
            ).fetchall()
            for entry in entries:
                local_date = now.astimezone(timetable_timezone(entry["timezone"])).date()
                if local_date.weekday() != entry["weekday"]:
                    continue
                start = self.occurrence(entry, local_date)
                if start is None or now < start:
                    continue
                occurrence_date = local_date.isoformat()
                if connection.execute(
                    "SELECT 1 FROM timetable_runs WHERE entry_id=? AND occurrence_date=?",
                    (entry["id"], occurrence_date),
                ).fetchone():
                    continue
                end = start + timedelta(minutes=entry["duration_minutes"])
                session_id = None
                if now >= start + timedelta(minutes=entry["checkpoint_window_minutes"]):
                    outcome, message = "missed", "First checkpoint window was missed; no attendance session created"
                elif connection.execute(
                    """SELECT 1 FROM monitor_sessions WHERE status!='closed'
                    AND starts_at<? AND ends_at>? LIMIT 1""", (datetime_text(end), datetime_text(start)),
                ).fetchone():
                    outcome, message = "conflict", "An existing session overlaps this class; it was left unchanged"
                else:
                    try:
                        session_id = self.database._create_session(
                            connection, entry["title"], start, entry["duration_minutes"],
                            entry["checkpoint_interval_minutes"], entry["checkpoint_window_minutes"],
                            entry["group_id"], now, replace_active=False,
                        )
                        created.append(session_id)
                        outcome, message = "created", "Attendance session created with the current class roster"
                    except ValueError as error:
                        outcome, message = "empty_roster", str(error)
                connection.execute(
                    """INSERT INTO timetable_runs(entry_id,occurrence_date,session_id,outcome,message,
                    title_snapshot,group_name_snapshot,starts_at,checked_at) VALUES (?,?,?,?,?,?,?,?,?)""",
                    (entry["id"], occurrence_date, session_id, outcome, message, entry["title"],
                     entry["group_name"], datetime_text(start), datetime_text(now)),
                )
            connection.execute(
                """UPDATE monitor_sessions SET status='closed' WHERE status!='closed' AND ends_at<=?
                AND id IN (SELECT session_id FROM timetable_runs WHERE session_id IS NOT NULL)""",
                (datetime_text(now),),
            )
        return created

    def runs(self):
        with self.database.session() as connection:
            return [dict(row) for row in connection.execute(
                "SELECT * FROM timetable_runs ORDER BY checked_at DESC,entry_id DESC LIMIT 30"
            )]

    def upcoming(self, now=None):
        now = ensure_aware(now or utc_now())
        with self.database.session() as connection:
            consumed = {(r["entry_id"], r["occurrence_date"]) for r in connection.execute(
                "SELECT entry_id,occurrence_date FROM timetable_runs WHERE occurrence_date>=?",
                ((now - timedelta(days=1)).date().isoformat(),),
            )}
        upcoming = []
        for entry in self.entries():
            if not entry["enabled"]:
                continue
            today = now.astimezone(timetable_timezone(entry["timezone"])).date()
            for offset in range(8):
                day = today + timedelta(days=offset)
                if day.weekday() != entry["weekday"] or (entry["id"], day.isoformat()) in consumed:
                    continue
                start = self.occurrence(entry, day)
                if start is not None and start >= now:
                    upcoming.append({**entry, "next_start": datetime_text(start)})
                    break
        return sorted(upcoming, key=lambda row: row["next_start"])


class TimetableScheduler:
    """Background runner for the local app; request checks also support other launchers."""
    def __init__(self, timetable, interval=15):
        self.timetable = timetable
        self.interval = interval
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = None
        self._last_tick = None
        self.last_error = None

    def check_due(self, force=False):
        with self._lock:
            if not force and self._last_tick is not None and time.monotonic() - self._last_tick < self.interval:
                return
            try:
                self.timetable.run_due()
                self.last_error = None
            except Exception:
                logging.getLogger(__name__).exception("Timetable check failed")
                self.last_error = "Timetable check failed. Check the server log; the next check will retry."
            self._last_tick = time.monotonic()

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="attendance-timetable")
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            self.check_due(force=True)
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def status(self):
        return {"background_running": self._thread is not None and self._thread.is_alive(),
                "last_error": self.last_error, "interval_seconds": self.interval}
