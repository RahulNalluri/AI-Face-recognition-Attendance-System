"""Atomic section metadata, roster and weekly-grid editing."""
import json
from datetime import datetime

from timetable import Timetable, timetable_timezone


def default_grid(timezone_name="Asia/Kolkata"):
    return {"day_start": "09:30", "timezone": timezone_name, "enabled": False,
            "lunch_after": 3,
            "periods": [{"duration": 60, "break_after": 60 if i == 2 else 0, "repeat": 0, "window": 10} for i in range(6)],
            "subjects": [[""] * 6 for _ in range(6)]}


def normalize_grid(raw):
    if not isinstance(raw, dict):
        raise ValueError("Timetable grid must be an object")
    periods, subjects = raw.get("periods"), raw.get("subjects")
    if not isinstance(periods, list) or not 1 <= len(periods) <= 10:
        raise ValueError("Choose between 1 and 10 periods")
    if not isinstance(subjects, list) or len(subjects) != 6 or any(
        not isinstance(row, list) or len(row) != len(periods) for row in subjects
    ):
        raise ValueError("Provide a Monday–Saturday subject grid matching the period count")
    # Old saved grids retain their timings until lunch is explicitly selected.
    lunch_after = raw.get("lunch_after")
    if lunch_after is not None:
        try:
            lunch_after = int(str(lunch_after))
        except (TypeError, ValueError):
            raise ValueError("Choose a valid period for lunch") from None
        if not 1 <= lunch_after <= len(periods):
            raise ValueError("Lunch must follow an existing period")
    try:
        parsed = datetime.strptime(raw.get("day_start", ""), "%H:%M")
        minute = parsed.hour * 60 + parsed.minute
        clean_periods = []
        for index, period in enumerate(periods):
            duration, gap, repeat, window = (int(str(period[field])) for field in ("duration", "break_after", "repeat", "window"))
            if not 1 <= duration <= 1440 or not 0 <= gap <= 1440:
                raise ValueError("Each period needs a positive duration and a nonnegative break")
            if lunch_after == index + 1:
                gap = 60
            if not 0 <= repeat <= duration or not 1 <= window <= (repeat or duration):
                raise ValueError("Attendance window must fit within Repeat after (or the period for a single check)")
            if minute + duration > 1440:
                raise ValueError("All periods must finish by midnight; reduce durations or breaks")
            clean_periods.append({"duration": duration, "break_after": gap, "repeat": repeat,
                                  "window": window, "start_time": f"{minute // 60:02d}:{minute % 60:02d}"})
            minute += duration + gap
            if lunch_after == index + 1 and minute > 1440:
                raise ValueError("The one-hour lunch break must finish by midnight")
    except (KeyError, TypeError, OverflowError):
        raise ValueError("Provide valid numbers for every period") from None
    zone = str(raw.get("timezone", "Asia/Kolkata"))
    timetable_timezone(zone)
    if not isinstance(raw.get("enabled", False), bool):
        raise ValueError("Timetable enabled must be true or false")
    clean_subjects = []
    for row in subjects:
        if any(not isinstance(subject, str) or len(subject.strip()) > 100 for subject in row):
            raise ValueError("Each subject must be text with at most 100 characters")
        clean_subjects.append([subject.strip() for subject in row])
    return {"day_start": parsed.strftime("%H:%M"), "timezone": zone,
            "enabled": raw.get("enabled", False), "lunch_after": lunch_after,
            "periods": clean_periods, "subjects": clean_subjects}


class SectionSetup:
    def __init__(self, database):
        self.database = database
        self.timetable = Timetable(database)

    def profile(self, group_id, timezone_name="Asia/Kolkata"):
        with self.database.session() as connection:
            row = connection.execute("SELECT * FROM section_profiles WHERE group_id=?", (group_id,)).fetchone()
        if row:
            return {"department": row["department"], "semester": row["semester"],
                    "academic_year": row["academic_year"], "grid": json.loads(row["grid_json"])}
        return {"department": "", "semester": "", "academic_year": "", "grid": default_grid(timezone_name)}

    def save(self, *, name, description, student_ids, department, semester, academic_year, grid, group_id=None):
        grid = normalize_grid(grid)
        metadata = [str(value).strip() for value in (department, semester, academic_year)]
        if any(len(value) > 80 for value in metadata):
            raise ValueError("Department, semester and academic year must each be 80 characters or fewer")
        with self.database.session() as connection:
            connection.execute("BEGIN IMMEDIATE")
            group_id = self.database.save_class_group(name, description, student_ids, group_id, _connection=connection)
            # Temporarily disable this grid inside the transaction so time edits don't conflict
            # with its previous layout. Other standalone entries are still checked for overlap.
            connection.execute(
                "UPDATE timetable_entries SET enabled=0 WHERE id IN (SELECT entry_id FROM section_grid_cells WHERE group_id=?)",
                (group_id,),
            )
            mappings = {(row["weekday"], row["period_index"]): row["entry_id"] for row in connection.execute(
                "SELECT * FROM section_grid_cells WHERE group_id=?", (group_id,)
            )}
            for weekday, subjects in enumerate(grid["subjects"]):
                for index, subject in enumerate(subjects):
                    if not subject:
                        continue
                    period = grid["periods"][index]
                    old_id = mappings.get((weekday, index))
                    entry_id = self.timetable.save(
                        group_id=group_id, title=subject, weekday=weekday,
                        start_time=period["start_time"], duration_minutes=period["duration"],
                        checkpoint_interval_minutes=period["repeat"] or period["duration"],
                        checkpoint_window_minutes=period["window"], timezone_name=grid["timezone"],
                        enabled=grid["enabled"], entry_id=old_id, _connection=connection,
                    )
                    if old_id is None:
                        connection.execute(
                            "INSERT INTO section_grid_cells(group_id,weekday,period_index,entry_id) VALUES (?,?,?,?)",
                            (group_id, weekday, index, entry_id),
                        )
            # Blank or removed cells retain their disabled IDs and run history, avoiding
            # duplicate occurrences if the user later fills the same slot again.
            connection.execute(
                """INSERT INTO section_profiles(group_id,department,semester,academic_year,grid_json)
                VALUES (?,?,?,?,?) ON CONFLICT(group_id) DO UPDATE SET department=excluded.department,
                semester=excluded.semester,academic_year=excluded.academic_year,grid_json=excluded.grid_json""",
                (group_id, *metadata, json.dumps(grid)),
            )
        return group_id
