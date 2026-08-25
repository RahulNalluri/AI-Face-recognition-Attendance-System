"""Command-line setup for the local attendance monitor."""

import argparse
import json
from pathlib import Path
from attendance_db import AttendanceDatabase


ROOT = Path(__file__).resolve().parent.parent
LABELS_PATH = ROOT / "models" / "sface" / "labels.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init", help="Create or upgrade the local database.")
    commands.add_parser("list-identities", help="List identities seen by the recognizer.")
    args = parser.parse_args()
    database = AttendanceDatabase()
    database.initialize()
    if LABELS_PATH.exists():
        payload = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
        labels = list(payload.values()) if isinstance(payload, dict) else list(payload)
        database.sync_identities([str(label) for label in labels])
        database.backfill_missing_rosters()
    if args.command == "init":
        print(f"Database ready: {database.path}")
    elif args.command == "list-identities":
        for student in database.students():
            print(f"{student['id']:>4}  {student['registration_number']:<16}  {student['identity_label']:<22}  {student['display_name']}")


if __name__ == "__main__":
    main()
