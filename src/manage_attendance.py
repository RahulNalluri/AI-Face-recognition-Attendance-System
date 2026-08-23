"""Command-line setup for the local attendance monitor."""

import argparse
from attendance_db import AttendanceDatabase


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init", help="Create or upgrade the local database.")
    commands.add_parser("list-identities", help="List identities seen by the recognizer.")
    args = parser.parse_args()
    database = AttendanceDatabase()
    database.initialize()
    if args.command == "init":
        print(f"Database ready: {database.path}")
    elif args.command == "list-identities":
        for student in database.students():
            print(f"{student['id']:>4}  {student['registration_number']:<16}  {student['identity_label']:<22}  {student['display_name']}")


if __name__ == "__main__":
    main()
