"""Read a client export, validate it, and load it into SQLite.

The client sends the same seven CSVs every Monday. Loading a fresh export must
not require a developer, so everything here is designed around one rule: check
the whole export first, report every problem in plain language, and only then
write. A bad export leaves the live database untouched.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from .config import (
    DEFAULT_DATA_DIR,
    DEFAULT_DB_PATH,
    MAX_SHIFT_HOURS,
    SOURCE_FILES,
    SourceFile,
)
from .db import connect, data_range, init_schema, staged_database, table_counts

DATE_FORMAT = "%Y-%m-%d"
TIME_FORMAT = "%H:%M"
STAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

#: How many offending values to name before summarising the rest.
EXAMPLE_LIMIT = 3


class IngestError(RuntimeError):
    """Raised when an export cannot be loaded. Carries the full report."""

    def __init__(self, report: IngestReport) -> None:
        super().__init__(report.render())
        self.report = report


@dataclass(frozen=True)
class Problem:
    """Something wrong, or merely worth knowing, about an export."""

    source: str
    message: str
    fatal: bool = True

    def __str__(self) -> str:
        marker = "ERROR" if self.fatal else "note "
        return f"  {marker}  {self.source}: {self.message}"


@dataclass
class IngestReport:
    """Everything the ingestion found and did."""

    source_dir: str
    problems: list[Problem] = field(default_factory=list)
    rows_read: dict[str, int] = field(default_factory=dict)
    rows_in_db: dict[str, int] = field(default_factory=dict)
    date_range: dict[str, object] = field(default_factory=dict)
    written: bool = False

    @property
    def errors(self) -> list[Problem]:
        return [p for p in self.problems if p.fatal]

    @property
    def notes(self) -> list[Problem]:
        return [p for p in self.problems if not p.fatal]

    @property
    def ok(self) -> bool:
        return not self.errors

    def render(self) -> str:
        lines = [f"Export: {self.source_dir}"]

        if self.errors:
            lines.append("")
            lines.append(f"{len(self.errors)} problem(s) stopped this export loading:")
            lines.extend(str(p) for p in self.errors)

        if self.notes:
            lines.append("")
            lines.append("Worth knowing:")
            lines.extend(str(p) for p in self.notes)

        if self.rows_read:
            lines.append("")
            lines.append("Rows in this export:")
            width = max(len(t) for t in self.rows_read)
            for table, count in self.rows_read.items():
                total = self.rows_in_db.get(table)
                suffix = f"   (database now holds {total:,})" if total is not None else ""
                lines.append(f"  {table:<{width}}  {count:>7,}{suffix}")

        if self.date_range:
            lines.append("")
            lines.append(
                "Shift dates: {min_shift_date} to {max_shift_date}".format(**self.date_range)
            )
            lines.append(
                "Reporting week: {current_week_start} to {current_week_end}".format(
                    **self.date_range
                )
            )

        lines.append("")
        if not self.ok:
            lines.append("Nothing was written. The existing data is unchanged.")
        elif self.written:
            lines.append("Loaded successfully.")
        else:
            lines.append("Checks passed. Nothing was written (dry run).")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #


def _examples(values: Iterable[object]) -> str:
    """Name a few offending values so the reader can go and look at them."""
    shown = [str(v) for v in list(values)[:EXAMPLE_LIMIT]]
    return ", ".join(repr(v) for v in shown)


def read_export(source_dir: Path) -> tuple[dict[str, pd.DataFrame], list[Problem]]:
    """Read every CSV as text. Nothing is coerced yet, so validation sees the file
    as it actually is rather than as pandas guessed it."""
    frames: dict[str, pd.DataFrame] = {}
    problems: list[Problem] = []

    for source in SOURCE_FILES:
        path = source_dir / source.filename
        if not path.exists():
            problems.append(Problem(source.filename, "file is missing from the export"))
            continue

        try:
            # Only a truly empty field counts as missing; values like "NA" are kept
            # as the text the client actually sent.
            frame = pd.read_csv(
                path, dtype=str, keep_default_na=False, na_values=[""], skipinitialspace=True
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the user verbatim
            problems.append(Problem(source.filename, f"could not be read as a CSV ({exc})"))
            continue

        frame.columns = [str(c).strip() for c in frame.columns]
        missing = [c for c in source.columns if c not in frame.columns]
        extra = [c for c in frame.columns if c not in source.columns]

        if missing:
            problems.append(
                Problem(
                    source.filename,
                    f"expected column(s) not found: {', '.join(missing)}"
                    + (f" (the file has: {', '.join(frame.columns)})" if extra else ""),
                )
            )
            continue
        if extra:
            problems.append(
                Problem(
                    source.filename,
                    f"unexpected column(s) ignored: {', '.join(extra)}",
                    fatal=False,
                )
            )

        frames[source.table] = frame[list(source.columns)].copy()

    return frames, problems


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def _check_parsing(source: SourceFile, frame: pd.DataFrame) -> list[Problem]:
    problems: list[Problem] = []

    for column in source.date_columns:
        raw = frame[column]
        parsed = pd.to_datetime(raw, format=DATE_FORMAT, errors="coerce")
        bad = raw[parsed.isna() & raw.notna()]
        if len(bad):
            problems.append(
                Problem(
                    source.filename,
                    f"{len(bad)} value(s) in '{column}' are not a YYYY-MM-DD date "
                    f"(e.g. {_examples(bad)})",
                )
            )
        if raw.isna().any():
            problems.append(
                Problem(source.filename, f"{int(raw.isna().sum())} row(s) have no '{column}'")
            )

    for column in source.time_columns:
        raw = frame[column]
        parsed = pd.to_datetime(raw, format=TIME_FORMAT, errors="coerce")
        bad = raw[parsed.isna() & raw.notna()]
        if len(bad):
            problems.append(
                Problem(
                    source.filename,
                    f"{len(bad)} value(s) in '{column}' are not an HH:MM time "
                    f"(e.g. {_examples(bad)})",
                )
            )
        blank = int(raw.isna().sum())
        if blank and column in source.nullable_columns:
            problems.append(
                Problem(
                    source.filename,
                    f"{blank} row(s) have no '{column}' - kept, and treated as an "
                    "unresolved shift downstream",
                    fatal=False,
                )
            )
        elif blank:
            problems.append(
                Problem(source.filename, f"{blank} row(s) have no '{column}'")
            )

    for column in source.numeric_columns:
        raw = frame[column]
        parsed = pd.to_numeric(raw, errors="coerce")
        bad = raw[parsed.isna() & raw.notna()]
        if len(bad):
            problems.append(
                Problem(
                    source.filename,
                    f"{len(bad)} value(s) in '{column}' are not a number "
                    f"(e.g. {_examples(bad)})",
                )
            )

    already_reported = set(source.date_columns) | set(source.time_columns)
    for column in source.columns:
        if column in already_reported:
            continue
        blank = int(frame[column].isna().sum())
        if not blank:
            continue
        if column in source.nullable_columns:
            problems.append(
                Problem(
                    source.filename,
                    f"{blank} row(s) have a blank '{column}' - expected, and kept as blank",
                    fatal=False,
                )
            )
        else:
            problems.append(
                Problem(source.filename, f"{blank} row(s) have no '{column}'")
            )

    return problems


def _check_keys(source: SourceFile, frame: pd.DataFrame) -> list[Problem]:
    duplicated = frame.duplicated(subset=list(source.key), keep=False)
    if not duplicated.any():
        return []
    offenders = frame.loc[duplicated, list(source.key)].drop_duplicates()
    label = " + ".join(source.key)
    return [
        Problem(
            source.filename,
            f"{len(offenders)} duplicated {label} value(s) "
            f"(e.g. {_examples(offenders.astype(str).agg(' / '.join, axis=1))})",
        )
    ]


def _known_ids(conn: sqlite3.Connection, table: str, column: str) -> set[str]:
    return {
        row[0]
        for row in conn.execute(f"SELECT {column} FROM {table}")  # noqa: S608
        if row[0] is not None
    }


def _check_references(
    source: SourceFile,
    frame: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    conn: sqlite3.Connection,
) -> list[Problem]:
    """Check foreign keys against the export *and* what the database already holds,
    so an export containing only the new week still validates."""
    problems: list[Problem] = []

    for column, parent_table, parent_column in source.references:
        known = _known_ids(conn, parent_table, parent_column)
        if parent_table in frames:
            known |= set(frames[parent_table][parent_column].dropna())

        values = frame[column].dropna()
        unknown = sorted(set(values) - known)
        if unknown:
            problems.append(
                Problem(
                    source.filename,
                    f"{len(unknown)} '{column}' value(s) do not match any "
                    f"{parent_table}.{parent_column} we know about "
                    f"(e.g. {_examples(unknown)})",
                )
            )

    return problems


def validate(
    frames: dict[str, pd.DataFrame], conn: sqlite3.Connection
) -> list[Problem]:
    """Every check we can make before writing a single row."""
    problems: list[Problem] = []
    for source in SOURCE_FILES:
        frame = frames.get(source.table)
        if frame is None:
            continue
        if frame.empty:
            problems.append(Problem(source.filename, "the file has no rows", fatal=False))
            continue
        problems.extend(_check_parsing(source, frame))
        problems.extend(_check_keys(source, frame))
        problems.extend(_check_references(source, frame, frames, conn))
    return problems


# --------------------------------------------------------------------------- #
# Derived shift times
# --------------------------------------------------------------------------- #


def resolve_shift_times(shifts: pd.DataFrame) -> tuple[pd.DataFrame, list[Problem]]:
    """Turn shift_date + clock times into real start/end datetimes.

    1010 of the 8863 shifts in the current export clock out before they clock in,
    because they run through the night. Resolving that here - once - means nothing
    downstream has to remember to add 24 hours, which is the sort of thing that
    gets forgotten exactly once and then silently corrupts an answer.
    """
    problems: list[Problem] = []
    resolved = shifts.copy()

    day = pd.to_datetime(resolved["shift_date"], format=DATE_FORMAT, errors="coerce")
    start_time = pd.to_datetime(resolved["clock_in_time"], format=TIME_FORMAT, errors="coerce")
    end_time = pd.to_datetime(resolved["clock_out_time"], format=TIME_FORMAT, errors="coerce")

    start = day + (start_time - start_time.dt.normalize())
    end = day + (end_time - end_time.dt.normalize())

    # Clocking out at or before you clocked in means the shift crossed midnight.
    crossed_midnight = end.notna() & start.notna() & (end <= start)
    end = end.where(~crossed_midnight, end + timedelta(days=1))

    duration_hours = (end - start).dt.total_seconds() / 3600
    too_long = duration_hours > MAX_SHIFT_HOURS
    if too_long.any():
        problems.append(
            Problem(
                "shifts.csv",
                f"{int(too_long.sum())} shift(s) resolve to more than {MAX_SHIFT_HOURS:g} "
                f"hours, which is not a plausible overnight shift "
                f"(e.g. {_examples(resolved.loc[too_long, 'shift_id'])})",
            )
        )

    unresolved = end.isna() & resolved["clock_out_time"].notna()
    if unresolved.any():
        problems.append(
            Problem(
                "shifts.csv",
                f"{int(unresolved.sum())} shift(s) could not be resolved into a start and "
                "end datetime - check their shift_date and clock times "
                f"(e.g. {_examples(resolved.loc[unresolved, 'shift_id'])})",
            )
        )

    if crossed_midnight.any():
        problems.append(
            Problem(
                "shifts.csv",
                f"{int(crossed_midnight.sum())} shift(s) cross midnight and were rolled "
                "onto the following day",
                fatal=False,
            )
        )

    resolved["start_at"] = start.dt.strftime(STAMP_FORMAT)
    resolved["end_at"] = end.dt.strftime(STAMP_FORMAT)
    resolved["end_at"] = resolved["end_at"].astype(object).where(end.notna(), None)

    return resolved, problems


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #


def _coerce_for_sqlite(source: SourceFile, frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the only type conversions the database cares about."""
    out = frame.copy()
    for column in source.numeric_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def _upsert(conn: sqlite3.Connection, source: SourceFile, frame: pd.DataFrame) -> int:
    """Insert, replacing any row already held under the same key.

    A re-sent export therefore corrects what we hold rather than duplicating it,
    and loading the same file twice is a no-op.
    """
    columns = list(source.columns) + list(source.derived_columns)
    updatable = [c for c in columns if c not in source.key]

    statement = (
        f"INSERT INTO {source.table} ({', '.join(columns)}) "  # noqa: S608
        f"VALUES ({', '.join('?' * len(columns))}) "
        f"ON CONFLICT({', '.join(source.key)}) DO UPDATE SET "
        + ", ".join(f"{c} = excluded.{c}" for c in updatable)
    )

    payload = frame[columns].astype(object).where(pd.notna(frame[columns]), None)
    rows = list(payload.itertuples(index=False, name=None))
    conn.executemany(statement, rows)
    return len(rows)


def _write(
    conn: sqlite3.Connection, frames: dict[str, pd.DataFrame], source_dir: Path
) -> dict[str, int]:
    written: dict[str, int] = {}
    for source in SOURCE_FILES:
        frame = frames.get(source.table)
        if frame is None or frame.empty:
            continue
        written[source.table] = _upsert(conn, source, _coerce_for_sqlite(source, frame))
    return written


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def ingest(
    source_dir: Path | str = DEFAULT_DATA_DIR,
    db_path: Path | str = DEFAULT_DB_PATH,
    *,
    validate_only: bool = False,
) -> IngestReport:
    """Load an export into `db_path`, or check it without writing.

    Raises `IngestError` if the export cannot be loaded, in which case the
    database at `db_path` is left exactly as it was.
    """
    source_dir = Path(source_dir)
    report = IngestReport(source_dir=str(source_dir))

    if not source_dir.is_dir():
        report.problems.append(Problem(str(source_dir), "is not a directory"))
        raise IngestError(report)

    frames, problems = read_export(source_dir)
    report.problems.extend(problems)
    report.rows_read = {table: len(frame) for table, frame in frames.items()}

    if report.errors:
        raise IngestError(report)

    tables = [s.table for s in SOURCE_FILES]

    with staged_database(Path(db_path), commit=not validate_only) as staging:
        conn = connect(staging)
        try:
            init_schema(conn)

            report.problems.extend(validate(frames, conn))

            if "shifts" in frames:
                frames["shifts"], shift_problems = resolve_shift_times(frames["shifts"])
                report.problems.extend(shift_problems)

            if report.errors:
                raise IngestError(report)

            written = _write(conn, frames, source_dir)
            report.rows_in_db = table_counts(conn, tables)
            report.date_range = data_range(conn)

            conn.execute(
                "INSERT INTO ingest_log (ingested_at, source_dir, row_counts, rows_written,"
                " min_shift_date, max_shift_date) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    datetime.now().isoformat(timespec="seconds"),
                    str(source_dir.resolve()),
                    json.dumps(report.rows_in_db),
                    json.dumps(written),
                    report.date_range.get("min_shift_date"),
                    report.date_range.get("max_shift_date"),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    report.written = not validate_only
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="uv run python -m src.ingest",
        description="Load a weekly client export into the shifts database.",
    )
    parser.add_argument(
        "source_dir",
        nargs="?",
        default=DEFAULT_DATA_DIR,
        type=Path,
        help="directory holding the seven exported CSVs (default: data/)",
    )
    parser.add_argument(
        "--db", default=DEFAULT_DB_PATH, type=Path, help="database to load into"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="check the export and report, without writing anything",
    )
    args = parser.parse_args(argv)

    try:
        report = ingest(args.source_dir, args.db, validate_only=args.validate_only)
    except IngestError as exc:
        print(exc.report.render(), file=sys.stderr)
        return 1

    print(report.render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
