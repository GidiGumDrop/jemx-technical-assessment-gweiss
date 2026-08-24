"""Paths and the expected shape of a client export."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_DB_PATH = PROJECT_ROOT / "shifts.db"

SCHEMA_SQL = Path(__file__).with_name("schema.sql")


@dataclass(frozen=True)
class SourceFile:
    """One CSV in the client's weekly export."""

    filename: str
    table: str
    columns: tuple[str, ...]
    key: tuple[str, ...]
    #: Columns that must parse as a date (YYYY-MM-DD).
    date_columns: tuple[str, ...] = ()
    #: Columns that must parse as HH:MM. Blanks are tolerated.
    time_columns: tuple[str, ...] = ()
    #: Columns that must parse as a number.
    numeric_columns: tuple[str, ...] = ()
    #: (column, table, column) foreign keys checked against already-loaded frames.
    references: tuple[tuple[str, str, str], ...] = ()
    #: Columns where blanks are expected rather than a defect.
    nullable_columns: tuple[str, ...] = ()
    #: Extra columns added by ingestion, not present in the CSV.
    derived_columns: tuple[str, ...] = field(default=())


# Order matters: referential integrity is checked against tables already read.
SOURCE_FILES: tuple[SourceFile, ...] = (
    SourceFile(
        filename="employees.csv",
        table="employees",
        columns=(
            "employee_id",
            "full_name",
            "id_number",
            "role",
            "primary_site_id",
            "shift_pattern",
            "contract_ordinary_hours",
            "employment_type",
        ),
        key=("employee_id",),
        numeric_columns=("contract_ordinary_hours",),
    ),
    SourceFile(
        filename="sites.csv",
        table="sites",
        columns=("site_id", "site_name", "province"),
        key=("site_id",),
    ),
    SourceFile(
        filename="shifts.csv",
        table="shifts",
        columns=(
            "shift_id",
            "employee_id",
            "site_id",
            "shift_date",
            "clock_in_time",
            "clock_out_time",
        ),
        key=("shift_id",),
        date_columns=("shift_date",),
        time_columns=("clock_in_time", "clock_out_time"),
        references=(
            ("employee_id", "employees", "employee_id"),
            ("site_id", "sites", "site_id"),
        ),
        nullable_columns=("clock_out_time",),
        derived_columns=("start_at", "end_at"),
    ),
    SourceFile(
        filename="shift_notes.csv",
        table="shift_notes",
        columns=("shift_id", "logged_by", "note"),
        key=("shift_id",),
        references=(("shift_id", "shifts", "shift_id"),),
        nullable_columns=("note",),
    ),
    SourceFile(
        filename="public_holidays.csv",
        table="public_holidays",
        columns=("date", "name"),
        key=("date",),
        date_columns=("date",),
    ),
    SourceFile(
        filename="weekly_summary.csv",
        table="weekly_summary",
        columns=(
            "employee_id",
            "week_starting",
            "total_hours",
            "overtime_hours",
            "breached",
        ),
        key=("employee_id", "week_starting"),
        date_columns=("week_starting",),
        numeric_columns=("total_hours", "overtime_hours", "breached"),
        references=(("employee_id", "employees", "employee_id"),),
    ),
    SourceFile(
        filename="payroll_details.csv",
        table="payroll_details",
        columns=(
            "employee_id",
            "full_name",
            "id_number",
            "bank_name",
            "branch_code",
            "account_number",
            "account_type",
            "tax_number",
            "hourly_rate",
            "pay_frequency",
        ),
        key=("employee_id",),
        numeric_columns=("hourly_rate",),
        references=(("employee_id", "employees", "employee_id"),),
    ),
)

SOURCE_BY_TABLE = {source.table: source for source in SOURCE_FILES}

#: A resolved shift longer than this is treated as a defect, not an overnight shift.
MAX_SHIFT_HOURS = 16.0

# The BCEA caps a week at 45 ordinary hours plus 10 of overtime. The client's own
# weekly_summary.csv applies exactly this rule, so a breach is a week over 55 hours.
ORDINARY_HOURS = 45.0
OVERTIME_CAP = 10.0
BREACH_HOURS = ORDINARY_HOURS + OVERTIME_CAP
