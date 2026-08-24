"""What the rest of the week probably looks like for each employee.

The export records shifts that were *worked*, never shifts that were *scheduled* -
there is no forward roster anywhere in the data. So the days still to come have to
be inferred from how each person has actually worked in previous weeks. Everything
this module returns is a probability, not a plan, and the dashboard says so.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import numpy as np
import pandas as pd

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

#: Below this historical rate we do not claim the person is likely to work that day.
LIKELY_THRESHOLD = 0.35


@dataclass(frozen=True)
class ProbableShift:
    """A day the employee will probably work, and roughly how long for."""

    shift_ref: str
    employee_id: str
    weekday: int
    hours: float
    probability: float

    @property
    def day_name(self) -> str:
        return DAY_NAMES[self.weekday]

    @property
    def expected_hours(self) -> float:
        return self.hours * self.probability


def weekday_profile(conn: sqlite3.Connection) -> pd.DataFrame:
    """Per employee and weekday: how often they work it, and how long when they do."""
    shifts = pd.read_sql(
        "SELECT employee_id, shift_date, start_at, end_at FROM shifts",
        conn, parse_dates=["shift_date", "start_at", "end_at"],
    )
    shifts["hours"] = (shifts.end_at - shifts.start_at).dt.total_seconds() / 3600
    shifts["week"] = shifts.shift_date - pd.to_timedelta(shifts.shift_date.dt.weekday, unit="D")
    shifts["weekday"] = shifts.shift_date.dt.weekday

    current_week = shifts.week.max()
    past = shifts[shifts.week < current_week]
    weeks_observed = past.groupby("employee_id").week.nunique()

    worked = (past.groupby(["employee_id", "weekday"])
                  .agg(days_worked=("week", "nunique"), mean_hours=("hours", "mean"))
                  .reset_index())
    worked["weeks_observed"] = worked.employee_id.map(weeks_observed)
    worked["rate"] = worked.days_worked / worked.weeks_observed
    return worked


def probable_remaining(conn: sqlite3.Connection, days_known: int) -> list[ProbableShift]:
    """The shifts each employee will probably still work this week."""
    profile = weekday_profile(conn)
    remaining = profile[profile.weekday >= days_known]

    return [
        ProbableShift(
            shift_ref=f"{row.employee_id}_{DAY_NAMES[row.weekday][:3].lower()}",
            employee_id=row.employee_id,
            weekday=int(row.weekday),
            hours=float(row.mean_hours),
            probability=float(row.rate),
        )
        for row in remaining.itertuples()
        if row.rate > 0 and np.isfinite(row.mean_hours)
    ]


def already_working(conn: sqlite3.Connection) -> set[tuple[str, int]]:
    """(employee, weekday) pairs already clocked this week - nobody can be moved onto one."""
    shifts = pd.read_sql("SELECT employee_id, shift_date FROM shifts", conn,
                         parse_dates=["shift_date"])
    shifts["week"] = shifts.shift_date - pd.to_timedelta(shifts.shift_date.dt.weekday, unit="D")
    live = shifts[shifts.week == shifts.week.max()]
    return set(zip(live.employee_id, live.shift_date.dt.weekday))
