"""Turn the shift log into the table the risk model is trained on.

One row per (employee, week, day-of-week we can see up to). The data currently
stops on a Wednesday, but nothing here assumes that: an export ending on a
Tuesday or a Thursday produces the same features with a different `days_elapsed`.
Training across every cut point is also what makes the sample size workable -
it turns ~1,900 employee-weeks into ~11,000 rows.
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from .config import BREACH_HOURS

DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]

#: Everything the model is allowed to see. Ordered roughly most- to least-obvious.
FEATURES = [
    "hours_so_far",      # hours already worked this week
    "days_worked",       # days already worked this week
    "days_elapsed",      # days of the week we can see
    "days_left",         # days still to come
    "hours_to_breach",   # 55 - hours_so_far: the remaining budget
    "rate_so_far",       # hours per elapsed day
    "prior_total",       # their average week, from earlier weeks only
    "prior_days",        # their average days worked per week
    "prior_hpd",         # their average hours per worked day
    "prior_max",         # their heaviest week so far
    "exp_hours_left",    # hours the days still to come usually bring them
    "exp_total",         # hours_so_far + exp_hours_left
    "frac_of_usual",     # share of a usual week already done
    "ahead_of_usual",    # hours ahead of where they normally are by this day
    "is_night",
]


def daily_hours_matrix(conn: sqlite3.Connection) -> tuple[np.ndarray, pd.DataFrame]:
    """Hours per weekday, one row per employee-week. Shape (n, 7), Monday first."""
    shifts = pd.read_sql(
        "SELECT employee_id, site_id, shift_date, start_at, end_at FROM shifts",
        conn, parse_dates=["shift_date", "start_at", "end_at"],
    )
    shifts["hours"] = (shifts.end_at - shifts.start_at).dt.total_seconds() / 3600
    shifts["week"] = shifts.shift_date - pd.to_timedelta(shifts.shift_date.dt.weekday, unit="D")
    shifts["dow"] = shifts.shift_date.dt.weekday

    grid = (shifts.pivot_table(index=["employee_id", "week"], columns="dow",
                               values="hours", aggfunc="sum")
                  .reindex(columns=range(7)).fillna(0.0))
    return grid.to_numpy(), grid.index.to_frame(index=False)


def _employee_history(hours: np.ndarray, keys: pd.DataFrame) -> dict[str, np.ndarray]:
    """Per employee-week summaries of that employee's *earlier* weeks only.

    Everything is built by walking weeks in order and only ever looking backwards,
    so no row can see itself or anything that came after it.
    """
    n = len(keys)
    out = {name: np.full(n, np.nan) for name in
           ("prior_total", "prior_days", "prior_hpd", "prior_max")}
    dow_rate = np.full((n, 7), np.nan)     # how often they work each weekday
    dow_hours = np.full((n, 7), np.nan)    # how long, when they do

    order = np.lexsort((keys.week.to_numpy().astype("datetime64[ns]").astype("int64"),
                        keys.employee_id.to_numpy()))
    seen: dict[str, list[np.ndarray]] = {}
    for i in order:
        employee = keys.employee_id.to_numpy()[i]
        past = seen.get(employee)
        if past:
            weeks = np.vstack(past)
            totals, days = weeks.sum(1), (weeks > 0).sum(1)
            out["prior_total"][i] = totals.mean()
            out["prior_days"][i] = days.mean()
            out["prior_max"][i] = totals.max()
            out["prior_hpd"][i] = totals.sum() / max(days.sum(), 1)
            worked = (weeks > 0).sum(0)
            dow_rate[i] = worked / len(weeks)
            dow_hours[i] = np.divide(weeks.sum(0), worked, out=np.zeros(7), where=worked > 0)
        seen.setdefault(employee, []).append(hours[i])

    out["dow_rate"] = dow_rate
    out["dow_hours"] = dow_hours
    return out


def build_panel(conn: sqlite3.Connection) -> pd.DataFrame:
    """One row per (employee, week, cut day), with the label and every feature."""
    hours, keys = daily_hours_matrix(conn)
    history = _employee_history(hours, keys)
    employees = pd.read_sql("SELECT employee_id, shift_pattern, role, primary_site_id "
                            "FROM employees", conn).set_index("employee_id")
    is_night = (employees.shift_pattern.reindex(keys.employee_id).to_numpy() == "night").astype(int)

    week_total = hours.sum(1)
    rows = []
    for cut in range(1, 7):                      # we can see Monday..cut
        seen, rest = hours[:, :cut], hours[:, cut:]
        so_far = seen.sum(1)
        # What their own roster history says about the days still to come. Days
        # worked explains most of the variance in remaining hours, so this
        # carries more than the raw averages do.
        expected_left = (history["dow_rate"][:, cut:] * history["dow_hours"][:, cut:]).sum(1)
        usual_by_now = (history["dow_rate"][:, :cut] * history["dow_hours"][:, :cut]).sum(1)

        rows.append(pd.DataFrame({
            "employee_id": keys.employee_id.to_numpy(),
            "week": keys.week.to_numpy(),
            "cut": cut,
            "hours_so_far": so_far,
            "days_worked": (seen > 0).sum(1),
            "days_elapsed": float(cut),
            "days_left": float(7 - cut),
            "hours_to_breach": BREACH_HOURS - so_far,
            "rate_so_far": so_far / cut,
            "prior_total": history["prior_total"],
            "prior_days": history["prior_days"],
            "prior_hpd": history["prior_hpd"],
            "prior_max": history["prior_max"],
            "exp_hours_left": expected_left,
            "exp_total": so_far + expected_left,
            "frac_of_usual": so_far / np.where(history["prior_total"] > 0,
                                               history["prior_total"], np.nan),
            "ahead_of_usual": so_far - usual_by_now,
            "is_night": is_night,
            "hours_remaining": rest.sum(1),          # the truth, for diagnostics
            "week_total": week_total,
            "breached": (week_total > BREACH_HOURS).astype(int),
        }))

    panel = pd.concat(rows, ignore_index=True)
    panel["site_id"] = employees.primary_site_id.reindex(panel.employee_id).to_numpy()
    panel["role"] = employees.role.reindex(panel.employee_id).to_numpy()
    return panel


def naive_projection(panel: pd.DataFrame) -> pd.Series:
    """The obvious thing to do without a model.

    Take the hours per day they have averaged so far this week, assume the days
    still to come look the same, and see whether that clears 55.
    """
    return panel.hours_so_far + panel.rate_so_far * panel.days_left
