"""Turn the risk list into something a contract manager can act on before Sunday.

Three things per person: how far over they are heading, why the hours are
happening, and who should take the work instead. The last of those is solved
across the whole site at once rather than name by name - see reassign.py.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from .config import BREACH_HOURS, ORDINARY_HOURS
from .db import connect, data_range
from .model import predict_current_week
from .reassign import Candidate, Move, Plan, build_program, solve_greedy, solve_with_clingo
from .notes import OPERATIONAL as NOTE_OPERATIONAL
from .notes import classify_hybrid, train_naive_bayes
from .roster import DAY_NAMES, already_working, probable_remaining

# Overtime is paid at 1.5x. Sunday and public-holiday hours are paid at 2x, but we
# apply 1.5x throughout: working out which specific future hours land on a Sunday
# means guessing a roster we do not have. Every rand figure is therefore a floor,
# and the dashboard says so rather than implying a precision we cannot support.
OVERTIME_MULTIPLIER = 1.5
#: Only offer someone as cover if they have real room below the 45-hour line.
MIN_COVER_ROOM = 4.0

# Nobody is ever "definitely going to breach" - the week is not over and the model
# is right about a minority of the people it flags. So the list is banded by how
# close the projection sits to the cap, and the wording says at risk, not will.
BAND_OVER = "over the cap"        # projected past 55h
BAND_CLOSE = "close to the line"  # projected within 5h of the cap
BAND_WATCH = "worth watching"     # under that, but the model still flags them
BAND_CLEAR = "clear"
BANDS = [BAND_OVER, BAND_CLOSE, BAND_WATCH, BAND_CLEAR]

#: Hours below the cap that still counts as close.
CLOSE_MARGIN = 5.0

OPERATIONAL = sorted(NOTE_OPERATIONAL)

REASON_LABEL = {
    "relief_no_show": "relief not arriving",
    "absence_cover": "covering absent colleagues",
    "equipment_failure": "equipment failures",
    "late_handover": "late handovers",
    "client_requested": "work the client asked for",
    "client_requested_unapproved": "client asked, nobody signed it off",
    "no_information": "no reason recorded",
}


@dataclass
class Action:
    """One person, and what to do about them."""

    employee_id: str
    name: str
    site_id: str
    role: str
    risk_score: float
    hours_so_far: float
    projected_hours: float
    headroom: float               # hours before 55
    overshoot: float              # projected hours above 55
    reason: str | None
    cost_at_risk: float           # rand of overtime premium if nothing changes
    band: str = ""
    moves: list[Move] = field(default_factory=list)
    resolved: bool = False
    blocked_reason: str | None = None
    cover_room: float = 0.0
    #: Their biggest likely remaining shift, so advice can name a day rather than
    #: a number of hours floating free of any actual shift.
    biggest_day: str = ""
    biggest_hours: float = 0.0

    @property
    def headline(self) -> str:
        if self.overshoot <= 0:
            return (f"{self.name} is at {self.hours_so_far:.1f}h, on track for "
                    f"{self.projected_hours:.1f}h - {self.headroom:.1f}h of headroom left.")
        return (f"{self.name} is at {self.hours_so_far:.1f}h and tracking toward "
                f"{self.projected_hours:.1f}h - {self.overshoot:.1f}h past the cap.")

    @property
    def first_name(self) -> str:
        return self.name.split()[0] if self.name else self.employee_id

    def as_dict(self) -> dict:
        return {
            "employee_id": self.employee_id, "name": self.name, "site_id": self.site_id,
            "role": self.role, "band": self.band, "status": self.status,
            "risk_score": round(self.risk_score, 3),
            "hours_so_far": round(self.hours_so_far, 2),
            "projected_hours": round(self.projected_hours, 2),
            "headroom": round(self.headroom, 2), "overshoot": round(self.overshoot, 2),
            "reason": self.reason, "reason_label": REASON_LABEL.get(self.reason or ""),
            "cost_at_risk": round(self.cost_at_risk, 2),
            "headline": self.headline, "instruction": self.instruction,
            "moves": [{"day": m.day_name, "hours": round(m.hours, 2),
                       "to": m.to_employee} for m in self.moves],
        }

    @property
    def status(self) -> str:
        if self.overshoot <= 0.25:
            return "monitor"
        if self.moves:
            return "covered"
        return "needs a decision" if self.blocked_reason else "not planned yet"

    @property
    def instruction(self) -> str:
        """One short thing to do. No jargon, no hedging - a manager reads this in a car."""
        if self.moves:
            return "; ".join(
                f"Give {m.day_name}'s shift ({m.hours:.0f}h) to {m.to_name or m.to_employee}"
                for m in self.moves)
        if self.overshoot <= 0.25:
            return "Nothing to do today"
        if self.blocked_reason == "no_cover":
            return f"No one free at {self.site_id} - bring in a {self.role.lower()} from another site"
        if self.blocked_reason == "not_enough_room":
            if self.biggest_day:
                return f"Cut {self.overshoot:.1f}h off their {self.biggest_day} shift"
            return f"Cut {self.overshoot:.1f}h off their longest remaining shift"
        # Before a plan has been worked out: name the shift that would fix it.
        if self.biggest_day and self.biggest_hours >= self.overshoot:
            return f"Drop their {self.biggest_day} shift ({self.biggest_hours:.0f}h)"
        if self.biggest_day:
            return (f"Drop their {self.biggest_day} shift ({self.biggest_hours:.0f}h) "
                    f"and cut {max(self.overshoot - self.biggest_hours, 0):.1f}h more")
        return f"Cut {self.overshoot:.1f}h from their remaining shifts"


def _shift_reasons(conn: sqlite3.Connection) -> pd.DataFrame:
    """Tag every noted shift this week with why the hours ran long."""
    notes = pd.read_sql(
        "SELECT n.shift_id, n.note, s.employee_id, s.site_id, s.shift_date,"
        "       (julianday(s.end_at) - julianday(s.start_at)) * 24 AS hours "
        "FROM shift_notes n JOIN shifts s ON s.shift_id = n.shift_id",
        conn, parse_dates=["shift_date"])
    model = train_naive_bayes(conn)
    notes["category"] = [classify_hybrid(t, model) for t in notes.note.fillna("")]
    return notes


def _cover_pool(live: pd.DataFrame) -> list[Candidate]:
    room = ORDINARY_HOURS - live.projected_hours
    pool = live[(live.will_breach == 0) & (room > MIN_COVER_ROOM)]
    return [
        Candidate(employee_id=r.employee_id,
                  room_hours=float(ORDINARY_HOURS - r.projected_hours),
                  site_id=r.site_id, role=r.role, pattern=r.shift_pattern)
        for r in pool.itertuples()
    ]


def _band(projected: float, risk: float, threshold: float) -> str:
    if projected > BREACH_HOURS:
        return BAND_OVER
    if projected > BREACH_HOURS - CLOSE_MARGIN:
        return BAND_CLOSE
    return BAND_WATCH if risk >= threshold else BAND_CLEAR


def assess(conn: sqlite3.Connection) -> tuple[pd.DataFrame, list[Action], dict]:
    """Score everyone, band them, and attach cause and cost. No solver involved."""
    live, meta = predict_current_week(conn)
    people = pd.read_sql("SELECT employee_id, full_name, role, primary_site_id AS site_id,"
                         " shift_pattern FROM employees", conn)
    rates = pd.read_sql("SELECT employee_id, hourly_rate FROM payroll_details", conn)
    live = (live.drop(columns=[c for c in ("role", "site_id") if c in live.columns])
                .merge(people, on="employee_id").merge(rates, on="employee_id"))
    live["projected_hours"] = live.exp_total.fillna(live.hours_so_far)

    threshold = float(meta.get("threshold", 0.5))
    live["overshoot"] = (live.projected_hours - BREACH_HOURS).clip(lower=0)
    live["headroom"] = BREACH_HOURS - live.hours_so_far
    live["band"] = [
        _band(r.projected_hours, r.risk_score, threshold) for r in live.itertuples()
    ]
    # Overtime hours are everything above 45. What they cost is those hours at 1.5x.
    live["overtime_hours"] = (live.projected_hours - ORDINARY_HOURS).clip(lower=0)
    live["overtime_cost"] = live.overtime_hours * live.hourly_rate * OVERTIME_MULTIPLIER
    # The headline rand figure counts only people projected past the 55h cap, so the
    # money on screen is the money attached to the compliance problem - not the
    # ordinary, lawful overtime the client expects to pay anyway.
    live["cost_at_risk"] = live.overtime_cost.where(live.band == BAND_OVER, 0.0)

    reasons = _shift_reasons(conn)
    week_start = pd.Timestamp(str(meta["current_week_start"]))
    this_week = reasons[reasons.shift_date >= week_start]
    per_person = (this_week[this_week.category.isin(OPERATIONAL + ["client_requested"])]
                  .groupby("employee_id").category
                  .agg(lambda s: s.value_counts().idxmax()).to_dict())
    per_site = (this_week[this_week.category.isin(OPERATIONAL)]
                .groupby("site_id").category
                .agg(lambda s: s.value_counts().idxmax()).to_dict())

    # Biggest likely remaining shift per person, so advice can name a day.
    biggest: dict[str, tuple[str, float]] = {}
    for shift in probable_remaining(conn, int(meta["days_known"])):
        if shift.probability < 0.35:
            continue
        current = biggest.get(shift.employee_id)
        if current is None or shift.hours > current[1]:
            biggest[shift.employee_id] = (shift.day_name, shift.hours)

    candidates = _cover_pool(live)
    room_by_key: dict[tuple[str, str, str], float] = {}
    for c in candidates:
        key = (c.site_id, c.role, c.pattern)
        room_by_key[key] = room_by_key.get(key, 0.0) + c.room_hours

    listed = live[live.band != BAND_CLEAR].sort_values(
        ["projected_hours", "risk_score"], ascending=False)
    actions = [
        Action(employee_id=r.employee_id, name=r.full_name, site_id=r.site_id, role=r.role,
               risk_score=float(r.risk_score), hours_so_far=float(r.hours_so_far),
               projected_hours=float(r.projected_hours), headroom=float(r.headroom),
               overshoot=float(r.overshoot), band=r.band,
               reason=per_person.get(r.employee_id) or per_site.get(r.site_id),
               cost_at_risk=float(r.cost_at_risk),
               cover_room=room_by_key.get((r.site_id, r.role, r.shift_pattern), 0.0),
               biggest_day=biggest.get(r.employee_id, ("", 0.0))[0],
               biggest_hours=biggest.get(r.employee_id, ("", 0.0))[1])
        for r in listed.itertuples()
    ]

    meta.update({
        "threshold": threshold,
        "band_counts": live.band.value_counts().reindex(BANDS, fill_value=0).to_dict(),
        "needing_action": int((live.overshoot > 0.25).sum()),
        "cost_at_risk": round(float(live.cost_at_risk.sum()), 2),
        "overtime_cost_total": round(float(live.overtime_cost.sum()), 2),
        "overtime_hours_total": round(float(live.overtime_hours.sum()), 1),
        "sites": _site_summary(this_week, live).reset_index().to_dict("records"),
        "quality": _quality(conn),
    })
    return live, actions, meta


def make_plan(conn: sqlite3.Connection, live: pd.DataFrame, actions: list[Action],
              days_known: int) -> tuple[Plan, list[Action]]:
    """Run the solver. Kept separate so the dashboard loads without waiting on it."""
    people = {a.employee_id: a for a in actions}
    at_risk = {a.employee_id: a.overshoot for a in actions if a.overshoot > 0.25}

    shifts = probable_remaining(conn, days_known)
    busy = already_working(conn)
    profile = {r.employee_id: (r.site_id, r.role, r.shift_pattern) for r in live.itertuples()}
    candidates = _cover_pool(live)

    program = build_program(at_risk, shifts, candidates, busy, profile)
    plan = solve_with_clingo(program, at_risk, shifts)
    if plan is None:
        plan = solve_greedy(at_risk, shifts, candidates, busy, profile)

    names = live.set_index("employee_id").full_name.to_dict()
    plan.moves = [replace(m, to_name=names.get(m.to_employee, m.to_employee))
                  for m in plan.moves]

    by_person = plan.by_employee()
    for employee, need in at_risk.items():
        action = people[employee]
        action.moves = by_person.get(employee, [])
        action.resolved = employee not in plan.unresolved
        if not action.resolved:
            action.blocked_reason = "no_cover" if action.cover_room <= 0 else "not_enough_room"
    return plan, actions


def _quality(conn: sqlite3.Connection) -> dict:
    """Defects worth showing the manager rather than quietly absorbing."""
    overlaps = conn.execute(
        "SELECT COUNT(*) FROM shifts a JOIN shifts b"
        " ON a.employee_id = b.employee_id AND a.shift_date = b.shift_date"
        " AND a.shift_id < b.shift_id AND a.site_id != b.site_id").fetchone()[0]
    missing = conn.execute("SELECT COUNT(*) FROM shifts WHERE end_at IS NULL").fetchone()[0]
    return {"overlapping_shift_pairs": overlaps, "missing_clock_outs": missing}


def build_actions(conn: sqlite3.Connection) -> tuple[list[Action], Plan, dict]:
    """Assess and solve in one go - used by the CLI."""
    live, actions, meta = assess(conn)
    plan, actions = make_plan(conn, live, actions, int(meta["days_known"]))
    meta.update({"solver": plan.solver, "moves": len(plan.moves),
                 "hours_moved": round(plan.hours_moved, 2),
                 "unresolved": plan.unresolved,
                 "status_counts": pd.Series([a.status for a in actions]).value_counts().to_dict()})
    return actions, plan, {"meta": meta, "sites": pd.DataFrame(meta["sites"])}


def _site_summary(this_week: pd.DataFrame, live: pd.DataFrame) -> pd.DataFrame:
    """Per site: how many are over the cap, what it costs, and why it is happening.

    The two hour columns come from the supervisors' notes, not the clock data. Each
    note on a shift worked this week is sorted into a cause; the hours of the shifts
    carrying those notes are then totalled per site. So they describe *noted* hours -
    supervisors only write when something happened - rather than all hours worked.
    """
    hours = this_week.groupby(["site_id", "category"]).hours.sum().unstack(fill_value=0.0)
    for column in OPERATIONAL + ["client_requested", "no_information"]:
        if column not in hours:
            hours[column] = 0.0

    out = pd.DataFrame(index=pd.Index(sorted(live.site_id.unique()), name="site_id"))
    out["headcount"] = live.groupby("site_id").size()
    out["over_cap"] = live[live.band == BAND_OVER].groupby("site_id").size()
    out["close"] = live[live.band == BAND_CLOSE].groupby("site_id").size()
    out["cost_at_risk"] = live.groupby("site_id").cost_at_risk.sum().round(0)

    out["operational_hours"] = hours[OPERATIONAL].sum(axis=1).round(1)
    out["client_hours"] = hours["client_requested"].round(1)
    explained = out.operational_hours + out.client_hours
    out["explained_hours"] = explained.round(1)
    out["operational_share"] = (out.operational_hours / explained.replace(0, np.nan))

    top = (this_week[this_week.category.isin(OPERATIONAL)]
           .groupby("site_id").category.agg(lambda s: s.value_counts().idxmax()))
    out["main_cause"] = top.map(REASON_LABEL)
    out["main_cause_share"] = (this_week[this_week.category.isin(OPERATIONAL)]
                               .groupby("site_id").category
                               .agg(lambda s: s.value_counts(normalize=True).max()))
    return out.fillna({"over_cap": 0, "close": 0, "cost_at_risk": 0,
                       "operational_hours": 0, "client_hours": 0, "explained_hours": 0}
                      ).sort_values("over_cap", ascending=False)


def main() -> int:
    conn = connect("shifts.db")
    try:
        actions, plan, report = build_actions(conn)
    finally:
        conn.close()

    meta = report["meta"]
    print(f"Week starting {meta['current_week_start']}, {meta['days_known']} days known")
    print(f"{len(actions)} people at risk | R{meta['cost_at_risk']:,.0f} of overtime premium "
          f"at stake if nothing changes")
    print(f"{meta['needing_action']} are projected over the cap and need work taken off them")
    print(f"plan ({meta['solver']}): {meta['moves']} reassignments moving {meta['hours_moved']}h, "
          f"{len(meta['unresolved'])} could not be covered")
    print(f"status: {meta['status_counts']}\n")

    for action in actions:
        if action.status == "monitor":
            continue
        cause = f" Driver: {REASON_LABEL[action.reason]}." if action.reason else ""
        print(f"[{action.risk_score:.2f}] {action.headline}{cause}")
        print(f"       R{action.cost_at_risk:,.0f} at risk. {action.instruction}\n")

    print("Per site:")
    print(report["sites"].to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
