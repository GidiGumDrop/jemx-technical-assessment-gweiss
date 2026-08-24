"""The ops-room dashboard.

The person using this is a contract manager between sites, on a phone, with ten
minutes. So the API answers three questions and nothing else: who is at risk, why,
and what to do about it - plus a way to load next Monday's export without calling
a developer.

Scoring the week takes a few seconds, so it is computed once and cached. The
solver is deliberately *not* part of that: it runs when the manager asks for a
plan, because a plan is a decision rather than a page load.
"""

from __future__ import annotations

import io
import shutil
import tempfile
import threading
import zipfile
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response

from .actions import BANDS, REASON_LABEL, assess, make_plan
from .config import BREACH_HOURS, DEFAULT_DB_PATH, ORDINARY_HOURS, SOURCE_FILES
from .db import connect
from .ingest import IngestError, ingest

WEB_DIR = Path(__file__).resolve().parents[1] / "web"

app = FastAPI(title="Overtime ops room")

_lock = threading.Lock()
_state: dict = {}


def _load(force: bool = False) -> dict:
    """Score the week, or hand back the cached scoring."""
    with _lock:
        if _state and not force:
            return _state
        conn = connect(DEFAULT_DB_PATH)
        try:
            live, actions, meta = assess(conn)
        finally:
            conn.close()
        _state.clear()
        _state.update({"live": live, "actions": actions, "meta": meta, "plan": None})
        return _state


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (WEB_DIR / "index.html").read_text()


@app.get("/api/summary")
def summary() -> dict:
    state = _load()
    meta = state["meta"]
    return {
        "week_start": meta["current_week_start"],
        "days_known": meta["days_known"],
        "employees": meta["employees_scored"],
        "bands": meta["band_counts"],
        "needing_action": meta["needing_action"],
        "cost_at_risk": meta["cost_at_risk"],
        "overtime_cost_total": meta["overtime_cost_total"],
        "overtime_hours_total": meta["overtime_hours_total"],
        "quality": meta["quality"],
        "model": {
            "auc": round(meta["auc"], 3),
            "brier": round(meta["brier"], 4),
            "threshold": round(meta["threshold"], 3),
            "weeks_backtested": meta["weeks_evaluated"],
            "base_rate": round(meta["base_rate"], 4),
            "naive_f1": round(meta["naive"]["f1"], 3),
            "model_f1": round(meta["f1"], 3),
        },
        "planned": state["plan"] is not None,
        "caps": {"ordinary": ORDINARY_HOURS, "breach": BREACH_HOURS},
    }


@app.get("/api/risk")
def risk(site: str | None = None, band: str | None = None) -> dict:
    state = _load()
    rows = [a.as_dict() for a in state["actions"]]
    if site:
        rows = [r for r in rows if r["site_id"] == site]
    if band:
        rows = [r for r in rows if r["band"] == band]
    return {"count": len(rows), "bands": BANDS, "people": rows}


@app.get("/api/sites")
def sites() -> dict:
    state = _load()
    conn = connect(DEFAULT_DB_PATH)
    try:
        names = pd.read_sql("SELECT site_id, site_name, province FROM sites", conn)
    finally:
        conn.close()
    lookup = names.set_index("site_id").to_dict("index")

    out = []
    for row in state["meta"]["sites"]:
        site_id = row["site_id"]
        info = lookup.get(site_id, {})
        share = row.get("operational_share")
        cause_share = row.get("main_cause_share")
        out.append({
            "site_id": site_id,
            "site_name": info.get("site_name", site_id),
            "province": info.get("province"),
            "headcount": int(row.get("headcount") or 0),
            "over_cap": int(row.get("over_cap") or 0),
            "close": int(row.get("close") or 0),
            "cost_at_risk": float(row.get("cost_at_risk") or 0),
            "operational_hours": float(row.get("operational_hours") or 0),
            "client_hours": float(row.get("client_hours") or 0),
            "explained_hours": float(row.get("explained_hours") or 0),
            "operational_share": None if pd.isna(share) else round(float(share), 3),
            "main_cause": row.get("main_cause"),
            "main_cause_share": None if pd.isna(cause_share) else round(float(cause_share), 3),
        })
    return {"sites": sorted(out, key=lambda s: (-s["over_cap"], -s["cost_at_risk"]))}


@app.get("/api/sites/{site_id}")
def site_detail(site_id: str) -> dict:
    state = _load()
    live = state["live"]
    here = live[live.site_id == site_id]
    if here.empty:
        raise HTTPException(404, f"No site {site_id}")

    reasons = {}
    for action in state["actions"]:
        if action.site_id == site_id and action.reason:
            label = REASON_LABEL.get(action.reason, action.reason)
            reasons[label] = reasons.get(label, 0) + 1

    return {
        "site_id": site_id,
        "headcount": int(len(here)),
        "hours_this_week": round(float(here.hours_so_far.sum()), 1),
        "projected_hours": round(float(here.projected_hours.sum()), 1),
        "bands": here.band.value_counts().reindex(BANDS, fill_value=0).to_dict(),
        "cost_at_risk": round(float(here.cost_at_risk.sum()), 2),
        "causes": sorted(reasons.items(), key=lambda kv: -kv[1]),
        "people": [a.as_dict() for a in state["actions"] if a.site_id == site_id],
    }


@app.post("/api/plan")
def plan() -> dict:
    """Run the solver. This is the button - it is a decision, not a page load."""
    state = _load()
    conn = connect(DEFAULT_DB_PATH)
    try:
        result, actions = make_plan(conn, state["live"], state["actions"],
                                    int(state["meta"]["days_known"]))
    finally:
        conn.close()
    state["plan"] = result
    state["actions"] = actions

    covered = [a for a in actions if a.status == "covered"]
    blocked = [a for a in actions if a.status == "needs a decision"]
    live = state["live"]
    indexed = live.set_index("employee_id")
    names = indexed.full_name.to_dict()
    room = (ORDINARY_HOURS - indexed.projected_hours).to_dict()

    # Room left after each move, so the table can show what the receiver has spare.
    assignments, taken = [], {}
    for move in result.moves:
        taken[move.to_employee] = taken.get(move.to_employee, 0.0) + move.hours
        assignments.append({
            "from": move.from_employee, "from_name": names.get(move.from_employee),
            "to": move.to_employee, "to_name": names.get(move.to_employee),
            "day": move.day_name, "hours": round(move.hours, 2),
            "to_room": round(room.get(move.to_employee, 0.0) - taken[move.to_employee], 2),
        })
    return {
        "solver": result.solver,
        "moves": len(result.moves),
        "hours_moved": round(result.hours_moved, 1),
        "covered": len(covered),
        "blocked": len(blocked),
        "cost_avoided": round(sum(a.cost_at_risk for a in covered), 2),
        "assignments": assignments,
        "unresolved": [
            {"employee_id": a.employee_id, "name": a.name, "site_id": a.site_id,
             "role": a.role, "overshoot": round(a.overshoot, 2),
             "why": a.blocked_reason, "instruction": a.instruction} for a in blocked
        ],
    }


@app.post("/api/upload")
async def upload(files: list[UploadFile]) -> dict:
    """Load next week's export. Same seven CSVs, or a zip of them.

    Nothing is written until the whole export validates, so a bad upload leaves
    the running dashboard exactly as it was.
    """
    staging = Path(tempfile.mkdtemp(prefix="export-"))
    try:
        for item in files:
            payload = await item.read()
            name = Path(item.filename or "").name
            if name.lower().endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                    for member in archive.namelist():
                        inner = Path(member).name
                        if inner.lower().endswith(".csv"):
                            (staging / inner).write_bytes(archive.read(member))
            elif name.lower().endswith(".csv"):
                (staging / name).write_bytes(payload)

        expected = {s.filename for s in SOURCE_FILES}
        missing = sorted(expected - {p.name for p in staging.glob("*.csv")})
        if missing:
            raise HTTPException(400, {
                "message": "That export is missing files.",
                "problems": [f"{name}: not in the upload" for name in missing]})

        try:
            report = ingest(staging, DEFAULT_DB_PATH)
        except IngestError as exc:
            raise HTTPException(400, {
                "message": "That export could not be loaded. Nothing was changed.",
                "problems": [f"{p.source}: {p.message}" for p in exc.report.errors],
            }) from exc
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    _load(force=True)
    return {
        "message": "Loaded.",
        "rows": report.rows_read,
        "week_start": str(report.date_range.get("current_week_start")),
        "notes": [f"{p.source}: {p.message}" for p in report.notes],
    }


@app.get("/api/export/predictions.csv")
def export_predictions() -> Response:
    state = _load()
    live = state["live"]
    csv = live[["employee_id", "will_breach", "risk_score"]].sort_values(
        "employee_id").to_csv(index=False)
    return Response(csv, media_type="text/csv", headers={
        "Content-Disposition": "attachment; filename=predictions.csv"})


@app.get("/api/export/note_classifications.csv")
def export_notes() -> Response:
    from .notes import classify_all
    conn = connect(DEFAULT_DB_PATH)
    try:
        frame = classify_all(conn)
    finally:
        conn.close()
    csv = frame[["shift_id", "category", "note"]].to_csv(index=False)
    return Response(csv, media_type="text/csv", headers={
        "Content-Disposition": "attachment; filename=note_classifications.csv"})


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}
