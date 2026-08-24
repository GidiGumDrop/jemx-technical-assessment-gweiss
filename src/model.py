"""Who will breach the 10-hour overtime cap by Sunday.

A breach is a week over 55 hours, and the hours already worked are known exactly.
So the model answers one question: given what this person has done so far this
week and how they normally work, what is the chance they finish over 55?

Logistic regression, trained across every day-of-week cut point so it does not
care which day the export happens to stop on, then isotonically calibrated so
the score it reports is a probability rather than just a ranking.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import BREACH_HOURS, DEFAULT_DB_PATH, PROJECT_ROOT
from .db import connect, data_range
from .features import FEATURES, build_panel, naive_projection

#: Weeks of history an employee needs before the model will score them.
MIN_HISTORY_WEEKS = 1
#: Weeks of training data before the first backtest fold.
MIN_TRAIN_WEEKS = 2


def _fit_calibrator(raw: np.ndarray, y: np.ndarray):
    """Map raw scores onto honest probabilities (Platt scaling).

    The model already emits something probability-shaped, but trained on all six
    cut points it is systematically off for any one of them. Fitting a one-
    parameter curve on out-of-fold scores corrects the level while leaving the
    ranking - and therefore every score distinct - intact.
    """
    logit = np.log(np.clip(raw, 1e-6, 1 - 1e-6) / (1 - np.clip(raw, 1e-6, 1 - 1e-6)))
    platt = LogisticRegression(max_iter=1000).fit(logit.reshape(-1, 1), y)
    return lambda r: platt.predict_proba(
        np.log(np.clip(r, 1e-6, 1 - 1e-6) / (1 - np.clip(r, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    )[:, 1]


def _new_model() -> Pipeline:
    # Few features, few positives, and a scored artefact that has to be a
    # probability: regularised logistic on standardised inputs.
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, C=0.5, class_weight=None)),
    ])


def _design(frame: pd.DataFrame, fill: pd.Series | None = None) -> tuple[np.ndarray, pd.Series]:
    """Feature matrix, with a fallback for employees we have never seen before."""
    X = frame[FEATURES].copy()
    fill = X.median(numeric_only=True) if fill is None else fill
    return X.fillna(fill).to_numpy(), fill


@dataclass
class Backtest:
    """Out-of-fold predictions from walking forward one week at a time."""

    frame: pd.DataFrame
    threshold: float = 0.5
    metrics: dict = field(default_factory=dict)

    def at(self, cut: int) -> pd.DataFrame:
        return self.frame[self.frame.cut == cut]


def _auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y)
    positives, negatives = y.sum(), len(y) - y.sum()
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = pd.Series(np.asarray(score, float)).rank().to_numpy()
    return float((ranks[y == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def _prf(y: np.ndarray, flag: np.ndarray) -> tuple[float, float, float]:
    tp = int(((flag == 1) & (y == 1)).sum())
    predicted, actual = int(flag.sum()), int(y.sum())
    precision = tp / predicted if predicted else 0.0
    recall = tp / actual if actual else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def backtest(panel: pd.DataFrame, current_week: pd.Timestamp) -> Backtest:
    """Walk forward: for each completed week, train only on the weeks before it."""
    history = panel[panel.week < current_week]
    weeks = np.sort(history.week.unique())

    folds = []
    for i, week in enumerate(weeks):
        if i < MIN_TRAIN_WEEKS:
            continue
        train = history[(history.week < week) & history.prior_total.notna()]
        test = history[history.week == week]
        if train.breached.sum() < 5 or test.empty:
            continue
        Xtr, fill = _design(train)
        Xte, _ = _design(test, fill)
        model = _new_model().fit(Xtr, train.breached.to_numpy())
        folds.append(test.assign(raw=model.predict_proba(Xte)[:, 1]))

    frame = pd.concat(folds, ignore_index=True)

    # Calibrate on out-of-fold scores only, so the mapping is never fitted to
    # predictions the model has already seen the answers for. Platt rather than
    # isotonic: with 55 positives isotonic collapses into a handful of plateaus,
    # which would leave a dozen people tied on the same score and unrankable.
    calibrator = _fit_calibrator(frame.raw.to_numpy(), frame.breached.to_numpy())
    frame["risk"] = calibrator(frame.raw.to_numpy())

    wednesday = frame[frame.cut == 3]
    y = wednesday.breached.to_numpy()

    # Threshold chosen on backtested F1, not on a round number.
    grid = np.unique(np.round(wednesday.risk, 4))
    scored = [(t, _prf(y, (wednesday.risk >= t).to_numpy().astype(int))[2]) for t in grid if t > 0]
    threshold = max(scored, key=lambda kv: kv[1])[0] if scored else 0.5

    precision, recall, f1 = _prf(y, (wednesday.risk >= threshold).to_numpy().astype(int))
    naive_flag = (naive_projection(wednesday) > BREACH_HOURS).to_numpy().astype(int)
    n_precision, n_recall, n_f1 = _prf(y, naive_flag)

    metrics = {
        "weeks_evaluated": int(wednesday.week.nunique()),
        "employee_weeks": int(len(wednesday)),
        "breaches": int(y.sum()),
        "base_rate": float(y.mean()),
        "auc": _auc(y, wednesday.risk.to_numpy()),
        "brier": float(np.mean((wednesday.risk - y) ** 2)),
        "mean_predicted_risk": float(wednesday.risk.mean()),
        "threshold": float(threshold),
        "precision": precision, "recall": recall, "f1": f1,
        "flagged": int(naive_flag.size and (wednesday.risk >= threshold).sum()),
        "naive": {"rule": "hours so far + (hours per elapsed day x days left) > 55",
                  "precision": n_precision, "recall": n_recall, "f1": n_f1,
                  "flagged": int(naive_flag.sum())},
        "by_days_known": {},
    }
    for cut in sorted(frame.cut.unique()):
        part = frame[frame.cut == cut]
        metrics["by_days_known"][int(cut)] = {
            "auc": _auc(part.breached.to_numpy(), part.risk.to_numpy()),
            "brier": float(np.mean((part.risk - part.breached) ** 2)),
        }
    return Backtest(frame=frame, threshold=float(threshold), metrics=metrics)


def predict_current_week(conn: sqlite3.Connection) -> tuple[pd.DataFrame, dict]:
    """Score every employee for the week still in progress."""
    panel = build_panel(conn)
    window = data_range(conn)
    current_week = pd.Timestamp(str(window["current_week_start"]))
    # How much of the week we can see, derived from the data rather than assumed.
    cut = int(pd.Timestamp(str(window["max_shift_date"])).weekday()) + 1

    result = backtest(panel, current_week)

    train = panel[(panel.week < current_week) & panel.prior_total.notna()]
    Xtr, fill = _design(train)
    model = _new_model().fit(Xtr, train.breached.to_numpy())
    calibrator = _fit_calibrator(result.frame.raw.to_numpy(),
                                 result.frame.breached.to_numpy())

    live = panel[(panel.week == current_week) & (panel.cut == cut)].copy()
    Xte, _ = _design(live, fill)
    live["raw"] = model.predict_proba(Xte)[:, 1]
    live["risk_score"] = calibrator(live.raw.to_numpy())
    live["will_breach"] = (live.risk_score >= result.threshold).astype(int)
    live["naive_projection"] = naive_projection(live)

    # Every employee on the register needs a row, including anyone who has not
    # worked this week - they simply cannot breach from a standing start.
    register = pd.read_sql("SELECT employee_id FROM employees", conn)
    live = register.merge(live, on="employee_id", how="left")
    live["risk_score"] = live.risk_score.fillna(0.0).clip(0.0, 1.0).round(4)
    live["will_breach"] = live.will_breach.fillna(0).astype(int)
    for column, default in (("hours_so_far", 0.0), ("days_worked", 0.0),
                            ("naive_projection", 0.0), ("exp_total", 0.0)):
        live[column] = live[column].fillna(default)
    live["hours_to_breach"] = BREACH_HOURS - live.hours_so_far

    meta = dict(result.metrics)
    meta.update({"current_week_start": str(current_week.date()),
                 "days_known": cut,
                 "employees_scored": int(len(live)),
                 "flagged_this_week": int(live.will_breach.sum())})
    return live.sort_values("risk_score", ascending=False), meta


def main() -> int:
    conn = connect(DEFAULT_DB_PATH)
    try:
        predictions, meta = predict_current_week(conn)
    finally:
        conn.close()

    (predictions[["employee_id", "will_breach", "risk_score"]]
        .sort_values("employee_id").to_csv(PROJECT_ROOT / "predictions.csv", index=False))
    (PROJECT_ROOT / "outputs").mkdir(exist_ok=True)
    (PROJECT_ROOT / "outputs" / "metrics.json").write_text(
        json.dumps(meta, indent=2, default=float))

    naive = meta["naive"]
    print(f"Week starting {meta['current_week_start']}, {meta['days_known']} days known\n")
    print(f"Backtest over {meta['weeks_evaluated']} weeks "
          f"({meta['employee_weeks']:,} employee-weeks, {meta['breaches']} breaches, "
          f"base rate {meta['base_rate']:.2%})")
    print(f"  AUC {meta['auc']:.3f}   Brier {meta['brier']:.4f}   "
          f"mean predicted risk {meta['mean_predicted_risk']:.4f}")
    print(f"  at threshold {meta['threshold']:.3f}: "
          f"precision {meta['precision']:.2%}  recall {meta['recall']:.2%}  F1 {meta['f1']:.3f}")
    print(f"  naive rule ({naive['rule']}):")
    print(f"    precision {naive['precision']:.2%}  recall {naive['recall']:.2%}  "
          f"F1 {naive['f1']:.3f}  flagged {naive['flagged']}")
    print("\n  AUC by how much of the week is known:")
    for days, m in meta["by_days_known"].items():
        print(f"    {days} day(s): AUC {m['auc']:.3f}")
    print(f"\nFlagged this week: {meta['flagged_this_week']} of {meta['employees_scored']}")
    print(predictions.head(12)[["employee_id", "hours_so_far", "days_worked",
                                "exp_total", "naive_projection", "risk_score",
                                "will_breach"]].round(2).to_string(index=False))
    print("\nWrote predictions.csv and outputs/metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
