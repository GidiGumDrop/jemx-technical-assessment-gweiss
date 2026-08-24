"""Write note_classifications.csv — one row per supervisor note, with its category."""

from __future__ import annotations

from .config import DEFAULT_DB_PATH, PROJECT_ROOT
from .db import connect
from .notes import classify_all


def main() -> int:
    conn = connect(DEFAULT_DB_PATH)
    try:
        notes = classify_all(conn)
    finally:
        conn.close()

    out = PROJECT_ROOT / "note_classifications.csv"
    notes[["shift_id", "category", "note"]].to_csv(out, index=False)
    print(f"Wrote {out.name}: {len(notes):,} notes")
    print(notes.category.value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
