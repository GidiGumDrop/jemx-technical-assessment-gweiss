-- Raw client export, stored verbatim. The only added columns are shifts.start_at
-- and shifts.end_at, which resolve the overnight-shift ambiguity once at ingest
-- so no downstream consumer has to remember to add 24 hours.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS employees (
    employee_id             TEXT PRIMARY KEY,
    full_name               TEXT,
    id_number               TEXT,
    role                    TEXT,
    primary_site_id         TEXT,
    shift_pattern           TEXT,
    contract_ordinary_hours REAL,
    employment_type         TEXT
);

CREATE TABLE IF NOT EXISTS sites (
    site_id   TEXT PRIMARY KEY,
    site_name TEXT,
    province  TEXT
);

CREATE TABLE IF NOT EXISTS shifts (
    shift_id       TEXT PRIMARY KEY,
    employee_id    TEXT NOT NULL REFERENCES employees(employee_id),
    site_id        TEXT NOT NULL REFERENCES sites(site_id),
    shift_date     TEXT NOT NULL,   -- YYYY-MM-DD, as exported
    clock_in_time  TEXT NOT NULL,   -- HH:MM, as exported
    clock_out_time TEXT,            -- HH:MM, as exported; NULL when never clocked out
    start_at       TEXT NOT NULL,   -- derived: shift_date + clock_in_time
    end_at         TEXT             -- derived: rolled to the next day for overnight shifts
);

CREATE TABLE IF NOT EXISTS shift_notes (
    shift_id  TEXT PRIMARY KEY REFERENCES shifts(shift_id),
    logged_by TEXT,
    note      TEXT               -- NULL where the supervisor typed nothing
);

CREATE TABLE IF NOT EXISTS public_holidays (
    date TEXT PRIMARY KEY,
    name TEXT
);

-- The client's own weekly roll-up, kept as exported so our recomputation can be
-- reconciled against it rather than quietly replacing it.
CREATE TABLE IF NOT EXISTS weekly_summary (
    employee_id    TEXT NOT NULL REFERENCES employees(employee_id),
    week_starting  TEXT NOT NULL,
    total_hours    REAL,
    overtime_hours REAL,
    breached       INTEGER,
    PRIMARY KEY (employee_id, week_starting)
);

CREATE TABLE IF NOT EXISTS payroll_details (
    employee_id    TEXT PRIMARY KEY REFERENCES employees(employee_id),
    full_name      TEXT,
    id_number      TEXT,
    bank_name      TEXT,
    branch_code    TEXT,
    account_number TEXT,
    account_type   TEXT,
    tax_number     TEXT,
    hourly_rate    REAL,
    pay_frequency  TEXT
);

-- One row per load: what arrived, from where, and when.
CREATE TABLE IF NOT EXISTS ingest_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ingested_at   TEXT NOT NULL,
    source_dir    TEXT NOT NULL,
    row_counts    TEXT NOT NULL,   -- JSON: table -> rows now in the database
    rows_written  TEXT NOT NULL,   -- JSON: table -> rows in this export
    min_shift_date TEXT,
    max_shift_date TEXT
);

CREATE INDEX IF NOT EXISTS idx_shifts_employee_date ON shifts(employee_id, shift_date);
CREATE INDEX IF NOT EXISTS idx_shifts_site_date     ON shifts(site_id, shift_date);
CREATE INDEX IF NOT EXISTS idx_shifts_start_at      ON shifts(start_at);
CREATE INDEX IF NOT EXISTS idx_weekly_summary_week  ON weekly_summary(week_starting);

-- The reporting week is always derived from the data, never hardcoded, so next
-- Monday's export moves it forward on its own. SQLite's %w is 0=Sunday, so the
-- Monday of a week is the date shifted back by ((%w + 6) % 7) days.
DROP VIEW IF EXISTS v_data_range;
CREATE VIEW v_data_range AS
SELECT
    MIN(shift_date) AS min_shift_date,
    MAX(shift_date) AS max_shift_date,
    DATE(
        MIN(shift_date),
        '-' || ((CAST(STRFTIME('%w', MIN(shift_date)) AS INTEGER) + 6) % 7) || ' days'
    ) AS first_week_start,
    DATE(
        MAX(shift_date),
        '-' || ((CAST(STRFTIME('%w', MAX(shift_date)) AS INTEGER) + 6) % 7) || ' days'
    ) AS current_week_start,
    DATE(
        MAX(shift_date),
        '+' || (6 - ((CAST(STRFTIME('%w', MAX(shift_date)) AS INTEGER) + 6) % 7)) || ' days'
    ) AS current_week_end,
    CAST(STRFTIME('%w', MAX(shift_date)) AS INTEGER) AS max_shift_dow
FROM shifts;
