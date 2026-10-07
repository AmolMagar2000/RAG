"""Mock eMigrant store (SQLite). Deterministic: the LLM only sees pre-computed facts, never raw SQL access.

Every query is parameterised and scoped by employer_id (tenant). In production, derive employer_id
from the authenticated session - never from the request body or the LLM.
"""
import sqlite3
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import get_settings

settings = get_settings()


def _conn() -> sqlite3.Connection:
    Path(settings.DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _d(days: int) -> str:
    return (date.today() + timedelta(days=days)).isoformat()


def init_db() -> None:
    with _conn() as c:
        c.execute("CREATE TABLE IF NOT EXISTS employers (id INTEGER PRIMARY KEY, name TEXT NOT NULL)")
        c.execute("""CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY AUTOINCREMENT, employer_id INTEGER NOT NULL, full_name TEXT NOT NULL,
            job_title TEXT, soc_code TEXT, salary_gbp INTEGER, hours_per_week INTEGER, work_location TEXT,
            visa_type TEXT, visa_expiry TEXT, rtw_check_date TEXT, rtw_followup_due TEXT)""")
        if c.execute("SELECT COUNT(*) FROM employers").fetchone()[0]:
            return
        c.executemany("INSERT INTO employers VALUES (?, ?)", [(1, "Northbridge Care Ltd"), (2, "Harlow Engineering Ltd")])
        c.executemany(
            "INSERT INTO employees (employer_id, full_name, job_title, soc_code, salary_gbp, hours_per_week, "
            "work_location, visa_type, visa_expiry, rtw_check_date, rtw_followup_due) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            [
                (1, "Priya Sharma", "Senior Care Worker", "6136", 33400, 40, "Leeds - Head Office", "Skilled Worker", _d(412), _d(-200), _d(212)),
                (1, "Priya Shah", "Care Worker", "6135", 30500, 37, "Leeds - Head Office", "Skilled Worker", _d(25), _d(-340), _d(25)),
                (1, "Mohammed Khan", "Care Team Leader", "6136", 34800, 40, "Bradford", "Skilled Worker", _d(300), _d(-370), _d(-9)),
                (1, "Olga Petrova", "Registered Nurse", "2231", 38200, 37, "Leeds - Head Office", "Health and Care Worker", _d(800), _d(-120), _d(240)),
                (1, "Tomasz Nowak", "Care Worker", "6135", 30500, 40, "Bradford", "Skilled Worker", _d(58), _d(-300), _d(58)),
                (1, "Aisha Bello", "Care Worker", "6135", 30500, 40, "Leeds - Head Office", "Skilled Worker", _d(600), _d(-60), _d(300)),
                (2, "Daniel Okafor", "Software Developer", "2136", 52000, 37, "Harlow", "Skilled Worker", _d(500), _d(-90), _d(410)),
                (2, "Mei Lin Chen", "Systems Engineer", "2137", 49000, 40, "Harlow", "Skilled Worker", _d(40), _d(-320), _d(40)),
                (2, "Priya Sharma", "Test Engineer", "2139", 46000, 40, "Harlow", "Skilled Worker", _d(90), _d(-275), _d(90)),
            ],
        )


def list_employers() -> List[Dict[str, Any]]:
    with _conn() as c:
        return [dict(r) for r in c.execute("SELECT id, name FROM employers ORDER BY id")]


def list_employees(employer_id: int) -> List[Dict[str, Any]]:
    with _conn() as c:
        rows = c.execute("SELECT id, full_name FROM employees WHERE employer_id = ? ORDER BY id", (employer_id,))
        return [{"id": r["id"], "name": r["full_name"]} for r in rows]


def _facts(row: sqlite3.Row, today: date) -> Dict[str, Any]:
    days_visa = (date.fromisoformat(row["visa_expiry"]) - today).days
    days_rtw = (date.fromisoformat(row["rtw_followup_due"]) - today).days
    return {
        "job_title": row["job_title"], "soc_code": row["soc_code"], "salary_gbp": row["salary_gbp"],
        "hours_per_week": row["hours_per_week"], "work_location": row["work_location"], "visa_type": row["visa_type"],
        "visa_expiry": row["visa_expiry"], "days_to_visa_expiry": days_visa,
        "visa_status": "expired" if days_visa < 0 else "expiring_within_90_days" if days_visa <= 90 else "valid",
        "last_rtw_check": row["rtw_check_date"], "rtw_followup_due": row["rtw_followup_due"],
        "days_to_rtw_followup": days_rtw,
        "rtw_followup_status": "overdue" if days_rtw < 0 else "due_within_30_days" if days_rtw <= 30 else "ok",
        "as_of": today.isoformat(),
    }


def get_employee_facts(employer_id: int, employee_id: int, today: Optional[date] = None) -> Optional[Dict[str, Any]]:
    with _conn() as c:
        row = c.execute("SELECT * FROM employees WHERE employer_id = ? AND id = ?", (employer_id, employee_id)).fetchone()
    return _facts(row, today or date.today()) if row else None


def get_workforce_alerts(employer_id: int, today: Optional[date] = None, limit: int = 20) -> List[Dict[str, Any]]:
    """Employees needing attention (visa expiring/expired, RTW follow-up due/overdue). Includes name+facts."""
    today = today or date.today()
    with _conn() as c:
        rows = c.execute("SELECT * FROM employees WHERE employer_id = ? ORDER BY id", (employer_id,)).fetchall()
    out = []
    for r in rows:
        f = _facts(r, today)
        if f["visa_status"] != "valid" or f["rtw_followup_status"] != "ok":
            out.append({"employee_id": r["id"], "name": r["full_name"], "facts": f})
    return out[:limit]
