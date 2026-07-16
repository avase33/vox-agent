"""A tiny SQLite-backed appointment book that the agent's tools operate on.

This is the "real world" the voice agent can *act* on: it can look up open slots
and actually create bookings that persist. Uses stdlib ``sqlite3`` only; an
in-memory database is the default so tests/demos need no files.
"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Optional

_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
# Default business hours (24h) offered to callers.
_DEFAULT_SLOTS = ["09:00", "10:00", "11:00", "13:00", "14:00", "15:00", "16:00"]


@dataclass
class Booking:
    id: int
    day: str
    time: str
    name: str
    reason: str


class CalendarDB:
    def __init__(self, path: str = ":memory:", seed: bool = True) -> None:
        # check_same_thread=False so the async server can share one connection.
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._create()
        if seed:
            self._seed()

    def _create(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS bookings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    day TEXT NOT NULL,
                    time TEXT NOT NULL,
                    name TEXT NOT NULL,
                    reason TEXT DEFAULT '',
                    created TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(day, time)
                )
                """
            )
            self._conn.commit()

    def _seed(self) -> None:
        # Pre-fill a couple of slots so "check availability" has something to
        # exclude and the demo feels real.
        tue = self._next_weekday("tuesday")
        self.book(tue, "10:00", "Existing Customer", "follow-up", ignore_errors=True)
        self.book(tue, "14:00", "Existing Customer", "consult", ignore_errors=True)

    # -- date helpers -----------------------------------------------------
    @staticmethod
    def _next_weekday(day_name: str, ref: Optional[date] = None) -> str:
        ref = ref or date.today()
        day_name = day_name.strip().lower()
        if day_name in ("today",):
            return ref.isoformat()
        if day_name in ("tomorrow",):
            return (ref + timedelta(days=1)).isoformat()
        if day_name not in _WEEKDAYS:
            return ref.isoformat()
        target = _WEEKDAYS.index(day_name)
        delta = (target - ref.weekday()) % 7
        delta = delta or 7  # always the *next* occurrence, not today
        return (ref + timedelta(days=delta)).isoformat()

    # -- tool operations --------------------------------------------------
    def check_availability(self, day: str, time: Optional[str] = None) -> dict:
        iso = self._next_weekday(day) if not _looks_iso(day) else day
        with self._lock:
            rows = self._conn.execute(
                "SELECT time FROM bookings WHERE day = ?", (iso,)
            ).fetchall()
        taken = {r["time"] for r in rows}
        free = [s for s in _DEFAULT_SLOTS if s not in taken]
        if time:
            norm = _normalise_time(time)
            return {
                "day": iso,
                "requested_time": norm,
                "available": norm in free,
                "open_slots": free,
            }
        return {"day": iso, "open_slots": free, "available": bool(free)}

    def book(
        self,
        day: str,
        time: str,
        name: str,
        reason: str = "",
        ignore_errors: bool = False,
    ) -> dict:
        iso = self._next_weekday(day) if not _looks_iso(day) else day
        norm = _normalise_time(time)
        try:
            with self._lock:
                cur = self._conn.execute(
                    "INSERT INTO bookings (day, time, name, reason) VALUES (?, ?, ?, ?)",
                    (iso, norm, name or "Caller", reason),
                )
                self._conn.commit()
                return {"ok": True, "booking_id": cur.lastrowid, "day": iso, "time": norm}
        except sqlite3.IntegrityError:
            if ignore_errors:
                return {"ok": False, "reason": "slot_taken", "day": iso, "time": norm}
            return {"ok": False, "reason": "slot_taken", "day": iso, "time": norm}

    def list_bookings(self) -> list[Booking]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, day, time, name, reason FROM bookings ORDER BY day, time"
            ).fetchall()
        return [Booking(r["id"], r["day"], r["time"], r["name"], r["reason"]) for r in rows]

    def close(self) -> None:
        self._conn.close()


def _looks_iso(s: str) -> bool:
    try:
        datetime.strptime(s, "%Y-%m-%d")
        return True
    except (ValueError, TypeError):
        return False


def _normalise_time(t: str) -> str:
    """Accept '3pm', '3 pm', '15:00', '3:30pm' -> 'HH:MM'."""
    t = t.strip().lower().replace(" ", "")
    ampm = None
    if t.endswith("am"):
        ampm, t = "am", t[:-2]
    elif t.endswith("pm"):
        ampm, t = "pm", t[:-2]
    if ":" in t:
        hh, mm = t.split(":", 1)
    else:
        hh, mm = t, "00"
    try:
        h = int(hh)
        m = int(mm)
    except ValueError:
        return t
    if ampm == "pm" and h < 12:
        h += 12
    if ampm == "am" and h == 12:
        h = 0
    return f"{h:02d}:{m:02d}"
