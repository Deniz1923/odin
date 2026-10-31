"""Private local gallery and event history. Raw enrollment photos are not retained."""

import json
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from cryptography.fernet import Fernet

from .lbp import SPACE


class Store:
    def __init__(self, directory: Path, retention_days: int = 7):
        if not 1 <= retention_days <= 365:
            raise ValueError("Event retention must be 1–365 days")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.directory.chmod(0o700)
        self.path = self.directory / "odin.sqlite3"
        self.retention_days = retention_days
        key = self.directory / "templates.key"
        if not key.exists():
            if self.path.exists():
                raise ValueError("Template key missing; restore it before opening this gallery")
            with os.fdopen(os.open(key, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as f:
                f.write(Fernet.generate_key())
        key.chmod(0o600)
        self.cipher = Fernet(key.read_bytes())
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS people (
                  id TEXT PRIMARY KEY, name TEXT NOT NULL, alert_enabled INTEGER NOT NULL,
                  authorized_at REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS templates (
                  person_id TEXT NOT NULL REFERENCES people(id) ON DELETE CASCADE,
                  space TEXT NOT NULL, payload BLOB NOT NULL);
                CREATE TABLE IF NOT EXISTS events (
                  id TEXT PRIMARY KEY,
                  person_id TEXT NOT NULL REFERENCES people(id) ON DELETE CASCADE,
                  source TEXT NOT NULL, session TEXT NOT NULL, track INTEGER NOT NULL,
                  distance REAL NOT NULL, observations INTEGER NOT NULL, policy TEXT NOT NULL,
                  created REAL NOT NULL, acknowledged INTEGER NOT NULL DEFAULT 0,
                  UNIQUE(session, track, person_id));
            """)
        self.path.chmod(0o600)
        self.sweep()

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def people(self) -> list[dict]:
        with self.connect() as db:
            return [
                dict(r)
                for r in db.execute("""
                SELECT p.*, COUNT(t.person_id) AS samples FROM people p
                LEFT JOIN templates t ON t.person_id=p.id GROUP BY p.id ORDER BY p.name
            """)
            ]

    def enroll(
        self, name: str, descriptors: list, *, authorized: bool, alert_enabled: bool = False
    ) -> dict:
        if not authorized or not name.strip() or len(name.strip()) > 80:
            raise ValueError("Authorization and a pseudonym of 1–80 characters are required")
        if not 1 <= len(descriptors) <= 32:
            raise ValueError("Enrollment needs 1–32 descriptors")
        encrypted = []
        for descriptor in descriptors:
            value = np.asarray(descriptor, dtype=np.float32)
            if (
                value.shape != (16384,)
                or not np.isfinite(value).all()
                or (value < 0).any()
                or value.sum() <= 0
            ):
                raise ValueError("Invalid LBP descriptor")
            encrypted.append(self.cipher.encrypt(value.astype("<f4").tobytes()))
        person = dict(id=uuid.uuid4().hex, name=name.strip(), alert_enabled=bool(alert_enabled))
        with self.connect() as db:
            db.execute(
                "INSERT INTO people VALUES (?,?,?,?)",
                (person["id"], person["name"], int(alert_enabled), time.time()),
            )
            db.executemany(
                "INSERT INTO templates VALUES (?,?,?)",
                [(person["id"], SPACE, payload) for payload in encrypted],
            )
        return person

    def gallery(self) -> dict[str, list[np.ndarray]]:
        gallery = {}
        with self.connect() as db:
            for row in db.execute("SELECT * FROM templates"):
                if row["space"] != SPACE:
                    raise ValueError("Incompatible gallery descriptor space")
                value = np.frombuffer(self.cipher.decrypt(row["payload"]), dtype="<f4").copy()
                if value.shape != (16384,) or not np.isfinite(value).all():
                    raise ValueError("Corrupt gallery descriptor")
                gallery.setdefault(row["person_id"], []).append(value)
        return gallery

    def delete(self, person_id: str):
        with self.connect() as db:
            if not db.execute("DELETE FROM people WHERE id=?", (person_id,)).rowcount:
                raise KeyError(person_id)
        # Logical deletion and compaction; not a physical SSD erasure guarantee.
        with self.connect() as db:
            db.execute("VACUUM")

    def set_alert(self, person_id: str, enabled: bool):
        with self.connect() as db:
            if not db.execute(
                "UPDATE people SET alert_enabled=? WHERE id=?", (int(enabled), person_id)
            ).rowcount:
                raise KeyError(person_id)

    def sweep(self, now: float | None = None):
        now = time.time() if now is None else now
        with self.connect() as db:
            db.execute("DELETE FROM events WHERE created<?", (now - self.retention_days * 86400,))

    def add_event(
        self,
        person_id,
        source,
        session,
        track,
        distance,
        observations,
        policy,
        *,
        now=None,
        cooldown=30,
    ):
        now = time.time() if now is None else now
        event_id = uuid.uuid4().hex
        with self.connect() as db:
            eligible = db.execute(
                "SELECT alert_enabled FROM people WHERE id=?", (person_id,)
            ).fetchone()
            if not eligible or not eligible[0]:
                return None
            last = db.execute(
                "SELECT MAX(created) FROM events WHERE person_id=? AND source=?",
                (person_id, source),
            ).fetchone()[0]
            if last is not None and now - last < cooldown:
                return None
            result = db.execute(
                "INSERT OR IGNORE INTO events VALUES (?,?,?,?,?,?,?,?,?,0)",
                (
                    event_id,
                    person_id,
                    source,
                    session,
                    track,
                    distance,
                    observations,
                    json.dumps(policy),
                    now,
                ),
            )
            return event_id if result.rowcount else None

    def events(self, *, now=None) -> list[dict]:
        self.sweep(now)
        with self.connect() as db:
            return [
                dict(r)
                for r in db.execute("""
                SELECT e.*,p.name FROM events e JOIN people p ON p.id=e.person_id
                ORDER BY created DESC LIMIT 200
            """)
            ]

    def acknowledge(self, event_id):
        with self.connect() as db:
            if not db.execute("UPDATE events SET acknowledged=1 WHERE id=?", (event_id,)).rowcount:
                raise KeyError(event_id)
