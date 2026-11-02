from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

from .models import canonical, digest


def now() -> str:
    return datetime.now(UTC).isoformat()


TERMINAL = frozenset({"succeeded", "failed", "timed_out", "cancelled", "lost", "invalid"})


class ResearchStore:
    """Persistent project/job ledger. Every state change and event share a transaction."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS research_projects (
                id TEXT PRIMARY KEY, definition TEXT NOT NULL, digest TEXT NOT NULL,
                created_at TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'ready');
            CREATE TABLE IF NOT EXISTS research_jobs (
                id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES research_projects(id),
                definition TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'queued',
                result TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS research_jobs_project ON research_jobs(project_id, status);
            CREATE TABLE IF NOT EXISTS research_events (
                seq INTEGER PRIMARY KEY AUTOINCREMENT, project_id TEXT NOT NULL,
                job_id TEXT NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL, created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS research_decisions (
                id TEXT PRIMARY KEY, project_id TEXT NOT NULL, payload TEXT NOT NULL, created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS research_cancellations (
                job_id TEXT PRIMARY KEY REFERENCES research_jobs(id), created_at TEXT NOT NULL);
        """)

    def close(self) -> None:
        self.db.close()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self.db.rollback()
            raise
        else:
            self.db.commit()

    def create_project(self, project_id: str, definition: dict[str, Any]) -> bool:
        with self.transaction():
            old = self.db.execute("SELECT digest FROM research_projects WHERE id=?", (project_id,)).fetchone()
            if old:
                if old["digest"] != digest(definition):
                    raise ValueError("project ID already exists with a different frozen definition")
                return False
            self.db.execute("INSERT INTO research_projects(id,definition,digest,created_at) VALUES(?,?,?,?)",
                            (project_id, canonical(definition), digest(definition), now()))
            self._event(project_id, "", "project_created", {"definition_digest": digest(definition)})
        return True

    def project(self, project_id: str) -> dict[str, Any]:
        row = self.db.execute("SELECT * FROM research_projects WHERE id=?", (project_id,)).fetchone()
        if row is None:
            raise KeyError(project_id)
        return {**dict(row), "definition": json.loads(row["definition"])}

    def enqueue(self, project_id: str, job: dict[str, Any]) -> bool:
        with self.transaction():
            previous = self.db.execute("SELECT definition FROM research_jobs WHERE id=?", (job["job_id"],)).fetchone()
            if previous:
                if previous["definition"] != canonical(job):
                    raise ValueError("job ID conflicts with frozen parameters")
                return False
            timestamp = now()
            self.db.execute("""INSERT INTO research_jobs
                (id,project_id,definition,created_at,updated_at) VALUES(?,?,?,?,?)""",
                            (job["job_id"], project_id, canonical(job), timestamp, timestamp))
            self._event(project_id, job["job_id"], "job_queued", {})
        return True

    def jobs(self, project_id: str) -> list[dict[str, Any]]:
        return [self._job(row) for row in self.db.execute(
            "SELECT * FROM research_jobs WHERE project_id=? ORDER BY rowid", (project_id,))]

    def job(self, job_id: str) -> dict[str, Any]:
        row = self.db.execute("SELECT * FROM research_jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return self._job(row)

    @staticmethod
    def _job(row: sqlite3.Row) -> dict[str, Any]:
        return {**dict(row), "definition": json.loads(row["definition"]),
                "result": json.loads(row["result"]) if row["result"] else None}

    def claim(self, job_id: str, max_concurrent: int, total_seconds: float) -> bool:
        with self.transaction():
            job = self.job(job_id)
            if job["status"] != "queued":
                return False
            jobs = self.jobs(job["project_id"])
            active = [j for j in jobs if j["status"] in {"starting", "running"}]
            if len(active) >= max_concurrent:
                return False
            used = sum((j["result"] or {}).get("elapsed_seconds", 0) for j in jobs)
            reserved = sum(j["definition"]["timeout_seconds"] for j in active)
            if used + reserved + job["definition"]["timeout_seconds"] > total_seconds:
                return False
            self.db.execute("UPDATE research_jobs SET status='starting',updated_at=? WHERE id=?", (now(), job_id))
            self._event(job["project_id"], job_id, "job_claimed", {})
        return True

    def transition(self, job_id: str, status: str, result: dict[str, Any] | None = None) -> bool:
        if status not in TERMINAL | {"running"}:
            raise ValueError("unsupported job transition")
        with self.transaction():
            job = self.job(job_id)
            if job["status"] in TERMINAL:
                if job["status"] != status or (result is not None and job["result"] != result):
                    raise ValueError("terminal job records are immutable")
                return False
            if job["status"] not in {"starting", "running"}:
                raise ValueError("a job must be claimed before execution")
            if job["status"] == status:
                return False
            if status in TERMINAL and result is None:
                raise ValueError("terminal transitions require an execution record")
            self.db.execute("UPDATE research_jobs SET status=?,result=?,updated_at=? WHERE id=?",
                            (status, canonical(result) if result is not None else None, now(), job_id))
            self._event(job["project_id"], job_id, f"job_{status}", result or {})
        return True

    def add_event(self, project_id: str, kind: str, payload: dict[str, Any], job_id: str = "") -> None:
        with self.transaction():
            self._event(project_id, job_id, kind, payload)

    def _event(self, project_id: str, job_id: str, kind: str, payload: dict[str, Any]) -> None:
        self.db.execute("INSERT INTO research_events(project_id,job_id,kind,payload,created_at) VALUES(?,?,?,?,?)",
                        (project_id, job_id, kind, canonical(payload), now()))

    def events(self, project_id: str) -> list[dict[str, Any]]:
        return [{**dict(row), "payload": json.loads(row["payload"])} for row in self.db.execute(
            "SELECT * FROM research_events WHERE project_id=? ORDER BY seq", (project_id,))]

    def record_decision(self, project_id: str, payload: dict[str, Any]) -> None:
        with self.transaction():
            self.db.execute("INSERT OR IGNORE INTO research_decisions VALUES(?,?,?,?)",
                            (digest({"project_id": project_id, "payload": payload}), project_id, canonical(payload), now()))

    def decisions(self, project_id: str) -> list[dict[str, Any]]:
        return [{**dict(row), "payload": json.loads(row["payload"])} for row in self.db.execute(
            "SELECT * FROM research_decisions WHERE project_id=? ORDER BY rowid", (project_id,))]

    def set_project_status(self, project_id: str, status: str) -> None:
        with self.transaction():
            self.db.execute("UPDATE research_projects SET status=? WHERE id=?", (status, project_id))

    def cancel_queued(self, job_id: str) -> None:
        with self.transaction():
            job = self.job(job_id)
            if job["status"] != "queued":
                raise ValueError("only queued jobs can be cancelled without a supervisor")
            result = {"status": "cancelled", "elapsed_seconds": 0, "exit_code": None,
                      "error": "cancelled before execution"}
            self.db.execute("UPDATE research_jobs SET status='cancelled',result=?,updated_at=? WHERE id=?",
                            (canonical(result), now(), job_id))
            self._event(job["project_id"], job_id, "job_cancelled", result)

    def request_cancel(self, job_id: str) -> None:
        with self.transaction():
            row = self.job(job_id)
            inserted = self.db.execute("INSERT OR IGNORE INTO research_cancellations VALUES(?,?)",
                                       (job_id, now())).rowcount
            if inserted:
                self._event(row["project_id"], job_id, "job_cancel_requested", {})

    def cancellation_requested(self, job_id: str) -> bool:
        return self.db.execute("SELECT 1 FROM research_cancellations WHERE job_id=?", (job_id,)).fetchone() is not None
