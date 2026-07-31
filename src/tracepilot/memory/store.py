from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

from .fingerprints import (
    canonical_json,
    content_fingerprint,
    source_fingerprint,
    to_jsonable,
    workspace_fingerprint,
)
from .records import (
    CheckpointEntry,
    DecisionMemoryEntry,
    EntityEntry,
    RunHistoryEntry,
    TaskRecovery,
    WorkingMemoryEntry,
    model_from_payload,
)


SCHEMA_VERSION = 2
DECISION_STATUSES = frozenset({"active", "disabled", "stale", "superseded", "deleted"})


class MemoryStoreError(RuntimeError):
    pass


class IdempotencyConflictError(MemoryStoreError):
    pass


class ImmutableRecordConflictError(MemoryStoreError):
    pass


class RecordNotFoundError(MemoryStoreError):
    pass


class WorkspaceScopeError(MemoryStoreError):
    pass


class MemoryStateConflictError(MemoryStoreError):
    pass


class MemoryStore:
    """Workspace-scoped SQLite persistence for TracePilot runtime state."""

    schema_version = SCHEMA_VERSION

    def __init__(
        self,
        path: str | Path,
        *,
        timeout: float = 5.0,
        clock: Callable[[], str] | None = None,
    ) -> None:
        self.path = Path(path) if str(path) != ":memory:" else Path(":memory:")
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._clock = clock or _utc_now
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            str(self.path),
            timeout=timeout,
            isolation_level=None,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        with self._lock:
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute("PRAGMA synchronous = NORMAL")
            self._connection.execute(f"PRAGMA busy_timeout = {max(1, int(timeout * 1000))}")
            self._journal_mode = str(self._connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]).lower()
            self._initialize_schema()

    def __enter__(self) -> MemoryStore:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    @property
    def journal_mode(self) -> str:
        return self._journal_mode

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    # Working memory -----------------------------------------------------

    def put_working_memory(
        self,
        workspace_id: str,
        key: str,
        value: Any,
        *,
        task_id: str = "",
        source: Any = None,
        workspace_state: Any = None,
        idempotency_key: str | None = None,
    ) -> WorkingMemoryEntry:
        workspace_id = _required(workspace_id, "workspace_id")
        key = _required(key, "key")
        task_id = str(task_id or "")
        payload = to_jsonable(value)
        if isinstance(payload, Mapping):
            _validate_payload_workspace(payload, workspace_id)
        content_fp = content_fingerprint(payload)
        source_fp = _source_fp(payload, source)
        workspace_fp = _workspace_fp(
            payload if isinstance(payload, Mapping) else {},
            workspace_id,
            workspace_state,
        )
        request = {
            "task_id": task_id,
            "key": key,
            "payload": payload,
            "source_fingerprint": source_fp,
            "workspace_fingerprint": workspace_fp,
        }
        with self._transaction() as connection:
            replay = self._idempotency_replay(
                connection, workspace_id, "working_memory.put", idempotency_key, request
            )
            if replay:
                row = self._select_working(connection, workspace_id, task_id, key)
                if row is None:
                    raise MemoryStoreError("idempotent working-memory resource no longer exists")
                return _working_from_row(row)
            now = self._clock()
            existing = self._select_working(connection, workspace_id, task_id, key)
            if existing is not None and _same_fingerprints(existing, content_fp, source_fp, workspace_fp):
                entry = _working_from_row(existing)
            else:
                created_at = existing["created_at"] if existing is not None else now
                connection.execute(
                    """
                    INSERT INTO working_memory (
                        workspace_id, task_id, memory_key, payload_json,
                        content_fingerprint, source_fingerprint, workspace_fingerprint,
                        schema_version, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(workspace_id, task_id, memory_key) DO UPDATE SET
                        payload_json = excluded.payload_json,
                        content_fingerprint = excluded.content_fingerprint,
                        source_fingerprint = excluded.source_fingerprint,
                        workspace_fingerprint = excluded.workspace_fingerprint,
                        schema_version = excluded.schema_version,
                        updated_at = excluded.updated_at
                    """,
                    (
                        workspace_id,
                        task_id,
                        key,
                        _dump(payload),
                        content_fp,
                        source_fp,
                        workspace_fp,
                        SCHEMA_VERSION,
                        created_at,
                        now,
                    ),
                )
                entry = _working_from_row(
                    self._select_working(connection, workspace_id, task_id, key, required=True)
                )
            self._remember_idempotency(
                connection,
                workspace_id,
                "working_memory.put",
                idempotency_key,
                request,
                f"{task_id}\0{key}",
            )
            return entry

    set_working_memory = put_working_memory

    def get_working_memory_entry(
        self,
        workspace_id: str,
        key: str,
        *,
        task_id: str = "",
    ) -> WorkingMemoryEntry | None:
        with self._lock:
            row = self._select_working(self._connection, workspace_id, str(task_id or ""), key)
        return _working_from_row(row) if row is not None else None

    def get_working_memory(
        self,
        workspace_id: str,
        key: str,
        *,
        task_id: str = "",
        model_type: type[Any] | None = None,
    ) -> Any:
        entry = self.get_working_memory_entry(workspace_id, key, task_id=task_id)
        return None if entry is None else entry.to_model(model_type)

    def list_working_memory(
        self,
        workspace_id: str,
        *,
        task_id: str | None = None,
    ) -> tuple[WorkingMemoryEntry, ...]:
        query = "SELECT * FROM working_memory WHERE workspace_id = ?"
        parameters: list[Any] = [workspace_id]
        if task_id is not None:
            query += " AND task_id = ?"
            parameters.append(task_id)
        query += " ORDER BY task_id, memory_key"
        with self._lock:
            rows = self._connection.execute(query, parameters).fetchall()
        return tuple(_working_from_row(row) for row in rows)

    def delete_working_memory(
        self,
        workspace_id: str,
        key: str,
        *,
        task_id: str = "",
        idempotency_key: str | None = None,
    ) -> bool:
        workspace_id = _required(workspace_id, "workspace_id")
        key = _required(key, "key")
        task_id = str(task_id or "")
        request = {"task_id": task_id, "key": key}
        with self._transaction() as connection:
            replay = self._idempotency_replay(
                connection,
                workspace_id,
                "working_memory.delete",
                idempotency_key,
                request,
            )
            if replay is not None:
                return replay == "deleted"
            cursor = connection.execute(
                "DELETE FROM working_memory WHERE workspace_id = ? AND task_id = ? AND memory_key = ?",
                (workspace_id, task_id, key),
            )
            deleted = cursor.rowcount > 0
            self._remember_idempotency(
                connection,
                workspace_id,
                "working_memory.delete",
                idempotency_key,
                request,
                "deleted" if deleted else "missing",
            )
            return deleted

    # Checkpoints --------------------------------------------------------

    def save_checkpoint(
        self,
        workspace_id: str,
        checkpoint: Any,
        *,
        task_id: str | None = None,
        checkpoint_id: str | None = None,
        run_id: str | None = None,
        step_id: str | None = None,
        source: Any = None,
        workspace_state: Any = None,
        idempotency_key: str | None = None,
    ) -> CheckpointEntry:
        workspace_id = _required(workspace_id, "workspace_id")
        payload = _payload(checkpoint)
        state_payload = payload.get("run") if isinstance(payload.get("run"), dict) else payload
        _validate_payload_workspace(state_payload, workspace_id)
        nested_request = (
            state_payload.get("request")
            if isinstance(state_payload.get("request"), dict)
            else {}
        )
        declared_workspace = _text(nested_request, "workspace_id")
        if declared_workspace and declared_workspace != workspace_id:
            raise WorkspaceScopeError(
                f"checkpoint workspace_id {declared_workspace!r} does not match scope {workspace_id!r}"
            )
        task_id = _required(task_id or _text(state_payload, "task_id"), "task_id")
        requested_id = checkpoint_id or _text(payload, "checkpoint_id")
        run_id = str(run_id or _text(state_payload, "run_id"))
        step_id = str(step_id or _text(state_payload, "step_id"))
        content_fp = content_fingerprint(payload)
        source_fp = _source_fp(state_payload, source)
        workspace_fp = _workspace_fp(payload, workspace_id, workspace_state)
        request = {
            "checkpoint_id": requested_id,
            "task_id": task_id,
            "run_id": run_id,
            "step_id": step_id,
            "payload": payload,
            "source_fingerprint": source_fp,
            "workspace_fingerprint": workspace_fp,
        }
        with self._transaction() as connection:
            replay_id = self._idempotency_replay(
                connection, workspace_id, "checkpoint.save", idempotency_key, request
            )
            if replay_id:
                return _checkpoint_from_row(
                    self._select_checkpoint(connection, workspace_id, replay_id, required=True)
                )
            checkpoint_id = requested_id or f"cp_{uuid.uuid4().hex}"
            existing = self._select_checkpoint(connection, workspace_id, checkpoint_id)
            if existing is not None:
                if not _same_fingerprints(existing, content_fp, source_fp, workspace_fp) or any(
                    (
                        existing["task_id"] != task_id,
                        existing["run_id"] != run_id,
                        existing["step_id"] != step_id,
                    )
                ):
                    raise ImmutableRecordConflictError(
                        f"checkpoint {checkpoint_id!r} already exists with different content"
                    )
                entry = _checkpoint_from_row(existing)
            else:
                now = self._clock()
                connection.execute(
                    """
                    INSERT INTO checkpoints (
                        workspace_id, checkpoint_id, task_id, run_id, step_id,
                        payload_json, content_fingerprint, source_fingerprint,
                        workspace_fingerprint, schema_version, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        workspace_id,
                        checkpoint_id,
                        task_id,
                        run_id,
                        step_id,
                        _dump(payload),
                        content_fp,
                        source_fp,
                        workspace_fp,
                        SCHEMA_VERSION,
                        now,
                    ),
                )
                entry = _checkpoint_from_row(
                    self._select_checkpoint(connection, workspace_id, checkpoint_id, required=True)
                )
            self._remember_idempotency(
                connection,
                workspace_id,
                "checkpoint.save",
                idempotency_key,
                request,
                checkpoint_id,
            )
            return entry

    def get_checkpoint(self, workspace_id: str, checkpoint_id: str) -> CheckpointEntry | None:
        with self._lock:
            row = self._select_checkpoint(self._connection, workspace_id, checkpoint_id)
        return _checkpoint_from_row(row) if row is not None else None

    def load_checkpoint(
        self,
        workspace_id: str,
        checkpoint_id: str,
        *,
        model_type: type[Any] | None = None,
    ) -> Any:
        entry = self.get_checkpoint(workspace_id, checkpoint_id)
        return None if entry is None else entry.to_model(model_type)

    def latest_checkpoint(self, workspace_id: str, task_id: str) -> CheckpointEntry | None:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT * FROM checkpoints
                WHERE workspace_id = ? AND task_id = ?
                ORDER BY sequence DESC LIMIT 1
                """,
                (workspace_id, task_id),
            ).fetchone()
        return _checkpoint_from_row(row) if row is not None else None

    get_latest_checkpoint = latest_checkpoint

    def load_latest_checkpoint(
        self,
        workspace_id: str,
        task_id: str,
        *,
        model_type: type[Any] | None = None,
    ) -> Any:
        entry = self.latest_checkpoint(workspace_id, task_id)
        return None if entry is None else entry.to_model(model_type)

    restore_checkpoint = load_latest_checkpoint

    # Immutable run history ---------------------------------------------

    def append_run(
        self,
        workspace_id: str,
        run: Any,
        *,
        run_id: str | None = None,
        task_id: str | None = None,
        status: str | None = None,
        source: Any = None,
        workspace_state: Any = None,
        idempotency_key: str | None = None,
    ) -> RunHistoryEntry:
        workspace_id = _required(workspace_id, "workspace_id")
        payload = _payload(run)
        _validate_payload_workspace(payload, workspace_id)
        requested_id = run_id or _text(payload, "run_id")
        task_id = str(task_id or _text(payload, "task_id"))
        status = str(status or _text(payload, "status") or "completed")
        content_fp = content_fingerprint(payload)
        source_fp = _source_fp(payload, source)
        workspace_fp = _workspace_fp(payload, workspace_id, workspace_state)
        request = {
            "run_id": requested_id,
            "task_id": task_id,
            "status": status,
            "payload": payload,
            "source_fingerprint": source_fp,
            "workspace_fingerprint": workspace_fp,
        }
        with self._transaction() as connection:
            replay_id = self._idempotency_replay(
                connection, workspace_id, "run.append", idempotency_key, request
            )
            if replay_id:
                return _run_from_row(self._select_run(connection, workspace_id, replay_id, required=True))
            run_id = requested_id or f"run_{uuid.uuid4().hex}"
            existing = self._select_run(connection, workspace_id, run_id)
            if existing is not None:
                if (
                    not _same_fingerprints(existing, content_fp, source_fp, workspace_fp)
                    or existing["task_id"] != task_id
                    or existing["status"] != status
                ):
                    raise ImmutableRecordConflictError(
                        f"run {run_id!r} is immutable and already has different content"
                    )
                entry = _run_from_row(existing)
            else:
                now = self._clock()
                connection.execute(
                    """
                    INSERT INTO run_history (
                        workspace_id, run_id, task_id, status, payload_json,
                        content_fingerprint, source_fingerprint, workspace_fingerprint,
                        schema_version, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        workspace_id,
                        run_id,
                        task_id,
                        status,
                        _dump(payload),
                        content_fp,
                        source_fp,
                        workspace_fp,
                        SCHEMA_VERSION,
                        now,
                    ),
                )
                entry = _run_from_row(self._select_run(connection, workspace_id, run_id, required=True))
            self._remember_idempotency(
                connection, workspace_id, "run.append", idempotency_key, request, run_id
            )
            return entry

    record_run = append_run
    save_run = append_run
    save_run_history = append_run

    def get_run(self, workspace_id: str, run_id: str) -> RunHistoryEntry | None:
        with self._lock:
            row = self._select_run(self._connection, workspace_id, run_id)
        return _run_from_row(row) if row is not None else None

    get_run_history = get_run

    def load_run(
        self,
        workspace_id: str,
        run_id: str,
        *,
        model_type: type[Any] | None = None,
    ) -> Any:
        entry = self.get_run(workspace_id, run_id)
        return None if entry is None else entry.to_model(model_type)

    def list_run_history(
        self,
        workspace_id: str,
        *,
        task_id: str | None = None,
        status: str | None = None,
        limit: int | None = None,
    ) -> tuple[RunHistoryEntry, ...]:
        query = "SELECT * FROM run_history WHERE workspace_id = ?"
        parameters: list[Any] = [workspace_id]
        if task_id is not None:
            query += " AND task_id = ?"
            parameters.append(task_id)
        if status is not None:
            query += " AND status = ?"
            parameters.append(status)
        query += " ORDER BY sequence DESC"
        if limit is not None:
            if limit < 1:
                return ()
            query += " LIMIT ?"
            parameters.append(limit)
        with self._lock:
            rows = self._connection.execute(query, parameters).fetchall()
        return tuple(_run_from_row(row) for row in rows)

    list_runs = list_run_history

    # Decision memory ----------------------------------------------------

    def promote_decision_memory(
        self,
        workspace_id: str,
        record: Any,
        *,
        memory_id: str | None = None,
        source: Any = None,
        workspace_state: Any = None,
        idempotency_key: str | None = None,
    ) -> DecisionMemoryEntry:
        workspace_id = _required(workspace_id, "workspace_id")
        payload = _payload(record)
        _validate_payload_workspace(payload, workspace_id)
        requested_id = memory_id or _text(payload, "memory_id")
        prepared = self._prepare_decision(
            workspace_id,
            payload,
            memory_id=requested_id,
            source=source,
            workspace_state=workspace_state,
            supersedes_id=_text(payload, "supersedes_id"),
        )
        request = {"memory_id": requested_id, **prepared}
        with self._transaction() as connection:
            replay_id = self._idempotency_replay(
                connection, workspace_id, "decision.promote", idempotency_key, request
            )
            if replay_id:
                return _decision_from_row(
                    self._select_decision(connection, workspace_id, replay_id, required=True)
                )
            memory_id = requested_id
            if not memory_id:
                duplicate = connection.execute(
                    """
                    SELECT * FROM decision_memories
                    WHERE workspace_id = ? AND kind = ? AND content_fingerprint = ?
                      AND source_fingerprint = ? AND workspace_fingerprint = ?
                      AND status = 'active'
                    ORDER BY sequence DESC LIMIT 1
                    """,
                    (
                        workspace_id,
                        prepared["kind"],
                        prepared["content_fingerprint"],
                        prepared["source_fingerprint"],
                        prepared["workspace_fingerprint"],
                    ),
                ).fetchone()
                if duplicate is not None:
                    entry = _decision_from_row(duplicate)
                    self._remember_idempotency(
                        connection,
                        workspace_id,
                        "decision.promote",
                        idempotency_key,
                        request,
                        entry.memory_id,
                    )
                    return entry
                memory_id = f"mem_{uuid.uuid4().hex}"
            existing = self._select_decision(connection, workspace_id, memory_id)
            if existing is not None:
                if (
                    existing["content_fingerprint"] != prepared["content_fingerprint"]
                    or existing["source_fingerprint"] != prepared["source_fingerprint"]
                    or existing["workspace_fingerprint"] != prepared["workspace_fingerprint"]
                    or existing["kind"] != prepared["kind"]
                    or existing["source_run_id"] != prepared["source_run_id"]
                    or existing["supersedes_id"] != prepared["supersedes_id"]
                ):
                    raise ImmutableRecordConflictError(
                        f"decision memory {memory_id!r} already exists with different content"
                    )
                entry = _decision_from_row(existing)
            else:
                entry = self._insert_decision(connection, workspace_id, memory_id, prepared)
            self._remember_idempotency(
                connection,
                workspace_id,
                "decision.promote",
                idempotency_key,
                request,
                memory_id,
            )
            return entry

    promote = promote_decision_memory

    def get_decision_memory(
        self,
        workspace_id: str,
        memory_id: str,
        *,
        include_deleted: bool = False,
    ) -> DecisionMemoryEntry | None:
        with self._lock:
            row = self._select_decision(self._connection, workspace_id, memory_id)
        if row is None or (row["status"] == "deleted" and not include_deleted):
            return None
        return _decision_from_row(row)

    get_memory = get_decision_memory

    def load_decision_memory(
        self,
        workspace_id: str,
        memory_id: str,
        *,
        model_type: type[Any] | None = None,
        include_deleted: bool = False,
    ) -> Any:
        entry = self.get_decision_memory(workspace_id, memory_id, include_deleted=include_deleted)
        return None if entry is None else entry.to_model(model_type)

    def list_decision_memories(
        self,
        workspace_id: str,
        *,
        status: str | None = "active",
        kind: str | None = None,
        include_deleted: bool = False,
        limit: int | None = None,
    ) -> tuple[DecisionMemoryEntry, ...]:
        query = "SELECT * FROM decision_memories WHERE workspace_id = ?"
        parameters: list[Any] = [workspace_id]
        if status is not None:
            query += " AND status = ?"
            parameters.append(status)
        elif not include_deleted:
            query += " AND status != 'deleted'"
        if kind is not None:
            query += " AND kind = ?"
            parameters.append(kind)
        query += " ORDER BY sequence DESC"
        if limit is not None:
            if limit < 1:
                return ()
            query += " LIMIT ?"
            parameters.append(limit)
        with self._lock:
            rows = self._connection.execute(query, parameters).fetchall()
        return tuple(_decision_from_row(row) for row in rows)

    list_memories = list_decision_memories

    def disable_decision_memory(
        self,
        workspace_id: str,
        memory_id: str,
        *,
        idempotency_key: str | None = None,
    ) -> DecisionMemoryEntry:
        return self._transition_decision(
            workspace_id,
            memory_id,
            operation="decision.disable",
            target_status="disabled",
            idempotency_key=idempotency_key,
        )

    disable_memory = disable_decision_memory

    def delete_decision_memory(
        self,
        workspace_id: str,
        memory_id: str,
        *,
        idempotency_key: str | None = None,
    ) -> DecisionMemoryEntry:
        return self._transition_decision(
            workspace_id,
            memory_id,
            operation="decision.delete",
            target_status="deleted",
            idempotency_key=idempotency_key,
        )

    delete_memory = delete_decision_memory

    def mark_decision_memory_stale(
        self,
        workspace_id: str,
        memory_id: str,
        *,
        reason: str = "",
        idempotency_key: str | None = None,
    ) -> DecisionMemoryEntry:
        return self._transition_decision(
            workspace_id,
            memory_id,
            operation="decision.stale",
            target_status="stale",
            idempotency_key=idempotency_key,
            stale_reason=reason,
        )

    mark_stale = mark_decision_memory_stale

    def supersede_decision_memory(
        self,
        workspace_id: str,
        memory_id: str,
        replacement: Any,
        *,
        replacement_id: str | None = None,
        source: Any = None,
        workspace_state: Any = None,
        idempotency_key: str | None = None,
    ) -> DecisionMemoryEntry:
        workspace_id = _required(workspace_id, "workspace_id")
        payload = _payload(replacement)
        _validate_payload_workspace(payload, workspace_id)
        with self._lock:
            previous = self._select_decision(
                self._connection,
                workspace_id,
                memory_id,
            )
        if previous is not None:
            supplied_workspace_fp = _text(payload, "workspace_fingerprint")
            supplied_source_fp = _text(payload, "source_fingerprint")
            if previous["status"] == "stale" and (
                (workspace_state is None and not supplied_workspace_fp)
                or (source is None and not supplied_source_fp)
            ):
                raise MemoryStateConflictError(
                    "superseding stale memory requires current source and "
                    "workspace state, normally from a completed task"
                )
            if workspace_state is None and not supplied_workspace_fp:
                payload["workspace_fingerprint"] = previous["workspace_fingerprint"]
            if source is None and not supplied_source_fp:
                payload["source_fingerprint"] = previous["source_fingerprint"]
            if not _text(payload, "source_run_id"):
                payload["source_run_id"] = previous["source_run_id"]
        requested_id = replacement_id or _text(payload, "memory_id")
        declared_supersedes = _text(payload, "supersedes_id")
        if declared_supersedes and declared_supersedes != memory_id:
            raise MemoryStateConflictError(
                f"replacement supersedes {declared_supersedes!r}, not {memory_id!r}"
            )
        prepared = self._prepare_decision(
            workspace_id,
            payload,
            memory_id=requested_id,
            source=source,
            workspace_state=workspace_state,
            supersedes_id=memory_id,
        )
        request = {
            "old_memory_id": memory_id,
            "replacement_id": requested_id,
            **prepared,
        }
        with self._transaction() as connection:
            replay_id = self._idempotency_replay(
                connection, workspace_id, "decision.supersede", idempotency_key, request
            )
            if replay_id:
                return _decision_from_row(
                    self._select_decision(connection, workspace_id, replay_id, required=True)
                )
            old = self._select_decision(connection, workspace_id, memory_id)
            if old is None or old["status"] == "deleted":
                raise RecordNotFoundError(f"decision memory {memory_id!r} was not found")
            if old["status"] == "superseded":
                existing_replacement = str(old["superseded_by_id"] or "")
                if requested_id and requested_id != existing_replacement:
                    raise MemoryStateConflictError(
                        f"decision memory {memory_id!r} is already superseded "
                        f"by {existing_replacement!r}"
                    )
                if existing_replacement:
                    replacement_row = self._select_decision(
                        connection, workspace_id, existing_replacement, required=True
                    )
                    if _decision_matches_prepared(replacement_row, prepared):
                        return _decision_from_row(replacement_row)
                raise MemoryStateConflictError(f"decision memory {memory_id!r} is already superseded")
            new_id = requested_id or f"mem_{uuid.uuid4().hex}"
            existing = self._select_decision(connection, workspace_id, new_id)
            if existing is not None:
                if not _decision_matches_prepared(existing, prepared):
                    raise ImmutableRecordConflictError(
                        f"replacement decision memory {new_id!r} has different content"
                    )
                replacement_entry = _decision_from_row(existing)
            else:
                replacement_entry = self._insert_decision(
                    connection, workspace_id, new_id, prepared
                )
            now = self._clock()
            connection.execute(
                """
                UPDATE decision_memories
                SET status = 'superseded', superseded_by_id = ?, updated_at = ?
                WHERE workspace_id = ? AND memory_id = ?
                """,
                (new_id, now, workspace_id, memory_id),
            )
            self._remember_idempotency(
                connection,
                workspace_id,
                "decision.supersede",
                idempotency_key,
                request,
                new_id,
            )
            return replacement_entry

    supersede_memory = supersede_decision_memory

    def is_decision_memory_stale(
        self,
        workspace_id: str,
        memory_id: str,
        *,
        current_content_fingerprint: str | None = None,
        current_source_fingerprint: str | None = None,
        current_workspace_fingerprint: str | None = None,
    ) -> bool:
        entry = self.get_decision_memory(workspace_id, memory_id, include_deleted=True)
        if entry is None:
            raise RecordNotFoundError(f"decision memory {memory_id!r} was not found")
        if entry.status == "stale":
            return True
        comparisons = (
            (current_content_fingerprint, entry.content_fingerprint),
            (current_source_fingerprint, entry.source_fingerprint),
            (current_workspace_fingerprint, entry.workspace_fingerprint),
        )
        return any(current is not None and current != stored for current, stored in comparisons)

    memory_is_stale = is_decision_memory_stale

    def list_stale_decision_memories(
        self,
        workspace_id: str,
        *,
        current_source_fingerprint: str | None = None,
        current_workspace_fingerprint: str | None = None,
    ) -> tuple[DecisionMemoryEntry, ...]:
        candidates = self.list_decision_memories(
            workspace_id, status=None, include_deleted=False
        )
        return tuple(
            entry
            for entry in candidates
            if entry.status == "stale"
            or (
                current_source_fingerprint is not None
                and entry.source_fingerprint != current_source_fingerprint
            )
            or (
                current_workspace_fingerprint is not None
                and entry.workspace_fingerprint != current_workspace_fingerprint
            )
        )

    list_stale_memories = list_stale_decision_memories

    # Task, step, approval, event persistence ---------------------------

    def save_task(
        self,
        workspace_id: str,
        task: Any,
        *,
        task_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> EntityEntry:
        return self._save_mutable_entity(
            table="tasks",
            operation="task.save",
            workspace_id=workspace_id,
            value=task,
            entity_id=task_id,
            entity_field="task_id",
            idempotency_key=idempotency_key,
        )

    put_task = save_task

    def get_task(self, workspace_id: str, task_id: str) -> EntityEntry | None:
        return self._get_entity("tasks", "task_id", workspace_id, task_id)

    def load_task(
        self,
        workspace_id: str,
        task_id: str,
        *,
        model_type: type[Any] | None = None,
    ) -> Any:
        entry = self.get_task(workspace_id, task_id)
        return None if entry is None else entry.to_model(model_type)

    def list_tasks(
        self, workspace_id: str, *, status: str | None = None
    ) -> tuple[EntityEntry, ...]:
        return self._list_entities("tasks", "task_id", workspace_id, status=status)

    def save_step(
        self,
        workspace_id: str,
        task_id: str,
        step: Any,
        *,
        step_id: str | None = None,
        ordinal: int | None = None,
        idempotency_key: str | None = None,
    ) -> EntityEntry:
        return self._save_mutable_entity(
            table="steps",
            operation="step.save",
            workspace_id=workspace_id,
            value=step,
            entity_id=step_id,
            entity_field="step_id",
            task_id=task_id,
            ordinal=ordinal,
            idempotency_key=idempotency_key,
        )

    put_step = save_step

    def get_step(self, workspace_id: str, step_id: str) -> EntityEntry | None:
        return self._get_entity("steps", "step_id", workspace_id, step_id)

    def load_step(
        self,
        workspace_id: str,
        step_id: str,
        *,
        model_type: type[Any] | None = None,
    ) -> Any:
        entry = self.get_step(workspace_id, step_id)
        return None if entry is None else entry.to_model(model_type)

    def list_steps(
        self,
        workspace_id: str,
        *,
        task_id: str | None = None,
        status: str | None = None,
    ) -> tuple[EntityEntry, ...]:
        return self._list_entities(
            "steps", "step_id", workspace_id, task_id=task_id, status=status
        )

    def save_approval(
        self,
        workspace_id: str,
        approval: Any,
        *,
        approval_id: str | None = None,
        task_id: str | None = None,
        step_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> EntityEntry:
        return self._save_mutable_entity(
            table="approvals",
            operation="approval.save",
            workspace_id=workspace_id,
            value=approval,
            entity_id=approval_id,
            entity_field="approval_id",
            task_id=task_id,
            step_id=step_id,
            idempotency_key=idempotency_key,
        )

    put_approval = save_approval

    def get_approval(self, workspace_id: str, approval_id: str) -> EntityEntry | None:
        return self._get_entity("approvals", "approval_id", workspace_id, approval_id)

    def load_approval(
        self,
        workspace_id: str,
        approval_id: str,
        *,
        model_type: type[Any] | None = None,
    ) -> Any:
        entry = self.get_approval(workspace_id, approval_id)
        return None if entry is None else entry.to_model(model_type)

    def list_approvals(
        self,
        workspace_id: str,
        *,
        task_id: str | None = None,
        status: str | None = None,
    ) -> tuple[EntityEntry, ...]:
        return self._list_entities(
            "approvals", "approval_id", workspace_id, task_id=task_id, status=status
        )

    def append_event(
        self,
        workspace_id: str,
        event: Any,
        *,
        event_id: str | None = None,
        task_id: str | None = None,
        step_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> EntityEntry:
        workspace_id = _required(workspace_id, "workspace_id")
        payload = _payload(event)
        _validate_payload_workspace(payload, workspace_id)
        requested_id = event_id or _text(payload, "event_id")
        task_id = str(task_id or _text(payload, "task_id"))
        step_id = str(step_id or _text(payload, "step_id"))
        kind = _text(payload, "kind") or _text(payload, "event_type")
        timestamp = _text(payload, "timestamp") or self._clock()
        content_fp = content_fingerprint(payload)
        workspace_fp = _workspace_fp(payload, workspace_id, None)
        request = {
            "event_id": requested_id,
            "task_id": task_id,
            "step_id": step_id,
            "kind": kind,
            "timestamp": timestamp,
            "payload": payload,
        }
        with self._transaction() as connection:
            replay_id = self._idempotency_replay(
                connection, workspace_id, "event.append", idempotency_key, request
            )
            if replay_id:
                return _entity_from_row(
                    self._select_entity(
                        connection, "events", "event_id", workspace_id, replay_id, required=True
                    ),
                    "event_id",
                )
            event_id = requested_id or f"evt_{uuid.uuid4().hex}"
            existing = self._select_entity(
                connection, "events", "event_id", workspace_id, event_id
            )
            if existing is not None:
                if (
                    existing["content_fingerprint"] != content_fp
                    or existing["task_id"] != task_id
                    or existing["step_id"] != step_id
                    or existing["kind"] != kind
                    or existing["timestamp"] != timestamp
                ):
                    raise ImmutableRecordConflictError(
                        f"event {event_id!r} is immutable and has different content"
                    )
                entry = _entity_from_row(existing, "event_id")
            else:
                now = self._clock()
                connection.execute(
                    """
                    INSERT INTO events (
                        workspace_id, event_id, task_id, step_id, kind, timestamp,
                        payload_json, content_fingerprint, workspace_fingerprint,
                        schema_version, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        workspace_id,
                        event_id,
                        task_id,
                        step_id,
                        kind,
                        timestamp,
                        _dump(payload),
                        content_fp,
                        workspace_fp,
                        SCHEMA_VERSION,
                        now,
                    ),
                )
                entry = _entity_from_row(
                    self._select_entity(
                        connection, "events", "event_id", workspace_id, event_id, required=True
                    ),
                    "event_id",
                )
            self._remember_idempotency(
                connection, workspace_id, "event.append", idempotency_key, request, event_id
            )
            return entry

    record_event = append_event
    save_trace_event = append_event

    def get_event(self, workspace_id: str, event_id: str) -> EntityEntry | None:
        return self._get_entity("events", "event_id", workspace_id, event_id)

    def load_event(
        self,
        workspace_id: str,
        event_id: str,
        *,
        model_type: type[Any] | None = None,
    ) -> Any:
        entry = self.get_event(workspace_id, event_id)
        return None if entry is None else entry.to_model(model_type)

    def list_events(
        self,
        workspace_id: str,
        *,
        task_id: str | None = None,
        kind: str | None = None,
    ) -> tuple[EntityEntry, ...]:
        return self._list_entities(
            "events", "event_id", workspace_id, task_id=task_id, kind=kind
        )

    def recover_task(self, workspace_id: str, task_id: str) -> TaskRecovery:
        return TaskRecovery(
            workspace_id=workspace_id,
            task_id=task_id,
            task=self.get_task(workspace_id, task_id),
            checkpoint=self.latest_checkpoint(workspace_id, task_id),
            steps=self.list_steps(workspace_id, task_id=task_id),
            approvals=self.list_approvals(workspace_id, task_id=task_id),
            events=self.list_events(workspace_id, task_id=task_id),
            runs=self.list_run_history(workspace_id, task_id=task_id),
        )

    restore_task = recover_task
    upsert_task = save_task
    upsert_step = save_step
    upsert_approval = save_approval
    save_working_memory = put_working_memory
    write_checkpoint = save_checkpoint
    list_trace_events = list_events

    # Internal entity helpers ------------------------------------------

    def _save_mutable_entity(
        self,
        *,
        table: str,
        operation: str,
        workspace_id: str,
        value: Any,
        entity_id: str | None,
        entity_field: str,
        idempotency_key: str | None,
        task_id: str | None = None,
        step_id: str | None = None,
        ordinal: int | None = None,
    ) -> EntityEntry:
        if table not in {"tasks", "steps", "approvals"}:
            raise ValueError(f"unsupported mutable entity table: {table}")
        workspace_id = _required(workspace_id, "workspace_id")
        payload = _payload(value)
        _validate_payload_workspace(payload, workspace_id)
        requested_id = entity_id or _text(payload, entity_field)
        resolved_task_id = str(task_id or _text(payload, "task_id"))
        if table == "tasks":
            resolved_task_id = requested_id or _text(payload, "task_id")
        if table in {"steps", "approvals"}:
            resolved_task_id = _required(resolved_task_id, "task_id")
        resolved_step_id = str(step_id or _text(payload, "step_id"))
        status = _text(payload, "status")
        kind = _text(payload, "kind")
        resolved_ordinal = int(ordinal if ordinal is not None else _int(payload, "ordinal", 0))
        timestamp = _text(payload, "timestamp")
        content_fp = content_fingerprint(payload)
        workspace_fp = _workspace_fp(payload, workspace_id, None)
        request = {
            entity_field: requested_id,
            "task_id": resolved_task_id,
            "step_id": resolved_step_id,
            "status": status,
            "kind": kind,
            "ordinal": resolved_ordinal,
            "payload": payload,
        }
        with self._transaction() as connection:
            replay_id = self._idempotency_replay(
                connection, workspace_id, operation, idempotency_key, request
            )
            if replay_id:
                return _entity_from_row(
                    self._select_entity(
                        connection, table, entity_field, workspace_id, replay_id, required=True
                    ),
                    entity_field,
                )
            resolved_id = requested_id or f"{entity_field.removesuffix('_id')}_{uuid.uuid4().hex}"
            now = self._clock()
            existing = self._select_entity(
                connection, table, entity_field, workspace_id, resolved_id
            )
            created_at = existing["created_at"] if existing is not None else now
            if table == "tasks":
                connection.execute(
                    """
                    INSERT INTO tasks (
                        workspace_id, task_id, status, kind, payload_json,
                        content_fingerprint, workspace_fingerprint, schema_version,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(workspace_id, task_id) DO UPDATE SET
                        status = excluded.status,
                        kind = excluded.kind,
                        payload_json = excluded.payload_json,
                        content_fingerprint = excluded.content_fingerprint,
                        workspace_fingerprint = excluded.workspace_fingerprint,
                        schema_version = excluded.schema_version,
                        updated_at = excluded.updated_at
                    """,
                    (
                        workspace_id,
                        resolved_id,
                        status,
                        kind,
                        _dump(payload),
                        content_fp,
                        workspace_fp,
                        SCHEMA_VERSION,
                        created_at,
                        now,
                    ),
                )
            elif table == "steps":
                connection.execute(
                    """
                    INSERT INTO steps (
                        workspace_id, step_id, task_id, status, kind, ordinal,
                        payload_json, content_fingerprint, workspace_fingerprint,
                        schema_version, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(workspace_id, step_id) DO UPDATE SET
                        task_id = excluded.task_id,
                        status = excluded.status,
                        kind = excluded.kind,
                        ordinal = excluded.ordinal,
                        payload_json = excluded.payload_json,
                        content_fingerprint = excluded.content_fingerprint,
                        workspace_fingerprint = excluded.workspace_fingerprint,
                        schema_version = excluded.schema_version,
                        updated_at = excluded.updated_at
                    """,
                    (
                        workspace_id,
                        resolved_id,
                        resolved_task_id,
                        status,
                        kind,
                        resolved_ordinal,
                        _dump(payload),
                        content_fp,
                        workspace_fp,
                        SCHEMA_VERSION,
                        created_at,
                        now,
                    ),
                )
            else:
                connection.execute(
                    """
                    INSERT INTO approvals (
                        workspace_id, approval_id, task_id, step_id, status, kind,
                        payload_json, content_fingerprint, workspace_fingerprint,
                        schema_version, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(workspace_id, approval_id) DO UPDATE SET
                        task_id = excluded.task_id,
                        step_id = excluded.step_id,
                        status = excluded.status,
                        kind = excluded.kind,
                        payload_json = excluded.payload_json,
                        content_fingerprint = excluded.content_fingerprint,
                        workspace_fingerprint = excluded.workspace_fingerprint,
                        schema_version = excluded.schema_version,
                        updated_at = excluded.updated_at
                    """,
                    (
                        workspace_id,
                        resolved_id,
                        resolved_task_id,
                        resolved_step_id,
                        status,
                        kind,
                        _dump(payload),
                        content_fp,
                        workspace_fp,
                        SCHEMA_VERSION,
                        created_at,
                        now,
                    ),
                )
            entry = _entity_from_row(
                self._select_entity(
                    connection, table, entity_field, workspace_id, resolved_id, required=True
                ),
                entity_field,
            )
            self._remember_idempotency(
                connection, workspace_id, operation, idempotency_key, request, resolved_id
            )
            return entry

    def _get_entity(
        self,
        table: str,
        id_field: str,
        workspace_id: str,
        entity_id: str,
    ) -> EntityEntry | None:
        with self._lock:
            row = self._select_entity(
                self._connection, table, id_field, workspace_id, entity_id
            )
        return _entity_from_row(row, id_field) if row is not None else None

    def _list_entities(
        self,
        table: str,
        id_field: str,
        workspace_id: str,
        *,
        task_id: str | None = None,
        status: str | None = None,
        kind: str | None = None,
    ) -> tuple[EntityEntry, ...]:
        if table not in {"tasks", "steps", "approvals", "events"}:
            raise ValueError(f"unsupported entity table: {table}")
        query = f"SELECT * FROM {table} WHERE workspace_id = ?"
        parameters: list[Any] = [workspace_id]
        if task_id is not None:
            query += " AND task_id = ?"
            parameters.append(task_id)
        if status is not None and table != "events":
            query += " AND status = ?"
            parameters.append(status)
        if kind is not None:
            query += " AND kind = ?"
            parameters.append(kind)
        if table == "steps":
            query += " ORDER BY ordinal ASC, sequence ASC"
        elif table == "events":
            query += " ORDER BY timestamp ASC, sequence ASC"
        else:
            query += " ORDER BY sequence ASC"
        with self._lock:
            rows = self._connection.execute(query, parameters).fetchall()
        return tuple(_entity_from_row(row, id_field) for row in rows)

    # Decision helpers --------------------------------------------------

    def _prepare_decision(
        self,
        workspace_id: str,
        payload: dict[str, Any],
        *,
        memory_id: str | None,
        source: Any,
        workspace_state: Any,
        supersedes_id: str,
    ) -> dict[str, Any]:
        content = payload.get("content", payload)
        supplied_content_hash = _text(payload, "content_hash")
        content_fp = supplied_content_hash or content_fingerprint(content)
        source_run_id = _text(payload, "source_run_id") or _text(payload, "source_task_id")
        source_identity: Any = source
        if source_identity is None:
            source_identity = {
                "source_run_id": source_run_id,
                "source_versions": payload.get("source_versions", {}),
            }
            if not source_run_id and not source_identity["source_versions"]:
                source_identity = None
        source_fp = _text(payload, "source_fingerprint") or source_fingerprint(
            source_identity
        )
        workspace_fp = _workspace_fp(payload, workspace_id, workspace_state)
        kind = _text(payload, "kind") or "decision"
        status = _text(payload, "status") or "active"
        if status not in DECISION_STATUSES:
            raise ValueError(f"invalid decision memory status: {status}")
        if status != "active":
            # Promotion creates a usable memory. Other states are reached through transitions.
            status = "active"
        stored_payload = dict(payload)
        if memory_id:
            stored_payload["memory_id"] = memory_id
        stored_payload["workspace_id"] = workspace_id
        return {
            "kind": kind,
            "content": to_jsonable(content),
            "source_run_id": source_run_id,
            "status": status,
            "supersedes_id": supersedes_id,
            "payload": stored_payload,
            "content_fingerprint": content_fp,
            "source_fingerprint": source_fp,
            "workspace_fingerprint": workspace_fp,
            "created_at": _text(payload, "created_at"),
            "updated_at": _text(payload, "updated_at"),
        }

    def _insert_decision(
        self,
        connection: sqlite3.Connection,
        workspace_id: str,
        memory_id: str,
        prepared: Mapping[str, Any],
    ) -> DecisionMemoryEntry:
        now = self._clock()
        created_at = str(prepared.get("created_at") or now)
        updated_at = str(prepared.get("updated_at") or created_at)
        connection.execute(
            """
            INSERT INTO decision_memories (
                workspace_id, memory_id, kind, content_json, source_run_id, status,
                supersedes_id, superseded_by_id, stale_reason, payload_json,
                content_fingerprint, source_fingerprint, workspace_fingerprint,
                schema_version, created_at, updated_at, disabled_at, deleted_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, '', '', ?, ?, ?, ?, ?, ?, ?, '', '')
            """,
            (
                workspace_id,
                memory_id,
                prepared["kind"],
                _dump(prepared["content"]),
                prepared["source_run_id"],
                prepared["status"],
                prepared["supersedes_id"],
                _dump(prepared["payload"]),
                prepared["content_fingerprint"],
                prepared["source_fingerprint"],
                prepared["workspace_fingerprint"],
                SCHEMA_VERSION,
                created_at,
                updated_at,
            ),
        )
        return _decision_from_row(
            self._select_decision(connection, workspace_id, memory_id, required=True)
        )

    def _transition_decision(
        self,
        workspace_id: str,
        memory_id: str,
        *,
        operation: str,
        target_status: str,
        idempotency_key: str | None,
        stale_reason: str = "",
    ) -> DecisionMemoryEntry:
        request = {
            "memory_id": memory_id,
            "target_status": target_status,
            "stale_reason": stale_reason,
        }
        with self._transaction() as connection:
            replay_id = self._idempotency_replay(
                connection, workspace_id, operation, idempotency_key, request
            )
            if replay_id:
                return _decision_from_row(
                    self._select_decision(connection, workspace_id, replay_id, required=True)
                )
            row = self._select_decision(connection, workspace_id, memory_id)
            if row is None:
                raise RecordNotFoundError(f"decision memory {memory_id!r} was not found")
            if row["status"] == "deleted" and target_status != "deleted":
                raise MemoryStateConflictError(f"decision memory {memory_id!r} is deleted")
            if row["status"] == "superseded" and target_status not in {"deleted", "superseded"}:
                raise MemoryStateConflictError(f"decision memory {memory_id!r} is superseded")
            if row["status"] != target_status or (
                target_status == "stale" and stale_reason != row["stale_reason"]
            ):
                now = self._clock()
                disabled_at = now if target_status == "disabled" else str(row["disabled_at"] or "")
                deleted_at = now if target_status == "deleted" else str(row["deleted_at"] or "")
                connection.execute(
                    """
                    UPDATE decision_memories
                    SET status = ?, stale_reason = ?, disabled_at = ?, deleted_at = ?, updated_at = ?
                    WHERE workspace_id = ? AND memory_id = ?
                    """,
                    (
                        target_status,
                        stale_reason if target_status == "stale" else str(row["stale_reason"] or ""),
                        disabled_at,
                        deleted_at,
                        now,
                        workspace_id,
                        memory_id,
                    ),
                )
            entry = _decision_from_row(
                self._select_decision(connection, workspace_id, memory_id, required=True)
            )
            self._remember_idempotency(
                connection, workspace_id, operation, idempotency_key, request, memory_id
            )
            return entry

    # SQL helpers -------------------------------------------------------

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                yield self._connection
            except Exception:
                self._connection.execute("ROLLBACK")
                raise
            else:
                self._connection.execute("COMMIT")

    def _idempotency_replay(
        self,
        connection: sqlite3.Connection,
        workspace_id: str,
        operation: str,
        idempotency_key: str | None,
        request: Any,
    ) -> str | None:
        if not idempotency_key:
            return None
        row = connection.execute(
            """
            SELECT request_fingerprint, resource_id
            FROM idempotency_keys
            WHERE workspace_id = ? AND operation = ? AND idempotency_key = ?
            """,
            (workspace_id, operation, idempotency_key),
        ).fetchone()
        if row is None:
            return None
        request_fp = content_fingerprint(request)
        if row["request_fingerprint"] != request_fp:
            raise IdempotencyConflictError(
                f"idempotency key {idempotency_key!r} was reused with a different request"
            )
        return str(row["resource_id"])

    def _remember_idempotency(
        self,
        connection: sqlite3.Connection,
        workspace_id: str,
        operation: str,
        idempotency_key: str | None,
        request: Any,
        resource_id: str,
    ) -> None:
        if not idempotency_key:
            return
        connection.execute(
            """
            INSERT INTO idempotency_keys (
                workspace_id, operation, idempotency_key,
                request_fingerprint, resource_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                workspace_id,
                operation,
                idempotency_key,
                content_fingerprint(request),
                resource_id,
                self._clock(),
            ),
        )

    @staticmethod
    def _select_working(
        connection: sqlite3.Connection,
        workspace_id: str,
        task_id: str,
        key: str,
        *,
        required: bool = False,
    ) -> sqlite3.Row | None:
        row = connection.execute(
            """
            SELECT * FROM working_memory
            WHERE workspace_id = ? AND task_id = ? AND memory_key = ?
            """,
            (workspace_id, task_id, key),
        ).fetchone()
        return _required_row(row, "working memory", key) if required else row

    @staticmethod
    def _select_checkpoint(
        connection: sqlite3.Connection,
        workspace_id: str,
        checkpoint_id: str,
        *,
        required: bool = False,
    ) -> sqlite3.Row | None:
        row = connection.execute(
            "SELECT * FROM checkpoints WHERE workspace_id = ? AND checkpoint_id = ?",
            (workspace_id, checkpoint_id),
        ).fetchone()
        return _required_row(row, "checkpoint", checkpoint_id) if required else row

    @staticmethod
    def _select_run(
        connection: sqlite3.Connection,
        workspace_id: str,
        run_id: str,
        *,
        required: bool = False,
    ) -> sqlite3.Row | None:
        row = connection.execute(
            "SELECT * FROM run_history WHERE workspace_id = ? AND run_id = ?",
            (workspace_id, run_id),
        ).fetchone()
        return _required_row(row, "run", run_id) if required else row

    @staticmethod
    def _select_decision(
        connection: sqlite3.Connection,
        workspace_id: str,
        memory_id: str,
        *,
        required: bool = False,
    ) -> sqlite3.Row | None:
        row = connection.execute(
            "SELECT * FROM decision_memories WHERE workspace_id = ? AND memory_id = ?",
            (workspace_id, memory_id),
        ).fetchone()
        return _required_row(row, "decision memory", memory_id) if required else row

    @staticmethod
    def _select_entity(
        connection: sqlite3.Connection,
        table: str,
        id_field: str,
        workspace_id: str,
        entity_id: str,
        *,
        required: bool = False,
    ) -> sqlite3.Row | None:
        if (table, id_field) not in {
            ("tasks", "task_id"),
            ("steps", "step_id"),
            ("approvals", "approval_id"),
            ("events", "event_id"),
        }:
            raise ValueError("unsupported entity table/id field")
        row = connection.execute(
            f"SELECT * FROM {table} WHERE workspace_id = ? AND {id_field} = ?",
            (workspace_id, entity_id),
        ).fetchone()
        return _required_row(row, table.removesuffix("s"), entity_id) if required else row

    def _initialize_schema(self) -> None:
        current = int(self._connection.execute("PRAGMA user_version").fetchone()[0])
        if current > SCHEMA_VERSION:
            raise MemoryStoreError(
                f"database schema version {current} is newer than supported version {SCHEMA_VERSION}"
            )
        self._connection.executescript(_SCHEMA_SQL)
        self._connection.execute(
            """
            INSERT INTO schema_meta(key, value) VALUES ('schema_version', ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (str(SCHEMA_VERSION),),
        )
        self._connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")


def _working_from_row(row: sqlite3.Row) -> WorkingMemoryEntry:
    return WorkingMemoryEntry(
        workspace_id=row["workspace_id"],
        task_id=row["task_id"],
        key=row["memory_key"],
        value=_load(row["payload_json"]),
        content_fingerprint=row["content_fingerprint"],
        source_fingerprint=row["source_fingerprint"],
        workspace_fingerprint=row["workspace_fingerprint"],
        schema_version=int(row["schema_version"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _checkpoint_from_row(row: sqlite3.Row) -> CheckpointEntry:
    return CheckpointEntry(
        workspace_id=row["workspace_id"],
        checkpoint_id=row["checkpoint_id"],
        task_id=row["task_id"],
        run_id=row["run_id"],
        step_id=row["step_id"],
        payload=_load(row["payload_json"]),
        content_fingerprint=row["content_fingerprint"],
        source_fingerprint=row["source_fingerprint"],
        workspace_fingerprint=row["workspace_fingerprint"],
        schema_version=int(row["schema_version"]),
        created_at=row["created_at"],
    )


def _run_from_row(row: sqlite3.Row) -> RunHistoryEntry:
    return RunHistoryEntry(
        workspace_id=row["workspace_id"],
        run_id=row["run_id"],
        task_id=row["task_id"],
        status=row["status"],
        payload=_load(row["payload_json"]),
        content_fingerprint=row["content_fingerprint"],
        source_fingerprint=row["source_fingerprint"],
        workspace_fingerprint=row["workspace_fingerprint"],
        schema_version=int(row["schema_version"]),
        created_at=row["created_at"],
    )


def _decision_from_row(row: sqlite3.Row) -> DecisionMemoryEntry:
    return DecisionMemoryEntry(
        workspace_id=row["workspace_id"],
        memory_id=row["memory_id"],
        kind=row["kind"],
        content=_load(row["content_json"]),
        source_run_id=row["source_run_id"],
        status=row["status"],
        supersedes_id=row["supersedes_id"],
        superseded_by_id=row["superseded_by_id"],
        stale_reason=row["stale_reason"],
        payload=_load(row["payload_json"]),
        content_fingerprint=row["content_fingerprint"],
        source_fingerprint=row["source_fingerprint"],
        workspace_fingerprint=row["workspace_fingerprint"],
        schema_version=int(row["schema_version"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        disabled_at=row["disabled_at"],
        deleted_at=row["deleted_at"],
    )


def _entity_from_row(row: sqlite3.Row, id_field: str) -> EntityEntry:
    keys = set(row.keys())
    entity_id = row[id_field]
    task_id = row["task_id"] if "task_id" in keys else ""
    if id_field == "task_id":
        task_id = entity_id
    return EntityEntry(
        workspace_id=row["workspace_id"],
        entity_id=entity_id,
        task_id=task_id,
        step_id=row["step_id"] if "step_id" in keys else "",
        status=row["status"] if "status" in keys else "",
        kind=row["kind"] if "kind" in keys else "",
        ordinal=int(row["ordinal"]) if "ordinal" in keys else 0,
        timestamp=row["timestamp"] if "timestamp" in keys else "",
        payload=_load(row["payload_json"]),
        content_fingerprint=row["content_fingerprint"],
        workspace_fingerprint=row["workspace_fingerprint"],
        schema_version=int(row["schema_version"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"] if "updated_at" in keys else row["created_at"],
    )


def _payload(value: Any) -> dict[str, Any]:
    normalized = to_jsonable(value)
    if not isinstance(normalized, dict):
        raise TypeError("persisted model must serialize to a JSON object")
    return normalized


def _text(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    return "" if value is None else str(value)


def _int(payload: Mapping[str, Any], key: str, default: int) -> int:
    value = payload.get(key)
    if value is None or value == "":
        return default
    return int(value)


def _required(value: str | None, name: str) -> str:
    resolved = str(value or "").strip()
    if not resolved:
        raise ValueError(f"{name} must be non-empty")
    return resolved


def _validate_payload_workspace(payload: Mapping[str, Any], workspace_id: str) -> None:
    scopes: list[tuple[str, Mapping[str, Any]]] = [("payload", payload)]
    nested_run = payload.get("run")
    if isinstance(nested_run, Mapping):
        scopes.append(("run", nested_run))
    for parent_name, candidate in tuple(scopes):
        for child_name in ("request", "context"):
            child = candidate.get(child_name)
            if isinstance(child, Mapping):
                scopes.append((f"{parent_name}.{child_name}", child))
    for scope_name, candidate in scopes:
        declared = _text(candidate, "workspace_id")
        is_default_context = scope_name.endswith(".context") and declared == "default"
        if declared and declared != workspace_id and not is_default_context:
            raise WorkspaceScopeError(
                f"{scope_name} workspace_id {declared!r} does not match "
                f"scope {workspace_id!r}"
            )


def _workspace_fp(
    payload: Mapping[str, Any],
    workspace_id: str,
    workspace_state: Any,
) -> str:
    nested_run = payload.get("run")
    state_payload = nested_run if isinstance(nested_run, Mapping) else payload
    supplied = _text(payload, "workspace_fingerprint") or _text(
        state_payload, "workspace_fingerprint"
    )
    if supplied:
        return supplied
    state = workspace_state
    if state is None:
        context_value = state_payload.get("context")
        context = context_value if isinstance(context_value, Mapping) else {}
        request_value = state_payload.get("request")
        request = request_value if isinstance(request_value, Mapping) else {}
        revision = _text(state_payload, "workspace_revision") or _text(
            context, "workspace_revision"
        )
        workspace_hash = _text(state_payload, "workspace_hash") or _text(
            context, "workspace_hash"
        )
        context_path = _text(state_payload, "workspace_path") or _text(
            context, "workspace_path"
        )
        request_path = _text(request, "workspace_path")
        identity = {
            key: value
            for key, value in {
                "revision": revision,
                "hash": workspace_hash,
                "context_path": context_path,
                "request_path": request_path,
            }.items()
            if value
        }
        state = identity or None
    return workspace_fingerprint(workspace_id, state)


def _source_fp(payload: Any, source: Any) -> str:
    if source is not None:
        return source_fingerprint(source)
    if not isinstance(payload, Mapping):
        return ""
    nested_run = payload.get("run")
    state_payload = nested_run if isinstance(nested_run, Mapping) else payload
    context_value = state_payload.get("context")
    context = context_value if isinstance(context_value, Mapping) else {}
    request_value = state_payload.get("request")
    request = request_value if isinstance(request_value, Mapping) else {}
    source_versions = state_payload.get("source_versions") or context.get(
        "source_versions"
    )
    source_urls = state_payload.get("source_urls") or request.get("source_urls")
    source_run_id = (
        _text(state_payload, "source_run_id")
        or _text(state_payload, "source_task_id")
    )
    identity = {
        key: value
        for key, value in {
            "source_versions": source_versions,
            "source_urls": source_urls,
            "source_run_id": source_run_id,
        }.items()
        if value
    }
    return source_fingerprint(identity) if identity else ""


def _same_fingerprints(
    row: sqlite3.Row,
    content_fp: str,
    source_fp: str,
    workspace_fp: str,
) -> bool:
    return (
        row["content_fingerprint"] == content_fp
        and row["source_fingerprint"] == source_fp
        and row["workspace_fingerprint"] == workspace_fp
    )


def _decision_matches_prepared(
    row: sqlite3.Row,
    prepared: Mapping[str, Any],
) -> bool:
    return (
        row["content_fingerprint"] == prepared["content_fingerprint"]
        and row["source_fingerprint"] == prepared["source_fingerprint"]
        and row["workspace_fingerprint"] == prepared["workspace_fingerprint"]
        and row["kind"] == prepared["kind"]
        and row["source_run_id"] == prepared["source_run_id"]
        and row["supersedes_id"] == prepared["supersedes_id"]
    )


def _required_row(
    row: sqlite3.Row | None,
    kind: str,
    identifier: str,
) -> sqlite3.Row:
    if row is None:
        raise RecordNotFoundError(f"{kind} {identifier!r} was not found")
    return row


def _dump(value: Any) -> str:
    return canonical_json(value)


def _load(value: str) -> Any:
    return json.loads(value)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS idempotency_keys (
    workspace_id TEXT NOT NULL,
    operation TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    request_fingerprint TEXT NOT NULL,
    resource_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (workspace_id, operation, idempotency_key)
);

CREATE TABLE IF NOT EXISTS working_memory (
    workspace_id TEXT NOT NULL,
    task_id TEXT NOT NULL DEFAULT '',
    memory_key TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    content_fingerprint TEXT NOT NULL,
    source_fingerprint TEXT NOT NULL DEFAULT '',
    workspace_fingerprint TEXT NOT NULL,
    schema_version INTEGER NOT NULL DEFAULT 2,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (workspace_id, task_id, memory_key)
);

CREATE TABLE IF NOT EXISTS checkpoints (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    workspace_id TEXT NOT NULL,
    checkpoint_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    run_id TEXT NOT NULL DEFAULT '',
    step_id TEXT NOT NULL DEFAULT '',
    payload_json TEXT NOT NULL,
    content_fingerprint TEXT NOT NULL,
    source_fingerprint TEXT NOT NULL DEFAULT '',
    workspace_fingerprint TEXT NOT NULL,
    schema_version INTEGER NOT NULL DEFAULT 2,
    created_at TEXT NOT NULL,
    UNIQUE (workspace_id, checkpoint_id)
);
CREATE INDEX IF NOT EXISTS idx_checkpoints_workspace_task
    ON checkpoints(workspace_id, task_id, sequence DESC);

CREATE TABLE IF NOT EXISTS run_history (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    workspace_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    task_id TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT '',
    payload_json TEXT NOT NULL,
    content_fingerprint TEXT NOT NULL,
    source_fingerprint TEXT NOT NULL DEFAULT '',
    workspace_fingerprint TEXT NOT NULL,
    schema_version INTEGER NOT NULL DEFAULT 2,
    created_at TEXT NOT NULL,
    UNIQUE (workspace_id, run_id)
);
CREATE INDEX IF NOT EXISTS idx_runs_workspace_task
    ON run_history(workspace_id, task_id, sequence DESC);

CREATE TABLE IF NOT EXISTS decision_memories (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    workspace_id TEXT NOT NULL,
    memory_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    content_json TEXT NOT NULL,
    source_run_id TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active'
        CHECK(status IN ('active', 'disabled', 'stale', 'superseded', 'deleted')),
    supersedes_id TEXT NOT NULL DEFAULT '',
    superseded_by_id TEXT NOT NULL DEFAULT '',
    stale_reason TEXT NOT NULL DEFAULT '',
    payload_json TEXT NOT NULL,
    content_fingerprint TEXT NOT NULL,
    source_fingerprint TEXT NOT NULL DEFAULT '',
    workspace_fingerprint TEXT NOT NULL,
    schema_version INTEGER NOT NULL DEFAULT 2,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    disabled_at TEXT NOT NULL DEFAULT '',
    deleted_at TEXT NOT NULL DEFAULT '',
    UNIQUE (workspace_id, memory_id)
);
CREATE INDEX IF NOT EXISTS idx_decisions_workspace_status
    ON decision_memories(workspace_id, status, sequence DESC);
CREATE INDEX IF NOT EXISTS idx_decisions_fingerprints
    ON decision_memories(workspace_id, content_fingerprint, source_fingerprint);

CREATE TABLE IF NOT EXISTS tasks (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    workspace_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT '',
    kind TEXT NOT NULL DEFAULT '',
    payload_json TEXT NOT NULL,
    content_fingerprint TEXT NOT NULL,
    workspace_fingerprint TEXT NOT NULL,
    schema_version INTEGER NOT NULL DEFAULT 2,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (workspace_id, task_id)
);

CREATE TABLE IF NOT EXISTS steps (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    workspace_id TEXT NOT NULL,
    step_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT '',
    kind TEXT NOT NULL DEFAULT '',
    ordinal INTEGER NOT NULL DEFAULT 0,
    payload_json TEXT NOT NULL,
    content_fingerprint TEXT NOT NULL,
    workspace_fingerprint TEXT NOT NULL,
    schema_version INTEGER NOT NULL DEFAULT 2,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (workspace_id, step_id)
);
CREATE INDEX IF NOT EXISTS idx_steps_workspace_task
    ON steps(workspace_id, task_id, ordinal, sequence);

CREATE TABLE IF NOT EXISTS approvals (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    workspace_id TEXT NOT NULL,
    approval_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    step_id TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT '',
    kind TEXT NOT NULL DEFAULT '',
    payload_json TEXT NOT NULL,
    content_fingerprint TEXT NOT NULL,
    workspace_fingerprint TEXT NOT NULL,
    schema_version INTEGER NOT NULL DEFAULT 2,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (workspace_id, approval_id)
);
CREATE INDEX IF NOT EXISTS idx_approvals_workspace_task
    ON approvals(workspace_id, task_id, sequence);

CREATE TABLE IF NOT EXISTS events (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    workspace_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    task_id TEXT NOT NULL DEFAULT '',
    step_id TEXT NOT NULL DEFAULT '',
    kind TEXT NOT NULL DEFAULT '',
    timestamp TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    content_fingerprint TEXT NOT NULL,
    workspace_fingerprint TEXT NOT NULL,
    schema_version INTEGER NOT NULL DEFAULT 2,
    created_at TEXT NOT NULL,
    UNIQUE (workspace_id, event_id)
);
CREATE INDEX IF NOT EXISTS idx_events_workspace_task
    ON events(workspace_id, task_id, timestamp, sequence);
"""
