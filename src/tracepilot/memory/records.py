from __future__ import annotations

import dataclasses
import inspect
from dataclasses import dataclass
from typing import Any, Generic, TypeVar


T = TypeVar("T")


def model_from_payload(payload: Any, model_type: type[T] | None = None) -> Any | T:
    if model_type is None:
        return payload
    if isinstance(payload, dict) and model_type.__name__ == "AgentRun":
        try:
            from tracepilot.serialization import checkpoint_from_dict, run_from_dict

            if "schema_version" in payload and "run" in payload:
                return checkpoint_from_dict(payload)
            return run_from_dict(payload)
        except ImportError:
            pass
    validator = getattr(model_type, "model_validate", None)
    if callable(validator):
        return validator(payload)
    if not isinstance(payload, dict):
        return model_type(payload)
    try:
        return model_type(**payload)
    except TypeError:
        accepted = _accepted_fields(model_type)
        if accepted is None:
            raise
        return model_type(**{key: value for key, value in payload.items() if key in accepted})


def _accepted_fields(model_type: type[Any]) -> set[str] | None:
    if dataclasses.is_dataclass(model_type):
        return {field.name for field in dataclasses.fields(model_type)}
    try:
        signature = inspect.signature(model_type)
    except (TypeError, ValueError):
        return None
    if any(parameter.kind == parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return None
    return {
        name
        for name, parameter in signature.parameters.items()
        if name != "self"
        and parameter.kind
        in {
            parameter.POSITIONAL_OR_KEYWORD,
            parameter.KEYWORD_ONLY,
        }
    }


@dataclass(frozen=True)
class WorkingMemoryEntry:
    workspace_id: str
    task_id: str
    key: str
    value: Any
    content_fingerprint: str
    source_fingerprint: str
    workspace_fingerprint: str
    schema_version: int
    created_at: str
    updated_at: str

    @property
    def content_hash(self) -> str:
        return self.content_fingerprint

    def to_model(self, model_type: type[T] | None = None) -> Any | T:
        return model_from_payload(self.value, model_type)


@dataclass(frozen=True)
class CheckpointEntry:
    workspace_id: str
    checkpoint_id: str
    task_id: str
    run_id: str
    step_id: str
    payload: Any
    content_fingerprint: str
    source_fingerprint: str
    workspace_fingerprint: str
    schema_version: int
    created_at: str

    @property
    def content_hash(self) -> str:
        return self.content_fingerprint

    def to_model(self, model_type: type[T] | None = None) -> Any | T:
        return model_from_payload(self.payload, model_type)


@dataclass(frozen=True)
class RunHistoryEntry:
    workspace_id: str
    run_id: str
    task_id: str
    status: str
    payload: Any
    content_fingerprint: str
    source_fingerprint: str
    workspace_fingerprint: str
    schema_version: int
    created_at: str

    @property
    def content_hash(self) -> str:
        return self.content_fingerprint

    def to_model(self, model_type: type[T] | None = None) -> Any | T:
        return model_from_payload(self.payload, model_type)


@dataclass(frozen=True)
class DecisionMemoryEntry:
    workspace_id: str
    memory_id: str
    kind: str
    content: Any
    source_run_id: str
    status: str
    supersedes_id: str
    superseded_by_id: str
    stale_reason: str
    payload: dict[str, Any]
    content_fingerprint: str
    source_fingerprint: str
    workspace_fingerprint: str
    schema_version: int
    created_at: str
    updated_at: str
    disabled_at: str
    deleted_at: str

    @property
    def content_hash(self) -> str:
        return self.content_fingerprint

    def as_memory_payload(self) -> dict[str, Any]:
        payload = dict(self.payload)
        payload.update(
            {
                "memory_id": self.memory_id,
                "workspace_id": self.workspace_id,
                "kind": self.kind,
                "content": self.content,
                "source_run_id": self.source_run_id,
                "content_hash": self.content_fingerprint,
                "workspace_fingerprint": self.workspace_fingerprint,
                "status": self.status,
                "supersedes_id": self.supersedes_id,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
            }
        )
        return payload

    def to_model(self, model_type: type[T] | None = None) -> dict[str, Any] | T:
        payload = self.as_memory_payload()
        if model_type is not None and model_type.__name__ == "MemoryRecord":
            accepted = _accepted_fields(model_type)
            if accepted is not None and "workspace_revision" in accepted:
                # The public MemoryRecord deliberately has only active/stale/disabled.
                # Preserve tombstone detail in metadata while hydrating to that contract.
                if payload["status"] in {"superseded", "deleted"}:
                    payload["metadata"] = {
                        **dict(payload.get("metadata") or {}),
                        "persistence_status": payload["status"],
                        "superseded_by_id": self.superseded_by_id,
                    }
                    payload["status"] = "disabled"
        return model_from_payload(payload, model_type)


@dataclass(frozen=True)
class EntityEntry:
    workspace_id: str
    entity_id: str
    task_id: str
    step_id: str
    status: str
    kind: str
    ordinal: int
    timestamp: str
    payload: Any
    content_fingerprint: str
    workspace_fingerprint: str
    schema_version: int
    created_at: str
    updated_at: str

    @property
    def content_hash(self) -> str:
        return self.content_fingerprint

    def to_model(self, model_type: type[T] | None = None) -> Any | T:
        return model_from_payload(self.payload, model_type)


@dataclass(frozen=True)
class TaskRecovery:
    workspace_id: str
    task_id: str
    task: EntityEntry | None
    checkpoint: CheckpointEntry | None
    steps: tuple[EntityEntry, ...]
    approvals: tuple[EntityEntry, ...]
    events: tuple[EntityEntry, ...]
    runs: tuple[RunHistoryEntry, ...]

    @property
    def resumable(self) -> bool:
        return self.checkpoint is not None
