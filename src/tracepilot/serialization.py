from __future__ import annotations

import json
import re
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from .models import (
    AgentRun,
    ApprovalRequest,
    EvidenceChunk,
    EvidenceClaim,
    MemoryRecord,
    PlanStep,
    TaskContext,
    TaskPlan,
    TaskRequest,
    ToolResult,
    TraceEvent,
)


SCHEMA_VERSION = 2

_PUBLIC_SECRET_KEYS = frozenset(
    {
        "api_key",
        "authorization",
        "cookie",
        "password",
        "private_key",
        "secret",
        "secret_key",
        "token",
    }
)
_PUBLIC_SECRET_SUFFIXES = (
    "_access_token",
    "_api_key",
    "_auth_token",
    "_client_secret",
    "_password",
    "_private_key",
    "_secret_key",
)
_PUBLIC_SECRET_PATTERNS = (
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{12,}"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(
        r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----.*?"
        r"-----END (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
        re.DOTALL,
    ),
)


class CheckpointVersionError(ValueError):
    pass


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def run_to_dict(run: AgentRun) -> dict[str, Any]:
    return to_jsonable(run)


def sanitize_public_payload(
    value: Any,
    *,
    roots: Mapping[str, str | Path] | None = None,
) -> Any:
    """Return a JSON-compatible copy with local paths and common secrets redacted."""

    normalized_roots = _normalize_public_roots(roots or {})
    return _sanitize_public_value(to_jsonable(value), normalized_roots)


def run_to_public_dict(
    run: AgentRun,
    *,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    roots: dict[str, str | Path] = {
        "workspace": run.request.workspace_path,
        "home": Path.home(),
    }
    if project_root is not None:
        roots["project"] = project_root
    return sanitize_public_payload(run, roots=roots)


def checkpoint_to_dict(run: AgentRun) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "tracepilot_agent_run",
        "run": run_to_dict(run),
    }


def write_checkpoint(run: AgentRun, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(checkpoint_to_dict(run), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output


def load_checkpoint(path: str | Path) -> AgentRun:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return checkpoint_from_dict(raw)


def checkpoint_from_dict(raw: dict[str, Any]) -> AgentRun:
    version = raw.get("schema_version")
    if version is None and _looks_like_legacy_bioml_state(raw):
        return migrate_v1_bioml_checkpoint(raw)
    if version == 1:
        state = raw.get("state", raw.get("run", raw))
        if not isinstance(state, dict) or not _looks_like_legacy_bioml_state(state):
            raise CheckpointVersionError("schema v1 checkpoint is not a recognized BioML graph state")
        return migrate_v1_bioml_checkpoint(state)
    if version != SCHEMA_VERSION:
        raise CheckpointVersionError(
            f"unsupported checkpoint schema_version={version!r}; expected 1 or {SCHEMA_VERSION}"
        )
    if raw.get("kind") != "tracepilot_agent_run":
        raise ValueError(f"unsupported checkpoint kind: {raw.get('kind')!r}")
    payload = raw.get("run")
    if not isinstance(payload, dict):
        raise ValueError("checkpoint is missing a run object")
    return run_from_dict(payload)


def run_from_dict(raw: dict[str, Any]) -> AgentRun:
    request = TaskRequest(**raw.get("request", {}))
    context_raw = raw.get("context", {})
    context = TaskContext(
        task_id=context_raw.get("task_id", request.task_id),
        workspace_id=context_raw.get("workspace_id", request.workspace_id),
        workspace_path=context_raw.get("workspace_path", request.workspace_path),
        workspace_revision=context_raw.get("workspace_revision", ""),
        workspace_hash=context_raw.get("workspace_hash", ""),
        source_versions=context_raw.get("source_versions", {}),
        evidence=[EvidenceChunk(**item) for item in context_raw.get("evidence", [])],
        memories=[MemoryRecord(**item) for item in context_raw.get("memories", [])],
        state=context_raw.get("state", {}),
        metadata=context_raw.get("metadata", {}),
    )
    plan_raw = raw.get("plan", {})
    plan = TaskPlan(
        task_id=plan_raw.get("task_id", request.task_id),
        steps=tuple(PlanStep(**item) for item in plan_raw.get("steps", [])),
        revision=plan_raw.get("revision", 1),
        rationale=plan_raw.get("rationale", ""),
        max_concurrency=plan_raw.get("max_concurrency", 4),
        created_at=plan_raw.get("created_at", ""),
        metadata=plan_raw.get("metadata", {}),
    )
    evidence: list[EvidenceChunk | EvidenceClaim] = []
    for item in raw.get("evidence", []):
        if "evidence_ids" in item or "claim_id" in item:
            evidence.append(EvidenceClaim(**item))
        else:
            evidence.append(EvidenceChunk(**item))
    return AgentRun(
        run_id=raw.get("run_id", ""),
        task_id=raw.get("task_id", request.task_id),
        request=request,
        context=context,
        plan=plan,
        evidence=evidence,
        tool_results=[ToolResult(**item) for item in raw.get("tool_results", [])],
        approvals=[ApprovalRequest(**item) for item in raw.get("approvals", [])],
        trace=[TraceEvent(**item) for item in raw.get("trace", [])],
        status=raw.get("status", "created"),
        summary=raw.get("summary", ""),
        error=raw.get("error", ""),
        created_at=raw.get("created_at", ""),
        updated_at=raw.get("updated_at", ""),
        metadata=raw.get("metadata", {}),
    )


def migrate_v1_bioml_checkpoint(raw: dict[str, Any]) -> AgentRun:
    question = raw.get("question", {})
    goal = question.get("text", "") if isinstance(question, dict) else str(question)
    project_name = str(raw.get("project_name", "BioML"))
    request = TaskRequest(
        goal=goal,
        workspace_id=f"legacy:{project_name}",
        workspace_path=".",
        metadata={
            "migrated_from": "bioml_agent_checkpoint_v1",
            "legacy_domain": question.get("domain", "") if isinstance(question, dict) else "",
        },
    )
    legacy_nodes = (
        "search_literature",
        "read_experiment_context",
        "score_transfer_fit",
        "draft_experiment_plan",
        "create_review_checkpoint",
    )
    next_node = str(raw.get("next_node", legacy_nodes[0]))
    completed = next_node == "done"
    steps = tuple(
        PlanStep(
            step_id=f"legacy_{name}",
            description=f"Migrated BioML step: {name}",
            tool_name=name,
            depends_on=() if index == 0 else (f"legacy_{legacy_nodes[index - 1]}",),
            effect_class="read",
            metadata={
                "legacy_status": (
                    "completed"
                    if completed or (next_node in legacy_nodes and index < legacy_nodes.index(next_node))
                    else "pending"
                )
            },
        )
        for index, name in enumerate(legacy_nodes)
    )
    claims = [
        EvidenceClaim(
            claim_id=str(item.get("evidence_id", f"legacy_evidence_{index + 1}")),
            text=str(item.get("supports", item.get("title", ""))),
            evidence_ids=(str(item.get("paper_id", "")),) if item.get("paper_id") else (),
            supported=True,
            confidence=float(item.get("relevance", 0.5)),
            uncertainty=str(item.get("cautions", "")),
            metadata={"legacy_evidence": item},
        )
        for index, item in enumerate(raw.get("evidence", []))
        if isinstance(item, dict)
    ]
    trace = [
        TraceEvent(
            task_id=request.task_id,
            step_id=str(item.get("step", "")),
            event_type="legacy_tool_call",
            tool_name=str(item.get("tool", "")),
            inputs=item.get("inputs", {}),
            outputs=item.get("outputs", {}),
            status="completed",
            metadata={"migrated": True},
        )
        for item in raw.get("trace", [])
        if isinstance(item, dict)
    ]
    failure = raw.get("failure")
    status = "completed" if completed and not failure else "needs_revision" if failure else "researching"
    plan_summary = raw.get("plan", {}) if isinstance(raw.get("plan"), dict) else {}
    return AgentRun(
        task_id=request.task_id,
        request=request,
        context=TaskContext(
            task_id=request.task_id,
            workspace_id=request.workspace_id,
            workspace_path=request.workspace_path,
            state={"legacy_graph_state": raw},
            metadata={"migrated": True},
        ),
        plan=TaskPlan(
            task_id=request.task_id,
            steps=steps,
            rationale="Mapped from the fixed BioML graph; resume with the generic planner.",
            metadata={"legacy_next_node": next_node},
        ),
        evidence=claims,
        trace=trace,
        status=status,
        summary=str(plan_summary.get("summary", "")),
        error=str(failure.get("message", "")) if isinstance(failure, dict) else "",
        metadata={"schema_version": SCHEMA_VERSION, "migrated_from": 1},
    )


def _looks_like_legacy_bioml_state(raw: dict[str, Any]) -> bool:
    return "question" in raw and (
        "next_node" in raw or "project_name" in raw or "candidates" in raw
    )


def _normalize_public_roots(
    roots: Mapping[str, str | Path],
) -> tuple[tuple[str, str], ...]:
    normalized: list[tuple[str, str]] = []
    seen: set[str] = set()
    for label, raw_root in roots.items():
        root = str(Path(raw_root).expanduser().resolve(strict=False)).rstrip("/")
        if not root or root == "." or root in seen:
            continue
        safe_label = re.sub(r"[^a-z0-9_-]", "-", label.lower()).strip("-")
        if not safe_label:
            safe_label = "local"
        seen.add(root)
        normalized.append((root, safe_label))
    normalized.sort(key=lambda item: len(item[0]), reverse=True)
    return tuple(normalized)


def _sanitize_public_value(
    value: Any,
    roots: tuple[tuple[str, str], ...],
) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            public_key = _sanitize_public_text(key, roots)
            if _is_public_secret_key(key):
                sanitized[public_key] = "[REDACTED]"
            else:
                sanitized[public_key] = _sanitize_public_value(item, roots)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_public_value(item, roots) for item in value]
    if isinstance(value, str):
        return _sanitize_public_text(value, roots)
    return value


def _sanitize_public_text(
    value: str,
    roots: tuple[tuple[str, str], ...],
) -> str:
    sanitized = value
    for root, label in roots:
        replacement = f"{label}://"
        sanitized = sanitized.replace(f"file://{root}/", replacement)
        sanitized = sanitized.replace(f"file://{root}", replacement)
        sanitized = sanitized.replace(f"{root}/", replacement)
        sanitized = sanitized.replace(root, replacement)
    for pattern in _PUBLIC_SECRET_PATTERNS:
        sanitized = pattern.sub("[REDACTED]", sanitized)
    return sanitized


def _is_public_secret_key(key: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
    return normalized in _PUBLIC_SECRET_KEYS or normalized.endswith(_PUBLIC_SECRET_SUFFIXES)


__all__ = [
    "CheckpointVersionError",
    "SCHEMA_VERSION",
    "checkpoint_from_dict",
    "checkpoint_to_dict",
    "load_checkpoint",
    "migrate_v1_bioml_checkpoint",
    "run_from_dict",
    "run_to_dict",
    "run_to_public_dict",
    "sanitize_public_payload",
    "to_jsonable",
    "write_checkpoint",
]
