from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, is_dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from .models import (
    ApprovalRequest,
    ApprovalStatus,
    PlanStep,
    ToolResult,
    TraceEvent,
    utc_now,
)
from .tooling import EffectClass, ToolContract, ToolRegistry


class ExecutionPolicyError(RuntimeError):
    pass


class ApprovalRequiredError(ExecutionPolicyError):
    pass


class ApprovalNotFoundError(ExecutionPolicyError, KeyError):
    pass


class ApprovalHashMismatchError(ExecutionPolicyError):
    pass


class WorkspaceBoundaryError(ExecutionPolicyError):
    pass


class IdempotencyConflictError(ExecutionPolicyError):
    pass


class ExecutionRuntime:
    """Run registered tools with approval, idempotency, and workspace guards."""

    def __init__(
        self,
        registry: ToolRegistry,
        workspace_path: str | Path,
        *,
        allowed_permissions: tuple[str, ...] | None = None,
    ) -> None:
        self.registry = registry
        self.workspace_path = Path(workspace_path).expanduser().resolve()
        self.allowed_permissions = frozenset(
            allowed_permissions
            or (
                "workspace_read",
                "workspace_write",
                "internal_state",
                "network",
                "subprocess",
            )
        )
        self._approvals: dict[str, ApprovalRequest] = {}
        self._approval_arguments: dict[str, dict[str, Any]] = {}
        self._idempotency: dict[str, tuple[str, ToolResult]] = {}
        self._trace: list[TraceEvent] = []

    def prepare(
        self,
        tool: str | PlanStep,
        arguments: Mapping[str, Any] | None = None,
        *,
        task_id: str = "",
        run_id: str = "",
        step_id: str = "",
    ) -> ApprovalRequest:
        name, payload, resolved_step_id = _resolve_invocation(
            tool, arguments, step_id=step_id
        )
        contract = self.registry.contract(name)
        if contract.effect_class != EffectClass.MUTATION:
            raise ExecutionPolicyError(
                f"tool '{name}' is {contract.effect_class.value}; approval is only "
                "prepared for mutation tools"
            )
        self._check_contract_permissions(contract)
        self._guard_workspace(payload)
        snapshot = _canonicalize_mapping(payload)
        request = ApprovalRequest(
            task_id=task_id,
            run_id=run_id,
            step_id=resolved_step_id,
            tool_name=name,
            arguments=snapshot,
            arguments_hash=hash_arguments(snapshot),
            status=ApprovalStatus.PENDING,
        )
        self._approvals[request.approval_id] = request
        self._approval_arguments[request.approval_id] = snapshot
        self._trace.append(
            TraceEvent(
                task_id=task_id,
                run_id=run_id,
                step_id=resolved_step_id,
                event_type="approval_requested",
                tool_name=name,
                inputs=snapshot,
                status=ApprovalStatus.PENDING.value,
                metadata={"approval_id": request.approval_id},
            )
        )
        return request

    def approve(
        self,
        approval: str | ApprovalRequest,
        *,
        approved_by: str = "human",
        arguments: Mapping[str, Any] | None = None,
    ) -> ApprovalRequest:
        current = self._get_approval(approval)
        if current.status not in (ApprovalStatus.PENDING, ApprovalStatus.MODIFIED):
            raise ExecutionPolicyError(
                f"approval '{current.approval_id}' is already {current.status.value}"
            )
        expected = self._approval_arguments[current.approval_id]
        supplied = (
            _canonicalize_mapping(arguments)
            if arguments is not None
            else _canonicalize_mapping(current.arguments)
        )
        if hash_arguments(supplied) != hash_arguments(expected):
            raise ApprovalHashMismatchError(
                "approved arguments differ from the frozen approval parameters; "
                "prepare a new or modified approval"
            )
        approved = replace(
            current,
            arguments=dict(expected),
            arguments_hash=hash_arguments(expected),
            status=ApprovalStatus.APPROVED,
            decided_at=utc_now(),
            decided_by=approved_by,
            rejection_reason="",
        )
        self._approvals[approved.approval_id] = approved
        self._trace.append(
            TraceEvent(
                task_id=approved.task_id,
                run_id=approved.run_id,
                step_id=approved.step_id,
                event_type="approval_decided",
                tool_name=approved.tool_name,
                status=ApprovalStatus.APPROVED.value,
                metadata={
                    "approval_id": approved.approval_id,
                    "decided_by": approved_by,
                },
            )
        )
        return approved

    def modify(
        self,
        approval: str | ApprovalRequest,
        arguments: Mapping[str, Any],
        *,
        modified_by: str = "human",
    ) -> ApprovalRequest:
        current = self._get_approval(approval)
        if current.status not in (ApprovalStatus.PENDING, ApprovalStatus.MODIFIED):
            raise ExecutionPolicyError(
                f"approval '{current.approval_id}' cannot be modified from "
                f"{current.status.value}"
            )
        snapshot = _canonicalize_mapping(arguments)
        self._guard_workspace(snapshot)
        modified = replace(
            current,
            arguments=snapshot,
            arguments_hash=hash_arguments(snapshot),
            status=ApprovalStatus.MODIFIED,
            decided_at="",
            decided_by=modified_by,
            rejection_reason="",
        )
        self._approvals[modified.approval_id] = modified
        self._approval_arguments[modified.approval_id] = snapshot
        self._trace.append(
            TraceEvent(
                task_id=modified.task_id,
                run_id=modified.run_id,
                step_id=modified.step_id,
                event_type="approval_modified",
                tool_name=modified.tool_name,
                inputs=snapshot,
                status=ApprovalStatus.MODIFIED.value,
                metadata={"approval_id": modified.approval_id},
            )
        )
        return modified

    def reject(
        self,
        approval: str | ApprovalRequest,
        *,
        reason: str = "",
        rejected_by: str = "human",
    ) -> ApprovalRequest:
        current = self._get_approval(approval)
        if current.status not in (ApprovalStatus.PENDING, ApprovalStatus.MODIFIED):
            raise ExecutionPolicyError(
                f"approval '{current.approval_id}' cannot be rejected from "
                f"{current.status.value}"
            )
        rejected = replace(
            current,
            status=ApprovalStatus.REJECTED,
            decided_at=utc_now(),
            decided_by=rejected_by,
            rejection_reason=reason,
        )
        self._approvals[rejected.approval_id] = rejected
        self._trace.append(
            TraceEvent(
                task_id=rejected.task_id,
                run_id=rejected.run_id,
                step_id=rejected.step_id,
                event_type="approval_decided",
                tool_name=rejected.tool_name,
                status=ApprovalStatus.REJECTED.value,
                error=reason,
                metadata={"approval_id": rejected.approval_id},
            )
        )
        return rejected

    def execute(
        self,
        tool: str | PlanStep,
        arguments: Mapping[str, Any] | None = None,
        *,
        task_id: str = "",
        run_id: str = "",
        step_id: str = "",
        approval: str | ApprovalRequest | None = None,
        approval_id: str = "",
        idempotency_key: str = "",
        state: Mapping[str, Any] | None = None,
    ) -> ToolResult:
        name, payload, resolved_step_id = _resolve_invocation(
            tool, arguments, step_id=step_id
        )
        contract = self.registry.contract(name)
        self._check_contract_permissions(contract)
        self._guard_workspace(payload)
        payload_hash = hash_arguments(payload)
        resolved_approval: ApprovalRequest | None = None
        if contract.effect_class == EffectClass.MUTATION:
            approval_reference = approval or approval_id
            if not approval_reference:
                raise ApprovalRequiredError(
                    f"mutation tool '{name}' requires an approved frozen request"
                )
            resolved_approval = self._get_approval(approval_reference)
            self._validate_approval(
                resolved_approval,
                name=name,
                payload_hash=payload_hash,
                task_id=task_id,
                step_id=resolved_step_id,
            )
            if not idempotency_key:
                idempotency_key = resolved_approval.approval_id

        fingerprint = f"{name}:{payload_hash}"
        if idempotency_key and idempotency_key in self._idempotency:
            previous_fingerprint, previous_result = self._idempotency[idempotency_key]
            if previous_fingerprint != fingerprint:
                raise IdempotencyConflictError(
                    f"idempotency key '{idempotency_key}' was already used for "
                    "different tool parameters"
                )
            return replace(previous_result, cached=True)

        started_at = utc_now()
        started = time.monotonic()
        output: Any = None
        error = ""
        success = False
        try:
            output = self.registry.call(name, payload, state=state)
            success = True
        except Exception as exc:  # Tool failures are persisted as results.
            error = f"{type(exc).__name__}: {exc}"
        duration_ms = (time.monotonic() - started) * 1000
        result = ToolResult(
            task_id=task_id,
            run_id=run_id,
            step_id=resolved_step_id,
            tool_name=name,
            effect_class=contract.effect_class.value,
            success=success,
            output=output,
            error=error,
            idempotency_key=idempotency_key,
            started_at=started_at,
            finished_at=utc_now(),
            duration_ms=duration_ms,
            metadata={
                "approval_id": (
                    resolved_approval.approval_id if resolved_approval else ""
                ),
                "arguments_hash": payload_hash,
            },
        )
        if idempotency_key:
            self._idempotency[idempotency_key] = (fingerprint, result)
        event = TraceEvent(
            task_id=task_id,
            run_id=run_id,
            step_id=resolved_step_id,
            event_type="tool_completed" if success else "tool_failed",
            tool_name=name,
            inputs=_canonicalize_mapping(payload),
            outputs=_summarize_output(output) if success else {},
            status="succeeded" if success else "failed",
            error=error,
            duration_ms=duration_ms,
            metadata={
                "effect_class": contract.effect_class.value,
                "idempotency_key": idempotency_key,
            },
        )
        self._trace.append(event)
        return result

    def approval(self, approval_id: str) -> ApprovalRequest:
        return self._get_approval(approval_id)

    def restore_approval(self, request: ApprovalRequest) -> ApprovalRequest:
        """Restore a persisted approval while rechecking its frozen parameters."""

        contract = self.registry.contract(request.tool_name)
        if contract.effect_class != EffectClass.MUTATION:
            raise ExecutionPolicyError(
                f"persisted approval references non-mutation tool "
                f"'{request.tool_name}'"
            )
        self._check_contract_permissions(contract)
        snapshot = _canonicalize_mapping(request.arguments)
        actual_hash = hash_arguments(snapshot)
        if not request.arguments_hash or request.arguments_hash != actual_hash:
            raise ApprovalHashMismatchError(
                "persisted approval arguments do not match arguments_hash"
            )
        self._guard_workspace(snapshot)
        existing = self._approvals.get(request.approval_id)
        if existing is not None and (
            existing.tool_name != request.tool_name
            or existing.arguments_hash != request.arguments_hash
        ):
            raise ApprovalHashMismatchError(
                "approval id is already associated with different parameters"
            )
        restored = replace(request, arguments=snapshot)
        self._approvals[restored.approval_id] = restored
        self._approval_arguments[restored.approval_id] = snapshot
        self._trace.append(
            TraceEvent(
                task_id=restored.task_id,
                run_id=restored.run_id,
                step_id=restored.step_id,
                event_type="approval_restored",
                tool_name=restored.tool_name,
                status=restored.status.value,
                metadata={"approval_id": restored.approval_id},
            )
        )
        return restored

    def load_approval(self, request: ApprovalRequest) -> ApprovalRequest:
        return self.restore_approval(request)

    @property
    def approvals(self) -> tuple[ApprovalRequest, ...]:
        return tuple(self._approvals.values())

    @property
    def trace(self) -> tuple[TraceEvent, ...]:
        return tuple(self._trace)

    def _get_approval(
        self,
        approval: str | ApprovalRequest,
    ) -> ApprovalRequest:
        approval_id = (
            approval.approval_id if isinstance(approval, ApprovalRequest) else approval
        )
        try:
            stored = self._approvals[approval_id]
        except KeyError as exc:
            raise ApprovalNotFoundError(
                f"unknown approval request: {approval_id}"
            ) from exc
        if isinstance(approval, ApprovalRequest):
            if approval.arguments_hash != stored.arguments_hash:
                raise ApprovalHashMismatchError(
                    "approval object does not match the stored frozen parameters"
                )
        return stored

    def _validate_approval(
        self,
        approval: ApprovalRequest,
        *,
        name: str,
        payload_hash: str,
        task_id: str,
        step_id: str,
    ) -> None:
        if approval.status != ApprovalStatus.APPROVED:
            raise ApprovalRequiredError(
                f"approval '{approval.approval_id}' is {approval.status.value}, "
                "not approved"
            )
        if approval.tool_name != name:
            raise ApprovalHashMismatchError(
                f"approval is for '{approval.tool_name}', not '{name}'"
            )
        if approval.arguments_hash != payload_hash:
            raise ApprovalHashMismatchError(
                "execution parameters differ from the approved frozen parameters"
            )
        if task_id and approval.task_id and task_id != approval.task_id:
            raise ApprovalHashMismatchError("approval belongs to a different task")
        if step_id and approval.step_id and step_id != approval.step_id:
            raise ApprovalHashMismatchError("approval belongs to a different plan step")

    def _check_contract_permissions(self, contract: ToolContract) -> None:
        denied = sorted(set(contract.permissions) - self.allowed_permissions)
        if denied:
            raise ExecutionPolicyError(
                f"tool '{contract.name}' requires denied permissions: "
                f"{', '.join(denied)}"
            )

    def _guard_workspace(self, payload: Mapping[str, Any]) -> None:
        for key, value in _walk_mapping(payload):
            if not _looks_like_path_key(key):
                continue
            values = value if isinstance(value, (list, tuple, set)) else (value,)
            for item in values:
                if not isinstance(item, (str, Path)) or not str(item):
                    continue
                if "://" in str(item):
                    continue
                path = Path(item).expanduser()
                resolved = (
                    path.resolve()
                    if path.is_absolute()
                    else (self.workspace_path / path).resolve()
                )
                if not _is_relative_to(resolved, self.workspace_path):
                    raise WorkspaceBoundaryError(
                        f"path argument '{key}' escapes workspace: {item}"
                    )


def hash_arguments(arguments: Mapping[str, Any]) -> str:
    payload = json.dumps(
        _canonicalize(arguments),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resolve_invocation(
    tool: str | PlanStep,
    arguments: Mapping[str, Any] | None,
    *,
    step_id: str,
) -> tuple[str, dict[str, Any], str]:
    if isinstance(tool, PlanStep):
        if arguments is not None:
            raise ValueError("arguments must be omitted when executing a PlanStep")
        return tool.tool_name, dict(tool.arguments), step_id or tool.step_id
    if not isinstance(tool, str) or not tool:
        raise ValueError("tool name must not be empty")
    return tool, dict(arguments or {}), step_id


def _canonicalize_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    canonical = _canonicalize(dict(value or {}))
    if not isinstance(canonical, dict):
        raise TypeError("tool arguments must be a mapping")
    return canonical


def _canonicalize(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _canonicalize(asdict(value))
    if isinstance(value, Enum):
        return _canonicalize(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _canonicalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_canonicalize(item) for item in value]
        return sorted(
            normalized,
            key=lambda item: json.dumps(item, sort_keys=True, default=str),
        )
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite numbers cannot be approved or hashed")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _walk_mapping(value: Mapping[str, Any]) -> list[tuple[str, Any]]:
    items: list[tuple[str, Any]] = []
    for key, item in value.items():
        items.append((str(key), item))
        if isinstance(item, Mapping):
            items.extend(_walk_mapping(item))
        elif isinstance(item, (list, tuple, set)):
            for nested in item:
                if isinstance(nested, Mapping):
                    items.extend(_walk_mapping(nested))
    return items


def _looks_like_path_key(key: str) -> bool:
    normalized = key.lower()
    exact = {
        "cwd",
        "file",
        "filename",
        "path",
        "directory",
        "workspace",
        "workspace_path",
        "root",
        "target",
    }
    return (
        normalized in exact
        or normalized.endswith("_path")
        or normalized.endswith("_file")
        or normalized.endswith("_dir")
        or normalized.endswith("_root")
    )


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _summarize_output(output: Any) -> dict[str, Any]:
    if output is None:
        return {"type": "NoneType"}
    if isinstance(output, Mapping):
        return {
            "type": type(output).__name__,
            "keys": [str(key) for key in list(output)[:20]],
        }
    if isinstance(output, (list, tuple, set)):
        return {"type": type(output).__name__, "count": len(output)}
    text = str(output)
    return {
        "type": type(output).__name__,
        "preview": text[:500],
        "truncated": len(text) > 500,
    }


__all__ = [
    "ApprovalHashMismatchError",
    "ApprovalNotFoundError",
    "ApprovalRequiredError",
    "ExecutionPolicyError",
    "ExecutionRuntime",
    "IdempotencyConflictError",
    "WorkspaceBoundaryError",
    "hash_arguments",
]
