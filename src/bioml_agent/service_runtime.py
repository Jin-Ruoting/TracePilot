from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .quality_gate import PROMOTION_QUESTION
from .schemas import (
    EvidenceItem,
    ExperimentContext,
    ExperimentPlan,
    PaperCandidate,
    ReviewCheckpoint,
    TraceEvent,
)
from .serialization import to_jsonable
from .service_interface import HttpRouteSpec, ServiceInterfaceReport, build_service_interface, service_interface_to_dict
from .tools import ToolRegistry, build_default_registry


DEFAULT_MARKDOWN = "traces/service-runtime-demo.md"
DEFAULT_JSON = "traces/service-runtime-demo.json"


@dataclass
class ServiceSessionState:
    session_id: str = "portfolio-service-demo"
    candidates: list[PaperCandidate] = field(default_factory=list)
    context: ExperimentContext | None = None
    evidence: list[EvidenceItem] = field(default_factory=list)
    plan: ExperimentPlan | None = None
    review: ReviewCheckpoint | None = None
    trace: list[TraceEvent] = field(default_factory=list)


@dataclass(frozen=True)
class ServiceInvocationResult:
    method: str
    path: str
    tool_name: str
    request: dict[str, Any]
    ok: bool
    status_code: int
    output: Any
    error: str
    state_before: tuple[str, ...]
    state_after: tuple[str, ...]
    trace_events_added: int
    approval_required: bool


@dataclass(frozen=True)
class ServiceRuntimeDemoReport:
    passed: bool
    route_count: int
    invocation_count: int
    trace_event_count: int
    final_state_keys: tuple[str, ...]
    approval_required: bool
    interface: ServiceInterfaceReport
    invocations: tuple[ServiceInvocationResult, ...]


class ServiceRuntime:
    def __init__(
        self,
        registry: ToolRegistry | None = None,
        interface: ServiceInterfaceReport | None = None,
    ) -> None:
        self.registry = registry or build_default_registry()
        self.interface = interface or build_service_interface()
        self._routes = {(route.method, route.path): route for route in self.interface.http_routes}

    @property
    def routes(self) -> tuple[HttpRouteSpec, ...]:
        return self.interface.http_routes

    def invoke(
        self,
        method: str,
        path: str,
        request: dict[str, Any] | None = None,
        state: ServiceSessionState | None = None,
    ) -> ServiceInvocationResult:
        payload = request or {}
        session = state or ServiceSessionState()
        route = self._routes.get((method.upper(), path))
        state_before = _state_keys(session)
        trace_before = len(session.trace)
        if route is None:
            return _failed_invocation(
                method=method.upper(),
                path=path,
                tool_name="",
                request=payload,
                status_code=404,
                error=f"unknown route: {method.upper()} {path}",
                state=session,
                state_before=state_before,
                trace_before=trace_before,
            )

        validation_error = _validate_request(payload, route.request_schema)
        if validation_error:
            return _failed_invocation(
                method=route.method,
                path=route.path,
                tool_name=route.tool_name,
                request=payload,
                status_code=422,
                error=validation_error,
                state=session,
                state_before=state_before,
                trace_before=trace_before,
            )
        dependency_error = _state_dependency_error(session, route.state_dependencies)
        if dependency_error:
            return _failed_invocation(
                method=route.method,
                path=route.path,
                tool_name=route.tool_name,
                request=payload,
                status_code=409,
                error=dependency_error,
                state=session,
                state_before=state_before,
                trace_before=trace_before,
            )

        try:
            output = self.registry.call(route.tool_name, session.trace, **_tool_kwargs(route, payload, session))
        except Exception as exc:  # pragma: no cover - defensive boundary for service adapters
            return _failed_invocation(
                method=route.method,
                path=route.path,
                tool_name=route.tool_name,
                request=payload,
                status_code=500,
                error=f"{type(exc).__name__}: {exc}",
                state=session,
                state_before=state_before,
                trace_before=trace_before,
            )
        _update_state(route.tool_name, output, session)
        return ServiceInvocationResult(
            method=route.method,
            path=route.path,
            tool_name=route.tool_name,
            request=payload,
            ok=True,
            status_code=200,
            output=output,
            error="",
            state_before=state_before,
            state_after=_state_keys(session),
            trace_events_added=len(session.trace) - trace_before,
            approval_required=_approval_required(output),
        )


def build_service_runtime(registry: ToolRegistry | None = None) -> ServiceRuntime:
    return ServiceRuntime(registry=registry)


def run_service_runtime_demo(runtime: ServiceRuntime | None = None) -> ServiceRuntimeDemoReport:
    service = runtime or build_service_runtime()
    state = ServiceSessionState()
    calls = (
        ("POST", "/tools/search_literature", {"query": PROMOTION_QUESTION, "top_k": 4}),
        ("POST", "/tools/read_experiment_context", {"project_name": "EpiCETRA"}),
        ("POST", "/tools/score_transfer_fit", {}),
        ("POST", "/tools/draft_experiment_plan", {}),
        ("POST", "/tools/create_review_checkpoint", {}),
    )
    results = tuple(service.invoke(method, path, payload, state) for method, path, payload in calls)
    approval_required = any(result.approval_required for result in results)
    passed = (
        all(result.ok for result in results)
        and len(results) == service.interface.tool_count
        and approval_required
        and "review" in _state_keys(state)
    )
    return ServiceRuntimeDemoReport(
        passed=passed,
        route_count=len(service.routes),
        invocation_count=len(results),
        trace_event_count=len(state.trace),
        final_state_keys=_state_keys(state),
        approval_required=approval_required,
        interface=service.interface,
        invocations=results,
    )


def service_runtime_demo_to_dict(report: ServiceRuntimeDemoReport) -> dict[str, Any]:
    return {
        "passed": report.passed,
        "route_count": report.route_count,
        "invocation_count": report.invocation_count,
        "trace_event_count": report.trace_event_count,
        "final_state_keys": list(report.final_state_keys),
        "approval_required": report.approval_required,
        "interface": service_interface_to_dict(report.interface),
        "invocations": [_invocation_to_dict(invocation) for invocation in report.invocations],
    }


def format_service_runtime_demo(report: ServiceRuntimeDemoReport | None = None) -> str:
    resolved = report or run_service_runtime_demo()
    lines = [
        "# TracePilot Service Runtime Demo",
        f"Passed: {resolved.passed}",
        f"Routes: {resolved.route_count}",
        f"Invocations: {resolved.invocation_count}",
        f"Trace events: {resolved.trace_event_count}",
        f"Final state keys: {', '.join(resolved.final_state_keys)}",
        f"Approval required: {resolved.approval_required}",
        "",
        "## What This Proves",
        "The same contract-derived route specs can execute the full TracePilot planning workflow through a session-backed service boundary.",
        "State-dependent tools receive candidates, context, evidence, and plan from service session state instead of trusting the caller to fabricate internal objects.",
        "",
        "## Invocation Trace",
    ]
    for index, invocation in enumerate(resolved.invocations, start=1):
        status = "PASS" if invocation.ok else "FAIL"
        state_before = ", ".join(invocation.state_before) if invocation.state_before else "(empty)"
        state_after = ", ".join(invocation.state_after) if invocation.state_after else "(empty)"
        lines.extend(
            [
                "",
                f"### {index}. {status} {invocation.method} {invocation.path}",
                f"- tool: `{invocation.tool_name}`",
                f"- status code: `{invocation.status_code}`",
                f"- request keys: {_format_keys(invocation.request)}",
                f"- state before: {state_before}",
                f"- state after: {state_after}",
                f"- trace events added: {invocation.trace_events_added}",
                f"- approval required: {invocation.approval_required}",
            ]
        )
        if invocation.error:
            lines.append(f"- error: {invocation.error}")
        else:
            lines.append(f"- output type: `{type(invocation.output).__name__}`")
    return "\n".join(lines)


def write_service_runtime_demo(
    path: str | Path,
    report: ServiceRuntimeDemoReport | None = None,
    json_output: bool = False,
) -> None:
    resolved = report or run_service_runtime_demo()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = (
        json.dumps(service_runtime_demo_to_dict(resolved), indent=2, ensure_ascii=False)
        if json_output
        else format_service_runtime_demo(resolved)
    )
    output_path.write_text(text, encoding="utf-8")


def create_fastapi_app(runtime: ServiceRuntime | None = None) -> Any:
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        raise RuntimeError("Install FastAPI to use create_fastapi_app: pip install fastapi uvicorn") from exc

    service = runtime or build_service_runtime()
    app = FastAPI(title="TracePilot Service")
    app.state.session = ServiceSessionState()

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "route_count": len(service.routes)}

    @app.get("/service-interface")
    def service_interface() -> dict[str, Any]:
        return service_interface_to_dict(service.interface)

    @app.post("/tools/{tool_name}")
    def call_tool(tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        result = service.invoke("POST", f"/tools/{tool_name}", payload, app.state.session)
        if not result.ok:
            raise HTTPException(status_code=result.status_code, detail=result.error)
        return _invocation_to_dict(result)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a session-backed service runtime demo.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument("--out", help="Write the report to this path.")
    args = parser.parse_args()

    report = run_service_runtime_demo()
    text = (
        json.dumps(service_runtime_demo_to_dict(report), indent=2, ensure_ascii=False)
        if args.json
        else format_service_runtime_demo(report)
    )
    if args.out:
        output_path = Path(args.out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    else:
        print(text)
    raise SystemExit(0 if report.passed else 1)


def _tool_kwargs(route: HttpRouteSpec, payload: dict[str, Any], state: ServiceSessionState) -> dict[str, Any]:
    kwargs = dict(payload)
    for dependency in route.state_dependencies:
        kwargs[dependency] = _state_value(state, dependency)
    return kwargs


def _state_dependency_error(state: ServiceSessionState, dependencies: tuple[str, ...]) -> str:
    missing = tuple(name for name in dependencies if not _has_state_value(state, name))
    if not missing:
        return ""
    return f"missing session state: {', '.join(missing)}"


def _validate_request(payload: dict[str, Any], schema: dict[str, Any]) -> str:
    if not isinstance(payload, dict):
        return "request body must be a JSON object"
    required = tuple(schema.get("required") or ())
    missing = tuple(name for name in required if name not in payload)
    if missing:
        return f"missing required fields: {', '.join(missing)}"
    allowed = set((schema.get("properties") or {}).keys())
    if schema.get("additionalProperties") is False:
        extra = tuple(sorted(set(payload) - allowed))
        if extra:
            return f"unknown fields: {', '.join(extra)}"
    properties = schema.get("properties") or {}
    for name, value in payload.items():
        error = _property_error(name, value, properties.get(name, {}))
        if error:
            return error
    return ""


def _property_error(name: str, value: Any, schema: dict[str, Any]) -> str:
    expected = schema.get("type")
    if expected == "string" and not isinstance(value, str):
        return f"{name} must be a string"
    if expected == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
        return f"{name} must be an integer"
    minimum = schema.get("minimum")
    maximum = schema.get("maximum")
    if isinstance(value, int) and minimum is not None and value < minimum:
        return f"{name} must be >= {minimum}"
    if isinstance(value, int) and maximum is not None and value > maximum:
        return f"{name} must be <= {maximum}"
    return ""


def _update_state(tool_name: str, output: Any, state: ServiceSessionState) -> None:
    if tool_name == "search_literature":
        state.candidates = output
    elif tool_name == "read_experiment_context":
        state.context = output
    elif tool_name == "score_transfer_fit":
        state.evidence = output
    elif tool_name == "draft_experiment_plan":
        state.plan = output
    elif tool_name == "create_review_checkpoint":
        state.review = output


def _failed_invocation(
    method: str,
    path: str,
    tool_name: str,
    request: dict[str, Any],
    status_code: int,
    error: str,
    state: ServiceSessionState,
    state_before: tuple[str, ...],
    trace_before: int,
) -> ServiceInvocationResult:
    return ServiceInvocationResult(
        method=method,
        path=path,
        tool_name=tool_name,
        request=request,
        ok=False,
        status_code=status_code,
        output=None,
        error=error,
        state_before=state_before,
        state_after=_state_keys(state),
        trace_events_added=len(state.trace) - trace_before,
        approval_required=False,
    )


def _state_keys(state: ServiceSessionState) -> tuple[str, ...]:
    keys: list[str] = []
    if state.candidates:
        keys.append("candidates")
    if state.context is not None:
        keys.append("context")
    if state.evidence:
        keys.append("evidence")
    if state.plan is not None:
        keys.append("plan")
    if state.review is not None:
        keys.append("review")
    if state.trace:
        keys.append("trace")
    return tuple(keys)


def _has_state_value(state: ServiceSessionState, name: str) -> bool:
    value = _state_value(state, name)
    if isinstance(value, list):
        return bool(value)
    return value is not None


def _state_value(state: ServiceSessionState, name: str) -> Any:
    if not hasattr(state, name):
        return None
    return getattr(state, name)


def _approval_required(output: Any) -> bool:
    return isinstance(output, ReviewCheckpoint) and output.status == "requires_approval"


def _invocation_to_dict(invocation: ServiceInvocationResult) -> dict[str, Any]:
    return {
        "method": invocation.method,
        "path": invocation.path,
        "tool_name": invocation.tool_name,
        "request": to_jsonable(invocation.request),
        "ok": invocation.ok,
        "status_code": invocation.status_code,
        "output": to_jsonable(invocation.output),
        "error": invocation.error,
        "state_before": list(invocation.state_before),
        "state_after": list(invocation.state_after),
        "trace_events_added": invocation.trace_events_added,
        "approval_required": invocation.approval_required,
    }


def _format_keys(value: dict[str, Any]) -> str:
    if not value:
        return "(none)"
    return ", ".join(sorted(value))


if __name__ == "__main__":
    main()
