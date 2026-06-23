from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .service_interface import HttpRouteSpec, ServiceInterfaceReport, build_service_interface, service_interface_to_dict
from .service_runtime import create_fastapi_app


DEFAULT_MARKDOWN = "traces/service-api-readiness.md"
DEFAULT_JSON = "traces/service-api-readiness.json"


@dataclass(frozen=True)
class ApiEndpointSpec:
    method: str
    path: str
    summary: str
    request_schema: dict[str, Any]
    response_schema: dict[str, Any]
    state_dependencies: tuple[str, ...]
    approval_policy: str
    source: str


@dataclass(frozen=True)
class OptionalDependencyStatus:
    name: str
    installed: bool
    purpose: str
    install_hint: str


@dataclass(frozen=True)
class ServiceApiReadinessReport:
    passed: bool
    runnable_now: bool
    endpoint_count: int
    tool_endpoint_count: int
    optional_dependencies: tuple[OptionalDependencyStatus, ...]
    interface: ServiceInterfaceReport
    endpoints: tuple[ApiEndpointSpec, ...]
    openapi_preview: dict[str, Any]
    launch_command: str


def build_service_api_readiness(
    interface: ServiceInterfaceReport | None = None,
) -> ServiceApiReadinessReport:
    resolved_interface = interface or build_service_interface()
    dependencies = (
        _dependency_status("fastapi", "Build the ASGI app and expose path operations."),
        _dependency_status("uvicorn", "Run the local ASGI development server."),
    )
    endpoints = _build_endpoint_specs(resolved_interface)
    openapi_preview = _build_openapi_preview(endpoints)
    runnable_now = all(dependency.installed for dependency in dependencies)
    return ServiceApiReadinessReport(
        passed=resolved_interface.passed and len(endpoints) == resolved_interface.tool_count + 2,
        runnable_now=runnable_now,
        endpoint_count=len(endpoints),
        tool_endpoint_count=resolved_interface.tool_count,
        optional_dependencies=dependencies,
        interface=resolved_interface,
        endpoints=endpoints,
        openapi_preview=openapi_preview,
        launch_command="PYTHONPATH=src python -m bioml_agent.service_api serve --host 127.0.0.1 --port 8787",
    )


def service_api_readiness_to_dict(report: ServiceApiReadinessReport) -> dict[str, Any]:
    return {
        "passed": report.passed,
        "runnable_now": report.runnable_now,
        "endpoint_count": report.endpoint_count,
        "tool_endpoint_count": report.tool_endpoint_count,
        "optional_dependencies": [
            {
                "name": dependency.name,
                "installed": dependency.installed,
                "purpose": dependency.purpose,
                "install_hint": dependency.install_hint,
            }
            for dependency in report.optional_dependencies
        ],
        "interface": service_interface_to_dict(report.interface),
        "endpoints": [_endpoint_to_dict(endpoint) for endpoint in report.endpoints],
        "openapi_preview": report.openapi_preview,
        "launch_command": report.launch_command,
    }


def format_service_api_readiness(report: ServiceApiReadinessReport | None = None) -> str:
    resolved = report or build_service_api_readiness()
    lines = [
        "# TracePilot Service API Readiness",
        f"Passed: {resolved.passed}",
        f"Runnable now: {resolved.runnable_now}",
        f"Endpoints: {resolved.endpoint_count}",
        f"Tool endpoints: {resolved.tool_endpoint_count}",
        "",
        "## Optional Dependencies",
    ]
    for dependency in resolved.optional_dependencies:
        status = "installed" if dependency.installed else "missing"
        lines.extend(
            [
                f"- `{dependency.name}`: {status}",
                f"  - Purpose: {dependency.purpose}",
                f"  - Install: `{dependency.install_hint}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Endpoint Plan",
        ]
    )
    for endpoint in resolved.endpoints:
        state = ", ".join(endpoint.state_dependencies) if endpoint.state_dependencies else "(none)"
        required = endpoint.request_schema.get("required") or []
        required_text = ", ".join(required) if required else "(none)"
        lines.extend(
            [
                "",
                f"### {endpoint.method} {endpoint.path}",
                endpoint.summary,
                f"- source: `{endpoint.source}`",
                f"- request required: {required_text}",
                f"- response schema: `{endpoint.response_schema.get('x-python-type', endpoint.response_schema.get('type', 'object'))}`",
                f"- session state dependencies: {state}",
                f"- approval policy: {endpoint.approval_policy}",
            ]
        )
    lines.extend(
        [
            "",
            "## Launch",
            "Install the optional API extras first:",
            "",
            "```bash",
            "python -m pip install -e '.[api]'",
            "```",
            "",
            "Then run:",
            "",
            "```bash",
            resolved.launch_command,
            "```",
            "",
            "FastAPI will expose `/openapi.json` and `/docs` when the optional dependencies are installed.",
        ]
    )
    return "\n".join(lines)


def write_service_api_readiness(
    path: str | Path,
    report: ServiceApiReadinessReport | None = None,
    json_output: bool = False,
) -> None:
    resolved = report or build_service_api_readiness()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = (
        json.dumps(service_api_readiness_to_dict(resolved), indent=2, ensure_ascii=False)
        if json_output
        else format_service_api_readiness(resolved)
    )
    output_path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect or serve the optional TracePilot FastAPI adapter.")
    subparsers = parser.add_subparsers(dest="command")

    report_parser = subparsers.add_parser("report", help="Generate service API readiness report.")
    report_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    report_parser.add_argument("--out", help="Write the report to this path.")

    serve_parser = subparsers.add_parser("serve", help="Run the optional FastAPI app with uvicorn.")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Host to bind.")
    serve_parser.add_argument("--port", type=int, default=8787, help="Port to bind.")

    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument("--out", help="Write the report to this path.")
    args = parser.parse_args()

    if args.command == "serve":
        _serve(host=args.host, port=args.port)
        return

    report = build_service_api_readiness()
    json_output = bool(getattr(args, "json", False))
    out = getattr(args, "out", None)
    text = (
        json.dumps(service_api_readiness_to_dict(report), indent=2, ensure_ascii=False)
        if json_output
        else format_service_api_readiness(report)
    )
    if out:
        output_path = Path(out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    else:
        print(text)
    raise SystemExit(0 if report.passed else 1)


def _build_endpoint_specs(interface: ServiceInterfaceReport) -> tuple[ApiEndpointSpec, ...]:
    base = (
        ApiEndpointSpec(
            method="GET",
            path="/health",
            summary="Return service liveness and route count.",
            request_schema=_object_schema({}),
            response_schema=_object_schema({"ok": {"type": "boolean"}, "route_count": {"type": "integer"}}),
            state_dependencies=(),
            approval_policy="no approval required; liveness read is read-only",
            source="service_runtime.create_fastapi_app",
        ),
        ApiEndpointSpec(
            method="GET",
            path="/service-interface",
            summary="Return the contract-derived MCP-style descriptors and HTTP route specs.",
            request_schema=_object_schema({}),
            response_schema={"type": "object", "x-python-type": "ServiceInterfaceReport"},
            state_dependencies=(),
            approval_policy="no approval required; interface read is read-only",
            source="service_runtime.create_fastapi_app",
        ),
    )
    tools = tuple(_endpoint_from_route(route) for route in interface.http_routes)
    return base + tools


def _endpoint_from_route(route: HttpRouteSpec) -> ApiEndpointSpec:
    return ApiEndpointSpec(
        method=route.method,
        path=route.path,
        summary=f"Invoke `{route.tool_name}` through the session-backed service runtime.",
        request_schema=route.request_schema,
        response_schema={"type": "object", "x-python-type": "ServiceInvocationResult"},
        state_dependencies=route.state_dependencies,
        approval_policy=route.approval_policy,
        source="service_runtime.ServiceRuntime.invoke",
    )


def _build_openapi_preview(endpoints: tuple[ApiEndpointSpec, ...]) -> dict[str, Any]:
    paths: dict[str, Any] = {}
    for endpoint in endpoints:
        operation = {
            "summary": endpoint.summary,
            "x-source": endpoint.source,
            "x-state-dependencies": list(endpoint.state_dependencies),
            "x-approval-policy": endpoint.approval_policy,
            "responses": {
                "200": {
                    "description": "Successful response.",
                    "content": {"application/json": {"schema": endpoint.response_schema}},
                }
            },
        }
        if endpoint.method == "POST":
            operation["requestBody"] = _request_body(endpoint.request_schema)
        paths.setdefault(endpoint.path, {})[endpoint.method.lower()] = operation
    return {
        "openapi": "3.1.0",
        "info": {
            "title": "TracePilot Service",
            "version": "0.1.0",
            "description": "Optional FastAPI adapter for the session-backed TracePilot planning service runtime.",
        },
        "paths": paths,
    }


def _request_body(schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "required": True,
        "content": {
            "application/json": {
                "schema": schema,
            }
        },
    }


def _dependency_status(name: str, purpose: str) -> OptionalDependencyStatus:
    installed = importlib.util.find_spec(name) is not None
    return OptionalDependencyStatus(
        name=name,
        installed=installed,
        purpose=purpose,
        install_hint="python -m pip install -e '.[api]'",
    )


def _serve(host: str, port: int) -> None:
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit("Install API extras first: python -m pip install -e '.[api]'") from exc
    try:
        app = create_fastapi_app()
    except RuntimeError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(str(exc)) from exc
    uvicorn.run(app, host=host, port=port)


def _endpoint_to_dict(endpoint: ApiEndpointSpec) -> dict[str, Any]:
    return {
        "method": endpoint.method,
        "path": endpoint.path,
        "summary": endpoint.summary,
        "request_schema": endpoint.request_schema,
        "response_schema": endpoint.response_schema,
        "state_dependencies": list(endpoint.state_dependencies),
        "approval_policy": endpoint.approval_policy,
        "source": endpoint.source,
    }


def _object_schema(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


if __name__ == "__main__":
    main()
