from __future__ import annotations

import argparse
import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .serialization import to_jsonable
from .tool_contracts import ToolContract, build_tool_contracts


DEFAULT_MARKDOWN = "traces/service-interface-report.md"
DEFAULT_JSON = "traces/service-interface-report.json"
_SAFE_ROUTE_RE = re.compile(r"^/tools/[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class McpToolDescriptor:
    name: str
    description: str
    input_schema: dict[str, Any]
    annotations: dict[str, bool]
    output_type: str
    approval_policy: str
    failure_modes: tuple[str, ...]
    trace_required: bool


@dataclass(frozen=True)
class HttpRouteSpec:
    method: str
    path: str
    tool_name: str
    request_schema: dict[str, Any]
    response_type: str
    state_dependencies: tuple[str, ...]
    approval_policy: str
    failure_modes: tuple[str, ...]
    trace_required: bool


@dataclass(frozen=True)
class ServiceInterfaceValidation:
    passed: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class ServiceInterfaceReport:
    version: int
    passed: bool
    tool_count: int
    mcp_tools: tuple[McpToolDescriptor, ...]
    http_routes: tuple[HttpRouteSpec, ...]
    errors: tuple[str, ...]
    warnings: tuple[str, ...]


def build_service_interface(
    contracts: tuple[ToolContract, ...] | None = None,
) -> ServiceInterfaceReport:
    resolved_contracts = contracts or build_tool_contracts()
    mcp_tools = tuple(_mcp_descriptor_from_contract(contract) for contract in resolved_contracts)
    http_routes = tuple(_http_route_from_contract(contract) for contract in resolved_contracts)
    validation = validate_service_interface_components(
        contracts=resolved_contracts,
        mcp_tools=mcp_tools,
        http_routes=http_routes,
    )
    return ServiceInterfaceReport(
        version=1,
        passed=validation.passed,
        tool_count=len(resolved_contracts),
        mcp_tools=mcp_tools,
        http_routes=http_routes,
        errors=validation.errors,
        warnings=validation.warnings,
    )


def validate_service_interface(
    report: ServiceInterfaceReport,
    contracts: tuple[ToolContract, ...] | None = None,
) -> ServiceInterfaceValidation:
    return validate_service_interface_components(
        contracts=contracts or build_tool_contracts(),
        mcp_tools=report.mcp_tools,
        http_routes=report.http_routes,
    )


def validate_service_interface_components(
    contracts: tuple[ToolContract, ...],
    mcp_tools: tuple[McpToolDescriptor, ...],
    http_routes: tuple[HttpRouteSpec, ...],
) -> ServiceInterfaceValidation:
    errors: list[str] = []
    warnings: list[str] = []
    contract_names = tuple(contract.name for contract in contracts)
    mcp_names = tuple(tool.name for tool in mcp_tools)
    route_names = tuple(route.tool_name for route in http_routes)

    if len(mcp_tools) != len(contracts):
        errors.append(f"MCP descriptor count drifted from contracts: {len(mcp_tools)} != {len(contracts)}")
    if len(http_routes) != len(contracts):
        errors.append(f"HTTP route count drifted from contracts: {len(http_routes)} != {len(contracts)}")
    if mcp_names != contract_names:
        errors.append(f"MCP descriptor names drifted from contracts: {mcp_names} != {contract_names}")
    if route_names != contract_names:
        errors.append(f"HTTP route names drifted from contracts: {route_names} != {contract_names}")

    seen_paths: set[str] = set()
    contract_by_name = {contract.name: contract for contract in contracts}
    for tool in mcp_tools:
        errors.extend(_mcp_descriptor_errors(tool))
        contract = contract_by_name.get(tool.name)
        if contract and tool.input_schema != contract.model_input_schema:
            errors.append(f"{tool.name}: MCP inputSchema must be derived from model_input_schema")
        if tool.name == "create_review_checkpoint" and "requires_approval" not in tool.approval_policy:
            errors.append("create_review_checkpoint approval_policy must mention requires_approval")

    for route in http_routes:
        errors.extend(_http_route_errors(route))
        if route.path in seen_paths:
            errors.append(f"duplicate HTTP route path: {route.path}")
        seen_paths.add(route.path)
        contract = contract_by_name.get(route.tool_name)
        if contract:
            if route.request_schema != contract.model_input_schema:
                errors.append(f"{route.tool_name}: HTTP request_schema must be derived from model_input_schema")
            if route.state_dependencies and not route.request_schema.get("required"):
                warnings.append(
                    f"{route.tool_name}: route relies on session-injected state: "
                    f"{', '.join(route.state_dependencies)}"
                )

    return ServiceInterfaceValidation(
        passed=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def service_interface_to_dict(report: ServiceInterfaceReport) -> dict[str, Any]:
    return {
        "version": report.version,
        "passed": report.passed,
        "tool_count": report.tool_count,
        "mcp_tools": [_mcp_descriptor_to_dict(tool) for tool in report.mcp_tools],
        "http_routes": [_http_route_to_dict(route) for route in report.http_routes],
        "errors": list(report.errors),
        "warnings": list(report.warnings),
    }


def format_service_interface(report: ServiceInterfaceReport | None = None) -> str:
    resolved = report or build_service_interface()
    lines = [
        "# TracePilot Service Interface Report",
        f"Passed: {resolved.passed}",
        f"Tool count: {resolved.tool_count}",
        "",
        "## What This Is",
        "This report derives MCP-style tool descriptors and HTTP route specs from the canonical tool contracts.",
        "It is a service boundary plan, not a claim that this repository already runs a full MCP server or FastAPI app.",
        "",
        "## MCP-style Tool Descriptors",
    ]
    for tool in resolved.mcp_tools:
        required = _format_required(tool.input_schema)
        annotations = ", ".join(f"{key}={value}" for key, value in sorted(tool.annotations.items()))
        lines.extend(
            [
                "",
                f"### {tool.name}",
                tool.description,
                f"- inputSchema required: {required}",
                f"- output: `{tool.output_type}`",
                f"- annotations: {annotations}",
                f"- approval policy: {tool.approval_policy}",
                f"- failure modes: {', '.join(tool.failure_modes)}",
                f"- trace required: {tool.trace_required}",
            ]
        )

    lines.append("")
    lines.append("## HTTP Route Specs")
    for route in resolved.http_routes:
        state = ", ".join(route.state_dependencies) if route.state_dependencies else "(none)"
        lines.extend(
            [
                "",
                f"### {route.method} {route.path}",
                f"- tool: `{route.tool_name}`",
                f"- request schema required: {_format_required(route.request_schema)}",
                f"- response type: `{route.response_type}`",
                f"- session-injected state: {state}",
                f"- approval policy: {route.approval_policy}",
                f"- failure modes: {', '.join(route.failure_modes)}",
                f"- trace required: {route.trace_required}",
            ]
        )

    lines.extend(["", "## Validation"])
    if resolved.errors:
        lines.append("Errors:")
        lines.extend(f"- {item}" for item in resolved.errors)
    else:
        lines.append("Errors: none")
    if resolved.warnings:
        lines.append("Warnings:")
        lines.extend(f"- {item}" for item in resolved.warnings)
    else:
        lines.append("Warnings: none")
    return "\n".join(lines)


def write_service_interface(
    path: str | Path,
    report: ServiceInterfaceReport | None = None,
    json_output: bool = False,
) -> None:
    resolved = report or build_service_interface()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = (
        json.dumps(service_interface_to_dict(resolved), indent=2, ensure_ascii=False)
        if json_output
        else format_service_interface(resolved)
    )
    output_path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export TracePilot tool contracts as service interface specs.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument("--out", help="Write the report to this path.")
    args = parser.parse_args()

    report = build_service_interface()
    text = (
        json.dumps(service_interface_to_dict(report), indent=2, ensure_ascii=False)
        if args.json
        else format_service_interface(report)
    )
    if args.out:
        output_path = Path(args.out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    else:
        print(text)
    raise SystemExit(0 if report.passed else 1)


def _mcp_descriptor_from_contract(contract: ToolContract) -> McpToolDescriptor:
    return McpToolDescriptor(
        name=contract.name,
        description=contract.description,
        input_schema=copy.deepcopy(contract.model_input_schema),
        annotations={
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": contract.name == "search_literature",
            "readOnlyHint": True,
        },
        output_type=contract.output_type,
        approval_policy=contract.approval_policy,
        failure_modes=contract.failure_modes,
        trace_required=contract.trace_required,
    )


def _http_route_from_contract(contract: ToolContract) -> HttpRouteSpec:
    return HttpRouteSpec(
        method="POST",
        path=f"/tools/{contract.name}",
        tool_name=contract.name,
        request_schema=copy.deepcopy(contract.model_input_schema),
        response_type=contract.output_type,
        state_dependencies=contract.state_dependencies,
        approval_policy=contract.approval_policy,
        failure_modes=contract.failure_modes,
        trace_required=contract.trace_required,
    )


def _mcp_descriptor_errors(tool: McpToolDescriptor) -> list[str]:
    errors: list[str] = []
    if not tool.name:
        errors.append("MCP descriptor missing name")
    if not tool.description:
        errors.append(f"{tool.name}: missing MCP description")
    if tool.input_schema.get("type") != "object":
        errors.append(f"{tool.name}: MCP inputSchema must be an object schema")
    if tool.input_schema.get("additionalProperties") is not False:
        errors.append(f"{tool.name}: MCP inputSchema must disable additionalProperties")
    if "required" not in tool.input_schema:
        errors.append(f"{tool.name}: MCP inputSchema must include required")
    if not tool.approval_policy:
        errors.append(f"{tool.name}: missing approval_policy")
    if not tool.failure_modes:
        errors.append(f"{tool.name}: missing failure_modes")
    if tool.trace_required is not True:
        errors.append(f"{tool.name}: trace_required must be true")
    if tool.annotations.get("destructiveHint") is not False:
        errors.append(f"{tool.name}: destructiveHint must be false")
    if tool.annotations.get("readOnlyHint") is not True:
        errors.append(f"{tool.name}: readOnlyHint must be true")
    return errors


def _http_route_errors(route: HttpRouteSpec) -> list[str]:
    errors: list[str] = []
    if route.method != "POST":
        errors.append(f"{route.tool_name}: HTTP method must be POST")
    if not _SAFE_ROUTE_RE.match(route.path):
        errors.append(f"{route.tool_name}: route path is not deterministic URL-safe: {route.path}")
    if route.request_schema.get("type") != "object":
        errors.append(f"{route.tool_name}: HTTP request_schema must be an object schema")
    if route.request_schema.get("additionalProperties") is not False:
        errors.append(f"{route.tool_name}: HTTP request_schema must disable additionalProperties")
    if "required" not in route.request_schema:
        errors.append(f"{route.tool_name}: HTTP request_schema must include required")
    if not route.approval_policy:
        errors.append(f"{route.tool_name}: missing approval_policy")
    if not route.failure_modes:
        errors.append(f"{route.tool_name}: missing failure_modes")
    if route.trace_required is not True:
        errors.append(f"{route.tool_name}: trace_required must be true")
    return errors


def _mcp_descriptor_to_dict(tool: McpToolDescriptor) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description,
        "inputSchema": copy.deepcopy(tool.input_schema),
        "annotations": dict(tool.annotations),
        "output_type": tool.output_type,
        "approval_policy": tool.approval_policy,
        "failure_modes": list(tool.failure_modes),
        "trace_required": tool.trace_required,
    }


def _http_route_to_dict(route: HttpRouteSpec) -> dict[str, Any]:
    return to_jsonable(route)


def _format_required(schema: dict[str, Any]) -> str:
    required = schema.get("required")
    if not required:
        return "(none)"
    return ", ".join(str(item) for item in required)


if __name__ == "__main__":
    main()
