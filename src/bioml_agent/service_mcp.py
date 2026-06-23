from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .serialization import to_jsonable
from .service_interface import McpToolDescriptor, ServiceInterfaceReport, build_service_interface, service_interface_to_dict
from .service_runtime import ServiceRuntime, ServiceSessionState, build_service_runtime, run_service_runtime_demo, service_runtime_demo_to_dict


DEFAULT_MARKDOWN = "traces/service-mcp-readiness.md"
DEFAULT_JSON = "traces/service-mcp-readiness.json"
SERVER_NAME = "TracePilot MCP"


@dataclass(frozen=True)
class McpToolBindingSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    output_type: str
    annotations: dict[str, bool]
    state_dependencies: tuple[str, ...]
    approval_policy: str
    runtime_path: str


@dataclass(frozen=True)
class McpResourceSpec:
    uri: str
    name: str
    description: str
    source: str
    read_only: bool


@dataclass(frozen=True)
class McpPromptSpec:
    name: str
    description: str
    arguments: tuple[str, ...]
    source: str


@dataclass(frozen=True)
class McpTransportSpec:
    name: str
    launch_command: str
    purpose: str


@dataclass(frozen=True)
class McpOptionalDependencyStatus:
    name: str
    installed: bool
    purpose: str
    install_hint: str


@dataclass(frozen=True)
class ServiceMcpReadinessReport:
    passed: bool
    runnable_now: bool
    server_name: str
    tool_binding_count: int
    resource_count: int
    prompt_count: int
    transport_count: int
    optional_dependencies: tuple[McpOptionalDependencyStatus, ...]
    interface: ServiceInterfaceReport
    tool_bindings: tuple[McpToolBindingSpec, ...]
    resources: tuple[McpResourceSpec, ...]
    prompts: tuple[McpPromptSpec, ...]
    transports: tuple[McpTransportSpec, ...]


def build_service_mcp_readiness(
    interface: ServiceInterfaceReport | None = None,
) -> ServiceMcpReadinessReport:
    resolved_interface = interface or build_service_interface()
    dependencies = (
        _dependency_status("mcp", "Build and run the optional FastMCP server."),
    )
    bindings = tuple(_binding_from_tool(tool) for tool in resolved_interface.mcp_tools)
    resources = _resource_specs()
    prompts = _prompt_specs()
    transports = _transport_specs()
    runnable_now = all(dependency.installed for dependency in dependencies)
    return ServiceMcpReadinessReport(
        passed=(
            resolved_interface.passed
            and len(bindings) == resolved_interface.tool_count
            and len(resources) == 2
            and len(prompts) == 1
            and len(transports) == 2
        ),
        runnable_now=runnable_now,
        server_name=SERVER_NAME,
        tool_binding_count=len(bindings),
        resource_count=len(resources),
        prompt_count=len(prompts),
        transport_count=len(transports),
        optional_dependencies=dependencies,
        interface=resolved_interface,
        tool_bindings=bindings,
        resources=resources,
        prompts=prompts,
        transports=transports,
    )


def service_mcp_readiness_to_dict(report: ServiceMcpReadinessReport) -> dict[str, Any]:
    return {
        "passed": report.passed,
        "runnable_now": report.runnable_now,
        "server_name": report.server_name,
        "tool_binding_count": report.tool_binding_count,
        "resource_count": report.resource_count,
        "prompt_count": report.prompt_count,
        "transport_count": report.transport_count,
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
        "tool_bindings": [_binding_to_dict(binding) for binding in report.tool_bindings],
        "resources": [to_jsonable(resource) for resource in report.resources],
        "prompts": [to_jsonable(prompt) for prompt in report.prompts],
        "transports": [to_jsonable(transport) for transport in report.transports],
    }


def format_service_mcp_readiness(report: ServiceMcpReadinessReport | None = None) -> str:
    resolved = report or build_service_mcp_readiness()
    lines = [
        "# TracePilot MCP Readiness",
        f"Passed: {resolved.passed}",
        f"Runnable now: {resolved.runnable_now}",
        f"Server: {resolved.server_name}",
        f"Tool bindings: {resolved.tool_binding_count}",
        f"Resources: {resolved.resource_count}",
        f"Prompts: {resolved.prompt_count}",
        f"Transports: {resolved.transport_count}",
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

    lines.append("")
    lines.append("## Tool Bindings")
    for binding in resolved.tool_bindings:
        state = ", ".join(binding.state_dependencies) if binding.state_dependencies else "(none)"
        annotations = ", ".join(f"{key}={value}" for key, value in sorted(binding.annotations.items()))
        required = binding.input_schema.get("required") or []
        required_text = ", ".join(required) if required else "(none)"
        lines.extend(
            [
                "",
                f"### {binding.name}",
                binding.description,
                f"- input required: {required_text}",
                f"- output: `{binding.output_type}`",
                f"- annotations: {annotations}",
                f"- session state dependencies: {state}",
                f"- approval policy: {binding.approval_policy}",
                f"- runtime path: `{binding.runtime_path}`",
            ]
        )

    lines.append("")
    lines.append("## Resources")
    for resource in resolved.resources:
        lines.extend(
            [
                "",
                f"### {resource.uri}",
                resource.description,
                f"- name: {resource.name}",
                f"- source: `{resource.source}`",
                f"- read only: {resource.read_only}",
            ]
        )

    lines.append("")
    lines.append("## Prompts")
    for prompt in resolved.prompts:
        lines.extend(
            [
                "",
                f"### {prompt.name}",
                prompt.description,
                f"- arguments: {', '.join(prompt.arguments)}",
                f"- source: `{prompt.source}`",
            ]
        )

    lines.extend(["", "## Transports"])
    for transport in resolved.transports:
        lines.extend(
            [
                "",
                f"### {transport.name}",
                transport.purpose,
                "```bash",
                transport.launch_command,
                "```",
            ]
        )
    return "\n".join(lines)


def write_service_mcp_readiness(
    path: str | Path,
    report: ServiceMcpReadinessReport | None = None,
    json_output: bool = False,
) -> None:
    resolved = report or build_service_mcp_readiness()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = (
        json.dumps(service_mcp_readiness_to_dict(resolved), indent=2, ensure_ascii=False)
        if json_output
        else format_service_mcp_readiness(resolved)
    )
    output_path.write_text(text, encoding="utf-8")


def create_fastmcp_server(runtime: ServiceRuntime | None = None) -> Any:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Install MCP extras first: python -m pip install -e '.[mcp]'") from exc

    service = runtime or build_service_runtime()
    state = ServiceSessionState(session_id="mcp-session")
    mcp = FastMCP(SERVER_NAME, json_response=True)

    @mcp.tool()
    def search_literature(query: str, top_k: int) -> dict[str, Any]:
        """Search the configured literature provider for source candidates."""
        return _invoke_for_mcp(service, state, "POST", "/tools/search_literature", {"query": query, "top_k": top_k})

    @mcp.tool()
    def read_experiment_context(project_name: str = "EpiCETRA") -> dict[str, Any]:
        """Load trusted baseline, split, metric, and gate constraints for a project."""
        return _invoke_for_mcp(service, state, "POST", "/tools/read_experiment_context", {"project_name": project_name})

    @mcp.tool()
    def score_transfer_fit() -> dict[str, Any]:
        """Score already retrieved candidates against the loaded experiment context."""
        return _invoke_for_mcp(service, state, "POST", "/tools/score_transfer_fit", {})

    @mcp.tool()
    def draft_experiment_plan() -> dict[str, Any]:
        """Draft an experiment matrix, promotion gate, risks, recommendation, and cited claims."""
        return _invoke_for_mcp(service, state, "POST", "/tools/draft_experiment_plan", {})

    @mcp.tool()
    def create_review_checkpoint() -> dict[str, Any]:
        """Create the human approval checkpoint for the drafted experiment plan."""
        return _invoke_for_mcp(service, state, "POST", "/tools/create_review_checkpoint", {})

    @mcp.resource("bioml://service/interface")
    def service_interface_resource() -> dict[str, Any]:
        """Return the contract-derived MCP tool descriptors and HTTP route specs."""
        return service_interface_to_dict(service.interface)

    @mcp.resource("bioml://service/runtime-demo")
    def service_runtime_demo_resource() -> dict[str, Any]:
        """Return a deterministic service runtime demo trace."""
        return service_runtime_demo_to_dict(run_service_runtime_demo(service))

    @mcp.prompt(title="TracePilot Decision Review")
    def bioml_experiment_review(question: str, focus: str = "approval gate") -> str:
        """Generate a review prompt for a TracePilot planning decision."""
        return (
            "Review this TracePilot planning-agent decision with emphasis on "
            f"{focus}. Question: {question}. Check evidence ids, source refs, "
            "chromosome-aware split gates, and whether promotion requires human approval."
        )

    return mcp


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect or run the optional TracePilot MCP adapter.")
    subparsers = parser.add_subparsers(dest="command")

    report_parser = subparsers.add_parser("report", help="Generate MCP readiness report.")
    report_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    report_parser.add_argument("--out", help="Write the report to this path.")

    serve_parser = subparsers.add_parser("serve", help="Run the optional FastMCP server.")
    serve_parser.add_argument(
        "--transport",
        choices=("stdio", "streamable-http"),
        default="stdio",
        help="MCP transport to use.",
    )

    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument("--out", help="Write the report to this path.")
    args = parser.parse_args()

    if args.command == "serve":
        _serve(transport=args.transport)
        return

    report = build_service_mcp_readiness()
    json_output = bool(getattr(args, "json", False))
    out = getattr(args, "out", None)
    text = (
        json.dumps(service_mcp_readiness_to_dict(report), indent=2, ensure_ascii=False)
        if json_output
        else format_service_mcp_readiness(report)
    )
    if out:
        output_path = Path(out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    else:
        print(text)
    raise SystemExit(0 if report.passed else 1)


def _binding_from_tool(tool: McpToolDescriptor) -> McpToolBindingSpec:
    return McpToolBindingSpec(
        name=tool.name,
        description=tool.description,
        input_schema=tool.input_schema,
        output_type=tool.output_type,
        annotations=tool.annotations,
        state_dependencies=_state_dependencies_for(tool.name),
        approval_policy=tool.approval_policy,
        runtime_path=f"ServiceRuntime.invoke('POST', '/tools/{tool.name}', ...)",
    )


def _resource_specs() -> tuple[McpResourceSpec, ...]:
    return (
        McpResourceSpec(
            uri="bioml://service/interface",
            name="Service Interface",
            description="Contract-derived MCP descriptors and HTTP route specs.",
            source="service_interface.service_interface_to_dict",
            read_only=True,
        ),
        McpResourceSpec(
            uri="bioml://service/runtime-demo",
            name="Service Runtime Demo",
            description="Deterministic trace showing the session-backed route workflow.",
            source="service_runtime.run_service_runtime_demo",
            read_only=True,
        ),
    )


def _prompt_specs() -> tuple[McpPromptSpec, ...]:
    return (
        McpPromptSpec(
            name="bioml_experiment_review",
            description="Review a TracePilot planning decision for evidence, gates, source refs, and approval requirements.",
            arguments=("question", "focus"),
            source="service_mcp.create_fastmcp_server",
        ),
    )


def _transport_specs() -> tuple[McpTransportSpec, ...]:
    return (
        McpTransportSpec(
            name="stdio",
            launch_command="PYTHONPATH=src python -m bioml_agent.service_mcp serve --transport stdio",
            purpose="Local desktop/client integration with a process-launched MCP server.",
        ),
        McpTransportSpec(
            name="streamable-http",
            launch_command="PYTHONPATH=src python -m bioml_agent.service_mcp serve --transport streamable-http",
            purpose="Remote-capable MCP transport served at the SDK default streamable HTTP path.",
        ),
    )


def _state_dependencies_for(tool_name: str) -> tuple[str, ...]:
    mapping = {
        "score_transfer_fit": ("candidates", "context"),
        "draft_experiment_plan": ("evidence", "context"),
        "create_review_checkpoint": ("plan", "evidence", "context"),
    }
    return mapping.get(tool_name, ())


def _dependency_status(name: str, purpose: str) -> McpOptionalDependencyStatus:
    installed = importlib.util.find_spec(name) is not None
    return McpOptionalDependencyStatus(
        name=name,
        installed=installed,
        purpose=purpose,
        install_hint="python -m pip install -e '.[mcp]'",
    )


def _invoke_for_mcp(
    service: ServiceRuntime,
    state: ServiceSessionState,
    method: str,
    path: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    result = service.invoke(method, path, payload, state)
    return {
        "ok": result.ok,
        "status_code": result.status_code,
        "tool_name": result.tool_name,
        "output": to_jsonable(result.output),
        "error": result.error,
        "state_after": list(result.state_after),
        "trace_events_added": result.trace_events_added,
        "approval_required": result.approval_required,
    }


def _serve(transport: str) -> None:
    try:
        mcp = create_fastmcp_server()
    except RuntimeError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(str(exc)) from exc
    mcp.run(transport=transport)


def _binding_to_dict(binding: McpToolBindingSpec) -> dict[str, Any]:
    return {
        "name": binding.name,
        "description": binding.description,
        "input_schema": binding.input_schema,
        "output_type": binding.output_type,
        "annotations": binding.annotations,
        "state_dependencies": list(binding.state_dependencies),
        "approval_policy": binding.approval_policy,
        "runtime_path": binding.runtime_path,
    }


if __name__ == "__main__":
    main()
