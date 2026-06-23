from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .tools import ToolRegistry, build_default_registry


@dataclass(frozen=True)
class ToolContract:
    name: str
    description: str
    registry_input_schema: dict[str, Any]
    model_input_schema: dict[str, Any]
    output_type: str
    state_dependencies: tuple[str, ...]
    failure_modes: tuple[str, ...]
    approval_policy: str
    trace_required: bool = True


@dataclass(frozen=True)
class ToolContractValidation:
    passed: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    registered_tools: tuple[str, ...]
    manifest_tools: tuple[str, ...]


def build_tool_contracts() -> tuple[ToolContract, ...]:
    empty_model_schema = _object_schema({}, ())
    candidates_schema = _array_schema(_ref("PaperCandidate"))
    evidence_schema = _array_schema(_ref("EvidenceItem"))
    context_schema = _ref("ExperimentContext")
    plan_schema = _ref("ExperimentPlan")
    return (
        ToolContract(
            name="search_literature",
            description="Search the configured literature provider for source candidates.",
            registry_input_schema=_object_schema(
                {
                    "query": {"type": "string", "description": "Research question or method keyword query."},
                    "top_k": {
                        "type": "integer",
                        "description": "Maximum number of candidates to return.",
                        "minimum": 1,
                        "maximum": 10,
                    },
                },
                ("query", "top_k"),
            ),
            model_input_schema=_object_schema(
                {
                    "query": {"type": "string", "description": "Research question or method keyword query."},
                    "top_k": {
                        "type": "integer",
                        "description": "Maximum number of candidates to return.",
                        "minimum": 1,
                        "maximum": 10,
                    },
                },
                ("query", "top_k"),
            ),
            output_type="list[PaperCandidate]",
            state_dependencies=(),
            failure_modes=("no candidates", "provider timeout", "candidate source metadata missing", "source poisoning"),
            approval_policy="no approval required; retrieval is read-only",
        ),
        ToolContract(
            name="read_experiment_context",
            description="Load trusted baseline, split, metric, and gate constraints for a project.",
            registry_input_schema=_object_schema(
                {
                    "project_name": {
                        "type": "string",
                        "description": "Project context name. Use EpiCETRA unless another project is specified.",
                    }
                },
                ("project_name",),
            ),
            model_input_schema=_object_schema(
                {
                    "project_name": {
                        "type": "string",
                        "description": "Project context name. Use EpiCETRA unless another project is specified.",
                    }
                },
                ("project_name",),
            ),
            output_type="ExperimentContext",
            state_dependencies=(),
            failure_modes=("context file missing", "unknown project", "trusted baseline missing", "split unclear"),
            approval_policy="no approval required; context read is read-only",
        ),
        ToolContract(
            name="score_transfer_fit",
            description="Score already retrieved candidates against the loaded experiment context.",
            registry_input_schema=_object_schema(
                {"candidates": candidates_schema, "context": context_schema},
                ("candidates", "context"),
            ),
            model_input_schema=empty_model_schema,
            output_type="list[EvidenceItem]",
            state_dependencies=("candidates", "context"),
            failure_modes=("task mismatch", "evidence too weak", "source metadata cannot be propagated", "source poisoning"),
            approval_policy="no approval required; scoring is an internal read-only transformation",
        ),
        ToolContract(
            name="draft_experiment_plan",
            description="Draft an experiment matrix, promotion gate, risks, recommendation, and cited claims.",
            registry_input_schema=_object_schema(
                {"evidence": evidence_schema, "context": context_schema},
                ("evidence", "context"),
            ),
            model_input_schema=empty_model_schema,
            output_type="ExperimentPlan",
            state_dependencies=("evidence", "context"),
            failure_modes=("missing promotion gate", "unsupported recommendation", "ablation matrix incomplete"),
            approval_policy="proposal only; execution still requires review checkpoint",
        ),
        ToolContract(
            name="create_review_checkpoint",
            description="Create the human approval checkpoint for the drafted experiment plan.",
            registry_input_schema=_object_schema(
                {"plan": plan_schema, "evidence": evidence_schema, "context": context_schema},
                ("plan", "evidence", "context"),
            ),
            model_input_schema=empty_model_schema,
            output_type="ReviewCheckpoint",
            state_dependencies=("plan", "evidence", "context"),
            failure_modes=("promotion lacks approval", "review actions underspecified"),
            approval_policy="promotion must produce requires_approval before any experiment execution",
        ),
    )


def build_openai_tool_specs_from_contracts(
    contracts: tuple[ToolContract, ...] | None = None,
) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "name": contract.name,
            "description": contract.description,
            "parameters": copy.deepcopy(contract.model_input_schema),
            "strict": True,
        }
        for contract in (contracts or build_tool_contracts())
    ]


def tool_manifest_to_dict(contracts: tuple[ToolContract, ...] | None = None) -> dict[str, Any]:
    return {
        "version": 1,
        "tool_count": len(contracts or build_tool_contracts()),
        "tools": [_contract_to_dict(contract) for contract in (contracts or build_tool_contracts())],
    }


def validate_tool_contracts(
    registry: ToolRegistry | None = None,
    contracts: tuple[ToolContract, ...] | None = None,
) -> ToolContractValidation:
    registry = registry or build_default_registry()
    contracts = contracts or build_tool_contracts()
    registered = tuple(registry.names)
    manifest = tuple(contract.name for contract in contracts)
    errors: list[str] = []
    warnings: list[str] = []

    duplicate_names = sorted({name for name in manifest if manifest.count(name) > 1})
    for name in duplicate_names:
        errors.append(f"duplicate contract name: {name}")
    missing = sorted(set(manifest) - set(registered))
    extra = sorted(set(registered) - set(manifest))
    for name in missing:
        errors.append(f"contract has no registered tool: {name}")
    for name in extra:
        errors.append(f"registered tool has no contract: {name}")

    for contract in contracts:
        errors.extend(_contract_errors(contract))
        if contract.name == "create_review_checkpoint" and "requires_approval" not in contract.approval_policy:
            warnings.append("create_review_checkpoint approval policy should mention requires_approval")

    return ToolContractValidation(
        passed=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        registered_tools=registered,
        manifest_tools=manifest,
    )


def write_tool_manifest(path: str | Path, contracts: tuple[ToolContract, ...] | None = None) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(tool_manifest_to_dict(contracts), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def format_tool_manifest(contracts: tuple[ToolContract, ...] | None = None) -> str:
    lines = ["# TracePilot Tool Manifest"]
    for contract in contracts or build_tool_contracts():
        lines.extend(
            [
                "",
                f"## {contract.name}",
                contract.description,
                f"- Output: {contract.output_type}",
                f"- State dependencies: {', '.join(contract.state_dependencies) if contract.state_dependencies else '(none)'}",
                f"- Approval policy: {contract.approval_policy}",
                f"- Failure modes: {', '.join(contract.failure_modes)}",
            ]
        )
    return "\n".join(lines)


def format_validation(validation: ToolContractValidation) -> str:
    lines = [
        "# Tool Contract Validation",
        f"Passed: {validation.passed}",
        f"Registered tools: {', '.join(validation.registered_tools)}",
        f"Manifest tools: {', '.join(validation.manifest_tools)}",
    ]
    if validation.errors:
        lines.append("Errors:")
        lines.extend(f"- {item}" for item in validation.errors)
    if validation.warnings:
        lines.append("Warnings:")
        lines.extend(f"- {item}" for item in validation.warnings)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect or validate TracePilot tool contracts.")
    parser.add_argument("--json", action="store_true", help="Emit tool manifest JSON.")
    parser.add_argument("--validate", action="store_true", help="Validate manifest names and schemas against registry.")
    parser.add_argument("--out", help="Write JSON manifest to this path.")
    args = parser.parse_args()

    if args.out:
        write_tool_manifest(args.out)
    if args.validate:
        validation = validate_tool_contracts()
        if args.json:
            print(json.dumps(_validation_to_dict(validation), indent=2, ensure_ascii=False))
        else:
            print(format_validation(validation))
        raise SystemExit(0 if validation.passed else 1)
    if args.json:
        print(json.dumps(tool_manifest_to_dict(), indent=2, ensure_ascii=False))
    else:
        print(format_tool_manifest())


def _contract_to_dict(contract: ToolContract) -> dict[str, Any]:
    return {
        "name": contract.name,
        "description": contract.description,
        "registry_input_schema": copy.deepcopy(contract.registry_input_schema),
        "model_input_schema": copy.deepcopy(contract.model_input_schema),
        "output_type": contract.output_type,
        "state_dependencies": list(contract.state_dependencies),
        "failure_modes": list(contract.failure_modes),
        "approval_policy": contract.approval_policy,
        "trace_required": contract.trace_required,
    }


def _validation_to_dict(validation: ToolContractValidation) -> dict[str, Any]:
    return {
        "passed": validation.passed,
        "errors": list(validation.errors),
        "warnings": list(validation.warnings),
        "registered_tools": list(validation.registered_tools),
        "manifest_tools": list(validation.manifest_tools),
    }


def _contract_errors(contract: ToolContract) -> list[str]:
    errors: list[str] = []
    if not contract.name:
        errors.append("contract missing name")
    if not contract.description:
        errors.append(f"{contract.name}: missing description")
    for field_name, schema in (
        ("registry_input_schema", contract.registry_input_schema),
        ("model_input_schema", contract.model_input_schema),
    ):
        if schema.get("type") != "object":
            errors.append(f"{contract.name}: {field_name} must be object schema")
        if schema.get("additionalProperties") is not False:
            errors.append(f"{contract.name}: {field_name} must disable additionalProperties")
        required = schema.get("required")
        if not isinstance(required, list):
            errors.append(f"{contract.name}: {field_name} must include required list")
    if not contract.output_type:
        errors.append(f"{contract.name}: missing output_type")
    if contract.trace_required is not True:
        errors.append(f"{contract.name}: trace_required must be true")
    return errors


def _object_schema(properties: dict[str, Any], required: tuple[str, ...]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": copy.deepcopy(properties),
        "required": list(required),
        "additionalProperties": False,
    }


def _array_schema(items: dict[str, Any]) -> dict[str, Any]:
    return {"type": "array", "items": copy.deepcopy(items)}


def _ref(name: str) -> dict[str, Any]:
    return {"type": "object", "x-python-type": name}


if __name__ == "__main__":
    main()
