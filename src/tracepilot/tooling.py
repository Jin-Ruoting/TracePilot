from __future__ import annotations

import asyncio
import copy
import inspect
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from enum import Enum
from types import UnionType
from typing import Any, get_args, get_origin, get_type_hints

ToolFn = Callable[..., Any]


class EffectClass(str, Enum):
    READ = "read"
    INTERNAL_WRITE = "internal_write"
    MUTATION = "mutation"


class ToolError(RuntimeError):
    pass


class ToolRegistrationError(ToolError, ValueError):
    pass


class ToolInputError(ToolError, ValueError):
    pass


class ToolTimeoutError(ToolError, TimeoutError):
    pass


@dataclass(frozen=True)
class ToolContract:
    name: str = ""
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_type: str = "Any"
    effect_class: EffectClass | str | None = None
    timeout_seconds: float = 30.0
    idempotent: bool = True
    permissions: tuple[str, ...] = ()
    state_dependencies: tuple[str, ...] = ()
    failure_modes: tuple[str, ...] = ()
    trace_required: bool = True

    def __post_init__(self) -> None:
        if self.effect_class is not None:
            try:
                effect = EffectClass(self.effect_class)
            except ValueError as exc:
                raise ToolRegistrationError(
                    f"{self.name or '<unnamed>'}: invalid effect_class '{self.effect_class}'"
                ) from exc
            object.__setattr__(self, "effect_class", effect)
        object.__setattr__(self, "input_schema", copy.deepcopy(self.input_schema))
        object.__setattr__(self, "permissions", tuple(self.permissions))
        object.__setattr__(self, "state_dependencies", tuple(self.state_dependencies))
        object.__setattr__(self, "failure_modes", tuple(self.failure_modes))

    def validate(self) -> None:
        errors: list[str] = []
        if not self.name.strip():
            errors.append("tool contract name is required")
        if self.effect_class is None:
            errors.append(f"{self.name or '<unnamed>'}: effect_class must be declared")
        if self.timeout_seconds <= 0:
            errors.append(f"{self.name or '<unnamed>'}: timeout_seconds must be positive")
        schema = self.input_schema
        if schema:
            if schema.get("type") != "object":
                errors.append(f"{self.name}: input_schema must be an object schema")
            if schema.get("additionalProperties") is not False:
                errors.append(f"{self.name}: input_schema must disable additionalProperties")
            if not isinstance(schema.get("required"), list):
                errors.append(f"{self.name}: input_schema must include a required list")
        if errors:
            raise ToolRegistrationError("; ".join(errors))

    def openai_spec(self) -> dict[str, Any]:
        self.validate()
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": copy.deepcopy(self.input_schema),
            "strict": True,
        }


@dataclass(frozen=True)
class _RegisteredTool:
    contract: ToolContract
    fn: ToolFn


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, _RegisteredTool] = {}

    def register(
        self,
        name_or_contract: str | ToolContract | None = None,
        fn: ToolFn | None = None,
        *,
        contract: ToolContract | None = None,
        effect_class: EffectClass | str | None = None,
        description: str = "",
        input_schema: dict[str, Any] | None = None,
        output_type: str = "Any",
        timeout_seconds: float = 30.0,
        idempotent: bool = True,
        permissions: tuple[str, ...] = (),
        state_dependencies: tuple[str, ...] = (),
        failure_modes: tuple[str, ...] = (),
    ) -> ToolFn | Callable[[ToolFn], ToolFn]:
        resolved_contract = _resolve_contract(
            name_or_contract,
            contract=contract,
            fn=fn,
            effect_class=effect_class,
            description=description,
            input_schema=input_schema,
            output_type=output_type,
            timeout_seconds=timeout_seconds,
            idempotent=idempotent,
            permissions=permissions,
            state_dependencies=state_dependencies,
            failure_modes=failure_modes,
        )

        def decorator(handler: ToolFn) -> ToolFn:
            final_contract = resolved_contract
            if not final_contract.input_schema:
                final_contract = _contract_with_derived_schema(final_contract, handler)
            final_contract.validate()
            if final_contract.name in self._tools:
                raise ToolRegistrationError(
                    f"tool already registered: {final_contract.name}"
                )
            if not callable(handler):
                raise ToolRegistrationError(
                    f"tool handler must be callable: {final_contract.name}"
                )
            self._tools[final_contract.name] = _RegisteredTool(final_contract, handler)
            return handler

        if fn is None:
            return decorator
        return decorator(fn)

    def contract(self, name: str) -> ToolContract:
        try:
            return self._tools[name].contract
        except KeyError as exc:
            raise KeyError(f"unknown tool: {name}") from exc

    def call(
        self,
        tool_name: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        state: Mapping[str, Any] | None = None,
        timeout_seconds: float | None = None,
        **kwargs: Any,
    ) -> Any:
        try:
            registered = self._tools[tool_name]
        except KeyError as exc:
            raise KeyError(f"unknown tool: {tool_name}") from exc
        payload = dict(arguments or {})
        overlap = set(payload).intersection(kwargs)
        if overlap:
            raise ToolInputError(
                f"{tool_name}: duplicate arguments: {', '.join(sorted(overlap))}"
            )
        payload.update(kwargs)
        _validate_state_dependencies(registered.contract, state)
        _validate_input(registered.contract, payload)
        timeout = timeout_seconds or registered.contract.timeout_seconds
        if registered.contract.effect_class == EffectClass.MUTATION:
            # A Python thread cannot be force-cancelled safely. Returning a
            # timeout while a mutation continues would create an uncertain
            # side-effect state, so mutation handlers must enforce their own
            # cancellable timeout and are awaited here.
            return _invoke(registered.fn, payload)
        return _call_with_timeout(registered.fn, payload, timeout, tool_name)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))

    @property
    def contracts(self) -> tuple[ToolContract, ...]:
        return tuple(self._tools[name].contract for name in self.names)

    def openai_specs(self) -> list[dict[str, Any]]:
        return [item.openai_spec() for item in self.contracts]

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)


def _resolve_contract(
    name_or_contract: str | ToolContract | None,
    *,
    contract: ToolContract | None,
    fn: ToolFn | None,
    effect_class: EffectClass | str | None,
    description: str,
    input_schema: dict[str, Any] | None,
    output_type: str,
    timeout_seconds: float,
    idempotent: bool,
    permissions: tuple[str, ...],
    state_dependencies: tuple[str, ...],
    failure_modes: tuple[str, ...],
) -> ToolContract:
    if isinstance(name_or_contract, ToolContract):
        if contract is not None:
            raise ToolRegistrationError("contract was provided twice")
        if effect_class is not None:
            raise ToolRegistrationError(
                "effect_class must be declared on the provided ToolContract"
            )
        return name_or_contract
    if contract is not None:
        if name_or_contract is not None and name_or_contract != contract.name:
            raise ToolRegistrationError(
                f"tool name '{name_or_contract}' does not match contract '{contract.name}'"
            )
        return contract
    if not isinstance(name_or_contract, str) or not name_or_contract:
        raise ToolRegistrationError("register requires a tool name or ToolContract")
    schema = copy.deepcopy(input_schema)
    if schema is None and fn is not None:
        schema = derive_input_schema(fn)
    return ToolContract(
        name=name_or_contract,
        description=description,
        input_schema=schema or {},
        output_type=output_type,
        effect_class=effect_class,
        timeout_seconds=timeout_seconds,
        idempotent=idempotent,
        permissions=permissions,
        state_dependencies=state_dependencies,
        failure_modes=failure_modes,
    )


def _contract_with_derived_schema(
    contract: ToolContract,
    fn: ToolFn,
) -> ToolContract:
    return ToolContract(
        name=contract.name,
        description=contract.description,
        input_schema=derive_input_schema(fn),
        output_type=contract.output_type,
        effect_class=contract.effect_class,
        timeout_seconds=contract.timeout_seconds,
        idempotent=contract.idempotent,
        permissions=contract.permissions,
        state_dependencies=contract.state_dependencies,
        failure_modes=contract.failure_modes,
        trace_required=contract.trace_required,
    )


def derive_input_schema(fn: ToolFn) -> dict[str, Any]:
    signature = inspect.signature(fn)
    try:
        type_hints = get_type_hints(fn)
    except (NameError, TypeError):
        type_hints = {}
    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, parameter in signature.parameters.items():
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        properties[name] = _annotation_schema(
            type_hints.get(name, parameter.annotation)
        )
        if parameter.default is inspect.Parameter.empty:
            required.append(name)
        else:
            properties[name]["default"] = parameter.default
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _annotation_schema(annotation: Any) -> dict[str, Any]:
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {}
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in (list, tuple, set, frozenset):
        return {
            "type": "array",
            "items": _annotation_schema(args[0]) if args else {},
        }
    if origin in (dict, Mapping):
        return {"type": "object"}
    if origin in (UnionType, getattr(__import__("typing"), "Union")):
        schemas = [_annotation_schema(item) for item in args]
        return {"anyOf": schemas}
    primitive = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        type(None): "null",
    }.get(annotation)
    return {"type": primitive} if primitive else {}


def _validate_state_dependencies(
    contract: ToolContract,
    state: Mapping[str, Any] | None,
) -> None:
    if not contract.state_dependencies:
        return
    state = state or {}
    missing = [
        dependency
        for dependency in contract.state_dependencies
        if dependency not in state or state[dependency] is None
    ]
    if missing:
        raise ToolInputError(
            f"{contract.name}: missing state dependencies: {', '.join(missing)}"
        )


def _validate_input(contract: ToolContract, payload: dict[str, Any]) -> None:
    schema = contract.input_schema
    if not schema:
        return
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    missing = [name for name in required if name not in payload]
    if missing:
        raise ToolInputError(
            f"{contract.name}: missing required arguments: {', '.join(missing)}"
        )
    if schema.get("additionalProperties") is False:
        extra = sorted(set(payload) - set(properties))
        if extra:
            raise ToolInputError(
                f"{contract.name}: unexpected arguments: {', '.join(extra)}"
            )
    for name, value in payload.items():
        item_schema = properties.get(name)
        if item_schema:
            _validate_value(contract.name, name, value, item_schema)


def _validate_value(
    tool_name: str,
    argument_name: str,
    value: Any,
    schema: dict[str, Any],
) -> None:
    if "anyOf" in schema:
        if any(_value_matches(value, item) for item in schema["anyOf"]):
            return
        raise ToolInputError(f"{tool_name}: invalid value for '{argument_name}'")
    if not _value_matches(value, schema):
        expected = schema.get("type", "declared schema")
        raise ToolInputError(
            f"{tool_name}: argument '{argument_name}' must match {expected}"
        )
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise ToolInputError(
                f"{tool_name}: argument '{argument_name}' is below minimum"
            )
        if "maximum" in schema and value > schema["maximum"]:
            raise ToolInputError(
                f"{tool_name}: argument '{argument_name}' exceeds maximum"
            )
    if "enum" in schema and value not in schema["enum"]:
        raise ToolInputError(
            f"{tool_name}: argument '{argument_name}' is not an allowed value"
        )


def _value_matches(value: Any, schema: dict[str, Any]) -> bool:
    expected = schema.get("type")
    if expected is None:
        return True
    checks = {
        "string": lambda item: isinstance(item, str),
        "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
        "number": lambda item: isinstance(item, (int, float))
        and not isinstance(item, bool),
        "boolean": lambda item: isinstance(item, bool),
        "array": lambda item: isinstance(item, (list, tuple)),
        "object": lambda item: isinstance(item, Mapping),
        "null": lambda item: item is None,
    }
    check = checks.get(expected)
    return True if check is None else check(value)


def _call_with_timeout(
    fn: ToolFn,
    payload: dict[str, Any],
    timeout_seconds: float,
    name: str,
) -> Any:
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"tracepilot-{name}")
    future = executor.submit(_invoke, fn, payload)
    try:
        return future.result(timeout=timeout_seconds)
    except FutureTimeoutError as exc:
        future.cancel()
        raise ToolTimeoutError(
            f"tool '{name}' exceeded timeout of {timeout_seconds:g}s"
        ) from exc
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _invoke(fn: ToolFn, payload: dict[str, Any]) -> Any:
    value = fn(**payload)
    if not inspect.isawaitable(value):
        return value
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(value)
    raise ToolError(
        "async tool returned an awaitable while an event loop is already running; "
        "use a synchronous adapter"
    )


__all__ = [
    "EffectClass",
    "ToolContract",
    "ToolError",
    "ToolFn",
    "ToolInputError",
    "ToolRegistrationError",
    "ToolRegistry",
    "ToolTimeoutError",
    "derive_input_schema",
]
