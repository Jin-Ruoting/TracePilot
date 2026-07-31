from __future__ import annotations

import copy
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


class LLMError(RuntimeError):
    pass


class LLMResponseError(LLMError, ValueError):
    pass


@dataclass(frozen=True)
class LLMResponse:
    text: str = ""
    parsed: Any = None
    response_id: str = ""
    model: str = ""
    prompt_version: str = "v1"
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0
    retries: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def content(self) -> str:
        return self.text

    @property
    def data(self) -> Any:
        return self.parsed

    @property
    def usage(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class LLMMetrics:
    calls: int = 0
    retries: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost: float = 0.0

    def record(self, response: LLMResponse) -> None:
        self.calls += 1
        self.retries += response.retries
        self.input_tokens += response.input_tokens
        self.output_tokens += response.output_tokens
        self.estimated_cost += response.estimated_cost


@dataclass(frozen=True)
class LLMCall:
    prompt: str = ""
    instructions: str = ""
    response_schema: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class LLMGateway(Protocol):
    def complete(
        self,
        prompt: str,
        *,
        instructions: str = "",
        response_schema: dict[str, Any] | None = None,
        schema_name: str = "tracepilot_response",
        metadata: Mapping[str, Any] | None = None,
    ) -> LLMResponse: ...


class DeterministicLLM:
    """Scriptable LLM test double with the same interface as the live gateway."""

    def __init__(
        self,
        responses: Sequence[str | Mapping[str, Any]] | str | Mapping[str, Any] | None = None,
        *,
        default_response: str | Mapping[str, Any] | None = None,
        model: str = "deterministic",
        prompt_version: str = "v1",
    ) -> None:
        if responses is None:
            scripted: list[str | Mapping[str, Any]] = []
        elif isinstance(responses, (str, Mapping)):
            scripted = [responses]
        else:
            scripted = list(responses)
        self._responses = copy.deepcopy(scripted)
        self._cursor = 0
        self.default_response = copy.deepcopy(
            {} if default_response is None else default_response
        )
        self.model = model
        self.prompt_version = prompt_version
        self.calls: list[LLMCall] = []
        self.metrics = LLMMetrics()

    def complete(
        self,
        prompt: str,
        *,
        instructions: str = "",
        response_schema: dict[str, Any] | None = None,
        schema_name: str = "tracepilot_response",
        metadata: Mapping[str, Any] | None = None,
    ) -> LLMResponse:
        del schema_name
        call = LLMCall(
            prompt=prompt,
            instructions=instructions,
            response_schema=copy.deepcopy(response_schema),
            metadata=dict(metadata or {}),
        )
        self.calls.append(call)
        if self._cursor < len(self._responses):
            raw = copy.deepcopy(self._responses[self._cursor])
            self._cursor += 1
        else:
            raw = copy.deepcopy(self.default_response)
        parsed: Any = None
        if isinstance(raw, Mapping):
            parsed = dict(raw)
            text = json.dumps(parsed, ensure_ascii=False, sort_keys=True)
        else:
            text = str(raw)
            if response_schema is not None:
                parsed = _parse_json(text)
        if response_schema is not None:
            if parsed is None:
                parsed = _parse_json(text)
            _validate_schema(parsed, response_schema)
        input_tokens = _estimate_tokens(f"{instructions}\n{prompt}")
        output_tokens = _estimate_tokens(text)
        response = LLMResponse(
            text=text,
            parsed=parsed,
            response_id=f"deterministic-{len(self.calls)}",
            model=self.model,
            prompt_version=self.prompt_version,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            metadata=dict(metadata or {}),
        )
        self.metrics.record(response)
        return response

    def generate(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> LLMResponse:
        return self.complete(prompt, **kwargs)

    def generate_structured(
        self,
        prompt: str,
        response_schema: dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        return self.complete(
            prompt,
            response_schema=response_schema,
            **kwargs,
        ).parsed

    def reset(self) -> None:
        self._cursor = 0
        self.calls.clear()
        self.metrics = LLMMetrics()


class OpenAILLMGateway:
    """Optional OpenAI Responses API adapter.

    The SDK is imported lazily so deterministic and offline installations keep
    working without the optional ``llm`` dependency.
    """

    def __init__(
        self,
        model: str,
        *,
        client: Any | None = None,
        prompt_version: str = "v1",
        max_retries: int = 2,
        retry_delay_seconds: float = 0.0,
        input_cost_per_million: float = 0.0,
        output_cost_per_million: float = 0.0,
    ) -> None:
        if not model:
            raise ValueError("model is required")
        if max_retries < 0:
            raise ValueError("max_retries must not be negative")
        self.model = model
        self._client = client
        self.prompt_version = prompt_version
        self.max_retries = max_retries
        self.retry_delay_seconds = max(0.0, retry_delay_seconds)
        self.input_cost_per_million = max(0.0, input_cost_per_million)
        self.output_cost_per_million = max(0.0, output_cost_per_million)
        self.metrics = LLMMetrics()

    def complete(
        self,
        prompt: str,
        *,
        instructions: str = "",
        response_schema: dict[str, Any] | None = None,
        schema_name: str = "tracepilot_response",
        metadata: Mapping[str, Any] | None = None,
    ) -> LLMResponse:
        client = self._client or self._build_client()
        request: dict[str, Any] = {
            "model": self.model,
            "input": prompt,
            "metadata": {
                **{str(key): str(value) for key, value in (metadata or {}).items()},
                "prompt_version": self.prompt_version,
            },
        }
        if instructions:
            request["instructions"] = instructions
        if response_schema is not None:
            request["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "schema": copy.deepcopy(response_schema),
                    "strict": True,
                }
            }

        response: Any = None
        last_error: Exception | None = None
        retries = 0
        for attempt in range(self.max_retries + 1):
            try:
                response = client.responses.create(**request)
                retries = attempt
                break
            except Exception as exc:  # Provider-specific exceptions stay optional.
                last_error = exc
                if attempt >= self.max_retries:
                    raise LLMError(
                        f"OpenAI Responses API failed after {attempt + 1} attempts: {exc}"
                    ) from exc
                if self.retry_delay_seconds:
                    time.sleep(self.retry_delay_seconds)
        if response is None:
            raise LLMError(f"OpenAI Responses API returned no response: {last_error}")

        text = _extract_output_text(response)
        parsed = _field(response, "output_parsed")
        if response_schema is not None:
            if parsed is None:
                parsed = _parse_json(text)
            _validate_schema(parsed, response_schema)
        usage = _extract_usage(response)
        estimated_cost = (
            usage["input_tokens"] * self.input_cost_per_million
            + usage["output_tokens"] * self.output_cost_per_million
        ) / 1_000_000
        result = LLMResponse(
            text=text,
            parsed=parsed,
            response_id=str(_field(response, "id") or ""),
            model=str(_field(response, "model") or self.model),
            prompt_version=self.prompt_version,
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            total_tokens=usage["total_tokens"],
            estimated_cost=estimated_cost,
            retries=retries,
            metadata=dict(metadata or {}),
        )
        self.metrics.record(result)
        return result

    def generate(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> LLMResponse:
        return self.complete(prompt, **kwargs)

    def generate_structured(
        self,
        prompt: str,
        response_schema: dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        return self.complete(
            prompt,
            response_schema=response_schema,
            **kwargs,
        ).parsed

    @staticmethod
    def _build_client() -> Any:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise LLMError(
                "OpenAI gateway requires the optional OpenAI SDK; install the "
                "project with the 'llm' extra"
            ) from exc
        return OpenAI()


def _extract_output_text(response: Any) -> str:
    direct = _field(response, "output_text")
    if direct is not None:
        return str(direct)
    pieces: list[str] = []
    for item in _field(response, "output") or ():
        content = _field(item, "content")
        if content is None:
            text = _field(item, "text")
            if text is not None:
                pieces.append(str(text))
            continue
        for part in content:
            text = _field(part, "text")
            if text is not None:
                pieces.append(str(text))
    return "".join(pieces)


def _extract_usage(response: Any) -> dict[str, int]:
    usage = _field(response, "usage") or {}
    input_tokens = int(_field(usage, "input_tokens") or 0)
    output_tokens = int(_field(usage, "output_tokens") or 0)
    total_tokens = int(
        _field(usage, "total_tokens") or input_tokens + output_tokens
    )
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _parse_json(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMResponseError(
            f"model response is not valid JSON: {exc.msg}"
        ) from exc


def _validate_schema(value: Any, schema: Mapping[str, Any], path: str = "$") -> None:
    if "anyOf" in schema:
        errors: list[Exception] = []
        for option in schema["anyOf"]:
            try:
                _validate_schema(value, option, path)
                return
            except LLMResponseError as exc:
                errors.append(exc)
        raise LLMResponseError(f"{path} does not match any allowed schema")

    expected = schema.get("type")
    checks = {
        "object": lambda item: isinstance(item, Mapping),
        "array": lambda item: isinstance(item, list),
        "string": lambda item: isinstance(item, str),
        "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
        "number": lambda item: isinstance(item, (int, float))
        and not isinstance(item, bool),
        "boolean": lambda item: isinstance(item, bool),
        "null": lambda item: item is None,
    }
    if expected in checks and not checks[expected](value):
        raise LLMResponseError(f"{path} must be {expected}")
    if expected == "object":
        required = schema.get("required", ())
        missing = [name for name in required if name not in value]
        if missing:
            raise LLMResponseError(
                f"{path} is missing required fields: {', '.join(missing)}"
            )
        if schema.get("additionalProperties") is False:
            extra = sorted(set(value) - set(schema.get("properties", {})))
            if extra:
                raise LLMResponseError(
                    f"{path} has unexpected fields: {', '.join(extra)}"
                )
        for name, child_schema in schema.get("properties", {}).items():
            if name in value:
                _validate_schema(value[name], child_schema, f"{path}.{name}")
    if expected == "array" and "items" in schema:
        for index, item in enumerate(value):
            _validate_schema(item, schema["items"], f"{path}[{index}]")
    if "enum" in schema and value not in schema["enum"]:
        raise LLMResponseError(f"{path} is not an allowed value")


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


__all__ = [
    "DeterministicLLM",
    "LLMCall",
    "LLMError",
    "LLMGateway",
    "LLMMetrics",
    "LLMResponse",
    "LLMResponseError",
    "OpenAILLMGateway",
]
