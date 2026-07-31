from __future__ import annotations

import dataclasses
import hashlib
import json
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


def to_jsonable(value: Any) -> Any:
    """Convert common model/value types into stable JSON-compatible data."""
    if isinstance(value, Enum):
        return to_jsonable(value.value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return {"__bytes__": value.hex()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if dataclasses.is_dataclass(value):
        return to_jsonable(dataclasses.asdict(value))
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return to_jsonable(model_dump(mode="json"))
        except TypeError:
            return to_jsonable(model_dump())
    legacy_dict = getattr(value, "dict", None)
    if callable(legacy_dict):
        return to_jsonable(legacy_dict())
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [to_jsonable(item) for item in value]
        return sorted(normalized, key=canonical_json)
    if hasattr(value, "__dict__"):
        return {
            str(key): to_jsonable(item)
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        }
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")


def canonical_json(value: Any) -> str:
    return json.dumps(
        to_jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def content_fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def source_fingerprint(source: Any) -> str:
    if source is None or source == "":
        return ""
    return content_fingerprint({"source": source})


def workspace_fingerprint(workspace: str | Path, metadata: Any = None) -> str:
    if isinstance(workspace, Path):
        normalized = str(workspace.expanduser().resolve(strict=False))
    else:
        normalized = str(workspace).strip()
    if not normalized:
        raise ValueError("workspace must be non-empty")
    return content_fingerprint({"workspace": normalized, "metadata": metadata})


# Verb-first aliases read naturally at call sites.
fingerprint_content = content_fingerprint
fingerprint_source = source_fingerprint
fingerprint_workspace = workspace_fingerprint
