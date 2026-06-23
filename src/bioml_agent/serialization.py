from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from .schemas import AgentRun


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def run_to_dict(run: AgentRun) -> dict[str, Any]:
    return to_jsonable(run)


def write_run_json(run: AgentRun, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(run_to_dict(run), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def review_packet_to_dict(run: AgentRun) -> dict[str, Any]:
    data = run_to_dict(run)
    return {
        "question": data["question"],
        "recommendation": data["plan"]["recommendation"],
        "summary": data["plan"]["summary"],
        "review": data["review"],
        "cited_claims": data["plan"]["cited_claims"],
        "promotion_gate": data["plan"]["promotion_gate"],
        "risks": data["plan"]["risks"],
    }


def write_review_packet(run: AgentRun, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(review_packet_to_dict(run), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
