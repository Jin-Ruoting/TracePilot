from __future__ import annotations

import json
import math
import os
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol


ROLES = {
    "research": "Read evidence and propose falsifiable hypotheses or experiment specifications.",
    "coding": "Implement or repair code and prepare a reproducible experiment.",
    "review": "Audit completed experiment artifacts and interpret comparative evidence.",
    "escalate": "Request additional context or coordination for an ambiguous decision.",
}


def validate_probabilities(probabilities: dict[str, float], options: dict[str, str]) -> None:
    if not isinstance(probabilities, dict) or set(probabilities) != set(options):
        raise ValueError("decision output does not cover exactly the legal options")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           or v < 0 or v > 1 for v in probabilities.values()):
        raise ValueError("invalid decision probabilities")
    if abs(sum(probabilities.values()) - 1) > 1e-4:
        raise ValueError("decision probabilities do not sum to one")


@dataclass(frozen=True)
class DecisionScores:
    probabilities: dict[str, float]
    confidence: float
    model: str = ""
    usage: dict[str, Any] = field(default_factory=dict)


class DecisionGateway(Protocol):
    def score(self, state: dict[str, Any], options: dict[str, str]) -> DecisionScores: ...


class SystemOneGateway:
    def __init__(self, endpoint: str, model: str, *, api_key_env: str = "TYPESAFE_API_KEY",
                 timeout: float = 20) -> None:
        self.endpoint, self.model, self.api_key_env, self.timeout = endpoint, model, api_key_env, timeout

    def score(self, state: dict[str, Any], options: dict[str, str]) -> DecisionScores:
        payload = {"model": self.model, "state": state, "questions": {
            "route": {"type": "choice", "instructions": "Which worker should handle this event?",
                      "criteria": options}}}
        headers = {"Content-Type": "application/json"}
        if os.environ.get(self.api_key_env):
            headers["Authorization"] = "Bearer " + os.environ[self.api_key_env]
        request = urllib.request.Request(self.endpoint, data=json.dumps(payload).encode(),
                                         headers=headers, method="POST")
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            raw = response.read(1_000_001)
        if len(raw) > 1_000_000:
            raise ValueError("decision response exceeds limit")
        response = json.loads(raw)
        answer = response["answers"]["route"]
        probabilities = answer["probabilities"]
        validate_probabilities(probabilities, options)
        if answer.get("choice") not in options or probabilities[answer["choice"]] != max(probabilities.values()):
            raise ValueError("choice does not agree with its probability distribution")
        return DecisionScores(probabilities, answer["confidence"], response.get("model", self.model),
                              response.get("usage", {}))


@dataclass(frozen=True)
class RouteDecision:
    role: str
    source: str
    state_version: int
    probabilities: dict[str, float]
    elapsed_seconds: float
    reason: str = ""
    confidence: float | None = None
    model: str = ""
    usage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CascadeRouter:
    def __init__(self, gateway: DecisionGateway | None = None, *, threshold: float = 1.0) -> None:
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be in [0, 1]")
        self.gateway, self.threshold = gateway, threshold

    def route(self, event: dict[str, Any], *, state_version: int) -> RouteDecision:
        start = time.monotonic()
        # Job ownership and known protocol transitions never need an extra model call.
        role = {"project_created": "research", "experiment_spec_ready": "coding",
                "job_succeeded": "review", "job_failed": "coding",
                "job_invalid": "review", "job_timed_out": "coding",
                "job_lost": "review", "job_cancelled": "review"}.get(event.get("kind"))
        if event.get("owner") in ROLES:
            role = event["owner"]
        if role:
            return RouteDecision(role, "rule", state_version, {}, time.monotonic() - start)
        probabilities: dict[str, float] = {}
        confidence = None
        model = ""
        usage: dict[str, Any] = {}
        reason = "no decision model configured"
        if self.gateway:
            try:
                scores = self.gateway.score(dict(event, state_version=state_version), ROLES)
                probabilities, confidence, model, usage = (
                    scores.probabilities, scores.confidence, scores.model, scores.usage)
                validate_probabilities(probabilities, ROLES)
                if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) \
                        or not math.isfinite(confidence) or not 0 <= confidence <= 1:
                    raise ValueError("invalid decision confidence")
                role = max(probabilities, key=probabilities.get)
                if confidence >= self.threshold:
                    return RouteDecision(role, "decision_model", state_version, probabilities,
                                         time.monotonic() - start, confidence=confidence, model=model, usage=usage)
                reason = "below the configured per-workload confidence threshold"
            except Exception as exc:
                probabilities = {}
                confidence = None
                model, usage = "", {}
                reason = type(exc).__name__ + ": decision service failed"
        return RouteDecision("escalate", "fallback", state_version, probabilities,
                             time.monotonic() - start, reason, confidence, model, usage)
