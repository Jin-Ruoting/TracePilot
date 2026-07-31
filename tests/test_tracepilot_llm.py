from __future__ import annotations

import unittest
from types import SimpleNamespace

from tracepilot.llm import (
    DeterministicLLM,
    LLMResponseError,
    OpenAILLMGateway,
)


SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


class _FakeResponses:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.attempts = 0

    def create(self, **kwargs):
        self.attempts += 1
        self.calls.append(kwargs)
        if self.attempts == 1:
            raise RuntimeError("temporary")
        return SimpleNamespace(
            id="response-1",
            model="test-model",
            output_text='{"answer":"ok"}',
            usage=SimpleNamespace(
                input_tokens=10,
                output_tokens=5,
                total_tokens=15,
            ),
        )


class TracePilotLLMTest(unittest.TestCase):
    def test_deterministic_structured_generation_and_metrics(self) -> None:
        llm = DeterministicLLM([{"answer": "stable"}], prompt_version="plan-v1")

        result = llm.generate_structured("make a plan", SCHEMA)

        self.assertEqual(result, {"answer": "stable"})
        self.assertEqual(llm.metrics.calls, 1)
        self.assertEqual(llm.calls[0].prompt, "make a plan")

    def test_deterministic_response_rejects_schema_violation(self) -> None:
        llm = DeterministicLLM([{"wrong": "field"}])

        with self.assertRaises(LLMResponseError):
            llm.generate_structured("make a plan", SCHEMA)

    def test_openai_gateway_uses_responses_api_and_retries(self) -> None:
        responses = _FakeResponses()
        client = SimpleNamespace(responses=responses)
        gateway = OpenAILLMGateway(
            "test-model",
            client=client,
            max_retries=1,
            prompt_version="decision-v2",
            input_cost_per_million=1.0,
            output_cost_per_million=2.0,
        )

        result = gateway.complete(
            "question",
            instructions="ground every claim",
            response_schema=SCHEMA,
        )

        self.assertEqual(result.parsed, {"answer": "ok"})
        self.assertEqual(result.retries, 1)
        self.assertEqual(result.total_tokens, 15)
        self.assertGreater(result.estimated_cost, 0)
        request = responses.calls[-1]
        self.assertEqual(request["metadata"]["prompt_version"], "decision-v2")
        self.assertEqual(
            request["text"]["format"]["schema"],
            SCHEMA,
        )


if __name__ == "__main__":
    unittest.main()
