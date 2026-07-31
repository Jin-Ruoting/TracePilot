from __future__ import annotations

import time
import unittest

from tracepilot.tooling import (
    EffectClass,
    ToolContract,
    ToolInputError,
    ToolRegistrationError,
    ToolRegistry,
    ToolTimeoutError,
)


def _schema(properties: dict, required: list[str]) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


class TracePilotToolingTest(unittest.TestCase):
    def test_registration_requires_explicit_effect_class(self) -> None:
        registry = ToolRegistry()

        with self.assertRaisesRegex(ToolRegistrationError, "effect_class"):
            registry.register("unsafe", lambda: None)

    def test_contract_call_and_openai_schema(self) -> None:
        registry = ToolRegistry()
        contract = ToolContract(
            name="double",
            description="Double an integer.",
            input_schema=_schema({"value": {"type": "integer"}}, ["value"]),
            output_type="integer",
            effect_class=EffectClass.READ,
            state_dependencies=("ready",),
        )
        registry.register(contract, lambda value: value * 2)

        self.assertEqual(registry.call("double", {"value": 4}, state={"ready": True}), 8)
        self.assertEqual(registry.contract("double"), contract)
        self.assertEqual(registry.openai_specs()[0]["name"], "double")
        with self.assertRaises(ToolInputError):
            registry.call("double", {"value": "4"}, state={"ready": True})
        with self.assertRaisesRegex(ToolInputError, "state dependencies"):
            registry.call("double", {"value": 4}, state={})

    def test_decorator_derives_strict_schema(self) -> None:
        registry = ToolRegistry()

        @registry.register(
            "greet",
            effect_class="read",
            description="Create a greeting.",
        )
        def greet(name):
            return f"hello {name}"

        self.assertEqual(registry.call("greet", name="Ada"), "hello Ada")
        schema = registry.contract("greet").input_schema
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["required"], ["name"])

    def test_future_annotations_are_resolved_for_strict_validation(self) -> None:
        registry = ToolRegistry()

        @registry.register("typed", effect_class="read")
        def typed(count: int, labels: list[str] | None = None) -> int:
            return count + len(labels or [])

        schema = registry.contract("typed").input_schema
        self.assertEqual(schema["properties"]["count"]["type"], "integer")
        self.assertEqual(
            schema["properties"]["labels"]["anyOf"][0]["type"],
            "array",
        )
        with self.assertRaisesRegex(ToolInputError, "count"):
            registry.call("typed", {"count": "2"})

    def test_state_dependencies_are_required_when_state_is_omitted(self) -> None:
        registry = ToolRegistry()
        registry.register(
            ToolContract(
                name="dependent",
                input_schema=_schema({}, []),
                effect_class="read",
                state_dependencies=("indexed",),
            ),
            lambda: True,
        )

        with self.assertRaisesRegex(ToolInputError, "indexed"):
            registry.call("dependent")

    def test_timeout_is_reported(self) -> None:
        registry = ToolRegistry()

        def slow() -> None:
            time.sleep(0.05)

        registry.register(
            "slow",
            slow,
            effect_class="read",
            timeout_seconds=0.005,
        )
        with self.assertRaises(ToolTimeoutError):
            registry.call("slow")

    def test_mutation_does_not_report_timeout_while_handler_continues(self) -> None:
        registry = ToolRegistry()
        effects: list[str] = []

        def slow_mutation() -> str:
            time.sleep(0.02)
            effects.append("finished")
            return "done"

        registry.register(
            "slow_mutation",
            slow_mutation,
            effect_class="mutation",
            timeout_seconds=0.001,
        )

        started = time.monotonic()
        result = registry.call("slow_mutation")

        self.assertEqual(result, "done")
        self.assertEqual(effects, ["finished"])
        self.assertGreaterEqual(time.monotonic() - started, 0.015)


if __name__ == "__main__":
    unittest.main()
