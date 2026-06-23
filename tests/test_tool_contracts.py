from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from bioml_agent.openai_controller import build_openai_tool_specs
from bioml_agent.tool_contracts import (
    build_tool_contracts,
    format_tool_manifest,
    format_validation,
    tool_manifest_to_dict,
    validate_tool_contracts,
    write_tool_manifest,
)
from bioml_agent.tools import build_default_registry


class ToolContractsTest(unittest.TestCase):
    def test_manifest_matches_default_registry(self) -> None:
        registry = build_default_registry()
        validation = validate_tool_contracts(registry)
        self.assertTrue(validation.passed)
        self.assertEqual(validation.errors, ())
        self.assertEqual(set(validation.manifest_tools), set(registry.names))

    def test_openai_tool_specs_are_derived_from_contracts(self) -> None:
        contracts = build_tool_contracts()
        specs = build_openai_tool_specs()

        self.assertEqual([spec["name"] for spec in specs], [contract.name for contract in contracts])
        for spec, contract in zip(specs, contracts, strict=True):
            self.assertEqual(spec["description"], contract.description)
            self.assertEqual(spec["parameters"], contract.model_input_schema)
            self.assertTrue(spec["strict"])
            self.assertFalse(spec["parameters"]["additionalProperties"])

    def test_tool_manifest_is_serializable_and_human_readable(self) -> None:
        data = tool_manifest_to_dict()
        text = format_tool_manifest()

        self.assertEqual(data["tool_count"], 5)
        self.assertIn("search_literature", text)
        self.assertIn("Approval policy", text)
        self.assertTrue(all("registry_input_schema" in item for item in data["tools"]))

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "tool_manifest.json"
            write_tool_manifest(path)
            loaded = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(loaded["tool_count"], 5)

    def test_validation_reports_missing_contract(self) -> None:
        contracts = build_tool_contracts()[1:]
        validation = validate_tool_contracts(build_default_registry(), contracts=contracts)
        self.assertFalse(validation.passed)
        self.assertIn("registered tool has no contract: search_literature", validation.errors)
        self.assertIn("Passed: False", format_validation(validation))

    def test_validation_rejects_loose_model_schema(self) -> None:
        contracts = list(build_tool_contracts())
        broken_schema = dict(contracts[0].model_input_schema)
        broken_schema["additionalProperties"] = True
        contracts[0] = replace(contracts[0], model_input_schema=broken_schema)

        validation = validate_tool_contracts(build_default_registry(), contracts=tuple(contracts))

        self.assertFalse(validation.passed)
        self.assertIn("search_literature: model_input_schema must disable additionalProperties", validation.errors)


if __name__ == "__main__":
    unittest.main()
