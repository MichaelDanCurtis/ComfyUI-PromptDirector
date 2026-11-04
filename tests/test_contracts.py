import json
import unittest
from pathlib import Path

from prompt_director.contracts import (
    CONTRACT_NAMES,
    ContractValidationError,
    load_contract_schema,
    validate_contract,
)
from prompt_director.manifest import module_manifest


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "contracts"


class ContractTests(unittest.TestCase):
    def test_every_published_contract_has_a_schema_and_valid_golden_fixture(self):
        for contract_name in CONTRACT_NAMES:
            with self.subTest(contract_name=contract_name):
                schema = load_contract_schema(contract_name)
                fixture = json.loads((FIXTURES / f"{contract_name}.json").read_text(encoding="utf-8"))

                self.assertEqual(schema["type"], "object")
                self.assertTrue(schema["$id"].endswith(f"{contract_name}.schema.json"))
                self.assertEqual(validate_contract(contract_name, fixture), fixture)

    def test_validation_preserves_unknown_additive_fields(self):
        fixture = json.loads((FIXTURES / "model_context.v1.json").read_text(encoding="utf-8"))

        validated = validate_contract("model_context.v1", fixture)

        self.assertEqual(validated["future_additive_field"], {"preserved": True})

    def test_validation_rejects_missing_fields_and_unknown_major_versions(self):
        with self.assertRaisesRegex(ContractValidationError, "missing required field 'model'"):
            validate_contract(
                "model_context.v1",
                {
                    "schema_version": "1.0",
                    "target_model": "qwen_edit",
                    "loras": [],
                    "required_trigger_tokens": [],
                    "warnings": [],
                },
            )
        with self.assertRaisesRegex(ContractValidationError, "major version 1"):
            validate_contract(
                "diagnostic.v1",
                {
                    "schema_version": "2.0",
                    "code": "future",
                    "severity": "info",
                    "message": "future",
                    "advisory": True,
                },
            )

    def test_generated_manifest_is_valid_and_declares_safety_behavior(self):
        manifest = module_manifest()

        validate_contract("module_manifest.v1", manifest)
        self.assertIn("PromptDirectorAuto", manifest["node_types"])
        self.assertIn("structured_diagnostics", manifest["capabilities"])
        self.assertFalse(manifest["behavior"]["automatic_model_replacement"])
        self.assertFalse(manifest["behavior"]["agent_required"])


if __name__ == "__main__":
    unittest.main()
