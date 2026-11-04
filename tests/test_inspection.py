import unittest

from prompt_director.inspection import (
    clear_inspections,
    list_inspections,
    list_inspections_v1,
    record_inspection,
)


class InspectionTests(unittest.TestCase):
    def tearDown(self):
        clear_inspections()

    def test_runtime_inspection_is_keyed_by_node_and_redacts_secret_like_fields(self):
        record_inspection(
            "42",
            "prompt_director_auto",
            {
                "final_prompt": "make the jacket red",
                "api_key": "must-not-leak",
                "context": {"access_token": "must-not-leak", "target_model": "qwen_edit"},
            },
        )

        payload = list_inspections(node_id="42")

        self.assertEqual(payload["inspections"][0]["node_id"], "42")
        self.assertEqual(payload["inspections"][0]["payload"]["api_key"], "[redacted]")
        self.assertEqual(payload["inspections"][0]["payload"]["context"]["access_token"], "[redacted]")
        self.assertEqual(payload["inspections"][0]["payload"]["context"]["target_model"], "qwen_edit")

    def test_versioned_inspection_identifies_module_and_adds_structured_diagnostics(self):
        record_inspection(
            "9",
            "prompt_director_context",
            {"warnings": ["model_context_missing"]},
        )

        payload = list_inspections_v1(node_id="9")

        self.assertEqual(payload["module_id"], "comfyui-prompt-director")
        self.assertEqual(payload["module_version"], "0.3.0.dev0")
        diagnostic = payload["inspections"][0]["payload"]["diagnostics"][0]
        self.assertEqual(diagnostic["code"], "model_context_missing")
        self.assertTrue(diagnostic["advisory"])


if __name__ == "__main__":
    unittest.main()
