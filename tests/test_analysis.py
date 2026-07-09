import json
import unittest

from prompt_director.analysis import (
    DEFAULT_ANALYSIS,
    merge_analysis,
    parse_provider_analysis,
)


class AnalysisTests(unittest.TestCase):
    def test_provider_json_is_normalized_to_prompt_director_schema(self):
        raw = """
        ```json
        {
          "subject_count": "2",
          "main_subject": "woman in jacket",
          "visible_text": "yes",
          "objects_in_hands": "phone",
          "risks": ["extra_people"]
        }
        ```
        """

        analysis = parse_provider_analysis(raw)

        self.assertEqual(analysis["subject_count"], 2)
        self.assertEqual(analysis["main_subject"], "woman in jacket")
        self.assertTrue(analysis["visible_text"])
        self.assertEqual(analysis["objects_in_hands"], ["phone"])
        self.assertIn("extra_people", analysis["risks"])

    def test_invalid_provider_json_returns_default_schema_with_warning(self):
        analysis = parse_provider_analysis("not json")

        self.assertEqual(set(DEFAULT_ANALYSIS), set(analysis))
        self.assertIn("provider_invalid_json", analysis["warnings"])

    def test_manual_analysis_overrides_provider_defaults_without_losing_risks(self):
        provider = {
            **DEFAULT_ANALYSIS,
            "subject_count": 1,
            "risks": ["text_artifacts"],
        }
        manual = json.dumps({"subject_count": 3, "risks": ["extra_people"]})

        merged, warnings = merge_analysis(manual, provider)

        self.assertEqual(merged["subject_count"], 3)
        self.assertEqual(merged["risks"], ["text_artifacts", "extra_people"])
        self.assertEqual(warnings, [])


if __name__ == "__main__":
    unittest.main()
