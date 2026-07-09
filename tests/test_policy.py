import json
import unittest

from prompt_director.policy import build_prompt_director_result


class PromptPolicyTests(unittest.TestCase):
    def test_qwen_edit_adds_strict_editing_policy_and_analysis_warnings(self):
        analysis = {
            "subject_count": 2,
            "visible_text": True,
            "objects_in_hands": ["phone"],
        }

        result = build_prompt_director_result(
            base_prompt="Make the jacket red while keeping the same person.",
            model_target="qwen_edit",
            analysis_json=json.dumps(analysis),
        )

        self.assertIn("Make the jacket red", result.final_prompt)
        self.assertIn("QWEN IMAGE EDIT POLICY", result.final_prompt)
        self.assertIn("Create exactly one main person", result.final_prompt)
        self.assertIn("Use reference images as appearance guidance only", result.final_prompt)
        self.assertIn("Do not add extra people", result.final_prompt)
        self.assertIn("extra_people", result.warnings)
        self.assertIn("text_artifacts", result.warnings)
        self.assertIn("object_in_hands", result.warnings)
        self.assertEqual(json.loads(result.analysis_json)["subject_count"], 2)

    def test_invalid_analysis_json_keeps_prompt_and_reports_warning(self):
        result = build_prompt_director_result(
            base_prompt="Replace the background with a studio setup.",
            model_target="qwen_edit",
            analysis_json="{bad json",
        )

        self.assertIn("Replace the background", result.final_prompt)
        self.assertIn("invalid_analysis_json", result.warnings)
        self.assertEqual(json.loads(result.analysis_json), {})

    def test_krea_policy_is_compositional_not_edit_locking(self):
        result = build_prompt_director_result(
            base_prompt="A product photo of a matte black headset.",
            model_target="krea2",
        )

        self.assertIn("KREA2 PROMPT POLICY", result.final_prompt)
        self.assertIn("scene composition", result.final_prompt)
        self.assertNotIn("QWEN IMAGE EDIT POLICY", result.final_prompt)

    def test_ltx_policy_emphasizes_motion_and_identity_continuity(self):
        result = build_prompt_director_result(
            base_prompt="The subject turns toward camera.",
            model_target="ltx",
        )

        self.assertIn("VIDEO PROMPT POLICY", result.final_prompt)
        self.assertIn("frame continuity", result.final_prompt)
        self.assertIn("identity", result.final_prompt)


if __name__ == "__main__":
    unittest.main()
