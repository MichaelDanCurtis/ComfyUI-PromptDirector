import unittest

from nodes import PromptDirectorDebugPreview


class PromptDirectorDebugPreviewTests(unittest.TestCase):
    def test_debug_preview_formats_prompt_analysis_and_warnings(self):
        node = PromptDirectorDebugPreview()

        (debug_text,) = node.run(
            final_prompt="QWEN IMAGE EDIT POLICY:\nUSER INTENT:\nMake jacket red.",
            analysis_json='{"subject_count": 2, "visible_text": true}',
            warnings='["extra_people", "text_artifacts"]',
        )

        self.assertIn("Final Prompt", debug_text)
        self.assertIn("Make jacket red", debug_text)
        self.assertIn("subject_count", debug_text)
        self.assertIn("extra_people", debug_text)


if __name__ == "__main__":
    unittest.main()
