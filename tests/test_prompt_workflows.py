import json
import unittest
from unittest.mock import patch

from nodes import PromptDirectorImageDescribe, PromptDirectorPromptEnhancer, PromptProducer


class FakeProvider:
    def describe_image(self, image_data_uri, focus_prompt="", detail_level="detailed"):
        return {
            "description": "Woman with long reddish-brown hair laughing in front of a waterfall.",
            "analysis": {"subject_count": 1, "visible_text": False},
            "warnings": [],
        }

    def enhance_prompt(
        self,
        base_prompt,
        model_target,
        image_description="",
        lora_metadata=None,
        global_rules="",
    ):
        keywords = (lora_metadata or {}).get("trigger_tokens", [])
        keyword_text = f" using {', '.join(keywords)}" if keywords else ""
        return {
            "enhanced_prompt": (
                f"A highly detailed photorealistic edit{keyword_text}: {base_prompt}. "
                "Preserve identity, pose, lighting, and background."
            ),
            "rationale": "Expanded the edit prompt for model adherence.",
            "lora_keywords": keywords,
            "warnings": [],
        }


class PromptWorkflowNodeTests(unittest.TestCase):
    def test_image_describe_node_returns_provider_description_and_analysis_json(self):
        node = PromptDirectorImageDescribe()

        with (
            patch("nodes.get_provider", return_value=FakeProvider()),
            patch("nodes.image_to_png_data_uri", return_value="data:image/png;base64,abc"),
        ):
            description, analysis_json, warnings = node.run(
                image=object(),
                detail_level="prompt_ready",
                focus_prompt="describe wardrobe and scene",
                config={"analysis_provider": "grok"},
            )

        self.assertIn("reddish-brown hair", description)
        self.assertEqual(json.loads(analysis_json)["subject_count"], 1)
        self.assertEqual(json.loads(warnings), [])

    def test_prompt_enhancer_node_uses_provider_to_expand_basic_prompt(self):
        node = PromptDirectorPromptEnhancer()

        with patch("nodes.get_provider", return_value=FakeProvider()):
            enhanced_prompt, rationale, warnings = node.run(
                base_prompt="Image of a woman in a red bikini",
                model_target="qwen_edit",
                image_description="Woman laughing in front of waterfall.",
                global_rules="Preserve the background.",
                config={"analysis_provider": "grok"},
            )

        self.assertIn("highly detailed photorealistic edit", enhanced_prompt)
        self.assertIn("Preserve identity", enhanced_prompt)
        self.assertIn("model adherence", rationale)
        self.assertEqual(json.loads(warnings), [])

    def test_prompt_producer_uses_lora_metadata_keywords_when_enhancing_prompt(self):
        node = PromptProducer()
        metadata_json = json.dumps(
            {
                "trigger_tokens": ["clownnose"],
                "model_family": ["qwen_image_edit"],
                "prompt_guidance": "Use clownnose for the comic nose effect.",
            }
        )

        with patch("nodes.get_provider", return_value=FakeProvider()):
            enhanced_prompt, lora_keywords, metadata_out, warnings = node.run(
                base_prompt="give her a funny nose",
                model_target="qwen_edit",
                lora_name="none",
                lora_metadata_json=metadata_json,
                image_description="Portrait of a laughing woman.",
                config={"analysis_provider": "grok"},
            )

        self.assertIn("clownnose", enhanced_prompt)
        self.assertEqual(json.loads(lora_keywords), ["clownnose"])
        self.assertEqual(json.loads(metadata_out)["trigger_tokens"], ["clownnose"])
        self.assertEqual(json.loads(warnings), [])


if __name__ == "__main__":
    unittest.main()
