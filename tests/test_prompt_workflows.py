import json
import unittest
from unittest.mock import patch

from nodes import (
    PromptDirectorAuto,
    PromptDirectorImageDescribe,
    PromptDirectorPromptEnhancer,
    PromptDirectorResultCritic,
    PromptProducer,
)


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
        model_context=None,
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

    def create_edit_plan(self, base_prompt, model_target, analysis=None, model_context=None, references=None, mask_info=None, global_rules=""):
        return {
            "changes": [f"Change only the requested attribute: {base_prompt}"],
            "preserve": ["Preserve identity, pose, lighting, and background."],
            "negative_constraints": ["no extra people", "no readable text"],
            "warnings": [],
        }

    def critique_result(self, source_image_data_uri, result_image_data_uri, user_intent, edit_plan=None, model_context=None):
        return {
            "verdict": "needs_revision",
            "scores": {"intent": 0.9, "preservation": 0.6, "identity": 0.7},
            "observations": ["The jacket changed correctly but the background also changed."],
            "unintended_changes": ["background"],
            "revised_prompt": "Make the jacket red and preserve the background exactly.",
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

    def test_auto_node_builds_provider_plan_and_compiles_source_context(self):
        node = PromptDirectorAuto()
        context = {
            "target_model": "qwen_edit",
            "model": {"name": "qwen-2511.safetensors", "model_card": {"model_family": ["qwen_image_edit"]}},
            "loras": [],
            "warnings": [],
        }

        with (
            patch("nodes.get_provider", return_value=FakeProvider()),
            patch("nodes.image_to_png_data_uri", return_value="data:image/png;base64,abc"),
        ):
            final_prompt, edit_plan_json, context_json, analysis_json, warnings = node.run(
                base_prompt="make the jacket red",
                model_target="auto",
                mode="edit",
                image=object(),
                context=context,
                config={"analysis_provider": "grok"},
                unique_id="91",
            )

        self.assertIn("make the jacket red", final_prompt)
        self.assertIn("Woman with long reddish-brown hair", final_prompt)
        self.assertEqual(json.loads(edit_plan_json)["target_model"], "qwen_edit")
        self.assertEqual(json.loads(context_json)["model"]["name"], "qwen-2511.safetensors")
        self.assertEqual(json.loads(analysis_json)["subject_count"], 1)
        self.assertEqual(json.loads(warnings), [])

    def test_result_critic_returns_structured_revision(self):
        node = PromptDirectorResultCritic()

        with (
            patch("nodes.get_provider", return_value=FakeProvider()),
            patch("nodes.image_to_png_data_uri", side_effect=["source-uri", "result-uri"]),
        ):
            critique_json, revised_prompt, warnings = node.run(
                source_image=object(),
                result_image=object(),
                user_intent="make the jacket red",
                config={"analysis_provider": "grok"},
                unique_id="92",
            )

        critique = json.loads(critique_json)
        self.assertEqual(critique["verdict"], "needs_revision")
        self.assertIn("background", critique["unintended_changes"])
        self.assertIn("preserve the background", revised_prompt.lower())
        self.assertEqual(json.loads(warnings), [])


if __name__ == "__main__":
    unittest.main()
