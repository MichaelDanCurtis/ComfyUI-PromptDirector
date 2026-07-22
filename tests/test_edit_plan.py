import json
import unittest

from prompt_director.edit_plan import build_edit_plan, compile_edit_plan


class EditPlanTests(unittest.TestCase):
    def test_analysis_model_context_loras_references_and_mask_are_compiled(self):
        context = {
            "target_model": "qwen_edit",
            "model": {
                "name": "qwen_image_edit_2511_bf16.safetensors",
                "model_card": {
                    "display_name": "Qwen Image Edit 2511",
                    "model_family": ["qwen_image_edit"],
                },
            },
            "loras": [
                {
                    "name": "clown_nose.safetensors",
                    "strength_model": 0.8,
                    "strength_clip": 0.8,
                    "compatibility": "compatible",
                    "prompt_director": {
                        "trigger_tokens": ["clownnose"],
                        "negative_tokens": ["extra nose"],
                        "preservation_guidance": "Keep the rest of the face unchanged.",
                    },
                }
            ],
        }
        analysis = {
            "subject_count": 1,
            "main_subject": "woman with reddish-brown hair",
            "identity_notes": "same face and hair",
            "pose_notes": "laughing with hands on stomach",
            "scene_notes": "rocks and waterfall",
            "visible_text": False,
            "objects_in_hands": [],
            "risks": [],
        }

        plan = build_edit_plan(
            base_prompt="give her a funny nose",
            model_target="auto",
            analysis=analysis,
            context=context,
            references=[
                {
                    "role": "identity",
                    "label": "face reference",
                    "description": "same woman in neutral lighting",
                    "instructions": "Use only for facial identity.",
                }
            ],
            mask_info={"provided": True, "coverage_percent": 8.5},
            global_rules="Do not add text.",
        )
        prompt = compile_edit_plan(plan)

        self.assertEqual(plan["target_model"], "qwen_edit")
        self.assertIn("give her a funny nose", plan["changes"])
        self.assertIn("clownnose", plan["lora_trigger_tokens"])
        self.assertIn("same face and hair", prompt)
        self.assertIn("laughing with hands on stomach", prompt)
        self.assertIn("rocks and waterfall", prompt)
        self.assertIn("clownnose", prompt)
        self.assertIn("extra nose", prompt)
        self.assertIn("face reference", prompt)
        self.assertIn("outside the mask", prompt)
        self.assertIn("Do not add text", prompt)

    def test_provider_plan_is_normalized_but_deterministic_context_remains_authoritative(self):
        plan = build_edit_plan(
            base_prompt="make the jacket red",
            model_target="qwen_edit",
            analysis={"main_subject": "person in a jacket"},
            context={},
            provider_plan={
                "changes": ["Change the jacket to solid red."],
                "preserve": ["Preserve the face."],
                "negative_constraints": ["no logo"],
            },
        )

        self.assertIn("Change the jacket to solid red.", plan["changes"])
        self.assertIn("Preserve the face.", plan["preserve"])
        self.assertEqual(plan["source_context"]["main_subject"], "person in a jacket")
        self.assertEqual(plan["target_model"], "qwen_edit")


if __name__ == "__main__":
    unittest.main()
