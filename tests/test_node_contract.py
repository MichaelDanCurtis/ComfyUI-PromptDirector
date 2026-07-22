import json
import unittest

from nodes import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    PromptDirector,
    PromptDirectorAuto,
    PromptDirectorConfig,
    PromptDirectorContext,
    PromptDirectorDebugPreview,
    PromptDirectorImageDescribe,
    PromptDirectorPromptEnhancer,
    PromptDirectorReference,
    PromptDirectorResultCritic,
    PromptProducer,
)


class PromptDirectorNodeContractTests(unittest.TestCase):
    def test_node_contract_exposes_typed_string_outputs_without_api_key_widget(self):
        inputs = PromptDirector.INPUT_TYPES()
        config_inputs = PromptDirectorConfig.INPUT_TYPES()

        self.assertIn("PromptDirectorImageDescribe", NODE_CLASS_MAPPINGS)
        self.assertIn("PromptDirectorPromptEnhancer", NODE_CLASS_MAPPINGS)
        self.assertIn("PromptProducer", NODE_CLASS_MAPPINGS)
        self.assertIn("PromptDirectorContext", NODE_CLASS_MAPPINGS)
        self.assertIn("PromptDirectorReference", NODE_CLASS_MAPPINGS)
        self.assertIn("PromptDirectorAuto", NODE_CLASS_MAPPINGS)
        self.assertIn("PromptDirectorResultCritic", NODE_CLASS_MAPPINGS)
        self.assertIn("base_prompt", inputs["required"])
        self.assertIn("model_target", inputs["required"])
        self.assertIn("analysis_json", inputs["optional"])
        self.assertIn("image", inputs["optional"])
        self.assertNotIn("api_key", inputs["required"])
        self.assertNotIn("api_key", inputs["optional"])
        self.assertNotIn("api_key", config_inputs["required"])
        self.assertIn("credential_source", config_inputs["required"])
        self.assertIn("credential_profile", config_inputs["required"])
        self.assertIn("base_url", config_inputs["required"])
        self.assertIn("model", config_inputs["required"])
        providers = config_inputs["required"]["analysis_provider"][0]
        self.assertIn("glm", providers)
        self.assertIn("grok", providers)
        self.assertEqual(PromptDirector.RETURN_TYPES[:3], ("STRING", "STRING", "STRING"))
        self.assertEqual(PromptDirector.RETURN_NAMES[:3], ("final_prompt", "analysis_json", "warnings"))
        self.assertEqual(PromptDirector.RETURN_NAMES[3], "edit_plan_json")

    def test_nodes_use_single_flat_prompt_director_menu_category(self):
        node_classes = [
            PromptDirector,
            PromptDirectorAuto,
            PromptDirectorConfig,
            PromptDirectorContext,
            PromptDirectorImageDescribe,
            PromptDirectorPromptEnhancer,
            PromptDirectorReference,
            PromptDirectorResultCritic,
            PromptProducer,
            PromptDirectorDebugPreview,
        ]

        for node_class in node_classes:
            self.assertEqual(node_class.CATEGORY, "Prompt Director")
            self.assertNotIn("/", node_class.CATEGORY)

    def test_node_menu_preserves_prompt_director_names(self):
        expected = {
            "PromptDirector": "Prompt Director",
            "PromptDirectorContext": "Prompt Director Context",
            "PromptDirectorReference": "Prompt Director Reference",
            "PromptDirectorAuto": "Prompt Director Auto",
            "PromptDirectorResultCritic": "Prompt Director Result Critic",
            "PromptDirectorConfig": "Prompt Director Config",
            "PromptDirectorImageDescribe": "Prompt Image Describe",
            "PromptDirectorPromptEnhancer": "Prompt Enhancer",
            "PromptProducer": "Prompt Producer",
            "PromptDirectorDebugPreview": "Prompt Director Debug Preview",
        }

        self.assertEqual(NODE_DISPLAY_NAME_MAPPINGS, expected)
        self.assertEqual(list(NODE_DISPLAY_NAME_MAPPINGS), list(expected))
        self.assertEqual(list(NODE_CLASS_MAPPINGS), list(expected))
        for display_name in expected.values():
            self.assertTrue(display_name.isascii())
            self.assertNotIn("_", display_name)

    def test_config_node_returns_local_profile_settings_without_secret_values(self):
        node = PromptDirectorConfig()

        (config,) = node.run(
            analysis_provider="kimi",
            credential_source="local_profile",
            credential_profile="default",
            base_url="https://api.moonshot.ai/v1",
            model="kimi-k2.6",
            debug=True,
        )

        self.assertEqual(config["analysis_provider"], "kimi")
        self.assertEqual(config["credential_source"], "local_profile")
        self.assertEqual(config["credential_profile"], "default")
        self.assertEqual(config["base_url"], "https://api.moonshot.ai/v1")
        self.assertEqual(config["model"], "kimi-k2.6")
        self.assertTrue(config["debug"])
        self.assertNotIn("api_key", config)

    def test_node_exposes_rich_info_tab_metadata(self):
        inputs = PromptDirector.INPUT_TYPES()
        visible_inputs = {
            **inputs["required"],
            **inputs["optional"],
        }

        self.assertGreater(len(PromptDirector.DESCRIPTION), 5000)
        self.assertIn("Qwen Image Edit 2511", PromptDirector.DESCRIPTION)
        self.assertIn("Recommended wiring", PromptDirector.DESCRIPTION)
        self.assertIn("Troubleshooting", PromptDirector.DESCRIPTION)
        self.assertIn("What to verify", PromptDirector.DESCRIPTION)
        self.assertIn("Secrets", PromptDirector.DESCRIPTION)
        self.assertEqual(len(PromptDirector.OUTPUT_TOOLTIPS), len(PromptDirector.RETURN_TYPES))

        for name, spec in visible_inputs.items():
            self.assertIn(
                "tooltip",
                spec[1],
                msg=f"{name} is missing a ComfyUI info tooltip",
            )
            self.assertGreater(
                len(spec[1]["tooltip"]),
                160,
                msg=f"{name} tooltip is too thin for the info tab",
            )

        for name, tooltip in zip(PromptDirector.RETURN_NAMES, PromptDirector.OUTPUT_TOOLTIPS):
            self.assertGreater(
                len(tooltip),
                160,
                msg=f"{name} output tooltip is too thin for the info tab",
            )

    def test_support_nodes_expose_rich_info_tab_metadata(self):
        support_nodes = [
            (
                PromptDirectorConfig,
                2500,
                ["Environment variables", "Secrets", "Provider behavior", "When to use none"],
            ),
            (
                PromptDirectorDebugPreview,
                1800,
                ["What this preview proves", "Recommended wiring", "How to read warnings"],
            ),
            (
                PromptDirectorImageDescribe,
                1600,
                ["What this node does", "Recommended wiring", "Provider requirements"],
            ),
            (
                PromptDirectorPromptEnhancer,
                1800,
                ["What this node does", "Recommended wiring", "Model-aware enhancement"],
            ),
            (
                PromptProducer,
                2200,
                ["LoRA metadata", "Recommended wiring", "Trigger tokens"],
            ),
            (
                PromptDirectorContext,
                1100,
                ["canonical metadata", "LoRA stack", "Warnings"],
            ),
            (
                PromptDirectorReference,
                650,
                ["reference image", "Reference roles", "semantic role"],
            ),
            (
                PromptDirectorAuto,
                1700,
                ["complete Prompt Director", "Recommended wiring", "runtime"],
            ),
            (
                PromptDirectorResultCritic,
                1000,
                ["source image", "structured verdict", "runtime inspection"],
            ),
        ]

        for node_cls, minimum_description_length, required_phrases in support_nodes:
            with self.subTest(node=node_cls.__name__):
                self.assertGreater(len(node_cls.DESCRIPTION), minimum_description_length)
                for phrase in required_phrases:
                    self.assertIn(phrase, node_cls.DESCRIPTION)
                self.assertEqual(len(node_cls.OUTPUT_TOOLTIPS), len(node_cls.RETURN_TYPES))

                inputs = node_cls.INPUT_TYPES()
                visible_inputs = {
                    **inputs.get("required", {}),
                    **inputs.get("optional", {}),
                }
                for name, spec in visible_inputs.items():
                    self.assertIn("tooltip", spec[1], msg=f"{name} missing tooltip")
                    self.assertGreater(
                        len(spec[1]["tooltip"]),
                        140,
                        msg=f"{node_cls.__name__}.{name} tooltip is too thin",
                    )

                for name, tooltip in zip(node_cls.RETURN_NAMES, node_cls.OUTPUT_TOOLTIPS):
                    self.assertGreater(
                        len(tooltip),
                        140,
                        msg=f"{node_cls.__name__}.{name} output tooltip is too thin",
                    )

    def test_node_run_returns_final_prompt_analysis_and_warnings_strings(self):
        node = PromptDirector()

        final_prompt, analysis_json, warnings, edit_plan_json = node.run(
            base_prompt="Remove the logo from the shirt.",
            model_target="qwen_edit",
            analysis_json='{"visible_text": true}',
        )

        self.assertIn("Remove the logo", final_prompt)
        self.assertIn("QWEN IMAGE EDIT POLICY", final_prompt)
        self.assertTrue(json.loads(analysis_json)["visible_text"])
        self.assertIn("text_artifacts", warnings)
        self.assertEqual(json.loads(edit_plan_json)["target_model"], "qwen_edit")


if __name__ == "__main__":
    unittest.main()
