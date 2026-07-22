import json
import struct
import tempfile
import unittest
from pathlib import Path

from prompt_director.model_context import (
    build_prompt_context,
    load_prompt_context,
    read_asset_metadata,
    resolve_model_target,
)


def _write_fake_safetensors(path: Path, metadata: dict[str, str]):
    header = {
        "__metadata__": metadata,
        "weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
    }
    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\x00\x00\x00\x00")


class ModelContextTests(unittest.TestCase):
    def test_reads_model_explorer_model_card_and_prompt_director_namespaces(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "qwen.safetensors"
            _write_fake_safetensors(
                path,
                {
                    "model_card": json.dumps(
                        {
                            "schema_version": "1.0",
                            "asset_type": "diffusion_model",
                            "display_name": "Qwen Image Edit 2511",
                            "base_model": "Qwen-Image",
                            "model_family": ["qwen_image_edit"],
                            "architecture": "qwen_image",
                            "precision": "BF16",
                        }
                    ),
                    "prompt_director": json.dumps(
                        {
                            "schema_version": "1.0",
                            "preservation_guidance": "Preserve source identity and framing.",
                        }
                    ),
                    "modelspec.title": "Qwen Image Edit 2511",
                },
            )

            asset = read_asset_metadata(path, category="diffusion_models", name=path.name)

        self.assertEqual(asset["model_card"]["asset_type"], "diffusion_model")
        self.assertEqual(asset["model_card"]["model_family"], ["qwen_image_edit"])
        self.assertEqual(asset["prompt_director"]["preservation_guidance"], "Preserve source identity and framing.")
        self.assertEqual(asset["modelspec"]["modelspec.title"], "Qwen Image Edit 2511")

    def test_multi_lora_context_checks_compatibility_strengths_and_trigger_tokens(self):
        model = {
            "category": "diffusion_models",
            "name": "qwen.safetensors",
            "model_card": {
                "asset_type": "diffusion_model",
                "model_family": ["qwen_image_edit"],
                "compatible_families": ["qwen_image_edit"],
            },
            "prompt_director": {},
            "modelspec": {},
        }
        loras = [
            {
                "category": "loras",
                "name": "good.safetensors",
                "strength_model": 0.8,
                "strength_clip": 0.7,
                "model_card": {"compatible_families": ["qwen_image_edit"]},
                "prompt_director": {
                    "trigger_tokens": ["goodtrigger"],
                    "strength_min": 0.5,
                    "strength_max": 1.0,
                },
                "modelspec": {},
            },
            {
                "category": "loras",
                "name": "sdxl-only.safetensors",
                "strength_model": 1.4,
                "strength_clip": 1.0,
                "model_card": {"compatible_families": ["sdxl"]},
                "prompt_director": {
                    "trigger_tokens": ["wrongtrigger"],
                    "strength_min": 0.2,
                    "strength_max": 1.0,
                },
                "modelspec": {},
            },
        ]

        context, warnings = build_prompt_context(model_asset=model, lora_assets=loras)

        self.assertEqual(context["target_model"], "qwen_edit")
        self.assertEqual(context["target"]["source"], "model_card.model_family")
        self.assertEqual(context["target"]["confidence"], 1.0)
        self.assertEqual(context["loras"][0]["compatibility"], "compatible")
        self.assertEqual(context["loras"][0]["compatibility_detail"]["reason"], "explicit_family_match")
        self.assertEqual(context["loras"][1]["compatibility"], "incompatible")
        self.assertIn("lora_incompatible:sdxl-only.safetensors:qwen_edit", warnings)
        self.assertIn("lora_strength_above_max:sdxl-only.safetensors:1.4:1.0", warnings)
        self.assertEqual(context["required_trigger_tokens"], ["goodtrigger", "wrongtrigger"])
        self.assertIn("lora_incompatible", [item["code"] for item in context["diagnostics"]])
        self.assertIn("lora_strength_above_max", [item["code"] for item in context["diagnostics"]])

    def test_family_weight_override_supplies_advisory_strength_guidance(self):
        model = {
            "name": "qwen.safetensors",
            "model_card": {"model_family": ["qwen_image_edit"]},
            "prompt_director": {},
            "modelspec": {},
        }
        lora = {
            "name": "override.safetensors",
            "strength_model": 1.1,
            "strength_clip": 0.7,
            "model_card": {
                "compatible_families": ["qwen_image_edit"],
                "family_weight_overrides": {
                    "qwen_image_edit": {
                        "model": 0.75,
                        "clip": 0.65,
                        "min": 0.5,
                        "max": 0.9,
                    }
                },
            },
            "prompt_director": {"strength_min": 0.2, "strength_max": 1.4},
            "modelspec": {},
        }

        context, warnings = build_prompt_context(model_asset=model, lora_assets=[lora])

        guidance = context["loras"][0]["strength_guidance"]
        self.assertEqual(guidance["source"], "model_card.family_weight_overrides.qwen_image_edit")
        self.assertEqual(guidance["default_strength_model"], 0.75)
        self.assertEqual(guidance["default_strength_clip"], 0.65)
        self.assertEqual(guidance["strength_max"], 0.9)
        self.assertIn("lora_strength_above_max:override.safetensors:1.1:0.9", warnings)

    def test_explicit_model_choice_remains_authoritative_over_metadata(self):
        model = {
            "name": "sdxl.safetensors",
            "model_card": {"model_family": ["sdxl"]},
            "prompt_director": {},
            "modelspec": {},
        }

        resolution = resolve_model_target(model, requested="qwen_edit")
        context, _warnings = build_prompt_context(
            model_asset=model,
            lora_assets=[],
            model_target="qwen_edit",
        )

        self.assertEqual(resolution["resolved"], "qwen_edit")
        self.assertEqual(resolution["source"], "user_selection")
        self.assertEqual(context["target_model"], "qwen_edit")
        self.assertEqual(context["target"]["source"], "user_selection")

    def test_supplied_context_preserves_future_fields_and_gains_target_evidence(self):
        supplied = {
            "schema_version": "1.1",
            "target_model": "qwen_edit",
            "model": {},
            "loras": [],
            "required_trigger_tokens": [],
            "warnings": [],
            "future": {"kept": True},
        }

        context, warnings = load_prompt_context(context_json=json.dumps(supplied))

        self.assertEqual(warnings, [])
        self.assertEqual(context["target"]["source"], "context.target_model")
        self.assertEqual(context["future"], {"kept": True})

    def test_structured_diagnostics_do_not_duplicate_legacy_warning_evidence(self):
        model = {
            "name": "qwen.safetensors",
            "model_card": {"model_family": ["qwen_image_edit"]},
            "prompt_director": {},
            "modelspec": {},
        }
        lora = {
            "name": "high.safetensors",
            "strength_model": 1.5,
            "strength_clip": 1.0,
            "model_card": {"compatible_families": ["qwen_image_edit"]},
            "prompt_director": {"strength_max": 1.0},
            "modelspec": {},
        }

        context, _warnings = build_prompt_context(model_asset=model, lora_assets=[lora])

        codes = [item["code"] for item in context["diagnostics"]]
        self.assertEqual(codes.count("lora_strength_above_max"), 1)

        missing_context, _missing_warnings = build_prompt_context(
            model_asset={},
            lora_assets=[],
            model_target="qwen_edit",
        )
        missing_codes = [item["code"] for item in missing_context["diagnostics"]]
        self.assertEqual(missing_codes.count("model_context_missing"), 1)


if __name__ == "__main__":
    unittest.main()
