import json
import struct
import tempfile
import unittest
from pathlib import Path

from prompt_director.lora_metadata import (
    extract_prompt_metadata,
    read_safetensors_metadata,
    summarize_lora_metadata,
)


class LoraMetadataTests(unittest.TestCase):
    def test_reads_prompt_director_metadata_from_safetensors_header(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "clown_nose_lora.safetensors"
            header = {
                "__metadata__": {
                    "prompt_director": json.dumps(
                        {
                            "schema_version": "1.0",
                            "display_name": "Clown Nose LoRA",
                            "trigger_tokens": ["clownnose"],
                            "model_family": ["qwen_image_edit"],
                            "prompt_guidance": "Use clownnose when requesting the nose effect.",
                        }
                    )
                },
                "dummy.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
            }
            header_bytes = json.dumps(header).encode("utf-8")
            path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + b"\x00\x00\x00\x00")

            raw_metadata = read_safetensors_metadata(path)
            prompt_metadata = extract_prompt_metadata(raw_metadata)

        self.assertEqual(prompt_metadata["display_name"], "Clown Nose LoRA")
        self.assertEqual(prompt_metadata["trigger_tokens"], ["clownnose"])
        self.assertEqual(prompt_metadata["model_family"], ["qwen_image_edit"])

    def test_extracts_flat_metadata_keys_and_summarizes_for_ai_context(self):
        prompt_metadata = extract_prompt_metadata(
            {
                "prompt_director.trigger_tokens": '["clownnose"]',
                "prompt_director.model_family": '["qwen_image_edit"]',
                "prompt_director.prompt_guidance": "Use clownnose for the comic red nose effect.",
                "prompt_director.default_strength_model": "0.85",
            }
        )

        summary = summarize_lora_metadata("clown_nose_lora.safetensors", prompt_metadata)

        self.assertEqual(prompt_metadata["trigger_tokens"], ["clownnose"])
        self.assertEqual(prompt_metadata["default_strength_model"], 0.85)
        self.assertIn("clown_nose_lora.safetensors", summary)
        self.assertIn("clownnose", summary)
        self.assertIn("qwen_image_edit", summary)


if __name__ == "__main__":
    unittest.main()
