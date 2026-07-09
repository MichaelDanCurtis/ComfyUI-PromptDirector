import json
import tempfile
import unittest
from pathlib import Path

from prompt_director.providers import ProviderUnavailable, get_provider
from prompt_director.providers.glm import GlmVisionProvider
from prompt_director.providers.kimi import KimiVisionProvider
from prompt_director.providers.openai_compatible import OpenAICompatibleVisionProvider
from prompt_director.providers.xai import XaiVisionProvider


class ProviderTests(unittest.TestCase):
    def test_openai_compatible_provider_reads_secrets_from_environment_only(self):
        provider = get_provider(
            "openai_compatible",
            env={
                "OPENAI_API_KEY": "secret-key",
                "OPENAI_API_BASE": "https://example.test/v1",
                "PROMPT_DIRECTOR_OPENAI_MODEL": "vision-model",
            },
        )

        self.assertIsInstance(provider, OpenAICompatibleVisionProvider)
        public = provider.public_config()
        self.assertEqual(public["provider"], "openai_compatible")
        self.assertEqual(public["model"], "vision-model")
        self.assertNotIn("secret-key", json.dumps(public))

    def test_provider_is_unavailable_without_required_api_key(self):
        with self.assertRaises(ProviderUnavailable):
            get_provider("openai_compatible", env={})

    def test_kimi_provider_reads_api_key_from_local_profile_and_public_config_stays_clean(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            credentials_path = Path(tmpdir) / "credentials.json"
            credentials_path.write_text(
                json.dumps({"profiles": {"default": {"kimi": {"api_key": "local-kimi-secret"}}}}),
                encoding="utf-8",
            )

            provider = get_provider(
                "kimi",
                env={"PROMPT_DIRECTOR_CREDENTIALS_FILE": str(credentials_path)},
                config={
                    "credential_source": "local_profile",
                    "credential_profile": "default",
                    "base_url": "https://api.moonshot.ai/v1",
                    "model": "kimi-k2.6",
                },
            )

        self.assertIsInstance(provider, KimiVisionProvider)
        public = provider.public_config()
        self.assertEqual(public["provider"], "kimi")
        self.assertEqual(public["base_url"], "https://api.moonshot.ai/v1")
        self.assertEqual(public["model"], "kimi-k2.6")
        self.assertNotIn("local-kimi-secret", json.dumps(public))

    def test_kimi_provider_omits_temperature_for_k2_models(self):
        calls = []

        def fake_http_post(url, headers, payload, timeout):
            calls.append({"url": url, "headers": headers, "payload": payload, "timeout": timeout})
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "description": "A subject in a simple scene.",
                                    "analysis": {},
                                    "warnings": [],
                                }
                            )
                        }
                    }
                ]
            }

        provider = KimiVisionProvider(
            api_key="kimi-secret",
            base_url="https://api.moonshot.ai/v1",
            model="kimi-k2.6",
            http_post=fake_http_post,
        )

        provider.describe_image("data:image/png;base64,abc")

        self.assertNotIn("temperature", calls[0]["payload"])

    def test_local_profile_provider_allows_profile_defaults_when_ui_fields_are_blank(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            credentials_path = Path(tmpdir) / "credentials.json"
            credentials_path.write_text(
                json.dumps(
                    {
                        "profiles": {
                            "default": {
                                "kimi": {
                                    "api_key": "local-kimi-secret",
                                    "base_url": "https://profile.moonshot.test/v1",
                                    "model": "profile-kimi-model",
                                }
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            provider = get_provider(
                "kimi",
                env={"PROMPT_DIRECTOR_CREDENTIALS_FILE": str(credentials_path)},
                config={
                    "credential_source": "local_profile",
                    "credential_profile": "default",
                    "base_url": "",
                    "model": "",
                },
            )

        public = provider.public_config()
        self.assertEqual(public["base_url"], "https://profile.moonshot.test/v1")
        self.assertEqual(public["model"], "profile-kimi-model")

    def test_glm_provider_reads_local_profile_with_provider_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            credentials_path = Path(tmpdir) / "credentials.json"
            credentials_path.write_text(
                json.dumps({"profiles": {"default": {"glm": {"api_key": "local-glm-secret"}}}}),
                encoding="utf-8",
            )

            provider = get_provider(
                "glm",
                env={"PROMPT_DIRECTOR_CREDENTIALS_FILE": str(credentials_path)},
                config={
                    "credential_source": "local_profile",
                    "credential_profile": "default",
                    "base_url": "",
                    "model": "",
                },
            )

        self.assertIsInstance(provider, GlmVisionProvider)
        public = provider.public_config()
        self.assertEqual(public["provider"], "glm")
        self.assertEqual(public["base_url"], "https://api.z.ai/api/paas/v4")
        self.assertEqual(public["model"], "glm-5v-turbo")
        self.assertNotIn("local-glm-secret", json.dumps(public))

    def test_grok_provider_reads_local_profile_with_provider_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            credentials_path = Path(tmpdir) / "credentials.json"
            credentials_path.write_text(
                json.dumps({"profiles": {"default": {"grok": {"api_key": "local-grok-secret"}}}}),
                encoding="utf-8",
            )

            provider = get_provider(
                "grok",
                env={"PROMPT_DIRECTOR_CREDENTIALS_FILE": str(credentials_path)},
                config={
                    "credential_source": "local_profile",
                    "credential_profile": "default",
                    "base_url": "",
                    "model": "",
                },
            )

        self.assertIsInstance(provider, XaiVisionProvider)
        public = provider.public_config()
        self.assertEqual(public["provider"], "grok")
        self.assertEqual(public["base_url"], "https://api.x.ai/v1")
        self.assertEqual(public["model"], "grok-4.3")
        self.assertNotIn("local-grok-secret", json.dumps(public))

    def test_openai_compatible_provider_sends_image_and_parses_analysis(self):
        calls = []

        def fake_http_post(url, headers, payload, timeout):
            calls.append(
                {
                    "url": url,
                    "headers": headers,
                    "payload": payload,
                    "timeout": timeout,
                }
            )
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "subject_count": 1,
                                    "main_subject": "person",
                                    "visible_text": False,
                                }
                            )
                        }
                    }
                ]
            }

        provider = OpenAICompatibleVisionProvider(
            api_key="secret-key",
            base_url="https://example.test/v1",
            model="vision-model",
            http_post=fake_http_post,
        )

        analysis = provider.analyze_image(
            image_data_uri="data:image/png;base64,abc",
            base_prompt="make the jacket red",
            model_target="qwen_edit",
        )

        self.assertEqual(analysis["main_subject"], "person")
        self.assertEqual(calls[0]["url"], "https://example.test/v1/chat/completions")
        self.assertEqual(calls[0]["headers"]["Authorization"], "Bearer secret-key")
        content = calls[0]["payload"]["messages"][1]["content"]
        self.assertEqual(content[1]["image_url"]["url"], "data:image/png;base64,abc")

    def test_openai_compatible_provider_describes_image_as_prompt_ready_text(self):
        calls = []

        def fake_http_post(url, headers, payload, timeout):
            calls.append({"url": url, "headers": headers, "payload": payload, "timeout": timeout})
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "description": (
                                        "Woman with long reddish-brown hair and bangs, wearing a multicolored "
                                        "string bikini, laughing in front of a waterfall."
                                    ),
                                    "analysis": {"subject_count": 1, "visible_text": False},
                                    "warnings": [],
                                }
                            )
                        }
                    }
                ]
            }

        provider = OpenAICompatibleVisionProvider(
            api_key="secret-key",
            base_url="https://example.test/v1",
            model="vision-model",
            http_post=fake_http_post,
        )

        result = provider.describe_image(
            image_data_uri="data:image/png;base64,abc",
            focus_prompt="describe wardrobe and scene",
            detail_level="prompt_ready",
        )

        self.assertIn("reddish-brown hair", result["description"])
        self.assertEqual(result["analysis"]["subject_count"], 1)
        self.assertEqual(calls[0]["url"], "https://example.test/v1/chat/completions")
        content = calls[0]["payload"]["messages"][1]["content"]
        self.assertEqual(content[1]["image_url"]["url"], "data:image/png;base64,abc")

    def test_openai_compatible_provider_enhances_prompt_with_lora_metadata(self):
        calls = []

        def fake_http_post(url, headers, payload, timeout):
            calls.append({"url": url, "headers": headers, "payload": payload, "timeout": timeout})
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "enhanced_prompt": (
                                        "A highly detailed photorealistic edit: add a funny clownnose nose "
                                        "while preserving the subject identity and lighting."
                                    ),
                                    "rationale": "Inserted the LoRA trigger token from metadata.",
                                    "lora_keywords": ["clownnose"],
                                    "warnings": [],
                                }
                            )
                        }
                    }
                ]
            }

        provider = OpenAICompatibleVisionProvider(
            api_key="secret-key",
            base_url="https://example.test/v1",
            model="vision-model",
            http_post=fake_http_post,
        )

        result = provider.enhance_prompt(
            base_prompt="give her a funny nose",
            model_target="qwen_edit",
            image_description="Woman laughing near a waterfall.",
            lora_metadata={"trigger_tokens": ["clownnose"], "model_family": ["qwen_image_edit"]},
            global_rules="Preserve identity.",
        )

        self.assertIn("clownnose", result["enhanced_prompt"])
        self.assertEqual(result["lora_keywords"], ["clownnose"])
        payload_text = json.dumps(calls[0]["payload"])
        self.assertIn("give her a funny nose", payload_text)
        self.assertIn("clownnose", payload_text)

    def test_grok_provider_sends_responses_payload_and_parses_analysis(self):
        calls = []

        def fake_http_post(url, headers, payload, timeout):
            calls.append(
                {
                    "url": url,
                    "headers": headers,
                    "payload": payload,
                    "timeout": timeout,
                }
            )
            return {
                "output": [
                    {
                        "content": [
                            {
                                "type": "output_text",
                                "text": json.dumps(
                                    {
                                        "subject_count": 1,
                                        "main_subject": "product photo",
                                        "visible_text": True,
                                    }
                                ),
                            }
                        ]
                    }
                ]
            }

        provider = XaiVisionProvider(
            api_key="xai-secret",
            base_url="https://api.x.ai/v1",
            model="grok-4.3",
            http_post=fake_http_post,
        )

        analysis = provider.analyze_image(
            image_data_uri="data:image/png;base64,abc",
            base_prompt="remove the logo",
            model_target="qwen_edit",
        )

        self.assertEqual(analysis["main_subject"], "product photo")
        self.assertTrue(analysis["visible_text"])
        self.assertEqual(calls[0]["url"], "https://api.x.ai/v1/responses")
        self.assertEqual(calls[0]["headers"]["Authorization"], "Bearer xai-secret")
        content = calls[0]["payload"]["input"][0]["content"]
        self.assertEqual(content[0]["type"], "input_image")
        self.assertEqual(content[0]["image_url"], "data:image/png;base64,abc")

    def test_grok_provider_describes_image_with_responses_payload(self):
        calls = []

        def fake_http_post(url, headers, payload, timeout):
            calls.append({"url": url, "headers": headers, "payload": payload, "timeout": timeout})
            return {
                "output": [
                    {
                        "content": [
                            {
                                "type": "output_text",
                                "text": json.dumps(
                                    {
                                        "description": "Woman laughing joyfully in front of a cascading waterfall.",
                                        "analysis": {"subject_count": 1},
                                        "warnings": [],
                                    }
                                ),
                            }
                        ]
                    }
                ]
            }

        provider = XaiVisionProvider(
            api_key="xai-secret",
            base_url="https://api.x.ai/v1",
            model="grok-4.3",
            http_post=fake_http_post,
        )

        result = provider.describe_image(
            image_data_uri="data:image/png;base64,abc",
            focus_prompt="describe subject",
            detail_level="detailed",
        )

        self.assertIn("waterfall", result["description"])
        self.assertEqual(calls[0]["url"], "https://api.x.ai/v1/responses")
        content = calls[0]["payload"]["input"][0]["content"]
        self.assertEqual(content[0]["type"], "input_image")
        self.assertEqual(content[1]["type"], "input_text")


if __name__ == "__main__":
    unittest.main()
