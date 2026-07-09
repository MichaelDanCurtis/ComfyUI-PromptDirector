from __future__ import annotations

import json
from typing import Any, Callable
from urllib import request

from ..analysis import parse_provider_analysis
from ..provider_tasks import (
    DESCRIBE_IMAGE_SYSTEM_PROMPT,
    ENHANCE_PROMPT_SYSTEM_PROMPT,
    normalize_description_response,
    normalize_enhancement_response,
)
from .openai_compatible import ANALYSIS_SYSTEM_PROMPT


class XaiVisionProvider:
    provider_name = "grok"

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        http_post: Callable[[str, dict[str, str], dict[str, Any], float], dict[str, Any]] | None = None,
        timeout: float = 60.0,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.http_post = http_post or _urllib_json_post
        self.timeout = timeout

    def public_config(self) -> dict[str, Any]:
        return {
            "provider": self.provider_name,
            "base_url": self.base_url,
            "model": self.model,
            "timeout": self.timeout,
        }

    def analyze_image(self, image_data_uri: str, base_prompt: str, model_target: str) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "temperature": 0,
            "store": False,
            "instructions": ANALYSIS_SYSTEM_PROMPT,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_image", "image_url": image_data_uri},
                        {
                            "type": "input_text",
                            "text": (
                                "Analyze this image for prompt policy. "
                                f"Model target: {model_target}. Base prompt: {base_prompt or ''}"
                            ),
                        },
                    ],
                }
            ],
        }
        response = self.http_post(
            f"{self.base_url}/responses",
            headers,
            payload,
            self.timeout,
        )
        content = _extract_responses_text(response)
        return parse_provider_analysis(content)

    def describe_image(
        self,
        image_data_uri: str,
        focus_prompt: str = "",
        detail_level: str = "detailed",
    ) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "temperature": 0,
            "store": False,
            "instructions": DESCRIBE_IMAGE_SYSTEM_PROMPT,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_image", "image_url": image_data_uri},
                        {
                            "type": "input_text",
                            "text": (
                                "Describe this image for prompt production. "
                                f"Detail level: {detail_level}. Focus: {focus_prompt or 'general image edit context'}."
                            ),
                        },
                    ],
                }
            ],
        }
        response = self.http_post(
            f"{self.base_url}/responses",
            self._headers(),
            payload,
            self.timeout,
        )
        return normalize_description_response(_extract_responses_text(response))

    def enhance_prompt(
        self,
        base_prompt: str,
        model_target: str,
        image_description: str = "",
        lora_metadata: dict[str, Any] | None = None,
        global_rules: str = "",
    ) -> dict[str, Any]:
        context = {
            "base_prompt": base_prompt or "",
            "model_target": model_target or "",
            "image_description": image_description or "",
            "global_rules": global_rules or "",
            "lora_metadata": lora_metadata or {},
        }
        payload = {
            "model": self.model,
            "temperature": 0,
            "store": False,
            "instructions": ENHANCE_PROMPT_SYSTEM_PROMPT,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": json.dumps(context, ensure_ascii=True)},
                    ],
                }
            ],
        }
        response = self.http_post(
            f"{self.base_url}/responses",
            self._headers(),
            payload,
            self.timeout,
        )
        return normalize_enhancement_response(_extract_responses_text(response))

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }


def _extract_responses_text(response: dict[str, Any]) -> str:
    output = response.get("output") if isinstance(response, dict) else None
    if not output:
        return ""

    text_parts = []
    for item in output:
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if isinstance(content, dict) and content.get("type") == "output_text":
                text_parts.append(str(content.get("text", "")))
    return "\n".join(part for part in text_parts if part)


def _urllib_json_post(url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=body, headers=headers, method="POST")
    with request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))
