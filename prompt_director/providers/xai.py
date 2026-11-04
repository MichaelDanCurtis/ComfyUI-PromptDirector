from __future__ import annotations

import json
from typing import Any, Callable
from urllib import request

from ..analysis import parse_provider_analysis
from ..provider_tasks import (
    CRITIQUE_RESULT_SYSTEM_PROMPT,
    DESCRIBE_IMAGE_SYSTEM_PROMPT,
    EDIT_PLAN_SYSTEM_PROMPT,
    ENHANCE_PROMPT_SYSTEM_PROMPT,
    normalize_critique_response,
    normalize_description_response,
    normalize_edit_plan_response,
    normalize_enhancement_response,
)
from ..request_auth import apply_provider_auth, sanitize_provider_error
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
        auth_style: str = "bearer",
        extra_headers: dict[str, str] | None = None,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.http_post = http_post or _urllib_json_post
        self.timeout = timeout
        self.auth_style = auth_style or "bearer"
        self.extra_headers = dict(extra_headers or {})

    def public_config(self) -> dict[str, Any]:
        return {
            "provider": self.provider_name,
            "base_url": self.base_url,
            "model": self.model,
            "timeout": self.timeout,
            "auth_style": self.auth_style,
            "header_names": sorted(self.extra_headers),
        }

    def analyze_image(self, image_data_uri: str, base_prompt: str, model_target: str) -> dict[str, Any]:
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
        response = self._post(payload)
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
        response = self._post(payload)
        return normalize_description_response(_extract_responses_text(response))

    def enhance_prompt(
        self,
        base_prompt: str,
        model_target: str,
        image_description: str = "",
        lora_metadata: dict[str, Any] | None = None,
        global_rules: str = "",
        model_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        context = {
            "base_prompt": base_prompt or "",
            "model_target": model_target or "",
            "image_description": image_description or "",
            "global_rules": global_rules or "",
            "lora_metadata": lora_metadata or {},
            "model_context": model_context or {},
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
        response = self._post(payload)
        return normalize_enhancement_response(_extract_responses_text(response))

    def create_edit_plan(
        self,
        base_prompt: str,
        model_target: str,
        analysis: dict[str, Any] | None = None,
        model_context: dict[str, Any] | None = None,
        references: list[dict[str, Any]] | None = None,
        mask_info: dict[str, Any] | None = None,
        global_rules: str = "",
    ) -> dict[str, Any]:
        context = {
            "base_prompt": base_prompt or "",
            "model_target": model_target or "",
            "analysis": analysis or {},
            "model_context": model_context or {},
            "references": references or [],
            "mask_info": mask_info or {},
            "global_rules": global_rules or "",
        }
        payload = {
            "model": self.model,
            "temperature": 0,
            "store": False,
            "instructions": EDIT_PLAN_SYSTEM_PROMPT,
            "input": [{"role": "user", "content": [{"type": "input_text", "text": json.dumps(context, ensure_ascii=True)}]}],
        }
        return normalize_edit_plan_response(_extract_responses_text(self._post(payload)))

    def critique_result(
        self,
        source_image_data_uri: str,
        result_image_data_uri: str,
        user_intent: str,
        edit_plan: dict[str, Any] | None = None,
        model_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        context = json.dumps(
            {"user_intent": user_intent or "", "edit_plan": edit_plan or {}, "model_context": model_context or {}},
            ensure_ascii=True,
        )
        payload = {
            "model": self.model,
            "temperature": 0,
            "store": False,
            "instructions": CRITIQUE_RESULT_SYSTEM_PROMPT,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "SOURCE IMAGE"},
                        {"type": "input_image", "image_url": source_image_data_uri},
                        {"type": "input_text", "text": "EDITED RESULT"},
                        {"type": "input_image", "image_url": result_image_data_uri},
                        {"type": "input_text", "text": context},
                    ],
                }
            ],
        }
        return normalize_critique_response(_extract_responses_text(self._post(payload)))

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        url, headers = apply_provider_auth(
            f"{self.base_url}/responses",
            api_key=self.api_key,
            auth_style=self.auth_style,
            extra_headers=self.extra_headers,
        )
        try:
            return self.http_post(url, headers, payload, self.timeout)
        except Exception as exc:
            secrets = [self.api_key, *self.extra_headers.values()]
            raise RuntimeError(sanitize_provider_error(exc, secrets)) from exc


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
