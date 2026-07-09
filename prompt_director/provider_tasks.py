from __future__ import annotations

import json
import re
from typing import Any


DESCRIBE_IMAGE_SYSTEM_PROMPT = """You describe source/reference images for ComfyUI image-edit prompt workflows.
Return only a compact JSON object with:
description, analysis, warnings.
description must be one prompt-ready sentence or compact paragraph describing visible subject, pose,
expression, clothing, colors, environment, lighting, camera/framing, and visible text.
analysis should include subject_count, main_subject, attire, pose, expression, environment, lighting,
visible_text, objects_in_hands, and edit_risks when known.
Do not include markdown or code fences."""


ENHANCE_PROMPT_SYSTEM_PROMPT = """You rewrite user prompts for ComfyUI image, image-edit, or video models.
Return only a compact JSON object with:
enhanced_prompt, rationale, lora_keywords, warnings.
Use the user's base prompt as intent, the image description as visual context, model_target as the target
model family, global_rules as hard constraints, and LoRA metadata as activation guidance.
If LoRA trigger tokens are provided, include the required trigger tokens naturally and exactly.
For image-edit models, preserve identity, pose, composition, lighting, shadows, and background unless the
user explicitly asks to change them. Do not add unrelated subjects, text, logos, or objects."""


def normalize_description_response(content: str) -> dict[str, Any]:
    parsed = _parse_json_object(content)
    if parsed:
        description = _string_value(parsed, "description") or _string_value(parsed, "prompt_ready_description")
        analysis = parsed.get("analysis") if isinstance(parsed.get("analysis"), dict) else {}
        if not analysis:
            analysis = {
                key: parsed[key]
                for key in (
                    "subject_count",
                    "main_subject",
                    "attire",
                    "pose",
                    "expression",
                    "environment",
                    "lighting",
                    "visible_text",
                    "objects_in_hands",
                    "edit_risks",
                )
                if key in parsed
            }
        return {
            "description": description or content.strip(),
            "analysis": analysis,
            "warnings": _string_list(parsed.get("warnings")),
        }

    return {
        "description": content.strip(),
        "analysis": {},
        "warnings": ["provider_non_json_description"] if content.strip() else ["provider_empty_response"],
    }


def normalize_enhancement_response(content: str) -> dict[str, Any]:
    parsed = _parse_json_object(content)
    if parsed:
        enhanced_prompt = _string_value(parsed, "enhanced_prompt") or _string_value(parsed, "final_prompt")
        return {
            "enhanced_prompt": enhanced_prompt or content.strip(),
            "rationale": _string_value(parsed, "rationale"),
            "lora_keywords": _string_list(parsed.get("lora_keywords")),
            "warnings": _string_list(parsed.get("warnings")),
        }

    return {
        "enhanced_prompt": content.strip(),
        "rationale": "",
        "lora_keywords": [],
        "warnings": ["provider_non_json_enhancement"] if content.strip() else ["provider_empty_response"],
    }


def _parse_json_object(content: str) -> dict[str, Any]:
    raw = (content or "").strip()
    if not raw:
        return {}
    raw = _strip_code_fence(raw)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return parsed if isinstance(parsed, dict) else {}


def _strip_code_fence(raw: str) -> str:
    if not raw.startswith("```"):
        return raw
    lines = raw.splitlines()
    if len(lines) >= 3 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return raw


def _string_value(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if value is None:
        return ""
    return str(value).strip()


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []
