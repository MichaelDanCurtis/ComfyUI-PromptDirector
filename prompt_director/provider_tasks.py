from __future__ import annotations

import json
import re
from typing import Any

from .diagnostics import diagnostics_from_warnings


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

EDIT_PLAN_SYSTEM_PROMPT = """You create structured edit plans for ComfyUI image and video workflows.
Return only one compact JSON object. Allowed keys are:
changes, preserve, negative_constraints, model_guidance, warnings.
Keep the user's intent authoritative. Use source analysis, assigned reference roles, mask information, model
metadata, and every LoRA's prompt_director metadata. Never turn a reference person into an extra subject.
Do not write a final prompt; a deterministic compiler will do that after validating your plan."""

CRITIQUE_RESULT_SYSTEM_PROMPT = """You compare a source image and edited result for a ComfyUI workflow.
Return only one compact JSON object with:
verdict, scores, observations, requested_changes_met, preservation_failures, unintended_changes,
lora_activation, revised_prompt, warnings.
scores must contain numeric values from 0 to 1 for intent, preservation, identity, composition, and overall.
Judge only the user's requested edit and the supplied edit plan. Call out extra people, altered identity,
pose/background drift, unwanted text or objects, and missing LoRA effects. revised_prompt should be empty when
the result is acceptable and specific when another attempt is warranted."""


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


def normalize_edit_plan_response(content: str) -> dict[str, Any]:
    parsed = _parse_json_object(content)
    if not parsed:
        return {"warnings": ["provider_invalid_edit_plan"] if content.strip() else ["provider_empty_response"]}
    return {
        "changes": _string_list(parsed.get("changes") or parsed.get("requested_changes")),
        "preserve": _string_list(parsed.get("preserve") or parsed.get("preservation_constraints")),
        "negative_constraints": _string_list(parsed.get("negative_constraints")),
        "model_guidance": _string_list(parsed.get("model_guidance")),
        "warnings": _string_list(parsed.get("warnings")),
    }


def normalize_critique_response(content: str) -> dict[str, Any]:
    parsed = _parse_json_object(content)
    if not parsed:
        warnings = ["provider_invalid_critique"] if content.strip() else ["provider_empty_response"]
        return {
            "schema_version": "1.0",
            "verdict": "not_run",
            "scores": {},
            "observations": [],
            "requested_changes_met": [],
            "preservation_failures": [],
            "unintended_changes": [],
            "lora_activation": [],
            "revised_prompt": "",
            "warnings": warnings,
            "diagnostics": diagnostics_from_warnings(warnings),
        }
    scores = parsed.get("scores") if isinstance(parsed.get("scores"), dict) else {}
    normalized_scores = {}
    for key in ("intent", "preservation", "identity", "composition", "overall"):
        try:
            if key in scores:
                normalized_scores[key] = max(0.0, min(1.0, float(scores[key])))
        except (TypeError, ValueError):
            continue
    warnings = _string_list(parsed.get("warnings"))
    return {
        "schema_version": "1.0",
        "verdict": _string_value(parsed, "verdict") or "review",
        "scores": normalized_scores,
        "observations": _string_list(parsed.get("observations")),
        "requested_changes_met": _string_list(parsed.get("requested_changes_met")),
        "preservation_failures": _string_list(parsed.get("preservation_failures")),
        "unintended_changes": _string_list(parsed.get("unintended_changes")),
        "lora_activation": _string_list(parsed.get("lora_activation")),
        "revised_prompt": _string_value(parsed, "revised_prompt"),
        "warnings": warnings,
        "diagnostics": diagnostics_from_warnings(warnings),
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
