from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any


QWEN_EDIT_POLICY = """QWEN IMAGE EDIT POLICY:
Create exactly one main person unless the user explicitly asks for more.
Do not add extra people, selfie companions, duplicate bodies, crowds, signs, placards, captions, text, writing, logos, or watermarks.
Empty hands unless the user specifically asks for an object.
Preserve the source/reference identity, body proportions, pose intent, and camera framing.
Use reference images as appearance guidance only; do not insert them as additional subjects."""

KREA2_POLICY = """KREA2 PROMPT POLICY:
Describe the intended scene composition, subject placement, lighting, material detail, and realism/style target.
Favor clear spatial layout and visual specificity.
Avoid strict edit-locking language unless the prompt is explicitly image-edit driven."""

VIDEO_POLICY = """VIDEO PROMPT POLICY:
Describe temporal motion, camera movement, frame continuity, and identity continuity.
Keep the subject consistent across frames and avoid unnecessary identity, wardrobe, or scene changes."""


@dataclass(frozen=True)
class PromptDirectorResult:
    final_prompt: str
    analysis_json: str
    warnings: list[str]


def build_prompt_director_result(
    base_prompt: str,
    model_target: str,
    analysis_json: str | None = None,
    mode: str = "edit",
    global_rules: str | None = None,
    extra_warnings: list[str] | None = None,
) -> PromptDirectorResult:
    analysis, warnings = _parse_analysis(analysis_json)
    warnings.extend(_warnings_from_analysis(analysis))
    warnings.extend(extra_warnings or [])

    policy = _policy_for_model(model_target)
    final_prompt = _join_prompt_parts(
        [
            global_rules,
            policy,
            _mode_hint(mode),
            "USER INTENT:",
            (base_prompt or "").strip(),
        ]
    )

    return PromptDirectorResult(
        final_prompt=final_prompt,
        analysis_json=json.dumps(analysis, ensure_ascii=True),
        warnings=_dedupe(warnings),
    )


def _parse_analysis(analysis_json: str | None) -> tuple[dict[str, Any], list[str]]:
    if not analysis_json or not analysis_json.strip():
        return {}, []

    try:
        parsed = json.loads(analysis_json)
    except json.JSONDecodeError:
        return {}, ["invalid_analysis_json"]

    if not isinstance(parsed, dict):
        return {}, ["analysis_json_not_object"]

    return parsed, []


def _warnings_from_analysis(analysis: dict[str, Any]) -> list[str]:
    warnings: list[str] = []

    subject_count = analysis.get("subject_count")
    if isinstance(subject_count, (int, float)) and subject_count > 1:
        warnings.append("extra_people")

    if analysis.get("visible_text") is True:
        warnings.append("text_artifacts")

    objects_in_hands = analysis.get("objects_in_hands")
    if isinstance(objects_in_hands, list) and len(objects_in_hands) > 0:
        warnings.append("object_in_hands")

    risks = analysis.get("risks")
    if isinstance(risks, list):
        warnings.extend(str(risk) for risk in risks if risk)

    return warnings


def _policy_for_model(model_target: str) -> str:
    normalized = (model_target or "").strip().lower().replace("-", "_")
    if normalized in {"qwen", "qwen_edit", "qwen_image_edit", "qwen_image_edit_2511"}:
        return QWEN_EDIT_POLICY
    if normalized in {"krea", "krea2", "krea2t"}:
        return KREA2_POLICY
    if normalized in {"ltx", "wan", "video", "ltx_wan"}:
        return VIDEO_POLICY
    return QWEN_EDIT_POLICY


def _mode_hint(mode: str) -> str:
    normalized = (mode or "").strip().lower()
    if normalized == "generate":
        return "MODE: generation prompt; describe the desired final image directly."
    if normalized == "video":
        return "MODE: video prompt; preserve continuity across frames."
    return "MODE: image edit prompt; describe only the intended change and preservation constraints."


def _join_prompt_parts(parts: list[str | None]) -> str:
    return "\n\n".join(part.strip() for part in parts if part and part.strip())


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        key = str(item).strip()
        if key and key not in seen:
            seen.add(key)
            result.append(key)
    return result
