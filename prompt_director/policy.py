from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from .edit_plan import build_edit_plan, compile_edit_plan
from .model_policies import (
    KREA2_POLICY,
    QWEN_EDIT_POLICY,
    VIDEO_POLICY,
    mode_hint,
    policy_for_model,
)


@dataclass(frozen=True)
class PromptDirectorResult:
    final_prompt: str
    analysis_json: str
    warnings: list[str]
    edit_plan_json: str


def build_prompt_director_result(
    base_prompt: str,
    model_target: str,
    analysis_json: str | None = None,
    mode: str = "edit",
    global_rules: str | None = None,
    extra_warnings: list[str] | None = None,
    context: dict[str, Any] | None = None,
    references: list[dict[str, Any]] | None = None,
    mask_info: dict[str, Any] | None = None,
    provider_plan: dict[str, Any] | None = None,
) -> PromptDirectorResult:
    analysis, warnings = _parse_analysis(analysis_json)
    warnings.extend(_warnings_from_analysis(analysis))
    warnings.extend(extra_warnings or [])

    plan = build_edit_plan(
        base_prompt=base_prompt,
        model_target=model_target,
        analysis=analysis,
        context=context,
        references=references,
        mask_info=mask_info,
        global_rules=global_rules or "",
        mode=mode,
        provider_plan=provider_plan,
    )
    warnings.extend(plan.get("warnings", []))
    final_prompt = compile_edit_plan(plan)

    return PromptDirectorResult(
        final_prompt=final_prompt,
        analysis_json=json.dumps(analysis, ensure_ascii=True),
        warnings=_dedupe(warnings),
        edit_plan_json=json.dumps(plan, ensure_ascii=True),
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
    return policy_for_model(model_target)


def _mode_hint(mode: str) -> str:
    return mode_hint(mode)


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
