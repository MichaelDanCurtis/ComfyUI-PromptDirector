from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from .model_policies import mode_hint, normalize_model_target, policy_for_model


EDIT_PLAN_SCHEMA_VERSION = "1.0"


def build_edit_plan(
    *,
    base_prompt: str,
    model_target: str,
    analysis: Mapping[str, Any] | None = None,
    context: Mapping[str, Any] | None = None,
    references: list[Mapping[str, Any]] | None = None,
    mask_info: Mapping[str, Any] | None = None,
    global_rules: str = "",
    mode: str = "edit",
    provider_plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    analysis_data = deepcopy(dict(analysis or {}))
    context_data = deepcopy(dict(context or {}))
    provider = deepcopy(dict(provider_plan or {}))
    target = _resolve_target(model_target, context_data)

    changes = _string_list(provider.get("changes")) or _string_list(provider.get("requested_changes"))
    if not changes and (base_prompt or "").strip():
        changes = [(base_prompt or "").strip()]

    preserve = _string_list(provider.get("preserve")) + _string_list(provider.get("preservation_constraints"))
    negatives = _string_list(provider.get("negative_constraints"))
    warnings = _string_list(provider.get("warnings")) + _string_list(context_data.get("warnings"))
    lora_tokens: list[str] = []
    lora_guidance: list[str] = []

    model = context_data.get("model") if isinstance(context_data.get("model"), dict) else {}
    model_prompt = model.get("prompt_director") if isinstance(model.get("prompt_director"), dict) else {}
    preserve.extend(_string_list(model_prompt.get("preservation_guidance")))
    lora_guidance.extend(_string_list(model_prompt.get("prompt_guidance")))

    for lora in context_data.get("loras", []) if isinstance(context_data.get("loras"), list) else []:
        if not isinstance(lora, dict):
            continue
        prompt = lora.get("prompt_director") if isinstance(lora.get("prompt_director"), dict) else {}
        lora_tokens.extend(_string_list(prompt.get("trigger_tokens")) or _string_list(lora.get("trigger_tokens")))
        lora_guidance.extend(_string_list(prompt.get("prompt_guidance")))
        preserve.extend(_string_list(prompt.get("preservation_guidance")))
        negatives.extend(_string_list(prompt.get("negative_tokens")))

    lora_tokens.extend(_string_list(context_data.get("required_trigger_tokens")))
    if analysis_data.get("visible_text") is True:
        negatives.append("Do not reproduce visible text, captions, signs, logos, or watermarks unless explicitly requested.")
    if _string_list(analysis_data.get("objects_in_hands")):
        negatives.append("Do not add or retain objects in the subject's hands unless the requested edit requires them.")
    negatives.extend(_risk_constraints(_string_list(analysis_data.get("risks"))))

    mask = dict(mask_info or {})
    if mask.get("provided"):
        preserve.append("Apply the requested change only inside the mask and preserve everything outside the mask.")

    source_context = {
        key: analysis_data.get(key)
        for key in (
            "subject_count",
            "main_subject",
            "identity_notes",
            "pose_notes",
            "scene_notes",
            "objects_in_hands",
            "visible_text",
        )
        if analysis_data.get(key) not in (None, "", [], {})
    }
    target_hint_key = {
        "qwen_edit": "qwen_edit_prompt",
        "krea2": "krea_prompt",
        "ltx": "ltx_motion_prompt",
        "wan": "ltx_motion_prompt",
    }.get(target)
    model_guidance = _string_list(provider.get("model_guidance"))
    if target_hint_key:
        model_guidance.extend(_string_list(analysis_data.get(target_hint_key)))

    normalized_references = []
    for reference in references or []:
        if not isinstance(reference, Mapping):
            continue
        normalized_references.append(
            {
                key: str(reference.get(key, "")).strip()
                for key in ("role", "label", "description", "instructions")
                if str(reference.get(key, "")).strip()
            }
        )

    return {
        "schema_version": EDIT_PLAN_SCHEMA_VERSION,
        "mode": mode or "edit",
        "target_model": target,
        "original_intent": (base_prompt or "").strip(),
        "changes": _dedupe(changes),
        "preserve": _dedupe(preserve),
        "source_context": source_context,
        "references": normalized_references,
        "mask": mask,
        "lora_trigger_tokens": _dedupe(lora_tokens),
        "lora_guidance": _dedupe(lora_guidance),
        "model_guidance": _dedupe(model_guidance),
        "negative_constraints": _dedupe(negatives),
        "global_rules": (global_rules or "").strip(),
        "warnings": _dedupe(warnings),
    }


def compile_edit_plan(plan: Mapping[str, Any]) -> str:
    target = normalize_model_target(str(plan.get("target_model") or "qwen_edit"))
    parts = [policy_for_model(target), mode_hint(str(plan.get("mode") or "edit"))]

    changes = _string_list(plan.get("changes"))
    if changes:
        parts.append("REQUESTED CHANGE:\n" + _bullets(changes))

    source_lines = _source_context_lines(plan.get("source_context"))
    if source_lines:
        parts.append("SOURCE CONTEXT:\n" + _bullets(source_lines))

    preserve = _string_list(plan.get("preserve"))
    if preserve:
        parts.append("PRESERVE:\n" + _bullets(preserve))

    references = _reference_lines(plan.get("references"))
    if references:
        parts.append("REFERENCE ROLES:\n" + _bullets(references))

    trigger_tokens = _string_list(plan.get("lora_trigger_tokens"))
    lora_guidance = _string_list(plan.get("lora_guidance"))
    if trigger_tokens or lora_guidance:
        lines = []
        if trigger_tokens:
            lines.append("Include these exact trigger tokens once: " + ", ".join(trigger_tokens))
        lines.extend(lora_guidance)
        parts.append("LORA ACTIVATION:\n" + _bullets(lines))

    guidance = _string_list(plan.get("model_guidance"))
    if guidance:
        parts.append("MODEL-SPECIFIC GUIDANCE:\n" + _bullets(guidance))

    negatives = _string_list(plan.get("negative_constraints"))
    if negatives:
        parts.append("AVOID:\n" + _bullets(negatives))

    global_rules = str(plan.get("global_rules") or "").strip()
    if global_rules:
        parts.append("GLOBAL RULES:\n" + global_rules)

    return "\n\n".join(part for part in parts if part)


def mask_summary(mask: Any) -> dict[str, Any]:
    if mask is None:
        return {"provided": False}
    try:
        import numpy as np
        import torch

        array = mask.detach().cpu().float().numpy() if torch.is_tensor(mask) else np.asarray(mask, dtype=float)
        coverage = float(np.clip(array, 0.0, 1.0).mean() * 100.0) if array.size else 0.0
        return {
            "provided": True,
            "coverage_percent": round(coverage, 2),
            "scope": "Change only the masked region; preserve all unmasked pixels and scene content.",
        }
    except Exception:
        return {"provided": True, "scope": "Change only the masked region; preserve everything outside it."}


def public_references(reference_set: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    references = (reference_set or {}).get("references", [])
    result = []
    for reference in references if isinstance(references, list) else []:
        if not isinstance(reference, dict):
            continue
        result.append(
            {
                key: value
                for key, value in reference.items()
                if key != "image" and value not in (None, "", [], {})
            }
        )
    return result


def _resolve_target(selected: str, context: Mapping[str, Any]) -> str:
    if selected and selected != "auto":
        return normalize_model_target(selected)
    return normalize_model_target(str(context.get("target_model") or "qwen_edit"))


def _risk_constraints(risks: list[str]) -> list[str]:
    result = []
    for risk in risks:
        normalized = risk.lower().replace("-", "_").replace(" ", "_")
        if normalized in {"extra_people", "duplicate_subject", "duplicate_bodies"}:
            result.append("Do not add extra people, duplicate subjects, or duplicate body parts.")
        elif normalized in {"text_artifacts", "visible_text"}:
            result.append("Do not add readable text, labels, captions, logos, or watermarks.")
        else:
            result.append(f"Avoid this identified edit risk: {risk}.")
    return result


def _source_context_lines(value: Any) -> list[str]:
    if not isinstance(value, Mapping):
        return []
    labels = {
        "subject_count": "Visible subject count",
        "main_subject": "Main subject",
        "identity_notes": "Identity",
        "pose_notes": "Pose",
        "scene_notes": "Scene",
        "objects_in_hands": "Objects in hands",
        "visible_text": "Visible text",
    }
    lines = []
    for key, label in labels.items():
        item = value.get(key)
        if item in (None, "", [], {}):
            continue
        rendered = ", ".join(_string_list(item)) if isinstance(item, (list, tuple)) else str(item)
        lines.append(f"{label}: {rendered}")
    return lines


def _reference_lines(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    lines = []
    for reference in value:
        if not isinstance(reference, Mapping):
            continue
        role = str(reference.get("role") or "reference")
        label = str(reference.get("label") or role)
        details = [str(reference.get(key)) for key in ("description", "instructions") if reference.get(key)]
        suffix = "; ".join(details)
        lines.append(f"{label} ({role})" + (f": {suffix}" if suffix else ""))
    return lines


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))


__all__ = [
    "EDIT_PLAN_SCHEMA_VERSION",
    "build_edit_plan",
    "compile_edit_plan",
    "mask_summary",
    "public_references",
]
