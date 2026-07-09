from __future__ import annotations

from copy import deepcopy
import json
import re
from typing import Any


DEFAULT_ANALYSIS: dict[str, Any] = {
    "subject_count": 0,
    "main_subject": "",
    "identity_notes": "",
    "pose_notes": "",
    "scene_notes": "",
    "objects_in_hands": [],
    "visible_text": False,
    "risks": [],
    "qwen_edit_prompt": "",
    "krea_prompt": "",
    "ltx_motion_prompt": "",
    "final_prompt": "",
    "warnings": [],
}


def parse_provider_analysis(raw: str | dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(raw, dict):
        return normalize_analysis(raw)

    if raw is None or not str(raw).strip():
        analysis = _default()
        analysis["warnings"].append("provider_empty_response")
        return analysis

    text = _strip_json_fence(str(raw).strip())
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        analysis = _default()
        analysis["warnings"].append("provider_invalid_json")
        return analysis

    if not isinstance(parsed, dict):
        analysis = _default()
        analysis["warnings"].append("provider_json_not_object")
        return analysis

    return normalize_analysis(parsed)


def normalize_analysis(data: dict[str, Any] | None) -> dict[str, Any]:
    analysis = _default()
    if not isinstance(data, dict):
        return analysis

    for key in analysis:
        if key in data:
            analysis[key] = data[key]

    analysis["subject_count"] = _coerce_subject_count(analysis["subject_count"])
    analysis["visible_text"] = _coerce_bool(analysis["visible_text"])
    analysis["objects_in_hands"] = _coerce_list(analysis["objects_in_hands"])
    analysis["risks"] = _coerce_list(analysis["risks"])
    analysis["warnings"] = _coerce_list(analysis["warnings"])

    for key in (
        "main_subject",
        "identity_notes",
        "pose_notes",
        "scene_notes",
        "qwen_edit_prompt",
        "krea_prompt",
        "ltx_motion_prompt",
        "final_prompt",
    ):
        analysis[key] = "" if analysis[key] is None else str(analysis[key])

    return analysis


def merge_analysis(
    manual_analysis_json: str | None,
    provider_analysis: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    merged = normalize_analysis(provider_analysis or {})
    warnings: list[str] = []

    if manual_analysis_json and manual_analysis_json.strip():
        try:
            manual = json.loads(manual_analysis_json)
        except json.JSONDecodeError:
            warnings.append("invalid_analysis_json")
        else:
            if isinstance(manual, dict):
                normalized_manual = normalize_analysis(manual)
                for key, value in normalized_manual.items():
                    if _has_meaningful_value(value):
                        if key == "risks":
                            merged[key] = _dedupe(merged[key] + value)
                        elif key == "warnings":
                            merged[key] = _dedupe(merged[key] + value)
                        else:
                            merged[key] = value
            else:
                warnings.append("analysis_json_not_object")

    warnings.extend(merged.get("warnings", []))
    merged["risks"] = _dedupe(merged.get("risks", []))
    merged["warnings"] = _dedupe(merged.get("warnings", []))
    return merged, _dedupe(warnings)


def _strip_json_fence(text: str) -> str:
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    return fence.group(1).strip() if fence else text


def _default() -> dict[str, Any]:
    return deepcopy(DEFAULT_ANALYSIS)


def _coerce_subject_count(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return max(0, int(value))
    if isinstance(value, str):
        try:
            return max(0, int(float(value.strip())))
        except ValueError:
            return 0
    return 0


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "visible"}
    if isinstance(value, (int, float)):
        return value != 0
    return False


def _coerce_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    return [str(value)]


def _has_meaningful_value(value: Any) -> bool:
    if value in ("", None, False, 0):
        return False
    if isinstance(value, list) and len(value) == 0:
        return False
    return True


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        key = str(item).strip()
        if key and key not in seen:
            seen.add(key)
            result.append(key)
    return result
