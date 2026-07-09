from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, Mapping


PROMPT_DIRECTOR_PREFIX = "prompt_director."
PROMPT_DIRECTOR_JSON_KEY = "prompt_director"


def read_safetensors_metadata(path: str | Path) -> dict[str, str]:
    file_path = Path(path)
    with file_path.open("rb") as handle:
        header_len_bytes = handle.read(8)
        if len(header_len_bytes) != 8:
            raise ValueError(f"Invalid safetensors file header: {file_path}")
        header_len = struct.unpack("<Q", header_len_bytes)[0]
        header = json.loads(handle.read(header_len).decode("utf-8"))

    metadata = header.get("__metadata__", {}) if isinstance(header, dict) else {}
    if not isinstance(metadata, dict):
        return {}
    return {str(key): str(value) for key, value in metadata.items()}


def extract_prompt_metadata(raw_metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    raw_metadata = raw_metadata or {}
    merged: dict[str, Any] = {}

    packed = raw_metadata.get(PROMPT_DIRECTOR_JSON_KEY)
    if packed:
        parsed = _parse_value(packed)
        if isinstance(parsed, dict):
            merged.update(parsed)

    for key, value in raw_metadata.items():
        if not str(key).startswith(PROMPT_DIRECTOR_PREFIX):
            continue
        short_key = str(key)[len(PROMPT_DIRECTOR_PREFIX) :]
        merged[short_key] = _parse_value(value)

    return _normalize_prompt_metadata(merged)


def parse_lora_metadata_json(raw_json: str) -> tuple[dict[str, Any], list[str]]:
    raw = (raw_json or "").strip()
    if not raw:
        return {}, []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {}, [f"invalid_lora_metadata_json:{exc.msg}"]
    if not isinstance(parsed, dict):
        return {}, ["invalid_lora_metadata_json:not_object"]
    return _normalize_prompt_metadata(parsed), []


def load_lora_prompt_metadata(lora_name: str, lora_metadata_json: str = "") -> tuple[dict[str, Any], list[str]]:
    metadata, warnings = parse_lora_metadata_json(lora_metadata_json)
    if metadata:
        return metadata, warnings

    normalized_name = (lora_name or "none").strip()
    if not normalized_name or normalized_name.lower() in {"none", "disabled"}:
        return {}, warnings

    try:
        import folder_paths  # type: ignore

        lora_path = folder_paths.get_full_path_or_raise("loras", normalized_name)
    except Exception as exc:
        warnings.append(f"lora_metadata_unavailable:{type(exc).__name__}:{exc}")
        return {}, warnings

    try:
        return extract_prompt_metadata(read_safetensors_metadata(lora_path)), warnings
    except Exception as exc:
        warnings.append(f"lora_metadata_unavailable:{type(exc).__name__}:{exc}")
        return {}, warnings


def get_lora_names() -> list[str]:
    try:
        import folder_paths  # type: ignore

        names = list(folder_paths.get_filename_list("loras"))
    except Exception:
        names = []
    return ["none"] + [name for name in names if name != "none"]


def summarize_lora_metadata(lora_name: str, metadata: Mapping[str, Any]) -> str:
    if not metadata:
        return f"LoRA: {lora_name or 'none'}; no Prompt Director metadata found."

    parts = [f"LoRA: {lora_name or metadata.get('display_name') or 'unspecified'}"]
    for key in (
        "display_name",
        "trigger_tokens",
        "activation_phrases",
        "model_family",
        "compatible_base_models",
        "prompt_guidance",
        "negative_prompt_guidance",
        "default_strength_model",
        "default_strength_clip",
    ):
        value = metadata.get(key)
        if value not in (None, "", [], {}):
            parts.append(f"{key}: {json.dumps(value, ensure_ascii=True)}")
    return "; ".join(parts)


def _normalize_prompt_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(metadata)
    for key in (
        "trigger_tokens",
        "activation_phrases",
        "negative_tokens",
        "model_family",
        "compatible_base_models",
        "example_prompts",
        "keyword_aliases",
    ):
        if key in normalized:
            normalized[key] = _string_list(normalized[key])

    for key in ("default_strength_model", "default_strength_clip", "strength_min", "strength_max"):
        if key in normalized:
            normalized[key] = _number_or_original(normalized[key])

    return normalized


def _parse_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return ""
    if text[:1] in "[{\"" or text in {"true", "false", "null"}:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return value
    try:
        return int(text) if text.isdigit() else float(text)
    except ValueError:
        return value


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        parsed = _parse_value(value)
        if isinstance(parsed, list):
            return _string_list(parsed)
        return [value.strip()] if value.strip() else []
    return [str(value).strip()] if str(value).strip() else []


def _number_or_original(value: Any) -> Any:
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value
