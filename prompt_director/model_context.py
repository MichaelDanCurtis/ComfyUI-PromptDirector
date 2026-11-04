from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping

from .diagnostics import diagnostics_from_warnings, make_diagnostic, merge_diagnostics
from .lora_metadata import read_safetensors_metadata
from .model_policies import normalize_model_target


MODEL_CONTEXT_SCHEMA_VERSION = "1.0"

_FAMILY_HINTS = (
    ("qwen_image_edit", "qwen_edit"),
    ("qwen-image-edit", "qwen_edit"),
    ("qwen image edit", "qwen_edit"),
    ("qwen_image", "qwen_image"),
    ("qwen-image", "qwen_image"),
    ("krea2", "krea2"),
    ("krea", "krea2"),
    ("ltx_video", "ltx"),
    ("ltx-video", "ltx"),
    ("ltx", "ltx"),
    ("wan_video", "wan"),
    ("wan (video)", "wan"),
    ("wan", "wan"),
    ("flux", "flux"),
    ("sdxl", "sdxl"),
    ("sd_xl", "sdxl"),
    ("z_image", "z_image"),
    ("z-image", "z_image"),
    ("lumina", "z_image"),
)


def read_asset_metadata(
    path: str | Path,
    *,
    category: str = "",
    name: str = "",
) -> dict[str, Any]:
    """Read the canonical Model Explorer namespaces without requiring its package."""
    file_path = Path(path)
    raw = read_safetensors_metadata(file_path)
    model_card = _parse_namespace(raw, "model_card")
    prompt_director = _parse_namespace(raw, "prompt_director")
    modelspec = {key: value for key, value in raw.items() if key.startswith("modelspec.")}
    classification: dict[str, Any] = {}
    try:
        from safetensors_metadata import classify  # type: ignore

        classified = classify(str(file_path))
        if isinstance(classified, dict):
            classification = classified
    except Exception:
        pass

    classification_fields = {
        "asset_type": classification.get("asset_type"),
        "base_model": classification.get("base_model"),
        "rank": classification.get("rank"),
        "precision": classification.get("precision"),
    }
    for key, value in classification_fields.items():
        if model_card.get(key) in (None, "", [], {}) and value not in (None, "", [], {}):
            model_card[key] = value

    if not model_card.get("display_name") and modelspec.get("modelspec.title"):
        model_card["display_name"] = modelspec["modelspec.title"]
    if not model_card.get("architecture") and modelspec.get("modelspec.architecture"):
        model_card["architecture"] = modelspec["modelspec.architecture"]
    if not model_card.get("base_model"):
        base_hint = raw.get("ss_base_model_version") or modelspec.get("modelspec.architecture")
        if base_hint:
            model_card["base_model"] = base_hint

    return {
        "category": category,
        "name": name or file_path.name,
        "model_card": model_card,
        "prompt_director": prompt_director,
        "modelspec": modelspec,
        "classify": classification,
    }


def resolve_asset_metadata(category: str, name: str) -> dict[str, Any]:
    if not name or name.strip().lower() in {"none", "disabled"}:
        return {}
    import folder_paths  # type: ignore

    path = folder_paths.get_full_path_or_raise(category, name)
    return read_asset_metadata(path, category=category, name=name)


def parse_lora_stack_json(raw_json: str) -> tuple[list[dict[str, Any]], list[str]]:
    raw = (raw_json or "").strip()
    if not raw:
        return [], []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return [], [f"invalid_lora_stack_json:{exc.msg}"]
    if not isinstance(parsed, list):
        return [], ["invalid_lora_stack_json:not_array"]

    entries: list[dict[str, Any]] = []
    warnings: list[str] = []
    for index, item in enumerate(parsed):
        if isinstance(item, str):
            entries.append({"name": item, "strength_model": 1.0, "strength_clip": 1.0})
            continue
        if not isinstance(item, dict) or not str(item.get("name", "")).strip():
            warnings.append(f"invalid_lora_stack_entry:{index}")
            continue
        entries.append(
            {
                "name": str(item["name"]).strip(),
                "strength_model": _number(item.get("strength_model"), 1.0),
                "strength_clip": _number(item.get("strength_clip"), 1.0),
            }
        )
    return entries, warnings


def load_prompt_context(
    *,
    model_category: str = "diffusion_models",
    model_name: str = "",
    lora_stack_json: str = "",
    context_json: str = "",
    model_target: str = "auto",
) -> tuple[dict[str, Any], list[str]]:
    if context_json and context_json.strip():
        try:
            parsed = json.loads(context_json)
        except json.JSONDecodeError as exc:
            return {}, [f"invalid_model_context_json:{exc.msg}"]
        if not isinstance(parsed, dict):
            return {}, ["invalid_model_context_json:not_object"]
        context = deepcopy(parsed)
        if model_target and model_target != "auto":
            resolution = resolve_model_target(context.get("model"), requested=model_target)
            context["target_model"] = resolution["resolved"]
            context["target"] = resolution
        elif not isinstance(context.get("target"), dict):
            existing_target = str(context.get("target_model") or "").strip()
            if existing_target:
                context["target"] = {
                    "requested": "auto",
                    "resolved": normalize_model_target(existing_target),
                    "source": "context.target_model",
                    "confidence": 1.0,
                }
            else:
                resolution = resolve_model_target(context.get("model"), requested="auto")
                context["target_model"] = resolution["resolved"]
                context["target"] = resolution
        warnings = _string_list(context.get("warnings"))
        if not isinstance(context.get("diagnostics"), list):
            context["diagnostics"] = diagnostics_from_warnings(warnings)
        return context, warnings

    warnings: list[str] = []
    model_asset: dict[str, Any] = {}
    if model_name and model_name.strip().lower() not in {"none", "disabled"}:
        try:
            model_asset = resolve_asset_metadata(model_category, model_name)
        except Exception as exc:
            warnings.append(f"model_metadata_unavailable:{type(exc).__name__}:{exc}")

    stack, stack_warnings = parse_lora_stack_json(lora_stack_json)
    warnings.extend(stack_warnings)
    lora_assets: list[dict[str, Any]] = []
    for entry in stack:
        try:
            asset = resolve_asset_metadata("loras", entry["name"])
        except Exception as exc:
            warnings.append(f"lora_metadata_unavailable:{entry['name']}:{type(exc).__name__}:{exc}")
            asset = {
                "category": "loras",
                "name": entry["name"],
                "model_card": {},
                "prompt_director": {},
                "modelspec": {},
            }
        asset.update(
            strength_model=entry["strength_model"],
            strength_clip=entry["strength_clip"],
        )
        lora_assets.append(asset)

    context, context_warnings = build_prompt_context(
        model_asset=model_asset,
        lora_assets=lora_assets,
        model_target=model_target,
    )
    merged_warnings = _dedupe(warnings + context_warnings)
    context["warnings"] = merged_warnings
    context["diagnostics"] = merge_diagnostics(
        list(context.get("diagnostics") or []),
        diagnostics_from_warnings(warnings),
    )
    return context, merged_warnings


def build_prompt_context(
    *,
    model_asset: Mapping[str, Any] | None = None,
    lora_assets: list[Mapping[str, Any]] | None = None,
    model_target: str = "auto",
) -> tuple[dict[str, Any], list[str]]:
    model = deepcopy(dict(model_asset or {}))
    inferred_target = infer_model_target(model)
    target_resolution = resolve_model_target(model, requested=model_target)
    selected_target = target_resolution["resolved"]
    warnings: list[str] = []
    diagnostics: list[dict[str, Any]] = []
    if model and not inferred_target:
        warning = f"model_family_unknown:{model.get('name', 'selected_model')}"
        warnings.append(warning)
        diagnostics.append(
            make_diagnostic(
                "model_family_unknown",
                "The selected model has no recognized family metadata; the configured fallback target remains active.",
                severity="warning",
                subject={"asset_name": str(model.get("name") or "selected_model")},
                evidence={"target_resolution": target_resolution, "legacy_warning": warning},
            )
        )
    if not model:
        warning = "model_context_missing"
        warnings.append(warning)
        diagnostics.append(
            make_diagnostic(
                "model_context_missing",
                "No model metadata is connected; Prompt Director is using the requested or default target policy.",
                severity="info",
                evidence={"target_resolution": target_resolution, "legacy_warning": warning},
            )
        )

    normalized_loras: list[dict[str, Any]] = []
    required_tokens: list[str] = []
    for raw_asset in lora_assets or []:
        asset = deepcopy(dict(raw_asset))
        name = str(asset.get("name") or "unnamed_lora")
        prompt = asset.get("prompt_director") if isinstance(asset.get("prompt_director"), dict) else {}
        model_card = asset.get("model_card") if isinstance(asset.get("model_card"), dict) else {}
        strength_model = _number(asset.get("strength_model"), 1.0)
        strength_clip = _number(asset.get("strength_clip"), 1.0)
        compatibility_detail = _compatibility_for_lora(model_card, prompt, selected_target, model)
        compatibility = compatibility_detail["status"]
        if compatibility == "incompatible":
            warning = f"lora_incompatible:{name}:{selected_target}"
            warnings.append(warning)
            diagnostics.append(
                make_diagnostic(
                    "lora_incompatible",
                    f"{name} metadata does not list compatibility with {selected_target}.",
                    subject={"asset_name": name},
                    evidence={**compatibility_detail, "legacy_warning": warning},
                )
            )

        strength_guidance = _strength_guidance(model_card, prompt, selected_target)
        minimum = _optional_number(strength_guidance.get("strength_min"))
        maximum = _optional_number(strength_guidance.get("strength_max"))
        if minimum is not None and strength_model < minimum:
            warning = f"lora_strength_below_min:{name}:{strength_model}:{minimum}"
            warnings.append(warning)
            diagnostics.append(
                make_diagnostic(
                    "lora_strength_below_min",
                    f"{name} model strength {strength_model} is below the tested minimum {minimum}.",
                    subject={"asset_name": name},
                    evidence={
                        "actual": strength_model,
                        "minimum": minimum,
                        "guidance_source": (strength_guidance.get("field_sources") or {}).get(
                            "strength_min", strength_guidance.get("source", "")
                        ),
                        "legacy_warning": warning,
                    },
                )
            )
        if maximum is not None and strength_model > maximum:
            warning = f"lora_strength_above_max:{name}:{strength_model}:{maximum}"
            warnings.append(warning)
            diagnostics.append(
                make_diagnostic(
                    "lora_strength_above_max",
                    f"{name} model strength {strength_model} is above the tested maximum {maximum}.",
                    subject={"asset_name": name},
                    evidence={
                        "actual": strength_model,
                        "maximum": maximum,
                        "guidance_source": (strength_guidance.get("field_sources") or {}).get(
                            "strength_max", strength_guidance.get("source", "")
                        ),
                        "legacy_warning": warning,
                    },
                )
            )

        triggers = _string_list(prompt.get("trigger_tokens"))
        required_tokens.extend(triggers)
        normalized_loras.append(
            {
                **asset,
                "name": name,
                "strength_model": strength_model,
                "strength_clip": strength_clip,
                "compatibility": compatibility,
                "compatibility_detail": compatibility_detail,
                "strength_guidance": strength_guidance,
                "trigger_tokens": triggers,
            }
        )

    context = {
        "schema_version": MODEL_CONTEXT_SCHEMA_VERSION,
        "target_model": selected_target,
        "target": target_resolution,
        "model": model,
        "loras": normalized_loras,
        "required_trigger_tokens": _dedupe(required_tokens),
        "warnings": _dedupe(warnings),
        "diagnostics": merge_diagnostics(diagnostics, diagnostics_from_warnings(warnings)),
    }
    return context, context["warnings"]


def infer_model_target(asset: Mapping[str, Any] | None) -> str:
    resolution = _inferred_target_resolution(asset)
    return resolution["resolved"] if resolution else ""


def resolve_model_target(
    asset: Mapping[str, Any] | None,
    *,
    requested: str = "auto",
) -> dict[str, Any]:
    selected = str(requested or "auto")
    if selected != "auto":
        return {
            "requested": selected,
            "resolved": normalize_model_target(selected),
            "source": "user_selection",
            "confidence": 1.0,
        }
    inferred = _inferred_target_resolution(asset)
    if inferred:
        return {"requested": "auto", **inferred}
    return {
        "requested": "auto",
        "resolved": "qwen_edit",
        "source": "default:qwen_edit",
        "confidence": 0.0,
    }


def _inferred_target_resolution(asset: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not asset:
        return None
    card = asset.get("model_card") if isinstance(asset.get("model_card"), dict) else {}
    prompt = asset.get("prompt_director") if isinstance(asset.get("prompt_director"), dict) else {}
    modelspec = asset.get("modelspec") if isinstance(asset.get("modelspec"), dict) else {}
    sources = (
        ("model_card.model_family", card.get("model_family"), 1.0),
        ("prompt_director.model_family", prompt.get("model_family"), 0.95),
        ("model_card.compatible_families", card.get("compatible_families"), 0.9),
        ("model_card.base_model", card.get("base_model"), 0.85),
        ("model_card.architecture", card.get("architecture"), 0.8),
        ("modelspec.architecture", modelspec.get("modelspec.architecture"), 0.75),
        ("asset.name", asset.get("name"), 0.5),
    )
    for source, value, confidence in sources:
        for candidate in _string_list(value):
            lowered = candidate.lower()
            for hint, target in _FAMILY_HINTS:
                if hint in lowered:
                    return {
                        "resolved": target,
                        "source": source,
                        "confidence": confidence,
                        "evidence": candidate,
                    }
    return None


def public_context(context: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return JSON-safe context and strip accidental runtime-only image tensors."""
    return _json_safe(deepcopy(dict(context or {})))


def _parse_namespace(raw: Mapping[str, Any], namespace: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    packed = raw.get(namespace)
    if packed:
        parsed = _parse_value(packed)
        if isinstance(parsed, dict):
            result.update(parsed)
    prefix = f"{namespace}."
    for key, value in raw.items():
        if not str(key).startswith(prefix):
            continue
        field = str(key)[len(prefix) :]
        if field not in result:
            result[field] = _parse_value(value)
    return result


def _compatibility_for_lora(
    card: Mapping[str, Any],
    prompt: Mapping[str, Any],
    target: str,
    model: Mapping[str, Any],
) -> dict[str, Any]:
    families = _dedupe(
        _string_list(card.get("model_family"))
        + _string_list(card.get("compatible_families"))
        + _string_list(prompt.get("model_family"))
    )
    normalized = {normalize_model_target(value) for value in families}
    if normalized:
        status = "compatible" if target in normalized else "incompatible"
        return {
            "status": status,
            "reason": "explicit_family_match" if status == "compatible" else "explicit_family_mismatch",
            "evidence": sorted(families),
            "target_model": target,
            "advisory": True,
        }

    compatible_models = _string_list(card.get("compatible_base_models")) + _string_list(
        prompt.get("compatible_base_models")
    )
    if compatible_models:
        model_text = " ".join(
            str(value)
            for value in (
                model.get("name"),
                (model.get("model_card") or {}).get("display_name") if isinstance(model.get("model_card"), dict) else "",
                (model.get("model_card") or {}).get("base_model") if isinstance(model.get("model_card"), dict) else "",
            )
            if value
        ).lower()
        status = "compatible" if any(item.lower() in model_text for item in compatible_models) else "incompatible"
        return {
            "status": status,
            "reason": "explicit_base_model_match" if status == "compatible" else "explicit_base_model_mismatch",
            "evidence": compatible_models,
            "target_model": target,
            "advisory": True,
        }
    return {
        "status": "unknown",
        "reason": "no_explicit_compatibility_metadata",
        "evidence": [],
        "target_model": target,
        "advisory": True,
    }


def _strength_guidance(
    card: Mapping[str, Any],
    prompt: Mapping[str, Any],
    target: str,
) -> dict[str, Any]:
    override = _family_override(card.get("family_weight_overrides"), target)
    override_source = f"model_card.family_weight_overrides.{override[0]}" if override else ""
    values = override[1] if override else {}
    guidance: dict[str, Any] = {"source": override_source or "prompt_director"}
    field_sources: dict[str, str] = {}
    field_aliases = {
        "default_strength_model": ("default_strength_model", "model"),
        "default_strength_clip": ("default_strength_clip", "clip"),
        "strength_min": ("strength_min", "min"),
        "strength_max": ("strength_max", "max"),
    }
    for output_field, aliases in field_aliases.items():
        value = next((values.get(alias) for alias in aliases if values.get(alias) not in (None, "")), None)
        source = override_source if value not in (None, "") else ""
        if value in (None, ""):
            value = prompt.get(output_field)
            if value not in (None, ""):
                source = f"prompt_director.{output_field}"
        parsed = _optional_number(value)
        if parsed is not None:
            guidance[output_field] = parsed
            field_sources[output_field] = source
    if field_sources:
        guidance["field_sources"] = field_sources
    return guidance


def _family_override(value: Any, target: str) -> tuple[str, Mapping[str, Any]] | None:
    if not isinstance(value, Mapping):
        return None
    for family, override in value.items():
        if normalize_model_target(str(family)) == target and isinstance(override, Mapping):
            return str(family), override
    return None


def _parse_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _number(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _optional_number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items() if key != "image"}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


__all__ = [
    "MODEL_CONTEXT_SCHEMA_VERSION",
    "build_prompt_context",
    "infer_model_target",
    "load_prompt_context",
    "parse_lora_stack_json",
    "public_context",
    "read_asset_metadata",
    "resolve_model_target",
    "resolve_asset_metadata",
]
