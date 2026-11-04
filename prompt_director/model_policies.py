from __future__ import annotations


QWEN_EDIT_POLICY = """QWEN IMAGE EDIT POLICY:
Create exactly one main person unless the user explicitly asks for more.
Do not add extra people, selfie companions, duplicate bodies, crowds, signs, placards, captions, text, writing, logos, or watermarks.
Empty hands unless the user specifically asks for an object.
Preserve the source/reference identity, body proportions, pose intent, and camera framing.
Use reference images as appearance guidance only and only for their assigned role; do not insert them as additional subjects."""

KREA2_POLICY = """KREA2 PROMPT POLICY:
Describe the intended scene composition, subject placement, lighting, material detail, and realism/style target.
Favor clear spatial layout and visual specificity.
Avoid strict edit-locking language unless the prompt is explicitly image-edit driven."""

VIDEO_POLICY = """VIDEO PROMPT POLICY:
Describe temporal motion, camera movement, frame continuity, and identity continuity.
Keep the subject consistent across frames and avoid unnecessary identity, wardrobe, or scene changes."""

GENERATION_POLICY = """IMAGE GENERATION POLICY:
Describe the desired final image directly with clear subject, composition, lighting, material, camera, and style details.
Keep required activation tokens exact and avoid contradictory visual instructions."""


def normalize_model_target(model_target: str) -> str:
    normalized = (model_target or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "qwen": "qwen_edit",
        "qwen_image_edit": "qwen_edit",
        "qwen_image_edit_2511": "qwen_edit",
        "krea": "krea2",
        "krea2t": "krea2",
        "ltx_video": "ltx",
        "wan_video": "wan",
        "flux_dev": "flux",
        "flux_schnell": "flux",
        "sd_xl": "sdxl",
        "zimage": "z_image",
    }
    return aliases.get(normalized, normalized or "qwen_edit")


def policy_for_model(model_target: str) -> str:
    normalized = normalize_model_target(model_target)
    if normalized == "qwen_edit":
        return QWEN_EDIT_POLICY
    if normalized == "krea2":
        return KREA2_POLICY
    if normalized in {"ltx", "wan"}:
        return VIDEO_POLICY
    return GENERATION_POLICY


def mode_hint(mode: str) -> str:
    normalized = (mode or "").strip().lower()
    if normalized == "generate":
        return "MODE: generation prompt; describe the desired final image directly."
    if normalized == "video":
        return "MODE: video prompt; preserve continuity across frames."
    return "MODE: image edit prompt; describe only the intended change and preservation constraints."


__all__ = [
    "GENERATION_POLICY",
    "KREA2_POLICY",
    "QWEN_EDIT_POLICY",
    "VIDEO_POLICY",
    "mode_hint",
    "normalize_model_target",
    "policy_for_model",
]
