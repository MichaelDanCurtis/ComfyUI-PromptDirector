from __future__ import annotations

import json

try:
    from .prompt_director.analysis import merge_analysis
    from .prompt_director.image_utils import image_to_png_data_uri
    from .prompt_director.lora_metadata import (
        get_lora_names,
        load_lora_prompt_metadata,
        parse_lora_metadata_json,
        summarize_lora_metadata,
    )
    from .prompt_director.policy import build_prompt_director_result
    from .prompt_director.providers import ProviderUnavailable, get_provider
except ImportError:
    from prompt_director.analysis import merge_analysis
    from prompt_director.image_utils import image_to_png_data_uri
    from prompt_director.lora_metadata import (
        get_lora_names,
        load_lora_prompt_metadata,
        parse_lora_metadata_json,
        summarize_lora_metadata,
    )
    from prompt_director.policy import build_prompt_director_result
    from prompt_director.providers import ProviderUnavailable, get_provider


PROMPT_DIRECTOR_DESCRIPTION = """
Prompt Director builds the exact final prompt string that should feed a model text encoder.

Primary use case:
Use this immediately before Qwen Image Edit 2511 text encoding, especially before
TextEncodeQwenImageEditPlus. The node is meant to replace hidden prompt-injection widgets,
buried text boxes, and stale prompt rails with a visible final_prompt output that can be routed
through a viewer before the conditioning node. The point is operational: you should be able to
open the workflow, click the node, inspect the info tab, inspect the rendered prompt text, and
know exactly what Qwen is receiving.

What it does:
1. Receives a human base prompt, model target, optional image-analysis JSON, optional global
   rules, and optionally a source/reference image plus provider config.
2. Applies model-specific prompt policy for Qwen Image Edit, Krea2, LTX, or WAN.
3. Returns three plain strings: final_prompt, normalized analysis_json, and warnings.
4. Keeps the model-facing language deterministic, so repeated runs with the same inputs produce
   the same prompt text when provider analysis is disabled.
5. Converts provider or manual analysis into compact policy hints without forcing the render to
   fail just because external image analysis is unavailable.

Recommended wiring:
base_prompt STRING -> Prompt Director
Prompt Director.final_prompt -> Show Text / prompt viewer
prompt viewer STRING -> TextEncodeQwenImageEditPlus prompt input
TextEncodeQwenImageEditPlus CONDITIONING -> sampler positive input

For Qwen Image Edit 2511, keep the viewer in the graph. It is the audit point that proves Qwen
receives this node's output rather than a stale widget value, an old group input, or a prompt
string saved inside a different node. If you branch to fast and slow Qwen encoders, route the
same final_prompt output to both branches unless you intentionally want two different policies.

Inputs:
base_prompt is the user's direct edit request. Keep it short and literal: "remove the logo from
the shirt", "make the jacket red", "replace the background with a clean studio wall". Put durable
workflow rules in global_rules instead of repeating them in every base prompt.

model_target selects the policy family. qwen_edit is strict about edit preservation and artifact
avoidance. krea2 is broader and more generative. ltx and wan are video-oriented targets that add
motion, frame continuity, and temporal identity language.

mode is a coarse intent flag. edit is the normal value for Qwen Image Edit. generate is useful
when the text goes to an image-generation model instead of an edit model. video is for text
consumers where motion and continuity matter more than pixel-level source preservation.

analysis_json lets another node, a manual note, or a provider describe what is visible in the
image. It must be a JSON object when provided. Useful keys include subject_count, main_subject,
visible_text, objects_in_hands, risks, style_notes, composition_notes, and preservation_notes.
Invalid JSON does not stop the graph; it becomes an empty object and adds invalid_analysis_json
to warnings so the failure is visible.

global_rules is for project-level constraints that should always apply. Examples: "do not add
extra people", "do not create readable text", "treat reference images as style guidance only",
"preserve identity, pose, clothing fit, camera angle, and body proportions unless explicitly
changed". Use this for repeatable guardrails, not for one-off edit intent.

image is optional. It is only inspected when a Prompt Director Config node enables a provider.
Without a provider config, the image input is ignored and no network call happens.

config is optional. It carries provider settings from Prompt Director Config, never API keys.
Secrets stay in a local credentials profile or environment variables so they do not get serialized
into workflow JSON.

Outputs:
final_prompt is the exact text that should go to the model text encoder. If the image edit is
misbehaving, inspect this first. It should contain the base prompt, global rules when supplied,
mode language, and the selected model policy.

analysis_json is normalized JSON after merging manual analysis and provider analysis. This output
is useful for debugging because it shows what policy inputs the node actually used.

warnings is a JSON array string. Typical warning values include extra_people, text_artifacts,
object_in_hands, invalid_analysis_json, provider_requires_image, provider_unavailable, and any
explicit risks carried in analysis_json. Route it to a text viewer when debugging prompt behavior.

Qwen policy:
The qwen_edit target adds strict edit language for common Qwen Image Edit failure modes: extra
people, duplicated bodies, reference images becoming additional subjects, signs, captions,
writing, logos, watermarks, and accidental objects in hands. It also preserves identity, body
proportions, pose intent, source/reference composition, clothing fit, and camera framing.
Reference images are described as appearance guidance only, not as new people or new objects to
insert into the edited image.

Krea2 policy:
The krea2 target favors scene composition, spatial layout, material detail, realism, lighting,
and style specificity. It avoids the tight preservation language used for edit models unless you
explicitly provide those constraints in the base prompt or global rules.

LTX and WAN policy:
The ltx and wan targets emphasize temporal motion, camera movement, frame continuity, action
readability, and identity continuity. Use these when the next text consumer is a video prompt
input rather than an image-edit text encoder.

What to verify:
1. final_prompt is connected to the actual TextEncodeQwenImageEditPlus prompt input.
2. A prompt viewer displays the same final_prompt text before the Qwen encoder.
3. The text encoder's old widget prompt is no longer the source of truth.
4. warnings is empty or contains expected, explainable values.
5. analysis_json reflects the connected analysis source instead of stale copied JSON.
6. The workflow JSON does not contain provider API keys.

Troubleshooting:
If edits still add people, check subject_count and risks in analysis_json, then add a direct
global rule such as "do not add extra people or duplicate the subject". If Qwen adds signs,
captions, labels, logos, or watermarks, check for visible_text or text_artifacts warnings and
keep the qwen_edit target selected. If the provider warning says provider_requires_image, connect
an image or set Prompt Director Config back to none. If provider_unavailable appears, inspect the
local credentials profile, provider base URL, provider model, or environment variables for the selected
provider. If the final prompt viewer does not change when the base prompt changes, the graph is likely
still wired around this node.

Secrets and providers:
This node intentionally does not expose API-key widgets. Do not paste API keys into the workflow.
Provider-backed image analysis reads secrets from a local credentials profile or environment variables.
Use Prompt Director Config to enable openai_compatible, zai, glm, kimi, or grok analysis. For the recommended local
profile path, the workflow stores only the provider, profile name, base URL, and model; the API key remains
in the ComfyUI user folder. When a provider is unavailable or misconfigured, the graph still returns
deterministic prompt policy and adds a warning such as provider_unavailable or provider_requires_image.

Limitations:
Provider-backed image inspection requires a connected image, provider config, and environment
variables. Without those, Prompt Director still works from base_prompt and analysis_json. Z.ai
and Kimi endpoint/model settings can be supplied from Prompt Director Config, a local profile, or
environment variables because those service settings vary by account and deployment. This node is a
prompt-policy and audit node; it does not replace the sampler, model loader, LoRA loader, or Qwen text
encoder.
""".strip()


class PromptDirector:
    DESCRIPTION = PROMPT_DIRECTOR_DESCRIPTION
    SEARCH_ALIASES = [
        "ComfyAI",
        "Prompt Policy",
        "Qwen Prompt",
        "Qwen Image Edit",
        "Prompt Injection",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": (
                            "The user's plain-language edit or generation intent before policy injection. "
                            "For Qwen edits, describe the desired change literally and keep reusable guardrails "
                            "in global_rules so the final_prompt remains easy to audit in a text viewer."
                        ),
                    },
                ),
                "model_target": (
                    ["qwen_edit", "krea2", "ltx", "wan"],
                    {
                        "default": "qwen_edit",
                        "tooltip": (
                            "Selects the prompt policy family. qwen_edit adds strict Qwen Image Edit preservation "
                            "and artifact-avoidance language; krea2 favors composition and realism; ltx and wan "
                            "emphasize motion, continuity, and video prompt readability."
                        ),
                    },
                ),
                "mode": (
                    ["edit", "generate", "video"],
                    {
                        "default": "edit",
                        "tooltip": (
                            "Adds a high-level mode hint to the final prompt. Use edit for image editing paths "
                            "such as Qwen Image Edit, generate for direct image generation, and video when the "
                            "next text consumer expects temporal or camera-motion language."
                        ),
                    },
                ),
            },
            "optional": {
                "analysis_json": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": (
                            "Optional structured image/context analysis as a JSON object. Useful keys include "
                            "subject_count, main_subject, visible_text, objects_in_hands, risks, style_notes, "
                            "composition_notes, and preservation_notes; invalid JSON becomes a warning."
                        ),
                    },
                ),
                "global_rules": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": (
                            "Optional reusable rules inserted before model policy. Use this for durable workflow "
                            "guardrails such as no extra people, no readable text, preserve identity, and treat "
                            "references as appearance guidance only; leave blank for one-off tests."
                        ),
                    },
                ),
                "image": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Optional source/reference image for provider-backed visual analysis. It is inspected "
                            "only when PromptDirectorConfig selects a provider; otherwise no network call happens "
                            "and deterministic prompt policy still runs from the text inputs."
                        ),
                    },
                ),
                "config": (
                    "PROMPT_DIRECTOR_CONFIG",
                    {
                        "tooltip": (
                            "Optional Prompt Director Config output. Use it to enable provider-backed image "
                            "analysis while keeping API keys in environment variables instead of serialized "
                            "workflow widgets; set provider to none for offline deterministic behavior."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("final_prompt", "analysis_json", "warnings")
    OUTPUT_TOOLTIPS = (
        "The exact model-facing prompt string that should feed the text encoder. Route this through a text viewer before TextEncodeQwenImageEditPlus so the workflow visibly proves which prompt Qwen receives.",
        "Normalized analysis JSON after merging manual analysis and optional provider analysis. Invalid or missing JSON becomes an empty object so the graph keeps running while warnings explain the issue.",
        "JSON array string of policy and provider warnings such as extra_people, text_artifacts, provider_requires_image, provider_unavailable, or invalid_analysis_json; route to a viewer during debugging.",
    )
    FUNCTION = "run"
    CATEGORY = "Prompt Director"

    def run(
        self,
        base_prompt: str,
        model_target: str,
        mode: str = "edit",
        analysis_json: str = "",
        global_rules: str = "",
        image=None,
        config=None,
    ):
        provider_analysis = {}
        provider_warnings = []

        if config and image is not None and config.get("analysis_provider", "none") != "none":
            try:
                provider = get_provider(config.get("analysis_provider", "none"), config=config)
                provider_analysis = provider.analyze_image(
                    image_data_uri=image_to_png_data_uri(image),
                    base_prompt=base_prompt,
                    model_target=model_target,
                )
            except ProviderUnavailable as exc:
                provider_warnings.append(f"provider_unavailable:{exc}")
            except Exception as exc:
                provider_warnings.append(f"provider_error:{type(exc).__name__}:{exc}")
        elif config and config.get("analysis_provider", "none") != "none" and image is None:
            provider_warnings.append("provider_requires_image")

        merged_analysis, merge_warnings = merge_analysis(analysis_json, provider_analysis)
        result = build_prompt_director_result(
            base_prompt=base_prompt,
            model_target=model_target,
            analysis_json=json.dumps(merged_analysis, ensure_ascii=True),
            mode=mode,
            global_rules=global_rules,
            extra_warnings=provider_warnings + merge_warnings,
        )
        return (
            result.final_prompt,
            result.analysis_json,
            json.dumps(result.warnings, ensure_ascii=True),
        )


class PromptDirectorConfig:
    DESCRIPTION = """
Configure optional provider-backed image analysis for Prompt Director.

Purpose:
Prompt Director can work fully offline from base_prompt, global_rules, and analysis_json. This Config
node only exists when you want a vision provider to inspect the connected image and return structured
analysis before the prompt policy is built. Keep it separate from Prompt Director so ordinary workflows
do not accidentally make network calls.

When to use none:
Use analysis_provider = none for the normal deterministic path, for local testing, for workflows you plan
to share, or when you already provide analysis_json from another node. In none mode, Prompt Director does
not inspect the image input, does not call any external service, and produces repeatable prompt text from
the visible workflow inputs.

Provider behavior:
openai_compatible sends the connected image as a data URI to a chat-completions-compatible endpoint and
expects a small JSON object describing subject count, visible text, objects in hands, risks, and relevant
preservation notes. zai, glm, and kimi follow the same public chat-completions contract. grok uses xAI's
Responses API because its current image-understanding examples use input_image/input_text payloads.
Provider failures do not stop the graph; Prompt Director falls back to deterministic policy and emits warnings.

Local profile credentials:
This is the recommended interface-first setup. Set credential_source to local_profile, set
credential_profile to a name such as default, then put only the API key in the local credentials file:
user/__credentials/provider_profiles.json inside the active ComfyUI user directory. This uses ComfyUI's
protected system-user directory convention and is blocked from the normal /userdata HTTP endpoints. The
workflow stores the profile name, base URL, and model, but it does not store the secret key. Prompt Director
also reads the older user/prompt_director/credentials.json and user/default/prompt_director/credentials.json
locations as fallback paths when the shared credentials file is missing.

Example credentials file:
{
  "profiles": {
    "default": {
      "kimi": {
        "api_key": "YOUR_KIMI_OR_MOONSHOT_KEY"
      },
      "glm": {
        "api_key": "YOUR_GLM_OR_ZAI_KEY"
      },
      "grok": {
        "api_key": "YOUR_XAI_KEY"
      }
    }
  }
}

The profile entry may also contain base_url and model, but visible UI values in this node take precedence.
That lets you keep the key local while still editing Kimi endpoint/model behavior in ComfyUI.

Environment variables:
Environment variables are still supported for launchers and headless runs. openai_compatible requires
OPENAI_API_KEY. Optional settings are OPENAI_API_BASE and PROMPT_DIRECTOR_OPENAI_MODEL.

glm and zai accept GLM_API_KEY, Z_AI_API_KEY, ZAI_API_KEY, or BIGMODEL_API_KEY. The default base URL is
https://api.z.ai/api/paas/v4 and the default vision model is glm-5v-turbo.

kimi requires KIMI_API_KEY or MOONSHOT_API_KEY plus provider-specific base URL and model variables. The
node does not assume a single Kimi endpoint.

grok accepts XAI_API_KEY or GROK_API_KEY. The default base URL is https://api.x.ai/v1 and the default
image-understanding model is grok-4.3.

Secrets:
This node deliberately does not expose API-key fields. API keys must come from a local credentials profile
or environment variables and must not be pasted into widgets, workflow JSON, screenshots, or saved
templates. The config output is a plain dictionary containing provider name, credential source, profile
name, base URL, model, and debug preference only; it is designed to be safe to serialize.

Debug:
The debug flag is reserved for verbose provider diagnostics. Keep it off for normal graph execution. When
debugging provider behavior, also route Prompt Director warnings and analysis_json into text viewers so you
can distinguish provider setup problems from prompt-policy problems.

Recommended wiring:
Prompt Director Config.config -> Prompt Director.config
source IMAGE -> Prompt Director.image
Prompt Director.final_prompt -> prompt viewer -> TextEncodeQwenImageEditPlus prompt

Troubleshooting:
If Prompt Director reports provider_requires_image, connect an image or set analysis_provider back to none.
If it reports provider_unavailable, check credential_source, credential_profile, the local credentials file,
and the provider base/model fields. If provider output appears stale, verify the graph is not reusing cached
text or analysis from an old workflow branch.
""".strip()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "analysis_provider": (
                    ["none", "openai_compatible", "zai", "glm", "kimi", "grok"],
                    {
                        "default": "none",
                        "tooltip": (
                            "Provider used to inspect the connected image before Prompt Director builds policy. "
                            "Choose none for offline deterministic behavior and shared workflows; choose kimi, glm, "
                            "or grok with local_profile when you want UI-configurable image analysis without storing "
                            "the API key inside workflow JSON."
                        ),
                    },
                ),
                "credential_source": (
                    ["local_profile", "environment"],
                    {
                        "default": "local_profile",
                        "tooltip": (
                            "Where the provider API key is resolved from. local_profile reads a named entry from "
                            "user/__credentials/provider_profiles.json in ComfyUI's protected system-user store; "
                            "environment keeps the older launcher-based env var behavior for headless runs or "
                            "external process managers."
                        ),
                    },
                ),
                "credential_profile": (
                    "STRING",
                    {
                        "default": "default",
                        "tooltip": (
                            "Name of the local credentials profile to use when credential_source is local_profile. "
                            "For Kimi, GLM, and Grok, the profile file should contain provider entries such as "
                            "profiles.<name>.kimi.api_key, profiles.<name>.glm.api_key, and profiles.<name>.grok.api_key."
                        ),
                    },
                ),
                "base_url": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Provider API base URL saved in the workflow. Leave blank to use provider defaults or "
                            "profile values. Common values: Kimi https://api.moonshot.ai/v1, GLM https://api.z.ai/api/paas/v4, "
                            "Grok https://api.x.ai/v1."
                        ),
                    },
                ),
                "model": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Vision-capable provider model saved in the workflow. Leave blank to use provider defaults "
                            "or profile values. Common defaults: Kimi kimi-k2.6, GLM glm-5v-turbo, Grok grok-4.3."
                        ),
                    },
                ),
                "debug": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Reserved flag for future verbose provider diagnostics. Leave it off for normal runs; "
                            "when debugging, pair it with visible analysis_json and warnings outputs so provider "
                            "setup issues are not confused with Qwen prompt-policy behavior."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("PROMPT_DIRECTOR_CONFIG",)
    RETURN_NAMES = ("config",)
    OUTPUT_TOOLTIPS = (
        "Provider configuration dictionary for Prompt Director. It contains provider, credential source, profile, base URL, model, and debug preference, never API keys.",
    )
    FUNCTION = "run"
    CATEGORY = "Prompt Director"

    def run(
        self,
        analysis_provider="none",
        credential_source="local_profile",
        credential_profile="default",
        base_url="",
        model="",
        debug=False,
    ):
        return (
            {
                "analysis_provider": analysis_provider,
                "credential_source": credential_source,
                "credential_profile": credential_profile,
                "base_url": base_url,
                "model": model,
                "debug": bool(debug),
            },
        )


class PromptDirectorImageDescribe:
    DESCRIPTION = """
Describe an input image through a configured vision provider and return prompt-ready visual text.

What this node does:
It sends the connected IMAGE to the provider selected by Prompt Director Config, then asks for a compact
description suitable for downstream prompt production. This is the node for the workflow step "what does
Grok/Kimi/GLM see in this image?" The description should capture subject, hair, clothing, colors, expression,
pose, environment, lighting, camera/framing, visible text, and edit risks.

Recommended wiring:
source IMAGE -> Prompt Image Describe.image
Prompt Director Config.config -> Prompt Image Describe.config
Prompt Image Describe.description -> Prompt Enhancer.image_description or Prompt Producer.image_description
Prompt Image Describe.analysis_json -> text viewer when debugging

Provider requirements:
This node requires a non-none provider config and local-profile or environment credentials for that provider.
It does not store API keys in workflow JSON. If the provider is missing or unavailable, the node returns an
empty description and a warnings JSON array explaining the setup problem.

Typical use:
Describe the source image first, then feed the description plus a short user edit prompt into Prompt Enhancer.
This gives the model enough visual context to produce edits like "change the patterned bikini to a solid matte
black bikini while preserving the same fit, ties, pose, lighting, rocks, and waterfall".

What to verify:
Open the returned description in a text viewer before trusting it. It should name the main subject, count obvious
subjects correctly, describe edit-sensitive clothing and accessories, and avoid inventing details that are not
visible. If the description is too vague, switch detail_level to detailed or add a focus_prompt. If it misses a
critical object, feed that correction into Prompt Enhancer as global rules or manual text.
""".strip()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Image to send to the configured vision provider for description. The image is encoded "
                            "as PNG data for the request; the API key still comes only from Prompt Director Config's "
                            "local profile or environment credential source."
                        ),
                    },
                ),
                "detail_level": (
                    ["concise", "detailed", "prompt_ready"],
                    {
                        "default": "prompt_ready",
                        "tooltip": (
                            "How much visual detail the provider should return. prompt_ready asks for a compact but "
                            "usable prompt paragraph; detailed asks for more inspection notes; concise keeps output short."
                        ),
                    },
                ),
            },
            "optional": {
                "focus_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": (
                            "Optional instruction for what to emphasize in the description, such as wardrobe, pose, "
                            "facial expression, background, product details, visible text, or edit-sensitive regions."
                        ),
                    },
                ),
                "config": (
                    "PROMPT_DIRECTOR_CONFIG",
                    {
                        "tooltip": (
                            "Prompt Director Config selecting grok, kimi, glm, zai, or another provider. Without a "
                            "non-none provider, this node returns provider_required in warnings instead of making a call."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("description", "analysis_json", "warnings")
    OUTPUT_TOOLTIPS = (
        "Prompt-ready image description returned by the provider. Connect this to Prompt Enhancer or Prompt Producer so the next rewrite step has concrete visual context.",
        "Structured visual analysis JSON returned by the provider when available. Route this to a text viewer when debugging subject count, visible text, or edit risks.",
        "JSON array of provider and setup warnings. Empty array means the description request completed without node-level warnings; provider_required means config is missing.",
    )
    FUNCTION = "run"
    CATEGORY = "Prompt Director"

    def run(self, image, detail_level="prompt_ready", focus_prompt="", config=None):
        warnings = []
        try:
            provider = _configured_provider(config)
            result = provider.describe_image(
                image_data_uri=image_to_png_data_uri(image),
                focus_prompt=focus_prompt,
                detail_level=detail_level,
            )
        except ProviderUnavailable as exc:
            return ("", "{}", json.dumps([f"provider_unavailable:{exc}"], ensure_ascii=True))
        except Exception as exc:
            return ("", "{}", json.dumps([f"provider_error:{type(exc).__name__}:{exc}"], ensure_ascii=True))

        warnings.extend(result.get("warnings", []))
        return (
            str(result.get("description", "")),
            json.dumps(result.get("analysis", {}) or {}, ensure_ascii=True),
            json.dumps(warnings, ensure_ascii=True),
        )


class PromptDirectorPromptEnhancer:
    DESCRIPTION = """
Enhance a short user prompt into a model-aware prompt using a configured inference provider.

What this node does:
It takes a basic prompt, optional image description, optional manual analysis JSON, optional LoRA metadata JSON,
and global rules, then asks the selected provider to produce a better model-facing prompt. This is the node for
"Image of a woman in a red bikini" becoming a precise edit prompt that preserves identity, pose, lighting, and
background while describing the requested change in terms the target model follows.

Model-aware enhancement:
The model_target input tells the provider whether to write for qwen_edit, krea2, ltx, or wan. For qwen_edit the
enhanced prompt should favor strict edit-preservation language. For generation models it can be more descriptive.
For video targets it can emphasize motion and temporal consistency.

Recommended wiring:
base prompt source -> Prompt Enhancer.base_prompt
Prompt Image Describe.description -> Prompt Enhancer.image_description
Prompt Director Config.config -> Prompt Enhancer.config
Prompt Enhancer.enhanced_prompt -> text viewer -> model text encoder

Provider requirements:
This node requires a configured provider to do AI rewriting. Without one, it returns the original prompt and a
provider_required warning so the graph remains runnable and visibly incomplete.

How to judge output:
The enhanced prompt should preserve the user's intent while adding concrete model-facing detail. For image edit
workflows it should say what changes and what must remain unchanged. For Qwen Image Edit, useful language often
includes "preserve pose", "preserve lighting and shadows", "same fit and style", "do not add extra people", and
"do not add readable text". If the enhanced prompt changes the user's intent, edit the base prompt or add a
global rule before sending it to the text encoder.

Relationship to Prompt Director:
Prompt Enhancer is provider-backed and creative; Prompt Director is deterministic policy injection. A practical
workflow is Prompt Image Describe -> Prompt Enhancer -> Prompt Director -> text viewer -> text encoder. That lets
the provider write a richer prompt, then lets Prompt Director apply repeatable model-specific guardrails.
""".strip()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": (
                            "The user's short prompt or edit intent before AI enhancement. Keep this close to the "
                            "actual intent; use image_description and global_rules for context and durable constraints."
                        ),
                    },
                ),
                "model_target": (
                    ["qwen_edit", "krea2", "ltx", "wan"],
                    {
                        "default": "qwen_edit",
                        "tooltip": (
                            "Target model family for the enhanced prompt. qwen_edit asks for edit-preservation wording; "
                            "krea2 asks for composition/realism wording; ltx and wan ask for video continuity wording."
                        ),
                    },
                ),
            },
            "optional": {
                "image_description": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": (
                            "Prompt-ready description from Prompt Image Describe. This lets the provider preserve "
                            "visible subject, pose, clothing, lighting, and background while rewriting the user prompt."
                        ),
                    },
                ),
                "lora_metadata_json": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": (
                            "Optional Prompt Director LoRA metadata JSON. Use Prompt Producer when you want the node "
                            "to read safetensors metadata automatically from a selected LoRA file."
                        ),
                    },
                ),
                "global_rules": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": (
                            "Reusable hard rules for the enhancement request, such as no extra people, preserve identity, "
                            "do not add text, keep camera angle, or treat references as appearance guidance only."
                        ),
                    },
                ),
                "config": (
                    "PROMPT_DIRECTOR_CONFIG",
                    {
                        "tooltip": (
                            "Prompt Director Config selecting the provider used to rewrite the prompt. No API key is "
                            "stored in this node; credentials come from local profile or environment configuration."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("enhanced_prompt", "rationale", "warnings")
    OUTPUT_TOOLTIPS = (
        "Provider-rewritten model-facing prompt. Route this through a text viewer before the text encoder so you can confirm the edit intent and preservation language.",
        "Provider rationale explaining what changed, especially preservation language, model-specific phrasing, image-context use, or LoRA trigger-token insertion.",
        "JSON array of setup or provider warnings. provider_required means no configured inference provider was connected, so the original prompt was returned.",
    )
    FUNCTION = "run"
    CATEGORY = "Prompt Director"

    def run(
        self,
        base_prompt,
        model_target,
        image_description="",
        lora_metadata_json="",
        global_rules="",
        config=None,
    ):
        lora_metadata, warnings = parse_lora_metadata_json(lora_metadata_json)
        try:
            provider = _configured_provider(config)
            result = provider.enhance_prompt(
                base_prompt=base_prompt,
                model_target=model_target,
                image_description=image_description,
                lora_metadata=lora_metadata,
                global_rules=global_rules,
            )
            warnings.extend(result.get("warnings", []))
            return (
                str(result.get("enhanced_prompt", base_prompt or "")),
                str(result.get("rationale", "")),
                json.dumps(warnings, ensure_ascii=True),
            )
        except ProviderUnavailable as exc:
            warnings.append(f"provider_unavailable:{exc}")
            return (base_prompt or "", "", json.dumps(warnings, ensure_ascii=True))
        except Exception as exc:
            warnings.append(f"provider_error:{type(exc).__name__}:{exc}")
            return (base_prompt or "", "", json.dumps(warnings, ensure_ascii=True))


class PromptProducer:
    DESCRIPTION = """
Produce a model-ready prompt from a user prompt plus LoRA metadata.

LoRA metadata:
Prompt Producer can read Prompt Director metadata embedded in a selected safetensors LoRA, or accept the same
metadata as JSON from another tool. The important fields are trigger_tokens, activation_phrases, model_family,
compatible_base_models, prompt_guidance, negative_tokens, and default strength hints. The common use case is a
LoRA such as clown_nose_lora.safetensors whose metadata says the required trigger token is clownnose.

Trigger tokens:
When trigger_tokens are present, Prompt Producer passes them to the inference provider and expects the enhanced
prompt to include the exact required token naturally. If no provider is configured, it falls back to a visible
deterministic insertion of missing trigger tokens so you can still see what the LoRA likely needs.

Recommended wiring:
user text -> Prompt Producer.base_prompt
LoRA file selector -> Prompt Producer.lora_name
Prompt Image Describe.description -> Prompt Producer.image_description
Prompt Director Config.config -> Prompt Producer.config
Prompt Producer.enhanced_prompt -> text viewer -> model text encoder

This node does not load or apply the LoRA weights. Keep using your normal LoRA loader for model/CLIP patching.
Prompt Producer only reads metadata and produces better text for the encoder.

Expected metadata shape:
The best LoRA metadata tells this node what token must appear, what model family the LoRA was trained for, what
strength range is sane, and what plain-language intent the LoRA represents. For example, a clown nose LoRA should
include trigger_tokens ["clownnose"], prompt_guidance explaining that the token activates the comic nose effect,
and model_family showing whether it was trained for qwen_image_edit, flux, sdxl, or another base family.

What to verify:
Always inspect lora_keywords and lora_metadata_json outputs. If trigger tokens are empty, the selected LoRA either
has no Prompt Director metadata or the metadata was not readable. If the enhanced prompt includes the wrong token,
fix the LoRA metadata in the manager rather than repeatedly hand-editing prompts. If the model output ignores the
LoRA even with the token present, check the actual LoRA loader strength and model compatibility.

Provider vs fallback:
With a provider configured, Prompt Producer asks the provider to rewrite the prompt using the metadata. Without a
provider, it only inserts missing trigger tokens deterministically. That fallback is intentionally simple; it is a
visibility aid, not a replacement for model-aware prompt rewriting.
""".strip()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": (
                            "The user's natural language prompt before LoRA-aware enhancement. Example: give her a "
                            "funny nose. The provider can rewrite this using trigger tokens found in LoRA metadata."
                        ),
                    },
                ),
                "model_target": (
                    ["qwen_edit", "krea2", "ltx", "wan"],
                    {
                        "default": "qwen_edit",
                        "tooltip": (
                            "Target model family for prompt production. This guides the provider toward edit-preserving, "
                            "generative, or video-oriented language while incorporating LoRA trigger metadata."
                        ),
                    },
                ),
                "lora_name": (
                    get_lora_names(),
                    {
                        "tooltip": (
                            "Selected LoRA safetensors file to inspect for Prompt Director metadata. Choose none when "
                            "you are providing lora_metadata_json directly from another LoRA manager."
                        ),
                    },
                ),
            },
            "optional": {
                "lora_metadata_json": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": (
                            "Optional metadata JSON from another app or LoRA manager. When provided, this takes precedence "
                            "over reading the selected safetensors file and should include trigger_tokens and model_family."
                        ),
                    },
                ),
                "image_description": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": (
                            "Optional prompt-ready description of the current image. Connect Prompt Image Describe here "
                            "so the provider can preserve visual details while applying LoRA trigger guidance."
                        ),
                    },
                ),
                "global_rules": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": (
                            "Optional hard constraints for the produced prompt, such as preserve identity, no extra people, "
                            "no text artifacts, or keep the original pose and lighting."
                        ),
                    },
                ),
                "config": (
                    "PROMPT_DIRECTOR_CONFIG",
                    {
                        "tooltip": (
                            "Prompt Director Config selecting the inference provider. Without a configured provider, "
                            "Prompt Producer falls back to deterministic trigger-token insertion and adds a warning."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("enhanced_prompt", "lora_keywords", "lora_metadata_json", "warnings")
    OUTPUT_TOOLTIPS = (
        "LoRA-aware prompt for the target model. Route this through a text viewer before the encoder to verify trigger tokens and preservation wording.",
        "JSON array of trigger tokens or LoRA keywords used during prompt production, useful for confirming metadata-driven activation terms such as clownnose.",
        "Normalized LoRA metadata JSON used by the node, from either safetensors metadata or the JSON input, for audit, debug viewers, and downstream tooling.",
        "JSON array of warnings about metadata parsing, provider setup, safetensors lookup, or deterministic fallback behavior when no inference provider is configured.",
    )
    FUNCTION = "run"
    CATEGORY = "Prompt Director"

    def run(
        self,
        base_prompt,
        model_target,
        lora_name,
        lora_metadata_json="",
        image_description="",
        global_rules="",
        config=None,
    ):
        lora_metadata, warnings = load_lora_prompt_metadata(lora_name, lora_metadata_json)
        trigger_tokens = [str(token) for token in lora_metadata.get("trigger_tokens", [])]

        try:
            provider = _configured_provider(config)
            result = provider.enhance_prompt(
                base_prompt=base_prompt,
                model_target=model_target,
                image_description=image_description,
                lora_metadata=lora_metadata,
                global_rules=global_rules,
            )
            warnings.extend(result.get("warnings", []))
            keywords = result.get("lora_keywords", trigger_tokens) or trigger_tokens
            return (
                str(result.get("enhanced_prompt", base_prompt or "")),
                json.dumps(keywords, ensure_ascii=True),
                json.dumps(lora_metadata, ensure_ascii=True),
                json.dumps(warnings, ensure_ascii=True),
            )
        except ProviderUnavailable as exc:
            warnings.append(f"provider_unavailable:{exc}")
            enhanced = _fallback_lora_prompt(base_prompt or "", trigger_tokens)
            return (
                enhanced,
                json.dumps(trigger_tokens, ensure_ascii=True),
                json.dumps(lora_metadata, ensure_ascii=True),
                json.dumps(warnings, ensure_ascii=True),
            )
        except Exception as exc:
            warnings.append(f"provider_error:{type(exc).__name__}:{exc}")
            enhanced = _fallback_lora_prompt(base_prompt or "", trigger_tokens)
            return (
                enhanced,
                json.dumps(trigger_tokens, ensure_ascii=True),
                json.dumps(lora_metadata, ensure_ascii=True),
                json.dumps(warnings, ensure_ascii=True),
            )


class PromptDirectorDebugPreview:
    DESCRIPTION = """
Formats Prompt Director outputs into a readable debug string.

Purpose:
This node turns final_prompt, analysis_json, and warnings into one readable STRING. It does not change the
prompt, call a provider, or feed the model by itself. It exists so the workflow has a compact audit surface
that can be connected to a Show Text node or any other text viewer.

What this preview proves:
The preview proves what Prompt Director produced at that point in the graph. It is useful when checking
whether Qwen is receiving the policy-injected prompt, whether manual or provider analysis was merged, and
whether warnings explain a likely edit failure. It does not prove the downstream text encoder is connected
correctly unless you also route final_prompt to that encoder.

Recommended wiring:
Prompt Director.final_prompt -> Prompt Director Debug Preview.final_prompt
Prompt Director.analysis_json -> Prompt Director Debug Preview.analysis_json
Prompt Director.warnings -> Prompt Director Debug Preview.warnings
Prompt Director Debug Preview.debug_text -> Show Text / text viewer
Prompt Director.final_prompt -> TextEncodeQwenImageEditPlus prompt

Keep the preview near the Qwen text encoder in the workflow. When a render produces extra people, accidental
text, unwanted objects, or weak prompt adherence, the preview should be the first place to inspect before
changing LoRAs, samplers, or model weights.

How to read warnings:
extra_people means the policy detected subject-count risk and added stronger no-duplicate-subject language.
text_artifacts means visible text, logos, captions, signs, or watermarks may need explicit suppression.
object_in_hands means the source or reference context may cause Qwen to place unintended objects in hands.
invalid_analysis_json means the manual analysis_json input was not parseable as a JSON object. Provider
warnings such as provider_requires_image and provider_unavailable point to config or environment setup.

Debug workflow:
1. Confirm Final Prompt includes the base edit intent and the selected model policy.
2. Confirm Analysis JSON reflects the current image or manual notes, not stale copied text.
3. Confirm Warnings are explainable. Empty warnings are fine; unexpected warnings deserve inspection.
4. If the text viewer changes but the render does not, inspect the downstream Qwen text encoder wiring.
5. If the preview never changes, inspect the upstream base_prompt and graph execution path.
""".strip()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "final_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "tooltip": (
                            "The final prompt string produced by Prompt Director. Feed the same value to this "
                            "preview and to the downstream text encoder so the debug text reflects the exact "
                            "model-facing prompt used by the graph."
                        ),
                    },
                ),
                "analysis_json": (
                    "STRING",
                    {
                        "multiline": True,
                        "tooltip": (
                            "The normalized image/context analysis JSON produced by Prompt Director after merging "
                            "manual and provider analysis. Inspect this when prompt policy seems stale or when "
                            "subject count, visible text, or object risks need confirmation."
                        ),
                    },
                ),
                "warnings": (
                    "STRING",
                    {
                        "multiline": True,
                        "tooltip": (
                            "JSON array string of warnings produced by Prompt Director. Route it here to see "
                            "invalid JSON, provider setup problems, extra-subject risks, text artifact risks, "
                            "and object-in-hands risks beside the final prompt."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("debug_text",)
    OUTPUT_TOOLTIPS = (
        "Readable prompt, analysis, and warning preview for text viewer nodes. This is an audit string only; keep final_prompt separately connected to the actual Qwen text encoder.",
    )
    FUNCTION = "run"
    CATEGORY = "Prompt Director"

    def run(self, final_prompt: str, analysis_json: str, warnings: str):
        debug_text = (
            "## Final Prompt\n"
            f"{final_prompt or ''}\n\n"
            "## Analysis JSON\n"
            f"{analysis_json or '{}'}\n\n"
            "## Warnings\n"
            f"{warnings or '[]'}"
        )
        return (debug_text,)


def _configured_provider(config):
    provider_name = (config or {}).get("analysis_provider", "none")
    if not config or provider_name in {"none", "", None}:
        raise ProviderUnavailable("provider_required")
    return get_provider(provider_name, config=config)


def _fallback_lora_prompt(base_prompt: str, trigger_tokens: list[str]) -> str:
    prompt = base_prompt.strip()
    missing = [token for token in trigger_tokens if token and token not in prompt]
    if not missing:
        return prompt
    prefix = ", ".join(missing)
    return f"{prefix}, {prompt}" if prompt else prefix


NODE_CLASS_MAPPINGS = {
    "PromptDirector": PromptDirector,
    "PromptDirectorConfig": PromptDirectorConfig,
    "PromptDirectorImageDescribe": PromptDirectorImageDescribe,
    "PromptDirectorPromptEnhancer": PromptDirectorPromptEnhancer,
    "PromptProducer": PromptProducer,
    "PromptDirectorDebugPreview": PromptDirectorDebugPreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptDirector": "Prompt Director",
    "PromptDirectorConfig": "Prompt Director Config",
    "PromptDirectorImageDescribe": "Prompt Image Describe",
    "PromptDirectorPromptEnhancer": "Prompt Enhancer",
    "PromptProducer": "Prompt Producer",
    "PromptDirectorDebugPreview": "Prompt Director Debug Preview",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "PromptDirector",
    "PromptDirectorConfig",
    "PromptDirectorImageDescribe",
    "PromptDirectorPromptEnhancer",
    "PromptProducer",
    "PromptDirectorDebugPreview",
]
