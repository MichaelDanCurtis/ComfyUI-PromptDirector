# LoRA Prompt Metadata Spec

Version: 1.0

Status: this document remains the detailed `prompt_director` namespace reference. The canonical cross-product contract is now the Safetensors Metadata Standard maintained by the shared `safetensors_metadata` library and ComfyUI Model Explorer. `model_card` owns general identity and compatibility; `prompt_director` owns activation and prompt behavior.

Purpose: define safetensors metadata that lets Prompt Director / Prompt Producer and LoRA manager tools produce better prompts automatically.

## Storage

Safetensors metadata values are strings. Store one packed JSON field named `prompt_director` whenever possible:

```json
{
  "prompt_director": "{\"schema_version\":\"1.0\",\"trigger_tokens\":[\"clownnose\"]}"
}
```

Flat keys are also supported:

```json
{
  "prompt_director.schema_version": "1.0",
  "prompt_director.trigger_tokens": "[\"clownnose\"]",
  "prompt_director.model_family": "[\"qwen_image_edit\"]"
}
```

Use packed JSON for new files. Use flat keys only when a writer cannot safely store a structured JSON object as a string.

## Minimal Metadata

Every prompt-aware LoRA should include:

```json
{
  "schema_version": "1.0",
  "display_name": "Clown Nose LoRA",
  "trigger_tokens": ["clownnose"],
  "semantic_intent": "Adds a comic clown nose to the subject.",
  "model_family": ["qwen_image_edit"],
  "prompt_guidance": "Use clownnose when requesting the nose effect. Keep the rest of the face unchanged for edit workflows."
}
```

## Recommended Fields

`schema_version`: string. Current value is `1.0`.

`display_name`: human-readable LoRA name.

`trigger_tokens`: array of exact tokens that must appear in the prompt.

`activation_phrases`: optional array of phrases that help the LoRA activate naturally.

`negative_tokens`: optional array of tokens or concepts to avoid.

`semantic_intent`: short plain-language description of what the LoRA does.

`model_family`: array of compatible model families, such as `qwen_image_edit`, `qwen_image`, `flux`, `sdxl`, `sd15`, `wan`, `ltx`, `krea2`.

`compatible_base_models`: array of specific base model names or checkpoints tested with the LoRA.

`prompt_guidance`: plain-language instruction for the prompt producer.

`negative_prompt_guidance`: optional instruction for negative prompts or avoid clauses.

`default_strength_model`: recommended model strength as number.

`default_strength_clip`: recommended CLIP/text strength as number, when applicable.

`strength_min` / `strength_max`: safe tested strength range.

`keyword_aliases`: optional array of user phrases that should map to the trigger. Example: `["funny nose", "clown nose", "red nose"]`.

`example_prompts`: array of tested prompt snippets.

`preservation_guidance`: for edit LoRAs, what should remain unchanged.

`content_notes`: optional free text for style, safety, subject, or dataset notes.

`authoring_tool`: name/version of the LoRA manager or trainer that wrote the metadata.

## Example: Clown Nose LoRA

```json
{
  "schema_version": "1.0",
  "display_name": "Clown Nose LoRA",
  "trigger_tokens": ["clownnose"],
  "activation_phrases": ["clownnose nose", "comic red clownnose"],
  "negative_tokens": ["extra nose", "deformed face"],
  "semantic_intent": "Adds a comic clown nose while preserving the subject identity.",
  "model_family": ["qwen_image_edit"],
  "compatible_base_models": ["Qwen Image Edit 2511"],
  "prompt_guidance": "Include clownnose exactly once. For edits, ask to preserve face shape, expression, lighting, and background.",
  "negative_prompt_guidance": "Avoid extra noses, face deformation, duplicate facial features, and costume changes unless requested.",
  "default_strength_model": 0.85,
  "default_strength_clip": 0.85,
  "strength_min": 0.55,
  "strength_max": 1.0,
  "keyword_aliases": ["funny nose", "clown nose", "red nose"],
  "example_prompts": [
    "clownnose, add a comic red nose while preserving identity and expression",
    "clownnose nose, same face, same lighting, no costume change"
  ],
  "preservation_guidance": "Do not change eyes, mouth, skin tone, hair, pose, camera angle, or background.",
  "authoring_tool": "LoraSuite"
}
```

## Prompt Producer Behavior

When Prompt Producer sees this metadata and the user enters:

```text
give her a funny nose
```

It should provide the inference model with:

- user intent: `give her a funny nose`
- required trigger token: `clownnose`
- aliases: `funny nose`, `clown nose`, `red nose`
- model compatibility: `qwen_image_edit`
- prompt guidance and preservation guidance

The expected enhanced prompt should include the exact trigger token:

```text
clownnose, add a comic red clown nose to the subject while preserving identity, facial expression, skin tone, hair, pose, lighting, camera angle, and background.
```

## Compatibility Notes

Do not rely only on Civitai tags or training captions. Those are useful context, but prompt production needs explicit activation metadata.

Do not store API keys, private dataset paths, or private user notes in safetensors metadata.

Keep trigger tokens exact. If the LoRA was trained on `clownnose`, do not write only `clown nose`.

When a LoRA supports multiple model families, include tested strength hints per family in a future `model_family_overrides` object. For version 1.0, prefer separate LoRA metadata profiles or conservative global strength values.
