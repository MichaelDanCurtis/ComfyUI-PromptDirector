# ComfyUI Prompt Director

Graph-native, model-aware prompt planning and edit evaluation for ComfyUI.

The first target is Qwen Image Edit 2511, but the pack also includes prompt policy for Krea2 and LTX/WAN video prompts.

## Nodes

- `Prompt Director Config`: selects optional provider-backed image analysis without exposing API-key widgets.
- `Prompt Director Context`: reads canonical Model Explorer `model_card` and `prompt_director` metadata for the active model and complete LoRA stack.
- `Prompt Director Reference`: marks reference images as identity, wardrobe, pose, style, object, or background guidance.
- `Prompt Image Describe`: sends an image to the configured provider and returns prompt-ready visual description text.
- `Prompt Enhancer`: rewrites a basic user prompt into a stronger model-aware prompt using provider inference.
- `Prompt Producer`: reads LoRA prompt metadata and produces text with the required trigger keywords.
- `Prompt Director`: deterministically compiles source analysis, model details, references, mask scope, and policy into `final_prompt` and `edit_plan_json`.
- `Prompt Director Auto`: runs source/reference description, structured AI planning, validation, and deterministic prompt compilation in one node.
- `Prompt Director Result Critic`: compares the source and rendered result, scores edit adherence and preservation, and returns a revision when needed.
- `Prompt Director Debug Preview`: formats prompt, analysis, and warning outputs into one readable text block.

## Recommended Qwen Wiring

```text
base_prompt -> Prompt Director
Prompt Director.final_prompt -> text viewer
text viewer -> TextEncodeQwenImageEditPlus
TextEncodeQwenImageEditPlus -> sampler positive conditioning
```

Keep the text viewer in the graph. It is the audit point proving Qwen receives this node's prompt output instead of a stale widget.

## Recommended Automatic Wiring

```text
Prompt Director Context.context -> Prompt Director Auto.context
Prompt Director Reference.references -> Prompt Director Auto.reference_set
source image + optional mask -> Prompt Director Auto
Prompt Director Config.config -> Prompt Director Auto.config
Prompt Director Auto.final_prompt -> text viewer -> TextEncodeQwenImageEditPlus
source image + rendered result -> Prompt Director Result Critic
```

`Prompt Director Auto` asks the provider for a structured edit plan, not an unchecked final prompt. The deterministic compiler remains responsible for the exact model-facing text.

## Model Explorer Metadata

Prompt Director implements the canonical Safetensors Metadata Standard used by ComfyUI Model Explorer:

- `model_card` supplies asset identity, model family, compatible families/base models, architecture, precision, rank, and optional family-specific weight overrides.
- `prompt_director` supplies exact triggers, activation and negative tokens, semantic intent, prompt/preservation guidance, strength defaults and tested ranges, aliases, and examples.
- `modelspec.*` and local header classification are read-only fallbacks when the canonical card is incomplete.

`Prompt Director Context` reads the tensor directly, checks every LoRA against the selected model family and actual strengths, and passes the complete metadata object to the prompt-building nodes. Unknown additive fields are preserved.

## Provider Prompt Workflow

For image-aware prompt production:

```text
source image -> Prompt Image Describe
basic user prompt + Prompt Image Describe.description -> Prompt Enhancer
Prompt Enhancer.enhanced_prompt -> Prompt Director
Prompt Director.final_prompt -> text viewer -> TextEncodeQwenImageEditPlus
```

For LoRA-aware prompt production:

```text
basic user prompt + lora_name -> Prompt Producer
Prompt Producer.enhanced_prompt -> Prompt Director or text viewer
normal LoRA loader still applies the LoRA weights
```

`Prompt Producer` reads Prompt Director metadata from safetensors headers when available. The metadata spec is in:

```text
LORA_PROMPT_METADATA_SPEC.md
```

## Provider Analysis

Provider-backed image analysis is optional. By default, Prompt Director is deterministic and does not make network calls.

Supported provider selectors:

- `openai_compatible`
- `zai`
- `glm`
- `kimi`
- `grok`

The shared profile interface is for API-key providers. Grok CLI OAuth can be used as an existing fallback; adding managed OAuth setup for other providers remains deferred.

### Recommended Setup: Credential Sidebar

Open the `AI Credentials` sidebar tab in ComfyUI. Select or enter a profile and provider, then save the API key, optional base URL/model, authentication style, and static headers. Secret values are write-only: the API and sidebar report only whether a key is configured.

The sidebar writes the product-neutral canonical store:

```text
ComfyUI/user/__credentials/provider_profiles.json
```

Other products should read and write this same store instead of creating another credential file.

### Prompt Director Config

Use `Prompt Director Config` with these UI values:

```text
analysis_provider: kimi | glm | grok | zai | openai_compatible
credential_source: local_profile
credential_profile: default
base_url: blank unless you want to override the provider default
model: blank unless you want to override the provider default
debug: false
```

For manual administration, the same file is in the active ComfyUI protected system-user credentials directory:

```text
ComfyUI/user/__credentials/provider_profiles.json
```

On disk, that resolves to:

```text
<ComfyUI root>/user/__credentials/provider_profiles.json
```

This uses ComfyUI's `folder_paths.get_system_user_directory("credentials")` convention. Those `user/__*`
directories are internal-only and are blocked from the normal `/userdata` HTTP endpoints, which makes this
the right local place for shared provider secrets across Prompt Director and related node packs. The built-in
Comfy Org hidden inputs (`AUTH_TOKEN_COMFY_ORG`, `API_KEY_COMFY_ORG`) are intentionally not reused for Kimi,
GLM, Grok, or other third-party provider keys.

File format:

```json
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
      },
      "openai_compatible": {
        "api_key": "YOUR_OPENAI_COMPATIBLE_KEY"
      }
    }
  }
}
```

The workflow stores only `credential_profile`, `base_url`, and `model`. It does not store the API key.

You may also put `base_url` and `model` in the profile entry, but the visible UI fields take precedence:

```json
{
  "profiles": {
    "default": {
      "kimi": {
        "api_key": "YOUR_KIMI_OR_MOONSHOT_KEY",
        "base_url": "https://api.moonshot.ai/v1",
        "model": "kimi-k2.6"
      },
      "glm": {
        "api_key": "YOUR_GLM_OR_ZAI_KEY",
        "base_url": "https://api.z.ai/api/paas/v4",
        "model": "glm-5v-turbo"
      },
      "grok": {
        "api_key": "YOUR_XAI_KEY",
        "base_url": "https://api.x.ai/v1",
        "model": "grok-4.3"
      }
    }
  }
}
```

Backward compatibility: if the shared credentials file does not exist, Prompt Director also checks the older
`ComfyUI/user/prompt_director/credentials.json` and `ComfyUI/user/default/prompt_director/credentials.json`
locations. New products should write only to the shared `user/__credentials/provider_profiles.json` file.
See `COMFYUI_SHARED_CREDENTIAL_STORE.md` for the shared-store contract intended for related products.

Provider defaults when `base_url` and `model` are blank:

| Provider | Base URL | Model |
| --- | --- | --- |
| `kimi` | `https://api.moonshot.ai/v1` | `kimi-k2.6` |
| `glm` / `zai` | `https://api.z.ai/api/paas/v4` | `glm-5v-turbo` |
| `grok` | `https://api.x.ai/v1` | `grok-4.3` |
| `openai_compatible` | `https://api.openai.com/v1` | `gpt-4o-mini` |

### Environment Variable Fallback

Environment variables are still supported for launchers and headless runs:

- `OPENAI_API_KEY`
- `GLM_API_KEY`, `Z_AI_API_KEY`, `ZAI_API_KEY`, or `BIGMODEL_API_KEY`
- `KIMI_API_KEY` or `MOONSHOT_API_KEY`
- `XAI_API_KEY` or `GROK_API_KEY`

Additional environment variables:

- `OPENAI_API_BASE` or `OPENAI_BASE_URL`
- `PROMPT_DIRECTOR_OPENAI_MODEL`
- `PROMPT_DIRECTOR_GLM_BASE_URL`
- `PROMPT_DIRECTOR_GLM_MODEL`
- `PROMPT_DIRECTOR_ZAI_BASE_URL`
- `PROMPT_DIRECTOR_ZAI_MODEL`
- `PROMPT_DIRECTOR_KIMI_BASE_URL`
- `PROMPT_DIRECTOR_KIMI_MODEL`
- `PROMPT_DIRECTOR_GROK_BASE_URL`
- `PROMPT_DIRECTOR_GROK_MODEL`

If a provider is unavailable or misconfigured, the graph still returns deterministic prompt policy and reports a warning.

## Comfy MCP Inspection

Prompt Director records a bounded, sanitized runtime snapshot for Context, Prompt Director, Auto, and Result Critic nodes. It includes the resolved model/LoRA context, source analysis, structured edit plan, exact final prompt, warnings, and critic result. Credentials and image tensors are never included.

- ComfyUI route: `GET /prompt_director/inspection[?node_id=<id>]`
- Comfy MCP tool: `prompt_director_inspect`
- Live panel tool: `panel_audit_prompt_director`

The live audit correlates runtime state with graph wiring, model loaders, and actual LoRA loader strengths. It returns observations and proposed changes, but is read-only. Comfy AI asks before applying a recommendation unless the user already explicitly requested the fix. See `PROMPT_DIRECTOR_AGENT_INSPECTION.md`.

## Versioned Integration Contracts

Prompt Director 0.3 publishes language-neutral JSON Schemas for model context, edit plans, structured diagnostics, critic results, runtime inspection, and module discovery. Existing node signatures and the legacy inspection route remain unchanged.

- Manifest: `GET /prompt_director/v1/manifest`
- Contract schema: `GET /prompt_director/v1/contracts/<contract-name>`
- Inspections: `GET /prompt_director/v1/inspections[?node_id=<id>]`
- One inspection: `GET /prompt_director/v1/inspections/<id>`

Model context now explains how the target was resolved, records compatibility evidence, consumes family-specific strength guidance, and includes structured diagnostics alongside legacy warning strings. Explicit model selection remains authoritative and no recommendation changes model or LoRA loader widgets automatically.

See `PROMPT_DIRECTOR_CONTRACTS.md` for compatibility rules, payload examples, and integration guidance.
