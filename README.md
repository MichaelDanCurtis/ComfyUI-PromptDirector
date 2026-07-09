# ComfyUI Prompt Director

Graph-native prompt policy nodes for ComfyUI.

The first target is Qwen Image Edit 2511, but the pack also includes prompt policy for Krea2 and LTX/WAN video prompts.

## Nodes

- `Prompt Director`: builds `final_prompt`, normalized `analysis_json`, and `warnings`.
- `Prompt Director Config`: enables optional provider-backed image analysis without exposing API-key widgets.
- `Prompt Image Describe`: sends an image to the configured provider and returns prompt-ready visual description text.
- `Prompt Enhancer`: rewrites a basic user prompt into a stronger model-aware prompt using provider inference.
- `Prompt Producer`: reads LoRA prompt metadata and produces a LoRA-aware prompt with the right trigger keywords.
- `Prompt Director Debug Preview`: formats Prompt Director outputs into a readable text block for viewer nodes.

## Recommended Qwen Wiring

```text
base_prompt -> Prompt Director
Prompt Director.final_prompt -> text viewer
text viewer -> TextEncodeQwenImageEditPlus
TextEncodeQwenImageEditPlus -> sampler positive conditioning
```

Keep the text viewer in the graph. It is the audit point proving Qwen receives this node's prompt output instead of a stale widget.

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

Prompt Producer reads Prompt Director metadata from safetensors headers when available. The metadata spec is in:

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

OAuth-style providers are intentionally deferred. The local profile path below is for API-key based providers.

### Recommended Setup: Local Profile

Use `Prompt Director Config` with these UI values:

```text
analysis_provider: kimi | glm | grok | zai | openai_compatible
credential_source: local_profile
credential_profile: default
base_url: blank unless you want to override the provider default
model: blank unless you want to override the provider default
debug: false
```

Create this local file in the active ComfyUI protected system-user credentials directory:

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
