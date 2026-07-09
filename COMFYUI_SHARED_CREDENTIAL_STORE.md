# ComfyUI Shared Credential Store

Prompt Director uses a product-neutral local credential file so related ComfyUI tools can share provider keys without putting secrets in workflow JSON.

## ComfyUI Credential Findings

ComfyUI has hidden prompt inputs named `AUTH_TOKEN_COMFY_ORG` and `API_KEY_COMFY_ORG`. They are specific to Comfy Org API nodes. The backend treats only those keys as sensitive prompt `extra_data`, and Comfy API node helpers forward them as Comfy Org auth headers.

Do not store third-party provider keys in those hidden fields. Kimi, GLM, Grok, OpenAI-compatible, and future provider keys should use the shared local store below.

## Canonical Path

Use ComfyUI's protected system-user directory:

```text
folder_paths.get_system_user_directory("credentials") / "provider_profiles.json"
```

On disk, that resolves to:

```text
<ComfyUI root>/user/__credentials/provider_profiles.json
```

ComfyUI blocks `user/__*` system-user directories from the normal `/userdata` HTTP endpoints. This makes `user/__credentials` a better local target than `user/default` for secrets that should be server-side only.

## File Format

```json
{
  "profiles": {
    "default": {
      "kimi": {
        "api_key": "YOUR_KIMI_OR_MOONSHOT_KEY",
        "base_url": "https://api.moonshot.ai/v1",
        "model": "kimi-k2.6",
        "auth_style": "bearer",
        "headers": {}
      },
      "glm": {
        "api_key": "YOUR_GLM_OR_ZAI_KEY"
      },
      "grok": {
        "api_key": "YOUR_XAI_KEY"
      },
      "anthropic": {
        "api_key": "YOUR_ANTHROPIC_API_KEY",
        "auth_style": "x-api-key",
        "headers": { "anthropic-version": "2023-06-01" }
      }
    }
  },
  "mcp": {
    "runcomfy": {
      "transport": "http",
      "url": "https://api.runcomfy.com/mcp",
      "requires": ["runcomfy"],
      "headers": {}
    },
    "comfyui": {
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "comfyui-mcp"],
      "requires": ["civitai", "huggingface"]
    }
  }
}
```

Provider keys are normalized to lowercase with hyphens converted to underscores. `api_key` is required for API-key providers. `base_url` and `model` are optional profile defaults; visible node UI values may override them.

## Provider Auth Style & Headers

Not every provider takes a plain `Authorization: Bearer <api_key>`. Two optional fields per provider entry tell a consumer exactly how to call it, so header knowledge never has to be hard-coded in each tool:

- **`auth_style`** — how the `api_key` is attached. One of:
  - `bearer` (default) → `Authorization: Bearer <api_key>`
  - `x-api-key` → `x-api-key: <api_key>` (Anthropic API)
  - `header:<Name>` → `<Name>: <api_key>` (arbitrary header)
  - `query:<param>` → appended as `?<param>=<api_key>`
- **`headers`** — a map of *additional static headers* the endpoint requires beyond auth. Examples: Anthropic needs `anthropic-version`; Kimi/Moonshot and other OpenAI-compatible gateways may require a routing/version header.

Known cases worth documenting (a consumer reads these from the file, never guesses):
- **Anthropic API**: `auth_style: x-api-key`, `headers: { "anthropic-version": "2023-06-01" }`. **Do NOT** use Claude Code / Claude subscription OAuth credentials programmatically — those are off-limits to tools; only a user-supplied Anthropic **API key** may be used here.
- **GitHub Copilot** (documented for completeness, **not in use** — cost): requires editor headers such as `Editor-Version` and `Copilot-Integration-Id`, plus a token exchange. Left out of active profiles on purpose.

## MCP Registry

The same file is where MCP servers are announced, so any tool (and any agent) can discover where an MCP lives and which credentials it needs — without hunting through configs. This half is owned by the MCP/tooling side (e.g. the comfyui-mcp fork); the provider half above is owned by Prompt Director.

`mcp.<name>` fields:
- **`transport`** — `http` or `stdio`.
- **`url`** — for `http` transport.
- **`command`** / **`args`** — for `stdio` transport (how to spawn it).
- **`requires`** — list of provider entry names (from `profiles.<profile>`) whose `api_key` this MCP needs. The launcher resolves the actual secret from the provider entry at spawn time — the MCP registry never duplicates raw keys.
- **`headers`** — optional static headers for an `http` MCP.

Rule for env-projecting consumers: when an MCP or tool turns a resolved `api_key` into a **process environment variable**, enforce a strict key allowlist so a corrupt/hand-edited file can never inject `NODE_OPTIONS`, `LD_PRELOAD`, `PATH`, etc. into a spawned subprocess.

## Resolution Order

Prompt Director resolves credentials in this order:

1. `PROMPT_DIRECTOR_CREDENTIALS_FILE`, if set.
2. `user/__credentials/provider_profiles.json`.
3. Legacy fallback: `user/prompt_director/credentials.json`.
4. Legacy fallback: `user/default/prompt_director/credentials.json`.

If the shared credentials file exists but is unreadable or invalid JSON, Prompt Director stops and reports that error instead of silently using a legacy secret.

## Rules For Other Products

- Read and write the same `provider_profiles.json` structure.
- Do not add API-key text widgets to workflow nodes.
- Store profile names in workflows, not secrets.
- Keep OAuth tokens in the same provider entry later, but use distinct fields such as `access_token`, `refresh_token`, and `expires_at`.
- Do not use Comfy Org hidden auth fields for third-party providers.
- Honor each provider's `auth_style` and `headers` from the file — never hard-code a provider's header rules inside a tool.
- Ownership split: the **`profiles`** (provider) half is owned by Prompt Director; the **`mcp`** registry half is owned by the MCP/tooling side. Read the half you don't own; only write your own.
- When projecting a resolved `api_key` into a process environment variable, enforce a strict env-key allowlist (see MCP Registry).
