from __future__ import annotations

import os
from typing import Any, Mapping

from ..credentials import CredentialUnavailable, load_provider_credentials
from ..grok_oauth import GrokOAuthUnavailable, resolve_grok_oauth_bearer
from .glm import GlmVisionProvider
from .kimi import KimiVisionProvider
from .openai_compatible import OpenAICompatibleVisionProvider
from .xai import XaiVisionProvider
from .zai import ZaiVisionProvider


class ProviderUnavailable(RuntimeError):
    pass


def get_provider(
    name: str,
    env: Mapping[str, str] | None = None,
    config: Mapping[str, Any] | None = None,
):
    env = os.environ if env is None else env
    config = config or {}
    normalized = (name or "none").strip().lower().replace("-", "_")

    if normalized in {"none", "off", "disabled"}:
        raise ProviderUnavailable("analysis provider is disabled")
    if normalized == "openai_compatible":
        resolved = _resolve_credentials(
            provider_name=normalized,
            env=env,
            config=config,
            api_key_env_names=("OPENAI_API_KEY",),
            base_url_env_names=("OPENAI_API_BASE", "OPENAI_BASE_URL"),
            model_env_names=("PROMPT_DIRECTOR_OPENAI_MODEL",),
            default_base_url="https://api.openai.com/v1",
            default_model="gpt-4o-mini",
        )
        api_key = resolved["api_key"]
        if not api_key:
            raise ProviderUnavailable("OPENAI_API_KEY is not set")
        return OpenAICompatibleVisionProvider(
            api_key=api_key,
            base_url=resolved["base_url"],
            model=resolved["model"],
        )
    if normalized in {"zai", "glm"}:
        resolved = _resolve_credentials(
            provider_name=normalized,
            env=env,
            config=config,
            api_key_env_names=("GLM_API_KEY", "Z_AI_API_KEY", "ZAI_API_KEY", "BIGMODEL_API_KEY"),
            base_url_env_names=(
                "PROMPT_DIRECTOR_GLM_BASE_URL",
                "GLM_BASE_URL",
                "PROMPT_DIRECTOR_ZAI_BASE_URL",
                "Z_AI_BASE_URL",
            ),
            model_env_names=("PROMPT_DIRECTOR_GLM_MODEL", "PROMPT_DIRECTOR_ZAI_MODEL"),
            default_base_url="https://api.z.ai/api/paas/v4",
            default_model="glm-5v-turbo",
        )
        api_key = resolved["api_key"]
        base_url = resolved["base_url"]
        model = resolved["model"]
        if not api_key:
            raise ProviderUnavailable("GLM_API_KEY or Z_AI_API_KEY is not set")
        if not base_url or not model:
            raise ProviderUnavailable("PROMPT_DIRECTOR_GLM_BASE_URL and PROMPT_DIRECTOR_GLM_MODEL must be set")
        provider_class = GlmVisionProvider if normalized == "glm" else ZaiVisionProvider
        return provider_class(api_key=api_key, base_url=base_url, model=model)
    if normalized == "kimi":
        resolved = _resolve_credentials(
            provider_name=normalized,
            env=env,
            config=config,
            api_key_env_names=("KIMI_API_KEY", "MOONSHOT_API_KEY"),
            base_url_env_names=("PROMPT_DIRECTOR_KIMI_BASE_URL", "KIMI_BASE_URL", "MOONSHOT_BASE_URL"),
            model_env_names=("PROMPT_DIRECTOR_KIMI_MODEL",),
            default_base_url="https://api.moonshot.ai/v1",
            default_model="kimi-k2.6",
        )
        api_key = resolved["api_key"]
        base_url = resolved["base_url"]
        model = resolved["model"]
        if not api_key:
            raise ProviderUnavailable("KIMI_API_KEY or MOONSHOT_API_KEY is not set")
        if not base_url or not model:
            raise ProviderUnavailable("PROMPT_DIRECTOR_KIMI_BASE_URL and PROMPT_DIRECTOR_KIMI_MODEL must be set")
        return KimiVisionProvider(api_key=api_key, base_url=base_url, model=model)
    if normalized in {"grok", "xai"}:
        resolved = _resolve_credentials(
            provider_name="grok",
            env=env,
            config=config,
            api_key_env_names=("XAI_API_KEY", "GROK_API_KEY"),
            base_url_env_names=("PROMPT_DIRECTOR_GROK_BASE_URL", "XAI_BASE_URL", "GROK_BASE_URL"),
            model_env_names=("PROMPT_DIRECTOR_GROK_MODEL", "XAI_MODEL", "GROK_MODEL"),
            default_base_url="https://api.x.ai/v1",
            default_model="grok-4.3",
        )
        api_key = resolved["api_key"]
        base_url = resolved["base_url"]
        model = resolved["model"]
        if not api_key:
            try:
                api_key = resolve_grok_oauth_bearer(env=env)
            except GrokOAuthUnavailable as exc:
                raise ProviderUnavailable(
                    "XAI_API_KEY, GROK_API_KEY, or Grok CLI OAuth (~/.grok/auth.json) is required"
                ) from exc
        if not base_url or not model:
            raise ProviderUnavailable("PROMPT_DIRECTOR_GROK_BASE_URL and PROMPT_DIRECTOR_GROK_MODEL must be set")
        return XaiVisionProvider(api_key=api_key, base_url=base_url, model=model)

    raise ProviderUnavailable(f"unknown analysis provider: {name}")


def _resolve_credentials(
    provider_name: str,
    env: Mapping[str, str],
    config: Mapping[str, Any],
    api_key_env_names: tuple[str, ...],
    base_url_env_names: tuple[str, ...],
    model_env_names: tuple[str, ...],
    default_base_url: str = "",
    default_model: str = "",
) -> dict[str, str]:
    source = _config_string(config, "credential_source") or "environment"
    profile_credentials: dict[str, str] = {}

    if source == "local_profile":
        try:
            profile_credentials = load_provider_credentials(
                provider_name=provider_name,
                profile_name=_config_string(config, "credential_profile") or "default",
                env=env,
            )
        except CredentialUnavailable as exc:
            raise ProviderUnavailable(str(exc)) from exc
        api_key = profile_credentials.get("api_key", "")
    else:
        api_key = _first_env_value(env, api_key_env_names)

    return {
        "api_key": api_key,
        "base_url": (
            _config_string(config, "base_url")
            or profile_credentials.get("base_url", "")
            or _first_env_value(env, base_url_env_names)
            or default_base_url
        ),
        "model": (
            _config_string(config, "model")
            or profile_credentials.get("model", "")
            or _first_env_value(env, model_env_names)
            or default_model
        ),
    }


def _first_env_value(env: Mapping[str, str], names: tuple[str, ...]) -> str:
    for name in names:
        value = env.get(name)
        if value:
            return value
    return ""


def _config_string(config: Mapping[str, Any], name: str) -> str:
    value = config.get(name)
    if value is None:
        return ""
    return str(value).strip()


__all__ = [
    "GlmVisionProvider",
    "KimiVisionProvider",
    "OpenAICompatibleVisionProvider",
    "ProviderUnavailable",
    "XaiVisionProvider",
    "ZaiVisionProvider",
    "get_provider",
]
