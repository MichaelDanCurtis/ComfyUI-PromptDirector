from __future__ import annotations

from .openai_compatible import OpenAICompatibleVisionProvider


class KimiVisionProvider(OpenAICompatibleVisionProvider):
    provider_name = "kimi"
    include_temperature = False
