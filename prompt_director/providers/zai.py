from __future__ import annotations

from .openai_compatible import OpenAICompatibleVisionProvider


class ZaiVisionProvider(OpenAICompatibleVisionProvider):
    provider_name = "zai"
