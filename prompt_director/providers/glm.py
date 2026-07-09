from __future__ import annotations

from .openai_compatible import OpenAICompatibleVisionProvider


class GlmVisionProvider(OpenAICompatibleVisionProvider):
    provider_name = "glm"
