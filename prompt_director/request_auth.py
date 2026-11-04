from __future__ import annotations

from typing import Any, Mapping
from urllib import parse


def apply_provider_auth(
    url: str,
    *,
    api_key: str,
    auth_style: str = "bearer",
    extra_headers: Mapping[str, Any] | None = None,
) -> tuple[str, dict[str, str]]:
    headers = {"Content-Type": "application/json"}
    for key, value in (extra_headers or {}).items():
        if value is not None:
            headers[str(key)] = str(value)

    style = (auth_style or "bearer").strip()
    if style == "bearer":
        headers["Authorization"] = f"Bearer {api_key}"
    elif style == "x-api-key":
        headers["x-api-key"] = api_key
    elif style.startswith("header:"):
        headers[style.split(":", 1)[1]] = api_key
    elif style.startswith("query:"):
        parameter = style.split(":", 1)[1]
        split = parse.urlsplit(url)
        query = parse.parse_qsl(split.query, keep_blank_values=True)
        query.append((parameter, api_key))
        url = parse.urlunsplit((split.scheme, split.netloc, split.path, parse.urlencode(query), split.fragment))
    else:
        raise ValueError(f"Unsupported provider auth_style: {style}")
    return url, headers


def sanitize_provider_error(error: Exception, secrets: list[str] | tuple[str, ...]) -> str:
    message = str(error)
    for secret in secrets:
        if secret:
            message = message.replace(str(secret), "[redacted]")
    return message


__all__ = ["apply_provider_auth", "sanitize_provider_error"]
