from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Mapping
from urllib import parse, request


XAI_OAUTH_CLIENT_ID = "b1a00492-073a-47ea-816f-4c329264a828"
XAI_OAUTH_TOKEN_URL = "https://auth.x.ai/oauth2/token"
GROK_AUTH_SCOPE_KEY = f"https://auth.x.ai::{XAI_OAUTH_CLIENT_ID}"
ACCESS_TOKEN_REFRESH_SKEW_MS = 120_000


class GrokOAuthUnavailable(RuntimeError):
    pass


def grok_auth_path(env: Mapping[str, str] | None = None) -> Path:
    env = os.environ if env is None else env
    home = Path(env.get("GROK_HOME") or env.get("HOME") or "~").expanduser()
    return home / ".grok" / "auth.json"


def resolve_grok_oauth_bearer(env: Mapping[str, str] | None = None) -> str:
    """Return a Grok CLI OAuth bearer from ~/.grok/auth.json, refreshing if needed."""
    env = os.environ if env is None else env
    path = grok_auth_path(env=env)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise GrokOAuthUnavailable(
            "Grok OAuth credentials not found (~/.grok/auth.json). Run `grok login` once."
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise GrokOAuthUnavailable(f"Cannot read Grok OAuth credentials from {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise GrokOAuthUnavailable(f"Grok OAuth credentials must be a JSON object: {path}")

    picked = _pick_grok_auth_entry(raw)
    if picked is None:
        raise GrokOAuthUnavailable(
            "No Grok OAuth credentials in ~/.grok/auth.json. Run `grok login` to sign in."
        )

    scope_key, entry = picked
    now_ms = int(time.time() * 1000)
    if _token_is_expiring(entry, now_ms) or not _access_token_from_entry(entry):
        entry = _refresh_grok_oauth_entry(entry)
        raw[scope_key] = entry
        _atomic_write_json(path, raw)

    bearer = _access_token_from_entry(entry)
    if not bearer:
        raise GrokOAuthUnavailable(
            "Grok OAuth access token is missing after refresh. Run `grok login` to sign in."
        )
    return bearer


def _access_token_from_entry(entry: dict[str, Any]) -> str:
    key = entry.get("key")
    if isinstance(key, str) and key.strip():
        return key.strip()
    access = entry.get("access_token")
    if isinstance(access, str) and access.strip():
        return access.strip()
    return ""


def _pick_grok_auth_entry(raw: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    preferred = raw.get(GROK_AUTH_SCOPE_KEY)
    if isinstance(preferred, dict) and (
        _string_value(preferred, "refresh_token") or _access_token_from_entry(preferred)
    ):
        return GROK_AUTH_SCOPE_KEY, preferred

    for scope_key, entry in raw.items():
        if not isinstance(entry, dict):
            continue
        if "auth.x.ai" in scope_key and (
            _string_value(entry, "refresh_token") or _access_token_from_entry(entry)
        ):
            return scope_key, entry

    for scope_key, entry in raw.items():
        if not isinstance(entry, dict):
            continue
        if _string_value(entry, "refresh_token") or _access_token_from_entry(entry):
            return scope_key, entry
    return None


def _parse_expires_at(expires_at: Any) -> int | None:
    if expires_at is None:
        return None
    if isinstance(expires_at, (int, float)) and float(expires_at) > 0:
        value = float(expires_at)
        return int(value * 1000) if value < 1_000_000_000_000 else int(value)
    if isinstance(expires_at, str) and expires_at.strip():
        try:
            from datetime import datetime

            dt = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
            return int(dt.timestamp() * 1000)
        except ValueError:
            return None
    return None


def _token_is_expiring(entry: dict[str, Any], now_ms: int, skew_ms: int = ACCESS_TOKEN_REFRESH_SKEW_MS) -> bool:
    expiry_ms = _parse_expires_at(entry.get("expires_at"))
    if expiry_ms is None:
        return False
    return now_ms >= expiry_ms - skew_ms


def _refresh_grok_oauth_entry(entry: dict[str, Any]) -> dict[str, Any]:
    refresh_token = _string_value(entry, "refresh_token")
    if not refresh_token:
        raise GrokOAuthUnavailable(
            "Grok OAuth refresh_token is missing. Run `grok login` to sign in again."
        )

    client_id = _string_value(entry, "oidc_client_id") or XAI_OAUTH_CLIENT_ID
    body = parse.urlencode(
        {
            "grant_type": "refresh_token",
            "client_id": client_id,
            "refresh_token": refresh_token,
        }
    ).encode("utf-8")
    req = request.Request(
        XAI_OAUTH_TOKEN_URL,
        data=body,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise GrokOAuthUnavailable(f"Grok OAuth token refresh failed: {exc}") from exc

    access_token = _string_value(payload, "access_token")
    if not access_token:
        raise GrokOAuthUnavailable("Grok OAuth token refresh response missing access_token.")

    rotated_refresh = _string_value(payload, "refresh_token") or refresh_token
    expires_in = payload.get("expires_in")
    expires_at = entry.get("expires_at")
    if isinstance(expires_in, (int, float)) and float(expires_in) > 0:
        from datetime import datetime, timezone

        expires_at = datetime.fromtimestamp(time.time() + float(expires_in), tz=timezone.utc).isoformat()

    updated = dict(entry)
    updated["key"] = access_token
    updated["access_token"] = access_token
    updated["refresh_token"] = rotated_refresh
    updated["expires_at"] = expires_at
    updated["oidc_client_id"] = client_id
    updated.setdefault("auth_mode", "oidc")
    return updated


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".auth-{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _string_value(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if isinstance(value, str):
        return value.strip()
    return ""