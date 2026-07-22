from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Any, Mapping


class CredentialUnavailable(RuntimeError):
    pass


class CredentialFileMissing(CredentialUnavailable):
    pass


class CredentialFileInvalid(CredentialUnavailable):
    pass


SHARED_CREDENTIALS_SYSTEM_USER = "credentials"
SHARED_CREDENTIALS_FILENAME = "provider_profiles.json"

_LEGACY_CREDENTIAL_PATHS = (
    ("prompt_director", "credentials.json"),
    ("default", "prompt_director", "credentials.json"),
)


def default_credentials_path(env: Mapping[str, str] | None = None) -> Path:
    env = os.environ if env is None else env
    override = env.get("PROMPT_DIRECTOR_CREDENTIALS_FILE")
    if override:
        return Path(override).expanduser()

    return _shared_credentials_path()


def _candidate_credentials_paths(env: Mapping[str, str] | None = None) -> tuple[Path, ...]:
    env = os.environ if env is None else env
    override = env.get("PROMPT_DIRECTOR_CREDENTIALS_FILE")
    if override:
        return (Path(override).expanduser(),)

    user_dir = _comfy_user_directory()
    candidates = [_shared_credentials_path()]
    candidates.extend(user_dir.joinpath(*parts) for parts in _LEGACY_CREDENTIAL_PATHS)
    return tuple(_dedupe_paths(candidates))


def _shared_credentials_path() -> Path:
    try:
        import folder_paths  # type: ignore

        system_dir = Path(folder_paths.get_system_user_directory(SHARED_CREDENTIALS_SYSTEM_USER))
    except Exception:
        system_dir = _comfy_user_directory() / f"__{SHARED_CREDENTIALS_SYSTEM_USER}"

    return system_dir / SHARED_CREDENTIALS_FILENAME


def _comfy_user_directory() -> Path:
    try:
        import folder_paths  # type: ignore

        return Path(folder_paths.get_user_directory())
    except Exception:
        return Path.cwd() / "user"


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return unique


def load_provider_credentials(
    provider_name: str,
    profile_name: str = "default",
    credentials_path: str | Path | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    provider = _normalize_provider_name(provider_name)
    profile = (profile_name or "default").strip() or "default"
    paths = (
        (Path(credentials_path).expanduser(),)
        if credentials_path is not None
        else _candidate_credentials_paths(env=env)
    )

    errors: list[str] = []
    for path in paths:
        try:
            data = _read_credentials_file(path)
        except CredentialFileMissing as exc:
            errors.append(str(exc))
            continue
        except CredentialFileInvalid:
            raise

        try:
            provider_data = _profile_provider_data(data, profile, provider, path)
            return _provider_credentials(provider_data, profile, provider, path)
        except CredentialUnavailable as exc:
            errors.append(str(exc))

    searched = ", ".join(str(path) for path in paths)
    details = " ".join(errors)
    raise CredentialUnavailable(
        f"Local credential profile '{profile}' for provider '{provider}' was not found. "
        f"Searched: {searched}. {details}".strip()
    )


def _provider_credentials(
    provider_data: dict[str, Any],
    profile: str,
    provider: str,
    path: Path,
) -> dict[str, Any]:
    api_key = _string_value(provider_data, "api_key") or _string_value(provider_data, "key")
    if not api_key:
        raise CredentialUnavailable(
            f"Local credential profile '{profile}' for provider '{provider}' is missing api_key in {path}"
        )

    credentials: dict[str, Any] = {"api_key": api_key}
    for key in ("base_url", "model", "auth_style"):
        value = _string_value(provider_data, key)
        if value:
            credentials[key] = value
    headers = provider_data.get("headers")
    if isinstance(headers, dict):
        credentials["headers"] = {
            str(key): str(value)
            for key, value in headers.items()
            if str(key).strip() and value is not None
        }
    return credentials


def list_provider_profiles(
    *,
    credentials_path: str | Path | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    path = Path(credentials_path).expanduser() if credentials_path is not None else default_credentials_path(env=env)
    try:
        data = _read_credentials_file(path)
    except CredentialFileMissing:
        data = {}
    profiles = data.get("profiles") if isinstance(data.get("profiles"), dict) else {}
    masked: dict[str, Any] = {}
    for profile_name, profile_data in profiles.items():
        if not isinstance(profile_data, dict):
            continue
        providers: dict[str, Any] = {}
        for provider_name, provider_data in profile_data.items():
            if not isinstance(provider_data, dict):
                continue
            headers = provider_data.get("headers") if isinstance(provider_data.get("headers"), dict) else {}
            providers[str(provider_name)] = {
                "configured": bool(_string_value(provider_data, "api_key") or _string_value(provider_data, "key")),
                "base_url": _string_value(provider_data, "base_url"),
                "model": _string_value(provider_data, "model"),
                "auth_style": _string_value(provider_data, "auth_style") or "bearer",
                "header_names": [str(key) for key in headers],
            }
        if providers:
            masked[str(profile_name)] = providers
    return {"profiles": masked}


def save_provider_profile(
    profile_name: str,
    provider_name: str,
    values: Mapping[str, Any],
    *,
    credentials_path: str | Path | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    profile = _validate_identifier(profile_name or "default", "profile")
    provider = _validate_identifier(_normalize_provider_name(provider_name), "provider")
    path = Path(credentials_path).expanduser() if credentials_path is not None else default_credentials_path(env=env)
    try:
        data = _read_credentials_file(path)
    except CredentialFileMissing:
        data = {}

    profiles = data.get("profiles") if isinstance(data.get("profiles"), dict) else {}
    profile_data = profiles.get(profile) if isinstance(profiles.get(profile), dict) else {}
    current = profile_data.get(provider) if isinstance(profile_data.get(provider), dict) else {}
    updated = dict(current)

    api_key = values.get("api_key")
    if isinstance(api_key, str) and api_key.strip():
        updated["api_key"] = api_key.strip()
    for key in ("base_url", "model"):
        if key in values:
            value = str(values.get(key) or "").strip()
            if value:
                updated[key] = value
            else:
                updated.pop(key, None)
    if "auth_style" in values:
        auth_style = _validate_auth_style(str(values.get("auth_style") or "bearer").strip())
        updated["auth_style"] = auth_style
    if "headers" in values:
        headers = values.get("headers")
        if headers in (None, ""):
            updated.pop("headers", None)
        elif not isinstance(headers, dict):
            raise CredentialFileInvalid("headers must be a JSON object")
        else:
            updated["headers"] = _validate_headers(headers)

    if not _string_value(updated, "api_key") and not _string_value(updated, "key"):
        raise CredentialUnavailable(f"Provider '{provider}' requires a non-empty api_key")

    profile_data[provider] = updated
    profiles[profile] = profile_data
    data["profiles"] = profiles
    _atomic_write_credentials(path, data)
    return list_provider_profiles(credentials_path=path)


def delete_provider_profile(
    profile_name: str,
    provider_name: str,
    *,
    credentials_path: str | Path | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    profile = _validate_identifier(profile_name or "default", "profile")
    provider = _validate_identifier(_normalize_provider_name(provider_name), "provider")
    path = Path(credentials_path).expanduser() if credentials_path is not None else default_credentials_path(env=env)
    try:
        data = _read_credentials_file(path)
    except CredentialFileMissing:
        return {"profiles": {}}
    profiles = data.get("profiles") if isinstance(data.get("profiles"), dict) else {}
    profile_data = profiles.get(profile) if isinstance(profiles.get(profile), dict) else {}
    profile_data.pop(provider, None)
    if profile_data:
        profiles[profile] = profile_data
    else:
        profiles.pop(profile, None)
    data["profiles"] = profiles
    _atomic_write_credentials(path, data)
    return list_provider_profiles(credentials_path=path)


def _read_credentials_file(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise CredentialFileMissing(f"Local credentials file was not found: {path}") from exc
    except OSError as exc:
        raise CredentialFileInvalid(f"Local credentials file cannot be read: {path}: {exc}") from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CredentialFileInvalid(f"Local credentials file is not valid JSON: {path}: {exc.msg}") from exc

    if not isinstance(data, dict):
        raise CredentialFileInvalid(f"Local credentials file must contain a JSON object: {path}")
    return data


def _profile_provider_data(
    data: dict[str, Any],
    profile: str,
    provider: str,
    path: Path,
) -> dict[str, Any]:
    profiles = data.get("profiles")
    if not isinstance(profiles, dict):
        raise CredentialUnavailable(f"Local credentials file is missing a profiles object: {path}")

    profile_data = profiles.get(profile)
    if not isinstance(profile_data, dict):
        raise CredentialUnavailable(f"Local credential profile '{profile}' was not found in {path}")

    provider_data = profile_data.get(provider)
    if not isinstance(provider_data, dict):
        raise CredentialUnavailable(
            f"Local credential profile '{profile}' has no provider '{provider}' entry in {path}"
        )
    return provider_data


def _normalize_provider_name(provider_name: str) -> str:
    return (provider_name or "").strip().lower().replace("-", "_")


def _validate_identifier(value: str, label: str) -> str:
    normalized = value.strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", normalized):
        raise CredentialUnavailable(f"Invalid {label} name")
    return normalized


def _validate_auth_style(value: str) -> str:
    if value in {"bearer", "x-api-key"}:
        return value
    if re.fullmatch(r"header:[A-Za-z0-9-]{1,80}", value):
        return value
    if re.fullmatch(r"query:[A-Za-z0-9_.-]{1,80}", value):
        return value
    raise CredentialUnavailable("auth_style must be bearer, x-api-key, header:<Name>, or query:<param>")


def _validate_headers(headers: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in headers.items():
        name = str(key).strip()
        if not re.fullmatch(r"[A-Za-z0-9-]{1,80}", name):
            raise CredentialUnavailable(f"Invalid header name: {name}")
        rendered = str(value)
        if "\r" in rendered or "\n" in rendered:
            raise CredentialUnavailable(f"Invalid header value for {name}")
        result[name] = rendered
    return result


def _atomic_write_credentials(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path.parent, 0o700)
    except OSError:
        pass
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
        os.chmod(path, 0o600)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _string_value(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if value is None:
        return ""
    return str(value).strip()
