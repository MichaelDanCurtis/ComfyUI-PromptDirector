from __future__ import annotations

import json
import os
from pathlib import Path
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
) -> dict[str, str]:
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
) -> dict[str, str]:
    api_key = _string_value(provider_data, "api_key") or _string_value(provider_data, "key")
    if not api_key:
        raise CredentialUnavailable(
            f"Local credential profile '{profile}' for provider '{provider}' is missing api_key in {path}"
        )

    credentials = {"api_key": api_key}
    for key in ("base_url", "model"):
        value = _string_value(provider_data, key)
        if value:
            credentials[key] = value
    return credentials


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


def _string_value(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if value is None:
        return ""
    return str(value).strip()
