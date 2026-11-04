from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


DIAGNOSTIC_SCHEMA_VERSION = "1.0"
_SEVERITIES = {"info", "warning", "error"}


def make_diagnostic(
    code: str,
    message: str,
    *,
    severity: str = "warning",
    subject: Mapping[str, Any] | None = None,
    evidence: Mapping[str, Any] | None = None,
    advisory: bool = True,
) -> dict[str, Any]:
    normalized_severity = severity if severity in _SEVERITIES else "warning"
    result: dict[str, Any] = {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "code": str(code or "unknown_diagnostic"),
        "severity": normalized_severity,
        "message": str(message or code or "Unknown diagnostic"),
        "advisory": bool(advisory),
    }
    if subject:
        result["subject"] = deepcopy(dict(subject))
    if evidence:
        result["evidence"] = deepcopy(dict(evidence))
    return result


def diagnostics_from_warnings(warnings: list[str]) -> list[dict[str, Any]]:
    return [_diagnostic_from_warning(warning) for warning in _dedupe(warnings)]


def merge_diagnostics(*groups: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for group in groups:
        for raw in group:
            item = deepcopy(dict(raw))
            evidence = item.get("evidence") if isinstance(item.get("evidence"), dict) else {}
            key = (
                str(item.get("code") or ""),
                str(evidence.get("legacy_warning") or item.get("message") or ""),
                str((item.get("subject") or {}).get("node_id") or ""),
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
    return merged


def _diagnostic_from_warning(warning: str) -> dict[str, Any]:
    text = str(warning or "").strip()
    code, _, detail = text.partition(":")
    message = detail.replace(":", "; ") if detail else code.replace("_", " ")
    severity = "info" if code.startswith(("provider_disabled", "model_context_missing")) else "warning"
    return make_diagnostic(
        code or "unknown_warning",
        message or "Unknown warning",
        severity=severity,
        evidence={"legacy_warning": text},
    )


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(str(item).strip() for item in items if str(item).strip()))


__all__ = [
    "DIAGNOSTIC_SCHEMA_VERSION",
    "diagnostics_from_warnings",
    "make_diagnostic",
    "merge_diagnostics",
]
