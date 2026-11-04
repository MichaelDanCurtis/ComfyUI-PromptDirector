from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Mapping

from .diagnostics import diagnostics_from_warnings
from .manifest import MODULE_ID, MODULE_VERSION


_MAX_INSPECTIONS = 64
_INSPECTIONS: OrderedDict[str, dict[str, Any]] = OrderedDict()
_SECRET_KEYS = {
    "api_key",
    "key",
    "access_token",
    "refresh_token",
    "authorization",
    "password",
    "secret",
}


def record_inspection(node_id: Any, kind: str, payload: Mapping[str, Any]) -> None:
    key = str(node_id if node_id not in (None, "") else f"unbound:{kind}")
    public_payload = _redact(deepcopy(dict(payload)))
    _add_missing_diagnostics(public_payload)
    _INSPECTIONS[key] = {
        "node_id": key,
        "kind": str(kind),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "payload": public_payload,
    }
    _INSPECTIONS.move_to_end(key)
    while len(_INSPECTIONS) > _MAX_INSPECTIONS:
        _INSPECTIONS.popitem(last=False)


def list_inspections(node_id: Any = None) -> dict[str, Any]:
    if node_id not in (None, ""):
        item = _INSPECTIONS.get(str(node_id))
        inspections = [deepcopy(item)] if item else []
    else:
        inspections = [deepcopy(item) for item in _INSPECTIONS.values()]
    return {"schema_version": "1.0", "inspections": inspections}


def list_inspections_v1(node_id: Any = None) -> dict[str, Any]:
    payload = list_inspections(node_id=node_id)
    return {
        "schema_version": payload["schema_version"],
        "module_id": MODULE_ID,
        "module_version": MODULE_VERSION,
        "inspections": payload["inspections"],
    }


def clear_inspections() -> None:
    _INSPECTIONS.clear()


def _redact(value: Any, key: str = "") -> Any:
    normalized = key.lower().replace("-", "_")
    if normalized in _SECRET_KEYS or normalized.endswith("_api_key") or normalized.endswith("_token"):
        return "[redacted]"
    if isinstance(value, dict):
        return {str(item_key): _redact(item, str(item_key)) for item_key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _add_missing_diagnostics(payload: dict[str, Any]) -> None:
    warnings = payload.get("warnings")
    if isinstance(warnings, list) and not isinstance(payload.get("diagnostics"), list):
        payload["diagnostics"] = diagnostics_from_warnings(warnings)
    critique = payload.get("critique")
    if isinstance(critique, dict):
        critique_warnings = critique.get("warnings")
        if isinstance(critique_warnings, list) and not isinstance(critique.get("diagnostics"), list):
            critique["diagnostics"] = diagnostics_from_warnings(critique_warnings)


__all__ = ["clear_inspections", "list_inspections", "list_inspections_v1", "record_inspection"]
