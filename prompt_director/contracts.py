from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping


CONTRACT_SCHEMA_VERSION = "1.0"
CONTRACT_NAMES = (
    "model_context.v1",
    "edit_plan.v1",
    "diagnostic.v1",
    "critic_result.v1",
    "prompt_director_inspection.v1",
    "module_manifest.v1",
)

_REQUIRED_FIELDS = {
    "model_context.v1": {
        "schema_version": str,
        "target_model": str,
        "model": dict,
        "loras": list,
        "required_trigger_tokens": list,
        "warnings": list,
    },
    "edit_plan.v1": {
        "schema_version": str,
        "mode": str,
        "target_model": str,
        "original_intent": str,
        "changes": list,
        "preserve": list,
        "references": list,
        "warnings": list,
    },
    "diagnostic.v1": {
        "schema_version": str,
        "code": str,
        "severity": str,
        "message": str,
        "advisory": bool,
    },
    "critic_result.v1": {
        "schema_version": str,
        "verdict": str,
        "scores": dict,
        "observations": list,
        "revised_prompt": str,
        "warnings": list,
    },
    "prompt_director_inspection.v1": {
        "schema_version": str,
        "inspections": list,
    },
    "module_manifest.v1": {
        "schema_version": str,
        "module_id": str,
        "module_version": str,
        "contracts": dict,
        "node_types": list,
        "capabilities": list,
        "routes": dict,
    },
}


class ContractValidationError(ValueError):
    pass


def contracts_directory() -> Path:
    return Path(__file__).resolve().parent.parent / "contracts"


def load_contract_schema(contract_name: str) -> dict[str, Any]:
    name = _validate_contract_name(contract_name)
    path = contracts_directory() / f"{name}.schema.json"
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ContractValidationError(f"Contract schema is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ContractValidationError(f"Contract schema is invalid JSON: {path}: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise ContractValidationError(f"Contract schema must contain an object: {path}")
    return parsed


def validate_contract(contract_name: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    name = _validate_contract_name(contract_name)
    if not isinstance(payload, Mapping):
        raise ContractValidationError(f"{name} payload must be an object")
    value = deepcopy(dict(payload))
    for field, expected_type in _REQUIRED_FIELDS[name].items():
        if field not in value:
            raise ContractValidationError(f"{name} is missing required field '{field}'")
        if not isinstance(value[field], expected_type):
            raise ContractValidationError(
                f"{name}.{field} must be {expected_type.__name__}, got {type(value[field]).__name__}"
            )
    _validate_schema_version(name, str(value["schema_version"]))
    if name == "diagnostic.v1" and value["severity"] not in {"info", "warning", "error"}:
        raise ContractValidationError("diagnostic.v1.severity must be info, warning, or error")
    if name == "model_context.v1":
        _validate_object_list(name, "loras", value["loras"])
        _validate_string_list(name, "required_trigger_tokens", value["required_trigger_tokens"])
        _validate_string_list(name, "warnings", value["warnings"])
    elif name == "prompt_director_inspection.v1":
        _validate_object_list(name, "inspections", value["inspections"])
    elif name == "module_manifest.v1":
        _validate_string_list(name, "node_types", value["node_types"])
        _validate_string_list(name, "capabilities", value["capabilities"])
    return value


def _validate_contract_name(contract_name: str) -> str:
    name = str(contract_name or "").strip()
    if name not in CONTRACT_NAMES:
        raise ContractValidationError(f"Unknown Prompt Director contract: {name}")
    return name


def _validate_schema_version(contract_name: str, version: str) -> None:
    major = version.split(".", 1)[0]
    if major != "1":
        raise ContractValidationError(f"{contract_name} requires schema major version 1, got {version}")


def _validate_string_list(contract_name: str, field: str, values: list[Any]) -> None:
    if any(not isinstance(item, str) for item in values):
        raise ContractValidationError(f"{contract_name}.{field} must contain only strings")


def _validate_object_list(contract_name: str, field: str, values: list[Any]) -> None:
    if any(not isinstance(item, dict) for item in values):
        raise ContractValidationError(f"{contract_name}.{field} must contain only objects")


__all__ = [
    "CONTRACT_NAMES",
    "CONTRACT_SCHEMA_VERSION",
    "ContractValidationError",
    "contracts_directory",
    "load_contract_schema",
    "validate_contract",
]
