from __future__ import annotations

from copy import deepcopy
from typing import Any

from .contracts import CONTRACT_NAMES


MODULE_ID = "comfyui-prompt-director"
MODULE_VERSION = "0.3.0.dev0"
MODULE_MANIFEST_SCHEMA_VERSION = "1.0"

_NODE_TYPES = (
    "PromptDirector",
    "PromptDirectorContext",
    "PromptDirectorReference",
    "PromptDirectorAuto",
    "PromptDirectorResultCritic",
    "PromptDirectorConfig",
    "PromptDirectorImageDescribe",
    "PromptDirectorPromptEnhancer",
    "PromptProducer",
    "PromptDirectorDebugPreview",
)

_MODEL_TARGETS = (
    "auto",
    "qwen_edit",
    "qwen_image",
    "krea2",
    "ltx",
    "wan",
    "flux",
    "sdxl",
    "z_image",
)


def module_manifest() -> dict[str, Any]:
    return deepcopy(
        {
            "schema_version": MODULE_MANIFEST_SCHEMA_VERSION,
            "module_id": MODULE_ID,
            "module_version": MODULE_VERSION,
            "contracts": {name: "1.0" for name in CONTRACT_NAMES},
            "node_types": list(_NODE_TYPES),
            "capabilities": [
                "contract_schemas",
                "deterministic_prompt_compilation",
                "model_context",
                "lora_activation",
                "reference_roles",
                "mask_awareness",
                "provider_image_analysis",
                "result_criticism",
                "runtime_inspection",
                "structured_diagnostics",
            ],
            "model_targets": list(_MODEL_TARGETS),
            "routes": {
                "manifest": "/prompt_director/v1/manifest",
                "inspections": "/prompt_director/v1/inspections",
                "inspection": "/prompt_director/v1/inspections/{node_id}",
                "contract_schema": "/prompt_director/v1/contracts/{contract_name}",
                "legacy_inspection": "/prompt_director/inspection",
            },
            "behavior": {
                "headless_execution": True,
                "agent_required": False,
                "recommendations_are_advisory": True,
                "automatic_model_replacement": False,
                "unknown_fields": "preserve",
            },
        }
    )


__all__ = [
    "MODULE_ID",
    "MODULE_MANIFEST_SCHEMA_VERSION",
    "MODULE_VERSION",
    "module_manifest",
]
