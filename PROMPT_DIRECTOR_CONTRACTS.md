# Prompt Director Integration Contracts

Version: 0.3 development contract

Prompt Director remains a standalone ComfyUI node pack. These contracts let an optional orchestrator, model manager, or diagnostic tool understand Prompt Director without importing its implementation or hard-coding every node widget.

## Compatibility Rules

- Existing node class names, inputs, output order, and legacy routes remain supported.
- Contract major version `1` accepts additive minor fields.
- Consumers must preserve unknown fields when relaying or storing a payload.
- Existing string warnings remain available. Structured diagnostics are additive.
- `target_model` and each LoRA's string `compatibility` remain available for existing consumers.
- An explicit model target is authoritative. Compatibility and strength findings are advisory.
- Prompt Director execution never requires an agent, MCP connection, or Model Explorer HTTP route.

## Published Contracts

Machine-readable JSON Schemas live in `contracts/`:

- `model_context.v1`
- `edit_plan.v1`
- `diagnostic.v1`
- `critic_result.v1`
- `prompt_director_inspection.v1`
- `module_manifest.v1`

Python consumers can use:

```python
from prompt_director.contracts import load_contract_schema, validate_contract

schema = load_contract_schema("model_context.v1")
validated = validate_contract("model_context.v1", payload)
```

The validator is dependency-free and checks the stable required surface. The JSON Schemas are the language-neutral contract for richer validators and generated TypeScript types.

## Module Discovery

```text
GET /prompt_director/v1/manifest
```

The manifest advertises:

- Module and contract versions.
- Prompt Director node class names.
- Supported model targets.
- Runtime and schema routes.
- Capabilities such as model context, LoRA activation, references, masks, result criticism, and structured diagnostics.
- Safety behavior including headless execution and no automatic model replacement.

Individual schemas are available without filesystem access:

```text
GET /prompt_director/v1/contracts/<contract-name>
```

Example:

```text
GET /prompt_director/v1/contracts/model_context.v1
```

## Model Context

`model_context.v1` keeps the original fields and adds integration evidence:

```json
{
  "schema_version": "1.0",
  "target_model": "qwen_edit",
  "target": {
    "requested": "auto",
    "resolved": "qwen_edit",
    "source": "model_card.model_family",
    "confidence": 1.0,
    "evidence": "qwen_image_edit"
  },
  "model": {},
  "loras": [],
  "required_trigger_tokens": [],
  "warnings": [],
  "diagnostics": []
}
```

Target resolution precedence is explicit user selection, `model_card.model_family`, `prompt_director.model_family`, compatible families, base model, architecture, ModelSpec architecture, filename, then the existing Qwen Edit fallback.

Each LoRA retains `compatibility: compatible|incompatible|unknown` and additionally includes:

- `compatibility_detail`: reason, evidence, target, and advisory status.
- `strength_guidance`: resolved defaults/ranges and the metadata source for each field.

When `model_card.family_weight_overrides` contains the resolved target family, its values take precedence over global `prompt_director` strength guidance. Prompt Director reports the recommendation and range but does not modify loader widgets.

## Structured Diagnostics

`diagnostic.v1` provides stable fields for agent and UI consumers:

```json
{
  "schema_version": "1.0",
  "code": "lora_strength_above_max",
  "severity": "warning",
  "message": "The actual strength is above the tested maximum.",
  "subject": { "asset_name": "example.safetensors" },
  "evidence": { "actual": 1.3, "maximum": 1.0 },
  "advisory": true
}
```

Diagnostics are included in model context, critic results, and runtime inspection when warnings exist. Secret-like fields are redacted before inspection is stored.

## Versioned Inspection

```text
GET /prompt_director/v1/inspections
GET /prompt_director/v1/inspections?node_id=<id>
GET /prompt_director/v1/inspections/<id>
```

The versioned envelope identifies the module and module version. The original route remains available:

```text
GET /prompt_director/inspection[?node_id=<id>]
```

Inspection remains in-memory, bounded to 64 node records, and cleared when ComfyUI restarts. It never contains image tensors or credentials.

## Golden Fixtures

Language-neutral examples are stored in `tests/fixtures/contracts/`. They cover model context, edit plans, diagnostics, critic results, inspections, and the module manifest. Other products can copy these fixtures into contract tests until a shared contract package is available.

## Integration Guidance

An external product should:

1. Read the manifest and check supported contract major versions.
2. Read versioned inspection only after the relevant graph path executes.
3. Prefer structured diagnostics while continuing to display human-readable warnings.
4. Treat compatibility and strength guidance as evidence, not permission to mutate the graph.
5. Ask before applying a graph change unless the user already explicitly requested the repair.
6. Fall back gracefully when Prompt Director is absent or reports a newer unsupported major version.

An external product should not:

- Read Prompt Director's Python globals directly.
- Depend on an exact additive field list.
- copy credentials from inspection or workflow JSON.
- replace the selected model because metadata reports an incompatibility.
- make Prompt Director workflow execution depend on the orchestrator.
