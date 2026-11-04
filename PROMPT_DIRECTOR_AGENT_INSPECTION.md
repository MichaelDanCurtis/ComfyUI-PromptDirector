# Prompt Director Agent Inspection

## Goal

Give Comfy AI and Comfy MCP enough read-only evidence to make specific observations about Prompt Director, model metadata, LoRA compatibility, graph wiring, and rendered edit quality before proposing a change.

## Evidence Layers

The agent uses two complementary sources:

1. Live graph state: node classes, widgets, execution modes, connections, model-loader filenames, LoRA-loader filenames, and actual model/CLIP strengths.
2. Prompt Director runtime state: resolved Model Explorer context, source analysis, structured edit plan, exact final prompt, warnings, and Result Critic verdict from the last execution of each node.

Graph state explains what is connected. Runtime state explains what Prompt Director actually resolved and compiled. Neither source contains provider credentials or image tensors.

## Interfaces

### ComfyUI Runtime Route

```text
GET /prompt_director/inspection
GET /prompt_director/inspection?node_id=<ComfyUI node id>
```

The registry is in memory, bounded to 64 entries, and cleared when ComfyUI restarts. Nodes must execute before they have runtime evidence.

Versioned consumers should use:

```text
GET /prompt_director/v1/manifest
GET /prompt_director/v1/inspections
GET /prompt_director/v1/inspections/<ComfyUI node id>
```

The versioned envelope identifies the module and contract version. Model context and critic payloads include
structured diagnostics with stable codes, subjects, evidence, and advisory status alongside legacy warning strings.
Contract schemas are available from `GET /prompt_director/v1/contracts/<contract-name>`.

### Comfy MCP Server Tool

```text
prompt_director_inspect(node_id?)
```

Use this for headless or MCP-level access to sanitized runtime evidence.

### Live Panel Tool

```text
panel_audit_prompt_director()
```

The panel audit correlates runtime evidence with the currently viewed graph. It detects:

- Prompt Director output not connected to a model prompt path.
- `model_target=auto` without Prompt Director Context.
- Context selecting a different model than the graph's loader.
- Loaded LoRAs missing from `lora_stack_json`.
- Actual LoRA strengths outside embedded tested ranges.
- Embedded family incompatibility.
- Auto receiving an image without provider analysis.
- Result Critic missing a provider or requesting revision.
- Prompt Director nodes that have not executed yet.

Recommendations contain a concrete `panel_set_widget` proposal and `requires_confirmation: true`. The audit never changes the graph.

## Agent Behavior

When Prompt Director nodes are present, Comfy AI should run the audit before declaring the prompt/model/LoRA setup correct or diagnosing a failed edit. It should report concise evidence, including a positive observation when context is coherent.

The agent may apply a proposed change immediately only when the user already asked it to fix the workflow. Otherwise it should explain the observation and ask whether to make the change. Normal panel snapshot and undo behavior still applies to approved edits.

## Example

```text
Observation: Qwen Image Edit 2511 is correctly resolved as qwen_edit, but
clown_nose.safetensors is loaded at 1.3 while its embedded tested maximum is 1.0.
The trigger clownnose is present in the compiled prompt.

Proposed change: set the LoRA loader model strength to 1.0. Apply it?
```
