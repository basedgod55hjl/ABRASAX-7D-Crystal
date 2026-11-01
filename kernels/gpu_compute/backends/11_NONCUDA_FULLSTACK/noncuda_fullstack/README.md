<!--
============================================================================
ABRASAX — Unified Xenoliving System
PRIMAL_HEX: 4f5349524953424c58434b | phi: 1.618033988749895
CREATOR: Sir Charles Spikes (BASEDGOD)
============================================================================
-->

# Non-CUDA Full Stack Runtime

This stack is an isolated full-stack runtime that avoids CUDA-specific kernels while keeping heterogeneous execution support.

## Layers

- `kernel/rust`: low-level compute abstractions and CPU/OpenCL/Vulkan backend dispatch.
- `kernel/cpp`: native bridge for backend detection and shared-memory transfer simulation.
- `runtime/python`: CBM orchestration, LM Studio integration, dev thinking logs.
- `service/python`: local HTTP control plane.
- `ui/typescript`: lightweight dashboard scaffold for status + logs.
- `ops/powershell`: Windows startup + health checks.
- `ops/python`: binary inspection and controlled patch workflow.
- `build`: one-command build/verify orchestrator.

## Model Lock

This stack is hard-locked to:

- LM Studio host: `http://127.0.0.1:1234`
- Model: `essentialai/rnj-1`

## Quick Start

```powershell
python noncuda_fullstack/build/build_noncuda_fullstack.py
powershell -ExecutionPolicy Bypass -File noncuda_fullstack/ops/powershell/Start-NonCudaStack.ps1
```

## Dev Thinking Log

CBM runtime writes newline-delimited JSON traces to:

- `runtime/noncuda/cbm_dev_thinking.log`

## Local LLM + Loki

<!-- local-llm-loki -->

**Local LLM**: Run a local model server (e.g. ollama, text-generation-webui, or another local API and set the application env var `LOCAL_LLM_URL` to its URL, e.g. `http://localhost:8080`.

**Loki (logs)**: Run a Loki instance (or use an existing endpoint) and set `LOKI_URL` to the Loki push/query URL, e.g. `http://localhost:3100`. Configure your logger or `promtail` to forward logs to that endpoint.

Example env vars:
``
LOCAL_LLM_URL=http://localhost:8080
LOKI_URL=http://localhost:3100
``

This section was added automatically to surface local LLM and Loki setup notes.