<!--
============================================================================
ABRASAX — Unified Xenoliving System
PRIMAL_HEX: 4f5349524953424c58434b | phi: 1.618033988749895
CREATOR: Sir Charles Spikes (BASEDGOD)
============================================================================
-->

This repository is structured in a way that files are licensed differently
   - [`cuda.python`](./cuda_python/LICENSE): NVIDIA Software License
   - [`cuda.bindings`](./cuda_bindings/LICENSE): NVIDIA Software License
   - [`cuda.core`](./cuda_core/LICENSE) and everything else in this repository: Apache 2.0

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