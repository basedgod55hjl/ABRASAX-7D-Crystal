<!--
============================================================================
ABRASAX — Unified Xenoliving System
PRIMAL_HEX: 4f5349524953424c58434b | phi: 1.618033988749895
CREATOR: Sir Charles Spikes (BASEDGOD)
============================================================================
-->

# GPU_COMPUTE — Master File Map
>
> All NVIDIA & AMD GPU files moved from `ABRASAX-AGI-MAIN` on 2026-02-21

## 01_MONITORING — GPU VRAM Surveillance

| File | Use Case |
|------|----------|
| `gpu_sovereign.py` | Real-time NVIDIA VRAM monitor via nvidia-smi. Triggers eviction protocols at >90% usage. 30s polling loop. |

## 02_NVIDIA_DEBUG — Diagnostics & Reference

| File | Use Case |
|------|----------|
| `debug_nvidia.py` | Debug nvidia-smi path resolution and GPU stat parsing |
| `nvidia-smi.json` | Cached nvidia-smi output snapshot for offline reference |
| `NVIDIA_L1B3RT4S_1-4.mkd` | NVIDIA documentation/notes (4 copies from L1B3RT4S project) |

## 03_CUDA_TOOLS — Core CUDA Infrastructure

| File | Use Case |
|------|----------|
| `cuda_path_mapper.py` | Finds CUDA installs, writes permanent system env vars, generates VRAM preloader script. Windows registry integration. |
| `cuda_load_balancer.py` | Maps all CUDA cores, enforces NVIDIA↔AMD app routing (LM Studio/Cursor on NVIDIA, background on AMD), auto-offloads at 85% threshold. |
| `cuda_core_activator.py` | Forces all CUDA cores active via matrix-multiply bursts. Keepalive mode maintains warm state. Generates `cuda_activation.conf`. |

## 04_CUDA_KERNELS — Custom GPU Kernels

| File | Use Case |
|------|----------|
| `flux_core.cuh` | CUDA header — flux core compute primitives |
| `sacred_sigmoid.cu` | Custom CUDA sigmoid activation kernel |
| `unfold_cbm.cu` | Custom CUDA unfold/transform kernel for CBM |

## 05_CUDA_RESEARCH — NVIDIA Research Archives

| Directory | Use Case |
|-----------|----------|
| `nvidia_research/cuda-samples/` | Official NVIDIA CUDA samples (Libraries, Features, Concepts) |
| `nvidia_research/cuda-python/` | CUDA Python bindings reference |

## 06_AMD_DIRECTML — AMD GPU Compute via DirectML

| File | Use Case |
|------|----------|
| `setup_amd_compute.py` | Creates isolated AMD venv, installs torch-directml, generates test script |
| `cbm_amd_env/` | Pre-built AMD DirectML virtual environment |

## 07_ROCM_BACKEND — AMD ROCm Compiler Backend

| File | Use Case |
|------|----------|
| `rocm.py` | TVM ROCm backend — ELF object linking, bitcode path resolution for AMD GPUs |
| `rocm_1767364909-911.py` | ROCm audit snapshots (3 timestamped versions) |

## 08_TENSORRT — NVIDIA Inference Optimization

| File | Use Case |
|------|----------|
| `tensorrt.py` | TVM TensorRT operator support — graph partitioning, op checkers for conv/pool/dense/etc, FP16 support |

## 09_VRAM_LOADER — Crystal ASI Model Loading

| File | Use Case |
|------|----------|
| `vram_manifold_loader.py` | 26-Dimensional VRAM manifold model loader. Maps attention heads to dims 1-8, MLP to dims 9-24. NVIDIA tensor core allocation. |

## 10_GPU_RUST — GPU Compute in Rust (7D Architecture)

| File | Use Case |
|------|----------|
| `gpu.rs` | Core GPU compute runtime in Rust |
| `wgpu_gpu.rs` | WebGPU backend implementation |
| `gpu_benchmark.rs` | GPU performance benchmark binary |
| `cuda.rs` | CUDA backend compiler in Rust |
| `gpu_launch.bat` | Windows GPU launcher batch script |
| `gpu_locked_launcher.py` | Python GPU locked launcher with resource pinning |

## 11_NONCUDA_FULLSTACK — Alternative GPU Backend

| Directory | Use Case |
|-----------|----------|
| `noncuda_fullstack/` | Full non-CUDA GPU backend: Rust kernel core, C++ bridge DLL, service layer, TypeScript UI, runtime |
| `runtime_noncuda/` | NonCUDA runtime build summary and config |

## 12_NVIDIA_LLM — NVIDIA LLM Templates & Notebooks

| File | Use Case |
|------|----------|
| `NVIDIA-Nemotron-Nano-v2.jinja` | Chat template for NVIDIA Nemotron Nano v2 model (llama.cpp) |
| `run-nvidia-1.ipynb` | NVIDIA GPU run notebook (Tools library) |
| `run-nvidia-2.ipynb` | NVIDIA GPU run notebook (ORB-TRADER project) |

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