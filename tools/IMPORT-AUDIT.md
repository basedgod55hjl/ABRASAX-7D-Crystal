# Import audit — ABRASAX-7D-Crystal

Ground truth for the claim *"`cuda_kernel_loader` is a local module, numpy present,
torch/cupy/scipy missing"*, plus the three defects that claim turned out to hide.

Re-verify everything with one command:

```
python tools/verify_all.py          # 14 stages, exit 0 == all green
python tools/verify_all.py --fast   # static half only (no CUDA stages)
```

## 1. Environment (measured, not assumed)

Interpreter — this box has exactly **one** install and one venv:

| | |
|---|---|
| `python` | 3.11.9, `C:\Users\lucar\AppData\Local\Programs\Python\Python311\python.exe` |
| `py -0p` | `-V:3.11 *` (only entry) |
| `C:\abrasax\.venv` | 3.11.9, `site-packages` holds **numpy only** |

| module | state | version / origin |
|---|---|---|
| numpy | **PRESENT** | 2.4.6 |
| psutil | PRESENT | 7.2.2 |
| PIL | PRESENT | 13.0.0 |
| torch | MISSING | — |
| cupy | MISSING | — |
| scipy | MISSING | — |
| numba | MISSING | — |
| tvm | MISSING | — |
| pycuda | MISSING | — |
| pyopencl | MISSING | — |

So the claim is correct, and the missing list is longer than stated: `numba`, `tvm`,
`pycuda`, `pyopencl` are absent too.

## 2. `cuda_kernel_loader` exists — twice

| path | size | what it is |
|---|---|---|
| `kernels/cuda/cuda_kernel_loader.py` | 49,723 B | **the real loader** (ctypes), binds 19 kernels |
| `cuda/cuda_kernel_loader.py` | 2,440 B | **a stub**: `KERNEL_REGISTRY` of `SimpleNamespace` placeholders |

Both import cleanly, both report the same device, and only one of them does
anything. Measured (`tools/_check_loader_shadow.py`):

```
cuda/ (stub first)     kernel_count=0   device="NVIDIA GeForce GTX 1660 Ti"
kernels/cuda/ (real)   kernel_count=19  device="NVIDIA GeForce GTX 1660 Ti"
```

The stub answers `device_name()` from the driver, so a stub-shadowed run looks
GPU-healthy while binding **zero** kernels. That is the dangerous shape of this
collision: no error, no warning, 19 kernels silently missing. Nothing in the repo
pins `kernels/cuda` ahead of `cuda/` on `sys.path`; it holds only because every
call site inserts the absolute `kernels/cuda` path itself.

## 3. Defects this audit found (all fixed)

### 3.1 `src/abrasax_pipeline.py:32` — phantom dependency, module unimportable

```python
from projects.neural_language.compiler import NeuralCompiler, QuantumSeed, PHI, ...
```

`projects` resolves to `C:\abrasax\PROJECTS`, which is **empty**, and no
`neural_language` package exists anywhere on `C:`. Result:

```
ModuleNotFoundError: No module named 'projects'
```

The file **compiles** perfectly — which is exactly why the compile gate was green
and this went unnoticed. Fixed by making the import lazy (PEP 562 `__getattr__`)
and failing at construction with the real diagnosis. `import abrasax_pipeline`
now succeeds; `ABRASAXPipeline()` raises a `ModuleNotFoundError` that names the
missing package. Proven by `tools/_check_phantom_fix.py` (4 checks).

### 3.2 `kernels/cuda/ctypes_wrapper.py` — 4 misqualified imports, pycuda backend dead

```python
from cuda.crystal_kernels import CrystalKernelContext, HAVE_PYCUDA   # line 94
from cuda.crystal_kernels import manifold_projection_7d as _pycu_fn  # line 160
...
```

`cuda/` contains no `__init__.py` and no `crystal_kernels.py` — only
`cuda_kernel_loader.py` and `build/`. Python 3 implicit namespace packages make
the import look plausible, but it can never resolve. Line 94 sits inside a
`try/except`, so the pycuda backend **always** failed to initialise and the
wrapper silently fell back to ctypes; with pycuda installed it would still never
activate. The intended module is the sibling `kernels/cuda/crystal_kernels.py`,
which defines every imported name (`CrystalKernelContext:157`, `HAVE_PYCUDA:45`,
`manifold_projection_7d:357`, `phi_harmonic_modulation:373`, `cbm_flux_unfold:390`).
Rewritten to the sibling form (`tools/_fix_sibling_imports.py`, backup
`.bak-phantomsibling`).

### 3.3 Scratch tools that only ran from the right cwd

`tools/_diag_grid_src.py` did `import cuda_kernel_loader` with no path setup:
running it from the repo root gave `ModuleNotFoundError`. `tools/_trace_grid.py`
worked, but via a hardcoded `C:\abrasax\...` string. Both now resolve the kernel
dir from `__file__` and run from any directory.

## 4. What is still legitimately missing (not fixed, by design)

`kernels/gpu_compute/**` imports modules this box does not have, at module scope:

```
torch   abrasax_cuda_engine.py:8, bridge_7d.py:9, gpu_pipeline.py:15, math_7d.py:8
tvm     backends/07_ROCM_BACKEND/rocm.py, backends/08_TENSORRT/tensorrt.py
```

28 sites, all in scope `external` in `tools/_import_matrix.py`, which is why it
reports `core_fatal=0, external_only=28`. Those are optional accelerated backends
with their own dependencies; nothing in the core crystal path imports them. They
will raise `ModuleNotFoundError` the moment anyone imports them here — that is a
packaging decision, not a bug to paper over with a `try/except`.

Everything inside the core path is guarded and degrades cleanly: `pycuda`
(9 sites), `numba` (2), `cuda` (1), `projects` (1).

## 5. Gates added

| tool | what it proves |
|---|---|
| `_import_matrix.py` | resolution + severity per import site (FATAL / DEFERRED / OPTIONAL), core vs external scope |
| `_phantom_imports.py` | catches `from pkg.mod import` where `pkg/mod.py` doesn't exist but a sibling does |
| `_import_controls.py` | 5 controls: both gates go red on the reverted files, and stay quiet when guarded |
| `_check_phantom_fix.py` | `abrasax_pipeline` importable, fails loudly and correctly on use |
| `_check_loader_shadow.py` | stub vs real loader, by kernel count, in isolated subprocesses |
| `_fix_phantom_import.py`, `_fix_sibling_imports.py` | idempotent, CRLF-aware fixers with backups |

Current status: `python tools/verify_all.py` → **14 stages, 14 pass, 0 fail**, exit 0.
