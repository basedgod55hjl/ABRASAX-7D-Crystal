# What the GPU verifications actually need

Question: *"the GPU ones need `cuda_kernel_loader` + a GPU + likely CUDA runtime;
probably cupy or torch CUDA — if so the GPU verifications can't run here."*

**They do not need cupy, torch, pycuda, numba or pyopencl — and they do run here.**
Measured, not assumed: `python tools\_loader_requirements.py`.

## The loader's real dependency closure

```
== A. loader imports ==
  stdlib     : __future__, ctypes, dataclasses, logging, os, pathlib, sys, time, typing
  third-party: numpy
```

That is the whole list. `cuda_kernel_loader.py` binds the **CUDA Driver API**
through `ctypes.WinDLL("nvcuda.dll")` and loads prebuilt modules — no framework.

```
== B. artifacts ==
  required  crystal_kernels_v2.ptx           present   149146 bytes
  required  crystal_kernels_v1.ptx           present    43800 bytes
  required  physx_crystal_dynamics.ptx       present     9264 bytes
  required  crystal_kernels_v2.cubin         present   119608 bytes
  required  crystal_kernels_v1.cubin         present    33848 bytes
  required  physx_crystal_dynamics.cubin     present    15912 bytes
  optional  crystal_kernels.dll              absent (PTX path used)
  ptx target: .target sm_75
```

The DLL is genuinely optional: `cuda_kernel_loader.py:443` loads it only
`if use_dll and _DLL_PATH.exists()`. The PTX modules are the primary path, which
is why 19 kernels bind with no DLL on disk.

```
== C. runtime ==
  driver        13050        (CUDA 13.5)
  device        NVIDIA GeForce GTX 1660 Ti
  cc            7.5          (matches the .target sm_75 above)
  mp_count      24
  total_mem     4.0 GiB
```

## The proof that the framework stack is irrelevant

Section E re-runs the GPU correctness gates with a `meta_path` hook that raises
`ImportError` for `cupy`, `torch`, `pycuda`, `numba` and `pyopencl`:

```
== E. GPU gates with the framework stack blocked ==
  kernel_math    PASS  0 failed
  mhq7d_quantum  PASS  0 failed
```

Same numbers as without the sandbox. The GPU verification does not merely happen
to work in their absence — it never touches them.

## So: what *would* stop the GPU gates here

| prerequisite | status now | if missing |
|---|---|---|
| NVIDIA driver (`nvcuda.dll`) | present, driver 13050 | `cuInit` fails, loader raises "NVIDIA drivers required" |
| A CUDA device | GTX 1660 Ti, 4 GiB | no device → kernels cannot launch |
| `build/*.ptx` + `*.cubin` | 6/6 present | nothing to load |
| `numpy` | 2.4.6 | the reference implementations and array plumbing |
| ptx arch matching the device | `sm_75` on cc 7.5 | `cuModuleLoadData` error |

Not on the list: cupy, torch, numba, pycuda, pyopencl, and the CUDA **toolkit**
(`nvcc` is only needed to rebuild the `.ptx`, not to run it).

## Where those frameworks *are* required

Only outside the live path, which is why `_import_matrix.py` reports them as
`scope=external`:

* `kernels/gpu_compute/**` — 28 module-level `torch`/`tvm` imports (optional
  accelerated backends, unused by the crystal path).
* `kernels/cuda/crystal_kernels.py` — the legacy `pycuda` implementation,
  `HAVE_PYCUDA = False` here, so `ctypes_wrapper.py` takes the ctypes route.
* `kernels/cuda/test_kernels.py` — the legacy pycuda test suite; its phases 3–5
  fail for that reason, which is why `test_kernel_loader.py --quick` is the loader
  gate that matters.

## Reproduce

```
python tools\_loader_requirements.py     # 8 checks: 8 pass, 0 fail
```
