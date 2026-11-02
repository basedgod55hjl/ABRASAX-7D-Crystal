# cuda_kernel_loader: location, imports, resolution

Answering "check location/imports" with measurements. Reproduce:
`python tools\_audit_loader_location.py` → **11 checks: 11 pass, 0 fail**.

## A. Location — three files carry the name, one is importable-and-real

| path | bytes | sha256[0:16] | binds `cuLaunchKernel` | importable |
|---|---|---|---|---|
| `kernels/cuda/cuda_kernel_loader.py` | 49,723 | `0b104712a7aa31cb` | **yes** — the real loader | yes |
| `cuda/cuda_kernel_loader.py` | 2,440 | `f2b48f4436af6f20` | no — stub (`SimpleNamespace` registry) | yes |
| `kernels/cuda/cuda_kernel_loader.py.bak-gridfix-20261101` | 50,486 | `f797204c30e0ecb6` | yes — the **pre-fix** loader, kept for forensics | no (not `.py`) |

Only one `.py` in the tree is a real loader, and it is the one in `kernels/cuda/`.

## B. Imports — numpy and the standard library, nothing else

```
module-level stdlib : __future__, ctypes, dataclasses, logging, os, pathlib, sys, time, typing
module-level third  : numpy
deferred stdlib     : none
deferred third      : none
dynamic mechanisms  : none          (no importlib / __import__ / exec / eval)
```

"Deferred: none" matters: there is no function-level import that a review could
miss, and no dynamic import that could pull in something invisible. The GPU path
binds `nvcuda.dll` through `ctypes.WinDLL` — that is a DLL load, not a Python
dependency.

## C. Resolution — every entry point binds the real file

Recorded from **real runs** with an `atexit` recorder in a `sitecustomize`, not
inferred from reading the source:

```
tools/verify_kernel_math.py            -> kernels\cuda\cuda_kernel_loader.py   kernels=19
tools/verify_kernel_math_quantum.py    -> kernels\cuda\cuda_kernel_loader.py   kernels=19
tools/diagnose_quantum_grid.py         -> kernels\cuda\cuda_kernel_loader.py   kernels=19
tools/_trace_grid.py                   -> kernels\cuda\cuda_kernel_loader.py   kernels=19
kernels/cuda/test_kernel_loader.py     -> kernels\cuda\cuda_kernel_loader.py   kernels=19
```

## D. Shadowing — the stub is reachable, and it would be silent

```
cuda/ first on sys.path -> cuda\cuda_kernel_loader.py (kernels=0)
repo files putting cuda/ on sys.path: none
```

The stub exposes **0 kernels** and no `available_kernels` attribute, yet its
`device_name()` still reports the GTX 1660 Ti (it queries the driver directly).
So a shadowed run looks GPU-healthy and binds nothing. Today that cannot happen
because every caller inserts the absolute `kernels/cuda` path itself and nothing
in the repo ever adds `cuda/` — verified, 0 references.

That is a convention, not a guarantee: `PYTHONPATH=C:\abrasax\SYSTEMS\ABRASAX-7D-Crystal\cuda`
would do it. If this ever needs hardening, make the stub refuse to instantiate
(or delete it), rather than relying on path order.
