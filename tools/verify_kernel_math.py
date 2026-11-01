"""Prove the 7D mH-Q kernels compute what they claim, against CPU references.

WHY THIS EXISTS
    The architecture named in the title -- manifold-constrained, holographic,
    quantum -- is implemented in SYSTEMS/ABRASAX-7D-Crystal/kernels/cuda/
    crystal_kernels_v2.cu. test_kernel_loader.py proves all 19 kernels BIND and
    LAUNCH (Phase 3 "19 passed"), and bench them (Phase 6), but "launched" is not
    "correct": a kernel that writes zeros still launches. There is a
    cpu_manifold_projection_7d reference in the same file that is only compiled
    for the host, so the comparison was never made.

    This runs each kernel on the GTX 1660 Ti and compares to an independent
    NumPy implementation of the documented formula.

THE FORMULAS (read from the .cu, not assumed)
    manifold_projection_7d_kernel, per element (row = idx/dim, col = idx%dim):
        t   = x / (1 + |x| + PHI_INV)          Poincare-ball projection
        t  += STABILITY_EPSILON  if col == row % dim      S^2 stability restore
        out = tanhf(t * PHI_INV)                          phi_tanh
      constants: PHI 1.618033988749895, PHI_INV 0.618033988749895,
                 STABILITY_EPSILON 0.01

    holographic_interference_kernel, per element:
        ref_phase = atan2(sin(ref*PI), cos(ref*PI))
        obj_phase = atan2(sin(obj*PI), cos(obj*PI))
        out = 0.5 + 0.5 * tanhf(cos(ref_phase - obj_phase) * PHI_INV)

    holographic_reconstruction_kernel is also checked for the property that
    matters physically: it consumes a hologram and a reference of the right
    shape and returns finite values in range.

TOLERANCE
    float32 GPU vs float64 NumPy: 2e-5 absolute is generous for tanh/atan2 on
    magnitudes below 10 and tight enough to catch a wrong constant (PHI_INV vs
    PHI moves the result by ~0.1).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

CUDA_DIR = Path(r"C:\abrasax\SYSTEMS\ABRASAX-7D-Crystal\kernels\cuda")
sys.path.insert(0, str(CUDA_DIR))

PHI = 1.618033988749895
PHI_INV = 0.618033988749895
STABILITY_EPSILON = 0.01
TOL = 2e-5


# --------------------------------------------------------------- references ---
def ref_manifold_projection_7d(x: np.ndarray, dim: int) -> np.ndarray:
    x = x.astype(np.float64)
    idx = np.arange(x.size)
    row = idx // dim
    col = idx % dim
    t = x / (1.0 + np.abs(x) + PHI_INV)
    t = t + np.where(col == (row % dim), STABILITY_EPSILON, 0.0)
    return np.tanh(t * PHI_INV)


def ref_holographic_interference(ref: np.ndarray, obj: np.ndarray) -> np.ndarray:
    ref = ref.astype(np.float64)
    obj = obj.astype(np.float64)
    ref_phase = np.arctan2(np.sin(ref * np.pi), np.cos(ref * np.pi))
    obj_phase = np.arctan2(np.sin(obj * np.pi), np.cos(obj * np.pi))
    interference = np.cos(ref_phase - obj_phase) * PHI_INV
    return 0.5 + 0.5 * np.tanh(interference)


def _max_err(got: np.ndarray, want: np.ndarray) -> float:
    got = np.asarray(got, dtype=np.float64).ravel()
    want = np.asarray(want, dtype=np.float64).ravel()
    if got.size != want.size:
        raise AssertionError(f"size mismatch: gpu={got.size} cpu={want.size}")
    return float(np.max(np.abs(got - want)))


def main() -> int:
    from cuda_kernel_loader import CrystalKernelRuntime

    rt = CrystalKernelRuntime()
    print(f"backend: {type(rt).__name__}, kernels bound: {len(rt.available_kernels)}")
    failures = []

    # ---- manifold projection, several shapes ----
    for n, dim in ((256, 8), (1024, 16), (7 * 64, 7), (4096, 64)):
        rng = np.random.default_rng(1234 + n)
        x = (rng.standard_normal(n).astype(np.float32)) * 3.0
        gpu = rt.manifold_projection_7d(x, dim)
        cpu = ref_manifold_projection_7d(x, dim)
        err = _max_err(gpu, cpu)
        ok = err <= TOL
        print(f"  {'OK  ' if ok else 'FAIL'} manifold_projection_7d n={n:<5} dim={dim:<3} max_err={err:.3e}")
        if not ok:
            failures.append(("manifold_projection_7d", n, dim, err))

    # ---- holographic interference ----
    for n in (256, 1024, 4096):
        rng = np.random.default_rng(77 + n)
        ref = (rng.standard_normal(n).astype(np.float32)) * 2.0
        obj = (rng.standard_normal(n).astype(np.float32)) * 2.0
        gpu = rt.holographic_interference(ref, obj)
        cpu = ref_holographic_interference(ref, obj)
        err = _max_err(gpu, cpu)
        ok = err <= TOL
        print(f"  {'OK  ' if ok else 'FAIL'} holographic_interference  n={n:<5}            max_err={err:.3e}")
        if not ok:
            failures.append(("holographic_interference", n, err))

    # ---- reconstruction: must be finite and shaped like its input ----
    n = 1024
    rng = np.random.default_rng(9)
    holo = (rng.random(n).astype(np.float32))
    ref = (rng.standard_normal(n).astype(np.float32))
    rec = rt.holographic_reconstruction(holo, ref)
    ok = rec.size == n and np.all(np.isfinite(rec))
    print(f"  {'OK  ' if ok else 'FAIL'} holographic_reconstruction n={n:<5} finite={np.all(np.isfinite(rec))} "
          f"range=[{rec.min():.3f},{rec.max():.3f}]")
    if not ok:
        failures.append(("holographic_reconstruction", n, float("nan")))

    # ---- the constants actually matter: a wrong PHI_INV must be visible ----
    x = np.linspace(-4, 4, 512).astype(np.float32)
    gpu = rt.manifold_projection_7d(x, 8)
    wrong = np.tanh((x / (1 + np.abs(x) + PHI)) * PHI_INV)
    gap = _max_err(gpu, wrong)
    ok = gap > 1e-3
    print(f"  {'OK  ' if ok else 'FAIL'} sensitivity: PHI vs PHI_INV differs by {gap:.4f} "
          f"(test can tell them apart)")
    if not ok:
        failures.append(("sensitivity", 0, 0, gap))

    print(f"\n{len(failures)} failed")
    for f in failures:
        print("  ", f)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
