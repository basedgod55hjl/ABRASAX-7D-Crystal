"""Verify the "Manifold-Constrained" half of mH-Q: mhq7d_attention.

Companion to verify_kernel_math.py (manifold + holographic) and the quantum
section here. Kernel source, crystal_kernels_v1.cu:

    h = blockIdx.x; i = threadIdx.x; if (i >= dim) return;
    base = Q[h*dim+i] * K[h*dim+i];
    acc = 0;
    for (d = 0; d < 7; d++) acc += base * (d + 1);   # 1+2+..+7 = 28
    acc = tanhf(acc);                                 # hyperbolic constraint
    O[h*dim+i] = acc * V[h*dim+i];

Two things a launch test cannot see, both checked here:
  * the interaction multiplier is 28, not 7, 21 or 49 -- with tanh saturation a
    wrong sum changes the answer by ~1e-1
  * the constraint is applied to the PRODUCT and then multiplied by V, not to
    the triple product

The loader signature is (Q, K, V, heads, dim) with blockDim.x covering dim.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

CUDA_DIR = Path(r"C:\abrasax\SYSTEMS\ABRASAX-7D-Crystal\kernels\cuda")
sys.path.insert(0, str(CUDA_DIR))

PHI = 1.618033988749895
PHI_INV = 0.618033988749895
TOL = 1e-6


def sacred_sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-(x + np.cos(x * PHI) * PHI_INV) * PHI))


def ref_quantum_evolution(field, phase, dt, toroidal=True):
    f = np.asarray(field, dtype=np.float64)
    h, w = f.shape
    if toroidal:
        left, right = np.roll(f, 1, axis=1), np.roll(f, -1, axis=1)
        up, down = np.roll(f, 1, axis=0), np.roll(f, -1, axis=0)
    else:
        fp = np.pad(f, 1, mode="edge")
        left, right = fp[1:-1, :-2], fp[1:-1, 2:]
        up, down = fp[:-2, 1:-1], fp[2:, 1:-1]
    lap = left + right + up + down - 4.0 * f
    inter = np.sin(f * PHI + phase) * np.cos(f * PHI_INV + phase)
    return sacred_sigmoid(f + dt * (0.1 * lap + inter * 0.1))


def ref_attention(Q, K, V, heads, dim, multiplier_sum=28):
    q = np.asarray(Q, np.float64).ravel()
    k = np.asarray(K, np.float64).ravel()
    v = np.asarray(V, np.float64).ravel()
    out = np.zeros(heads * dim)
    for h in range(heads):
        for i in range(dim):
            idx = h * dim + i
            base = q[idx] * k[idx]
            out[idx] = np.tanh(base * multiplier_sum) * v[idx]
    return out


def main() -> int:
    from cuda_kernel_loader import CrystalKernelRuntime

    rt = CrystalKernelRuntime()
    fails = []
    print(f"kernels bound: {len(rt.available_kernels)}")

    # ---------------- quantum evolution (toroidal) ----------------
    rng = np.random.default_rng(4242)
    for (h, w) in ((16, 16), (32, 32), (8, 64), (64, 8), (5, 100)):
        field = (rng.standard_normal((h, w)) * 0.5).astype(np.float32)
        gpu = rt.quantum_evolution_2d(field.copy(), phase=0.3, dt=0.1)
        cpu = ref_quantum_evolution(field, 0.3, 0.1, True)
        e = float(np.max(np.abs(np.asarray(gpu, np.float64) - cpu)))
        ok = e <= 2e-5
        print(f"  {'OK  ' if ok else 'FAIL'} quantum_evolution_2d {h}x{w:<4} max_err={e:.3e}")
        if not ok:
            fails.append(("quantum_evolution_2d", h, w, e))

    field = (rng.standard_normal((16, 16)) * 0.5).astype(np.float32)
    gpu = rt.quantum_evolution_2d(field.copy(), phase=0.3, dt=0.1)
    gap = float(np.max(np.abs(np.asarray(gpu, np.float64)
                              - ref_quantum_evolution(field, 0.3, 0.1, False))))
    ok = gap > 1e-6
    print(f"  {'OK  ' if ok else 'FAIL'} boundaries are toroidal (clamped differs {gap:.3e})")
    if not ok:
        fails.append(("toroidal", gap))

    # ---------------- mhq7d attention ----------------
    for heads, dim in ((1, 8), (2, 16), (4, 64)):
        n = heads * dim
        Q = (rng.standard_normal(n) * 0.2).astype(np.float32)
        K = (rng.standard_normal(n) * 0.2).astype(np.float32)
        V = (rng.standard_normal(n) * 0.5).astype(np.float32)
        gpu = rt.mhq7d_attention(Q, K, V, heads, dim)
        cpu = ref_attention(Q, K, V, heads, dim)
        e = float(np.max(np.abs(np.asarray(gpu, np.float64) - cpu)))
        ok = e <= TOL
        print(f"  {'OK  ' if ok else 'FAIL'} mhq7d_attention heads={heads} dim={dim:<3} "
              f"max_err={e:.3e}")
        if not ok:
            fails.append(("mhq7d_attention", heads, dim, e))

    # the 7D multiplier must be 28
    n = 16
    Q = (rng.standard_normal(n) * 0.3).astype(np.float32)
    K = (rng.standard_normal(n) * 0.3).astype(np.float32)
    V = (rng.standard_normal(n) * 0.5).astype(np.float32)
    gpu = rt.mhq7d_attention(Q, K, V, 1, n)
    for wrong in (7, 21, 49):
        alt = ref_attention(Q, K, V, 1, n, multiplier_sum=wrong)
        gap = float(np.max(np.abs(np.asarray(gpu, np.float64) - alt)))
        ok = gap > 1e-4
        print(f"  {'OK  ' if ok else 'FAIL'} distinguishes multiplier {wrong} vs 28 "
              f"(differs {gap:.3e})")
        if not ok:
            fails.append(("multiplier", wrong, gap))

    print(f"\n{len(fails)} failed")
    for f in fails:
        print("  ", f)
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
