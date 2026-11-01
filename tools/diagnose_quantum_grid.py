"""Confirm the 8x64 quantum-evolution mismatch is a GRID BUG, not a bad reference.

MEASURED 2026-11-01 by tools/verify_kernel_math_quantum.py:
    quantum_evolution_2d 16x16  max_err 8.4e-08   OK
    quantum_evolution_2d 32x32  max_err 7.9e-08   OK
    quantum_evolution_2d  8x64  max_err 8.5e-01   FAIL

The kernel is correct; the launch dimensions are not. The kernel indexes
    x = blockIdx.x * blockDim.x + threadIdx.x
    y = blockIdx.y * blockDim.y + threadIdx.y
with block = (BLOCK_SIZE_2D_X, BLOCK_SIZE_2D_Y) = (16, 16), and the loader
launches with NO grid argument, so one block is dispatched. That covers
x < 16 and y < 16 only.

  16x16  -> every element covered            -> exact (8e-08)
  32x32  -> x,y < 16 only, but the default was
            evidently large enough, or the
            uncovered region was untouched     -> exact (7.9e-08)
  8x64   -> height 8 fits, width 64 does NOT  -> columns 16..63 keep
            whatever the output buffer held   -> 0.85 error

This tool proves it by comparing element-by-element and showing WHERE the
error lives: if the mismatch is confined to x >= 16 and y >= 16, the untested
region is unbounded, and the fix is ceil-division grids.

It also reports the correct expected value for the uncovered region so a future
fix can be checked the same way.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

CUDA_DIR = Path(r"C:\abrasax\SYSTEMS\ABRASAX-7D-Crystal\kernels\cuda")
sys.path.insert(0, str(CUDA_DIR))

sys.path.insert(0, str(CUDA_DIR.parents[2] / "tools"))
from verify_kernel_math_quantum import ref_quantum_evolution  # noqa: E402

BLOCK = (16, 16)


def main() -> int:
    from cuda_kernel_loader import CrystalKernelRuntime

    rt = CrystalKernelRuntime()
    rng = np.random.default_rng(1)
    field = (rng.standard_normal((8, 64)) * 0.5).astype(np.float32)
    cpu = ref_quantum_evolution(field, 0.3, 0.1, toroidal=True)
    gpu = rt.quantum_evolution_2d(field.copy(), phase=0.3, dt=0.1)
    err = np.abs(np.asarray(gpu, np.float64) - cpu)

    cov_x = min(BLOCK[0], field.shape[1])
    cov_y = min(BLOCK[1], field.shape[0])
    covered = err[:cov_y, :cov_x]
    uncovered = err[:, cov_x:]

    print(f"shape={field.shape}  block={BLOCK} -> covers x<{cov_x}, y<{cov_y}")
    print(f"covered   region max_err = {covered.max():.3e}")
    print(f"uncovered region max_err = {uncovered.max():.3e}  (columns {cov_x}..{field.shape[1]-1})")
    print(f"uncovered region all-zero in output: {np.allclose(np.asarray(gpu)[:, cov_x:], 0.0)}")

    ok = covered.max() < 2e-5 and uncovered.max() > 1e-3
    print(f"\n{'CONFIRMED' if ok else 'NOT CONFIRMED'}: error is confined to the uncovered grid region")

    # what a correct launch would produce for the uncovered columns
    print(f"expected value at [0,{cov_x}] = {cpu[0, cov_x]:.6f}, gpu gave {np.asarray(gpu)[0, cov_x]:.6f}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
