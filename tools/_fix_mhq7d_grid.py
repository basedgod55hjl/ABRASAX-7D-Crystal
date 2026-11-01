"""mhq7d_attention launches ONE block, so only head 0 is ever computed.

MEASURED 2026-11-01 (tools/verify_kernel_math_quantum.py)
    mhq7d_attention heads=1 dim=8    max_err 3.4e-08   OK
    mhq7d_attention heads=2 dim=16   max_err 5.98e-01  FAIL
    mhq7d_attention heads=4 dim=64   max_err 1.17e+00  FAIL

KERNEL (crystal_kernels_v1.cu)
    int h = blockIdx.x;                 <-- the HEAD is the block
    int i = threadIdx.x;
    if (i >= dim) return;
    O[h*dim + i] = tanhf(Q[h*dim+i]*K[h*dim+i]*28) * V[h*dim+i]

LOADER (cuda_kernel_loader.py:1291)
    self.launch_kernel("mhq7d_attention", (..., heads, dim), first_array_n=n)
No grid is given, so _get_grid_block falls through to the 1D branch with
n = heads*dim and produces grid = ceil(n/256) = 1 for any n <= 256, and
blockIdx.x is always 0. Heads 1..H-1 are left as whatever d_o held.

    heads=1 -> the only head is head 0 -> correct
    heads>1 -> silent garbage in every other head

FIX
    Register the kernel as 2D and launch grid=(heads,1,1), block=(dim,1,1),
    mirroring the kernel's own indexing exactly. That is the only grid under
    which `h = blockIdx.x; i = threadIdx.x` can reach every element. Passing
    grid and block explicitly also short-circuits _get_grid_block, so the
    inference path cannot quietly override it.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

TARGET = Path(r"C:\abrasax\SYSTEMS\ABRASAX-7D-Crystal\kernels\cuda\cuda_kernel_loader.py")

OLD = """			self.launch_kernel(
				"mhq7d_attention",
				(d_q, d_k, d_v, d_o, np.int32(heads), np.int32(dim)),
				first_array_n=n,
			)"""

NEW = """			# h = blockIdx.x, i = threadIdx.x in the kernel, so the launch
			# must be one BLOCK per head and one THREAD per dim. With no grid
			# the loader defaults to (1,1,1) and only head 0 is ever computed
			# (measured 2026-11-01: heads=1 exact, heads=2 err 0.60,
			# heads=4 err 1.17).
			self.launch_kernel(
				"mhq7d_attention",
				(d_q, d_k, d_v, d_o, np.int32(heads), np.int32(dim)),
				grid=(heads, 1, 1),
				block=(dim, 1, 1),
			)"""


def main() -> int:
    src = io.open(TARGET, encoding="utf-8").read()
    if "one BLOCK per head" in src:
        print("already patched")
        return 0
    if OLD not in src:
        print("MISSING ANCHOR: mhq7d_attention launch")
        return 2
    io.open(TARGET, "w", encoding="utf-8", newline="").write(src.replace(OLD, NEW, 1))
    print("patched mhq7d_attention grid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
