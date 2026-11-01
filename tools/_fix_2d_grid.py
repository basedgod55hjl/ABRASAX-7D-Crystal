"""Fix the 2D grid inference: a flat element count cannot describe a shape.

MEASURED 2026-11-01 (tools/diagnose_quantum_grid.py)
    quantum_evolution_2d on an 8x64 field, block (16,16):
        covered   region (x<22) max_err 6.8e-08   correct
        uncovered region (x>=22) max_err 8.3e-01  wrong
    16x16 and 32x32 were exact, which is why this was never caught.

ROOT CAUSE  cuda_kernel_loader.py:767-772
        if n is not None:
            side = int(n ** 0.5)          # 8*64=512 -> side 22
        used_grid = (ceil(side/16), ceil(side/16), 1)
    A square side derived from n cannot describe a non-square field. For 8x64
    it launches ceil(22/16)=2 blocks on BOTH axes, so the kernel writes y<32
    (harmless, guarded) and x<32 (guarded against width 64), leaving columns
    32..63 never touched. Anything non-square with width > height * ... is
    silently wrong, and the error is invisible to a launch test.

FIX
    Two changes, both narrow:
      1. _get_grid_block accepts an optional `shape` hint and, when given a
         2D shape, derives the grid from (width, height) with ceil division
         per axis. Falls back to the old sqrt heuristic only when no shape is
         known, which keeps every existing caller behaving as before.
      2. launch_kernel gains `first_array_shape` and threads it through, and
         quantum_evolution_2d passes field.shape. This is the kernel that was
         measurably wrong; other 2D callers are left alone so the change
         cannot silently alter behaviour nobody has verified yet (flower_of_life
         and metatron_cube also take a size, not a shape).
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

TARGET = Path(r"C:\abrasax\SYSTEMS\ABRASAX-7D-Crystal\kernels\cuda\cuda_kernel_loader.py")


def main() -> int:
    src = io.open(TARGET, encoding="utf-8").read()
    if "first_array_shape" in src:
        print("already patched")
        return 0

    # ---- 1. _get_grid_block gains a shape hint ----
    old_sig = """	def _get_grid_block(
		self,
		name: str,
		grid: tuple[int, int, int] | None,
		block: tuple[int, int, int] | None,
		n: int | None = None,
	) -> tuple[tuple[int, int, int], tuple[int, int, int]]:"""
    new_sig = """	def _get_grid_block(
		self,
		name: str,
		grid: tuple[int, int, int] | None,
		block: tuple[int, int, int] | None,
		n: int | None = None,
		shape: tuple[int, ...] | None = None,
	) -> tuple[tuple[int, int, int], tuple[int, int, int]]:"""
    if old_sig not in src:
        print("MISSING ANCHOR: _get_grid_block signature")
        return 2
    src = src.replace(old_sig, new_sig, 1)

    old_infer = """		if kinfo.dim == 2:
			# 2D kernel -- infer from typical sizes
			if n is not None:
				side = int(n ** 0.5)
			else:
				side = 64
			used_grid = ((side + BLOCK_SIZE_2D_X - 1) // BLOCK_SIZE_2D_X,
						 (side + BLOCK_SIZE_2D_Y - 1) // BLOCK_SIZE_2D_Y, 1)
			used_block = (BLOCK_SIZE_2D_X, BLOCK_SIZE_2D_Y, 1)"""
    new_infer = """		if kinfo.dim == 2:
			# 2D kernel. A flat count cannot describe a shape: measured
			# 2026-11-01, an 8x64 field with n=512 gave side=22, so columns
			# 32..63 were never written (max_err 0.83) while 16x16 and 32x32
			# were exact. Prefer the real shape whenever the caller knows it.
			if shape is not None and len(shape) >= 2:
				height, width = int(shape[-2]), int(shape[-1])
			else:
				if n is not None:
					side = int(n ** 0.5)
				else:
					side = 64
				height = width = side
			used_grid = ((width + BLOCK_SIZE_2D_X - 1) // BLOCK_SIZE_2D_X,
						 (height + BLOCK_SIZE_2D_Y - 1) // BLOCK_SIZE_2D_Y, 1)
			used_block = (BLOCK_SIZE_2D_X, BLOCK_SIZE_2D_Y, 1)"""
    if old_infer not in src:
        print("MISSING ANCHOR: 2D inference block")
        return 2
    src = src.replace(old_infer, new_infer, 1)

    # ---- 2. launch_kernel gains first_array_shape ----
    old_kw = """		first_array_n: int | None = None,
	) -> None:"""
    new_kw = """		first_array_n: int | None = None,
		first_array_shape: tuple[int, ...] | None = None,
	) -> None:"""
    if old_kw not in src:
        print("MISSING ANCHOR: launch_kernel signature tail")
        return 2
    src = src.replace(old_kw, new_kw, 1)

    # thread it into the _get_grid_block call
    old_call = """			first_array_n=n,"""
    new_call = """			first_array_n=n,
			first_array_shape=first_array_shape,"""
    if old_call not in src:
        print("MISSING ANCHOR: _get_grid_block call site")
        return 2
    src = src.replace(old_call, new_call, 1)

    # ---- 3. quantum_evolution_2d passes the shape ----
    old_q = """			self.launch_kernel(
				"quantum_evolution_2d",
				(d_in, d_out, np.int32(w), np.int32(h),
				 np.float32(phase), np.float32(dt)),
				block=(BLOCK_SIZE_2D_X, BLOCK_SIZE_2D_Y, 1),
			)"""
    new_q = """			self.launch_kernel(
				"quantum_evolution_2d",
				(d_in, d_out, np.int32(w), np.int32(h),
				 np.float32(phase), np.float32(dt)),
				block=(BLOCK_SIZE_2D_X, BLOCK_SIZE_2D_Y, 1),
				first_array_shape=(h, w),
			)"""
    if old_q not in src:
        print("MISSING ANCHOR: quantum_evolution_2d launch")
        return 2
    src = src.replace(old_q, new_q, 1)

    io.open(TARGET, "w", encoding="utf-8", newline="").write(src)
    print("patched", TARGET)
    return 0


if __name__ == "__main__":
    sys.exit(main())
