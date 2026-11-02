"""Correct my own bad patch: forward first_array_shape from launch_kernel.

MEASURED 2026-11-01 (tools/_trace_grid.py)
    [trace] name=quantum_evolution_2d grid_in=None block_in=(16,16,1)
             n=512 shape=None -> grid=(2,2,1)
    shape arrived as None for an 8x64 field, so _get_grid_block fell back to
    side=int(512**0.5)=22 and launched 2x2 blocks = covers x<32 of 64 columns.
    Needed grid is x=4, y=1.

WHY
    tools/_fix_2d_grid.py inserted `first_array_shape=first_array_shape,` after
    the first occurrence of `first_array_n=n,`, which is inside _launch_1d --
    a different method that has no first_array_shape parameter and passes
    first_array_n straight to the kernel. The real call site is
    line 839:  grid, block = self._get_grid_block(name, grid, block, n=n_elems)

FIX
    1. remove the stray line from _launch_1d (it is dead there and would raise
       NameError the moment _launch_1d was called)
    2. pass shape=first_array_shape at the real _get_grid_block call
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

TARGET = Path(r"C:\abrasax\SYSTEMS\ABRASAX-7D-Crystal\kernels\cuda\cuda_kernel_loader.py")

BAD = """			self.launch_kernel(
				name,
				(d_input, d_output, *extra_args),
				first_array_n=n,
			first_array_shape=first_array_shape,
			)"""
GOOD = """			self.launch_kernel(
				name,
				(d_input, d_output, *extra_args),
				first_array_n=n,
			)"""

OLD_CALL = "		grid, block = self._get_grid_block(name, grid, block, n=n_elems)"
NEW_CALL = ("		grid, block = self._get_grid_block(\n"
            "			name, grid, block, n=n_elems, shape=first_array_shape\n"
            "		)")


def main() -> int:
    src = io.open(TARGET, encoding="utf-8").read()
    changed = []
    if BAD in src:
        src = src.replace(BAD, GOOD, 1)
        changed.append("removed stray forward in _launch_1d")
    if OLD_CALL in src:
        src = src.replace(OLD_CALL, NEW_CALL, 1)
        changed.append("forwarded shape at the real _get_grid_block call")
    if not changed:
        print("nothing to change (already correct?)")
        return 0
    io.open(TARGET, "w", encoding="utf-8", newline="").write(src)
    print("; ".join(changed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
