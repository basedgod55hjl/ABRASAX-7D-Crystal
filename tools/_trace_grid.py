"""Show the grid/block actually used for the 8x64 quantum-evolution launch.

The patch threads first_array_shape -> _get_grid_block, but the 8x64 case still
mismatches, so something earlier is supplying a grid. This prints what arrives.

Run from anywhere: the kernel-loader dir is resolved from __file__, not cwd.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kernels", "cuda"))

import cuda_kernel_loader as m

orig = m.CrystalKernelRuntime._get_grid_block


def traced(self, name, grid, block, n=None, shape=None):
    out = orig(self, name, grid, block, n=n, shape=shape)
    if "quantum" in name:
        print(f"  [trace] name={name} grid_in={grid} block_in={block} "
              f"n={n} shape={shape} -> grid={out[0]} block={out[1]}")
    return out


m.CrystalKernelRuntime._get_grid_block = traced

rt = m.CrystalKernelRuntime()
rng = np.random.default_rng(1)
field = (rng.standard_normal((8, 64)) * 0.5).astype(np.float32)
print(f"field shape={field.shape} n={field.size}")
out = rt.quantum_evolution_2d(field.copy(), phase=0.3, dt=0.1)
print("output shape:", out.shape)

# what a correct launch covers
print(f"needed grid: x={-(-64 // 16)}, y={-(-8 // 16)}")
