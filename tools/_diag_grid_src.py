"""Show what source actually reaches the 8x64 quantum-evolution launch.

Prints CrystalKernelRuntime.quantum_evolution_2d, whether it passes
first_array_shape through, the launch call, and _get_grid_block.

Run from anywhere: the kernel-loader dir is resolved from __file__, not cwd.
"""
import inspect
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kernels", "cuda"))

import cuda_kernel_loader as m

print("module:", m.__file__)
src = inspect.getsource(m.CrystalKernelRuntime.quantum_evolution_2d)
print("passes first_array_shape:", "first_array_shape" in src)
print("--- launch call ---")
i = src.find("launch_kernel")
print(src[max(0, i - 200): i + 400])
print("--- _get_grid_block: does it return early? ---")
g = inspect.getsource(m.CrystalKernelRuntime._get_grid_block)
print(g[:900])
