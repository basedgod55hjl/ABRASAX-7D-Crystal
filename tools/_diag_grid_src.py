import inspect
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
