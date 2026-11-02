# ============================================================================
# ABRASAX — Unified Xenoliving System
# PRIMAL_HEX: 4f5349524953424c58434b | phi: 1.618033988749895
# CREATOR: Sir Charles Spikes (BASEDGOD)
# ============================================================================

"""
crystal_kernels.py -- Python Bindings for 7D mH-Q Crystal CUDA Kernels
======================================================================

Provides Python callable wrappers for all 19 compiled CUDA kernels across
three translation units:

    crystal_kernels_v2.ptx   (13 kernels -- crystal.7d)
    crystal_kernels_v1.ptx   ( 5 kernels -- crystal_cortex.7d)
    physx_crystal_dynamics.ptx ( 1 kernel  -- physx_crystal_dynamics.7d)

Requires: pycuda (pip install pycuda)
Target:   GTX 1660 Ti, sm_75

Usage:
    import crystal_kernels as ck
    ctx = ck.CrystalKernelContext()
    out = ctx.manifold_projection_7d(input_tensor, dim=64)
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("crystal_kernels")

# ---------------------------------------------------------------------------
#  Attempt to import the CUDA driver
# ---------------------------------------------------------------------------
try:
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401 -- triggers context creation
    HAVE_PYCUDA = True
except ImportError:
    HAVE_PYCUDA = False
    cuda = None  # type: ignore[assignment]

try:
    import pycuda.gpuarray as gpuarray
    HAVE_GPUARRAY = True
except ImportError:
    HAVE_GPUARRAY = False
    gpuarray = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
#  Paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_BLD = _HERE / "build"

_PTX_FILES: dict[str, Path] = {
    "v2":   _BLD / "crystal_kernels_v2.ptx",
    "v1":   _BLD / "crystal_kernels_v1.ptx",
    "physx": _BLD / "physx_crystal_dynamics.ptx",
}


def ptx_path(name: str) -> Path:
    p = _PTX_FILES.get(name)
    if p is None:
        raise ValueError(f"Unknown PTX module: {name!r}. Options: {list(_PTX_FILES)}")
    return p


# ---------------------------------------------------------------------------
#  Launch configuration helpers
# ---------------------------------------------------------------------------
# Constants matching the CUDA #defines
BLOCK_SIZE_1D = 256
BLOCK_SIZE_2D_X = 16
BLOCK_SIZE_2D_Y = 16


def grid_1d(n: int, block_size: int = BLOCK_SIZE_1D) -> tuple[int, int, int]:
    """Return (grid_x, 1, 1) for a 1D kernel."""
    return ((n + block_size - 1) // block_size, 1, 1)


def block_1d(block_size: int = BLOCK_SIZE_1D) -> tuple[int, int, int]:
    return (block_size, 1, 1)


def grid_2d(w: int, h: int,
            bx: int = BLOCK_SIZE_2D_X,
            by: int = BLOCK_SIZE_2D_Y) -> tuple[int, int, int]:
    return ((w + bx - 1) // bx, (h + by - 1) // by, 1)


def block_2d(bx: int = BLOCK_SIZE_2D_X, by: int = BLOCK_SIZE_2D_Y) -> tuple[int, int, int]:
    return (bx, by, 1)


# ---------------------------------------------------------------------------
#  Kernel descriptor: binds a CUDA function name to its PTX module and
#  grid/block strategy.
# ---------------------------------------------------------------------------
class KernelDesc:
    """Describes a single kernel: which PTX module, function name, dimensionality."""

    __slots__ = ("module", "func_name", "dim")

    def __init__(self, module: str, func_name: str, dim: int = 1) -> None:
        self.module = module
        self.func_name = func_name
        self.dim = dim

    def bind(self, mod: cuda.Module) -> cuda.Function:
        return mod.get_function(self.func_name)


# ---------------------------------------------------------------------------
#  Registry of all 19 kernels
# ---------------------------------------------------------------------------
KERNEL_REGISTRY: dict[str, KernelDesc] = {
    # -- v2: crystal_kernels_v2.ptx (13 kernels) --
    "manifold_projection_7d":        KernelDesc("v2", "manifold_projection_7d_kernel"),
    "manifold_projection_batch":     KernelDesc("v2", "manifold_projection_batch_kernel"),
    "phi_harmonic_modulation":       KernelDesc("v2", "phi_harmonic_modulation_kernel"),
    "fibonacci_spiral_embedding":    KernelDesc("v2", "fibonacci_spiral_embedding_kernel"),
    "cbm_flux_unfold_v2":            KernelDesc("v2", "cbm_flux_unfold_v2_kernel"),
    "cbm_flux_fold":                 KernelDesc("v2", "cbm_flux_fold_kernel"),
    "holographic_interference":      KernelDesc("v2", "holographic_interference_kernel"),
    "holographic_reconstruction":    KernelDesc("v2", "holographic_reconstruction_kernel"),
    "quantum_evolution_2d":          KernelDesc("v2", "quantum_evolution_2d_kernel", dim=2),
    "crystal_resonance":             KernelDesc("v2", "crystal_resonance_kernel"),
    "crystal_entropy_mining":        KernelDesc("v2", "crystal_entropy_mining_kernel"),
    "flower_of_life":                KernelDesc("v2", "flower_of_life_kernel", dim=2),
    "metatron_cube":                 KernelDesc("v2", "metatron_cube_kernel", dim=2),

    # -- v1: crystal_kernels_v1.ptx (5 kernels) --
    "mhq7d_attention":               KernelDesc("v1", "mhq7d_attention"),
    "mhq7d_persistent":              KernelDesc("v1", "mhq7d_persistent"),
    "manifold_projection":           KernelDesc("v1", "manifold_projection_kernel"),
    "phi_modulation":                KernelDesc("v1", "phi_modulation_kernel"),
    "cbm_flux_unfold":               KernelDesc("v1", "cbm_flux_unfold_kernel"),

    # -- physx (1 kernel) --
    "physx_crystal_dynamics":        KernelDesc("physx", "physx_crystal_dynamics_kernel"),
}


# ---------------------------------------------------------------------------
#  CrystalKernelContext -- main entry point
# ---------------------------------------------------------------------------
class CrystalKernelContext:
    """Loads all PTX modules and exposes each kernel as a callable method.

    Typical usage:
        >>> ctx = CrystalKernelContext()
        >>> inp = np.random.randn(1024).astype(np.float32)
        >>> out = ctx.manifold_projection_7d(inp, dim=64)
    """

    def __init__(self, ptx_dir: str | Path | None = None) -> None:
        if not HAVE_PYCUDA:
            raise RuntimeError(
                "pycuda is required. Install: pip install pycuda"
            )

        if ptx_dir is not None:
            global _PTX_FILES
            p = Path(ptx_dir)
            _PTX_FILES["v2"] = p / "crystal_kernels_v2.ptx"
            _PTX_FILES["v1"] = p / "crystal_kernels_v1.ptx"
            _PTX_FILES["physx"] = p / "physx_crystal_dynamics.ptx"

        self._check_ptx_files()
        self._modules: dict[str, cuda.Module] = {}
        self._load_modules()
        self._bind_kernels()

    # -- public helpers ---------------------------------------------------

    @staticmethod
    def device_info() -> dict[str, Any]:
        """Return dict of device properties for the active GPU."""
        if not HAVE_PYCUDA:
            return {"error": "pycuda not available"}
        dev = cuda.Device(0)
        attrs = dev.get_attributes()
        info: dict[str, Any] = {
            "name": dev.name(),
            "compute_capability": f"{dev.compute_capability()[0]}.{dev.compute_capability()[1]}",
            "total_memory_mb": dev.total_memory() // (1024 * 1024),
            "max_threads_per_block": attrs.get(cuda.device_attribute.MAX_THREADS_PER_BLOCK, 0),
            "multiprocessor_count": attrs.get(cuda.device_attribute.MULTIPROCESSOR_COUNT, 0),
            "shared_memory_per_block_kb": attrs.get(cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK, 0) // 1024,
        }
        return info

    @staticmethod
    def alloc(size: int, dtype: type = np.float32) -> np.ndarray:
        """Allocate a device-side array of *size* elements."""
        return cuda.mem_alloc(size * np.dtype(dtype).itemsize)

    @staticmethod
    def to_device(arr: np.ndarray) -> cuda.DeviceAllocation:
        """Upload a numpy array to the device."""
        return cuda.to_device(arr)

    @staticmethod
    def from_device(dev_ptr: cuda.DeviceAllocation, shape: tuple[int, ...],
                    dtype: type = np.float32) -> np.ndarray:
        """Download device memory back to a numpy array."""
        out = np.empty(shape, dtype=dtype)
        cuda.memcpy_dtoh(out, dev_ptr)
        return out

    @staticmethod
    def stream() -> cuda.Stream:
        return cuda.Stream()

    # -- private ----------------------------------------------------------

    def _check_ptx_files(self) -> None:
        missing = [str(p) for p in _PTX_FILES.values() if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing compiled PTX files. Run compile_kernels.bat first.\n"
                + "\n".join(f"  MISSING: {m}" for m in missing)
            )

    def _load_modules(self) -> None:
        for name, path in _PTX_FILES.items():
            logger.info("Loading PTX module: %s -> %s", name, path)
            with open(str(path), "rb") as f:
                ptx_data = f.read()
            self._modules[name] = cuda.module_from_buffer(ptx_data)

    def _bind_kernels(self) -> None:
        for key, desc in KERNEL_REGISTRY.items():
            mod = self._modules.get(desc.module)
            if mod is None:
                raise RuntimeError(f"Module {desc.module!r} not loaded for kernel {key!r}")
            func = mod.get_function(desc.func_name)
            setattr(self, key, _KernelLauncher(self, key, desc, func))

    @property
    def available_kernels(self) -> list[str]:
        return list(KERNEL_REGISTRY.keys())


# ---------------------------------------------------------------------------
#  _KernelLauncher -- wraps a cuda.Function with convenient calling
# ---------------------------------------------------------------------------
class _KernelLauncher:
    """Thin wrapper around a cuda.Function with grid/block inference."""

    def __init__(self, ctx: CrystalKernelContext, name: str,
                 desc: KernelDesc, func: cuda.Function) -> None:
        self._ctx = ctx
        self._name = name
        self._desc = desc
        self._func = func

    @property
    def name(self) -> str:
        return self._name

    @property
    def function(self) -> cuda.Function:
        return self._func

    def __call__(self, *args: Any,
                 grid: tuple[int, int, int] | None = None,
                 block: tuple[int, int, int] | None = None,
                 stream: cuda.Stream | None = None,
                 shared_mem: int = 0,
                 texrefs: list[Any] | None = None) -> None:
        """Launch the kernel.

        Parameters
        ----------
        *args : arrays or scalars
            Kernel arguments in order. numpy arrays are automatically
            converted to device pointers.
        grid, block : tuple of 3 ints or None
            Launch configuration. If None, uses heuristics based on the
            first array argument's size (1D) or 16x16 blocks (2D).
        stream : cuda.Stream or None
            Optional stream for async execution.
        shared_mem : int
            Dynamic shared memory in bytes.
        """
        if grid is None or block is None:
            grid, block = self._infer_config(args)

        # Prepare args: upload numpy arrays
        prepared = []
        for a in args:
            if isinstance(a, np.ndarray):
                prepared.append(cuda.to_device(a))
            elif isinstance(a, cuda.DeviceAllocation):
                prepared.append(a)
            else:
                prepared.append(a)

        self._func(*prepared,
                   grid=grid,
                   block=block,
                   stream=stream,
                   shared=shared_mem,
                   texrefs=texrefs or [])

    def _infer_config(self, args: tuple[Any, ...]) -> tuple[tuple, tuple]:
        """Guess grid/block from the first array argument."""
        # Find first array argument
        for a in args:
            if isinstance(a, np.ndarray):
                n = a.size
                if self._desc.dim == 2:
                    w = a.shape[1] if a.ndim >= 2 else int(np.sqrt(n))
                    h = a.shape[0] if a.ndim >= 2 else int(np.sqrt(n))
                    block = (BLOCK_SIZE_2D_X, BLOCK_SIZE_2D_Y, 1)
                    grid = ((w + BLOCK_SIZE_2D_X - 1) // BLOCK_SIZE_2D_X,
                            (h + BLOCK_SIZE_2D_Y - 1) // BLOCK_SIZE_2D_Y, 1)
                    return grid, block
                else:
                    block = (BLOCK_SIZE_1D, 1, 1)
                    grid = ((n + BLOCK_SIZE_1D - 1) // BLOCK_SIZE_1D, 1, 1)
                    return grid, block
            if isinstance(a, int):
                # Could be n parameter
                pass
        # Default fallback
        return (1, 1, 1), (BLOCK_SIZE_1D, 1, 1)


# ---------------------------------------------------------------------------
#  Convenience: create a default context on import
# ---------------------------------------------------------------------------
_default_ctx: CrystalKernelContext | None = None


def get_context() -> CrystalKernelContext:
    global _default_ctx
    if _default_ctx is None:
        _default_ctx = CrystalKernelContext()
    return _default_ctx


# ---------------------------------------------------------------------------
#  Direct functional API (uses default context)
# ---------------------------------------------------------------------------
def manifold_projection_7d(input: np.ndarray, dim: int,
                           stream: cuda.Stream | None = None) -> np.ndarray:
    """Project input onto the 7D Poincare ball."""
    ctx = get_context()
    n = input.size
    out = np.empty_like(input)
    d_in = cuda.to_device(input)
    d_out = cuda.to_device(out)
    block = (BLOCK_SIZE_1D, 1, 1)
    grid = ((n + BLOCK_SIZE_1D - 1) // BLOCK_SIZE_1D, 1, 1)
    ctx.manifold_projection_7d(d_in, d_out, np.int32(n), np.int32(dim),
                                grid=grid, block=block, stream=stream)
    cuda.memcpy_dtoh(out, d_out)
    return out


def phi_harmonic_modulation(data: np.ndarray, phase: float = 0.0,
                            harmonic_order: int = 3,
                            stream: cuda.Stream | None = None) -> np.ndarray:
    """Apply PHI-harmonic modulation to data."""
    ctx = get_context()
    out = data.copy()
    n = out.size
    d_data = cuda.to_device(out)
    block = (BLOCK_SIZE_1D, 1, 1)
    grid = ((n + BLOCK_SIZE_1D - 1) // BLOCK_SIZE_1D, 1, 1)
    ctx.phi_harmonic_modulation(d_data, np.int32(n),
                                 np.float32(phase), np.int32(harmonic_order),
                                 grid=grid, block=block, stream=stream)
    cuda.memcpy_dtoh(out, d_data)
    return out


def cbm_flux_unfold(seed: np.ndarray, output_size: int, phi_flux: float = 0.1,
                    stream: cuda.Stream | None = None) -> np.ndarray:
    """Unfold seed DNA into full parameter space."""
    ctx = get_context()
    out = np.empty(output_size, dtype=np.float32)
    d_seed = cuda.to_device(seed)
    d_out = cuda.to_device(out)
    block = (BLOCK_SIZE_1D, 1, 1)
    grid = ((output_size + BLOCK_SIZE_1D - 1) // BLOCK_SIZE_1D, 1, 1)
    ctx.cbm_flux_unfold_v2(d_seed, d_out,
                            np.int32(seed.size), np.int32(output_size),
                            np.float32(phi_flux), np.float32(0.0),
                            grid=grid, block=block, stream=stream)
    cuda.memcpy_dtoh(out, d_out)
    return out


def generate_flower_of_life(width: int, height: int, scale: float = 0.3,
                            stream: cuda.Stream | None = None) -> np.ndarray:
    """Generate a Flower of Life sacred geometry pattern."""
    ctx = get_context()
    out = np.empty((height, width), dtype=np.float32)
    d_out = cuda.to_device(out)
    block = (BLOCK_SIZE_2D_X, BLOCK_SIZE_2D_Y, 1)
    grid = ((width + BLOCK_SIZE_2D_X - 1) // BLOCK_SIZE_2D_X,
            (height + BLOCK_SIZE_2D_Y - 1) // BLOCK_SIZE_2D_Y, 1)
    ctx.flower_of_life(d_out, np.int32(width), np.int32(height), np.float32(scale),
                        grid=grid, block=block, stream=stream)
    cuda.memcpy_dtoh(out, d_out)
    return out


def generate_metatron_cube(width: int, height: int, rotation: float = 0.0,
                           stream: cuda.Stream | None = None) -> np.ndarray:
    """Generate Metatron's Cube sacred geometry pattern."""
    ctx = get_context()
    out = np.empty((height, width), dtype=np.float32)
    d_out = cuda.to_device(out)
    block = (BLOCK_SIZE_2D_X, BLOCK_SIZE_2D_Y, 1)
    grid = ((width + BLOCK_SIZE_2D_X - 1) // BLOCK_SIZE_2D_X,
            (height + BLOCK_SIZE_2D_Y - 1) // BLOCK_SIZE_2D_Y, 1)
    ctx.metatron_cube(d_out, np.int32(width), np.int32(height), np.float32(rotation),
                       grid=grid, block=block, stream=stream)
    cuda.memcpy_dtoh(out, d_out)
    return out
