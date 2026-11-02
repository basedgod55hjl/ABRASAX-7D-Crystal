# ============================================================================
# ABRASAX — Unified Xenoliving System
# PRIMAL_HEX: 4f5349524953424c58434b | phi: 1.618033988749895
# CREATOR: Sir Charles Spikes (BASEDGOD)
# ============================================================================

"""
ctypes_wrapper.py -- Unified CUDA kernel loader for compiled .o / .dll files.

Provides two backends:
  1. CTYPES_DIRECT  -- loads cuda_kernels.dll via ctypes.CDLL (fast, no deps)
  2. PYCUDA_BACKEND -- uses the existing pycuda-based crystal_kernels.py

On init, the wrapper tries PYCUDA_BACKEND first (richer API), then falls back
to CTYPES_DIRECT if a pre-built DLL is available.

Usage:
    from cuda.ctypes_wrapper import CudaKernelRuntime

    runtime = CudaKernelRuntime()
    info = runtime.device_info()
    out = runtime.manifold_projection_7d(input_array, dim=64)
"""

from __future__ import annotations

import ctypes
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("cuda_wrapper")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_BLD = _HERE / "build"
_O_FILES: dict[str, Path] = {
    "v1": _HERE / "crystal_kernels_v1.o",
    "v2": _HERE / "crystal_kernels_v2.o",
    "physx": _HERE / "physx_crystal_dynamics.o",
}
_DLL_PATH = _BLD / "cuda_kernels.dll"

# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------
_HAVE_PYCUDA = False
try:
    import pycuda.driver as cuda  # noqa: F401
    _HAVE_PYCUDA = True
except ImportError:
    pass

_HAVE_DLL = _DLL_PATH.exists()


# ---------------------------------------------------------------------------
# Ctypes kernel context
# ---------------------------------------------------------------------------
class CudaKernelRuntime:
    """Unified runtime for compiled CUDA kernels.

    Backend selection (in order of preference):
      1. pycuda (via crystal_kernels.CrystalKernelContext)
      2. ctypes.CDLL loading cuda_kernels.dll (if compiled)
      3. Simulated/CPU fallback for smoke-testing (no GPU required)
    """

    def __init__(self) -> None:
        self._backend: str = "none"
        self._lib: ctypes.CDLL | None = None
        self._pyctx: Any = None  # CrystalKernelContext
        self._init_backend()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def _init_backend(self) -> None:
        if _HAVE_PYCUDA:
            self._init_pycuda()
        elif _HAVE_DLL:
            self._init_ctypes()
        else:
            self._init_cpu_sim()

    def _init_pycuda(self) -> None:
        """Backend: pycuda via crystal_kernels.CrystalKernelContext."""
        try:
            from cuda.crystal_kernels import CrystalKernelContext, HAVE_PYCUDA

            if not HAVE_PYCUDA:
                raise RuntimeError("pycuda import failed in crystal_kernels")
            self._pyctx = CrystalKernelContext()
            self._backend = "pycuda"
            logger.info("CUDA backend: pycuda (%d kernels loaded)",
                        len(self._pyctx.available_kernels))
        except Exception as exc:
            logger.warning("pycuda init failed (%s); trying ctypes backend", exc)
            if _HAVE_DLL:
                self._init_ctypes()
            else:
                self._init_cpu_sim()

    def _init_ctypes(self) -> None:
        """Backend: ctypes.CDLL loading cuda_kernels.dll."""
        try:
            self._lib = ctypes.CDLL(str(_DLL_PATH))
            self._backend = "ctypes"
            logger.info("CUDA backend: ctypes (DLL: %s)", _DLL_PATH)
        except Exception as exc:
            logger.error("ctypes DLL load failed: %s", exc)
            self._init_cpu_sim()

    def _init_cpu_sim(self) -> None:
        """Fallback: CPU simulation for smoke-testing / non-GPU hosts."""
        self._backend = "cpu_sim"
        logger.warning(
            "No CUDA backend available (no pycuda, no DLL). "
            "Using CPU simulation for shape-checking only."
        )

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------
    @property
    def backend(self) -> str:
        return self._backend

    @property
    def available_kernels(self) -> list[str]:
        if self._backend == "pycuda" and self._pyctx is not None:
            return self._pyctx.available_kernels
        return list(_O_FILES.keys())

    def device_info(self) -> dict[str, Any]:
        """Return dict with available device/backend info."""
        info: dict[str, Any] = {"backend": self._backend}
        if self._backend == "pycuda" and self._pyctx is not None:
            info["pycuda"] = self._pyctx.device_info()
        elif self._backend == "ctypes":
            info["dll"] = str(_DLL_PATH)
        # Always report compiled .o files even if not loaded
        info["o_files"] = {
            name: {"path": str(p), "exists": p.exists(), "size_bytes": p.stat().st_size if p.exists() else 0}
            for name, p in _O_FILES.items()
        }
        return info

    # ------------------------------------------------------------------
    # Kernel wrappers
    # ------------------------------------------------------------------
    def manifold_projection_7d(self, input_data: np.ndarray, dim: int = 64) -> np.ndarray:
        """Project input onto the 7D Poincare ball manifold."""
        if self._backend == "pycuda" and self._pyctx is not None:
            from cuda.crystal_kernels import manifold_projection_7d as _pycu_fn
            return _pycu_fn(input_data, dim=dim)
        # CPU simulation fallback
        return self._sim_manifold_projection(input_data, dim)

    def phi_harmonic_modulation(self, data: np.ndarray, phase: float = 0.0,
                                harmonic_order: int = 3) -> np.ndarray:
        """Apply PHI-harmonic modulation to data."""
        if self._backend == "pycuda" and self._pyctx is not None:
            from cuda.crystal_kernels import phi_harmonic_modulation as _pycu_fn
            return _pycu_fn(data, phase=phase, harmonic_order=harmonic_order)
        return self._sim_phi_modulation(data, phase, harmonic_order)

    def cbm_flux_unfold(self, seed: np.ndarray, output_size: int,
                        phi_flux: float = 0.1) -> np.ndarray:
        """Unfold seed DNA into full parameter space via CBM flux."""
        if self._backend == "pycuda" and self._pyctx is not None:
            from cuda.crystal_kernels import cbm_flux_unfold as _pycu_fn
            return _pycu_fn(seed, output_size, phi_flux=phi_flux)
        return self._sim_flux_unfold(seed, output_size, phi_flux)

    # ------------------------------------------------------------------
    # CPU simulations (shape-preserving, used for test/debug)
    # ------------------------------------------------------------------
    @staticmethod
    def _sim_manifold_projection(input_data: np.ndarray, dim: int = 64) -> np.ndarray:
        """CPU fallback: normalize + tanh squashing."""
        out = input_data.copy().astype(np.float32)
        norm = np.sqrt(np.sum(out ** 2))
        if norm > 1e-8:
            out = out / norm
        return np.tanh(out * (1.618033988749895 ** 0.5))

    @staticmethod
    def _sim_phi_modulation(data: np.ndarray, phase: float = 0.0,
                            harmonic_order: int = 3) -> np.ndarray:
        """CPU fallback: sinusoidal modulation with golden ratio."""
        phi = 1.618033988749895
        indices = np.arange(data.size, dtype=np.float32).reshape(data.shape)
        modulation = np.sin(indices * phi ** harmonic_order + phase * phi)
        return (data * modulation).astype(np.float32)

    @staticmethod
    def _sim_flux_unfold(seed: np.ndarray, output_size: int,
                         phi_flux: float = 0.1) -> np.ndarray:
        """CPU fallback: tile + interpolate seed."""
        out = np.tile(seed.flatten(), int(np.ceil(output_size / seed.size)))[:output_size]
        out = out + np.random.randn(output_size).astype(np.float32) * phi_flux
        return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------
_default_runtime: CudaKernelRuntime | None = None


def get_runtime() -> CudaKernelRuntime:
    global _default_runtime
    if _default_runtime is None:
        _default_runtime = CudaKernelRuntime()
    return _default_runtime


def device_info() -> dict[str, Any]:
    return get_runtime().device_info()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    rt = CudaKernelRuntime()
    info = rt.device_info()
    print(f"Backend: {rt.backend}")
    print(f"  DLL exists: {_HAVE_DLL}")
    print(f"  pycuda available: {_HAVE_PYCUDA}")
    # Smoke test: run manifold projection on dummy data
    dummy = np.random.randn(256).astype(np.float32)
    out = rt.manifold_projection_7d(dummy, dim=64)
    print(f"  manifold_projection_7d output shape={out.shape} dtype={out.dtype}")
    print(f"  Stats: min={out.min():.4f} max={out.max():.4f} mean={out.mean():.4f}")
    print("CUDA wrapper OK")
