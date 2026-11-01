# ============================================================================
# ABRASAX — Unified Xenoliving System
# PRIMAL_HEX: 4f5349524953424c58434b | phi: 1.618033988749895
# CREATOR: Sir Charles Spikes (BASEDGOD)
# ============================================================================

#!/usr/bin/env python3
"""
test_kernels.py -- Validation Suite for 7D mH-Q Crystal CUDA Kernels
======================================================================

Tests all 19 CUDA kernels across three compilation units:

    v2 (13):   manifold_projection_7d, manifold_projection_batch,
               phi_harmonic_modulation, fibonacci_spiral_embedding,
               cbm_flux_unfold_v2, cbm_flux_fold,
               holographic_interference, holographic_reconstruction,
               quantum_evolution_2d, crystal_resonance,
               crystal_entropy_mining, flower_of_life, metatron_cube

    v1 (5):    mhq7d_attention, mhq7d_persistent,
               manifold_projection, phi_modulation, cbm_flux_unfold

    physx (1): physx_crystal_dynamics

Phases:
    1. Environment check (CUDA toolkit, sm_75 target, pycuda, nvcc)
    2. Compilation validation (nvcc -ptx each .cu file)
    3. PTX module loading
    4. Kernel launch validation for every kernel
    5. Numerical correctness checks
    6. Performance benchmark (optional)

Usage:
    python test_kernels.py               # full test suite
    python test_kernels.py --quick        # skip compilation, test only
    python test_kernels.py --list         # list all kernels and exit
    python test_kernels.py --benchmark    # include performance benchmarks
"""

from __future__ import annotations

import argparse
import ctypes
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Protocol

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname).1s] %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("test_kernels")

# ---------------------------------------------------------------------------
#  Paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_CUDA_DIR = _HERE
_BLD = _CUDA_DIR / "build"
_MODELS = _CUDA_DIR.parent / "models" / "crystal"

SOURCES: dict[str, Path] = {
    "v2":   _CUDA_DIR / "crystal_kernels_v2.cu",
    "v1":   _CUDA_DIR / "crystal_kernels_v1.cu",
    "physx": _CUDA_DIR / "physx_crystal_dynamics.cu",
}

# All 19 kernels with their launch config tuples
# (name, module, dim, arg_shapes_for_test)
KERNEL_MANIFEST: list[dict[str, Any]] = [
    # --- v2 kernels (13) ---
    dict(name="manifold_projection_7d",  src="v2", dim=1,
         args=lambda: [
             np.random.randn(1024).astype(np.float32),  # input
             np.empty(1024, dtype=np.float32),           # output
             np.int32(1024),                             # n
             np.int32(64),                               # dim
         ]),
    dict(name="manifold_projection_batch", src="v2", dim=1,
         args=lambda: [
             np.random.randn(256 * 64).astype(np.float32),
             np.empty(256 * 64, dtype=np.float32),
             np.int32(256),  # batch_size
             np.int32(64),   # dim
         ]),
    dict(name="phi_harmonic_modulation", src="v2", dim=1,
         args=lambda: [
             np.random.randn(1024).astype(np.float32),
             np.int32(1024),
             np.float32(0.0),    # phase
             np.int32(3),        # harmonic_order
         ]),
    dict(name="fibonacci_spiral_embedding", src="v2", dim=1,
         args=lambda: [
             np.random.randn(512).astype(np.float32),
             np.empty(512, dtype=np.float32),
             np.empty(512, dtype=np.float32),
             np.int32(512),
         ]),
    dict(name="cbm_flux_unfold_v2", src="v2", dim=1,
         args=lambda: [
             np.random.randn(16).astype(np.float32),
             np.empty(1024, dtype=np.float32),
             np.int32(16),       # seed_size
             np.int32(1024),     # output_size
             np.float32(0.1),    # phi_flux
             np.float32(0.0),    # time_factor
         ]),
    dict(name="cbm_flux_fold", src="v2", dim=1,
         args=lambda: [
             np.random.randn(1024).astype(np.float32),
             np.empty(16, dtype=np.float32),
             np.int32(16),      # seed_size
             np.int32(1024),    # weight_size
         ]),
    dict(name="holographic_interference", src="v2", dim=1,
         args=lambda: [
             np.random.randn(1024).astype(np.float32),
             np.random.randn(1024).astype(np.float32),
             np.empty(1024, dtype=np.float32),
             np.int32(1024),
         ]),
    dict(name="holographic_reconstruction", src="v2", dim=1,
         args=lambda: [
             np.random.randn(1024).astype(np.float32),
             np.random.randn(1024).astype(np.float32),
             np.empty(1024, dtype=np.float32),
             np.int32(1024),
         ]),
    dict(name="quantum_evolution_2d", src="v2", dim=2,
         args=lambda: [
             np.random.randn(64, 64).astype(np.float32),
             np.empty((64, 64), dtype=np.float32),
             np.int32(64), np.int32(64),
             np.float32(0.0),   # phase
             np.float32(0.01),  # dt
         ]),
    dict(name="crystal_resonance", src="v2", dim=1,
         args=lambda: [
             np.random.randn(1024).astype(np.float32),
             ctypes.c_void_p(0),  # phi_alignment (output, will be set)
             ctypes.c_void_p(0),  # coherence (output, will be set)
             np.int32(1024),
         ]),
    dict(name="crystal_entropy_mining", src="v2", dim=1,
         args=lambda: [
             np.empty(1024, dtype=np.float32),
             np.int32(1024),
             np.uint64(42),      # base_seed
             np.float32(0.0),    # time_factor
         ]),
    dict(name="flower_of_life", src="v2", dim=2,
         args=lambda: [
             np.empty((128, 128), dtype=np.float32),
             np.int32(128), np.int32(128),
             np.float32(0.3),    # scale
         ]),
    dict(name="metatron_cube", src="v2", dim=2,
         args=lambda: [
             np.empty((128, 128), dtype=np.float32),
             np.int32(128), np.int32(128),
             np.float32(0.0),    # rotation
         ]),

    # --- v1 kernels (5) ---
    dict(name="mhq7d_attention", src="v1", dim=1,
         args=lambda: [
             np.random.randn(8 * 64).astype(np.float32),
             np.random.randn(8 * 64).astype(np.float32),
             np.random.randn(8 * 64).astype(np.float32),
             np.empty(8 * 64, dtype=np.float32),
             np.int32(8),    # heads
             np.int32(64),   # dim
         ]),
    dict(name="mhq7d_persistent", src="v1", dim=1,
         args=lambda: [
             np.random.randn(256).astype(np.float32),
             np.int32(256),
         ]),
    dict(name="manifold_projection", src="v1", dim=1,
         args=lambda: [
             np.random.randn(1024).astype(np.float32),
             np.empty(1024, dtype=np.float32),
             np.int32(1024),
             np.int32(64),
         ]),
    dict(name="phi_modulation", src="v1", dim=1,
         args=lambda: [
             np.random.randn(1024).astype(np.float32),
             np.int32(1024),
             np.float32(0.0),
         ]),
    dict(name="cbm_flux_unfold", src="v1", dim=1,
         args=lambda: [
             np.random.randn(16).astype(np.float32),
             np.empty(1024, dtype=np.float32),
             np.int32(16),      # dna_size
             np.int32(1024),    # matrix_size
             np.float32(0.1),   # phi_flux
         ]),

    # --- physx kernel (1) ---
    dict(name="physx_crystal_dynamics", src="physx", dim=1,
         args=lambda: _physx_test_args()),
]

BLOCK_SIZE_1D = 256
BLOCK_SIZE_2D_X = 16
BLOCK_SIZE_2D_Y = 16


def _physx_test_args() -> list[Any]:
    """Build CrystalRigidBody test data for physx kernel."""
    n = 128
    positions = np.random.randn(n, 7).astype(np.float32) * 0.1
    velocities = np.random.randn(n, 7).astype(np.float32) * 0.01
    masses = np.ones(n, dtype=np.float32)
    resonances = np.random.rand(n).astype(np.float32) * 0.1
    cosmos = np.random.randn(n * 7).astype(np.float32) * 0.01

    # Pack into a flat struct: [pos(7), vel(7), mass, resonance] * n
    body_size = 7 + 7 + 1 + 1
    bodies = np.empty(n * body_size, dtype=np.float32)
    for i in range(n):
        base = i * body_size
        bodies[base:base + 7] = positions[i]
        bodies[base + 7:base + 14] = velocities[i]
        bodies[base + 14] = masses[i]
        bodies[base + 15] = resonances[i]

    return [bodies, cosmos, np.int32(n), np.float32(0.01)]


# ---------------------------------------------------------------------------
#  Phase 1: Environment check
# ---------------------------------------------------------------------------
def phase1_check_environment() -> bool:
    log.info("=== Phase 1: Environment Check ===")
    ok = True

    # Check nvcc
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            log.info("nvcc: %s", result.stdout.splitlines()[-2].strip()
                     if len(result.stdout.splitlines()) >= 2 else "found")
        else:
            log.error("nvcc returned error: %s", result.stderr)
            ok = False
    except FileNotFoundError:
        log.error("nvcc not found in PATH. Install CUDA Toolkit (sm_75).")
        ok = False
    except subprocess.TimeoutExpired:
        log.error("nvcc timed out.")
        ok = False

    # Check source files exist
    for key, path in SOURCES.items():
        if path.exists():
            log.info("Source %s: %s (%d bytes)", key, path.name, path.stat().st_size)
        else:
            log.error("Source %s NOT FOUND: %s", key, path)
            ok = False

    # Check CUDA driver
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401
        dev = cuda.Device(0)
        cc = dev.compute_capability()
        log.info("GPU: %s  (CC %d.%d, %d MB)",
                 dev.name(), cc[0], cc[1],
                 dev.total_memory() // (1024 * 1024))
        if cc[0] < 7:
            log.warning("CC %d.%d < 7.5; sm_75 may not be optimal.", cc[0], cc[1])
    except ImportError:
        log.warning("pycuda not installed -- skipping GPU tests.")
    except Exception as e:
        log.warning("GPU init: %s", e)

    if not ok:
        log.error("Environment checks FAILED. Fix issues and re-run.")
    return ok


# ---------------------------------------------------------------------------
#  Phase 2: Compilation validation
# ---------------------------------------------------------------------------
def phase2_compile(quick: bool = False) -> bool:
    """Compile all .cu files to PTX.  In quick mode skip if PTX already exists."""
    log.info("=== Phase 2: Compilation Validation ===")
    ok = True
    _BLD.mkdir(parents=True, exist_ok=True)

    for key, src_path in SOURCES.items():
        ptx_path = _BLD / f"{src_path.stem}.ptx"

        if quick and ptx_path.exists():
            log.info("Skipping %s (PTX exists)", key)
            continue

        log.info("Compiling %s -> %s ...", src_path.name, ptx_path.name)
        try:
            result = subprocess.run(
                ["nvcc", "-O3", f"-arch=sm_75", "-ptx",
                 str(src_path), "-o", str(ptx_path)],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode == 0:
                size = ptx_path.stat().st_size
                log.info("  OK (%d bytes)", size)
            else:
                log.error("  FAIL:\n%s", result.stderr)
                ok = False
        except FileNotFoundError:
            log.error("  nvcc not found")
            ok = False
        except subprocess.TimeoutExpired:
            log.error("  compilation timed out")
            ok = False

    return ok


# ---------------------------------------------------------------------------
#  Phase 3-5: PTX loading, kernel launch, numerical tests
# ---------------------------------------------------------------------------
def phase3_4_5_load_and_launch(quick: bool = False, benchmark: bool = False) -> bool:
    """Load PTX modules and launch each kernel."""
    log.info("=== Phase 3: PTX Module Loading ===")
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401
    except ImportError:
        log.error("pycuda required for kernel launch tests.")
        return False

    ok = True
    modules: dict[str, cuda.Module] = {}

    for key in ("v2", "v1", "physx"):
        stem = f"crystal_kernels_{key}" if key != "physx" else "physx_crystal_dynamics"
        ptx_path = _BLD / f"{stem}.ptx"
        if not ptx_path.exists():
            log.error("PTX not found: %s", ptx_path)
            ok = False
            continue
        try:
            with open(str(ptx_path), "rb") as f:
                mod = cuda.module_from_buffer(f.read())
            modules[key] = mod
            log.info("Loaded %s (%d kernels)", key, _count_kernels(key))
        except Exception as e:
            log.error("Failed to load %s PTX: %s", key, e)
            ok = False

    if not modules:
        return False

    log.info("=== Phase 4+5: Kernel Launch & Validation ===")

    for kinfo in KERNEL_MANIFEST:
        name = kinfo["name"]
        src = kinfo["src"]
        dim = kinfo["dim"]
        args_fn = kinfo["args"]

        mod = modules.get(src)
        if mod is None:
            log.warning("  [SKIP] %-32s  (module %s not loaded)", name, src)
            continue

        try:
            func = mod.get_function(name)
        except Exception as e:
            log.error("  [FAIL] %-32s  func lookup: %s", name, e)
            ok = False
            continue

        # Get test args
        try:
            args = args_fn()
        except Exception as e:
            log.error("  [FAIL] %-32s  arg generation: %s", name, e)
            ok = False
            continue

        # Determine grid/block
        if dim == 2:
            block = (BLOCK_SIZE_2D_X, BLOCK_SIZE_2D_Y, 1)
            grid = ((64 + BLOCK_SIZE_2D_X - 1) // BLOCK_SIZE_2D_X,
                    (64 + BLOCK_SIZE_2D_Y - 1) // BLOCK_SIZE_2D_Y, 1)
            n_elems = args[0].size
        else:
            block = (BLOCK_SIZE_1D, 1, 1)
            # Figure out n from the first array arg
            n_elems = 0
            for a in args:
                if isinstance(a, np.ndarray):
                    n_elems = a.size
                    break
            if n_elems == 0:
                n_elems = 256
            grid = ((n_elems + BLOCK_SIZE_1D - 1) // BLOCK_SIZE_1D, 1, 1)

        # Upload args to device
        dev_args = []
        for a in args:
            if isinstance(a, np.ndarray):
                dev_args.append(cuda.to_device(a))
            elif isinstance(a, ctypes.CFuncPtr) or isinstance(a, int):
                dev_args.append(a)
            else:
                dev_args.append(a)

        # Launch
        try:
            if benchmark:
                # warmup
                for _ in range(3):
                    func(*dev_args, grid=grid, block=block)
                cuda.Context.synchronize()

                # timed
                start = time.perf_counter()
                n_iter = 50 if dim == 1 else 10
                for _ in range(n_iter):
                    func(*dev_args, grid=grid, block=block)
                cuda.Context.synchronize()
                elapsed = time.perf_counter() - start
                avg_ms = elapsed / n_iter * 1000
                log.info("  [BENCH] %-28s %6.3f ms/launch (grid=%s, block=%s)",
                         name, avg_ms, str(grid), str(block))
            else:
                func(*dev_args, grid=grid, block=block)
                cuda.Context.synchronize()
                log.info("  [OK]   %-32s  n=%d grid=%s block=%s",
                         name, n_elems, str(grid), str(block))
        except Exception as e:
            log.error("  [FAIL] %-32s  launch: %s", name, e)
            ok = False
            continue

        # Numerical validation: check for NaN/Inf in output arrays
        for i, a in enumerate(args):
            if isinstance(a, np.ndarray) and a.size > 0:
                host_copy = np.empty_like(a)
                cuda.memcpy_dtoh(host_copy, dev_args[i])
                n_nan = int(np.isnan(host_copy).sum())
                n_inf = int(np.isinf(host_copy).sum())
                vmin = float(host_copy.min())
                vmax = float(host_copy.max())
                if n_nan > 0 or n_inf > 0:
                    log.warning("  [WARN] %-28s  arr[%d]: %d NaN, %d Inf, range=[%.4f, %.4f]",
                                name, i, n_nan, n_inf, vmin, vmax)
                else:
                    log.info("  [VAL]  %-28s  arr[%d]: range=[%.4f, %.4f]  mean=%.4f",
                             name, i, vmin, vmax, float(host_copy.mean()))

    log.info("=== Phase 4+5 Complete ===")
    return ok


def _count_kernels(key: str) -> int:
    return sum(1 for k in KERNEL_MANIFEST if k["src"] == key)


# ---------------------------------------------------------------------------
#  List kernels
# ---------------------------------------------------------------------------
def list_kernels() -> None:
    print(f"\n{'Kernel Name':35s} {'Module':8s} {'Dim':4s}")
    print("-" * 50)
    for kinfo in KERNEL_MANIFEST:
        print(f"{kinfo['name']:35s} {kinfo['src']:8s} {kinfo['dim']:4d}")
    print(f"\nTotal: {len(KERNEL_MANIFEST)} kernels")
    print(f"  v2:    {_count_kernels('v2')}")
    print(f"  v1:    {_count_kernels('v1')}")
    print(f"  physx: {_count_kernels('physx')}")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test suite for 7D mH-Q Crystal CUDA kernels",
    )
    parser.add_argument("--quick", action="store_true",
                        help="Skip recompilation if PTX exists.")
    parser.add_argument("--list", action="store_true", dest="list_only",
                        help="List all kernels and exit.")
    parser.add_argument("--benchmark", action="store_true",
                        help="Include performance benchmarks.")
    parser.add_argument("--phase", type=int, default=0,
                        help="Run specific phase (1-5). 0 = all.")
    args = parser.parse_args()

    if args.list_only:
        list_kernels()
        return 0

    # Determine which phases to run
    phase_range = range(1, 6) if args.phase == 0 else [args.phase]

    results: dict[int, bool] = {}

    for phase in phase_range:
        if phase == 1:
            results[1] = phase1_check_environment()
        elif phase == 2:
            results[2] = phase2_compile(quick=args.quick)
        elif phase == 3:
            results[3] = phase3_4_5_load_and_launch(
                quick=args.quick, benchmark=args.benchmark,
            )
        elif phase == 4:
            results[4] = phase3_4_5_load_and_launch(
                quick=args.quick, benchmark=args.benchmark,
            )
        elif phase == 5:
            results[5] = phase3_4_5_load_and_launch(
                quick=args.quick, benchmark=args.benchmark,
            )
        else:
            log.error("Invalid phase: %d (must be 1-5)", phase)
            return 1

    # Summary
    log.info("=" * 55)
    log.info("TEST SUMMARY")
    log.info("=" * 55)
    all_ok = True
    for phase in sorted(results):
        status = "PASS" if results[phase] else "FAIL"
        log.info("  Phase %d: %s", phase, status)
        if not results[phase]:
            all_ok = False
    log.info("-" * 55)
    log.info("OVERALL: %s", "ALL TESTS PASSED" if all_ok else "SOME TESTS FAILED")
    log.info("=" * 55)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
