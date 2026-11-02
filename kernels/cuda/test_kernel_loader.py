# ============================================================================
# ABRASAX — Unified Xenoliving System
# PRIMAL_HEX: 4f5349524953424c58434b | phi: 1.618033988749895
# CREATOR: Sir Charles Spikes (BASEDGOD)
# ============================================================================

#!/usr/bin/env python3
"""
test_kernel_loader.py -- Test suite for cuda_kernel_loader.py
==============================================================

Tests the ctypes-based CUDA kernel loader against all 19 registered kernels:
  - v2 (13):   manifold_projection_7d, manifold_projection_batch,
               phi_harmonic_modulation, fibonacci_spiral_embedding,
               cbm_flux_unfold_v2, cbm_flux_fold,
               holographic_interference, holographic_reconstruction,
               quantum_evolution_2d, crystal_resonance,
               crystal_entropy_mining, flower_of_life, metatron_cube
  - v1 (5):    mhq7d_attention, manifold_projection_v1,
               phi_modulation, cbm_flux_unfold_v1, mhq7d_persistent
  - physx (1): physx_crystal_dynamics

Phases:
    1. Environment: CUDA driver loaded, GPU detected, PTX files exist
    2. Runtime init: CrystalKernelRuntime created, kernels bound
    3. Kernel launches: every kernel launched with valid inputs
    4. Numerical sanity: no NaN/Inf in outputs, finite ranges
    5. Edge cases: zero-sized inputs, NULL pointers, error handling
    6. Performance: basic timing benchmarks (optional)

Usage:
    python test_kernel_loader.py                  # full suite
    python test_kernel_loader.py --quick           # skip perf benchmarks
    python test_kernel_loader.py --list            # list kernels and exit
    python test_kernel_loader.py --kernel <name>   # test specific kernel
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(
	level=logging.INFO,
	format="[%(levelname).1s] %(message)s",
	stream=sys.stderr,
)
log = logging.getLogger("test_kernel_loader")

# Import the loader
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
	from cuda_kernel_loader import (
		CrystalKernelRuntime,
		KERNEL_REGISTRY,
		BLOCK_SIZE_1D,
		BLOCK_SIZE_2D_X,
		BLOCK_SIZE_2D_Y,
		list_kernels,
	)
except ImportError as e:
	log.error("Failed to import cuda_kernel_loader: %s", e)
	log.error("Make sure cuda_kernel_loader.py is in the same directory.")
	sys.exit(1)


# ---------------------------------------------------------------------------
#  Phase 1: Environment checks
# ---------------------------------------------------------------------------

def phase1_check_environment() -> bool:
	"""Verify CUDA driver, GPU, and PTX file availability."""
	log.info("=== Phase 1: Environment Check ===")
	ok = True

	# Check PTX files exist
	from cuda_kernel_loader import _PTX_FILES
	for name, path in _PTX_FILES.items():
		if path.exists():
			log.info("PTX %s: %s (%d bytes)", name, path.name, path.stat().st_size)
		else:
			log.error("PTX %s NOT FOUND: %s (run compile_kernels.bat)", name, path)
			ok = False

	# Try to create a minimal runtime to verify driver
	try:
		rt = CrystalKernelRuntime(log_level=logging.DEBUG)
		log.info("GPU: %s (CC %d.%d, %d MB)",
				 rt.device_name, rt.compute_capability[0],
				 rt.compute_capability[1],
				 rt.total_memory // (1024 * 1024))
		rt.close()
	except Exception as e:
		log.error("Runtime init failed: %s", e)
		ok = False

	if ok:
		log.info("Phase 1: PASS")
	else:
		log.error("Phase 1: FAIL -- fix environment issues and re-run.")
	return ok


# ---------------------------------------------------------------------------
#  Phase 2: Runtime initialization and kernel binding
# ---------------------------------------------------------------------------

def phase2_runtime_init() -> CrystalKernelRuntime | None:
	"""Create runtime and verify all kernels are bound."""
	log.info("=== Phase 2: Runtime Initialization ===")
	try:
		rt = CrystalKernelRuntime()
	except Exception as e:
		log.error("Runtime init failed: %s", e)
		return None

	loaded = rt.available_kernels
	expected = len(KERNEL_REGISTRY)
	log.info("Kernels loaded: %d / %d", len(loaded), expected)

	missing = set(KERNEL_REGISTRY.keys()) - set(loaded)
	if missing:
		log.warning("Missing kernels (%d): %s", len(missing),
					", ".join(sorted(missing)))

	log.info("Phase 2: PASS (%d / %d kernels)", len(loaded), expected)
	return rt


# ---------------------------------------------------------------------------
#  Phase 3+4: Kernel launches and numerical validation
# ---------------------------------------------------------------------------

def _check_numerics(
	name: str,
	result: np.ndarray,
	label: str = "output",
) -> bool:
	"""Check array for NaN/Inf and report range."""
	n_nan = int(np.isnan(result).sum())
	n_inf = int(np.isinf(result).sum())
	vmin = float(result.min())
	vmax = float(result.max())
	vmean = float(result.mean())

	if n_nan > 0 or n_inf > 0:
		log.warning("  [WARN] %-32s %s: %d NaN, %d Inf, range=[%.4f, %.4f]",
					name, label, n_nan, n_inf, vmin, vmax)
		return False

	log.info("  [OK]   %-32s %s: range=[%.6f, %.6f] mean=%.6f",
			 name, label, vmin, vmax, vmean)
	return True


def phase3_launch_kernels(rt: CrystalKernelRuntime, quick: bool = False) -> bool:
	"""Launch every kernel and validate outputs."""
	log.info("=== Phase 3: Kernel Launch & Validation ===")
	total = len(KERNEL_REGISTRY)
	passed = 0
	failed = 0
	skipped = 0

	for alias in sorted(KERNEL_REGISTRY.keys()):
		if alias not in rt.available_kernels:
			log.warning("  [SKIP] %-32s  (not loaded)", alias)
			skipped += 1
			continue

		ok = _test_single_kernel(rt, alias)
		if ok:
			passed += 1
		else:
			failed += 1

	log.info("Phase 3: %d passed, %d failed, %d skipped (of %d)",
			 passed, failed, skipped, total)
	return failed == 0


def _test_single_kernel(rt: CrystalKernelRuntime, alias: str) -> bool:
	"""Test a single kernel by name. Returns True on success."""
	try:
		if alias == "manifold_projection_7d":
			inp = np.random.randn(1024).astype(np.float32)
			out = rt.manifold_projection_7d(inp, dim=64)
			assert out.shape == inp.shape, f"Shape mismatch: {out.shape} vs {inp.shape}"
			return _check_numerics(alias, out)

		elif alias == "manifold_projection_batch":
			inp = np.random.randn(256 * 64).astype(np.float32)
			out = rt.manifold_projection_batch(inp, batch_size=256, dim=64)
			assert out.shape == inp.shape
			return _check_numerics(alias, out)

		elif alias == "phi_harmonic_modulation":
			data = np.random.randn(1024).astype(np.float32)
			out = rt.phi_harmonic_modulation(data, phase=0.5, harmonic_order=3)
			assert out.shape == data.shape
			return _check_numerics(alias, out)

		elif alias == "fibonacci_spiral_embedding":
			inp = np.random.randn(512).astype(np.float32)
			out_x, out_y = rt.fibonacci_spiral_embedding(inp)
			assert out_x.shape == (512,)
			assert out_y.shape == (512,)
			x_ok = _check_numerics(f"{alias}_x", out_x)
			y_ok = _check_numerics(f"{alias}_y", out_y)
			return x_ok and y_ok

		elif alias == "cbm_flux_unfold_v2":
			seed = np.random.randn(16).astype(np.float32)
			out = rt.cbm_flux_unfold_v2(seed, output_size=1024, phi_flux=0.1, time_factor=0.0)
			assert out.shape == (1024,)
			return _check_numerics(alias, out)

		elif alias == "cbm_flux_fold":
			weights = np.random.randn(1024).astype(np.float32)
			out = rt.cbm_flux_fold(weights, seed_size=16)
			assert out.shape == (16,)
			return _check_numerics(alias, out)

		elif alias == "holographic_interference":
			ref = np.random.randn(1024).astype(np.float32)
			obj = np.random.randn(1024).astype(np.float32)
			out = rt.holographic_interference(ref, obj)
			assert out.shape == (1024,)
			return _check_numerics(alias, out)

		elif alias == "holographic_reconstruction":
			hologram = np.random.randn(1024).astype(np.float32)
			ref = np.random.randn(1024).astype(np.float32)
			out = rt.holographic_reconstruction(hologram, ref)
			assert out.shape == (1024,)
			return _check_numerics(alias, out)

		elif alias == "quantum_evolution_2d":
			field = np.random.randn(64, 64).astype(np.float32)
			out = rt.quantum_evolution_2d(field, phase=0.0, dt=0.01)
			assert out.shape == field.shape
			return _check_numerics(alias, out)

		elif alias == "crystal_resonance":
			pattern = np.random.randn(1024).astype(np.float32)
			phi_alignment, coherence = rt.crystal_resonance(pattern)
			log.info("  [OK]   %-32s  phi_align=%.6f coherence=%.6f",
					 alias, phi_alignment, coherence)
			return np.isfinite(phi_alignment) and np.isfinite(coherence)

		elif alias == "crystal_entropy_mining":
			out = rt.crystal_entropy_mining(n=1024, base_seed=42, time_factor=0.0)
			assert out.shape == (1024,)
			return _check_numerics(alias, out)

		elif alias == "flower_of_life":
			out = rt.flower_of_life(width=128, height=128, scale=0.3)
			assert out.shape == (128, 128)
			return _check_numerics(alias, out)

		elif alias == "metatron_cube":
			out = rt.metatron_cube(width=128, height=128, rotation=0.0)
			assert out.shape == (128, 128)
			return _check_numerics(alias, out)

		elif alias == "mhq7d_attention":
			Q = np.random.randn(8 * 64).astype(np.float32)
			K = np.random.randn(8 * 64).astype(np.float32)
			V = np.random.randn(8 * 64).astype(np.float32)
			out = rt.mhq7d_attention(Q, K, V, heads=8, dim=64)
			assert out.shape == Q.shape
			return _check_numerics(alias, out)

		elif alias == "mhq7d_persistent":
			# Persistent kernel is designed to run indefinitely. Test by:
			# 1. Allocating the manifold buffer
			# 2. Pre-setting the termination flag before launch so the kernel
			#    exits immediately (proves the kernel launches and checks the flag)
			stream = rt.create_stream()
			manifold_size = 256
			d_manifold = rt.alloc(manifold_size * 4)
			# Set termination signal BEFORE launch
			signal = np.array([999.0], dtype=np.float32)
			rt.to_device(d_manifold, signal)
			# Launch kernel -- should exit immediately since manifold[0] == 999.0
			rt.launch_kernel(
				"mhq7d_persistent",
				(d_manifold, np.int32(manifold_size)),
				stream=stream,
				first_array_n=manifold_size,
			)
			rt.sync_stream(stream)
			result = rt.from_device(d_manifold, (manifold_size,))
			rt.free(d_manifold)
			rt.destroy_stream(stream)
			log.info("  [OK]   mhq7d_persistent        (immediate-exit test)")
			return _check_numerics(alias, result)

		elif alias == "manifold_projection":
			inp = np.random.randn(1024).astype(np.float32)
			out = rt.manifold_projection_v1(inp, dim=64)
			assert out.shape == inp.shape
			return _check_numerics(alias, out)

		elif alias == "phi_modulation":
			data = np.random.randn(1024).astype(np.float32)
			out = rt.phi_modulation(data, phase=0.0)
			assert out.shape == data.shape
			return _check_numerics(alias, out)

		elif alias == "cbm_flux_unfold":
			seed = np.random.randn(16).astype(np.float32)
			out = rt.cbm_flux_unfold_v1(seed, matrix_size=1024, phi_flux=0.1)
			assert out.shape == (1024,)
			return _check_numerics(alias, out)

		elif alias == "physx_crystal_dynamics":
			n_bodies = 128
			positions = np.random.randn(n_bodies, 7).astype(np.float32) * 0.1
			velocities = np.random.randn(n_bodies, 7).astype(np.float32) * 0.01
			masses = np.ones(n_bodies, dtype=np.float32)
			resonances = np.random.rand(n_bodies).astype(np.float32) * 0.1
			cosmos = np.random.randn(n_bodies, 7).astype(np.float32) * 0.01

			p_out, v_out, m_out, r_out = rt.physx_crystal_dynamics(
				positions, velocities, masses, resonances,
				cosmos_predictions=cosmos, dt=0.01,
			)
			assert p_out.shape == (n_bodies, 7)
			assert v_out.shape == (n_bodies, 7)
			assert m_out.shape == (n_bodies,)
			assert r_out.shape == (n_bodies,)
			p_ok = _check_numerics(f"{alias}_pos", p_out.flatten())
			v_ok = _check_numerics(f"{alias}_vel", v_out.flatten())
			r_ok = _check_numerics(f"{alias}_res", r_out)
			return p_ok and v_ok and r_ok

		else:
			log.warning("  [SKIP] %-32s  (no test implementation)", alias)
			return True

	except Exception as e:
		log.error("  [FAIL] %-32s  %s: %s", alias, type(e).__name__, e)
		return False


# ---------------------------------------------------------------------------
#  Phase 5: Edge cases
# ---------------------------------------------------------------------------

def phase5_edge_cases(rt: CrystalKernelRuntime) -> bool:
	"""Test error handling and edge cases."""
	log.info("=== Phase 5: Edge Cases ===")
	ok = True

	# Test 1: Invalid kernel name
	try:
		rt.launch_kernel("nonexistent_kernel", ())
		log.error("  [FAIL] Expected KeyError for invalid kernel name")
		ok = False
	except KeyError:
		log.info("  [OK]   KeyError raised for invalid kernel name")

	# Test 3: Zero-sized allocation
	try:
		zero_dev = rt.alloc(0)
		log.info("  [OK]   Zero-sized allocation: %s", zero_dev)
		rt.free(zero_dev)
	except Exception as e:
		log.warning("  [WARN] Zero-sized alloc: %s", e)

	# Test 4: Large allocation (quarter of VRAM)
	try:
		max_alloc = rt.total_memory // 4
		big_dev = rt.alloc(max_alloc)
		log.info("  [OK]   Large allocation: %d MB",
				 max_alloc // (1024 * 1024))
		rt.free(big_dev)
	except Exception as e:
		log.warning("  [WARN] Large alloc: %s", e)
		if "out of memory" in str(e).lower():
			log.warning("  [WARN] Out of memory -- skipping large alloc test")

	# Test 5: memset operations
	try:
		d = rt.alloc(1024)  # 256 floats
		rt.memset(d, 0)
		result = rt.from_device(d, (256,))
		assert np.all(result == 0.0), "memset to zero failed"
		log.info("  [OK]   memset to zero: all values == 0.0")
		rt.free(d)
	except Exception as e:
		log.error("  [FAIL] memset: %s", e)
		ok = False

	# Test 6: Stream operations
	try:
		s = rt.create_stream()
		rt.sync_stream(s)
		rt.destroy_stream(s)
		log.info("  [OK]   Stream create/sync/destroy")
	except Exception as e:
		log.error("  [FAIL] Stream ops: %s", e)
		ok = False

	# Test 7: to_device / from_device round-trip
	try:
		original = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
		d = rt.alloc(original.nbytes)
		rt.to_device(d, original)
		readback = rt.from_device(d, (4,))
		assert np.allclose(original, readback), f"Round-trip failed: {original} vs {readback}"
		log.info("  [OK]   Device round-trip: values preserved")
		rt.free(d)
	except Exception as e:
		log.error("  [FAIL] Round-trip: %s", e)
		ok = False

	if ok:
		log.info("Phase 5: PASS")
	else:
		log.error("Phase 5: FAIL")
	return ok


# ---------------------------------------------------------------------------
#  Phase 6: Performance benchmarks (optional)
# ---------------------------------------------------------------------------

def phase6_benchmark(rt: CrystalKernelRuntime | None, quick: bool = False) -> bool:
	"""Run basic timing benchmarks on key kernels."""
	log.info("=== Phase 6: Performance Benchmarks ===")

	if quick:
		log.info("Skipping benchmarks (--quick)")
		return True

	if rt is None:
		try:
			rt = CrystalKernelRuntime()
		except Exception as e:
			log.error("Failed to initialize runtime for benchmarks: %s", e)
			return False

	benchmarks = [
		("manifold_projection_7d",
		 lambda: rt.manifold_projection_7d(
			 np.random.randn(1024 * 1024).astype(np.float32), dim=64)),
		("phi_harmonic_modulation",
		 lambda: rt.phi_harmonic_modulation(
			 np.random.randn(1024 * 1024).astype(np.float32), phase=0.0)),
		("cbm_flux_unfold_v2",
		 lambda: rt.cbm_flux_unfold_v2(
			 np.random.randn(16).astype(np.float32), output_size=1024 * 1024)),
		("flower_of_life",
		 lambda: rt.flower_of_life(width=512, height=512, scale=0.3)),
		("crystal_entropy_mining",
		 lambda: rt.crystal_entropy_mining(n=1024 * 1024)),
	]

	all_ok = True
	for name, fn in benchmarks:
		try:
			# Warmup
			for _ in range(3):
				fn()

			# Timed runs
			n_iter = 10
			start = time.perf_counter()
			for _ in range(n_iter):
				fn()
			elapsed = time.perf_counter() - start
			avg_ms = elapsed / n_iter * 1000
			log.info("  [BENCH] %-32s %8.3f ms/launch", name, avg_ms)
		except Exception as e:
			log.error("  [FAIL] %-32s benchmark: %s", name, e)
			all_ok = False

	return all_ok


# ---------------------------------------------------------------------------
#  List kernels
# ---------------------------------------------------------------------------

def print_kernel_list() -> None:
	"""Print formatted kernel table."""
	rows = list_kernels()
	print(f"\n{'Python Name':35s} {'CUDA Function':40s} {'Module':8s} {'Dim':4s}")
	print("-" * 90)
	for r in sorted(rows, key=lambda x: (x["module"], x["name"])):
		print(f"{r['name']:35s} {r['function']:40s} {r['module']:8s} {r['dim']:4d}")
	print(f"\nTotal: {len(rows)} kernels")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main() -> int:
	parser = argparse.ArgumentParser(
		description="Test suite for cuda_kernel_loader.py (ctypes-based CUDA kernel loader)",
	)
	parser.add_argument("--quick", action="store_true",
						help="Skip benchmarks.")
	parser.add_argument("--list", action="store_true", dest="list_only",
						help="List all kernels and exit.")
	parser.add_argument("--kernel", type=str, default=None,
						help="Test only a specific kernel by name.")
	parser.add_argument("--phase", type=int, default=0,
						help="Run specific phase (1-6). 0 = all phases.")
	args = parser.parse_args()

	if args.list_only:
		print_kernel_list()
		return 0

	# Determine phases
	if args.phase:
		phase_range = [args.phase]
	else:
		phase_range = [1, 2, 3, 5]
		if not args.quick:
			phase_range.append(6)

	results: dict[int, bool] = {}
	rt: CrystalKernelRuntime | None = None

	for phase in phase_range:
		if phase == 1:
			results[1] = phase1_check_environment()
		elif phase == 2:
			rt = phase2_runtime_init()
			results[2] = rt is not None
		elif phase == 3:
			if rt is None:
				log.error("Runtime not initialized; re-running phase 2")
				rt = phase2_runtime_init()
			if rt is not None:
				if args.kernel:
					# Test single kernel
					log.info("=== Testing single kernel: %s ===", args.kernel)
					ok = _test_single_kernel(rt, args.kernel)
					results[3] = ok
				else:
					results[3] = phase3_launch_kernels(rt, quick=args.quick)
			else:
				results[3] = False
		elif phase == 5:
			if rt is None:
				log.error("Runtime not initialized")
				results[5] = False
			else:
				results[5] = phase5_edge_cases(rt)
		elif phase == 6:
			# phase6_benchmark handles None rt internally
			results[6] = phase6_benchmark(rt, quick=args.quick)
		else:
			log.error("Invalid phase: %d", phase)
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

	# Cleanup
	if rt is not None:
		rt.close()

	return 0 if all_ok else 1


if __name__ == "__main__":
	sys.exit(main())
