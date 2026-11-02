"""Quantum-evolution grid regression + forensic record of the GRID BUG.

THE BUG (real, fixed 2026-11-01 in kernels/cuda/cuda_kernel_loader.py):

    _get_grid_block inferred a 2-D launch grid with a SQUARE assumption,
        side = int(n ** 0.5)
        grid = (ceil(side/16), ceil(side/16), 1)
    so an 8x64 field (n=512) got side=22 -> a 2x2 grid: x covered 32 of 64
    columns, so columns 32..63 were never written. Measured against the numpy
    reference: max_err 8.348e-01 confined to columns 32..63, while columns
    0..31 were exact at 6.8e-08. Not "one block dispatched" -- two blocks of 16
    in x, and two rows in y that the 8-row field never used.

    Square fields hid it: 16x16 -> side=16 -> (1,1,1) covers all; 32x32 ->
    side=32 -> (2,2,1) covers all. Only non-square fields exposed it.

THE FIX: the caller passes first_array_shape=(h, w) and the grid is ceil
division per axis, so 8x64 -> (4, 1, 1) with block (16, 16, 1).

Two modes:

    python diagnose_quantum_grid.py            regression -- every shape exact
                                               AND the dispatched grid equals
                                               the ceil division of that shape
    python diagnose_quantum_grid.py --legacy   forensics -- load the pre-fix
                                               loader (.bak-gridfix-20261101)
                                               and reproduce 8.348e-01 with bad
                                               columns exactly 32..63

Exit 0 when the mode's assertions hold.
"""
from __future__ import annotations

import importlib.util
import inspect
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CUDA_DIR = os.path.join(ROOT, "kernels", "cuda")
LEGACY_LOADER = os.path.join(CUDA_DIR, "cuda_kernel_loader.py.bak-gridfix-20261101")
STAGED_NAME = "_hist_probe_loader.py"
BLOCK = (16, 16)
SHAPES = [(16, 16), (32, 32), (8, 64), (64, 8), (5, 100)]
TOL = 2e-5

sys.path.insert(0, CUDA_DIR)
sys.path.insert(0, HERE)

from verify_kernel_math_quantum import ref_quantum_evolution  # noqa: E402


def expected_grid(shape):
    h, w = shape
    return (-(-w // BLOCK[0]), -(-h // BLOCK[1]), 1)


def install_recorder(rt):
    """Record the grid the loader actually dispatches, whatever its signature."""
    recorder = []
    original = rt._get_grid_block
    params = inspect.signature(original).parameters

    def wrapped(name, grid, block, n=None, shape=None):
        kwargs = {}
        if "n" in params:
            kwargs["n"] = n
        if "shape" in params:
            kwargs["shape"] = shape
        out = original(name, grid, block, **kwargs)
        recorder.append(tuple(out[0]))
        return out

    rt._get_grid_block = wrapped
    return recorder


def run_shapes(rt, recorder, label, expect_ok):
    rng = np.random.default_rng(1)
    failures = []
    print("\n== %s ==" % label)
    for shape in SHAPES:
        field = (rng.standard_normal(shape) * 0.5).astype(np.float32)
        recorder.clear()
        gpu = np.asarray(rt.quantum_evolution_2d(field.copy(), phase=0.3, dt=0.1),
                         dtype=np.float64)
        cpu = ref_quantum_evolution(field, 0.3, 0.1, toroidal=True)
        err = float(np.abs(gpu - cpu).max())
        grid = recorder[-1] if recorder else None
        want = expected_grid(shape)
        num_ok = err <= TOL
        if not expect_ok:
            num_ok = True  # forensics: we are documenting, not asserting
        grid_ok = grid == want
        print("  %-9s grid=%-12s want=%-12s max_err=%.3e  %s%s"
              % ("%dx%d" % shape, grid, want, err,
                 "OK" if num_ok else "FAIL",
                 "" if grid_ok else "  GRID MISMATCH"))
        if not (num_ok and grid_ok):
            failures.append((shape, grid, want, err))
    return failures


def current_mode():
    from cuda_kernel_loader import CrystalKernelRuntime

    rt = CrystalKernelRuntime()
    print("loader: %s" % sys.modules["cuda_kernel_loader"].__file__)
    print("kernels bound: %d" % len(rt.available_kernels))
    recorder = install_recorder(rt)
    failures = run_shapes(rt, recorder, "current loader (regression)", True)
    print("\n%d shape(s) failed" % len(failures))
    for f in failures:
        print("  ", f)
    return 0 if not failures else 1


def load_legacy():
    staged = os.path.join(CUDA_DIR, STAGED_NAME)
    try:
        shutil.copy2(LEGACY_LOADER, staged)
        spec = importlib.util.spec_from_file_location("legacy_loader", staged)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["legacy_loader"] = mod
        spec.loader.exec_module(mod)
        rt = mod.CrystalKernelRuntime()
    finally:
        if os.path.exists(staged):
            os.remove(staged)
    return rt


def legacy_mode():
    if not os.path.exists(LEGACY_LOADER):
        print("legacy loader missing: %s" % LEGACY_LOADER)
        return 1
    rt = load_legacy()
    print("loader: %s" % LEGACY_LOADER)
    print("kernels bound: %d" % len(rt.available_kernels))
    recorder = install_recorder(rt)
    run_shapes(rt, recorder, "pre-fix loader (forensics)", False)

    rng = np.random.default_rng(1)
    field = (rng.standard_normal((8, 64)) * 0.5).astype(np.float32)
    gpu = np.asarray(rt.quantum_evolution_2d(field.copy(), phase=0.3, dt=0.1),
                     dtype=np.float64)
    cpu = ref_quantum_evolution(field, 0.3, 0.1, toroidal=True)
    err = np.abs(gpu - cpu)
    bad = [int(i) for i in np.nonzero(err.max(axis=0) > 1e-3)[0]]
    max_err = float(err.max())
    print("\n8x64 forensics: max_err=%.4e  bad_columns=%s" % (max_err, bad))
    print("documented 2026-11-01: 8.5e-01 ; measured now: %.4e" % max_err)

    ok = (0.80 < max_err < 0.90) and bad == list(range(32, 64))
    print("%s: pre-fix loader reproduces the documented failure"
          % ("CONFIRMED" if ok else "NOT CONFIRMED"))
    return 0 if ok else 1


def main():
    if "--legacy" in sys.argv:
        return legacy_mode()
    return current_mode()


if __name__ == "__main__":
    sys.exit(main())
