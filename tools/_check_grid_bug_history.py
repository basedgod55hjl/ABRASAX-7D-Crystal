"""Historical check: did the PRE-FIX loader really produce the 8.5e-01 8x64 error?

diagnose_quantum_grid.py claims, "MEASURED 2026-11-01", that quantum_evolution_2d
on an 8x64 field was off by 8.5e-01 because the launch grid was inferred wrongly.
cuda_kernel_loader.py.bak-gridfix-20261101 is the loader from before the fix.

Each version is loaded in its own subprocess. The loader resolves .ptx/.dll paths
from __file__ (module-level _BLD plus the _MODULE_PATHS dict), so the staged copy
is placed inside kernels/cuda under a throwaway name and removed afterwards --
patching those constants after import does not reach the already-built dict.

Exit 0 only if the old loader is measurably wrong, the shape of the error matches
the record (columns 32..63, not 16..63), and the current loader is exact -- which
together also proves the regression gate in diagnose_quantum_grid.py is not
vacuous: the pre-fix grids fail its ceil-division assertion.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CUDA_DIR = os.path.join(ROOT, "kernels", "cuda")
BACKUP = os.path.join(CUDA_DIR, "cuda_kernel_loader.py.bak-gridfix-20261101")
CURRENT = os.path.join(CUDA_DIR, "cuda_kernel_loader.py")
STAGED_NAME = "_hist_probe_loader.py"
PY = sys.executable
BLOCK = (16, 16)

PROBE = r'''
import importlib.util, inspect, json, sys
import numpy as np

loader_path, tools_dir = sys.argv[1], sys.argv[2]
sys.path.insert(0, tools_dir)

spec = importlib.util.spec_from_file_location("loader_under_test", loader_path)
mod = importlib.util.module_from_spec(spec)
sys.modules["loader_under_test"] = mod
spec.loader.exec_module(mod)

from verify_kernel_math_quantum import ref_quantum_evolution

rt = mod.CrystalKernelRuntime()
original = rt._get_grid_block
params = inspect.signature(original).parameters
seen = []


def wrapped(name, grid, block, n=None, shape=None):
    kwargs = {}
    if "n" in params:
        kwargs["n"] = n
    if "shape" in params:
        kwargs["shape"] = shape
    out = original(name, grid, block, **kwargs)
    seen.append(tuple(out[0]))
    return out


rt._get_grid_block = wrapped

rng = np.random.default_rng(1)
shapes = [(16, 16), (32, 32), (8, 64), (64, 8), (5, 100)]
per_shape = {}
for shape in shapes:
    field = (rng.standard_normal(shape) * 0.5).astype(np.float32)
    seen.clear()
    gpu = np.asarray(rt.quantum_evolution_2d(field.copy(), phase=0.3, dt=0.1),
                     dtype=np.float64)
    cpu = ref_quantum_evolution(field, 0.3, 0.1, toroidal=True)
    per_shape["%dx%d" % shape] = {
        "grid": list(seen[-1]) if seen else None,
        "max_err": float(np.abs(gpu - cpu).max()),
    }

field = (rng.standard_normal((8, 64)) * 0.5).astype(np.float32)
gpu = np.asarray(rt.quantum_evolution_2d(field.copy(), phase=0.3, dt=0.1),
                 dtype=np.float64)
cpu = ref_quantum_evolution(field, 0.3, 0.1, toroidal=True)
err = np.abs(gpu - cpu)

print(json.dumps({
    "per_shape": per_shape,
    "max_err_8x64": float(err.max()),
    "covered_x16": float(err[:, :16].max()),
    "uncovered_x16": float(err[:, 16:].max()),
    "bad_columns": [int(i) for i in np.nonzero(err.max(axis=0) > 1e-3)[0]],
    "uncovered_all_zero": bool(np.allclose(gpu[:, 16:], 0.0)),
    "kernels": len(list(getattr(rt, "available_kernels", []) or [])),
}))
'''


def expected_grid(shape):
    h, w = shape
    return (-(-w // BLOCK[0]), -(-h // BLOCK[1]), 1)


def probe(loader_path, label):
    staged = os.path.join(CUDA_DIR, STAGED_NAME)
    try:
        shutil.copy2(loader_path, staged)
        proc = subprocess.run([PY, "-c", PROBE, staged, HERE],
                              capture_output=True, text=True, timeout=300)
    finally:
        if os.path.exists(staged):
            os.remove(staged)
    line = [ln for ln in (proc.stdout or "").splitlines() if ln.startswith("{")]
    if not line:
        tail = (proc.stderr or "").strip().splitlines()[-2:]
        return {"error": " | ".join(tail) if tail else "no output", "label": label}
    data = json.loads(line[-1])
    data["label"] = label
    return data


def grids_ok(data):
    """Do this loader's dispatched grids equal the ceil-division of each shape?"""
    got = data.get("per_shape", {})
    bad = []
    for key, want_shape in (("16x16", (16, 16)), ("32x32", (32, 32)),
                            ("8x64", (8, 64)), ("64x8", (64, 8)),
                            ("5x100", (5, 100))):
        want = expected_grid(want_shape)
        have = tuple(got.get(key, {}).get("grid") or ())
        if have != want:
            bad.append((key, have, want))
    return bad


def main():
    if not os.path.exists(BACKUP):
        print("backup loader missing: %s" % BACKUP)
        return 1

    old = probe(BACKUP, "pre-fix")
    new = probe(CURRENT, "current")
    print("pre-fix  : %s" % json.dumps(old))
    print("current  : %s" % json.dumps(new))

    nan = float("nan")
    old_bad_grids = grids_ok(old)
    new_bad_grids = grids_ok(new)
    cases = [
        ("pre-fix 8x64 is measurably wrong",
         old.get("max_err_8x64", 0) > 1e-2,
         "max_err=%.3e" % old.get("max_err_8x64", nan)),
        ("pre-fix error matches the documented 8.5e-01",
         abs(old.get("max_err_8x64", 0) - 8.5e-01) < 5e-02,
         "max_err=%.3e vs documented 8.5e-01" % old.get("max_err_8x64", nan)),
        ("pre-fix only damages columns >= 16",
         old.get("covered_x16", 1) < 2e-5 and old.get("uncovered_x16", 0) > 1e-2,
         "x<16=%.2e x>=16=%.2e" % (old.get("covered_x16", nan),
                                   old.get("uncovered_x16", nan))),
        ("pre-fix damage starts at column 32, not 16",
         old.get("bad_columns") == list(range(32, 64)),
         "%d bad columns, first=%s"
         % (len(old.get("bad_columns") or []),
            (old.get("bad_columns") or ["none"])[0])),
        ("current loader is exact on 8x64",
         new.get("max_err_8x64", 1) < 2e-5,
         "max_err=%.3e" % new.get("max_err_8x64", nan)),
        ("current loader leaves no bad columns",
         new.get("bad_columns") == [], "bad_columns=%s" % new.get("bad_columns")),
        ("regression gate is not vacuous: pre-fix grids violate ceil-div",
         sorted(k for k, _, _ in old_bad_grids) == ["5x100", "64x8", "8x64"],
         "%d/%d shapes" % (len(old_bad_grids), 5)),
        ("current loader satisfies ceil-div on every shape",
         new_bad_grids == [], "mismatches=%s" % new_bad_grids),
    ]

    for name, ok, detail in cases:
        print("[%s] %-52s %s" % ("PASS" if ok else "FAIL", name, detail))
    bad = [c for c in cases if not c[1]]
    print("\n%d checks: %d pass, %d fail"
          % (len(cases), len(cases) - len(bad), len(bad)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
