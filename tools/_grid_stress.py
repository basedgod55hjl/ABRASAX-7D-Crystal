"""Is the KERNEL correct, and is the fault really only the launch geometry?

The diagnostic docstring asserts: "The kernel is correct; the launch dimensions
are not." That is a testable claim, and this separates the two halves.

PART A -- geometry isolation. One kernel binary, three launch paths, same field:
    pre-fix loader, as shipped          -> must be wrong on 8x64
    pre-fix loader, grid forced correct -> must be EXACT (only the dispatch
                                           geometry changes; .ptx/.o untouched)
    current loader                      -> must be exact
If the forced-grid run is exact, the kernel is vindicated and the defect was
purely inference -- the docstring's claim, proven mechanically.

The force is applied by wrapping launch_kernel (the pre-fix _get_grid_block has no
shape argument at all, which is the bug: it cannot know h and w).

PART B -- adversarial shape matrix. tiny (1x1), degenerate (1xN, Nx1), block
boundaries (15,16,17,31,32,33), primes (127x129), extreme aspect (3x1000), large
(256x256). Each must be exact AND must dispatch exactly the ceil division of its
own shape.

Exit 0 only if every expectation holds.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CUDA_DIR = os.path.join(ROOT, "kernels", "cuda")
LEGACY = os.path.join(CUDA_DIR, "cuda_kernel_loader.py.bak-gridfix-20261101")
CURRENT = os.path.join(CUDA_DIR, "cuda_kernel_loader.py")
STAGED_NAME = "_stress_probe_loader.py"
PY = sys.executable
BLOCK = (16, 16)
TOL = 2e-5
KEY_SHAPES = [(8, 64), (1, 64), (3, 1000), (31, 33)]

SHAPES = [(1, 1), (1, 64), (64, 1), (1, 100), (8, 64), (64, 8), (15, 15),
          (16, 16), (17, 17), (31, 33), (33, 31), (3, 1000), (1000, 3),
          (127, 129), (256, 256)]

PROBE = r'''
import hashlib, importlib.util, inspect, json, sys
import numpy as np

loader_path, tools_dir, mode, shapes_json = (sys.argv[1], sys.argv[2],
                                             sys.argv[3], sys.argv[4])
sys.path.insert(0, tools_dir)

spec = importlib.util.spec_from_file_location("loader_under_test", loader_path)
mod = importlib.util.module_from_spec(spec)
sys.modules["loader_under_test"] = mod
spec.loader.exec_module(mod)

from verify_kernel_math_quantum import ref_quantum_evolution

BLOCK = (16, 16)
state = {"shape": None}
seen = []


def ceil_grid(shape):
    h, w = shape
    return (-(-w // BLOCK[0]), -(-h // BLOCK[1]), 1)


rt = mod.CrystalKernelRuntime()

if mode == "forced":
    # Dispatch the correct geometry explicitly. The kernel and its .ptx are
    # untouched; only the grid handed to cuLaunchKernel changes.
    original_launch = rt.launch_kernel

    def launch(name, args, grid=None, block=None, **kw):
        if "quantum" in name and state["shape"] is not None:
            grid = ceil_grid(state["shape"])
            block = (BLOCK[0], BLOCK[1], 1)
        seen.append(tuple(grid) if grid else None)
        return original_launch(name, args, grid=grid, block=block, **kw)

    rt.launch_kernel = launch
else:
    # Record the grid the shipped inference actually produces.
    original = rt._get_grid_block
    params = inspect.signature(original).parameters

    def recording(name, grid, block, n=None, shape=None):
        kwargs = {}
        if "n" in params:
            kwargs["n"] = n
        if "shape" in params:
            kwargs["shape"] = shape
        out = original(name, grid, block, **kwargs)
        seen.append(tuple(out[0]))
        return out

    rt._get_grid_block = recording

rows = []
for shape in [tuple(s) for s in json.loads(shapes_json)]:
    state["shape"] = shape
    rng = np.random.default_rng(7)
    field = (rng.standard_normal(shape) * 0.5).astype(np.float32)
    seen.clear()
    gpu = np.asarray(rt.quantum_evolution_2d(field.copy(), phase=0.3, dt=0.1),
                     dtype=np.float64)
    cpu = ref_quantum_evolution(field, 0.3, 0.1, toroidal=True)
    rows.append({
        "shape": list(shape),
        "grid": list(seen[-1]) if seen and seen[-1] else None,
        "want": list(ceil_grid(shape)),
        "max_err": float(np.abs(gpu - cpu).max()),
        "finite": bool(np.isfinite(gpu).all()),
        "digest": hashlib.sha256(np.ascontiguousarray(gpu).tobytes()).hexdigest()[:16],
    })

print(json.dumps({"mode": mode, "kernels": len(list(rt.available_kernels)),
                  "rows": rows}))
'''


def probe(loader, mode, shapes):
    staged = os.path.join(CUDA_DIR, STAGED_NAME)
    try:
        shutil.copy2(loader, staged)
        proc = subprocess.run(
            [PY, "-c", PROBE, staged, HERE, mode, json.dumps(shapes)],
            capture_output=True, text=True, timeout=900)
    finally:
        if os.path.exists(staged):
            os.remove(staged)
    line = [ln for ln in (proc.stdout or "").splitlines() if ln.startswith("{")]
    if not line:
        tail = (proc.stderr or "").strip().splitlines()[-3:]
        return {"error": " | ".join(tail) if tail else "no output"}
    return json.loads(line[-1])


def err_for(data, shape):
    for row in data.get("rows", []):
        if row["shape"] == list(shape):
            return row["max_err"], row
    return float("nan"), {}


def main():
    legacy = probe(LEGACY, "shipped", SHAPES)
    forced = probe(LEGACY, "forced", SHAPES)
    current = probe(CURRENT, "shipped", SHAPES)

    for label, data in (("legacy shipped", legacy), ("legacy forced", forced),
                        ("current", current)):
        if "error" in data:
            print("%-16s ERROR %s" % (label, data["error"]))
            return 1
        print("%-16s kernels=%d" % (label, data["kernels"]))

    print("\n%-10s %-12s %-12s %-10s %-10s %-10s"
          % ("shape", "legacy grid", "want", "legacy err", "forced err",
             "current err"))
    for shape in SHAPES:
        _, lrow = err_for(legacy, shape)
        _, frow = err_for(forced, shape)
        _, crow = err_for(current, shape)
        print("%-10s %-12s %-12s %-10.2e %-10.2e %-10.2e"
              % ("%dx%d" % shape, lrow.get("grid"), lrow.get("want"),
                 lrow.get("max_err", float("nan")),
                 frow.get("max_err", float("nan")),
                 crow.get("max_err", float("nan"))))

    cases = []
    for shape in KEY_SHAPES:
        e_legacy, _ = err_for(legacy, shape)
        e_forced, frow = err_for(forced, shape)
        e_cur, crow = err_for(current, shape)
        cases.append(("legacy as shipped wrong on %dx%d" % shape, e_legacy > 1e-2,
                      "max_err=%.3e" % e_legacy))
        cases.append(("same kernel, forced grid exact on %dx%d" % shape,
                      e_forced <= TOL, "max_err=%.3e grid=%s"
                      % (e_forced, frow.get("grid"))))
        cases.append(("current loader exact on %dx%d" % shape, e_cur <= TOL,
                      "max_err=%.3e grid=%s" % (e_cur, crow.get("grid"))))

    cases.append(("all runs used the same kernel binaries",
                  legacy.get("kernels") == forced.get("kernels") == current.get("kernels"),
                  "kernels=%s/%s/%s" % (legacy.get("kernels"), forced.get("kernels"),
                                        current.get("kernels"))))

    bad_cur = [r for r in current.get("rows", [])
               if r["max_err"] > TOL or r["grid"] != r["want"] or not r["finite"]]
    cases.append(("current: exact + ceil-div grid on all %d shapes" % len(SHAPES),
                  not bad_cur, "%d bad: %s" % (len(bad_cur),
                                               [r["shape"] for r in bad_cur][:4])))

    bad_forced = [r for r in forced.get("rows", []) if r["max_err"] > TOL]
    cases.append(("forced grid exact on all %d shapes (kernel is shape-safe)"
                  % len(SHAPES), not bad_forced,
                  "%d off: %s" % (len(bad_forced),
                                  [r["shape"] for r in bad_forced][:4])))

    forced_map = {tuple(r["shape"]): r.get("digest") for r in forced.get("rows", [])}
    current_map = {tuple(r["shape"]): r.get("digest") for r in current.get("rows", [])}
    differ = [s for s in forced_map if forced_map[s] != current_map.get(s)]
    cases.append(("forced legacy output is bit-identical to current output",
                  not differ, "%d/%d shapes differ" % (len(differ), len(forced_map))))

    for name, ok, detail in cases:
        print("[%s] %-52s %s" % ("PASS" if ok else "FAIL", name, detail))
    bad = [c for c in cases if not c[1]]
    print("\n%d checks: %d pass, %d fail"
          % (len(cases), len(cases) - len(bad), len(bad)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
