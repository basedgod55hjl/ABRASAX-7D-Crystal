"""Aggregate verification for the ABRASAX-7D-Crystal tree.

One command -> every gate, in order, PASS/FAIL table, nonzero exit on any failure:

    0. manifest       verify_manifest.py            SHA-256 of every claimed artifact
    1. self_test      _self_test.py                 gates go red when they should
    2. compile        _compile_gate.py              per-file compile, this dir
    3. compile_tree   _tree_gate.py <root>          every .py in the tree
    4. static         _static_audit.py              imports/dups/pyc/bare-except
    5. imports        _import_matrix.py             resolution + severity (core FATAL)
    6. phantom        _phantom_imports.py           misqualified `from pkg.mod import`
    7. shadow         _check_loader_shadow.py       stub vs real cuda_kernel_loader
    8. import_ctl     _import_controls.py           controls for stages 5-6
    9. offline_ctl    _offline_controls.py          sandbox + manifest controls
   10. pipeline_fix   _check_phantom_fix.py         abrasax_pipeline importable
   11. grid_regress   diagnose_quantum_grid.py      5 shapes exact AND grid == ceil-div
   12. grid_forensic  diagnose_quantum_grid.py --legacy
   13. grid_stress    _grid_stress.py              15 shapes, kernel-vs-geometry
   14. grid_history   _check_grid_bug_history.py   pre-fix vs current, columns 32..63
   15. gitignore      verify_gitignore.py
   16. kernel_math    verify_kernel_math.py
   17. mhq7d_quantum  verify_kernel_math_quantum.py
   18. loader         kernels/cuda/test_kernel_loader.py --quick
   19. offline        verify_offline.py --fast     every static gate, network blocked
   19b vacuity       _vacuity_audit.py       every gate proves it ran, not just that it passed
   20. encoding       strict UTF-8 decode of every .py in the tree

Usage: python tools/verify_all.py [--fast]
       --fast          skip the runtime/CUDA stages (15-18), keep the static half
       --no-selftest   skip stage 1 (the self test's aggregator control uses it)
"""
from __future__ import annotations

import os
import subprocess
import sys

TOOLS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS)
PY = sys.executable
SKIP_DIRS = {".venv", "venv", "__pycache__", "node_modules", ".git", "target", "build"}


def run(name, argv, cwd, timeout, expect, skip):
    if skip:
        return ("SKIP", name, "fast mode")
    try:
        proc = subprocess.run(argv, cwd=cwd, capture_output=True, text=True,
                              timeout=timeout)
    except subprocess.TimeoutExpired:
        return ("FAIL", name, "timeout after %ss" % timeout)
    out = (proc.stdout or "") + (proc.stderr or "")
    tail = [ln.strip() for ln in out.splitlines() if ln.strip()]
    summary = tail[-1] if tail else "<no output>"
    detail = ""
    for ln in tail:
        if expect in ln:
            detail = ln  # last match wins: the summary is the useful line
    ok = proc.returncode == 0 and (expect in out if expect else True)
    return ("PASS" if ok else "FAIL", name, (detail or summary)[:110])


def encoding_scan():
    """Strict UTF-8 decode: a .py that is not valid UTF-8 needs a coding cookie."""
    bad = []
    n = 0
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            n += 1
            path = os.path.join(dirpath, fn)
            raw = open(path, "rb").read()
            if raw.startswith(b"\xef\xbb\xbf"):
                bad.append((os.path.relpath(path, ROOT), "UTF-8 BOM"))
                continue
            try:
                raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                bad.append((os.path.relpath(path, ROOT), str(exc)))
    return n, bad


def main():
    fast = "--fast" in sys.argv
    skip_self = "--no-selftest" in sys.argv
    T = TOOLS
    py = PY
    stages = [
        ("manifest", [py, os.path.join(T, "verify_manifest.py")], ROOT, 300,
         "changed=0 missing=0 extra=0", False),
        ("self_test", [py, os.path.join(T, "_self_test.py")], T, 300, ", 0 fail",
         skip_self),
        ("compile", [py, os.path.join(T, "_compile_gate.py")], T, 120,
         "failures=0", False),
        ("compile_tree", [py, os.path.join(T, "_tree_gate.py"), ROOT], T, 180,
         "failures=0", False),
        ("static", [py, os.path.join(T, "_static_audit.py")], T, 120, "fatal=0",
         False),
        ("imports", [py, os.path.join(T, "_import_matrix.py")], ROOT, 180,
         "core_fatal=0", False),
        ("phantom", [py, os.path.join(T, "_phantom_imports.py")], ROOT, 180,
         "phantom=0", False),
        ("shadow", [py, os.path.join(T, "_check_loader_shadow.py")], ROOT, 240,
         ", 0 fail", False),
        ("import_ctl", [py, os.path.join(T, "_import_controls.py")], ROOT, 300,
         ", 0 fail", False),
        ("offline_ctl", [py, os.path.join(T, "_offline_controls.py")], ROOT, 300,
         ", 0 fail", False),
        ("pipeline_fix", [py, os.path.join(T, "_check_phantom_fix.py")], ROOT, 120,
         ", 0 fail", False),
        ("grid_regress", [py, os.path.join(T, "diagnose_quantum_grid.py")], ROOT, 300,
         "0 shape(s) failed", fast),
        ("grid_forensic", [py, os.path.join(T, "diagnose_quantum_grid.py"),
                           "--legacy"], ROOT, 300,
         "CONFIRMED: pre-fix loader reproduces", fast),
        ("grid_stress", [py, os.path.join(T, "_grid_stress.py")], ROOT, 600,
         ", 0 fail", fast),
        ("grid_history", [py, os.path.join(T, "_check_grid_bug_history.py")], ROOT,
         300, ", 0 fail", fast),
        ("gitignore_ctl", [py, os.path.join(T, "_audit_verify_gitignore.py")], ROOT,
         300, ", 0 fail", False),
        ("gitignore", [py, os.path.join(T, "verify_gitignore.py")], ROOT, 300,
         "0 misconfigured", fast),
        ("loader_audit", [py, os.path.join(T, "_audit_loader_location.py")], ROOT,
         600, "0 fail", False),
        ("gpu_deps", [py, os.path.join(T, "_loader_requirements.py")], ROOT, 400,
         ", 0 fail", False),
        ("kernel_math", [py, os.path.join(T, "verify_kernel_math.py")], ROOT, 300,
         "0 failed", fast),
        ("mhq7d_quantum", [py, os.path.join(T, "verify_kernel_math_quantum.py")],
         ROOT, 300, "0 failed", fast),
        ("loader", [py, os.path.join(ROOT, "kernels", "cuda", "test_kernel_loader.py"),
                    "--quick"], ROOT, 420, "ALL TESTS PASSED", fast),
        ("vacuity", [py, os.path.join(T, "_vacuity_audit.py")], ROOT, 900,
         ", 0 failed", False),
        ("offline", [py, os.path.join(T, "verify_offline.py"), "--fast"], ROOT, 600,
         ", 0 fail", False),
    ]

    rows = []
    for name, argv, cwd, timeout, expect, skip in stages:
        state, nm, detail = run(name, argv, cwd, timeout, expect, skip)
        rows.append((state, nm, detail))
        print("[%-4s] %-15s %s" % (state, nm, detail))

    nfiles, bad = encoding_scan()
    state = "PASS" if not bad else "FAIL"
    detail = ("%d .py files valid UTF-8, no BOM" % nfiles) if not bad else bad[0][1]
    rows.append((state, "encoding", detail))
    print("[%-4s] %-15s %s" % (state, "encoding", detail))
    for rel, why in bad:
        print("        %s  %s" % (rel, why))

    failed = [r for r in rows if r[0] == "FAIL"]
    skipped = [r for r in rows if r[0] == "SKIP"]
    print("\n%d stages: %d pass, %d fail, %d skip"
          % (len(rows), len(rows) - len(failed) - len(skipped), len(failed),
             len(skipped)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
