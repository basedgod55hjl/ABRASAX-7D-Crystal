"""Audit cuda_kernel_loader: where it lives, what it imports, who resolves it.

Four questions, each answered by running something rather than reading a comment.

  A  INVENTORY   every file named cuda_kernel_loader* in the tree, hashed and
                 classified (real loader vs stub) by whether it binds
                 cuLaunchKernel
  B  IMPORTS     AST of the real loader: module-level vs deferred imports,
                 third-party set, and whether anything imports dynamically
                 (importlib / __import__ / exec / eval) where review cannot see
  C  RESOLUTION  run each entry point for real under an atexit recorder and
                 report which cuda_kernel_loader.__file__ it actually binds
  D  SHADOWING   put cuda/ (the stub dir) first on sys.path and show the cost,
                 then prove no repo file ever puts that dir on the path

Exit 0 only if there is exactly one real loader, every entry point resolves to it,
the stub binds nothing, nothing imports dynamically, and no repo code puts the stub
directory on sys.path.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CUDA = os.path.join(ROOT, "kernels", "cuda")
REAL = os.path.join(CUDA, "cuda_kernel_loader.py")
STUB_DIR = os.path.join(ROOT, "cuda")
PY = sys.executable
ENTRY_POINTS = [
    ("tools/verify_kernel_math.py", [], 300),
    ("tools/verify_kernel_math_quantum.py", [], 300),
    ("tools/diagnose_quantum_grid.py", [], 300),
    ("tools/_trace_grid.py", [], 300),
    ("kernels/cuda/test_kernel_loader.py", ["--quick"], 420),
]
DYNAMIC = ("__import__", "importlib", "exec", "eval", "runpy")

RECORDER = r'''
"""Record which cuda_kernel_loader a real run binds."""
import atexit
import json
import os
import sys

_out = os.environ.get("LOADER_AUDIT_OUT")


def _dump():
    if not _out:
        return
    mod = sys.modules.get("cuda_kernel_loader")
    if mod is None:
        payload = {"loaded": False}
    else:
        try:
            count = len(list(mod.CrystalKernelRuntime().available_kernels))
        except Exception:
            count = 0
        payload = {"loaded": True, "file": getattr(mod, "__file__", "?"),
                   "kernels": count}
    with open(_out, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(payload))


atexit.register(_dump)
'''


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def inventory():
    found = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in
                       {"__pycache__", ".venv", "build", "target", ".git"}]
        for fn in filenames:
            if fn.startswith("cuda_kernel_loader") and not fn.endswith(".pyc"):
                found.append(os.path.join(dirpath, fn))
    rows = []
    for path in sorted(found):
        text = open(path, "rb").read().decode("utf-8", "replace")
        binds = "cuLaunchKernel" in text
        rows.append({
            "path": os.path.relpath(path, ROOT).replace("\\", "/"),
            "bytes": os.path.getsize(path),
            "sha16": sha(path)[:16],
            "binds": binds,
            "importable": path.endswith(".py"),
            "kind": "real" if binds else "stub/other",
        })
    return rows


def imports_of(path):
    tree = ast.parse(open(path, "rb").read(), os.path.basename(path))
    module_level, deferred = set(), set()

    def walk(node, depth):
        for child in getattr(node, "body", []):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                walk(child, depth + 1)
                continue
            if isinstance(child, ast.Import):
                (module_level if depth == 0 else deferred).update(
                    a.name.split(".")[0] for a in child.names)
            elif isinstance(child, ast.ImportFrom) and child.module and not child.level:
                (module_level if depth == 0 else deferred).add(
                    child.module.split(".")[0])
            walk(child, depth)

    walk(tree, 0)
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    dynamic = sorted(names & set(DYNAMIC))

    def stdlib(mods):
        return sorted(m for m in mods
                      if m in sys.stdlib_module_names or m == "__future__")

    def third(mods):
        return sorted(m for m in mods
                      if m not in sys.stdlib_module_names and m != "__future__")

    return {
        "module_stdlib": stdlib(module_level),
        "module_third": third(module_level),
        "deferred_stdlib": stdlib(deferred),
        "deferred_third": third(deferred),
        "dynamic_names": dynamic,
    }


def resolution_check():
    """Run each entry point for real; ask the recorder what got imported."""
    sandbox = tempfile.mkdtemp(prefix="loader_audit_")
    out_file = os.path.join(sandbox, "loader.json")
    rows = []
    try:
        with open(os.path.join(sandbox, "sitecustomize.py"), "w",
                  encoding="utf-8") as fh:
            fh.write(RECORDER)
        env = dict(os.environ)
        env["PYTHONPATH"] = sandbox
        env["LOADER_AUDIT_OUT"] = out_file
        for rel, args, timeout in ENTRY_POINTS:
            script = os.path.join(ROOT, rel.replace("/", os.sep))
            if not os.path.isfile(script):
                rows.append((rel, None, None))
                continue
            if os.path.exists(out_file):
                os.remove(out_file)
            subprocess.run([PY, script] + args, cwd=ROOT, capture_output=True,
                           text=True, timeout=timeout, env=env)
            data = {}
            if os.path.isfile(out_file):
                with open(out_file, encoding="utf-8") as fh:
                    data = json.load(fh)
            rows.append((rel, data.get("file"), data.get("kernels")))
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)
    return rows


def shadow_check():
    probe = ("import sys;sys.path.insert(0, sys.argv[1]);"
             "import cuda_kernel_loader as m;"
             "rt=m.CrystalKernelRuntime();"
             "print(m.__file__);"
             "print(len(list(getattr(rt, 'available_kernels', []) or [])))")
    proc = subprocess.run([PY, "-c", probe, STUB_DIR], cwd=ROOT,
                          capture_output=True, text=True, timeout=180)
    lines = [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]
    path = lines[0] if lines else "?"
    count = int(lines[1]) if len(lines) > 1 and lines[1].isdigit() else -1
    return path, count


def stub_dir_references():
    """Does any repo file put cuda/ (the stub dir) on sys.path?"""
    hits = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in
                       {"__pycache__", ".venv", "build", "target", ".git"}]
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dirpath, fn)
            text = open(path, "rb").read().decode("utf-8", "replace")
            for line_no, line in enumerate(text.splitlines(), 1):
                if "sys.path" not in line and "PYTHONPATH" not in line:
                    continue
                code = line.split("#")[0]
                if '"cuda"' in code or "'cuda'" in code:
                    hits.append("%s:%d" % (os.path.relpath(path, ROOT), line_no))
    return hits


def main():
    cases = []

    print("== A. inventory ==")
    rows = inventory()
    for row in rows:
        print("  %-42s %7d B  %s  %-10s cuLaunchKernel=%s"
              % (row["path"], row["bytes"], row["sha16"], row["kind"],
                 row["binds"]))
    real = [r for r in rows if r["kind"] == "real" and r["importable"]]
    cases.append(("exactly one real loader", len(real) == 1,
                  "%d real, %d other" % (len(real), len(rows) - len(real))))
    cases.append(("the real loader lives in kernels/cuda/",
                  len(real) == 1
                  and real[0]["path"] == "kernels/cuda/cuda_kernel_loader.py",
                  real[0]["path"] if real else "<none>"))

    print("\n== B. imports (real loader) ==")
    info = imports_of(REAL)
    print("  module-level stdlib : %s" % ", ".join(info["module_stdlib"]))
    print("  module-level third  : %s" % (", ".join(info["module_third"]) or "none"))
    print("  deferred stdlib     : %s" % (", ".join(info["deferred_stdlib"]) or "none"))
    print("  deferred third      : %s" % (", ".join(info["deferred_third"]) or "none"))
    print("  dynamic mechanisms  : %s" % (", ".join(info["dynamic_names"]) or "none"))
    third = set(info["module_third"]) | set(info["deferred_third"])
    cases.append(("third-party set is numpy only", third <= {"numpy"},
                  "third=%s" % (sorted(third) or "none")))
    cases.append(("no dynamic import mechanisms", not info["dynamic_names"],
                  "dynamic=%s" % (info["dynamic_names"] or "none")))

    print("\n== C. resolution per entry point (recorded from real runs) ==")
    for rel, resolved, kernels in resolution_check():
        short = (os.path.relpath(resolved, ROOT)
                 if resolved and resolved != "<not imported>" else resolved)
        print("  %-38s -> %-40s kernels=%s" % (rel, short, kernels))
        cases.append(("%s binds the real loader" % os.path.basename(rel),
                      bool(resolved)
                      and os.path.normcase(resolved) == os.path.normcase(REAL),
                      short or "<not imported>"))

    print("\n== D. shadowing ==")
    stub_path, stub_count = shadow_check()
    print("  cuda/ first on sys.path -> %s (kernels=%d)"
          % (os.path.relpath(stub_path, ROOT) if stub_path not in ("?", "")
             else "?", stub_count))
    cases.append(("the stub dir really does shadow the loader", stub_count == 0,
                  "kernels=%d" % stub_count))
    refs = stub_dir_references()
    print("  repo files putting cuda/ on sys.path: %s" % (", ".join(refs) or "none"))
    cases.append(("no repo file puts the stub dir on sys.path", not refs,
                  "%d references" % len(refs)))

    for name, ok, detail in cases:
        print("[%s] %-52s %s" % ("PASS" if ok else "FAIL", name, detail))
    bad = [c for c in cases if not c[1]]
    print("\n%d checks: %d pass, %d fail"
          % (len(cases), len(cases) - len(bad), len(bad)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
