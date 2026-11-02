"""Negative controls for the gates: prove each gate goes red for the RIGHT reason.

A gate that only ever prints OK is worthless, and a gate that fails for an
unrelated reason is worse than useless. Each case injects exactly one defect
into a throwaway copy of this dir and asserts the matching diagnostic line
appears in the output AND the exit code is nonzero.

Cases:
  baseline      unmodified copy                -> compile gate exit 0
  syntax        append `def broken(:`          -> "FAIL <file>" + nonzero
  missing_mod   import nonexistent_mod_xyz     -> "UNRESOLVED IMPORT nonexistent_mod_xyz"
  dup_def       duplicate `def main()`         -> "DUPLICATE DEF main"
  stale_pyc     old .pyc beside an edited .py  -> "PYC source newer than pyc"
  aggregator    broken file in a copied tree   -> verify_all nonzero + [FAIL] compile_tree

The aggregator case invokes verify_all with --fast --no-selftest and also sets
NEGCTL_DEPTH: without both, a copied tree would re-run this script and fork
without bound (learned the hard way -- 1588 scratch dirs, 200 processes).
"""
from __future__ import annotations

import os
import py_compile
import shutil
import subprocess
import sys
import tempfile

TOOLS = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
GATES = ("_compile_gate.py", "_tree_gate.py", "_static_audit.py", "verify_all.py",
         "_clean_scratch.py")
VICTIM = "verify_gitignore.py"
DEPTH = int(os.environ.get("NEGCTL_DEPTH", "0"))


def sources():
    return tuple(sorted(f for f in os.listdir(TOOLS)
                        if f.endswith(".py") and not f.startswith("_")
                        and f not in GATES))


def build_copy():
    dst = tempfile.mkdtemp(prefix="negctl_")
    for name in GATES + sources():
        src = os.path.join(TOOLS, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst, name))
    return dst


def run_gate(dst, gate, args=()):
    proc = subprocess.run([PY, os.path.join(dst, gate), *args], cwd=dst,
                          capture_output=True, text=True, timeout=300)
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def last_summary(out, token="files="):
    key = [ln for ln in out.splitlines() if token in ln]
    return key[-1].strip() if key else "<no summary>"


def inject(path, text):
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(text)


def main():
    cases = []
    made = []

    dst = build_copy()
    made.append(dst)
    rc, out = run_gate(dst, "_compile_gate.py")
    cases.append(("baseline compile clean", rc == 0, last_summary(out)))

    dst = build_copy()
    made.append(dst)
    inject(os.path.join(dst, VICTIM), "\n\ndef broken(:\n    pass\n")
    rc, out = run_gate(dst, "_compile_gate.py")
    cases.append(("syntax error -> red",
                  rc != 0 and ("FAIL " + VICTIM) in out, last_summary(out)))

    dst = build_copy()
    made.append(dst)
    inject(os.path.join(dst, VICTIM), "\nimport nonexistent_mod_xyz  # injected\n")
    rc, out = run_gate(dst, "_static_audit.py")
    cases.append(("missing import -> red",
                  rc != 0 and "UNRESOLVED IMPORT nonexistent_mod_xyz" in out,
                  last_summary(out)))

    dst = build_copy()
    made.append(dst)
    inject(os.path.join(dst, VICTIM), "\n\ndef main():\n    return 0\n")
    rc, out = run_gate(dst, "_static_audit.py")
    cases.append(("duplicate def -> red",
                  rc != 0 and "DUPLICATE DEF main" in out, last_summary(out)))

    dst = build_copy()
    made.append(dst)
    victim = os.path.join(dst, VICTIM)
    pyc = os.path.join(dst, "__pycache__", VICTIM[:-3] + ".cpython-311.pyc")
    py_compile.compile(victim, cfile=pyc, doraise=True)
    inject(victim, "\n# edited after the pyc was written\n")
    rc, out = run_gate(dst, "_static_audit.py")
    cases.append(("stale pyc -> red",
                  rc != 0 and "PYC source newer than pyc" in out, last_summary(out)))

    if DEPTH > 0:
        cases.append(("aggregator -> skipped (nested)", True,
                      "NEGCTL_DEPTH=%d" % DEPTH))
    else:
        proj = tempfile.mkdtemp(prefix="negctl_proj_")
        made.append(proj)
        shutil.copytree(TOOLS, os.path.join(proj, "tools"),
                        ignore=shutil.ignore_patterns("__pycache__"))
        inject(os.path.join(proj, "broken_probe.py"), "def nope(:\n")
        env = dict(os.environ, NEGCTL_DEPTH="1")
        proc = subprocess.run(
            [PY, os.path.join(proj, "tools", "verify_all.py"), "--fast",
             "--no-selftest"],
            cwd=proj, capture_output=True, text=True, timeout=300, env=env)
        out = proc.stdout or ""
        cases.append(("aggregator -> red",
                      proc.returncode != 0 and "[FAIL] compile_tree" in out,
                      "%s (expected: the probe tree is broken on purpose)"
                      % last_summary(out, "stages:")))

    for name, ok, line in cases:
        print("[%s] %-26s %s" % ("PASS" if ok else "FAIL", name, line))
    bad = [c for c in cases if not c[1]]
    print("\n%d controls: %d pass, %d fail"
          % (len(cases), len(cases) - len(bad), len(bad)))
    for d in made:
        shutil.rmtree(d, ignore_errors=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
