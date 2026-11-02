"""Offline verification: everything that can be checked with no network, ever.

Two ideas:

  1. CAPABILITY MATRIX. Say up front what this box can and cannot prove here --
     numpy? CUDA device? git? nvcc? A check that silently skips is worse than a
     check that fails, so every skip is named and counted.

  2. NETWORK SANDBOX. Every stage runs in a subprocess whose sitecustomize
     replaces socket.connect / create_connection / urlopen with a raiser. If a
     gate ever reached for the network it would fail loudly instead of hanging.
     A probe stage proves the sandbox is actually armed (it must see our
     RuntimeError, not a refused connection).

Usage:
    python tools/verify_offline.py            run everything the box supports
    python tools/verify_offline.py --fast     skip GPU stages
    python tools/verify_offline.py --matrix   print only the capability matrix

Exit 0 only if every executed stage passed and none failed.
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable

SANDBOX = r'''
"""Armed by tools/verify_offline.py: no network in this process tree."""
import socket

_MARK = "offline verification: network access attempted"


class NetworkBlocked(RuntimeError):
    pass


def _deny(*args, **kwargs):
    raise NetworkBlocked(_MARK)


socket.socket.connect = _deny
socket.socket.connect_ex = _deny
socket.create_connection = _deny
try:
    import urllib.request
    urllib.request.urlopen = _deny
except Exception:
    pass
'''

PROBE = r'''
import socket, sys
sys.stderr.write("")
try:
    socket.create_connection(("127.0.0.1", 9), 0.2)
    print("SANDBOX_NOT_ARMED: connection succeeded")
except RuntimeError as exc:
    print("SANDBOX_ARMED" if "offline verification" in str(exc) else
          "SANDBOX_UNKNOWN: %s" % exc)
except Exception as exc:
    print("SANDBOX_NOT_ARMED: %s" % type(exc).__name__)
'''


def sandbox_dir():
    tmp = tempfile.mkdtemp(prefix="offline_sandbox_")
    with open(os.path.join(tmp, "sitecustomize.py"), "w", encoding="utf-8") as fh:
        fh.write(SANDBOX)
    return tmp


def run(argv, cwd, timeout, sandbox):
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = sandbox + (os.pathsep + existing if existing else "")
    try:
        proc = subprocess.run(argv, cwd=cwd, capture_output=True, text=True,
                              timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return 1, "timeout after %ss" % timeout
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, out


def capabilities():
    caps = {}

    proc = subprocess.run([PY, "-c", "import sys;print(sys.version.split()[0])"],
                          capture_output=True, text=True)
    caps["python"] = (proc.stdout.strip() or "?", True)

    try:
        spec = importlib.util.find_spec("numpy")
    except (ImportError, ValueError):
        spec = None
    if spec is None:
        caps["numpy"] = ("absent", False)
    else:
        proc = subprocess.run([PY, "-c", "import numpy;print(numpy.__version__)"],
                              capture_output=True, text=True)
        caps["numpy"] = (proc.stdout.strip() or "present", True)

    caps["git"] = (shutil.which("git") or "absent", bool(shutil.which("git")))
    caps["nvcc"] = (shutil.which("nvcc") or "absent", bool(shutil.which("nvcc")))

    probe = ("import sys;sys.path.insert(0, r'%s');"
             "from cuda_kernel_loader import CrystalKernelRuntime as R;"
             "rt=R();print(len(rt.available_kernels))"
             % os.path.join(ROOT, "kernels", "cuda"))
    proc = subprocess.run([PY, "-c", probe], capture_output=True, text=True,
                          timeout=180)
    if proc.returncode == 0 and proc.stdout.strip().isdigit():
        n = int(proc.stdout.strip())
        caps["cuda"] = ("%d kernels bound" % n, n > 0)
    else:
        caps["cuda"] = ("unavailable", False)
    return caps


def stages(caps, fast):
    T = HERE
    gpu = caps["cuda"][1] and not fast
    has_git = caps["git"][1]
    table = [
        ("manifest", [PY, os.path.join(T, "verify_manifest.py")], ROOT, 180,
         "changed=0 missing=0 extra=0", True),
        ("compile", [PY, os.path.join(T, "_compile_gate.py")], T, 120,
         "failures=0", True),
        ("compile_tree", [PY, os.path.join(T, "_tree_gate.py"), ROOT], T, 180,
         "failures=0", True),
        ("static", [PY, os.path.join(T, "_static_audit.py")], T, 120, "fatal=0",
         True),
        ("imports", [PY, os.path.join(T, "_import_matrix.py")], ROOT, 180,
         "core_fatal=0", True),
        ("phantom", [PY, os.path.join(T, "_phantom_imports.py")], ROOT, 180,
         "phantom=0", True),
        ("gitignore", [PY, os.path.join(T, "verify_gitignore.py")], ROOT, 300,
         "0 misconfigured", has_git),
        ("grid_regress", [PY, os.path.join(T, "diagnose_quantum_grid.py")], ROOT, 300,
         "0 shape(s) failed", gpu),
        ("kernel_math", [PY, os.path.join(T, "verify_kernel_math.py")], ROOT, 300,
         "0 failed", gpu),
        ("mhq7d_quantum", [PY, os.path.join(T, "verify_kernel_math_quantum.py")],
         ROOT, 300, "0 failed", gpu),
        ("grid_stress", [PY, os.path.join(T, "_grid_stress.py")], ROOT, 600,
         "0 fail", gpu),
        ("loader", [PY, os.path.join(ROOT, "kernels", "cuda",
                                     "test_kernel_loader.py"), "--quick"],
         ROOT, 420, "ALL TESTS PASSED", gpu),
        ("shadow", [PY, os.path.join(T, "_check_loader_shadow.py")], ROOT, 240,
         "0 fail", gpu),
    ]
    return table


def main():
    caps = capabilities()
    print("== capability matrix ==")
    for name in ("python", "numpy", "cuda", "git", "nvcc"):
        detail, ok = caps[name]
        print("  %-8s %-6s %s" % (name, "yes" if ok else "NO", detail))

    if "--matrix" in sys.argv:
        return 0

    sandbox = sandbox_dir()
    try:
        rc, out = run([PY, "-c", PROBE], ROOT, 60, sandbox)
        armed = "SANDBOX_ARMED" in out
        print("\n== sandbox ==")
        print("  %s  %s" % ("ARMED" if armed else "NOT ARMED",
                            out.strip().splitlines()[-1] if out.strip() else "no output"))
        if not armed:
            print("  refusing to report a green run without a working sandbox")
            return 1

        fast = "--fast" in sys.argv
        rows = []
        print("\n== stages (offline, network blocked) ==")
        for name, argv, cwd, timeout, expect, enabled in stages(caps, fast):
            if not enabled:
                reason = "fast mode" if fast else "missing capability"
                rows.append(("SKIP", name, reason))
                print("  [SKIP] %-14s %s" % (name, reason))
                continue
            rc, out = run(argv, cwd, timeout, sandbox)
            lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
            detail = ""
            for ln in lines:
                if expect in ln:
                    detail = ln
                    break
            blocked = "NetworkBlocked" in out or "network access attempted" in out
            ok = rc == 0 and expect in out and not blocked
            if blocked:
                detail = "NETWORK ATTEMPT: " + detail
            rows.append(("PASS" if ok else "FAIL", name, detail or (lines[-1] if lines else "")))
            print("  [%-4s] %-14s %s" % (rows[-1][0], name, rows[-1][2][:96]))
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)

    ran = [r for r in rows if r[0] != "SKIP"]
    failed = [r for r in rows if r[0] == "FAIL"]
    skipped = [r for r in rows if r[0] == "SKIP"]
    print("\n%d stages: %d executed (%d pass, %d fail), %d skipped"
          % (len(rows), len(ran), len(ran) - len(failed), len(failed), len(skipped)))
    if skipped:
        print("  skipped: %s" % ", ".join("%s (%s)" % (r[1], r[2]) for r in skipped))
        print("  a green here does NOT cover the skipped stages")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
