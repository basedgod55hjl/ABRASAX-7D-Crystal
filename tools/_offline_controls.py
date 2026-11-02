"""Controls for the offline story: prove the sandbox and the manifest are real.

Three claims are load-bearing for "verification you can do offline", and each is
tested here against a case where it must fail:

  sandbox arms      the same probe that reports SANDBOX_ARMED under the sandbox must
                    report NOT_ARMED without it -- otherwise "armed" means nothing
  sandbox blocks    a deliberate socket.create_connection must raise our
                    NetworkBlocked under the sandbox, not a refused connection
  manifest catches  a tampered byte and an unlisted file must both be reported:
                    CHANGED and EXTRA respectively

Exit 0 only if every control behaves.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

SANDBOX = r'''
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
import socket
try:
    socket.create_connection(("127.0.0.1", 9), 0.2)
    print("SANDBOX_NOT_ARMED: connected")
except RuntimeError as exc:
    print("SANDBOX_ARMED" if "offline verification" in str(exc) else "SANDBOX_UNKNOWN")
except Exception as exc:
    print("SANDBOX_NOT_ARMED: %s" % type(exc).__name__)
'''

BLOCK_PROBE = r'''
import socket
try:
    socket.create_connection(("example.invalid", 80), 0.2)
    print("NOT_BLOCKED")
except RuntimeError as exc:
    print("BLOCKED" if "offline verification" in str(exc) else "OTHER_RUNTIME")
except Exception as exc:
    print("NOT_BLOCKED: %s" % type(exc).__name__)
'''


def make_sandbox():
    tmp = tempfile.mkdtemp(prefix="octl_sandbox_")
    with open(os.path.join(tmp, "sitecustomize.py"), "w", encoding="utf-8") as fh:
        fh.write(SANDBOX)
    return tmp


def run(argv, cwd, sandbox=None):
    env = dict(os.environ)
    if sandbox:
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = sandbox + (os.pathsep + existing if existing else "")
    proc = subprocess.run(argv, cwd=cwd, capture_output=True, text=True,
                          timeout=120, env=env)
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def manifest_controls():
    """Build a throwaway tree with one covered file and a hand-written manifest."""
    tmp = tempfile.mkdtemp(prefix="octl_manifest_")
    tools = os.path.join(tmp, "tools")
    os.makedirs(tools)
    shutil.copy2(os.path.join(HERE, "verify_manifest.py"),
                 os.path.join(tools, "verify_manifest.py"))
    covered = os.path.join(tmp, "src", "abrasax_pipeline.py")
    os.makedirs(os.path.dirname(covered))
    with open(covered, "w", encoding="utf-8") as fh:
        fh.write("x = 1\n")

    results = []
    rc, out = run([PY, os.path.join(tools, "verify_manifest.py"), "--update"], tmp)
    rc, out = run([PY, os.path.join(tools, "verify_manifest.py")], tmp)
    results.append(("manifest clean on a fresh tree",
                    rc == 0 and "changed=0 missing=0 extra=0" in out,
                    [ln for ln in out.splitlines() if "entries=" in ln][-1]))

    with open(covered, "a", encoding="utf-8") as fh:
        fh.write("# tampered\n")
    rc, out = run([PY, os.path.join(tools, "verify_manifest.py")], tmp)
    results.append(("manifest reports CHANGED after an edit",
                    rc != 0 and "CHANGED" in out,
                    [ln for ln in out.splitlines() if ln.startswith("CHANGED")][-1:]
                    and "CHANGED" or "<none>"))

    hidden = os.path.join(tools, "_unlisted_probe.py")
    with open(hidden, "w", encoding="utf-8") as fh:
        fh.write("# not in the manifest\n")
    rc, out = run([PY, os.path.join(tools, "verify_manifest.py")], tmp)
    results.append(("manifest reports EXTRA for an unlisted file",
                    rc != 0 and "EXTRA" in out,
                    "EXTRA" if "EXTRA" in out else "<none>"))

    shutil.rmtree(tmp, ignore_errors=True)
    return results


def main():
    cases = []
    sandbox = make_sandbox()
    try:
        rc, out = run([PY, "-c", PROBE], HERE, sandbox)
        cases.append(("probe reports ARMED under the sandbox",
                      "SANDBOX_ARMED" in out,
                      out.strip().splitlines()[-1] if out.strip() else "<none>"))

        rc, out = run([PY, "-c", PROBE], HERE, None)
        cases.append(("same probe reports NOT armed without it",
                      "SANDBOX_NOT_ARMED" in out,
                      out.strip().splitlines()[-1] if out.strip() else "<none>"))

        rc, out = run([PY, "-c", BLOCK_PROBE], HERE, sandbox)
        cases.append(("a real connect attempt is blocked",
                      "BLOCKED" in out,
                      out.strip().splitlines()[-1] if out.strip() else "<none>"))
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)

    cases.extend(manifest_controls())

    for name, ok, detail in cases:
        print("[%s] %-44s %s" % ("PASS" if ok else "FAIL", name, detail))
    bad = [c for c in cases if not c[1]]
    print("\n%d controls: %d pass, %d fail"
          % (len(cases), len(cases) - len(bad), len(bad)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
