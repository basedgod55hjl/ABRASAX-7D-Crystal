"""Why does the mirrored .gitignore behave differently in a fresh repo?

In the real repo all 17 must-be-tracked paths come back 'ok'. In a throwaway repo
with the same two .gitignore files copied in, 12 of them come back IGNORED. Either
the harness is missing something (those paths do not exist there) or git's
directory-exclusion rules genuinely depend on the working tree.

This prints `git check-ignore -v` for one path in four states:
    A  path + parents absent
    B  parent directories created, file absent
    C  file created (empty)
    D  file created and staged in the index
so the answer is measured instead of guessed.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(ROOT))
SUB = "SYSTEMS/ABRASAX-7D-Crystal"
PROBE = "kernels/rust/Cargo.toml"


def sh(args, cwd):
    p = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def build():
    tmp = tempfile.mkdtemp(prefix="why_")
    sh(["git", "init", "-q", tmp], tmp)
    sub = os.path.join(tmp, SUB.replace("/", os.sep))
    os.makedirs(sub)
    shutil.copy2(os.path.join(REPO, ".gitignore"), os.path.join(tmp, ".gitignore"))
    child = os.path.join(REPO, SUB, ".gitignore")
    if os.path.isfile(child):
        shutil.copy2(child, os.path.join(sub, ".gitignore"))
    return tmp, sub


def report(label, sub, rc, out):
    rule = out.strip().splitlines()[0] if out.strip() else "<no rule>"
    state = "IGNORED" if rc == 0 else "not ignored"
    print("  %-34s %-12s %s" % (label, state, rule))


def main():
    tmp, sub = build()
    try:
        rc, out = sh(["git", "check-ignore", "-v", PROBE], sub)
        report("A path+parents absent", sub, rc, out)

        target = os.path.join(sub, PROBE.replace("/", os.sep))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        rc, out = sh(["git", "check-ignore", "-v", PROBE], sub)
        report("B parents created", sub, rc, out)

        open(target, "w", encoding="utf-8").write("x\n")
        rc, out = sh(["git", "check-ignore", "-v", PROBE], sub)
        report("C file created", sub, rc, out)

        sh(["git", "add", "-f", PROBE], sub)
        rc, out = sh(["git", "check-ignore", "-v", PROBE], sub)
        report("D file staged in the index", sub, rc, out)

        print("\nsame file, real repo:")
        rc, out = sh(["git", "check-ignore", "-v", PROBE], ROOT)
        report("real repo", ROOT, rc, out)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
