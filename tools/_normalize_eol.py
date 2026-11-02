"""Normalize line endings to LF for every .py in this dir, then report counts.

PowerShell string surgery inserts CRLF into otherwise-LF files; git then sees a
phantom diff and py_compile sees a stale .pyc. Run this, then
`python -m compileall -f .` to refresh the bytecode.
"""
from __future__ import annotations

import os
import sys


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    files = sorted(f for f in os.listdir(here) if f.endswith(".py"))
    changed = 0
    for name in files:
        path = os.path.join(here, name)
        raw = open(path, "rb").read()
        crlf = raw.count(b"\r\n")
        lone_cr = raw.count(b"\r") - crlf
        fixed = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        if fixed and not fixed.endswith(b"\n"):
            fixed += b"\n"
        if fixed != raw:
            with open(path, "wb") as fh:
                fh.write(fixed)
            changed += 1
            print("%-32s crlf=%-4d lone_cr=%-3d normalized" % (name, crlf, lone_cr))
    print("\nfiles=%d normalized=%d" % (len(files), changed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
