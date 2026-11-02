"""Refresh the bytecode of an edited tool so _static_audit.py does not read a
stale .pyc as a fatal defect.

The static audit deliberately treats "source newer than pyc" as red, and there
is a negative control that proves it does. What it does not do is regenerate the
pyc, because running a file as __main__ never writes one. So every edit to a
tool leaves the tree one step from red until something imports it. This closes
that loop from the patcher side instead of weakening the gate.

    python tools/_patchers/_refresh_pyc.py verify_all.py [...]
"""
from __future__ import annotations

import os
import py_compile
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.dirname(HERE)


def main() -> int:
    names = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not names:
        print("usage: _refresh_pyc.py <tool.py> [...]")
        return 2
    rc = 0
    for name in names:
        src = os.path.join(TOOLS, name if name.endswith(".py") else name + ".py")
        if not os.path.isfile(src):
            print("MISSING %s" % src)
            rc = 1
            continue
        try:
            py_compile.compile(src, doraise=True)
        except py_compile.PyCompileError as exc:
            print("FAIL %s %s" % (name, exc))
            rc = 1
            continue
        print("refreshed %s" % os.path.relpath(src, TOOLS))
    return rc


if __name__ == "__main__":
    sys.exit(main())
