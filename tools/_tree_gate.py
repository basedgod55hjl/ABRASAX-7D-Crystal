"""Recursive compile gate over a tree. Skips vendored venvs and build dirs.

Usage: python _tree_gate.py [root]     (defaults to the parent of this dir)
Exit 0 only if every .py under root compiles. Scratch dir is removed on exit.
"""
from __future__ import annotations

import os
import py_compile
import shutil
import sys
import tempfile

SKIP = {".venv", "venv", "__pycache__", "node_modules", ".git", "target", "build"}


def collect(root):
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP]
        for fn in filenames:
            if fn.endswith(".py"):
                out.append(os.path.join(dirpath, fn))
    return sorted(out)


def gate(root):
    files = collect(root)
    scratch = tempfile.mkdtemp(prefix="treegate_")
    bad = []
    try:
        for path in files:
            rel = os.path.relpath(path, root)
            try:
                with open(path, "rb") as fh:
                    src = fh.read()
                compile(src, rel, "exec", dont_inherit=True)
                py_compile.compile(path, cfile=os.path.join(scratch, "x.pyc"),
                                   doraise=True)
            except Exception as exc:
                msg = str(exc).splitlines()[0] if str(exc) else repr(exc)
                bad.append((rel, type(exc).__name__, msg))
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    print("root=%s files=%d failures=%d"
          % (os.path.abspath(root), len(files), len(bad)))
    for rel, kind, msg in bad:
        print("FAIL %-46s %s: %s" % (rel, kind, msg))
    return 0 if not bad else 1


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(here)
    return gate(root)


if __name__ == "__main__":
    sys.exit(main())
