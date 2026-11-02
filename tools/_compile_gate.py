"""Compile gate: hard-fails if any .py in this dir does not compile.

Per file:
  1. source -> compile() to a code object (full compile, not just a parse)
  2. py_compile emits real bytecode to a scratch dir, so stale .pyc files
     cannot mask a broken source
Also prints the source hash so a run can be quoted in a report.
Exit 0 only if every file compiles. Scratch dir is removed on the way out.
"""
from __future__ import annotations

import hashlib
import os
import py_compile
import shutil
import sys
import tempfile
import traceback


def gate(directory):
    files = sorted(f for f in os.listdir(directory) if f.endswith(".py"))
    scratch = tempfile.mkdtemp(prefix="cgate_")
    failures = []
    try:
        for name in files:
            path = os.path.join(directory, name)
            try:
                with open(path, "rb") as fh:
                    src = fh.read()
                code = compile(src, name, "exec", dont_inherit=True)
                py_compile.compile(path, cfile=os.path.join(scratch, name + "c"),
                                   doraise=True)
                print("OK   %-32s %6d bytes  %8s  %5d opcodes"
                      % (name, len(src),
                         hashlib.sha256(src).hexdigest()[:12], len(code.co_code)))
            except Exception as exc:
                failures.append(name)
                print("FAIL %-32s %s" % (name, exc))
                traceback.print_exc()
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    print("\nfiles=%d failures=%d" % (len(files), len(failures)))
    return 0 if not failures else 1


def main():
    return gate(os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    sys.exit(main())
