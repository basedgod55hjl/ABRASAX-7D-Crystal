"""Remove the scratch dirs left behind by the gates (and by one runaway run).

Only touches temp dirs whose names start with one of the gate prefixes.
Prints what it removed; refuses to touch anything outside %TEMP%.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile

PREFIXES = ("negctl", "cgate_", "treegate_")


def main():
    tmp = os.path.abspath(tempfile.gettempdir())
    removed = failed = 0
    for name in os.listdir(tmp):
        if not name.startswith(PREFIXES):
            continue
        path = os.path.join(tmp, name)
        if os.path.dirname(os.path.abspath(path)) != tmp:
            print("REFUSE %s" % path)
            continue
        if not os.path.isdir(path):
            continue
        try:
            shutil.rmtree(path)
            removed += 1
        except OSError as exc:
            failed += 1
            if failed <= 5:
                print("FAILED  %s  %s" % (name, exc))
    left = sum(1 for n in os.listdir(tmp) if n.startswith(PREFIXES))
    print("removed=%d failed=%d remaining=%d" % (removed, failed, left))
    return 0 if left == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
