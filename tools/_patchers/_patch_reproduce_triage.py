"""Append the exit-3 triage note to REPRODUCE.md (idempotent)."""
import io
import os
import sys

TARGET = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "REPRODUCE.md")

OLD = """4. Anything else \u2192 the log file has the full transcript including which control
   failed and with what message."""
NEW = """4. `gitignore` exits 3 \u2192 the path you gave is not the 7D-Crystal subtree (the
   marker `kernels/cuda/crystal_cortex.h` was not found, and no
   `SYSTEMS/ABRASAX-7D-Crystal` underneath it either). It refuses instead of
   reporting a vacuous pass -- pointing it at `C:\\abrasax` used to "succeed"
   while 16 of the 17 committable paths did not exist at that level.
5. Anything else -> the log file has the full transcript including which control
   failed and with what message."""


def main():
    src = io.open(TARGET, encoding="utf-8", newline="").read()
    nl = "\r\n" if "\r\n" in src else "\n"
    old = OLD.replace("\n", nl)
    new = NEW.replace("\n", nl)
    if new in src:
        print("already patched")
        return 0
    if old not in src:
        print("PATTERN NOT FOUND")
        return 1
    io.open(TARGET, "w", encoding="utf-8", newline="").write(src.replace(old, new, 1))
    print("patched REPRODUCE.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
