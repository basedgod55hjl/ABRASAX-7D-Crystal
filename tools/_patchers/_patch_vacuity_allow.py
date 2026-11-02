"""Add two allowances to _vacuity_audit.py's skip-line filter."""
import io
import os
import sys

TARGET = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "_vacuity_audit.py")

ANCHOR = '    re.compile(r"absent on disk\\]"),             # per-path annotation\n'
ADD = ('    re.compile(r"absent on disk\\]"),             # per-path annotation\n'
       '    re.compile(r"^\\s*(cupy|torch|pycuda|numba|pyopencl)\\s+absent"),  '
       '# section D table\n'
       '    re.compile(r"pyc=absent"),                   # no bytecode yet, not a skip\n')


def main():
    src = io.open(TARGET, encoding="utf-8", newline="").read()
    if "section D table" in src:
        print("already patched")
        return 0
    probe = ANCHOR.replace("\n", "\r\n") if "\r\n" in src else ANCHOR
    add = ADD.replace("\n", "\r\n") if "\r\n" in src else ADD
    if probe not in src:
        print("ANCHOR NOT FOUND")
        return 1
    io.open(TARGET, "w", encoding="utf-8", newline="").write(src.replace(probe, add, 1))
    print("patched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
