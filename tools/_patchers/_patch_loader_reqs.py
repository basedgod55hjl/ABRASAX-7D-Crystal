"""Tidy _loader_requirements.py: one import probe helper instead of two."""
import io
import os
import sys

TARGET = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "_loader_requirements.py")

OLD_CASE = '''    cases.append(("cupy/torch/pycuda/numba/pyopencl are not in the loader's path",
                  not any(importlib.util.find_spec(n) for n in NOT_REQUIRED
                          if _safe(n)), "see sandbox run below"))'''

NEW_CASE = '''    absent = [n for n in NOT_REQUIRED if not is_present(n)]
    cases.append(("cupy/torch/pycuda/numba/pyopencl are absent here",
                  len(absent) == len(NOT_REQUIRED),
                  "absent: %s" % ", ".join(absent)))'''

OLD_HELPER = '''def _safe(name):
    """find_spec can raise for odd names; treat that as absent."""
    try:
        importlib.util.find_spec(name)
        return True
    except (ImportError, ValueError):
        return False'''

NEW_HELPER = '''def is_present(name):
    """find_spec raises for some names; treat any failure as absent."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False'''


def main():
    src = io.open(TARGET, encoding="utf-8").read()
    if "is_present" in src:
        print("already patched")
        return 0
    for old, new in ((OLD_CASE, NEW_CASE), (OLD_HELPER, NEW_HELPER)):
        if old not in src:
            print("PATTERN NOT FOUND:\n%s" % old[:80])
            return 1
        src = src.replace(old, new, 1)
    # the helper must exist before main() runs; module level is fine either way
    io.open(TARGET, "w", encoding="utf-8", newline="").write(src)
    print("patched %s" % TARGET)
    return 0


if __name__ == "__main__":
    sys.exit(main())
