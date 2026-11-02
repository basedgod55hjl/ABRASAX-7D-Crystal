"""Fix three checks in _audit_loader_location.py that failed for my reasons.

1. inventory counted the .bak-gridfix backup as a second "real" loader -- it does
   bind cuLaunchKernel. Only importable .py files count as candidates.
2. therefore the "lives in kernels/cuda/" check saw two reals.
3. the shadow probe called list(rt.available_kernels) on the stub, which does not
   define that attribute, so the probe crashed instead of reporting 0.
"""
import io
import os
import sys

TARGET = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "_audit_loader_location.py")

PAIRS = [
    # 1. mark importability, keep backups visible but out of the candidate set
    ('''            "binds": binds,
            "kind": "real" if binds else "stub/other",''',
     '''            "binds": binds,
            "importable": path.endswith(".py"),
            "kind": "real" if binds else "stub/other",'''),
    ('''    real = [r for r in rows if r["kind"] == "real"]''',
     '''    real = [r for r in rows if r["kind"] == "real" and r["importable"]]'''),
    # 3. tolerate a stub runtime that does not expose available_kernels
    ('''             "rt=m.CrystalKernelRuntime();"
             "print(m.__file__);print(len(list(rt.available_kernels)))")''',
     '''             "rt=m.CrystalKernelRuntime();"
             "print(m.__file__);"
             "print(len(list(getattr(rt, 'available_kernels', []) or [])))")'''),
]


def main():
    src = io.open(TARGET, encoding="utf-8").read()
    for old, new in PAIRS:
        if old not in src:
            print("PATTERN NOT FOUND: %r" % old[:60])
            return 1
        src = src.replace(old, new, 1)
    io.open(TARGET, "w", encoding="utf-8", newline="").write(src)
    print("patched %s" % os.path.basename(TARGET))
    return 0


if __name__ == "__main__":
    sys.exit(main())
