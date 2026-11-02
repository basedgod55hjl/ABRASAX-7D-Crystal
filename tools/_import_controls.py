"""Controls for the two import gates.

1. _phantom_imports must find the OLD ctypes_wrapper imports (the .bak file) and
   must report clean on the fixed one.
2. _import_matrix must go FATAL when a core file gains an unguarded import of a
   module that does not exist.

Both are built in a throwaway tree so the real repo is never touched.
Exit 0 only if every case behaves.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import _phantom_imports  # noqa: E402
import _import_matrix  # noqa: E402

BACKUP = os.path.join(os.path.dirname(HERE), "kernels", "cuda",
                      "ctypes_wrapper.py.bak-phantomsibling")
FIXED = os.path.join(os.path.dirname(HERE), "kernels", "cuda", "ctypes_wrapper.py")


def layout(src_wrapper):
    """temp tree: cuda/ with only the loader, kernels/cuda/ with the wrapper."""
    tmp = tempfile.mkdtemp(prefix="phantctl_")
    os.makedirs(os.path.join(tmp, "cuda"))
    os.makedirs(os.path.join(tmp, "kernels", "cuda"))
    shutil.copy2(os.path.join(os.path.dirname(HERE), "cuda", "cuda_kernel_loader.py"),
                 os.path.join(tmp, "cuda", "cuda_kernel_loader.py"))
    shutil.copy2(src_wrapper, os.path.join(tmp, "kernels", "cuda", "ctypes_wrapper.py"))
    shutil.copy2(os.path.join(os.path.dirname(HERE), "kernels", "cuda",
                              "crystal_kernels.py"),
                 os.path.join(tmp, "kernels", "cuda", "crystal_kernels.py"))
    return tmp


def main():
    cases = []

    if os.path.exists(BACKUP):
        tmp = layout(BACKUP)
        try:
            _phantom_imports.ROOT = tmp
            found = _phantom_imports.scan()
            mods = sorted({f["import"] for f in found})
            cases.append(("phantom gate catches old imports",
                          len(found) == 4 and mods == ["cuda.crystal_kernels"],
                          "%d sites, imports=%s" % (len(found), mods)))
            cases.append(("phantom gate names the sibling",
                          all(f["sibling"].endswith("crystal_kernels.py") for f in found),
                          found[0]["sibling"] if found else "<none>"))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    else:
        cases.append(("phantom gate catches old imports", False,
                      "backup missing: %s" % BACKUP))

    tmp = layout(FIXED)
    try:
        _phantom_imports.ROOT = tmp
        found = _phantom_imports.scan()
        cases.append(("phantom gate clean on fixed file", len(found) == 0,
                      "%d sites" % len(found)))

        # matrix control: unguarded core import of a nonexistent module
        victim = os.path.join(tmp, "kernels", "cuda", "broken_probe.py")
        with open(victim, "w", encoding="utf-8") as fh:
            fh.write("import definitely_not_installed_xyz\n")
        _import_matrix.ROOT = tmp
        local = _import_matrix.local_candidates()
        findings = _import_matrix.scan(local)
        fatal = [f for f in findings
                 if f["scope"] == "core" and f["severity"] == "FATAL"
                 and f["module"] == "definitely_not_installed_xyz"]
        cases.append(("matrix flags unguarded core import", len(fatal) == 1,
                      "%d fatal" % len(fatal)))

        # and does NOT flag it once guarded
        with open(victim, "w", encoding="utf-8") as fh:
            fh.write("try:\n    import definitely_not_installed_xyz\nexcept ImportError:\n"
                     "    definitely_not_installed_xyz = None\n")
        findings = _import_matrix.scan(local)
        fatal = [f for f in findings
                 if f["scope"] == "core" and f["severity"] == "FATAL"
                 and f["module"] == "definitely_not_installed_xyz"]
        opt = [f for f in findings
               if f["scope"] == "core" and f["severity"] == "OPTIONAL"
               and f["module"] == "definitely_not_installed_xyz"]
        cases.append(("matrix downgrades guarded import", len(fatal) == 0 and len(opt) == 1,
                      "fatal=%d optional=%d" % (len(fatal), len(opt))))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    for name, ok, detail in cases:
        print("[%s] %-38s %s" % ("PASS" if ok else "FAIL", name, detail))
    bad = [c for c in cases if not c[1]]
    print("\n%d controls: %d pass, %d fail"
          % (len(cases), len(cases) - len(bad), len(bad)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
