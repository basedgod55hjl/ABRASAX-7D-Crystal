"""Call resolution_check() once instead of twice in _audit_loader_location.py."""
import io
import os
import sys

TARGET = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "_audit_loader_location.py")

OLD = '''    print("\\n== C. resolution per entry point (recorded from real runs) ==")
    for rel, resolved, kernels in resolution_check():
        short = resolved.replace(ROOT + os.sep, "") if resolved else resolved
        print("  %-38s -> %-40s kernels=%s" % (rel, short, kernels))
        cases.append(("%s binds the real loader" % os.path.basename(rel),
                      resolved is not None
                      and os.path.normcase(resolved) == os.path.normcase(REAL),
                      short or "<not imported>"))
    rows_c = resolution_check()
    cases = [c for c in cases if not c[0].endswith("binds the real loader")]
    for rel, resolved, kernels in rows_c:
        cases.append(("%s binds the real loader" % os.path.basename(rel),
                      resolved is not None
                      and os.path.normcase(resolved) == os.path.normcase(REAL),
                      os.path.relpath(resolved, ROOT) if resolved else "<not imported>"))'''

NEW = '''    print("\\n== C. resolution per entry point (recorded from real runs) ==")
    resolved_rows = resolution_check()
    for rel, resolved, kernels in resolved_rows:
        short = (os.path.relpath(resolved, ROOT)
                 if resolved and resolved != "<not imported>" else resolved)
        print("  %-38s -> %-40s kernels=%s" % (rel, short, kernels))
        cases.append(("%s binds the real loader" % os.path.basename(rel),
                      resolved is not None
                      and os.path.normcase(resolved) == os.path.normcase(REAL),
                      short or "<not imported>"))'''


def main():
    src = io.open(TARGET, encoding="utf-8").read()
    if OLD not in src:
        print("PATTERN NOT FOUND")
        return 1
    io.open(TARGET, "w", encoding="utf-8", newline="").write(src.replace(OLD, NEW, 1))
    print("patched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
