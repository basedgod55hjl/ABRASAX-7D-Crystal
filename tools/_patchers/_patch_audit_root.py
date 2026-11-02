"""Add root-resolution, determinism and wrong-directory checks to the audit."""
import io
import os
import sys

TARGET = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "_audit_verify_gitignore.py")

NEW_FUNC = '''def resolution_checks():
    """The tool must not report a verdict about paths that do not exist."""
    cases = []

    rc, out = run([PY, TOOL, REPO], REPO)
    cases.append(("pointing at the repo root descends into the subtree",
                  rc == 0 and "descended into" in out and "0 misconfigured" in out,
                  [ln.strip() for ln in out.splitlines() if ln.startswith("root:")][-1:]
                  and [ln.strip() for ln in out.splitlines()
                       if ln.startswith("root:")][0][:78] or "<none>"))

    tmp = tempfile.mkdtemp(prefix="vgnowhere_")
    try:
        rc, out = run([PY, TOOL, tmp], tmp)
        cases.append(("a directory with no marker is rejected, not passed",
                      rc == 3 and "refusing to report a verdict" in out,
                      "rc=%d" % rc))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    rc1, out1 = run([PY, TOOL, ROOT], ROOT)
    rc2, out2 = run([PY, TOOL, ROOT], ROOT)
    cases.append(("two runs produce identical output (deterministic)",
                  rc1 == rc2 == 0 and out1 == out2,
                  "%d bytes each" % len(out1)))
    return cases


'''

CALL_ANCHOR = "    cases.extend(red_checks())"


def main():
    src = io.open(TARGET, encoding="utf-8").read()
    if "resolution_checks" in src:
        print("already patched")
        return 0
    anchor = "def divergence_check():"
    if anchor not in src:
        print("ANCHOR NOT FOUND")
        return 1
    src = src.replace(anchor, NEW_FUNC + anchor, 1)
    if CALL_ANCHOR not in src:
        print("CALL SITE NOT FOUND")
        return 1
    src = src.replace(CALL_ANCHOR, CALL_ANCHOR + "\n    cases.extend(resolution_checks())", 1)
    io.open(TARGET, "w", encoding="utf-8", newline="").write(src)
    print("patched %s" % os.path.basename(TARGET))
    return 0


if __name__ == "__main__":
    sys.exit(main())
