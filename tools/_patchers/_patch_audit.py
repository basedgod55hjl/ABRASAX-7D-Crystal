"""Add the -q/-v divergence regression check to _audit_verify_gitignore.py."""
import io
import os
import sys

TARGET = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "_audit_verify_gitignore.py")

NEW_FUNC = '''def divergence_check():
    """-q and -v disagree on re-included paths; logic must parse the pattern."""
    rel = "kernels/cuda/__audit_probe_notreal.cu"
    q_rc, _ = run(["git", "check-ignore", "-q", rel], ROOT)
    v_rc, v_out = run(["git", "check-ignore", "-v", rel], ROOT)
    rule = v_out.split("\\t")[0].strip() if v_out else "none"
    return [("-q/-v divergence on a re-included path",
             q_rc != 0 and v_rc == 0 and "!" in v_out,
             "-q rc=%d, -v rc=%d, rule=%s" % (q_rc, v_rc, rule))]


'''


def main():
    src = io.open(TARGET, encoding="utf-8").read()
    if "divergence_check" in src:
        print("already patched")
        return 0
    anchor = "def attribution_checks():"
    if anchor not in src:
        print("ANCHOR NOT FOUND")
        return 1
    src = src.replace(anchor, NEW_FUNC + anchor, 1)
    call = "    attrib, negations = attribution_checks()"
    if call not in src:
        print("CALL SITE NOT FOUND")
        return 1
    src = src.replace(call, "    cases.extend(divergence_check())\n" + call, 1)
    io.open(TARGET, "w", encoding="utf-8", newline="").write(src)
    print("patched %s" % TARGET)
    return 0


if __name__ == "__main__":
    sys.exit(main())
