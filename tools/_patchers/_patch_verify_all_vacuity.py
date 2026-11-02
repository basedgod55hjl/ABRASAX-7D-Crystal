"""Wire the vacuity audit into the aggregate verifier as stage 23.

Why a patcher and not a hand edit: the anchor has to be found or the edit has to
refuse. A silent no-op here is exactly the failure mode _vacuity_audit.py exists
to catch, so this refuses to report success unless the stage is in the table.

    python tools/_patchers/_patch_verify_all_vacuity.py [--check]
"""
from __future__ import annotations

import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.dirname(HERE)
TARGET = os.path.join(TOOLS, "verify_all.py")

ANCHOR = (
    '        ("loader", [py, os.path.join(ROOT, "kernels", "cuda", '
    '"test_kernel_loader.py"),\n'
    '                    "--quick"], ROOT, 420, "ALL TESTS PASSED", fast),\n'
)
ADD = ANCHOR + (
    '        ("vacuity", [py, os.path.join(T, "_vacuity_audit.py")], ROOT, 900,\n'
    '         ", 0 failed", False),\n'
)
STAGE_MARK = '        ("vacuity", '
DOC_ANCHOR = "   19. offline        verify_offline.py --fast     every static gate, network blocked\n"
DOC_ADD = DOC_ANCHOR + (
    "   19b vacuity       _vacuity_audit.py       every gate proves it ran, not just that it passed\n"
)


def main() -> int:
    src = io.open(TARGET, encoding="utf-8", newline="").read()
    have_stage = STAGE_MARK in src
    have_doc = "19b vacuity" in src
    if "--check" in sys.argv:
        print("stage=%s doc=%s" % (have_stage, have_doc))
        return 0 if (have_stage and have_doc) else 1
    if have_stage and have_doc:
        print("already patched")
        return 0

    crlf = "\r\n" in src
    anchor, add = ANCHOR, ADD
    doc_anchor, doc_add = DOC_ANCHOR, DOC_ADD
    if crlf:
        anchor, add = anchor.replace("\n", "\r\n"), add.replace("\n", "\r\n")
        doc_anchor, doc_add = (doc_anchor.replace("\n", "\r\n"),
                               doc_add.replace("\n", "\r\n"))
    if not have_stage:
        if anchor not in src:
            print("ANCHOR NOT FOUND (stage table)")
            return 1
        src = src.replace(anchor, add, 1)
    if not have_doc:
        if doc_anchor not in src:
            print("ANCHOR NOT FOUND (docstring)")
            return 1
        src = src.replace(doc_anchor, doc_add, 1)
    io.open(TARGET, "w", encoding="utf-8", newline="").write(src)
    print("patched stage=%s doc=%s" % (True, True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
