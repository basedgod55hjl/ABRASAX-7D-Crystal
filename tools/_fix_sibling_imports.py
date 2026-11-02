"""Fix provably-phantom package imports in kernels/cuda/ctypes_wrapper.py.

`from cuda.crystal_kernels import ...` cannot resolve: C:\\abrasax\\SYSTEMS\\
ABRASAX-7D-Crystal\\cuda\\ has no __init__.py and no crystal_kernels.py (only
cuda_kernel_loader.py and a build dir). The module it wants is the SIBLING
kernels/cuda/crystal_kernels.py, which defines every name imported:

    CrystalKernelContext :157   HAVE_PYCUDA :45
    manifold_projection_7d :357  phi_harmonic_modulation :373  cbm_flux_unfold :390

Consequence today: line 94 sits in a try/except, so the pycuda backend always
fails to initialise and the wrapper silently falls back to ctypes. With pycuda
installed it would STILL never activate. This rewrites the four code-site
imports to the sibling form; the docstring example is left for a human.

Idempotent, CRLF-aware. Backup: .bak-phantomsibling
"""
from __future__ import annotations

import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(os.path.dirname(HERE), "kernels", "cuda", "ctypes_wrapper.py")
BACKUP = TARGET + ".bak-phantomsibling"
PATTERN = re.compile(r"^(\s*)from cuda\.crystal_kernels import ", re.MULTILINE)
REPLACEMENT = r"\1from crystal_kernels import "


def main() -> int:
    src = open(TARGET, "r", encoding="utf-8", newline="").read()
    hits = PATTERN.findall(src)
    if not hits:
        already = "from crystal_kernels import" in src
        print("%s: %s" % ("already fixed" if already else "NO MATCH", TARGET))
        return 0 if already else 1

    if not os.path.exists(BACKUP):
        shutil.copy2(TARGET, BACKUP)
        print("backup  -> %s" % BACKUP)

    out, n = PATTERN.subn(REPLACEMENT, src)
    with open(TARGET, "w", encoding="utf-8", newline="") as fh:
        fh.write(out)
    print("rewrote %d import(s) in %s" % (n, TARGET))
    return 0


if __name__ == "__main__":
    sys.exit(main())
