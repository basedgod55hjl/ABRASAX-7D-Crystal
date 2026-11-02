"""Make src/abrasax_pipeline.py survive its phantom dependency.

`from projects.neural_language.compiler import ...` at module scope killed the
whole module at import time: 'projects' resolves to C:\\abrasax\\PROJECTS, which
is empty, and no neural_language package exists anywhere on this box. The file
still compiled, so nothing noticed.

This patch replaces the hard import with a lazy one plus a module __getattr__
(PEP 562), so:
  * `import abrasax_pipeline` succeeds and tooling can inspect it
  * touching NeuralCompiler / QuantumSeed / PHI* raises ModuleNotFoundError with
    the actual reason instead of a bare "No module named 'projects'"

Idempotent, line-ending aware (the file is CRLF). Original saved as
.bak-phantomimport.
"""
from __future__ import annotations

import os
import shutil
import sys

TARGET = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "src", "abrasax_pipeline.py")
BACKUP = TARGET + ".bak-phantomimport"

OLD_IMPORT = (
    "from projects.neural_language.compiler import NeuralCompiler, QuantumSeed, "
    "PHI, PHI_INVERSE, S2_STABILITY\n"
)

NEW_BLOCK = '''# --- optional dependency: projects.neural_language.compiler ----------------
# This package is NOT present on this machine (C:\\abrasax\\PROJECTS is empty and
# no neural_language module exists anywhere on disk). Importing it at module
# scope made this whole file unimportable even though it compiles fine.
# tools/_import_matrix.py flags exactly this: core FATAL at
# src/abrasax_pipeline.py:32. Keep the names lazy so the rest of the module
# stays inspectable; fail loudly, with the real reason, only when you use them.
_NEURAL_COMPILER_NAMES = ("NeuralCompiler", "QuantumSeed", "PHI", "PHI_INVERSE",
                          "S2_STABILITY")
_neural_compiler_cache: dict[str, object] = {}


def _load_neural_compiler() -> dict[str, object]:
    """Import the neural compiler package, or raise with the actual diagnosis."""
    if _neural_compiler_cache:
        return _neural_compiler_cache
    try:
        from projects.neural_language.compiler import (  # noqa: F401
            NeuralCompiler, QuantumSeed, PHI, PHI_INVERSE, S2_STABILITY,
        )
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "abrasax_pipeline requires 'projects.neural_language.compiler', which "
            "is not installed on this machine (checked: no neural_language package "
            "on C:, and C:\\\\abrasax\\\\PROJECTS is empty). Restore that package, or "
            "run the parts of the pipeline that do not need it. "
            "Verified by tools/_import_matrix.py."
        ) from exc
    _neural_compiler_cache.update(
        NeuralCompiler=NeuralCompiler, QuantumSeed=QuantumSeed, PHI=PHI,
        PHI_INVERSE=PHI_INVERSE, S2_STABILITY=S2_STABILITY,
    )
    return _neural_compiler_cache


def __getattr__(name: str):
    """PEP 562: resolve the optional names on first access."""
    if name in _NEURAL_COMPILER_NAMES:
        return _load_neural_compiler()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
# --- end optional dependency ----------------------------------------------
'''

OLD_CTOR = ('    def __init__(self, output_dir: str = "./pipeline_output"):\n'
            '        self.compiler = NeuralCompiler(vram_limit_gb=6.0)\n')
NEW_CTOR = ('    def __init__(self, output_dir: str = "./pipeline_output"):\n'
            '        deps = _load_neural_compiler()  # fail here, not at import\n'
            '        self.compiler = deps["NeuralCompiler"](vram_limit_gb=6.0)\n')


def adapt(text, nl):
    return text.replace("\n", nl)


def main() -> int:
    src = open(TARGET, "r", encoding="utf-8", newline="").read()
    nl = "\r\n" if "\r\n" in src else "\n"

    if "optional dependency: projects.neural_language.compiler" in src:
        print("already patched: %s" % TARGET)
        return 0

    old_import = adapt(OLD_IMPORT, nl)
    old_ctor = adapt(OLD_CTOR, nl)
    if old_import not in src:
        print("IMPORT PATTERN NOT FOUND in %s" % TARGET)
        return 1
    if old_ctor not in src:
        print("CTOR PATTERN NOT FOUND in %s" % TARGET)
        return 1

    if not os.path.exists(BACKUP):
        shutil.copy2(TARGET, BACKUP)
        print("backup  -> %s" % BACKUP)

    out = src.replace(old_import, adapt(NEW_BLOCK, nl)).replace(
        old_ctor, adapt(NEW_CTOR, nl))
    with open(TARGET, "w", encoding="utf-8", newline="") as fh:
        fh.write(out)
    print("patched -> %s (%d -> %d bytes)" % (TARGET, len(src.encode()),
                                              len(out.encode())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
