"""Prove the 7D-Crystal .gitignore keeps the kernels and drops the noise.

Run inside the repo (or point it at one):
    python tools/verify_gitignore.py [path]

ROOT RESOLUTION. The rules being checked live in two places -- the repo root
C:\\abrasax\\.gitignore and the subtree SYSTEMS\\ABRASAX-7D-Crystal\\.gitignore --
and the 27 probe paths are subtree-relative. Pointing the tool at C:\\abrasax
used to "pass" while 16 of the 17 committable files did not exist at that level,
i.e. it verified nothing. It now requires a marker file and descends into the
subtree when pointed at its parent, and exits 3 when no marker exists at all.

CHECKED (measured 2026-11-01 against C:\\abrasax\\.gitignore as it actually
behaved -- `git check-ignore -v` names the matching rule):
  * the published tree was 2913 blobs / 573.8 MB, and 470 MB of that was Rust
    build output: .pdb 184.2 MB, .rmeta 180.9 MB, .rlib 110.8 MB.
  * before the fix, C:\\abrasax\\.gitignore line 38
    "SYSTEMS/ABRASAX-7D-Crystal/kernels/" ignored the ENTIRE kernels subtree --
    every .rs/.cu/.h source and the runtime .dll/.ptx -- and line 63 "BUILD/"
    matched any directory named `build`, including cuda/build.
  * now the sources and shipped kernel artifacts are committable and the build
    output still is not.

HOW THE COMMITTABLE ONES SURVIVE: 12 of the 17 MUST_BE_TRACKED paths are kept
only by `!` re-include rules later in the root file (`!**/cuda/build/*`,
`!SYSTEMS/ABRASAX-7D-Crystal/kernels/**`). Delete those negations and the kernel
artifacts silently become ignored again -- tools/_audit_verify_gitignore.py
demonstrates that in a throwaway repo.

WHY THE DECISION IS PARSED, NOT JUST THE EXIT CODE: `git check-ignore -q` and
`-v` disagree on negations. For an untracked path re-included by `!rule`,
`-q` exits 1 (not ignored) while `-v` exits 0 and prints the rule. Trusting
`-v`'s exit status alone marks committed kernel artifacts as IGNORED; trusting
`-q` alone throws away the rule name. So: read the pattern, and a pattern that
starts with `!` means re-included, i.e. not ignored.

Nonexistent paths are fine for git (it matches patterns without stat-ing the
file), but they are not fine as evidence, so absences are counted and printed.
Nothing here writes to the working tree.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

MARKER = "kernels/cuda/crystal_cortex.h"
SUBTREE = ("SYSTEMS", "ABRASAX-7D-Crystal")

MUST_BE_TRACKED = [
    "cuda/build/crystal_kernels_v2.ptx",
    "cuda/build/crystal_kernels_v2.cubin",
    "kernels/cuda/build/crystal_kernels_v2.dll",
    "kernels/cuda/build/crystal_kernels_v1.dll",
    "kernels/cuda/build/physx_crystal_dynamics.dll",
    "kernels/cuda/crystal_kernels_v2.cu",
    "kernels/cuda/crystal_cortex.h",
    "kernels/rust/Cargo.toml",
    "kernels/rust/src/main.rs",
    "kernels/rust/src/lib.rs",
    "kernels/rust/src/hyperbolic_ca.rs",
    "kernels/rust/src/vram_manager.rs",
    "src/abrasax_pipeline.py",
    "src/seed_xor_graph.py",
    "src/abs_engine.py",
    "README.md",
    "LICENSE",
]

# (probe path, note). Each must be ignored.
MUST_BE_IGNORED = [
    ("kernels/rust/target/release/abrasax_7d_core.exe", "rust exe"),
    ("kernels/rust/target/release/libabrasax_7d_core.rlib", "rust rlib"),
    ("kernels/rust/target/debug/deps/libsyn-a9ff29bc279fa3b9.rlib", "dep rlib"),
    ("kernels/rust/target/release/x.rmeta", "rust rmeta"),
    ("kernels/rust/target/release/abrasax_7d_core.pdb", "rust pdb"),
    ("kernels/cuda/__pycache__/cuda_kernel_loader.cpython-311.pyc", "pyc"),
    ("CRYSTAL/crystal_checkpoint_1778205523.npz", "runtime checkpoint"),
    ("logs/any-runtime-log.jsonl", "runtime log"),
    (".venv/Lib/site-packages/torch/__init__.py", "vendored venv"),
    ("kernels/gpu_compute/backends/.venv/pyvenv.cfg", "vendored venv (nested)"),
]


def resolve_root(given: Path) -> tuple[Path, str]:
    """Descend into the subtree when pointed at its parent. Marker or bust."""
    if (given / MARKER).is_file():
        return given, "marker %s present" % MARKER
    nested = given.joinpath(*SUBTREE)
    if (nested / MARKER).is_file():
        return nested, "descended into %s (marker found there)" % "/".join(SUBTREE)
    return given, "MARKER %s NOT FOUND" % MARKER


def _decision(repo: Path, rel: str) -> tuple[bool, str]:
    """(is_ignored, rule). A `!` pattern means re-included, so NOT ignored."""
    proc = subprocess.run(["git", "check-ignore", "-v", rel], cwd=str(repo),
                          capture_output=True, text=True)
    if proc.returncode != 0:
        return False, ""
    line = (proc.stdout or "").strip().splitlines()
    if not line:
        return False, ""
    rule = line[0].split("\t")[0].strip()
    parts = rule.rsplit(":", 2)
    pattern = parts[2] if len(parts) == 3 else rule
    return (not pattern.startswith("!")), rule


def main() -> int:
    given = Path(sys.argv[1] if len(sys.argv) > 1 else
                 r"C:\abrasax\SYSTEMS\ABRASAX-7D-Crystal")
    repo, why = resolve_root(given)
    print(f"root: {repo}  ({why})")
    if why.startswith("MARKER"):
        print(f"refusing to report a verdict: {repo} is not the 7D-Crystal subtree")
        return 3
    if not (repo / ".gitignore").is_file():
        print(f"no .gitignore in {repo}")
        return 2

    bad = 0
    absent = []
    for rel in MUST_BE_TRACKED:
        ig, rule = _decision(repo, rel)
        if not (repo / rel).exists():
            absent.append(rel)
        if ig:
            state = f"IGNORED by {rule}"
        elif rule:
            state = f"ok, rescued by {rule}"
        else:
            state = "ok"
        print(f"{'XX' if ig else '  '} must-be-committable {rel:<56}{state}"
              f"{' [absent on disk]' if rel in absent else ''}")
        bad += 1 if ig else 0
    for rel, note in MUST_BE_IGNORED:
        ig, rule = _decision(repo, rel)
        shown = (rule or "ignored") if ig else "COMMITTABLE"
        print(f"{'  ' if ig else 'XX'} must-be-ignored    {rel:<56}{shown}  ({note})")
        bad += 0 if ig else 1

    if absent:
        print(f"\nnote: {len(absent)} of {len(MUST_BE_TRACKED)} committable paths are "
              f"absent on disk; the verdict for those is about patterns, not files")
    print(f"\n{bad} misconfigured")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
