"""Prove the 7D-Crystal .gitignore keeps the kernels and drops the noise.

Run inside the repo:
    python tools/verify_gitignore.py [repo]

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

A path that does not exist on disk is probed as a directory-content check
instead, because git will not report a nonexistent path as ignored.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

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


def _ignored(repo: Path, rel: str, scratch: Path) -> bool:
    """Ignore-test a path, materialising a probe file when it does not exist."""
    target = repo / rel
    if target.exists():
        p = subprocess.run(["git", "check-ignore", "-q", rel], cwd=str(repo),
                           capture_output=True)
        return p.returncode == 0
    # nonexistent: git can still answer for a path, but not reliably for a file
    # inside an ignored directory, so probe with a real temporary name.
    probe = scratch / Path(rel).name
    probe.write_bytes(b"x")
    p = subprocess.run(["git", "check-ignore", "-q", rel], cwd=str(repo),
                       capture_output=True)
    return p.returncode == 0


def main() -> int:
    repo = Path(sys.argv[1] if len(sys.argv) > 1 else
                r"C:\abrasax\SYSTEMS\ABRASAX-7D-Crystal")
    if not (repo / ".gitignore").is_file():
        print(f"no .gitignore in {repo}")
        return 2
    bad = 0
    with tempfile.TemporaryDirectory() as td:
        scratch = Path(td)
        for rel in MUST_BE_TRACKED:
            ig = _ignored(repo, rel, scratch)
            missing = "" if (repo / rel).exists() else " [absent on disk]"
            print(f"{'XX' if ig else '  '} must-be-committable {rel:<56}"
                  f"{'IGNORED' if ig else 'ok'}{missing}")
            bad += 1 if ig else 0
        for rel, note in MUST_BE_IGNORED:
            ig = _ignored(repo, rel, scratch)
            print(f"{'  ' if ig else 'XX'} must-be-ignored    {rel:<56}"
                  f"{'ignored' if ig else 'COMMITTABLE'}  ({note})")
            bad += 0 if ig else 1
    print(f"\nprobe scratch={scratch} (removed)")
    print(f"{bad} misconfigured")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
