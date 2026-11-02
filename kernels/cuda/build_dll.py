# ============================================================================
# ABRASAX — Unified Xenoliving System
# PRIMAL_HEX: 4f5349524953424c58434b | phi: 1.618033988749895
# CREATOR: Sir Charles Spikes (BASEDGOD)
# ============================================================================

"""
build_dll.py -- Link compiled .o files into cuda_kernels.dll for ctypes loading.

Usage:
    python cuda/build_dll.py

Requirements:
    - nvcc from CUDA Toolkit (for compiling .cu -> .o if needed)
    - MSVC linker (link.exe) or MinGW ld for .o -> .dll

The produced cuda_kernels.dll can be loaded by ctypes_wrapper.py's
CTYPES_DIRECT backend (no Python CUDA deps required).
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("build_dll")

_HERE = Path(__file__).resolve().parent
_BLD = _HERE / "build"
_CUDA_DIR = _HERE

_O_FILES = [
    _CUDA_DIR / "crystal_kernels_v1.o",
    _CUDA_DIR / "crystal_kernels_v2.o",
    _CUDA_DIR / "physx_crystal_dynamics.o",
]

_CU_FILES = [
    _CUDA_DIR / "crystal_kernels_v1.cu",
    _CUDA_DIR / "crystal_kernels_v2.cu",
    _CUDA_DIR / "physx_crystal_dynamics.cu",
]


def _ensure_o_files() -> bool:
    """Recompile .cu -> .o if any .o is missing or stale."""
    nvcc = _find_nvcc()
    if nvcc is None:
        logger.warning("nvcc not found on PATH; cannot recompile .cu files")
        return all(p.exists() for p in _O_FILES)

    ok = True
    for cu in _CU_FILES:
        o = cu.with_suffix(".o")
        if not o.exists() or (cu.stat().st_mtime > o.stat().st_mtime):
            logger.info("Compiling %s -> %s", cu.name, o.name)
            try:
                subprocess.run(
                    [nvcc, "-o", str(o), "-c", str(cu), "-O3",
                     "-arch=sm_75", "-Xcompiler", "/MD"],
                    check=True, capture_output=True, text=True,
                )
            except subprocess.CalledProcessError as e:
                logger.error("nvcc failed for %s:\n%s", cu.name, e.stderr)
                ok = False
    return ok


def _find_nvcc() -> str | None:
    """Locate nvcc on PATH or in common CUDA install paths."""
    try:
        result = subprocess.run(["nvcc", "--version"],
                                capture_output=True, text=True, check=False)
        if result.returncode == 0:
            return "nvcc"
    except FileNotFoundError:
        pass
    # Try common Windows CUDA paths
    for ver in range(12, 10, -1):
        candidate = Path(f"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v{ver}/bin/nvcc.exe")
        if candidate.exists():
            return str(candidate)
    return None


def _build_dll() -> bool:
    """Link .o files into cuda_kernels.dll using MSVC linker."""
    _BLD.mkdir(parents=True, exist_ok=True)
    dll_path = _BLD / "cuda_kernels.dll"

    # Check all .o files exist
    missing = [p for p in _O_FILES if not p.exists()]
    if missing:
        logger.error("Missing .o files: %s", [str(p) for p in missing])
        logger.info("Run 'compile_kernels.bat' first or ensure nvcc is on PATH")
        return False

    # Try MSVC link.exe
    try:
        obj_list = [str(o) for o in _O_FILES]
        cmd = [
            "link.exe", "/DLL", "/OUT:" + str(dll_path),
            "/NOLOGO", "/NOENTRY",
            *obj_list,
        ]
        logger.info("Linking: %s", " ".join(cmd))
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("DLL created: %s (%d bytes)", dll_path, dll_path.stat().st_size)
        return True
    except FileNotFoundError:
        logger.warning("MSVC link.exe not found; trying MinGW dllwrap...")
    except subprocess.CalledProcessError as e:
        logger.error("MSVC link failed:\n%s", e.stderr)
        return False

    # Try MinGW dllwrap or gcc -shared
    for linker in ["dllwrap", "gcc"]:
        try:
            obj_list = [str(o) for o in _O_FILES]
            if linker == "dllwrap":
                cmd = ["dllwrap", "--output-def", str(_BLD / "cuda_kernels.def"),
                       "-o", str(dll_path), *obj_list]
            else:
                cmd = ["gcc", "-shared", "-o", str(dll_path),
                       "-Wl,--out-implib," + str(_BLD / "libcuda_kernels.a"),
                       *obj_list]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("DLL created (via %s): %s (%d bytes)",
                        linker, dll_path, dll_path.stat().st_size)
            return True
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            logger.warning("%s linking failed: %s", linker, e)
            continue

    logger.error(
        "No linker found. Install MSVC Build Tools or MinGW-w64, then:\n"
        "  link /DLL /OUT:%s /NOLOGO /NOENTRY %s" %
        (dll_path, " ".join(str(o) for o in _O_FILES))
    )
    return False


if __name__ == "__main__":
    # Step 1: ensure .o files exist
    if not _ensure_o_files():
        sys.exit(1)
    # Step 2: link into DLL
    if not _build_dll():
        sys.exit(1)
    logger.info("Done. cuda_kernels.dll is ready for ctypes_wrapper.py")
