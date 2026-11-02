# ============================================================================
# ABRASAX — Unified Xenoliving System
# PRIMAL_HEX: 4f5349524953424c58434b | phi: 1.618033988749895
# CREATOR: Sir Charles Spikes (BASEDGOD)
# ============================================================================

#!/usr/bin/env python3
"""
7D Extended CUDA Dialect Translator
====================================
Translates .7d files containing the 7D mH-Q Crystal dialect into
standard CUDA C suitable for compilation with nvcc (sm_75 target).

Dialect Mapping:
    quantum cortex void            -> __global__ void
    quantum logic __forceinline__  -> __forceinline__ __device__
    quantum logic <type>           -> __device__ <type>
    manifold_idx_7d().cell[N]      -> blockIdx.{x,y}
    manifold_idx_7d().lane[N]      -> threadIdx.{x,y}
    coherence_sync()               -> __syncthreads()
    superposition_fuse()           -> atomicAdd()
    HyperbolicReal                 -> float
    PHIf                           -> 1.618033988749895f
    PI_MANIFOLDf                   -> 3.141592653589793f

Usage:
    python translate_7d_to_cuda.py input.7d -o output.cu
    python translate_7d_to_cuda.py input.7d            # stdout
    python translate_7d_to_cuda.py --batch models/      # batch convert
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import IO


TRANSLATIONS: list[tuple[str, str]] = [
    ('extern "C" quantum cortex void',
     'extern "C" __global__ void'),
    ('quantum cortex void',
     '__global__ void'),
    ('quantum logic __forceinline__',
     '__forceinline__ __device__'),
    ('quantum logic HyperbolicReal',
     '__device__ float'),
    ('quantum logic float',
     '__device__ float'),
    ('quantum logic double',
     '__device__ double'),
    ('quantum logic int',
     '__device__ int'),
    ('quantum logic void',
     '__device__ void'),
    ('manifold_idx_7d().cell[1]',
     'blockIdx.y'),
    ('manifold_idx_7d().cell[0]',
     'blockIdx.x'),
    ('manifold_idx_7d().lane[1]',
     'threadIdx.y'),
    ('manifold_idx_7d().lane[0]',
     'threadIdx.x'),
    ('coherence_sync()',
     '__syncthreads()'),
    ('superposition_fuse',
     'atomicAdd'),
    ('HyperbolicReal',
     'float'),
    ('PHIf',
     '1.618033988749895f'),
    ('PI_MANIFOLDf',
     '3.141592653589793f'),
]


def translate_source(source: str) -> str:
    for pattern, replacement in TRANSLATIONS:
        source = source.replace(pattern, replacement)
    source = re.sub(
        r'(__global__ \w+\([^)]*\))\s*->\s*\w+\s*{',
        r'\1 {',
        source,
    )
    return source


def translate_file(input_path: Path, output: IO[str] | None = None) -> str:
    source = input_path.read_text(encoding="utf-8")
    translated = translate_source(source)
    if output is not None:
        output.write(translated)
    return translated


def infer_output_name(input_path: Path, suffix: str = ".cu") -> Path:
    stem = input_path.name
    for ext in (".7d", ".7ds", ".7DCRYSTAL", ".7dexe"):
        if stem.lower().endswith(ext.lower()):
            stem = stem[: -len(ext)]
            break
    stem = re.sub(r'-\d+$', '', stem)
    return input_path.parent / (stem + suffix)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Translate 7D Extended CUDA dialect (.7d) to standard CUDA (.cu).",
    )
    parser.add_argument("inputs", nargs="+", help="Source .7d files or directories.")
    parser.add_argument("-o", "--output", help="Output path (single-input mode).")
    parser.add_argument("--batch", action="store_true", help="Batch mode.")
    parser.add_argument("--suffix", default=".cu", help="Output suffix.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.batch:
        files: list[Path] = []
        for d in args.inputs:
            p = Path(d)
            if p.is_dir():
                files.extend(sorted(p.rglob("*.7d")))
                files.extend(sorted(p.rglob("*.7ds")))
            else:
                files.append(p)
        if not files:
            print("No .7d files found.", file=sys.stderr, flush=True)
            sys.exit(1)
        for fpath in files:
            out_path = infer_output_name(fpath, args.suffix)
            try:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "w", encoding="utf-8") as f:
                    translate_file(fpath, f)
                print(f"[OK]  {fpath.name} -> {out_path}")
            except Exception as exc:
                print(f"[FAIL] {fpath.name}: {exc}", file=sys.stderr)
        return

    if len(args.inputs) > 1 and args.output:
        print("ERROR: --output only valid with single input.", file=sys.stderr)
        sys.exit(1)

    for fpath_str in args.inputs:
        fpath = Path(fpath_str)
        if not fpath.exists():
            print(f"ERROR: {fpath} not found.", file=sys.stderr, flush=True)
            sys.exit(1)
        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                translate_file(fpath, f)
            print(f"[OK]  {fpath.name} -> {out_path}")
        else:
            print(translate_file(fpath), flush=True)


if __name__ == "__main__":
    main()
