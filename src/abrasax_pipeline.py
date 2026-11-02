# ============================================================================
# ABRASAX Ecosystem — Unified Pipeline
# Φ = 1.618033988749895 | K = 0.618033988749895
# ============================================================================
"""Unified ABRASAX Pipeline — Connects all ecosystem components.

Pipeline stages:
    1. Neural Language Compiler → Intent → 512D quantum seed
    2. Quantum Seed → Φ-modulated weight matrix (K-calibrated)
    3. Weight matrix → ABS-encoded PNG (All Bits Significant)
    4. ABS PNG → GGUF model file
    5. GGUF → Crystal Transformer inference
    6. (Optional) GGUF → Layer-split PNG sequence (distributed inference)

This module replaces the legacy "555" namespace with proper math:
    - ABS (All Bits Significant) — data encoding
    - K-Calibration (Φ-damped tracking) — optimization
    - Φ-Basis — 7D manifold projection
    - S² Stability — convergence bound
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np

# --- optional dependency: projects.neural_language.compiler ----------------
# This package is NOT present on this machine (C:\abrasax\PROJECTS is empty and
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
            "on C:, and C:\\abrasax\\PROJECTS is empty). Restore that package, or "
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
from abs_engine import (
    ABSEncoder, ABSDecoder, ABSLayerSplitter, KCalibration,
    ABS_MODE_ABS, ABS_MODE_LAYER,
)


# ═══════════════════════════════════════════════════════════════
# Unified Pipeline
# ═══════════════════════════════════════════════════════════════

class ABRASAXPipeline:
    """End-to-end ABRASAX compilation and deployment pipeline.

    Connects: Neural Compiler → K-Calibration → ABS Encoder → GGUF → Crystal Transformer
    """

    def __init__(self, output_dir: str = "./pipeline_output"):
        deps = _load_neural_compiler()  # fail here, not at import
        self.compiler = deps["NeuralCompiler"](vram_limit_gb=6.0)
        self.abs_encoder = ABSEncoder()
        self.abs_decoder = ABSDecoder()
        self.layer_splitter = ABSLayerSplitter(self.abs_encoder)
        self.k_calibration = KCalibration(k=PHI_INVERSE)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pipeline_log: list[dict] = []

    # ── Stage 1: Intent → Seed ──

    def stage1_intent_to_seed(self, intent: str) -> QuantumSeed:
        """Compile symbolic intent into quantum seed."""
        seed = self.compiler.compile_intent(intent)
        self.k_calibration.set_target(seed.vector.numpy())

        self._log("stage1", {"intent": intent, "hash": seed.intent_hash})
        return seed

    # ── Stage 2: Seed → Weights ──

    def stage2_seed_to_weights(
        self, seed: QuantumSeed, shape: tuple[int, int] = (64, 192)
    ) -> np.ndarray:
        """Expand seed into Φ-modulated weights with K-calibration."""
        weights = self.compiler.seed_to_weights(seed, shape)
        weights_np = weights.numpy()

        # Apply K-calibration optimization
        self.k_calibration.set_target(weights_np)
        for _ in range(10):  # 10 calibration steps
            gradient = np.random.randn(*shape).astype(np.float32) * 0.001
            weights_np = self.k_calibration.step(gradient)

        phi_coh = self.k_calibration.phi_coherence()
        self._log("stage2", {
            "shape": list(shape),
            "phi_coherence": phi_coh,
            "stable": self.k_calibration.is_stable,
        })
        return weights_np

    # ── Stage 3: Weights → ABS PNG ──

    def stage3_weights_to_abs(self, weights: np.ndarray, name: str = "weights") -> dict:
        """Encode weights into ABS PNG format."""
        # Save weights as binary first
        bin_path = self.output_dir / f"{name}.bin"
        weights.astype(np.float32).tofile(bin_path)

        # ABS encode
        results = self.abs_encoder.encode(
            str(bin_path),
            str(self.output_dir / "abs"),
            mode=ABS_MODE_ABS,
        )
        self._log("stage3", {"chunks": len(results), "mode": "ABS"})
        return {"chunks": results, "bin_path": str(bin_path)}

    # ── Stage 4: ABS → GGUF ──

    def stage4_abs_to_gguf(self, abs_chunks: list[dict], name: str = "model") -> str:
        """Decode ABS PNGs into GGUF file."""
        chunk_paths = [c["path"] for c in abs_chunks]
        gguf_path = str(self.output_dir / f"{name}.gguf")

        decoded_path = self.abs_decoder.decode(
            chunk_paths,
            str(self.output_dir / "decoded"),
        )
        # Rename to GGUF
        os.rename(decoded_path, gguf_path)

        self._log("stage4", {"gguf_path": gguf_path, "size": os.path.getsize(gguf_path)})
        return gguf_path

    # ── Stage 5: GGUF → Layer Split ──

    def stage5_gguf_to_layers(
        self,
        gguf_path: str,
        n_layers: int = 32,
        hidden_size: int = 4096,
    ) -> dict:
        """Split GGUF into per-layer ABS PNG sequence."""
        manifest = self.layer_splitter.split_gguf(
            gguf_path,
            str(self.output_dir / "layers"),
            n_layers=n_layers,
            hidden_size=hidden_size,
        )
        self._log("stage5", {"n_layers": n_layers, "layers": len(manifest["layers"])})
        return manifest

    # ── Full Pipeline ──

    def run_full(
        self,
        intent: str = "GENESIS_PROTOCOL_INIT",
        weight_shape: tuple[int, int] = (64, 192),
        layer_split: bool = False,
        n_layers: int = 32,
        hidden_size: int = 4096,
    ) -> dict:
        """Run the complete ABRASAX pipeline: intent → seed → weights → ABS → GGUF."""
        result = {"intent": intent, "stages": {}}

        # Stage 1
        seed = self.stage1_intent_to_seed(intent)
        result["seed"] = seed.intent_hash

        # Stage 2
        weights = self.stage2_seed_to_weights(seed, weight_shape)
        result["stages"]["weights"] = {"shape": list(weight_shape), "mean": float(weights.mean()), "std": float(weights.std())}

        # Stage 3
        abs_result = self.stage3_weights_to_abs(weights, seed.intent_hash)
        result["stages"]["abs"] = {"chunks": len(abs_result["chunks"])}

        # Stage 4
        gguf_path = self.stage4_abs_to_gguf(abs_result["chunks"], seed.intent_hash)
        result["stages"]["gguf"] = {"path": gguf_path, "size_bytes": os.path.getsize(gguf_path)}

        # Stage 5 (optional)
        if layer_split:
            manifest = self.stage5_gguf_to_layers(gguf_path, n_layers, hidden_size)
            result["stages"]["layers"] = {"n_layers": n_layers, "manifest": str(self.output_dir / "layers" / "manifest.json")}

        # Save pipeline log
        log_path = self.output_dir / "pipeline_log.json"
        with open(log_path, 'w') as f:
            json.dump({"pipeline": result, "steps": self.pipeline_log}, f, indent=2)

        return result

    def _log(self, stage: str, data: dict):
        self.pipeline_log.append({
            "stage": stage,
            "timestamp": time.time(),
            **data,
        })


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ABRASAX Unified Pipeline")
    parser.add_argument("intent", nargs="?", default="GENESIS_PROTOCOL_INIT")
    parser.add_argument("--output", "-o", default="./pipeline_output")
    parser.add_argument("--layers", action="store_true", help="GGUF→layer split")
    parser.add_argument("--batch", action="store_true", help="Run batch genesis intents")
    args = parser.parse_args()

    pipe = ABRASAXPipeline(output_dir=args.output)

    print("=" * 60)
    print("ABRASAX UNIFIED PIPELINE")
    print(f"Φ = {PHI:.15f}  |  K = {PHI_INVERSE:.15f}  |  S² = {S2_STABILITY}")
    print("=" * 60)

    if args.batch:
        intents = [
            "GENESIS_PROTOCOL_INIT", "MANIFOLD_UNFOLD_7D",
            "HOLOGRAPHIC_WEIGHT_MAP", "PHI_RESONANCE_LOCK",
            "QUANTUM_SEED_GERMINATE", "XOR_BRAIN_COLLECTIVE",
            "CBM_FIELD_HYDRATE", "BRIMSTONE_CATALYST",
            "S2_STABILITY_ENFORCE", "NEXUS_AGENT_SPAWN",
        ]
        for i, intent in enumerate(intents):
            print(f"\n[{i+1}/{len(intents)}] {intent}")
            result = pipe.run_full(intent, layer_split=args.layers)
            print(f"  Seed: {result['seed']}  |  ABS chunks: {result['stages']['abs']['chunks']}  |  GGUF: {os.path.basename(result['stages']['gguf']['path'])}")
    else:
        print(f"\nPipeline: '{args.intent}'")
        result = pipe.run_full(args.intent, layer_split=args.layers)
        print(f"\n✅ COMPLETE")
        print(f"  Seed:   {result['seed']}")
        print(f"  Weights: {result['stages']['weights']['shape']} (μ={result['stages']['weights']['mean']:.4f}, σ={result['stages']['weights']['std']:.4f})")
        print(f"  ABS:    {result['stages']['abs']['chunks']} chunks")
        print(f"  GGUF:   {os.path.basename(result['stages']['gguf']['path'])} ({result['stages']['gguf']['size_bytes']} bytes)")
