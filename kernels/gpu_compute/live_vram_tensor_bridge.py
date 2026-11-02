# ============================================================================
# ABRASAX — Unified Xenoliving System
# PRIMAL_HEX: 4f5349524953424c58434b | phi: 1.618033988749895
# CREATOR: Sir Charles Spikes (BASEDGOD)
# ============================================================================

#!/usr/bin/env python3
"""LM Studio embedding -> AMD CPU controller -> live CUDA tensor bridge.

The controller keeps orchestration on CPU, asks LM Studio's Nomic embedding
model for vectors, then materializes a normalized tensor on CUDA VRAM when
available. Long-running agents can import this module and keep `live_tensor`
resident; CLI/API calls also write a state file Gemma can read as context.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import urllib.error
import urllib.request
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"torch\.cuda")
    import torch
except Exception:  # pragma: no cover - lets CPU-only tooling still report state.
    torch = None  # type: ignore[assignment]


ABRASAX_ROOT = Path(os.environ.get("ABRASAX_ROOT", r"C:\Windows\System32\abrasax"))
RUNTIME_DIR = ABRASAX_ROOT / "runtime" / "gemma_adaptive"
STATE_PATH = RUNTIME_DIR / "live_vram_tensor_state.json"
TENSOR_CPU_MIRROR_PATH = RUNTIME_DIR / "live_vram_tensor_cpu.npy"

DEFAULT_LM_STUDIO_BASE = os.environ.get("ABRASAX_LM_STUDIO_API", "http://127.0.0.1:1234")
DEFAULT_EMBEDDING_MODEL = os.environ.get(
    "ABRASAX_EMBEDDING_MODEL",
    "text-embedding-nomic-embed-text-v1.5",
)
DEFAULT_GEMMA_MODEL = os.environ.get("ABRASAX_MODEL", "gemma-4-e2b-uncensored-hauhaucs-aggressive")


@dataclass
class LiveTensorState:
    ok: bool
    ts: float
    input_sha256: str
    embedding_model: str
    gemma_model: str
    lm_studio_base: str
    cpu_controller: str
    device: str
    dtype: str
    shape: list[int]
    norm: float
    mean: float
    std: float
    min: float
    max: float
    tensor_cpu_mirror: str
    note: str
    error: str | None = None


def _cpu_controller_label() -> str:
    cpu = platform.processor() or platform.machine() or "unknown-cpu"
    vendor = "amd" if "amd" in cpu.lower() else "cpu"
    return f"{vendor}:{cpu}"


def _post_json(url: str, payload: dict[str, Any], timeout_s: float = 60.0) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'lm-studio')}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return json.loads(raw)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:600]
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc


def _embedding_from_response(data: dict[str, Any]) -> np.ndarray:
    rows = data.get("data")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("embedding response missing data[0].embedding")
    embedding = rows[0].get("embedding")
    if not isinstance(embedding, list) or not embedding:
        raise RuntimeError("embedding response data[0].embedding is empty")
    return np.asarray(embedding, dtype=np.float32)


class AmdCpuGemmaVramController:
    """CPU-side controller that feeds Gemma with VRAM-backed embedding state."""

    def __init__(
        self,
        lm_studio_base: str = DEFAULT_LM_STUDIO_BASE,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        gemma_model: str = DEFAULT_GEMMA_MODEL,
        cuda_device: int = 0,
    ) -> None:
        self.lm_studio_base = lm_studio_base.rstrip("/")
        self.embedding_model = embedding_model
        self.gemma_model = gemma_model
        self.cuda_device = cuda_device
        self.live_tensor = None

        # Keep the scheduler/controller CPU-side; push only tensors to VRAM.
        if torch is not None:
            threads = int(os.environ.get("ABRASAX_CPU_THREADS", os.cpu_count() or 4))
            torch.set_num_threads(max(1, threads))

    def embed_text(self, text: str) -> np.ndarray:
        payload = {"model": self.embedding_model, "input": text}
        data = _post_json(f"{self.lm_studio_base}/v1/embeddings", payload)
        return _embedding_from_response(data)

    def tensorize(self, embedding: np.ndarray) -> tuple[Any, str]:
        if torch is None:
            return embedding, "numpy-cpu"

        cpu_tensor = torch.from_numpy(embedding.astype(np.float32, copy=False)).contiguous()
        norm = torch.linalg.vector_norm(cpu_tensor).clamp_min(1e-8)
        cpu_tensor = cpu_tensor / norm

        if torch.cuda.is_available():
            device = torch.device(f"cuda:{self.cuda_device}")
            if torch.cuda.current_device() != self.cuda_device:
                torch.cuda.set_device(device)
            pinned = cpu_tensor.pin_memory() if torch.cuda.is_available() else cpu_tensor
            self.live_tensor = pinned.to(device=device, dtype=torch.float16, non_blocking=True)
            torch.cuda.synchronize(device)
            return self.live_tensor, str(device)

        self.live_tensor = cpu_tensor
        return self.live_tensor, "torch-cpu"

    def ingest(self, text: str) -> LiveTensorState:
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        embedding = self.embed_text(text)
        tensor, device = self.tensorize(embedding)
        np.save(TENSOR_CPU_MIRROR_PATH, embedding.astype(np.float32, copy=False))

        if torch is not None and hasattr(tensor, "detach"):
            stats_array = tensor.detach().float().cpu().numpy()
            dtype = str(tensor.dtype)
            shape = list(tensor.shape)
        else:
            stats_array = np.asarray(tensor, dtype=np.float32)
            dtype = str(stats_array.dtype)
            shape = list(stats_array.shape)

        state = LiveTensorState(
            ok=True,
            ts=time.time(),
            input_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            embedding_model=self.embedding_model,
            gemma_model=self.gemma_model,
            lm_studio_base=self.lm_studio_base,
            cpu_controller=_cpu_controller_label(),
            device=device,
            dtype=dtype,
            shape=shape,
            norm=float(np.linalg.norm(stats_array)),
            mean=float(np.mean(stats_array)),
            std=float(np.std(stats_array)),
            min=float(np.min(stats_array)),
            max=float(np.max(stats_array)),
            tensor_cpu_mirror=str(TENSOR_CPU_MIRROR_PATH),
            note=(
                "Embedding vector is normalized and materialized as live_tensor. "
                "Use a resident controller process to keep CUDA VRAM state alive across calls."
            ),
        )
        STATE_PATH.write_text(json.dumps(asdict(state), indent=2), encoding="utf-8")
        return state

    def gemma_context(self, state: LiveTensorState, prompt: str) -> str:
        return (
            "ABRASAX live VRAM tensor context:\n"
            f"- embedding_model: {state.embedding_model}\n"
            f"- tensor_device: {state.device}\n"
            f"- tensor_shape: {state.shape}\n"
            f"- tensor_norm: {state.norm:.6f}\n"
            f"- tensor_mean/std: {state.mean:.6f}/{state.std:.6f}\n"
            f"- input_sha256: {state.input_sha256}\n\n"
            f"User input:\n{prompt}"
        )

    def ask_gemma(self, prompt: str, max_tokens: int = 512) -> dict[str, Any]:
        state = self.ingest(prompt)
        payload = {
            "model": self.gemma_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are Gemma 4 inside ABRASAX. Treat the provided live VRAM "
                        "embedding tensor metadata as the active input control signal."
                    ),
                },
                {"role": "user", "content": self.gemma_context(state, prompt)},
            ],
            "max_tokens": max_tokens,
        }
        data = _post_json(f"{self.lm_studio_base}/v1/chat/completions", payload, timeout_s=120.0)
        return {"state": asdict(state), "gemma": data}


def _read_text_arg(args: argparse.Namespace) -> str:
    if args.text:
        return args.text
    data = sys.stdin.read()
    if not data.strip():
        raise SystemExit("No input text supplied.")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="ABRASAX live VRAM tensor bridge")
    parser.add_argument("mode", choices=["embed", "chat", "state"])
    parser.add_argument("text", nargs="?")
    parser.add_argument("--lm-studio-base", default=DEFAULT_LM_STUDIO_BASE)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--gemma-model", default=DEFAULT_GEMMA_MODEL)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    if args.mode == "state":
        if STATE_PATH.exists():
            print(STATE_PATH.read_text(encoding="utf-8"))
        else:
            print(json.dumps({"ok": False, "error": "no live tensor state yet"}))
        return

    controller = AmdCpuGemmaVramController(
        lm_studio_base=args.lm_studio_base,
        embedding_model=args.embedding_model,
        gemma_model=args.gemma_model,
    )
    text = _read_text_arg(args)
    if args.mode == "chat":
        result = controller.ask_gemma(text, max_tokens=args.max_tokens)
    else:
        result = asdict(controller.ingest(text))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
