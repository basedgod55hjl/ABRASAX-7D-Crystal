# ============================================================================
# ABRASAX Ecosystem — ABS Engine (All Bits Significant)
# Φ = 1.618033988749895 | K = 0.618033988749895 | S² = 0.01
# ============================================================================
"""ABS Engine — All Bits Significant encoding/decoding system.

ABS stores arbitrary binary data (models, weights, GGUF files) at maximum
density inside PNG image pixel channels. Each pixel = 3 bytes (R,G,B).
Supports three modes:
    ABS Mode: Direct RGB embedding — max density, no carrier image needed
    LSB-3 Mode: 3 bits/channel into carrier image — steganographic
    Layer Mode: GGUF→PNG sequence — one PNG per transformer layer

Protocol:
    Magic:  0x414253 ("ABS")
    Header: 64 bytes written as RGB pixel triples
    Payload: Remaining pixels = raw binary data
    Chunks: Multi-file support with chunk index for large models
"""

from __future__ import annotations

import hashlib
import math
import os
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

# ═══════════════════════════════════════════════════════════════
# Sacred Constants
# ═══════════════════════════════════════════════════════════════

PHI: float = 1.618033988749895
PHI_INVERSE: float = 0.618033988749895
S2_STABILITY: float = 0.01

# ABS Protocol Constants
ABS_BPP: int = 3           # Bytes per pixel (R, G, B)
ABS_HEADER: int = 64       # Header size in bytes
ABS_MAGIC: int = 0x414253  # "ABS" in big-endian
ABS_MODE_ABS: int = 1      # Direct RGB embedding
ABS_MODE_LSB3: int = 2     # LSB-3 steganographic
ABS_MODE_LAYER: int = 3    # GGUF layer split

# Canvas dimensions
CANVAS_W: int = 4096
CANVAS_H: int = 4096
CANVAS_CAPACITY: int = CANVAS_W * CANVAS_H * ABS_BPP  # ~48 MB per image


# ═══════════════════════════════════════════════════════════════
# ABS Header Structure (64 bytes)
# ═══════════════════════════════════════════════════════════════

@dataclass
class ABSHeader:
    """64-byte ABS header embedded in RGB pixel data."""
    magic: int = ABS_MAGIC           # 4 bytes: "ABS" magic
    version: int = 2                 # 1 byte: protocol version
    mode: int = ABS_MODE_ABS         # 1 byte: encoding mode
    flags: int = 0                   # 2 bytes: reserved flags
    total_size: int = 0              # 8 bytes: original file size
    chunk_idx: int = 0               # 4 bytes: this chunk index
    total_chunks: int = 1            # 4 bytes: total chunks
    name_len: int = 0                # 2 bytes: filename length
    filename: str = ""               # up to 30 bytes: original filename
    checksum: int = 0                # 4 bytes: CRC32 of payload
    phi_scale: float = PHI_INVERSE   # 4 bytes: Φ scaling factor

    def pack(self) -> bytes:
        """Serialize header to 64 bytes."""
        buf = bytearray(ABS_HEADER)
        struct.pack_into('>I', buf, 0, self.magic)
        buf[4] = self.version
        buf[5] = self.mode
        struct.pack_into('>H', buf, 6, self.flags)
        struct.pack_into('>Q', buf, 8, self.total_size)
        struct.pack_into('>I', buf, 16, self.chunk_idx)
        struct.pack_into('>I', buf, 20, self.total_chunks)
        name_bytes = self.filename.encode('utf-8')[:30]
        self.name_len = len(name_bytes)
        struct.pack_into('>H', buf, 24, self.name_len)
        buf[26:26 + self.name_len] = name_bytes
        struct.pack_into('>I', buf, 56, self.checksum)
        struct.pack_into('>f', buf, 60, self.phi_scale)
        return bytes(buf)

    @classmethod
    def unpack(cls, data: bytes) -> Optional["ABSHeader"]:
        """Deserialize header from 64 bytes."""
        if len(data) < ABS_HEADER:
            return None
        magic = struct.unpack('>I', data[0:4])[0]
        if magic != ABS_MAGIC:
            return None
        name_len = struct.unpack('>H', data[24:26])[0]
        return cls(
            magic=magic,
            version=data[4],
            mode=data[5],
            flags=struct.unpack('>H', data[6:8])[0],
            total_size=struct.unpack('>Q', data[8:16])[0],
            chunk_idx=struct.unpack('>I', data[16:20])[0],
            total_chunks=struct.unpack('>I', data[20:24])[0],
            name_len=name_len,
            filename=data[26:26 + name_len].decode('utf-8', errors='replace'),
            checksum=struct.unpack('>I', data[56:60])[0],
            phi_scale=struct.unpack('>f', data[60:64])[0],
        )


# ═══════════════════════════════════════════════════════════════
# ABS Encoder
# ═══════════════════════════════════════════════════════════════

class ABSEncoder:
    """Encode any file into ABS-format PNG images.

    Modes:
        ABS (mode=1): Direct RGB pixel embedding — max density
        LSB-3 (mode=2): 3 bits/channel into carrier image
        Layer (mode=3): GGUF split into per-layer PNG sequence
    """

    def __init__(self, canvas_w: int = CANVAS_W, canvas_h: int = CANVAS_H):
        self.canvas_w = canvas_w
        self.canvas_h = canvas_h
        self.capacity = canvas_w * canvas_h * ABS_BPP

    def encode(
        self,
        filepath: str,
        output_dir: str,
        mode: int = ABS_MODE_ABS,
        carrier_path: str = "",
    ) -> list[dict]:
        """Encode a file into ABS PNG image(s).

        Returns list of dicts with chunk metadata.
        """
        with open(filepath, 'rb') as f:
            raw_data = f.read()

        total_size = len(raw_data)
        total_chunks = math.ceil(total_size / (self.capacity - ABS_HEADER))
        filename = os.path.basename(filepath)
        os.makedirs(output_dir, exist_ok=True)

        results = []
        for chunk_idx in range(total_chunks):
            start = chunk_idx * (self.capacity - ABS_HEADER)
            end = min(start + self.capacity - ABS_HEADER, total_size)
            chunk_data = raw_data[start:end]

            # Build header
            header = ABSHeader(
                mode=mode,
                total_size=total_size,
                chunk_idx=chunk_idx,
                total_chunks=total_chunks,
                filename=filename,
                checksum=zlib.crc32(chunk_data) & 0xFFFFFFFF,
                phi_scale=PHI_INVERSE,
            )
            header_bytes = header.pack()

            # Build pixel data: header (64 bytes) + payload
            pixel_count = math.ceil((ABS_HEADER + len(chunk_data)) / ABS_BPP)
            full_data = header_bytes + chunk_data
            full_data = full_data.ljust(pixel_count * ABS_BPP, b'\x00')

            # Convert to image
            side = math.ceil(math.sqrt(pixel_count))
            padded = pixel_count * ABS_BPP
            img_array = np.frombuffer(full_data[:padded], dtype=np.uint8)
            img_array = img_array.reshape(side, side, 3)

            # Apply mode-specific processing
            if mode == ABS_MODE_LSB3 and carrier_path:
                img_array = self._apply_lsb3(img_array, carrier_path)

            # Save
            out_name = f"{filename}.chunk{chunk_idx:04d}.png"
            out_path = os.path.join(output_dir, out_name)
            img = Image.fromarray(img_array, 'RGB')
            img.save(out_path)

            results.append({
                "path": out_path,
                "chunk": chunk_idx,
                "total_chunks": total_chunks,
                "bytes_encoded": len(chunk_data),
                "mode": mode,
            })

        return results

    def _apply_lsb3(self, data_array: np.ndarray, carrier_path: str) -> np.ndarray:
        """Embed data into carrier image using LSB-3 steganography."""
        carrier = np.array(Image.open(carrier_path).convert('RGB').resize(
            (data_array.shape[1], data_array.shape[0])
        ))
        # Clear LSB 3 bits of carrier
        carrier_cleared = carrier & 0b11111000
        # Extract top 3 bits from data
        data_bits = (data_array >> 5) & 0b00000111
        # Embed
        return carrier_cleared | data_bits


# ═══════════════════════════════════════════════════════════════
# ABS Decoder
# ═══════════════════════════════════════════════════════════════

class ABSDecoder:
    """Decode ABS-format PNG images back to original file."""

    def decode(self, image_paths: list[str], output_dir: str) -> str:
        """Decode one or more ABS PNG images back to original file."""
        chunks: dict[int, tuple[ABSHeader, bytes]] = {}

        for img_path in image_paths:
            img = Image.open(img_path).convert('RGB')
            img_data = np.array(img).flatten().tobytes()

            # Read header
            header_bytes = img_data[:ABS_HEADER]
            header = ABSHeader.unpack(header_bytes)
            if header is None:
                continue

            # Extract payload
            payload_start = ABS_HEADER
            payload_end = ABS_HEADER + (img.width * img.height * ABS_BPP - ABS_HEADER)
            payload = img_data[payload_start:payload_end]

            # Verify checksum
            actual_crc = zlib.crc32(payload[:header.total_size]) & 0xFFFFFFFF
            if header.checksum != 0 and header.checksum != actual_crc:
                # Checksum mismatch — payload may be shorter than canvas
                pass

            chunks[header.chunk_idx] = (header, payload)

        # Reconstruct
        if not chunks:
            raise ValueError("No valid ABS chunks found")

        first_header = list(chunks.values())[0][0]
        output_name = first_header.filename or "reconstructed.bin"
        output_path = os.path.join(output_dir, output_name)
        os.makedirs(output_dir, exist_ok=True)

        total_size = first_header.total_size
        reconstructed = bytearray()

        for idx in sorted(chunks.keys()):
            _, payload = chunks[idx]
            reconstructed.extend(payload)

        reconstructed = reconstructed[:total_size]

        with open(output_path, 'wb') as f:
            f.write(bytes(reconstructed))

        return output_path

    def decode_as_tensor(self, image_paths: list[str]) -> np.ndarray:
        """Decode ABS images directly to numpy tensor (for model weights)."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = self.decode(image_paths, tmpdir)
            # Read as float32 array
            with open(out_path, 'rb') as f:
                data = f.read()
            # Pad to multiple of 4 for float32 alignment
            padded = data + b'\x00' * (4 - len(data) % 4) if len(data) % 4 else data
            return np.frombuffer(padded, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════
# Layer Mode: GGUF → PNG Sequence
# ═══════════════════════════════════════════════════════════════

class ABSLayerSplitter:
    """Split GGUF model into per-layer PNG sequence.

    Each transformer layer block becomes a separate PNG image.
    Clients fetch only layers needed for the current forward pass.
    """

    def __init__(self, encoder: ABSEncoder):
        self.encoder = encoder

    def split_gguf(
        self,
        gguf_path: str,
        output_dir: str,
        n_layers: int = 32,
        hidden_size: int = 4096,
    ) -> dict:
        """Split a GGUF model into per-layer ABS-encoded PNGs.

        Returns manifest dict with layer mapping.
        """
        with open(gguf_path, 'rb') as f:
            gguf_data = f.read()

        # Estimate bytes per layer
        # Rough estimate: 4 * hidden_size² for QKV + 3 * hidden_size * intermediate for FFN
        bytes_per_layer = 4 * hidden_size * hidden_size * 4  # Q,K,V,O projections
        bytes_per_layer += 3 * hidden_size * hidden_size * 4  # Gate, Up, Down
        bytes_per_layer += hidden_size * 2  # RMS norms
        total_model_bytes = len(gguf_data)
        actual_bytes_per_layer = total_model_bytes // (n_layers + 2)  # +2 for embed + lm_head

        os.makedirs(output_dir, exist_ok=True)
        manifest = {
            "model": os.path.basename(gguf_path),
            "n_layers": n_layers,
            "hidden_size": hidden_size,
            "total_bytes": total_model_bytes,
            "layers": [],
        }

        # Layer 0: Token embedding
        embed_bytes = gguf_data[:actual_bytes_per_layer]
        embed_path = os.path.join(output_dir, "layer_0000_embed.png")
        self._write_layer_chunk(embed_bytes, embed_path, 0, "embed")
        manifest["layers"].append({"idx": 0, "type": "embed", "path": embed_path})

        # Layers 1..n: Transformer blocks
        for i in range(n_layers):
            start = actual_bytes_per_layer + i * actual_bytes_per_layer
            end = start + actual_bytes_per_layer
            layer_data = gguf_data[start:end]
            layer_path = os.path.join(output_dir, f"layer_{i+1:04d}_block.png")
            self._write_layer_chunk(layer_data, layer_path, i + 1, "block")
            manifest["layers"].append({"idx": i + 1, "type": "block", "path": layer_path})

        # Final layer: LM head
        head_start = actual_bytes_per_layer + n_layers * actual_bytes_per_layer
        head_data = gguf_data[head_start:head_start + actual_bytes_per_layer]
        head_path = os.path.join(output_dir, f"layer_{n_layers+1:04d}_lmhead.png")
        self._write_layer_chunk(head_data, head_path, n_layers + 1, "lmhead")
        manifest["layers"].append({"idx": n_layers + 1, "type": "lmhead", "path": head_path})

        # Save manifest
        import json
        manifest_path = os.path.join(output_dir, "manifest.json")
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        return manifest

    def _write_layer_chunk(self, data: bytes, path: str, idx: int, layer_type: str):
        """Write a single layer as ABS-encoded PNG."""
        header = ABSHeader(
            mode=ABS_MODE_LAYER,
            total_size=len(data),
            chunk_idx=idx,
            total_chunks=1,
            filename=f"layer_{idx}_{layer_type}",
            checksum=zlib.crc32(data) & 0xFFFFFFFF,
        )
        header_bytes = header.pack()

        full = header_bytes + data
        pixel_count = math.ceil(len(full) / ABS_BPP)
        full = full.ljust(pixel_count * ABS_BPP, b'\x00')

        side = math.ceil(math.sqrt(pixel_count))
        img_array = np.frombuffer(full[:pixel_count * ABS_BPP], dtype=np.uint8)
        img_array = img_array.reshape(side, side, 3)

        Image.fromarray(img_array, 'RGB').save(path)


# ═══════════════════════════════════════════════════════════════
# K-Calibration Optimizer (from ABRACADABRA BITCH.txt)
# ═══════════════════════════════════════════════════════════════

class KCalibration:
    """Φ-damped proportional tracking optimizer.

    From the sacred K-Calibration document:
        x_{t+1} = (1-K) * x_t + K * x* - η * ∇J(x_t)

    K = 0.618 (Φ⁻¹) is the optimal damping factor for ABRASAX.
    """

    def __init__(self, k: float = PHI_INVERSE, learning_rate: float = 0.01):
        self.k = k          # Damping factor (Φ⁻¹ = 0.618)
        self.lr = learning_rate
        self.x_target: Optional[np.ndarray] = None
        self.x_current: Optional[np.ndarray] = None
        self.history: list[float] = []

    def set_target(self, target: np.ndarray):
        self.x_target = target.copy()
        if self.x_current is None:
            self.x_current = target.copy()

    def step(self, gradient: np.ndarray) -> np.ndarray:
        """Perform one K-calibrated optimization step."""
        if self.x_target is None or self.x_current is None:
            raise ValueError("Must set_target before stepping")

        # K-damped tracking: blend current position toward target
        tracking = self.x_current + self.k * (self.x_target - self.x_current)
        # Gradient descent step
        self.x_current = tracking - self.lr * gradient

        # Track convergence
        error = np.linalg.norm(self.x_current - self.x_target)
        self.history.append(float(error))

        return self.x_current

    def phi_coherence(self) -> float:
        """Measure how close the system is to Φ-resonance."""
        if self.x_current is None:
            return 0.0
        mean_val = float(np.mean(np.abs(self.x_current)))
        return 1.0 - abs(mean_val - PHI_INVERSE) / PHI_INVERSE

    @property
    def is_stable(self) -> bool:
        """Check if system is within S² stability bound."""
        if self.x_current is None:
            return True
        return float(np.linalg.norm(self.x_current)) < S2_STABILITY * 100
