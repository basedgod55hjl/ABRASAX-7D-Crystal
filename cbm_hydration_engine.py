#!/usr/bin/env python3
"""
CBM HYDRATION ENGINE — Seed → Model expansion via K-calibrated weight modulation.

Takes a CBM quantum seed and injects it into an existing GGUF model's weights
via phi-harmonic modulation + K-calibrated optimization. The seed doesn't
generate weights from scratch — it transforms a trained model's personality.

Two modes:
  --inject   : Modulate an existing GGUF with a seed (fast, immediate results)
  --generate : Generate a new tiny model from seed alone (experimental)

Usage:
  python cbm_hydration_engine.py --seed genesis_seed.npy --inject existing.gguf --output seeded.gguf
  python cbm_hydration_engine.py --seed genesis_seed.npy --generate --target 10M --output new.gguf
"""
from __future__ import annotations

import argparse, json, math, os, struct, sys, time, zlib
from pathlib import Path
from typing import Optional

import numpy as np

# ── Constants ──────────────────────────────────────────────
PHI = (1 + math.sqrt(5)) / 2
KAPPA = 1.0 / PHI
PHI3 = PHI ** 3
SEED_DIM = 512

ABRASAX_ROOT = Path(os.environ.get("ABRASAX_ROOT", r"C:\Users\BASEDGOD\Desktop\ABRASAX"))
CBM_HEADER = b'CBM-GGUF-v1\x00\x00\x00\x00\x00'[:16].ljust(16, b'\x00')


# ═══════════════════════════════════════════════════════════
# SEED LOADING
# ═══════════════════════════════════════════════════════════

def load_seed(path: str | Path) -> np.ndarray:
	"""Load a seed from .npy, .cbm, or .bin into a normalized float32 array."""
	path = Path(path)
	suffix = path.suffix.lower()

	if suffix == '.npy':
		arr = np.load(path)
	elif suffix == '.cbm':
		with open(path, 'rb') as f:
			data = f.read()
		seed_bytes = data[272:272 + 1024]
		arr = np.frombuffer(seed_bytes, dtype=np.float32)
	elif suffix == '.bin':
		with open(path, 'rb') as f:
			arr = np.frombuffer(f.read(), dtype=np.float32)
	else:
		raise ValueError(f"Unknown seed format: {suffix}")

	arr = arr.astype(np.float32).flatten()[:SEED_DIM]
	if len(arr) < SEED_DIM:
		arr = np.pad(arr, (0, SEED_DIM - len(arr)))
	# Normalize to decent range
	arr = arr / (np.std(arr) + 1e-8) * 0.1
	return arr


# ═══════════════════════════════════════════════════════════
# GGUF READER — reads full tensor data from a GGUF file
# ═══════════════════════════════════════════════════════════

def read_gguf(model_path: str | Path) -> dict:
	"""Read a GGUF file and return its config + tensors as numpy arrays."""
	path = Path(model_path)
	with open(path, 'rb') as f:
		data = f.read()

	assert data[:4] == b'GGUF', f"Not a GGUF file: {path}"
	version = struct.unpack('<I', data[4:8])[0]
	n_tensors = struct.unpack('<Q', data[8:16])[0]
	n_kv = struct.unpack('<Q', data[16:24])[0]

	off = 24
	metadata = {}
	for _ in range(n_kv):
		key_len = struct.unpack('<Q', data[off:off+8])[0]; off += 8
		key = data[off:off+key_len].decode('utf-8', errors='replace'); off += key_len
		val_type = struct.unpack('<I', data[off:off+4])[0]; off += 4

		if val_type == 8:  # string
			str_len = struct.unpack('<Q', data[off:off+8])[0]; off += 8
			val = data[off:off+str_len]; off += str_len
			try:
				val = val.decode('utf-8')
			except Exception:
				pass
		elif val_type == 1:  # bool
			val = bool(data[off]); off += 1
		elif val_type in (4, 3):  # uint32, int32
			val = struct.unpack('<I', data[off:off+4])[0]; off += 4
		elif val_type in (5, 7):  # uint64, int64
			val = struct.unpack('<Q', data[off:off+8])[0]; off += 8
		elif val_type == 6:  # float64
			val = struct.unpack('<d', data[off:off+8])[0]; off += 8
		elif val_type == 0:  # uint8
			val = data[off]; off += 1
		elif val_type == 2:  # int16
			val = struct.unpack('<h', data[off:off+2])[0]; off += 2
		else:
			off += 8  # skip unknown
		metadata[key] = val

	# Align to 32
	while off % 32 != 0:
		off += 1

	# Read tensor infos
	tensors = {}
	tensor_infos = []
	for _ in range(n_tensors):
		name_len = struct.unpack('<Q', data[off:off+8])[0]; off += 8
		name = data[off:off+name_len].decode('utf-8', errors='replace'); off += name_len
		n_dims = struct.unpack('<I', data[off:off+4])[0]; off += 4
		dims = []
		for _ in range(n_dims):
			dims.append(struct.unpack('<Q', data[off:off+8])[0]); off += 8
		dtype = struct.unpack('<I', data[off:off+4])[0]; off += 4
		tensor_offset = struct.unpack('<Q', data[off:off+8])[0]; off += 8
		tensor_infos.append((name, dims, dtype, tensor_offset))

	# Read tensor data
	for name, dims, dtype, tensor_off in tensor_infos:
		n_elems = int(np.prod(dims))
		if dtype == 0:     # float32 / F32
			elem_size = 4
			arr = np.frombuffer(data[tensor_off:tensor_off + n_elems * elem_size], dtype=np.float32)
		elif dtype == 1:   # float16 / F16
			elem_size = 2
			arr = np.frombuffer(data[tensor_off:tensor_off + n_elems * elem_size], dtype=np.float16).astype(np.float32)
		elif dtype in (2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19):
			# Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K, IQ2, IQ3, IQ4, etc.
			# These are block-quantized. We need to dequantize.
			elem_size = 32  # block size for Q4_0
			block_count = n_elems // elem_size
			qdata = data[tensor_off:tensor_off + block_count * (elem_size // 2 + 2)]
			# Dequantize Q4_0: first 2 bytes = scale (fp16), then 16 bytes = 32 x 4-bit values
			values = np.zeros(block_count * 32, dtype=np.float32)
			for b in range(block_count):
				boff = b * (elem_size // 2 + 2)
				scale = np.frombuffer(qdata[boff:boff+2], dtype=np.float16)[0]
				quants = np.frombuffer(qdata[boff+2:boff+2+elem_size//2], dtype=np.uint8)
				# Each byte = 2 x 4-bit values
				for i in range(elem_size // 2):
					values[b * 32 + i * 2] = (quants[i] & 0x0F) - 8
					values[b * 32 + i * 2 + 1] = (quants[i] >> 4) - 8
				values[b * 32:b * 32 + 32] = values[b * 32:b * 32 + 32] * scale
			arr = values[:n_elems]
		else:
			continue  # skip unknown types

		if tuple(dims) != arr.shape:
			try:
				arr = arr.reshape(dims)
			except Exception:
				pass
		tensors[name] = arr

	return {'metadata': metadata, 'tensors': tensors}


# ═══════════════════════════════════════════════════════════
# GGUF WRITER
# ═══════════════════════════════════════════════════════════

def write_gguf(tensors: dict[str, np.ndarray], metadata: dict,
               output_path: str, seed_meta: dict | None = None):
	"""Write a GGUF file with seed-modulated weights."""

	# Merge seed metadata
	meta_items = list(metadata.items())
	if seed_meta:
		meta_items.append(("cbm.seed_injected", True))
		meta_items.append(("cbm.phi", PHI))
		meta_items.append(("cbm.kappa", KAPPA))
		meta_items.append(("cbm.phi3", PHI3))

	buf = bytearray()
	buf.extend(b'GGUF')
	buf.extend(struct.pack('<I', 3))
	buf.extend(struct.pack('<Q', len(tensors)))
	buf.extend(struct.pack('<Q', len(meta_items)))

	for key, val in meta_items:
		key_bytes = key.encode('utf-8') if isinstance(key, str) else key
		buf.extend(struct.pack('<Q', len(key_bytes)))
		buf.extend(key_bytes)
		if isinstance(val, str):
			val_bytes = val.encode('utf-8')
			buf.extend(struct.pack('<I', 8))
			buf.extend(struct.pack('<Q', len(val_bytes)))
			buf.extend(val_bytes)
		elif isinstance(val, bool):
			buf.extend(struct.pack('<I', 1))
			buf.extend(struct.pack('B', 1 if val else 0))
		elif isinstance(val, int):
			buf.extend(struct.pack('<I', 5))
			buf.extend(struct.pack('<Q', val))
		elif isinstance(val, float):
			buf.extend(struct.pack('<I', 6))
			buf.extend(struct.pack('<d', val))
		else:
			val_bytes = str(val).encode('utf-8')
			buf.extend(struct.pack('<I', 8))
			buf.extend(struct.pack('<Q', len(val_bytes)))
			buf.extend(val_bytes)

	# Align
	while len(buf) % 32 != 0:
		buf.extend(b'\x00')

	# Tensor info headers + compute offsets
	tensor_names = list(tensors.keys())
	info_start = len(buf)
	offset = info_start + len(tensor_names) * 64  # rough

	# First pass: write tensor info, track offset
	tensor_offsets = []
	cur_offset = offset
	for name in tensor_names:
		arr = tensors[name]
		name_bytes = name.encode('utf-8')
		rank = len(arr.shape)

		buf.extend(struct.pack('<Q', len(name_bytes)))
		buf.extend(name_bytes)
		buf.extend(struct.pack('<I', rank))
		for d in arr.shape:
			buf.extend(struct.pack('<Q', d))
		buf.extend(struct.pack('<I', 0))  # dtype = float32
		buf.extend(struct.pack('<Q', cur_offset))
		tensor_offsets.append(cur_offset)
		cur_offset += arr.nbytes

	# Align
	while len(buf) % 32 != 0:
		buf.extend(b'\x00')

	# Write tensor data
	assert len(buf) <= offset, f"Header too large: {len(buf)} > {offset}"
	buf.extend(b'\x00' * (offset - len(buf)))  # pad to first tensor offset

	for name, expected_off in zip(tensor_names, tensor_offsets):
		arr = tensors[name]
		assert len(buf) == expected_off, f"Offset mismatch for {name}: {len(buf)} != {expected_off}"
		buf.extend(arr.astype(np.float32).tobytes())

	with open(output_path, 'wb') as f:
		f.write(buf)

	size_mb = len(buf) / 1e6
	total_params = sum(int(np.prod(t.shape)) for t in tensors.values())
	print(f"  Wrote {output_path}: {size_mb:.1f} MB, {total_params:,} params")
	return output_path


# ═══════════════════════════════════════════════════════════
# SEED INJECTION — modulate existing model weights with seed
# ═══════════════════════════════════════════════════════════

def inject_seed(seed_path: str | Path, model_path: str | Path,
                output_path: str, modulation_strength: float = 0.1):
	"""Inject a CBM seed into an existing GGUF model by phi-modulating its weights.

	For each weight tensor, the seed array is interpolated to match the tensor shape
	and used to create a phi-harmonic modulation. K-calibration is then applied
	to pull the modulated weights toward their original values — creating a
	stable perturbation rather than random noise.

	The result is the SAME model with a SEED-DERIVED PERSONALITY SHIFT.
	"""
	print("=" * 60)
	print("  CBM HYDRATION ENGINE — SEED INJECTION MODE")
	print(f"  Seed:  {seed_path}")
	print(f"  Model: {model_path}")
	print(f"  φ={PHI:.6f}  K={KAPPA:.6f}  strength={modulation_strength}")
	print("=" * 60)

	# Load seed
	seed = load_seed(seed_path)
	print(f"\n[1/4] Seed loaded: {len(seed)} dims, mean={seed.mean():.4f}, std={seed.std():.4f}")

	# Load model
	print(f"[2/4] Reading GGUF model...")
	t0 = time.perf_counter()
	model_data = read_gguf(model_path)
	elapsed = time.perf_counter() - t0
	tensors = model_data['tensors']
	metadata = model_data['metadata']
	total_params = sum(int(np.prod(t.shape)) for t in tensors.values())
	print(f"  {len(tensors)} tensors, {total_params:,} params total")
	print(f"  Read in {elapsed:.1f}s")

	# Inject seed into each tensor via phi-harmonic modulation
	print(f"\n[3/4] Injecting seed via phi-harmonic modulation...")
	modulated = {}
	total_mod = 0.0
	for name, arr in tensors.items():
		# Skip norm/ln weights — keep them stable
		if 'norm' in name or 'ln' in name:
			modulated[name] = arr.copy()
			continue

		# Interpolate seed to match tensor shape
		target_size = arr.size
		if target_size <= 1:
			modulated[name] = arr.copy()
			continue

		# Resample seed to tensor dimensions using phi-spaced interpolation
		indices = np.linspace(0, len(seed) - 1, target_size)
		seed_mod = np.interp(indices, np.arange(len(seed)), seed)

		# Reshape to match tensor
		seed_mod = seed_mod.reshape(arr.shape)

		# Phi-harmonic modulation:
		#   w' = w * (1 + strength * sin(w * phi + seed_mod * pi))
		# The sin creates a structured perturbation; different weights shift differently
		modulation = 1.0 + modulation_strength * np.sin(arr * PHI + seed_mod * np.pi)
		arr_mod = arr * modulation

		# K-calibration: pull toward original with KAPPA strength
		# This ensures the model stays functional — only phi-resonant weights shift significantly
		arr_final = (1.0 - KAPPA * modulation_strength) * arr + KAPPA * modulation_strength * arr_mod

		modulated[name] = arr_final.astype(np.float32)
		mod_amount = float(np.mean(np.abs(arr_final - arr)))
		total_mod += mod_amount

	print(f"  Mean weight shift: {total_mod / max(1, len(tensors)):.6f}")

	# Write modulated model
	print(f"[4/4] Writing seeded GGUF...")
	metadata['general.name'] = metadata.get('general.name', 'model') + ' [SEEDED]'
	write_gguf(modulated, metadata, output_path,
	           seed_meta={'seed_source': str(seed_path), 'strength': modulation_strength})

	# Write seed connection manifest
	manifest = {
		'version': '1.0.0',
		'type': 'seed_injection',
		'seed_source': str(seed_path),
		'base_model': str(model_path),
		'modulation_strength': modulation_strength,
		'phi': PHI, 'kappa': KAPPA, 'phi3': PHI3,
		'mean_shift': float(total_mod / max(1, len(tensors))),
		'status': 'SEED_INJECTED',
	}
	manifest_path = output_path.replace('.gguf', '_seed_manifest.json')
	with open(manifest_path, 'w') as f:
		json.dump(manifest, f, indent=2)

	print("\n" + "=" * 60)
	print("  SEED INJECTION COMPLETE")
	print(f"  Output: {output_path}")
	print(f"  Chat:   python seed_chat.py --model {output_path}")
	print("=" * 60)
	return output_path


# ═══════════════════════════════════════════════════════════
# HYPERNETWORK GENERATION (experimental — from scratch)
# ═══════════════════════════════════════════════════════════

def generate_from_seed(seed_path: str | Path, target_params: str,
                       output_path: str):
	"""Generate a new model from a seed using hypernetwork expansion.

	This creates a model from scratch using the seed as the sole source of
	weight initialization. The model architecture is a minimal llama-compatible
	transformer. K-calibration stabilizes the weights.

	NOTE: This produces untrained weights. For coherent text, you need to
	train the model on a corpus afterward, OR use --inject mode instead.
	"""
	print("=" * 60)
	print("  CBM HYDRATION ENGINE — GENERATION MODE")
	print(f"  Seed: {seed_path}")
	print(f"  Target: {target_params} params")
	print("=" * 60)

	seed = load_seed(seed_path)
	print(f"\n[1/4] Seed loaded: {len(seed)} dims")

	# Parse target
	target = target_params.upper()
	if 'B' in target:
		target_n = int(float(target.replace('B', '')) * 1e9)
	elif 'M' in target:
		target_n = int(float(target.replace('M', '')) * 1e6)
	else:
		target_n = int(target.replace('_', '').replace(',', ''))

	print(f"[2/4] Designing architecture for ~{target_n:,} params...")

	# Model sizing heuristic
	if target_n >= 1e9:
		dim, n_layers = 2048, 24
	elif target_n >= 500e6:
		dim, n_layers = 1536, 16
	elif target_n >= 100e6:
		dim, n_layers = 1024, 12
	elif target_n >= 10e6:
		dim, n_layers = 512, 8
	else:
		dim, n_layers = 256, 4

	n_heads = max(4, dim // 64)
	n_kv_heads = max(2, n_heads // 4)
	vocab_size = 32000
	hidden_dim = dim * 4

	print(f"  dim={dim}, layers={n_layers}, heads={n_heads}, kv_heads={n_kv_heads}")
	print(f"  vocab={vocab_size}, hidden_dim={hidden_dim}")

	# Generate weights from seed
	print(f"\n[3/4] Expanding seed to weight tensors...")
	tensors = {}
	rng = np.random.RandomState(int(np.sum(np.abs(seed)) * 1e6) % (2**31 - 1))

	# Token embeddings: use seed harmonics to create structured embeddings
	emb = np.zeros((vocab_size, dim), dtype=np.float32)
	for i in range(vocab_size):
		phase = (i * PHI) % (2 * np.pi)
		for j in range(dim):
			seed_idx = (i * dim + j) % len(seed)
			emb[i, j] = seed[seed_idx] * np.sin(phase + j * KAPPA)
	# Normalize embeddings
	emb = emb * 0.02 / (np.std(emb) + 1e-8)
	tensors['token_embd.weight'] = emb.astype(np.float32)

	# Transformer layers
	for layer in range(n_layers):
		phase = layer / max(1, n_layers - 1)
		scale = 0.02 / np.sqrt(dim)

		# Attention weights — use seed to initialize
		for proj, shape in [('attn_q', (dim, dim)), ('attn_k', (dim, n_kv_heads * (dim // n_heads))),
		                    ('attn_v', (dim, n_kv_heads * (dim // n_heads))), ('attn_output', (dim, dim))]:
			w = np.zeros(shape, dtype=np.float32)
			for i in range(shape[0]):
				for j in range(shape[1]):
					seed_idx = (layer * dim * dim + i * shape[1] + j) % len(seed)
					# Xavier-like init modulated by seed
					w[i, j] = rng.normal(0, scale) * (1.0 + seed[seed_idx])
			tensors[f'blk.{layer}.{proj}.weight'] = w.astype(np.float32)

		# FFN weights
		for proj, shape in [('ffn_gate', (dim, hidden_dim)), ('ffn_up', (dim, hidden_dim)),
		                    ('ffn_down', (hidden_dim, dim))]:
			w = np.zeros(shape, dtype=np.float32)
			for i in range(shape[0]):
				for j in range(shape[1]):
					seed_idx = (layer * dim * hidden_dim + i * shape[1] + j) % len(seed)
					w[i, j] = rng.normal(0, scale) * (1.0 + seed[seed_idx])
			tensors[f'blk.{layer}.{proj}.weight'] = w.astype(np.float32)

		# Layer norms
		tensors[f'blk.{layer}.attn_norm.weight'] = np.ones(dim, dtype=np.float32)
		tensors[f'blk.{layer}.ffn_norm.weight'] = np.ones(dim, dtype=np.float32)

	# Output norm
	tensors['output_norm.weight'] = np.ones(dim, dtype=np.float32)
	# Output projection (tied with embedding)
	tensors['output.weight'] = emb.copy()

	total_p = sum(int(np.prod(t.shape)) for t in tensors.values())
	print(f"  Generated {total_p:,} parameters ({total_p * 4 / 1e6:.1f} MB fp32)")

	# Write GGUF
	print(f"\n[4/4] Writing GGUF...")
	metadata = {
		'general.architecture': 'llama',
		'general.name': f'CBM Hydrated {target_params}',
		'llama.context_length': 4096,
		'llama.embedding_length': dim,
		'llama.block_count': n_layers,
		'llama.feed_forward_length': hidden_dim,
		'llama.attention.head_count': n_heads,
		'llama.attention.head_count_kv': n_kv_heads,
		'llama.rope.dimension_count': dim // n_heads,
		'llama.vocab_size': vocab_size,
		'tokenizer.ggml.model': 'llama',
		'tokenizer.ggml.tokens': json.dumps(
			['<unk>'] + [f'<t{i}>' for i in range(vocab_size - 1)]),
		'tokenizer.ggml.scores': json.dumps(
			[0.0] + [-1000.0] * (vocab_size - 1)),
		'tokenizer.ggml.token_type': json.dumps(
			[2] + [1] * (vocab_size - 1)),
	}
	write_gguf(tensors, metadata, output_path,
	           seed_meta={'seed_source': str(seed_path)})

	print("\n" + "=" * 60)
	print("  MODEL GENERATED (untrained — train on corpus for coherence)")
	print(f"  Output: {output_path}")
	print(f"  To train: use llama.cpp finetune or K-calibration pipeline")
	print("=" * 60)
	return output_path


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="CBM Hydration Engine")
	parser.add_argument("--seed", required=True, help="Path to seed (.npy, .cbm, .bin)")
	parser.add_argument("--inject", help="Inject seed into existing GGUF model")
	parser.add_argument("--generate", action="store_true",
	                    help="Generate new model from scratch (experimental)")
	parser.add_argument("--target", default="100M",
	                    help="Target size for --generate mode (e.g. 10M, 100M, 1.5B)")
	parser.add_argument("--output", default="models/hydrated_model.gguf",
	                    help="Output GGUF path")
	parser.add_argument("--strength", type=float, default=0.1,
	                    help="Modulation strength for --inject (0.01-0.5)")
	args = parser.parse_args()

	if args.inject:
		inject_seed(args.seed, args.inject, args.output, args.strength)
	elif args.generate:
		generate_from_seed(args.seed, args.target, args.output)
	else:
		# Default: inject into the trained abrasax model
		default_model = ABRASAX_ROOT / "models" / "training_data" / "abrasax-7d-v1-trained-q4_0.gguf"
		if default_model.exists():
			print("No mode specified. Defaulting to --inject with trained abrasax model.\n")
			inject_seed(args.seed, str(default_model), args.output)
		else:
			print("No --inject model specified and default not found.")
			print("Use --generate for scratch generation or --inject <model.gguf>")
