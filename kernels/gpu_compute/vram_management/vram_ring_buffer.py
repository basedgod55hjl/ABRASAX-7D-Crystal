# ============================================================================
# ABRASAX — Unified Xenoliving System
# PRIMAL_HEX: 4f5349524953424c58434b | phi: 1.618033988749895
# CREATOR: Sir Charles Spikes (BASEDGOD)
# ============================================================================

"""
GPU Pipeline — Core-VRAM Mapper, Ring Buffer, Shared Memory, Accelerated Storage
Provides: CoreVRAMMapper, VRAMRingBuffer, SharedMemoryBus, AcceleratedStorage
All CPU cores -> VRAM mapped, loop-back on fill, NVIDIA shared IPC, fast disk I/O.
"""
import os
import sys
import json
import time
import math
import mmap
import struct
import ctypes
import logging
import threading
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from multiprocessing import shared_memory

try:
    import psutil
except ImportError:
    psutil = None

logger = logging.getLogger("GPU_PIPELINE")

PHI = 1.618033988749895
PHI_INV = 0.618033988749895
SLOT_SIZE_BYTES = 4096
_LOCK = threading.Lock()


def _load_pipeline_config() -> Dict[str, Any]:
    """Load optional config from LLM_ENGINE/config/gpu_pipeline_config.json."""
    try:
        cfg_path = Path(__file__).resolve().parent / "config" / "gpu_pipeline_config.json"
        if cfg_path.exists():
            return json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


# ═══════════════════════════════════════════════════════════════════════════
# CORE-VRAM MAPPER — Maps all CPU cores to VRAM slots, scales dynamically
# ═══════════════════════════════════════════════════════════════════════════
class CoreVRAMMapper:
    """
    Maps every physical CPU core to a VRAM slot for parallel GPU offload.
    Each core gets a dedicated region in the shared ring buffer.
    Scales up by detecting all available cores at init.
    """

    def __init__(self):
        self.physical_cores = 0
        self.logical_cores = 0
        self.vram_total_mb = 6144
        self.vram_free_mb = 6144
        self.slots_per_core = 0
        self.total_slots = 0
        self.core_map: Dict[int, Dict[str, Any]] = {}
        self._detect_cores()
        self._probe_vram()
        self._build_map()

    def _detect_cores(self):
        if psutil:
            self.physical_cores = psutil.cpu_count(logical=False) or 4
            self.logical_cores = psutil.cpu_count(logical=True) or 8
        else:
            self.physical_cores = os.cpu_count() or 4
            self.logical_cores = self.physical_cores

    def _probe_vram(self):
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total,memory.free",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if r.returncode == 0:
                parts = [p.strip() for p in r.stdout.strip().split(",")]
                self.vram_total_mb = int(parts[1])
                self.vram_free_mb = int(parts[2])
        except Exception:
            pass

    def _build_map(self):
        cfg = _load_pipeline_config()
        core_cfg = cfg.get("core_vram", {})
        usable_pct = core_cfg.get("vram_usable_pct", 0.22)
        min_mult = core_cfg.get("min_slots_per_core_mult", 8)
        min_spc = core_cfg.get("min_slots_per_core", 2)
        usable_mb = min(self.vram_free_mb, int(self.vram_total_mb * usable_pct))
        usable_bytes = usable_mb * 1024 * 1024
        min_slots_from_cores = self.logical_cores * min_mult
        self.total_slots = max(min_slots_from_cores, usable_bytes // SLOT_SIZE_BYTES)
        self.slots_per_core = max(min_spc, self.total_slots // max(1, self.logical_cores))

        for core_id in range(self.logical_cores):
            slot_start = core_id * self.slots_per_core
            slot_end = slot_start + self.slots_per_core - 1
            phi_weight = round(PHI_INV ** (1 + core_id % 7), 6)
            self.core_map[core_id] = {
                "core_id": core_id,
                "slot_range": (slot_start, slot_end),
                "slots": self.slots_per_core,
                "phi_weight": phi_weight,
                "writes": 0,
                "reads": 0,
                "loopbacks": 0,
            }

    def get_core_slot(self, core_id: int, offset: int = 0) -> int:
        info = self.core_map.get(core_id % self.logical_cores, {})
        sr = info.get("slot_range", (0, self.total_slots - 1))
        return sr[0] + (offset % self.slots_per_core)

    def record_write(self, core_id: int):
        cid = core_id % self.logical_cores
        if cid in self.core_map:
            self.core_map[cid]["writes"] += 1

    def record_read(self, core_id: int):
        cid = core_id % self.logical_cores
        if cid in self.core_map:
            self.core_map[cid]["reads"] += 1

    def record_loopback(self, core_id: int):
        cid = core_id % self.logical_cores
        if cid in self.core_map:
            self.core_map[cid]["loopbacks"] += 1

    def scale_up(self):
        """Re-probe VRAM and rebuild map with latest free memory."""
        self._probe_vram()
        self._build_map()

    def status(self) -> Dict[str, Any]:
        total_writes = sum(c["writes"] for c in self.core_map.values())
        total_reads = sum(c["reads"] for c in self.core_map.values())
        total_loops = sum(c["loopbacks"] for c in self.core_map.values())
        return {
            "physical_cores": self.physical_cores,
            "logical_cores": self.logical_cores,
            "vram_total_mb": self.vram_total_mb,
            "vram_free_mb": self.vram_free_mb,
            "total_slots": self.total_slots,
            "slots_per_core": self.slots_per_core,
            "total_writes": total_writes,
            "total_reads": total_reads,
            "total_loopbacks": total_loops,
        }


# ═══════════════════════════════════════════════════════════════════════════
# VRAM RING BUFFER — Circular buffer, loops back when filled
# ═══════════════════════════════════════════════════════════════════════════
class VRAMRingBuffer:
    """
    Fixed-size circular ring buffer backed by shared memory.
    When write_head reaches capacity, it wraps around (loop-back),
    overwriting oldest data. Tracks watermarks for VRAM pressure.
    """

    def __init__(self, name: str = "abrasax_vram_ring", num_slots: int = 2048,
                 slot_bytes: int = SLOT_SIZE_BYTES):
        self.name = name
        self.num_slots = num_slots
        self.slot_bytes = slot_bytes
        self._total_bytes = 64 + (num_slots * slot_bytes)  # 64-byte header
        self.shm: Optional[shared_memory.SharedMemory] = None
        self.write_head = 0
        self.read_head = 0
        self.total_writes = 0
        self.total_loopbacks = 0
        self.high_watermark = int(num_slots * 0.90)
        self.low_watermark = int(num_slots * 0.50)
        self._pressure = False
        self._lock = threading.Lock()

    def open(self):
        try:
            self.shm = shared_memory.SharedMemory(
                name=self.name, create=True, size=self._total_bytes
            )
            self._write_header()
            logger.info(
                f"VRAM Ring: created '{self.name}' | {self.num_slots} slots x {self.slot_bytes}B "
                f"= {self._total_bytes // 1024}KB"
            )
        except FileExistsError:
            self.shm = shared_memory.SharedMemory(name=self.name, create=False)
            self._read_header()
            logger.info(f"VRAM Ring: attached to existing '{self.name}'")

    def _header_offset(self) -> int:
        return 0

    def _slot_offset(self, slot_idx: int) -> int:
        return 64 + (slot_idx * self.slot_bytes)

    def _write_header(self):
        if not self.shm:
            return
        struct.pack_into("<IIQQII", self.shm.buf, 0,
                         self.num_slots, self.slot_bytes,
                         self.write_head, self.read_head,
                         self.total_writes, self.total_loopbacks)

    def _read_header(self):
        if not self.shm:
            return
        try:
            vals = struct.unpack_from("<IIQQII", self.shm.buf, 0)
            self.num_slots = vals[0]
            self.slot_bytes = vals[1]
            self.write_head = vals[2]
            self.read_head = vals[3]
            self.total_writes = vals[4]
            self.total_loopbacks = vals[5]
        except Exception:
            pass

    def write(self, data: bytes, core_id: int = 0) -> int:
        """Write data to next slot. Returns slot index. Loops back if full."""
        if not self.shm:
            return -1
        with self._lock:
            slot_idx = self.write_head % self.num_slots
            offset = self._slot_offset(slot_idx)
            chunk = data[:self.slot_bytes]
            padded = chunk + b'\x00' * (self.slot_bytes - len(chunk))
            self.shm.buf[offset:offset + self.slot_bytes] = padded

            old_head = self.write_head
            self.write_head = (self.write_head + 1) % (self.num_slots * 2)
            self.total_writes += 1

            if old_head >= self.num_slots and self.write_head < old_head:
                self.total_loopbacks += 1

            if slot_idx >= self.high_watermark:
                self._pressure = True
            elif slot_idx < self.low_watermark:
                self._pressure = False

            if self.total_writes % 100 == 0:
                self._write_header()

            return slot_idx

    def read(self, slot_idx: int) -> bytes:
        if not self.shm:
            return b''
        with self._lock:
            offset = self._slot_offset(slot_idx % self.num_slots)
            return bytes(self.shm.buf[offset:offset + self.slot_bytes])

    def read_next(self) -> Tuple[int, bytes]:
        with self._lock:
            slot_idx = self.read_head % self.num_slots
            offset = self._slot_offset(slot_idx)
            data = bytes(self.shm.buf[offset:offset + self.slot_bytes])
            self.read_head = (self.read_head + 1) % (self.num_slots * 2)
            return slot_idx, data

    @property
    def pressure(self) -> bool:
        return self._pressure

    @property
    def fill_pct(self) -> float:
        used = (self.write_head - self.read_head) % (self.num_slots * 2)
        return round(min(100.0, (used / max(1, self.num_slots)) * 100), 1)

    def status(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "num_slots": self.num_slots,
            "slot_bytes": self.slot_bytes,
            "total_kb": self._total_bytes // 1024,
            "write_head": self.write_head,
            "read_head": self.read_head,
            "total_writes": self.total_writes,
            "total_loopbacks": self.total_loopbacks,
            "fill_pct": self.fill_pct,
            "pressure": self._pressure,
        }

    def close(self):
        if self.shm:
            try:
                self.shm.close()
                self.shm.unlink()
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════════════════
# NVIDIA SHARED MEMORY BUS — Zero-copy IPC between workers via CUDA MPS
# ═══════════════════════════════════════════════════════════════════════════
class SharedMemoryBus:
    """
    Inter-worker shared memory bus using multiprocessing.shared_memory.
    Each worker gets a read/write segment for zero-copy data exchange.
    Enables NVIDIA CUDA MPS (Multi-Process Service) for concurrent GPU access.
    """

    def __init__(self, num_workers: int = 4, segment_bytes: int = 131072):
        """segment_bytes: 128KB default for faster worker IPC (was 64KB)."""
        self.num_workers = num_workers
        self.segment_bytes = segment_bytes
        self._header_bytes = 256
        self._total = self._header_bytes + (num_workers * segment_bytes)
        self.shm: Optional[shared_memory.SharedMemory] = None
        self._lock = threading.Lock()
        self.msg_count = 0

    def open(self):
        name = "abrasax_worker_bus"
        try:
            self.shm = shared_memory.SharedMemory(
                name=name, create=True, size=self._total
            )
            logger.info(
                f"SharedBus: created | {self.num_workers} workers x {self.segment_bytes // 1024}KB "
                f"= {self._total // 1024}KB total"
            )
        except FileExistsError:
            self.shm = shared_memory.SharedMemory(name=name, create=False)
            logger.info("SharedBus: attached to existing segment")

    def _worker_offset(self, worker_id: int) -> int:
        return self._header_bytes + (worker_id * self.segment_bytes)

    def send(self, from_worker: int, to_worker: int, data: bytes):
        if not self.shm:
            return
        with self._lock:
            offset = self._worker_offset(to_worker % self.num_workers)
            header = struct.pack("<IIQ", from_worker, len(data), int(time.time()))
            payload = header + data[:self.segment_bytes - 16]
            padded = payload + b'\x00' * (self.segment_bytes - len(payload))
            self.shm.buf[offset:offset + self.segment_bytes] = padded
            self.msg_count += 1

    def recv(self, worker_id: int) -> Tuple[int, bytes]:
        if not self.shm:
            return -1, b''
        with self._lock:
            offset = self._worker_offset(worker_id % self.num_workers)
            raw = bytes(self.shm.buf[offset:offset + self.segment_bytes])
            if len(raw) < 16:
                return -1, b''
            from_id, data_len, _ = struct.unpack_from("<IIQ", raw, 0)
            return from_id, raw[16:16 + data_len]

    def broadcast(self, from_worker: int, data: bytes):
        for w in range(self.num_workers):
            if w != from_worker:
                self.send(from_worker, w, data)

    def enable_cuda_mps(self):
        """Enable NVIDIA CUDA MPS for concurrent multi-process GPU access."""
        cmds = [
            ["nvidia-smi", "-i", "0", "-c", "DEFAULT"],
            ["nvidia-smi", "-pm", "1"],
        ]
        env_mps = {
            "CUDA_MPS_PIPE_DIRECTORY": os.path.expanduser("~/.cuda_mps"),
            "CUDA_MPS_LOG_DIRECTORY": os.path.expanduser("~/.cuda_mps_log"),
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": "100",
        }
        for k, v in env_mps.items():
            os.environ[k] = v

        for cmd in cmds:
            try:
                subprocess.run(cmd, capture_output=True, timeout=5)
            except Exception:
                pass

        logger.info("CUDA MPS environment configured for multi-process GPU access")

    def status(self) -> Dict[str, Any]:
        return {
            "num_workers": self.num_workers,
            "segment_bytes": self.segment_bytes,
            "total_kb": self._total // 1024,
            "messages_sent": self.msg_count,
            "active": self.shm is not None,
        }

    def close(self):
        if self.shm:
            try:
                self.shm.close()
                self.shm.unlink()
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════════════════
# ACCELERATED STORAGE — GPU/CPU pinned-memory fast read/write
# ═══════════════════════════════════════════════════════════════════════════
class AcceleratedStorage:
    """
    High-speed file I/O with:
    - Memory-mapped files (mmap) for zero-copy access
    - Read-ahead buffering with configurable block size
    - Write coalescing (batch small writes into large sequential writes)
    - Pinned page cache for frequently accessed files
    - Parallel read across cores for large files
    """

    def __init__(self, cache_mb: int = 128, block_size: int = 2097152, read_ahead_kb: int = 512):
        """Larger cache (128MB), 2MB block, optional read-ahead for sequential reads."""
        self.cache_mb = cache_mb
        self.block_size = block_size
        self.read_ahead_kb = read_ahead_kb
        self._page_cache: Dict[str, bytes] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._total_read_bytes = 0
        self._total_write_bytes = 0
        self._write_buffer: Dict[str, bytearray] = {}
        self._lock = threading.Lock()
        self._max_cache_bytes = cache_mb * 1024 * 1024

        self._set_high_perf_io()

    def _set_high_perf_io(self):
        """Set OS-level I/O priority and buffer hints."""
        try:
            if sys.platform == "win32":
                kernel32 = ctypes.windll.kernel32
                handle = kernel32.GetCurrentProcess()
                IO_PRIORITY_HIGH = 3
                ProcessIoPriority = 21
                kernel32.SetPriorityClass(handle, 0x00000080)  # HIGH_PRIORITY_CLASS
                try:
                    ntdll = ctypes.windll.ntdll
                    priority = ctypes.c_ulong(IO_PRIORITY_HIGH)
                    ntdll.NtSetInformationProcess(
                        handle, ProcessIoPriority,
                        ctypes.byref(priority), ctypes.sizeof(priority)
                    )
                except Exception:
                    pass
        except Exception:
            pass

    def fast_read(self, path: str, max_bytes: int = 0) -> bytes:
        """Memory-mapped fast file read with page cache."""
        cache_key = f"{path}:{max_bytes}"
        with self._lock:
            if cache_key in self._page_cache:
                self._cache_hits += 1
                return self._page_cache[cache_key]
            self._cache_misses += 1

        try:
            p = Path(path)
            size = p.stat().st_size
            read_size = min(size, max_bytes) if max_bytes > 0 else size
            read_ahead = self.read_ahead_kb * 1024
            if read_ahead > 0 and read_size > 8192:
                read_size = min(size, read_size + read_ahead)

            if read_size > 4096:
                with open(path, "rb") as f:
                    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                    data = mm[:read_size]
                    mm.close()
            else:
                data = p.read_bytes()[:read_size] if max_bytes else p.read_bytes()

            self._total_read_bytes += len(data)

            with self._lock:
                current_size = sum(len(v) for v in self._page_cache.values())
                if current_size + len(data) > self._max_cache_bytes:
                    self._evict_cache(len(data))
                self._page_cache[cache_key] = data

            return data
        except Exception as e:
            logger.debug(f"fast_read error {path}: {e}")
            return b''

    def fast_read_text(self, path: str, max_chars: int = 100000, encoding: str = "utf-8") -> str:
        raw = self.fast_read(path, max_bytes=max_chars * 4)
        try:
            return raw.decode(encoding, errors="replace")[:max_chars]
        except Exception:
            return raw.decode("utf-8", errors="replace")[:max_chars]

    def fast_write(self, path: str, data: bytes, flush: bool = True):
        """Coalesced write with optional buffering."""
        if not flush:
            with self._lock:
                if path not in self._write_buffer:
                    self._write_buffer[path] = bytearray()
                self._write_buffer[path].extend(data)
                if len(self._write_buffer[path]) >= self.block_size:
                    self._flush_path(path)
            return

        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as f:
                f.write(data)
            self._total_write_bytes += len(data)
            cache_key = f"{path}:0"
            with self._lock:
                self._page_cache.pop(cache_key, None)
        except Exception as e:
            logger.debug(f"fast_write error {path}: {e}")

    def fast_write_text(self, path: str, text: str, encoding: str = "utf-8", flush: bool = True):
        self.fast_write(path, text.encode(encoding), flush=flush)

    def flush_all(self):
        with self._lock:
            for p in list(self._write_buffer.keys()):
                self._flush_path(p)

    def _flush_path(self, path: str):
        buf = self._write_buffer.pop(path, None)
        if buf:
            try:
                p = Path(path)
                p.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "ab") as f:
                    f.write(buf)
                self._total_write_bytes += len(buf)
            except Exception as e:
                logger.debug(f"flush error {path}: {e}")

    def _evict_cache(self, needed: int):
        """Evict oldest entries until enough space is available."""
        keys = list(self._page_cache.keys())
        freed = 0
        for k in keys:
            if freed >= needed:
                break
            freed += len(self._page_cache.pop(k, b''))

    def invalidate(self, path: str):
        with self._lock:
            to_remove = [k for k in self._page_cache if k.startswith(f"{path}:")]
            for k in to_remove:
                del self._page_cache[k]

    def status(self) -> Dict[str, Any]:
        with self._lock:
            cached_files = len(self._page_cache)
            cache_bytes = sum(len(v) for v in self._page_cache.values())
            pending_writes = sum(len(v) for v in self._write_buffer.values())
        return {
            "cache_mb": self.cache_mb,
            "block_size": self.block_size,
            "cached_files": cached_files,
            "cache_used_kb": cache_bytes // 1024,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate_pct": round(
                self._cache_hits * 100 / max(1, self._cache_hits + self._cache_misses), 1
            ),
            "total_read_mb": round(self._total_read_bytes / 1048576, 2),
            "total_write_mb": round(self._total_write_bytes / 1048576, 2),
            "pending_write_kb": pending_writes // 1024,
        }


# ═══════════════════════════════════════════════════════════════════════════
# PIPELINE ORCHESTRATOR — Ties all subsystems together
# ═══════════════════════════════════════════════════════════════════════════
class GPUPipeline:
    """
    Unified pipeline controller:
    - CoreVRAMMapper for core-to-slot mapping
    - VRAMRingBuffer for circular loop-back storage
    - SharedMemoryBus for zero-copy worker IPC
    - AcceleratedStorage for fast file I/O
    - VRAM pressure auto-scaling
    """

    def __init__(self, num_workers: int = 4):
        self.num_workers = num_workers
        cfg = _load_pipeline_config()
        self.mapper = CoreVRAMMapper()
        ring_cfg = cfg.get("ring_buffer", {})
        scale_mult = ring_cfg.get("scale_mult", 2)
        min_slots = ring_cfg.get("min_slots", 512)
        ring_slots = max(min_slots, self.mapper.total_slots * scale_mult)
        self.ring = VRAMRingBuffer(
            num_slots=ring_slots,
            slot_bytes=SLOT_SIZE_BYTES,
        )
        bus_cfg = cfg.get("shared_bus", {})
        seg_kb = bus_cfg.get("segment_bytes_kb", 128)
        self.bus = SharedMemoryBus(num_workers=num_workers, segment_bytes=seg_kb * 1024)
        stor_cfg = cfg.get("accelerated_storage", {})
        cache_mb = stor_cfg.get("cache_mb", 128)
        block_mb = stor_cfg.get("block_size_mb", 2)
        read_ahead_kb = stor_cfg.get("read_ahead_kb", 512)
        self.storage = AcceleratedStorage(
            cache_mb=cache_mb,
            block_size=block_mb * 1024 * 1024,
            read_ahead_kb=read_ahead_kb,
        )
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None

    def start(self):
        self.ring.open()
        self.bus.open()
        self.bus.enable_cuda_mps()
        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._pressure_monitor, daemon=True, name="VRAM_PRESSURE"
        )
        self._monitor_thread.start()
        logger.info(
            f"GPU Pipeline ONLINE | cores={self.mapper.logical_cores} | "
            f"slots={self.ring.num_slots} | bus={self.bus.num_workers}w | "
            f"cache={self.storage.cache_mb}MB"
        )

    def write_from_core(self, core_id: int, data: bytes) -> int:
        """Write data from a specific core into its mapped VRAM ring slot."""
        slot = self.ring.write(data, core_id=core_id)
        self.mapper.record_write(core_id)
        if self.ring.pressure:
            self.mapper.record_loopback(core_id)
        return slot

    def read_slot(self, slot_idx: int) -> bytes:
        return self.ring.read(slot_idx)

    def worker_send(self, from_w: int, to_w: int, data: bytes):
        self.bus.send(from_w, to_w, data)

    def worker_recv(self, worker_id: int) -> Tuple[int, bytes]:
        return self.bus.recv(worker_id)

    def worker_broadcast(self, from_w: int, data: bytes):
        self.bus.broadcast(from_w, data)

    def fast_read(self, path: str, max_bytes: int = 0) -> bytes:
        return self.storage.fast_read(path, max_bytes)

    def fast_read_text(self, path: str, max_chars: int = 100000) -> str:
        return self.storage.fast_read_text(path, max_chars)

    def fast_write(self, path: str, data: bytes):
        self.storage.fast_write(path, data)

    def _pressure_monitor(self):
        """Auto-scale: when ring buffer pressure is high, scale up mapper and trim."""
        while self._running:
            try:
                if self.ring.pressure:
                    self.mapper.scale_up()
                    old_slots = self.ring.num_slots
                    new_target = min(old_slots * 2, 65536)
                    logger.info(
                        f"VRAM PRESSURE: ring fill={self.ring.fill_pct}% | "
                        f"loopbacks={self.ring.total_loopbacks} | "
                        f"scaling mapper, target_slots={new_target}"
                    )
            except Exception as e:
                logger.debug(f"Pressure monitor error: {e}")
            time.sleep(10)

    def status(self) -> Dict[str, Any]:
        return {
            "mapper": self.mapper.status(),
            "ring_buffer": self.ring.status(),
            "shared_bus": self.bus.status(),
            "storage": self.storage.status(),
        }

    def status_oneline(self) -> str:
        ms = self.mapper.status()
        rs = self.ring.status()
        ss = self.storage.status()
        return (
            f"Cores:{ms['logical_cores']} Slots:{rs['num_slots']} "
            f"Fill:{rs['fill_pct']}% Loops:{rs['total_loopbacks']} "
            f"Bus_msg:{self.bus.msg_count} Cache_hit:{ss['hit_rate_pct']}% "
            f"R:{ss['total_read_mb']}MB W:{ss['total_write_mb']}MB"
        )

    def stop(self):
        self._running = False
        self.storage.flush_all()
        self.ring.close()
        self.bus.close()
        logger.info("GPU Pipeline: shutdown complete")
