# ============================================================================
# ABRASAX — Unified Xenoliving System
# PRIMAL_HEX: 4f5349524953424c58434b | phi: 1.618033988749895
# CREATOR: Sir Charles Spikes (BASEDGOD)
# ============================================================================

"""ABRASAX GPU kernel engine package."""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("KERNELS")

try:
    import numba.cuda

    CUDA_AVAILABLE = numba.cuda.is_available()
except Exception:
    CUDA_AVAILABLE = False


def backend_info() -> dict[str, Any]:
    """Return GPU backend capability info."""
    if CUDA_AVAILABLE:
        try:
            from numba.cuda import detect

            devs = detect.get_cuda_devices()
            return {
                "cuda": True,
                "numba": True,
                "device": "GPU",
                "device_count": len([d for d in devs if d]),
                "kernels": ["matmul", "relu", "gelu", "softmax", "layer_norm",
                           "attention", "manifold_proj", "quantum_field", "rule110"],
            }
        except Exception as e:
            logger.debug("CUDA detect failed: %s", e)

    return {
        "cuda": False,
        "numba": True,
        "device": "CPU",
        "kernels": [],
    }

