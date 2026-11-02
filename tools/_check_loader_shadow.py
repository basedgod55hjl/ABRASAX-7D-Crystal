"""Which cuda_kernel_loader wins, and does it matter?

Two modules carry this name:

    cuda/cuda_kernel_loader.py           2.4 KB  stub: KERNEL_REGISTRY of
                                                 SimpleNamespace placeholders
    kernels/cuda/cuda_kernel_loader.py  49.7 KB  the real ctypes loader

Whatever lands first on sys.path wins, silently. This loads each one in its own
subprocess and reports origin, size, kernel count and device, so the shadowing
hazard is measured rather than assumed.

Exit 0 only if the stub is confirmed non-functional and the real one binds
kernels -- that asymmetry is what makes the name collision dangerous.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable

PROBE = r'''
import json, os, sys

sys.path.insert(0, sys.argv[1])
import cuda_kernel_loader as m


def call(obj, name):
    v = getattr(obj, name, None)
    if callable(v):
        try:
            v = v()
        except Exception as exc:
            return "error: %s" % type(exc).__name__
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return type(v).__name__


rt = m.CrystalKernelRuntime()
kernels = getattr(rt, "available_kernels", None)
try:
    names = sorted(kernels) if kernels is not None else []
except Exception:
    names = []
out = {
    "origin": m.__file__,
    "bytes": os.path.getsize(m.__file__),
    "kernel_count": len(names),
    "first_kernels": names[:3],
    "device": call(rt, "device_name"),
    "compute_capability": call(rt, "compute_capability"),
}
print(json.dumps(out))
'''


def probe(path):
    proc = subprocess.run([PY, "-c", PROBE, path], capture_output=True, text=True,
                          timeout=180)
    line = [ln for ln in (proc.stdout or "").splitlines() if ln.startswith("{")]
    if not line:
        tail = (proc.stderr or "").strip().splitlines()[-1:]
        return {"error": tail[0] if tail else "no output"}
    return json.loads(line[-1])


def main():
    stub = probe(os.path.join(ROOT, "cuda"))
    live = probe(os.path.join(ROOT, "kernels", "cuda"))

    for label, data in (("cuda/ (stub first)", stub), ("kernels/cuda/ (real)", live)):
        print("%-22s %s" % (label, json.dumps(data)))

    cases = [
        ("stub loads from cuda/",
         stub.get("bytes", 0) < 5000 and stub.get("origin", "").endswith("cuda\\cuda_kernel_loader.py"),
         stub.get("origin", "?")),
        ("real loads from kernels/cuda/",
         live.get("bytes", 0) > 40000
         and live.get("origin", "").endswith("kernels\\cuda\\cuda_kernel_loader.py"),
         "%s bytes=%s" % (live.get("origin", "?"), live.get("bytes"))),
        ("stub binds no real kernels", stub.get("kernel_count", -1) == 0,
         "kernel_count=%s" % stub.get("kernel_count")),
        ("real binds kernels", live.get("kernel_count", 0) > 0,
         "kernel_count=%s device=%s cc=%s"
         % (live.get("kernel_count"), live.get("device"),
            live.get("compute_capability"))),
    ]

    for name, ok, detail in cases:
        print("[%s] %-30s %s" % ("PASS" if ok else "FAIL", name, detail))
    bad = [c for c in cases if not c[1]]
    print("\n%d checks: %d pass, %d fail"
          % (len(cases), len(cases) - len(bad), len(bad)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
