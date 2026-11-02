"""What does cuda_kernel_loader actually need? Measured, not assumed.

The question raised: "the GPU ones need cuda_kernel_loader + a GPU + CUDA runtime;
probably cupy or torch CUDA -- if so the GPU verifications can't run here."

Short answer: it needs ctypes, numpy, nvcuda.dll (shipped with the NVIDIA driver)
and prebuilt .ptx/.cubin files. It does NOT need cupy, torch, pycuda, numba or
pyopencl -- and this script proves that the hard way, by re-running the GPU gates
in a process where those five are made unimportable.

  A  imports      AST of cuda_kernel_loader.py -> stdlib vs third-party
  B  artifacts    .ptx/.cubin it loads (required) and the optional .dll, with the
                  ptx target checked against the device
  C  runtime      nvcuda.dll, driver version, device name, compute capability,
                  SM count, total memory -- via the CUDA Driver API over ctypes
  D  forbidden    probe the framework stack: absent here
  E  sandbox      verify_kernel_math.py + verify_kernel_math_quantum.py run with
                  cupy/torch/pycuda/numba/pyopencl blocked through a meta_path hook

Note on the DLL: cuda_kernel_loader.py:443 loads crystal_kernels.dll only
`if use_dll and _DLL_PATH.exists()`, so a missing DLL is not a failure -- the PTX
modules are the primary path. That is why it is reported separately.

Exit 0 only if the loader needs nothing beyond stdlib+numpy, every required
artifact is present, the driver answers, and both GPU gates pass with the
framework stack blocked.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CUDA = os.path.join(ROOT, "kernels", "cuda")
LOADER = os.path.join(CUDA, "cuda_kernel_loader.py")
BUILD = os.path.join(CUDA, "build")
PY = sys.executable

REQUIRED = ["crystal_kernels_v2.ptx", "crystal_kernels_v1.ptx",
            "physx_crystal_dynamics.ptx", "crystal_kernels_v2.cubin",
            "crystal_kernels_v1.cubin", "physx_crystal_dynamics.cubin"]
OPTIONAL = ["crystal_kernels.dll"]
NOT_REQUIRED = ("cupy", "torch", "pycuda", "numba", "pyopencl")

BLOCKER = r'''
"""Block the deep-learning stack: the loader must not need it."""
import importlib.abc
import sys

BLOCKED = {"cupy", "torch", "pycuda", "numba", "pyopencl"}


class _Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in BLOCKED:
            raise ImportError("blocked by sandbox: %s" % fullname)
        return None


sys.meta_path.insert(0, _Block())
'''

PROBE = r'''
import ctypes, json

lib = ctypes.WinDLL("nvcuda.dll")
ver = ctypes.c_int()
rc = lib.cuDriverGetVersion(ctypes.byref(ver))
out = {"cuDriverGetVersion_rc": rc, "driver": int(ver.value)}
lib.cuInit(0)
dev = ctypes.c_int()
lib.cuDeviceGet(ctypes.byref(dev), 0)
name = ctypes.create_string_buffer(256)
lib.cuDeviceGetName(name, 256, dev)
out["device"] = name.value.decode(errors="replace")
for key, attr in (("cc_major", 75), ("cc_minor", 76), ("mp_count", 16)):
    v = ctypes.c_int()
    lib.cuDeviceGetAttribute(ctypes.byref(v), attr, dev)
    out[key] = int(v.value)
mem = ctypes.c_size_t()
lib.cuDeviceTotalMem(ctypes.byref(mem), dev)
out["total_mem_gib"] = round(int(mem.value) / (1024 ** 3), 2)
print(json.dumps(out))
'''


def sh(argv, cwd, timeout=300, env_extra=None):
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    proc = subprocess.run(argv, cwd=cwd, capture_output=True, text=True,
                          timeout=timeout, env=env)
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def is_present(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def loader_imports():
    tree = ast.parse(open(LOADER, "rb").read(), "cuda_kernel_loader.py")
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            mods.add(node.module.split(".")[0])
    stdlib = {m for m in mods if m in sys.stdlib_module_names or m == "__future__"}
    return sorted(mods - stdlib), sorted(stdlib)


def ptx_target(name="crystal_kernels_v2.ptx"):
    path = os.path.join(BUILD, name)
    if not os.path.isfile(path):
        return "?"
    with open(path, "r", errors="replace") as fh:
        for _ in range(40):
            line = fh.readline()
            if ".target" in line:
                return line.strip()
    return "no .target in header"


def main():
    cases = []

    third, stdlib = loader_imports()
    print("== A. loader imports ==")
    print("  stdlib     : %s" % ", ".join(stdlib))
    print("  third-party: %s" % (", ".join(third) or "none"))
    cases.append(("loader needs nothing beyond stdlib + numpy", set(third) <= {"numpy"},
                  "third-party=%s" % (third or "none")))

    print("\n== B. artifacts ==")
    missing = []
    for name in REQUIRED:
        path = os.path.join(BUILD, name)
        exists = os.path.isfile(path)
        size = os.path.getsize(path) if exists else 0
        print("  required  %-32s %s  %d bytes"
              % (name, "present" if exists else "MISSING", size))
        if not exists:
            missing.append(name)
    for name in OPTIONAL:
        path = os.path.join(BUILD, name)
        exists = os.path.isfile(path)
        size = os.path.getsize(path) if exists else 0
        print("  optional  %-32s %s  %d bytes"
              % (name, "present" if exists else "absent (PTX path used)", size))
    cases.append(("every required artifact is present", not missing,
                  "%d/%d" % (len(REQUIRED) - len(missing), len(REQUIRED))))
    target = ptx_target()
    print("  ptx target: %s" % target)
    cases.append(("ptx targets this GPU (sm_75)", "sm_75" in target, target))

    print("\n== C. runtime (CUDA Driver API) ==")
    rc, out = sh([PY, "-c", PROBE], ROOT, 120)
    if rc == 0 and out.strip().startswith("{"):
        info = json.loads(out.strip().splitlines()[-1])
        for key in ("driver", "device", "cc_major", "cc_minor", "mp_count",
                    "total_mem_gib"):
            print("  %-13s %s" % (key, info.get(key)))
        cases.append(("nvcuda.dll answers and a device is present",
                      info.get("cuDriverGetVersion_rc") == 0
                      and info.get("mp_count", 0) > 0,
                      "driver=%s device=%s" % (info.get("driver"), info.get("device"))))
        cc = "%d.%d" % (info.get("cc_major", 0), info.get("cc_minor", 0))
        cases.append(("device is sm_75, matching the ptx", cc == "7.5", "cc=%s" % cc))
    else:
        print("  driver probe failed: %s" % out.strip()[:160])
        cases.append(("nvcuda.dll answers and a device is present", False, "probe failed"))

    print("\n== D. framework stack (NOT required) ==")
    for name in NOT_REQUIRED:
        print("  %-9s %s" % (name, "installed" if is_present(name) else "absent"))
    absent = [n for n in NOT_REQUIRED if not is_present(n)]
    cases.append(("cupy/torch/pycuda/numba/pyopencl are absent here",
                  len(absent) == len(NOT_REQUIRED), "absent: %s" % ", ".join(absent)))

    print("\n== E. GPU gates with the framework stack blocked ==")
    sandbox = tempfile.mkdtemp(prefix="nodeps_")
    try:
        with open(os.path.join(sandbox, "sitecustomize.py"), "w",
                  encoding="utf-8") as fh:
            fh.write(BLOCKER)
        env = {"PYTHONPATH": sandbox}
        for label, script in (("kernel_math", "verify_kernel_math.py"),
                              ("mhq7d_quantum", "verify_kernel_math_quantum.py")):
            rc, out = sh([PY, os.path.join(HERE, script)], ROOT, 300, env)
            tail = [ln.strip() for ln in out.splitlines() if ln.strip()]
            summary = [ln for ln in tail if "failed" in ln]
            ok = rc == 0 and any("0 failed" in ln for ln in tail)
            print("  %-14s %s  %s" % (label, "PASS" if ok else "FAIL",
                                      summary[-1] if summary else "<none>"))
            cases.append(("%s passes with cupy/torch/pycuda blocked" % label, ok,
                          summary[-1] if summary else "<none>"))
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)

    for name, ok, detail in cases:
        print("[%s] %-52s %s" % ("PASS" if ok else "FAIL", name, detail))
    bad = [c for c in cases if not c[1]]
    print("\n%d checks: %d pass, %d fail" % (len(cases), len(cases) - len(bad), len(bad)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
