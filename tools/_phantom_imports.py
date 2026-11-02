"""Find misqualified package imports: `from pkg.mod import X` where pkg/mod.py
does not exist, but a SIBLING mod.py next to the importing file does.

That is the ctypes_wrapper.py bug: `from cuda.crystal_kernels import ...` when
cuda/ holds only cuda_kernel_loader.py, while kernels/cuda/crystal_kernels.py
sits right next to the file. Python 3 implicit namespace packages made the
import look plausible, so the only thing that catches it is checking the disk.

Exit 1 if any misqualified import remains.
"""
from __future__ import annotations

import ast
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_DIRS = {".venv", "venv", "__pycache__", "node_modules", ".git", "target", "build"}


def is_module(path):
    return os.path.isfile(path + ".py") or os.path.isfile(os.path.join(path, "__init__.py"))


def scan():
    findings = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in sorted(filenames):
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dirpath, fn)
            rel = os.path.relpath(path, ROOT).replace("\\", "/")
            try:
                tree = ast.parse(open(path, "rb").read(), rel)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom) or node.level or not node.module:
                    continue
                parts = node.module.split(".")
                if len(parts) < 2:
                    continue
                pkg_dir = os.path.join(ROOT, parts[0])
                if not os.path.isdir(pkg_dir):
                    continue  # not a local package at all; other gates cover it
                target = os.path.join(pkg_dir, *parts[1:])
                if is_module(target):
                    continue  # resolves fine inside the package
                sibling = os.path.join(dirpath, parts[-1] + ".py")
                findings.append({
                    "file": rel, "line": node.lineno, "import": node.module,
                    "sibling": os.path.relpath(sibling, ROOT).replace("\\", "/")
                    if os.path.isfile(sibling) else "",
                })
    return findings


def main():
    findings = scan()
    for f in findings:
        hint = ("  -> sibling exists: %s" % f["sibling"]) if f["sibling"] else ""
        print("PHANTOM %s:%d  from %s import ...%s" % (f["file"], f["line"],
                                                       f["import"], hint))
    print("\nsites=%d phantom=%d" % (len(findings), len(findings)))
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
