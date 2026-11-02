"""Import matrix for the 7D-Crystal tree, with severity.

Answers three questions the compile gate cannot:

  1. does the interpreter actually have the third-party modules the code names?
  2. does every import resolve on the search path the tools use?
  3. if something is missing, is it optional (guarded) or a hard break?

Severity per import site:
  FATAL     module-level, not guarded  -> the file cannot be imported at all
  DEFERRED  inside a function/class    -> breaks only when that code path runs
  OPTIONAL  inside try/except ImportError, or guarded by a find_spec probe

Scope:
  core  (tools/, kernels/cuda/, src/, repo-root modules) -> must be FATAL-free
  all   (also kernels/gpu_compute/**)                    -> reported, allowed to
        need torch/cupy/tvm; those are optional backends with their own deps

Exit 0 only if scope=core has zero FATAL sites.
Usage: python tools/_import_matrix.py [--all]
"""
from __future__ import annotations

import ast
import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEARCH = [ROOT,
          os.path.join(ROOT, "kernels", "cuda"),
          os.path.join(ROOT, "cuda"),
          os.path.join(ROOT, "src"),
          os.path.join(ROOT, "tools")]
SKIP_DIRS = {".venv", "venv", "__pycache__", "node_modules", ".git", "target", "build"}
PROBE = ("numpy", "torch", "cupy", "scipy", "numba", "tvm", "pycuda", "pyopencl",
         "psutil", "PIL")
GUARD_NAMES = {"ImportError", "ModuleNotFoundError", "Exception", "BaseException"}


def probe(names):
    rows = []
    for name in names:
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError):
            spec = None
        if spec is None:
            rows.append((name, "MISSING", "", ""))
            continue
        version, where = "?", spec.origin or "<namespace>"
        try:
            version = str(getattr(__import__(name), "__version__", "?"))
        except Exception as exc:
            version = "import-failed: %s" % type(exc).__name__
        rows.append((name, "PRESENT", version, where))
    return rows


def local_candidates():
    names = {}
    for base in SEARCH:
        if not os.path.isdir(base):
            continue
        for entry in os.listdir(base):
            full = os.path.join(base, entry)
            if entry.endswith(".py"):
                names.setdefault(entry[:-3], []).append(full)
            elif os.path.isdir(full) and os.path.exists(os.path.join(full, "__init__.py")):
                names.setdefault(entry, []).append(full)
    return names


def resolve(root, local):
    if root in sys.stdlib_module_names:
        return "stdlib", ""
    if root in local:
        return "local", local[root][0]
    try:
        spec = importlib.util.find_spec(root)
    except (ImportError, ValueError):
        spec = None
    if spec is not None:
        return "installed", spec.origin or "<namespace>"
    return "MISSING", ""


class SiteVisitor(ast.NodeVisitor):
    """Collect import sites with their guarding context."""

    def __init__(self):
        self.depth = 0          # inside function/class?
        self.guarded = 0        # inside try/except ImportError?
        self.probed = 0         # inside an `if find_spec(...)` style guard?
        self.sites = []         # (module_root, lineno, severity)

    def visit_FunctionDef(self, node):
        self.depth += 1
        self.generic_visit(node)
        self.depth -= 1

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        self.depth += 1
        self.generic_visit(node)
        self.depth -= 1

    def visit_Try(self, node):
        catches = any(
            h.type is None
            or (isinstance(h.type, ast.Name) and h.type.id in GUARD_NAMES)
            or (isinstance(h.type, ast.Tuple)
                and any(isinstance(e, ast.Name) and e.id in GUARD_NAMES
                        for e in h.type.elts))
            for h in node.handlers)
        if catches:
            self.guarded += 1
        for child in node.body:
            self.visit(child)
        if catches:
            self.guarded -= 1
        for child in node.handlers + node.orelse + node.finalbody:
            self.visit(child)

    def visit_If(self, node):
        probe_guard = "find_spec" in ast.unparse(node.test) or \
                      "importlib" in ast.unparse(node.test)
        if probe_guard:
            self.probed += 1
        for child in node.body:
            self.visit(child)
        if probe_guard:
            self.probed -= 1
        for child in node.orelse:
            self.visit(child)

    def visit_Import(self, node):
        for alias in node.names:
            self._record(alias.name, node.lineno)

    def visit_ImportFrom(self, node):
        if node.level:  # relative import inside a package
            return
        self._record(node.module or "", node.lineno)

    def _record(self, module, lineno):
        root = module.split(".")[0]
        if not root:
            return
        if self.guarded or self.probed:
            sev = "OPTIONAL"
        elif self.depth:
            sev = "DEFERRED"
        else:
            sev = "FATAL"
        self.sites.append((root, lineno, sev))


def scope_of(rel):
    return "all" if rel.startswith("kernels/gpu_compute/") else "core"


def scan(local):
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
            visitor = SiteVisitor()
            visitor.visit(tree)
            for root, lineno, sev in visitor.sites:
                kind, where = resolve(root, local)
                if kind != "MISSING":
                    continue
                findings.append({"module": root, "file": rel, "line": lineno,
                                 "severity": sev, "scope": scope_of(rel)})
    return findings


def main():
    show_all = "--all" in sys.argv
    print("interpreter: %s  %s" % (sys.version.split()[0], sys.executable))

    print("\n-- third-party probe --")
    for name, state, version, where in probe(PROBE):
        print("  %-9s %-8s %-10s %s" % (name, state, version, where))

    local = local_candidates()
    findings = scan(local)
    findings.sort(key=lambda f: (f["scope"] != "core", f["severity"], f["module"], f["file"]))

    print("\n-- unresolved imports --")
    if not findings:
        print("  none")
    for f in findings:
        if not show_all and f["scope"] != "core":
            continue
        print("  %-9s %-8s %-16s %s:%d"
              % (f["severity"], f["scope"], f["module"], f["file"], f["line"]))

    core_fatal = [f for f in findings if f["scope"] == "core" and f["severity"] == "FATAL"]
    core_soft = [f for f in findings if f["scope"] == "core" and f["severity"] != "FATAL"]
    ext = [f for f in findings if f["scope"] != "core"]
    ext_mods = sorted({f["module"] for f in ext})
    print("\n-- summary --")
    print("  core_fatal=%d core_deferred_or_optional=%d external_only=%d"
          % (len(core_fatal), len(core_soft), len(ext)))
    if ext_mods:
        print("  external scope is limited to: %s" % ", ".join(ext_mods))
    if not show_all and ext:
        print("  (pass --all to list the %d external sites)" % len(ext))
    for f in core_fatal:
        print("  FATAL %s  %s:%d" % (f["module"], f["file"], f["line"]))
    return 1 if core_fatal else 0


if __name__ == "__main__":
    sys.exit(main())
