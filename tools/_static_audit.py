"""Static audit of a Python tools dir.

Fatal:  unresolved imports, duplicate defs, bare except, stale/corrupt bytecode,
        module-level statements other than the sanctioned forms.
Sanctioned module-level: docstring, `sys.path.insert/append(...)`, bare `print(...)`.
"""
from __future__ import annotations

import ast
import importlib.util
import marshal
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
STDLIB = set(sys.stdlib_module_names)
PYC_DIR = os.path.join(HERE, "__pycache__")
EXTRA_PATHS = [os.path.join(ROOT, "kernels", "cuda")]


def bootstrap():
    """Put the kernel-loader dir on sys.path exactly once (lazy, not at import)."""
    for path in EXTRA_PATHS:
        if path not in sys.path:
            sys.path.insert(0, path)


def resolvable(mod_root, local):
    if not mod_root or mod_root in STDLIB or mod_root in local:
        return True
    bootstrap()
    try:
        return importlib.util.find_spec(mod_root) is not None
    except (ImportError, ValueError):
        return False


def check_imports(tree):
    local = {f[:-3] for f in os.listdir(HERE) if f.endswith(".py")}
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods = [node.module or ""]
        else:
            continue
        for m in mods:
            if not resolvable(m.split(".")[0], local):
                bad.append((node.lineno, m))
    return bad


def is_sanctioned(node):
    """Docstring, sys.path bootstrap, or a bare print() call."""
    if not isinstance(node, ast.Expr):
        return False
    if isinstance(node.value, ast.Constant):
        return True  # module docstring
    if not isinstance(node.value, ast.Call):
        return False
    fn = node.value.func
    if isinstance(fn, ast.Name) and fn.id == "print":
        return True
    if isinstance(fn, ast.Attribute) and fn.attr in ("insert", "append"):
        return ast.unparse(fn.value) == "sys.path"
    return False


def check_module_level(tree):
    out = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef,
                             ast.AsyncFunctionDef, ast.ClassDef, ast.Assign,
                             ast.AnnAssign)):
            continue
        if is_sanctioned(node):
            continue
        if isinstance(node, ast.If):
            test = node.test
            if (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name)
                    and test.left.id == "__name__"):
                continue
        out.append((node.lineno, type(node).__name__))
    return out


def check_duplicates(tree):
    seen, dups = {}, []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name in seen:
                dups.append((node.name, seen[node.name], node.lineno))
            seen[node.name] = node.lineno
    return dups


def check_bare_except(tree):
    return [n.lineno for n in ast.walk(tree)
            if isinstance(n, ast.ExceptHandler) and n.type is None]


def check_pyc(path):
    name = os.path.basename(path)[:-3]
    pyc = os.path.join(PYC_DIR, name + ".cpython-311.pyc")
    if not os.path.exists(pyc):
        return "absent", None
    with open(pyc, "rb") as fh:
        head = fh.read(16)
    magic = head[:4]
    _flags, mtime, size = struct.unpack("<III", head[4:16])
    live = os.stat(path)
    if magic != importlib.util.MAGIC_NUMBER:
        return "bad", "magic mismatch"
    if mtime and (int(mtime) != int(live.st_mtime) or size != live.st_size):
        return "stale", "source newer than pyc"
    try:
        with open(pyc, "rb") as fh:
            fh.read(16)
            marshal.load(fh)
    except Exception as exc:
        return "bad", "unmarshal: %s" % exc
    return "fresh", None


def main():
    files = sorted(f for f in os.listdir(HERE) if f.endswith(".py"))
    fatal = 0
    fresh = absent = 0
    for name in files:
        path = os.path.join(HERE, name)
        tree = ast.parse(open(path, "rb").read(), name)
        bad_imp = check_imports(tree)
        side = check_module_level(tree)
        dups = check_duplicates(tree)
        bare = check_bare_except(tree)
        state, why = check_pyc(path)
        fresh += state == "fresh"
        absent += state == "absent"
        ok = not (bad_imp or side or dups or bare or why)
        if not ok:
            fatal += 1
        defs = len([n for n in tree.body
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))])
        print("%s %-32s defs=%-2d pyc=%-7s"
              % ("OK  " if ok else "FAIL", name, defs, state))
        for ln, mod in bad_imp:
            print("        L%-4d UNRESOLVED IMPORT %s" % (ln, mod))
        for ln, what in side:
            print("        L%-4d UNSANCTIONED MODULE-LEVEL %s" % (ln, what))
        for nm, a, b in dups:
            print("        DUPLICATE DEF %s L%d/L%d" % (nm, a, b))
        for ln in bare:
            print("        L%-4d BARE EXCEPT" % ln)
        if why:
            print("        PYC %s" % why)
    print("\nfiles=%d fatal=%d  pyc_fresh=%d pyc_absent=%d"
          % (len(files), fatal, fresh, absent))
    return 1 if fatal else 0


if __name__ == "__main__":
    sys.exit(main())
