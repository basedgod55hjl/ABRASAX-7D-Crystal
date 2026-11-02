"""Prove the abrasax_pipeline patch behaves: import works, use fails clearly.

Checks:
  1. import abrasax_pipeline              -> succeeds
  2. abrasax_pipeline.NeuralCompiler      -> ModuleNotFoundError naming the dep
  3. ABRASAXPipeline(...)                 -> same, at construction, not at import
  4. a missing attribute still raises plain AttributeError (no lying)
Exit 0 only if all four behave.
"""
from __future__ import annotations

import importlib
import os
import sys

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC)


def check_import():
    try:
        return importlib.import_module("abrasax_pipeline"), True, "module loaded"
    except Exception as exc:
        return None, False, "%s: %s" % (type(exc).__name__, exc)


def check_lazy_name(mod):
    try:
        mod.NeuralCompiler
        return False, "no error raised"
    except ModuleNotFoundError as exc:
        return "neural_language" in str(exc), str(exc).splitlines()[0][:80]
    except Exception as exc:
        return False, type(exc).__name__


def check_ctor(mod):
    try:
        mod.ABRASAXPipeline()
        return False, "constructed without the dependency"
    except ModuleNotFoundError as exc:
        return "neural_language" in str(exc), str(exc).splitlines()[0][:80]
    except Exception as exc:
        return False, "%s: %s" % (type(exc).__name__, exc)


def check_unknown_attr(mod):
    try:
        mod.definitely_not_a_real_name
        return False, "no error raised"
    except AttributeError:
        return True, "AttributeError"
    except Exception as exc:
        return False, type(exc).__name__


def main():
    cases = []
    mod, ok, detail = check_import()
    cases.append(("import succeeds", ok, detail))
    if mod is not None:
        for name, fn in (("lazy name raises", check_lazy_name),
                         ("ctor raises", check_ctor),
                         ("unknown attr -> AttributeError", check_unknown_attr)):
            ok, detail = fn(mod)
            cases.append((name, ok, detail))

    for name, ok, detail in cases:
        print("[%s] %-30s %s" % ("PASS" if ok else "FAIL", name, detail))
    bad = [c for c in cases if not c[1]]
    print("\n%d checks: %d pass, %d fail"
          % (len(cases), len(cases) - len(bad), len(bad)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
