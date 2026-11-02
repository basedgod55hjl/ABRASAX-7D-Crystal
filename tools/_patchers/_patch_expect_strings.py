"""Tighten the aggregate runner's expectations and de-alarm the self test.

Two display bugs found by reading verify_all's own table:

  1. expect "0 fail" also matches the substring "10 fail". The self_test stage
     passed for the right reason but displayed "[PASS] aggregator -> red
     23 stages: 4 pass, 10 fail, 9 skip", which reads like a catastrophe.
  2. run() reported the FIRST matching line; the informative one is the LAST
     (the summary), since the output is a table.

Fixes: match ", 0 fail" (unambiguous), take the last match, and have _self_test
label the aggregator's deliberately-failing nested run as expected.
"""
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
VERIFY_ALL = os.path.join(HERE, "verify_all.py")
SELF_TEST = os.path.join(HERE, "_self_test.py")


def patch(path, pairs):
    src = io.open(path, encoding="utf-8").read()
    for old, new in pairs:
        if old not in src:
            print("NOT FOUND in %s: %r" % (os.path.basename(path), old[:60]))
            return False
        src = src.replace(old, new)
    io.open(path, "w", encoding="utf-8", newline="").write(src)
    print("patched %s" % os.path.basename(path))
    return True


def main():
    ok = patch(VERIFY_ALL, [
        ('    detail = ""\n    for ln in tail:\n        if expect in ln:\n'
         '            detail = ln\n            break\n',
         '    detail = ""\n    for ln in tail:\n        if expect in ln:\n'
         '            detail = ln  # last match wins: the summary is the useful line\n'),
        ('"0 fail"', '", 0 fail"'),
    ])
    if not ok:
        return 1
    ok = patch(SELF_TEST, [
        ('                      last_summary(out, "stages:")))',
         '                      "%s (expected: the probe tree is broken on purpose)"\n'
         '                      % last_summary(out, "stages:")))'),
    ])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
