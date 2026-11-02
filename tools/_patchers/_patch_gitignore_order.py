"""Swap two guards in verify_gitignore.py so the marker check comes first.

A directory that is not the subtree should say so, even when it also lacks a
.gitignore -- "refusing to report a verdict" is the more useful answer.
"""
import io
import os
import sys

TARGET = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "verify_gitignore.py")

GITIGNORE_GUARD = '''    if not (repo / ".gitignore").is_file():
        print(f"no .gitignore in {repo}")
        return 2
'''
MARKER_GUARD = '''    if why.startswith("MARKER"):
        print(f"refusing to report a verdict: {repo} is not the 7D-Crystal subtree")
        return 3
'''


def main():
    src = io.open(TARGET, encoding="utf-8", newline="").read()
    nl = "\r\n" if "\r\n" in src else "\n"
    g = GITIGNORE_GUARD.replace("\n", nl)
    m = MARKER_GUARD.replace("\n", nl)
    if g not in src or m not in src:
        print("PATTERN NOT FOUND (g=%s m=%s)" % (g in src, m in src))
        return 1
    if src.index(m) < src.index(g):
        print("already ordered")
        return 0
    src = src.replace(g + m, m + g, 1)
    io.open(TARGET, "w", encoding="utf-8", newline="").write(src)
    print("reordered guards")
    return 0


if __name__ == "__main__":
    sys.exit(main())
