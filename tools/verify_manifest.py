"""Generate and check a SHA-256 manifest of the artifacts this work claims.

Pure stdlib, no numpy, no CUDA, no git, no network -- this is the check anyone can
run on any machine to confirm the files are the ones that were verified, even if
they cannot run the GPU gates at all.

    python tools/verify_manifest.py            check every entry, exit 0 if clean
    python tools/verify_manifest.py --update   rewrite tools/MANIFEST.sha256
    python tools/verify_manifest.py --list     print the manifest without checking

Manifest lines:  <sha256>  <size>  <path relative to repo root>
Order is stable (sorted by path) so diffs are readable.
"""
from __future__ import annotations

import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MANIFEST = os.path.join(HERE, "MANIFEST.sha256")

# What this manifest covers: the verification tooling itself, the files it fixed,
# and the loader artifacts the claims rest on. Backups and caches are excluded.
TARGETS = [
    "tools",
    "src/abrasax_pipeline.py",
    "kernels/cuda/cuda_kernel_loader.py",
    "kernels/cuda/crystal_kernels.py",
    "kernels/cuda/ctypes_wrapper.py",
    "cuda/cuda_kernel_loader.py",
]
SKIP_SUFFIX = (".pyc", ".bak", ".png", ".o", ".dll", ".ptx", ".cubin")
SKIP_DIRS = {"__pycache__", ".venv", "build", "target", ".git"}


def is_covered(path):
    if path.endswith(SKIP_SUFFIX) or ".bak-" in path:
        return False
    return path.endswith((".py", ".md", ".sha256", ".txt", ".cmd"))


def collect():
    out = []
    for target in TARGETS:
        full = os.path.join(ROOT, target.replace("/", os.sep))
        if os.path.isfile(full):
            out.append(target)
            continue
        for dirpath, dirnames, filenames in os.walk(full):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fn in filenames:
                rel = os.path.relpath(os.path.join(dirpath, fn), ROOT).replace("\\", "/")
                if rel == "tools/MANIFEST.sha256":
                    continue
                if is_covered(rel):
                    out.append(rel)
    return sorted(set(out))


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build():
    lines = []
    for rel in collect():
        full = os.path.join(ROOT, rel.replace("/", os.sep))
        lines.append("%s  %d  %s" % (digest(full), os.path.getsize(full), rel))
    return lines


def read_manifest():
    entries = {}
    if not os.path.exists(MANIFEST):
        return entries
    with open(MANIFEST, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("  ")
            if len(parts) != 3:
                continue
            entries[parts[2]] = (parts[0], int(parts[1]))
    return entries


def write():
    lines = build()
    with open(MANIFEST, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# SHA-256 manifest for the ABRASAX-7D-Crystal verification work\n")
        fh.write("# regenerate: python tools/verify_manifest.py --update\n")
        fh.write("# check:      python tools/verify_manifest.py\n")
        for line in lines:
            fh.write(line + "\n")
    return len(lines)


def check():
    entries = read_manifest()
    if not entries:
        print("no manifest: %s (run --update to create it)" % MANIFEST)
        return 1
    missing, changed, extra, ok = [], [], [], 0
    for rel, (want_hash, want_size) in sorted(entries.items()):
        full = os.path.join(ROOT, rel.replace("/", os.sep))
        if not os.path.isfile(full):
            missing.append(rel)
            continue
        size = os.path.getsize(full)
        if size != want_size or digest(full) != want_hash:
            changed.append((rel, want_size, size))
            continue
        ok += 1
    on_disk = set(collect())
    extra = sorted(on_disk - set(entries))

    for rel, want, have in changed:
        print("CHANGED  %-46s %d -> %d bytes" % (rel, want, have))
    for rel in missing:
        print("MISSING  %s" % rel)
    for rel in extra:
        print("EXTRA    %s  (not in manifest)" % rel)
    print("\nentries=%d ok=%d changed=%d missing=%d extra=%d"
          % (len(entries), ok, len(changed), len(missing), len(extra)))
    return 0 if not (changed or missing or extra) else 1


def main():
    if "--update" in sys.argv:
        n = write()
        print("wrote %d entries -> %s" % (n, MANIFEST))
        return 0
    if "--list" in sys.argv:
        for line in build():
            print(line)
        return 0
    return check()


if __name__ == "__main__":
    sys.exit(main())
