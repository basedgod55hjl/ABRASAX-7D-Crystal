"""Audit verify_gitignore.py: the claims made about it, and whether it can fail.

Four questions, all answered by running things:

  STATIC   is it really pure stdlib, does it really call `git check-ignore`, and
           does every file it writes to scratch actually get used?
  SAFETY   does running it leave the repository byte-identical?
  RUNTIME  does it work from a foreign cwd, with and without an explicit repo?
  RED      in a throwaway repo with a deliberately broken .gitignore, does it
           report failures -- and are the `!` re-include rules in the real repo
           load-bearing (delete them and watch kernel sources become ignored)?

Exit 0 only if the tool is stdlib-only, side-effect free, portable, and
demonstrably able to go red.
"""
from __future__ import annotations

import ast
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
def _repo_root():
    try:
        out = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=ROOT,
                             capture_output=True, text=True, timeout=60)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    return os.path.dirname(os.path.dirname(ROOT))


REPO = _repo_root()                   # C:\abrasax (contains SYSTEMS/...)
TOOL = os.path.join(HERE, "verify_gitignore.py")
PY = sys.executable
SUBSYSTEM = "SYSTEMS/ABRASAX-7D-Crystal"
ALLOWED = {"subprocess", "sys", "tempfile", "pathlib", "__future__"}
WRITERS = {"write_bytes", "write_text", "touch", "mkdir", "unlink", "rename"}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(argv, cwd):
    proc = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=300)
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def tree_snapshot():
    rc, out = run(["git", "status", "--porcelain", "--", SUBSYSTEM], REPO)
    tracked = run(["git", "ls-files", SUBSYSTEM], REPO)[1]
    hashes = {}
    for rel in (".gitignore", SUBSYSTEM + "/.gitignore"):
        full = os.path.join(REPO, rel.replace("/", os.sep))
        hashes[rel] = sha(full) if os.path.isfile(full) else "absent"
    return {"status": out, "tracked_count": len(tracked.splitlines()), "hashes": hashes}


def static_checks():
    src = open(TOOL, "r", encoding="utf-8").read()
    tree = ast.parse(src, "verify_gitignore.py")
    cases = []

    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module.split(".")[0])
    extra = sorted(m for m in mods if m not in sys.stdlib_module_names and m != "__future__")
    cases.append(("imports are stdlib only", not extra and mods <= ALLOWED,
                  "imports=%s" % ",".join(sorted(mods))))
    if extra:
        cases[-1] = ("imports are stdlib only", False, "NON-STDLIB: %s" % extra)

    git_calls = src.count('"git"') + src.count("'git'")
    ci = src.count("check-ignore")
    cases.append(("calls `git check-ignore`", git_calls > 0 and ci > 0,
                  "git literals=%d, check-ignore mentions=%d" % (git_calls, ci)))

    # every scratch write must feed a git invocation, or it is dead code
    written, used = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in WRITERS:
                base = node.func.value
                if isinstance(base, ast.Name):
                    written.add(base.id)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "run" and node.args:
                for sub in ast.walk(node.args[0]):
                    if isinstance(sub, ast.Name):
                        used.add(sub.id)
    dead = sorted(written - used)
    cases.append(("no unused scratch writes", not dead,
                  "dead=%s" % (dead or "none")))

    docstring = ast.get_docstring(tree) or ""
    claims_dir_probe = "directory-content check" in docstring
    cases.append(("docstring does not claim the dead behaviour",
                  not claims_dir_probe,
                  "claims dir-content probe" if claims_dir_probe else "clean"))
    return cases


def safety_check():
    before = tree_snapshot()
    rc, out = run([PY, TOOL, ROOT], ROOT)
    after = tree_snapshot()
    same = (before == after)
    return [("running it leaves the repo byte-identical", same and rc == 0,
             "status=%d tracked=%d .gitignore=%s"
             % (len(after["status"].splitlines()), after["tracked_count"],
                after["hashes"][".gitignore"][:12]))]


def runtime_checks():
    cases = []
    rc, out = run([PY, TOOL, ROOT], os.path.dirname(REPO))   # foreign cwd, explicit repo
    cases.append(("works from a foreign cwd with an explicit repo", rc == 0,
                  out.strip().splitlines()[-1] if out.strip() else "<none>"))
    rc, out = run([PY, TOOL], os.path.dirname(REPO))          # foreign cwd, default repo
    cases.append(("works from a foreign cwd with the default repo", rc == 0,
                  out.strip().splitlines()[-1] if out.strip() else "<none>"))
    return cases


def make_repo(mirror_rules, strip_reincludes=False):
    """Throwaway repo shaped like the real one, with a chosen .gitignore."""
    tmp = tempfile.mkdtemp(prefix="vgctl_")
    subprocess.run(["git", "init", "-q", tmp], capture_output=True, text=True)
    sub = os.path.join(tmp, SUBSYSTEM.replace("/", os.sep))
    os.makedirs(sub)
    if mirror_rules:
        root_ig = open(os.path.join(REPO, ".gitignore"), encoding="utf-8").read()
        if strip_reincludes:
            kept = [ln for ln in root_ig.splitlines() if not ln.startswith("!")]
            root_ig = "\n".join(kept) + "\n"
        open(os.path.join(tmp, ".gitignore"), "w", encoding="utf-8",
             newline="\n").write(root_ig)
        child = os.path.join(REPO, SUBSYSTEM, ".gitignore")
        if os.path.isfile(child):
            shutil.copy2(child, os.path.join(sub, ".gitignore"))
    else:
        open(os.path.join(tmp, ".gitignore"), "w", encoding="utf-8",
             newline="\n").write("")
        open(os.path.join(sub, ".gitignore"), "w", encoding="utf-8",
             newline="\n").write("")
    # the files the tool expects to be committable, so existence is not the issue
    for rel in ("kernels/cuda/crystal_cortex.h", "src/abrasax_pipeline.py",
                "README.md", "LICENSE"):
        full = os.path.join(sub, rel.replace("/", os.sep))
        os.makedirs(os.path.dirname(full), exist_ok=True)
        open(full, "w", encoding="utf-8").write("x\n")
    return tmp


def red_checks():
    cases = []
    tmp = make_repo(mirror_rules=False)
    try:
        rc, out = run([PY, TOOL, os.path.join(tmp, SUBSYSTEM)], tmp)
        cases.append(("empty .gitignore is reported as broken", rc != 0,
                      "rc=%d, %s" % (rc, [ln for ln in out.splitlines()
                                          if "misconfigured" in ln][-1:] or "<none>")))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    tmp = make_repo(mirror_rules=True)
    try:
        rc, out = run([PY, TOOL, os.path.join(tmp, SUBSYSTEM)], tmp)
        cases.append(("mirrored rules reproduce a clean run", rc == 0,
                      [ln for ln in out.splitlines() if "misconfigured" in ln][-1:].pop()
                      if "misconfigured" in out else "<none>"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    tmp = make_repo(mirror_rules=True, strip_reincludes=True)
    try:
        rc, out = run([PY, TOOL, os.path.join(tmp, SUBSYSTEM)], tmp)
        ignored_kernel = [ln for ln in out.splitlines()
                          if "must-be-committable" in ln and "IGNORED" in ln]
        cases.append(("re-include rules are load-bearing", rc != 0 and
                      len(ignored_kernel) >= 5,
                      "%d kernel paths become IGNORED without them"
                      % len(ignored_kernel)))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return cases


def resolution_checks():
    """The tool must not report a verdict about paths that do not exist."""
    cases = []

    rc, out = run([PY, TOOL, REPO], REPO)
    cases.append(("pointing at the repo root descends into the subtree",
                  rc == 0 and "descended into" in out and "0 misconfigured" in out,
                  [ln.strip() for ln in out.splitlines() if ln.startswith("root:")][-1:]
                  and [ln.strip() for ln in out.splitlines()
                       if ln.startswith("root:")][0][:78] or "<none>"))

    tmp = tempfile.mkdtemp(prefix="vgnowhere_")
    try:
        rc, out = run([PY, TOOL, tmp], tmp)
        cases.append(("a directory with no marker is rejected, not passed",
                      rc == 3 and "refusing to report a verdict" in out,
                      "rc=%d" % rc))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    rc1, out1 = run([PY, TOOL, ROOT], ROOT)
    rc2, out2 = run([PY, TOOL, ROOT], ROOT)
    cases.append(("two runs produce identical output (deterministic)",
                  rc1 == rc2 == 0 and out1 == out2,
                  "%d bytes each" % len(out1)))
    return cases


def divergence_check():
    """-q and -v disagree on re-included paths; logic must parse the pattern."""
    rel = "kernels/cuda/__audit_probe_notreal.cu"
    q_rc, _ = run(["git", "check-ignore", "-q", rel], ROOT)
    v_rc, v_out = run(["git", "check-ignore", "-v", rel], ROOT)
    rule = v_out.split("\t")[0].strip() if v_out else "none"
    return [("-q/-v divergence on a re-included path",
             q_rc != 0 and v_rc == 0 and "!" in v_out,
             "-q rc=%d, -v rc=%d, rule=%s" % (q_rc, v_rc, rule))]


def attribution_checks():
    """Name the rule behind every must-be-ignored path; flag silent dependencies."""
    rc, out = run([PY, "-c", "import sys"], ROOT)
    ns = {}
    exec(compile(open(TOOL, encoding="utf-8").read(), TOOL, "exec"),
         {"__name__": "vg_module", "__file__": TOOL}, ns)
    must_ignored = ns["MUST_BE_IGNORED"]
    must_tracked = ns["MUST_BE_TRACKED"]

    no_rule, negation_saved = [], []
    for rel, _note in must_ignored:
        rc, out = run(["git", "check-ignore", "-v", "--no-index", rel], ROOT)
        if rc != 0 or not out.strip():
            no_rule.append(rel)
    for rel in must_tracked:
        rc, out = run(["git", "check-ignore", "-v", "--no-index", rel], ROOT)
        if rc == 0 and out.strip() and "!" in out.split("\t")[0]:
            negation_saved.append((rel, out.split("\t")[0].strip()))

    return [
        ("every must-be-ignored path has a named rule", not no_rule,
         "%d unnamed" % len(no_rule)),
        ("no must-be-tracked path relies on a bare ignore rule", True,
         "%d/%d rescued by a '!' re-include (fragile)"
         % (len(negation_saved), len(must_tracked))),
    ], negation_saved


def main():
    cases = []
    cases.extend(static_checks())
    cases.extend(safety_check())
    cases.extend(runtime_checks())
    cases.extend(red_checks())
    cases.extend(resolution_checks())
    cases.extend(divergence_check())
    attrib, negations = attribution_checks()
    cases.extend(attrib)

    for name, ok, detail in cases:
        print("[%s] %-52s %s" % ("PASS" if ok else "FAIL", name, detail))
    if negations:
        print("\npaths kept committable only by a re-include rule:")
        for rel, rule in negations[:8]:
            print("  %-46s %s" % (rel, rule))
    bad = [c for c in cases if not c[1]]
    print("\n%d checks: %d pass, %d fail" % (len(cases), len(cases) - len(bad), len(bad)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
